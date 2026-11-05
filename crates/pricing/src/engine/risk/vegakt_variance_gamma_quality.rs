//! Shared quality gates for variance-Gamma candidate estimates.
use crate::MonteCarloError;
use crate::risk::{
    RiskConfigError, VegaKtGammaRecoveryPolicy, VegaKtStepMoment, VegaKtVarianceGammaQuality,
};

fn invalid(s: &'static str) -> MonteCarloError {
    RiskConfigError::VegaKtChainInput { quantity: s }.into()
}

fn norm(v: impl Iterator<Item = f64>) -> f64 {
    v.fold(0., f64::hypot)
}
pub(super) fn quality(
    b: &[VegaKtStepMoment],
    r: &[VegaKtStepMoment],
    q: &[VegaKtStepMoment],
    p: &VegaKtGammaRecoveryPolicy,
) -> Result<VegaKtVarianceGammaQuality, MonteCarloError> {
    let metrics = |v: &[VegaKtStepMoment]| -> Result<(f64, f64), MonteCarloError> {
        let errors = v
            .iter()
            .map(|m| {
                m.standard_error()
                    .ok_or_else(|| invalid("at least two variance-Gamma samples required"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((norm(v.iter().map(|m| m.mean())), norm(errors.into_iter())))
    };
    let (b, bse) = metrics(b)?;
    let (r, rse) = metrics(r)?;
    let (q, qse) = metrics(q)?;
    if [b, bse, r, rse, q, qse].iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite variance-Gamma quality"));
    }
    let ratio = |a: f64, b: f64| (b > 0.).then(|| a / b).filter(|v| v.is_finite());
    let reconstruction_margin_ratio = ratio(
        r + p.mc_margin_multiplier() * rse,
        b - p.mc_margin_multiplier() * bse,
    );
    let gamma_relative_standard_error = ratio(qse, q);
    Ok(VegaKtVarianceGammaQuality {
        sensitivity_norm: b,
        sensitivity_mc_norm: bse,
        reconstruction_norm: r,
        reconstruction_mc_norm: rse,
        gamma_norm: q,
        gamma_mc_norm: qse,
        reconstruction_margin_ratio,
        gamma_relative_standard_error,
        passed: reconstruction_margin_ratio.is_some_and(|v| v <= p.reconstruction_relative_limit())
            && gamma_relative_standard_error
                .is_some_and(|v| v <= p.gamma_relative_standard_error_limit()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moment(mean: f64, error: f64) -> VegaKtStepMoment {
        VegaKtStepMoment {
            mean,
            sample_variance: Some(2. * error * error),
            standard_error: Some(error),
        }
    }

    fn policy(limit: f64) -> VegaKtGammaRecoveryPolicy {
        VegaKtGammaRecoveryPolicy::new(vec![0.01], 128, limit, limit, 2.).unwrap()
    }

    #[test]
    fn signed_estimates_use_mc_margins_and_inclusive_quality_limits() {
        let b = [moment(-3., 0.3), moment(4., 0.4)];
        let r = [moment(-0.5, 0.25)];
        let q = [moment(-2., 0.5)];
        assert_eq!(
            quality(&b, &r, &q, &policy(0.25)).unwrap(),
            VegaKtVarianceGammaQuality {
                sensitivity_norm: 5.,
                sensitivity_mc_norm: 0.5,
                reconstruction_norm: 0.5,
                reconstruction_mc_norm: 0.25,
                gamma_norm: 2.,
                gamma_mc_norm: 0.5,
                reconstruction_margin_ratio: Some(0.25),
                gamma_relative_standard_error: Some(0.25),
                passed: true,
            }
        );
        assert!(!quality(&b, &r, &q, &policy(0.249)).unwrap().passed);
    }

    #[test]
    fn unidentified_signal_or_zero_gamma_withholds_quality() {
        for signal in [0.5, 1.] {
            let result = quality(
                &[moment(signal, 0.5)],
                &[moment(0., 0.)],
                &[moment(0., 0.)],
                &policy(0.25),
            )
            .unwrap();
            assert_eq!(result.reconstruction_margin_ratio, None);
            assert_eq!(result.gamma_relative_standard_error, None);
            assert!(!result.passed);
        }
    }

    #[test]
    fn missing_uncertainty_and_nonfinite_metrics_keep_typed_errors() {
        let mut missing = moment(1., 0.);
        missing.standard_error = None;
        for (input, message) in [
            (missing, "at least two variance-Gamma samples required"),
            (
                moment(f64::INFINITY, 0.),
                "nonfinite variance-Gamma quality",
            ),
            (moment(1., f64::NAN), "nonfinite variance-Gamma quality"),
        ] {
            for slot in 0..3 {
                let mut inputs = [[moment(1., 0.)]; 3];
                inputs[slot] = [input];
                assert!(matches!(
                    quality(&inputs[0], &inputs[1], &inputs[2], &policy(0.25)),
                    Err(MonteCarloError::RiskConfig(RiskConfigError::VegaKtChainInput { quantity }))
                        if quantity == message
                ));
            }
        }
    }
}
