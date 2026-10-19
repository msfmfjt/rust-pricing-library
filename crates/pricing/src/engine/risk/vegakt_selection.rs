use crate::MonteCarloError;
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::EngineConfig;
use crate::risk::{
    RiskConfigError, VegaKtConvolutionDiagnostic, VegaKtGammaCandidateAudit,
    VegaKtGammaRecoveryAudit, VegaKtGammaRecoveryPolicy, VegaKtGammaRecoveryStatus,
    VegaKtGammaRowQuality, VegaKtStepMoment,
};

fn invalid(quantity: &'static str) -> MonteCarloError {
    RiskConfigError::VegaKtChainInput { quantity }.into()
}
fn norm(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0., f64::hypot)
}
fn means(v: &[VegaKtStepMoment]) -> f64 {
    norm(v.iter().map(|m| m.mean()))
}
fn errors(v: &[VegaKtStepMoment]) -> Result<f64, MonteCarloError> {
    let se = v
        .iter()
        .map(|m| {
            m.standard_error()
                .ok_or_else(|| invalid("at least two sampling units required for Gamma selection"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(norm(se.into_iter()))
}
fn ratio(a: f64, b: f64) -> Option<f64> {
    (b > 0.).then(|| a / b).filter(|v| v.is_finite())
}
fn audit(
    diagnostic: VegaKtConvolutionDiagnostic,
    p: &VegaKtGammaRecoveryPolicy,
) -> Result<VegaKtGammaCandidateAudit, MonteCarloError> {
    let quality = diagnostic
        .rows
        .iter()
        .map(|r| {
            let b = means(&r.local_vega);
            let bse = errors(&r.local_vega)?;
            let e = means(&r.reconstruction_difference);
            let ese = errors(&r.reconstruction_difference)?;
            let q = means(&r.gamma_masses);
            let qse = errors(&r.gamma_masses)?;
            if [b, bse, e, ese, q, qse].iter().any(|v| !v.is_finite()) {
                return Err(invalid("nonfinite Gamma quality norm"));
            }
            let reconstruction_margin_ratio = ratio(
                e + p.mc_margin_multiplier * ese,
                b - p.mc_margin_multiplier * bse,
            );
            let gamma_relative_standard_error = ratio(qse, q);
            let passed = reconstruction_margin_ratio
                .is_some_and(|r| r <= p.reconstruction_relative_limit)
                && gamma_relative_standard_error
                    .is_some_and(|r| r <= p.gamma_relative_standard_error_limit);
            Ok(VegaKtGammaRowQuality {
                source_maturity: r.source_maturity,
                local_vega_norm: b,
                local_vega_mc_norm: bse,
                reconstruction_difference_norm: e,
                reconstruction_mc_norm: ese,
                gamma_mass_norm: q,
                gamma_mc_norm: qse,
                reconstruction_margin_ratio,
                gamma_relative_standard_error,
                unobserved_source_nodes: r
                    .local_vega_nonzero_sampling_units
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &n)| (n == 0).then_some(i))
                    .collect(),
                passed,
            })
        })
        .collect::<Result<Vec<_>, MonteCarloError>>()?;
    let passed = !quality.is_empty() && quality.iter().all(|q| q.passed);
    Ok(VegaKtGammaCandidateAudit {
        diagnostic,
        quality,
        passed,
    })
}
impl SimulationPlan {
    pub(in crate::engine) fn evaluate_vega_kt_gamma_recovery_audit(
        &self,
        gamma_strikes: Vec<f64>,
        policy: &VegaKtGammaRecoveryPolicy,
    ) -> Result<VegaKtGammaRecoveryAudit, MonteCarloError> {
        let EngineConfig::PseudoMonteCarlo(engine) = self.engine else {
            return Err(invalid(
                "pseudo-MC required for split-sample Gamma recovery",
            ));
        };
        let count = engine.independent_sampling_units().get();
        if count < 2 {
            return Err(invalid("at least two validation sampling units required"));
        }
        let end = count
            .checked_add(policy.pilot_sampling_units)
            .ok_or_else(|| invalid("Gamma pilot counter range overflow"))?;
        let mut pilot_candidates = Vec::new();
        for &lambda in &policy.candidate_regularizations {
            let d = self.convolution_gamma_in_range_with_constraints(
                gamma_strikes.clone(),
                lambda,
                count,
                policy.pilot_sampling_units,
                policy.preserve_local_vega_total,
                &policy.local_vega_conservation_directions,
            )?;
            pilot_candidates.push(audit(d, policy)?);
        }
        // The strongest pilot-feasible lambda is frozen before looking at ANY
        // validation moments. A validation failure never triggers reselection.
        let selected_regularization = pilot_candidates
            .iter()
            .find(|c| c.passed)
            .map(|c| c.diagnostic.relative_regularization);
        let validation = selected_regularization
            .map(|lambda| {
                self.convolution_gamma_in_range_with_constraints(
                    gamma_strikes.clone(),
                    lambda,
                    0,
                    count,
                    policy.preserve_local_vega_total,
                    &policy.local_vega_conservation_directions,
                )
                .and_then(|d| audit(d, policy))
            })
            .transpose()?;
        let status = match &validation {
            None => VegaKtGammaRecoveryStatus::NoPilotCandidate,
            Some(v) if v.passed => VegaKtGammaRecoveryStatus::PositiveTimePolicyPassed,
            Some(_) => VegaKtGammaRecoveryStatus::ValidationFailed,
        };
        let initial = self.evaluate_vega_kt_initial_gamma_diagnostic(gamma_strikes)?;
        Ok(VegaKtGammaRecoveryAudit {
            policy_label: if !policy.local_vega_conservation_directions.is_empty() {
                "flat_variance_direction_constrained_split_sample_gamma_audit_v1"
            } else if policy.preserve_local_vega_total {
                "flat_variance_conservative_split_sample_gamma_audit_v1"
            } else {
                "flat_variance_split_sample_gamma_audit_v1"
            },
            request_fingerprint: self.request_fingerprint(),
            plan_fingerprint: self.plan_fingerprint(),
            policy: policy.clone(),
            status,
            validation_sampling_units: count,
            pilot_index_start: count,
            pilot_index_end_exclusive: end,
            pilot_candidates,
            selected_regularization,
            validation,
            initial,
        })
    }
}
