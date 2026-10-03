//! Fixed-kernel tangents of the numerical Riccati solution and Fourier integral.
use super::{FourierError, HestonFourierPlan, HestonFourierPrice, simpson_weight};
use pricing_numerics::{Complex64 as C, NeumaierSum, standard_normal_pdf};

/// Per-unit scalar parameter derivatives, not implied-vol Vega or calibrated risk.
/// Hurst, or every explicit lift weight/rate, is held fixed. At parameter-domain
/// boundaries these are the inward derivatives of the analytic continuation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HestonParameterSensitivities {
    pub initial_variance: f64,
    pub mean_reversion: f64,
    pub long_run_variance: f64,
    pub vol_of_vol: f64,
    pub correlation: f64,
}
impl HestonParameterSensitivities {
    /// Order: v0, kappa, theta, nu, rho. No unit scaling is applied.
    #[must_use]
    pub fn as_array(self) -> [f64; 5] {
        [
            self.initial_variance,
            self.mean_reversion,
            self.long_run_variance,
            self.vol_of_vol,
            self.correlation,
        ]
    }
}
impl From<[f64; 5]> for HestonParameterSensitivities {
    fn from(v: [f64; 5]) -> Self {
        Self {
            initial_variance: v[0],
            mean_reversion: v[1],
            long_run_variance: v[2],
            vol_of_vol: v[3],
            correlation: v[4],
        }
    }
}

/// Call and put have the same parameter derivatives at fixed F, K, D and T.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonFourierParameterRisk {
    pub price: HestonFourierPrice,
    pub sensitivities: HestonParameterSensitivities,
    /// Absolute Simpson N/N2 differences on the retained frequency interval.
    pub quadrature_differences: HestonParameterSensitivities,
    /// Derivative correction envelopes on [cutoff/2, cutoff], NOT omitted-tail
    /// bounds, Riccati-error estimates, total-error bounds, or sampling errors.
    pub tail_indicators: HestonParameterSensitivities,
}

/// Reusable per-maturity parameter derivatives. Compilation differentiates the
/// implicit root, rather than bumping model inputs or differentiating MC paths.
#[derive(Clone, Debug)]
pub struct HestonFourierParameterRiskPlan {
    price_plan: HestonFourierPlan,
    derivatives: Vec<[C; 5]>,
}
impl HestonFourierPlan {
    /// Compile/cache five fixed-kernel parameter tangents. The original price
    /// plan remains unchanged; evaluating further strikes requires no Riccati solve.
    /// At positive maturity initial/control variance must be strictly positive.
    /// At T=0 all parameter derivatives are zero, including ATM (unlike Delta).
    pub fn parameter_risk_plan(&self) -> Result<HestonFourierParameterRiskPlan, FourierError> {
        let n = self.config.integration_intervals;
        if self.riccati.parameter_work() * (n as u64 + 1) > 8_000_000_000 {
            return Err(FourierError::InvalidInput("parameter-risk work limit"));
        }
        let variance = self.riccati.control_variance();
        let t = self.riccati.maturity();
        if t > 0.0 && variance <= 0.0 {
            return Err(FourierError::InvalidInput(
                "parameter risk requires positive initial/control variance at positive maturity",
            ));
        }
        let mut derivatives = Vec::with_capacity(n + 1);
        for j in 0..=n {
            let u = j as f64 * self.config.cutoff / n as f64;
            let z = C::new(0.5, u);
            let c = (z * z - z) * 0.5;
            let (log_m, tangent) = self.riccati.log_transform_derivatives(z)?;
            let model = log_m.exp();
            let control_derivative = (c * variance).exp() * c * t;
            let mut node = [C::ZERO; 5];
            for q in 0..5 {
                node[q] = ((if q == 0 { control_derivative } else { C::ZERO })
                    - model * tangent[q])
                    / (u * u + 0.25);
            }
            if node.iter().any(|x| !x.is_finite()) {
                return Err(FourierError::NumericalFailure(
                    "parameter-risk Fourier node",
                ));
            }
            derivatives.push(node);
        }
        Ok(HestonFourierParameterRiskPlan {
            price_plan: self.clone(),
            derivatives,
        })
    }
}
impl HestonFourierParameterRiskPlan {
    /// Log-transform derivatives in the fixed order v0, kappa, theta, nu, rho.
    /// This explicit transform query performs a new Riccati/tangent solve.
    pub fn log_transform_derivatives(&self, z: C) -> Result<[C; 5], FourierError> {
        self.price_plan
            .riccati
            .log_transform_derivatives(z)
            .map(|(_, d)| d)
    }
    /// Fixed F/K/D/T parameter risk; call and put share these derivatives.
    pub fn price(
        &self,
        forward: f64,
        strike: f64,
        discount: f64,
    ) -> Result<HestonFourierParameterRisk, FourierError> {
        let price = self.price_plan.price(forward, strike, discount)?;
        let t = self.price_plan.riccati.maturity();
        if t == 0.0 {
            return Ok(HestonFourierParameterRisk {
                price,
                sensitivities: HestonParameterSensitivities::default(),
                quadrature_differences: HestonParameterSensitivities::default(),
                tail_indicators: HestonParameterSensitivities::default(),
            });
        }
        let n = self.price_plan.config.integration_intervals;
        let du = self.price_plan.config.cutoff / n as f64;
        let log_moneyness = forward.ln() - strike.ln();
        let scale = discount * forward.sqrt() * strike.sqrt() / std::f64::consts::PI;
        let mut fine = [NeumaierSum::new(); 5];
        let mut coarse = [NeumaierSum::new(); 5];
        let mut tail = [NeumaierSum::new(); 5];
        for (j, node) in self.derivatives.iter().enumerate() {
            let phase = C::new(0.0, j as f64 * du * log_moneyness).exp();
            for q in 0..5 {
                let value = (phase * node[q]).re;
                fine[q].add(simpson_weight(j, n) * value);
                if j.is_multiple_of(2) {
                    coarse[q].add(simpson_weight(j / 2, n / 2) * value);
                }
                if j >= n / 2 {
                    tail[q].add(simpson_weight(j - n / 2, n / 2) * node[q].abs());
                }
            }
        }
        let root = self.price_plan.riccati.control_variance().sqrt();
        let d1 = log_moneyness / root + 0.5 * root;
        let control_v0 = discount * forward * standard_normal_pdf(d1) * (0.5 * t / root);
        let mut values = [0.0; 5];
        let mut errors = [0.0; 5];
        let mut envelopes = [0.0; 5];
        for q in 0..5 {
            let correction = scale * du * fine[q].total() / 3.0;
            values[q] = correction + if q == 0 { control_v0 } else { 0.0 };
            errors[q] = (correction - scale * (2.0 * du) * coarse[q].total() / 3.0).abs();
            envelopes[q] = scale * du * tail[q].total() / 3.0;
        }
        if values
            .iter()
            .chain(errors.iter())
            .chain(envelopes.iter())
            .any(|x| !x.is_finite())
        {
            return Err(FourierError::NumericalFailure(
                "non-finite parameter sensitivity",
            ));
        }
        Ok(HestonFourierParameterRisk {
            price,
            sensitivities: values.into(),
            quadrature_differences: errors.into(),
            tail_indicators: envelopes.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::riccati::RiccatiPlan;
    use super::*;
    use crate::rough_volatility::{HestonFourierConfig, RoughHeston};
    #[test]
    fn tangent_budget_rejects_before_frequency_work() {
        let config = HestonFourierConfig::new(8192, 8192, 128.0).unwrap();
        let model = RoughHeston::new(0.1, 0.04, 0.7, 0.055, 0.18, -0.65)
            .unwrap()
            .into();
        let plan = HestonFourierPlan {
            config,
            riccati: RiccatiPlan::new(&model, 1.0, config).unwrap(),
            differences: Vec::new(),
        };
        assert!(matches!(
            plan.parameter_risk_plan(),
            Err(FourierError::InvalidInput("parameter-risk work limit"))
        ));
    }
}
