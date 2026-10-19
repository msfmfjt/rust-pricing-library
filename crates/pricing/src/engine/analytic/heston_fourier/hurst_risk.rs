//! Hurst-only tangent of the fractional kernel and Riccati/Fourier calculation.
use super::riccati::PowerHurstWeights;
use super::{FourierError, HestonFourierPlan, HestonFourierPrice, simpson_weight};
use pricing_numerics::{Complex64 as C, NeumaierSum};

/// Per-unit Hurst derivative at fixed F/K/D/T and v0/kappa/theta/nu/rho.
/// Call and put share this sensitivity. This is not a finite-lift factory risk,
/// an IV Vega, a recalibrated market risk, or a reverse-mode AAD result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonFourierHurstRisk {
    pub price: HestonFourierPrice,
    pub hurst_sensitivity: f64,
    /// Absolute Simpson N/N2 difference on `[0,cutoff]`, not a total error bound.
    pub quadrature_difference: f64,
    /// Derivative envelope on [cutoff/2,cutoff], NOT an omitted-tail bound.
    pub tail_indicator: f64,
}

/// Cached per-maturity Hurst sensitivity, only for Rough Heston.
#[derive(Clone, Debug)]
pub struct HestonFourierHurstRiskPlan {
    price_plan: HestonFourierPlan,
    weights: PowerHurstWeights,
    derivatives: Vec<C>,
}
impl HestonFourierPlan {
    /// Differentiate/cache the power-kernel weights and the implicit Riccati root.
    /// Other scalar parameters and the time unit (years) stay fixed.
    /// At H=.5 this is the derivative approached from inside H<.5; it is not zero
    /// merely because the base process is ordinary Heston. Finite lifts are rejected.
    pub fn hurst_risk_plan(&self) -> Result<HestonFourierHurstRiskPlan, FourierError> {
        let n = self.config.integration_intervals;
        if self.riccati.hurst_work() * (n as u64 + 1) > 8_000_000_000 {
            return Err(FourierError::InvalidInput("Hurst-risk work limit"));
        }
        let weights = self.riccati.hurst_weights()?;
        let mut derivatives = Vec::with_capacity(n + 1);
        for j in 0..=n {
            let u = j as f64 * self.config.cutoff / n as f64;
            let (log_m, tangent) = self
                .riccati
                .log_transform_hurst_derivative(C::new(0.5, u), &weights)?;
            // The Black control uses v0*T, which is independent of H.
            let value = -(log_m.exp() * tangent) / (u * u + 0.25);
            if !value.is_finite() {
                return Err(FourierError::NumericalFailure("Hurst-risk Fourier node"));
            }
            derivatives.push(value);
        }
        Ok(HestonFourierHurstRiskPlan {
            price_plan: self.clone(),
            weights,
            derivatives,
        })
    }
}
impl HestonFourierHurstRiskPlan {
    /// Explicit log-transform H derivative. Unlike price(), this performs a new solve.
    pub fn log_transform_derivative(&self, z: C) -> Result<C, FourierError> {
        self.price_plan
            .riccati
            .log_transform_hurst_derivative(z, &self.weights)
            .map(|(_, d)| d)
    }
    /// No parameter bump, recalibration or additional Riccati solve per strike.
    pub fn price(
        &self,
        forward: f64,
        strike: f64,
        discount: f64,
    ) -> Result<HestonFourierHurstRisk, FourierError> {
        let price = self.price_plan.price(forward, strike, discount)?;
        let n = self.price_plan.config.integration_intervals;
        let du = self.price_plan.config.cutoff / n as f64;
        let x = forward.ln() - strike.ln();
        let scale = discount * forward.sqrt() * strike.sqrt() / std::f64::consts::PI;
        let (mut fine, mut coarse, mut tail) =
            (NeumaierSum::new(), NeumaierSum::new(), NeumaierSum::new());
        for (j, &node) in self.derivatives.iter().enumerate() {
            let value = (C::new(0.0, j as f64 * du * x).exp() * node).re;
            fine.add(simpson_weight(j, n) * value);
            if j.is_multiple_of(2) {
                coarse.add(simpson_weight(j / 2, n / 2) * value);
            }
            if j >= n / 2 {
                tail.add(simpson_weight(j - n / 2, n / 2) * node.abs());
            }
        }
        let sensitivity = scale * du * fine.total() / 3.0;
        let result = HestonFourierHurstRisk {
            price,
            hurst_sensitivity: sensitivity,
            quadrature_difference: (sensitivity - scale * (2.0 * du) * coarse.total() / 3.0).abs(),
            tail_indicator: scale * du * tail.total() / 3.0,
        };
        if [
            result.hurst_sensitivity,
            result.quadrature_difference,
            result.tail_indicator,
        ]
        .iter()
        .any(|x| !x.is_finite())
        {
            return Err(FourierError::NumericalFailure(
                "non-finite Hurst sensitivity",
            ));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{HestonFourierConfig, RiccatiPlan};
    use super::*;
    use crate::rough_volatility::RoughHeston;
    #[test]
    fn hurst_budget_checked_before_transform_allocation() {
        let config = HestonFourierConfig::new(8192, 8192, 128.0).unwrap();
        let model = RoughHeston::new(0.1, 0.04, 0.7, 0.055, 0.18, -0.65)
            .unwrap()
            .into();
        let plan = HestonFourierPlan {
            config,
            riccati: RiccatiPlan::new(&model, 1.0, config).unwrap(),
            differences: vec![],
        };
        assert!(matches!(
            plan.hurst_risk_plan(),
            Err(FourierError::InvalidInput("Hurst-risk work limit"))
        ));
    }
}
