//! Analytic forward derivatives of the cached, finite Fourier quadrature.
//! Model parameters, maturity, strike and discount factor are held fixed.

use super::{FourierError, HestonFourierPlan, HestonFourierPrice, simpson_weight};
use pricing_numerics::{Complex64 as C, NeumaierSum, standard_normal_cdf, standard_normal_pdf};

/// European forward sensitivities, not physical-Spot or recalibrated Greeks.
/// Diagnostics compare frequency grids or retained envelopes, not total error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonFourierGreeks {
    /// Exactly the result of the unchanged `HestonFourierPlan::price` method.
    pub price: HestonFourierPrice,
    pub call_forward_delta: f64,
    pub put_forward_delta: f64,
    /// Common forward Gamma of call and put, per unit of forward squared.
    pub forward_gamma: f64,
    /// Absolute Delta difference between Simpson N and N/2 grids.
    pub delta_quadrature_difference: f64,
    /// Absolute Gamma difference between Simpson N and N/2 grids.
    pub gamma_quadrature_difference: f64,
    /// Delta correction envelope integrated over [cutoff/2, cutoff].
    /// This is NOT an upper bound for the omitted infinite tail.
    pub delta_tail_indicator: f64,
    /// Gamma correction envelope on the same retained half-interval.
    /// It does NOT include Riccati error or bound the omitted tail.
    pub gamma_tail_indicator: f64,
}

impl HestonFourierPlan {
    /// Prices and analytic Forward Delta/Gamma without bumps or new Riccati solves.
    /// The cached model, maturity, strike and discount factor remain fixed.
    ///
    /// At zero terminal variance the derivatives are piecewise intrinsic, except
    /// at the ATM kink, where this method fails explicitly. Non-degenerate laws
    /// with zero initial variance are rejected: the existing zero-variance Black
    /// control is singular and its truncated derivatives are not reliable.
    pub fn price_and_greeks(
        &self,
        forward: f64,
        strike: f64,
        discount: f64,
    ) -> Result<HestonFourierGreeks, FourierError> {
        let price = self.price(forward, strike, discount)?;
        let variance = self.riccati.control_variance();
        if self.riccati.is_zero_variance() {
            if forward == strike {
                return Err(FourierError::InvalidInput(
                    "forward derivatives are undefined at a deterministic ATM payoff",
                ));
            }
            return Ok(HestonFourierGreeks {
                price,
                call_forward_delta: if forward > strike { discount } else { 0.0 },
                put_forward_delta: if forward < strike { -discount } else { 0.0 },
                forward_gamma: 0.0,
                delta_quadrature_difference: 0.0,
                gamma_quadrature_difference: 0.0,
                delta_tail_indicator: 0.0,
                gamma_tail_indicator: 0.0,
            });
        }
        if variance <= 0.0 {
            return Err(FourierError::InvalidInput(
                "forward Greeks require positive control variance for a non-degenerate law",
            ));
        }
        let n = self.config.integration_intervals;
        let du = self.config.cutoff / n as f64;
        let log_moneyness = forward.ln() - strike.ln();
        // Division before multiplication avoids squaring the forward explicitly.
        let delta_scale = discount * (strike.sqrt() / forward.sqrt()) / std::f64::consts::PI;
        let gamma_scale = delta_scale / forward;
        let mut fine_delta = NeumaierSum::new();
        let mut coarse_delta = NeumaierSum::new();
        let mut fine_gamma = NeumaierSum::new();
        let mut coarse_gamma = NeumaierSum::new();
        let mut delta_tail = NeumaierSum::new();
        let mut gamma_tail = NeumaierSum::new();
        for (j, &z) in self.differences.iter().enumerate() {
            let u = j as f64 * du;
            let phase = C::new(0.0, u * log_moneyness).exp();
            let phased = z * phase;
            // d/dF [sqrt(F) exp(i*u*log(F/K))] supplies (1/2+i*u)/F.
            // The second derivative supplies -(u*u+1/4)/(F*F).
            let delta = (C::new(0.5, u) * phased).re;
            let gamma = -(u * u + 0.25) * phased.re;
            fine_delta.add(simpson_weight(j, n) * delta);
            fine_gamma.add(simpson_weight(j, n) * gamma);
            if j.is_multiple_of(2) {
                coarse_delta.add(simpson_weight(j / 2, n / 2) * delta);
                coarse_gamma.add(simpson_weight(j / 2, n / 2) * gamma);
            }
            if j >= n / 2 {
                let weight = simpson_weight(j - n / 2, n / 2);
                delta_tail.add(weight * (u * u + 0.25).sqrt() * z.abs());
                gamma_tail.add(weight * (u * u + 0.25) * z.abs());
            }
        }
        let delta_correction = delta_scale * du * fine_delta.total() / 3.0;
        let gamma_correction = gamma_scale * du * fine_gamma.total() / 3.0;
        let root_variance = variance.sqrt();
        let d1 = log_moneyness / root_variance + 0.5 * root_variance;
        let result = HestonFourierGreeks {
            price,
            call_forward_delta: discount * standard_normal_cdf(d1) + delta_correction,
            put_forward_delta: -discount * standard_normal_cdf(-d1) + delta_correction,
            forward_gamma: (discount / forward) * (standard_normal_pdf(d1) / root_variance)
                + gamma_correction,
            delta_quadrature_difference: (delta_correction
                - delta_scale * (2.0 * du) * coarse_delta.total() / 3.0)
                .abs(),
            gamma_quadrature_difference: (gamma_correction
                - gamma_scale * (2.0 * du) * coarse_gamma.total() / 3.0)
                .abs(),
            delta_tail_indicator: delta_scale * du * delta_tail.total() / 3.0,
            gamma_tail_indicator: gamma_scale * du * gamma_tail.total() / 3.0,
        };
        if [
            result.call_forward_delta,
            result.put_forward_delta,
            result.forward_gamma,
            result.delta_quadrature_difference,
            result.gamma_quadrature_difference,
            result.delta_tail_indicator,
            result.gamma_tail_indicator,
        ]
        .iter()
        .any(|x| !x.is_finite())
        {
            return Err(FourierError::NumericalFailure("non-finite forward Greek"));
        }
        // Necessary shape checks only, not accuracy certificates. Never clamp.
        let eps = 1e-10 * discount;
        if result.call_forward_delta < -eps
            || result.call_forward_delta > discount + eps
            || result.put_forward_delta < -discount - eps
            || result.put_forward_delta > eps
            || result.forward_gamma * forward < -eps
        {
            return Err(FourierError::NumericalFailure(
                "forward Greek bounds; refine time/frequency grid or extend cutoff",
            ));
        }
        Ok(result)
    }
}
