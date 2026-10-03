//! Experimental European forward prices for continuous-time rough/lifted Heston.
//! See docs/models/rough-heston-fourier.md for definitions, references and limits.

mod calibration;
pub use calibration::{
    HestonCalibrationError, HestonCalibrationEvaluation, HestonCalibrationParameter,
    HestonCalibrationProblem, HestonCalibrationQuote, HestonCalibrationResult,
    HestonCalibrationVariable,
};
mod greeks;
mod hurst_risk;
pub use hurst_risk::{HestonFourierHurstRisk, HestonFourierHurstRiskPlan};
mod parameter_risk;
mod riccati;
use crate::rough_volatility::RoughVolatilityModel;
pub use greeks::HestonFourierGreeks;
pub use parameter_risk::{
    HestonFourierParameterRisk, HestonFourierParameterRiskPlan, HestonParameterSensitivities,
};
use pricing_numerics::{Complex64 as C, NeumaierSum, standard_normal_cdf};
use riccati::RiccatiPlan;
use std::{error::Error, fmt};

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FourierError {
    UnsupportedModel,
    InvalidInput(&'static str),
    NumericalFailure(&'static str),
}
impl fmt::Display for FourierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedModel => write!(
                f,
                "Fourier pricing supports only Rough Heston and Lifted Heston"
            ),
            Self::InvalidInput(s) => write!(f, "invalid Fourier input: {s}"),
            Self::NumericalFailure(s) => write!(f, "Fourier numerical failure: {s}"),
        }
    }
}
impl Error for FourierError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonFourierConfig {
    time_steps: usize,
    integration_intervals: usize,
    cutoff: f64,
}
impl HestonFourierConfig {
    pub fn new(
        time_steps: usize,
        integration_intervals: usize,
        cutoff: f64,
    ) -> Result<Self, FourierError> {
        if !(2..=8192).contains(&time_steps) {
            return Err(FourierError::InvalidInput("time_steps must be 2..=8192"));
        }
        if !(8..=8192).contains(&integration_intervals) || !integration_intervals.is_multiple_of(4)
        {
            return Err(FourierError::InvalidInput(
                "integration_intervals must be 8..=8192 and divisible by 4",
            ));
        }
        if !cutoff.is_finite() || cutoff <= 0.0 || cutoff > 10000.0 {
            return Err(FourierError::InvalidInput(
                "cutoff must be finite in (0,10000]",
            ));
        }
        Ok(Self {
            time_steps,
            integration_intervals,
            cutoff,
        })
    }
    #[must_use]
    pub const fn time_steps(self) -> usize {
        self.time_steps
    }
    #[must_use]
    pub const fn integration_intervals(self) -> usize {
        self.integration_intervals
    }
    #[must_use]
    pub const fn cutoff(self) -> f64 {
        self.cutoff
    }
}
impl Default for HestonFourierConfig {
    fn default() -> Self {
        Self {
            time_steps: 512,
            integration_intervals: 512,
            cutoff: 128.0,
        }
    }
}

/// Deterministic estimates. Neither diagnostic is an error bound; neither
/// includes Riccati time-grid, model or Monte Carlo uncertainty.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonFourierPrice {
    pub call: f64,
    pub put: f64,
    /// Absolute difference between Simpson grids of N and N/2 on [0,cutoff].
    pub quadrature_difference: f64,
    /// Integral of the absolute control-variate integrand envelope on
    /// [cutoff/2,cutoff]. It does NOT bound the omitted infinite tail.
    pub tail_indicator: f64,
}

/// Caches a maturity's half-moment transform for reuse across strikes/forwards.
/// The input is a positive tradable forward with constant model parameters,
/// not a physical-Spot cash-dividend request. Discounting is supplied at pricing.
#[derive(Clone, Debug)]
pub struct HestonFourierPlan {
    config: HestonFourierConfig,
    riccati: RiccatiPlan,
    differences: Vec<C>,
}
impl HestonFourierPlan {
    pub fn compile(
        model: RoughVolatilityModel,
        maturity: f64,
        config: HestonFourierConfig,
    ) -> Result<Self, FourierError> {
        if !maturity.is_finite() || maturity < 0.0 {
            return Err(FourierError::InvalidInput(
                "maturity must be finite and nonnegative",
            ));
        }
        // Bound total arithmetic work before allocating/evaluating transform nodes.
        let cost = match &model {
            RoughVolatilityModel::RoughHeston(_) => (config.time_steps as u64).pow(2),
            RoughVolatilityModel::LiftedHeston(m) => {
                config.time_steps as u64 * m.weights().len() as u64
            }
            _ => return Err(FourierError::UnsupportedModel),
        };
        if cost * (config.integration_intervals as u64 + 1) > 8_000_000_000 {
            return Err(FourierError::InvalidInput("Fourier work limit"));
        }
        let riccati = RiccatiPlan::new(&model, maturity, config)?;
        let variance = riccati.control_variance();
        if !variance.is_finite() {
            return Err(FourierError::NumericalFailure("control variance overflow"));
        }
        let mut differences = Vec::with_capacity(config.integration_intervals + 1);
        for j in 0..=config.integration_intervals {
            let u = j as f64 * config.cutoff / config.integration_intervals as f64;
            let z = C::new(0.5, u);
            let control = ((z * z - z) * (0.5 * variance)).exp();
            let model = riccati.log_transform(z)?.exp();
            let value = (control - model) / (u * u + 0.25);
            if !value.is_finite() {
                return Err(FourierError::NumericalFailure("Fourier transform node"));
            }
            differences.push(value);
        }
        Ok(Self {
            config,
            riccati,
            differences,
        })
    }
    #[must_use]
    pub const fn config(&self) -> HestonFourierConfig {
        self.config
    }
    /// Returns log E[exp(z log(F_T/F_0))], only in the closed moment strip
    /// 0<=Re(z)<=1. Complex components are returned without taking a log of a CF.
    pub fn log_transform(&self, z: C) -> Result<C, FourierError> {
        self.riccati.log_transform(z)
    }
    pub fn characteristic_function(&self, frequency: f64) -> Result<C, FourierError> {
        Ok(self.log_transform(C::new(0.0, frequency))?.exp())
    }
    pub fn price(
        &self,
        forward: f64,
        strike: f64,
        discount: f64,
    ) -> Result<HestonFourierPrice, FourierError> {
        if [forward, strike, discount]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(FourierError::InvalidInput(
                "forward, strike and discount must be finite and positive",
            ));
        }
        let n = self.config.integration_intervals;
        let du = self.config.cutoff / n as f64;
        let log_moneyness = forward.ln() - strike.ln();
        let scale = discount * forward.sqrt() * strike.sqrt() / std::f64::consts::PI;
        let mut fine = NeumaierSum::new();
        let mut coarse = NeumaierSum::new();
        let mut tail = NeumaierSum::new();
        for (j, &z) in self.differences.iter().enumerate() {
            let phase = C::new(0.0, j as f64 * du * log_moneyness).exp();
            let value = (z * phase).re;
            fine.add(simpson_weight(j, n) * value);
            if j.is_multiple_of(2) {
                coarse.add(simpson_weight(j / 2, n / 2) * value);
            }
            if j >= n / 2 {
                tail.add(simpson_weight(j - n / 2, n / 2) * z.abs());
            }
        }
        let correction = scale * du * fine.total() / 3.0;
        let coarse_correction = scale * (2.0 * du) * coarse.total() / 3.0;
        let (black_call, black_put) =
            black_prices(forward, strike, discount, self.riccati.control_variance());
        let result = HestonFourierPrice {
            call: black_call + correction,
            put: black_put + correction,
            quadrature_difference: (correction - coarse_correction).abs(),
            tail_indicator: scale * du * tail.total() / 3.0,
        };
        if [
            result.call,
            result.put,
            result.quadrature_difference,
            result.tail_indicator,
        ]
        .iter()
        .any(|x| !x.is_finite())
        {
            return Err(FourierError::NumericalFailure("non-finite price"));
        }
        let eps = 1e-10 * (discount * forward + discount * strike).max(1.0);
        if result.call < discount * (forward - strike).max(0.0) - eps
            || result.call > discount * forward + eps
            || result.put < discount * (strike - forward).max(0.0) - eps
            || result.put > discount * strike + eps
        {
            return Err(FourierError::NumericalFailure(
                "option bounds; refine time/frequency grid or extend cutoff",
            ));
        }
        Ok(result)
    }
}
fn simpson_weight(j: usize, n: usize) -> f64 {
    if j == 0 || j == n {
        1.0
    } else if j.is_multiple_of(2) {
        2.0
    } else {
        4.0
    }
}
fn black_prices(f: f64, k: f64, d: f64, w: f64) -> (f64, f64) {
    if w == 0.0 {
        return (d * (f - k).max(0.0), d * (k - f).max(0.0));
    }
    let v = w.sqrt();
    let d1 = (f.ln() - k.ln()) / v + 0.5 * v;
    let d2 = d1 - v;
    // Compute the OTM side directly; use parity for the ITM side.
    if f <= k {
        let call = d * (f * standard_normal_cdf(d1) - k * standard_normal_cdf(d2));
        (call, call + d * (k - f))
    } else {
        let put = d * (k * standard_normal_cdf(-d2) - f * standard_normal_cdf(-d1));
        (put + d * (f - k), put)
    }
}

mod iv_calibration;
pub use iv_calibration::{
    HestonIvCalibrationEvaluation, HestonIvCalibrationProblem, HestonIvCalibrationQuote,
    HestonIvCalibrationResult, HestonIvGridValidation, HestonIvRefinementOptions,
    HestonIvRefinementResult, HestonIvRefinementStage,
};
