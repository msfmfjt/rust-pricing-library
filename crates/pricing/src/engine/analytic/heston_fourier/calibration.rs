//! Experimental multi-expiry price calibration with the Fourier tangent Jacobian.
use super::{FourierError, HestonFourierConfig, HestonFourierPlan};
use crate::rough_volatility::{LiftedHeston, RoughHeston, RoughVolatilityModel};
use pricing_numerics::least_squares::{
    LeastSquaresError, LeastSquaresEvaluation, LeastSquaresOptions, LeastSquaresResult,
    LeastSquaresVariable, bounded_least_squares,
};
use std::{error::Error, fmt};

/// Fitted coordinates. An explicit finite lift has no Hurst coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HestonCalibrationParameter {
    InitialVariance,
    MeanReversion,
    LongRunVariance,
    VolOfVol,
    Correlation,
    Hurst,
}
impl HestonCalibrationParameter {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::InitialVariance => "initial_variance",
            Self::MeanReversion => "mean_reversion",
            Self::LongRunVariance => "long_run_variance",
            Self::VolOfVol => "vol_of_vol",
            Self::Correlation => "correlation",
            Self::Hurst => "hurst",
        }
    }
    const fn index(self) -> usize {
        match self {
            Self::InitialVariance => 0,
            Self::MeanReversion => 1,
            Self::LongRunVariance => 2,
            Self::VolOfVol => 3,
            Self::Correlation => 4,
            Self::Hurst => 5,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonCalibrationVariable {
    pub parameter: HestonCalibrationParameter,
    pub lower: f64,
    pub upper: f64,
    pub scale: f64,
}
/// One positive-forward European price observation, with maturity in years.
/// `price_scale` is a fixed divisor: residual=(model_price-target)/price_scale.
/// It is neither a probability weight nor automatically a bid/ask or IV error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonCalibrationQuote {
    pub maturity: f64,
    pub forward: f64,
    pub strike: f64,
    pub discount: f64,
    pub is_call: bool,
    pub target_price: f64,
    pub price_scale: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct HestonCalibrationEvaluation {
    pub model_prices: Vec<f64>,
    pub scaled_residuals: Vec<f64>,
    /// Row-major, quote order then selected-variable order. Physical parameter
    /// derivatives divided by price_scale; variable scaling is NOT included.
    pub jacobian: Vec<f64>,
    pub quadrature_differences: Vec<f64>,
    pub tail_indicators: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct HestonCalibrationResult {
    pub model: RoughVolatilityModel,
    pub evaluation: HestonCalibrationEvaluation,
    pub optimizer: LeastSquaresResult,
    /// All quoted scaled residuals meet options.residual_tolerance. This does
    /// not establish parameter recovery, global optimality or continuum accuracy.
    pub fit_achieved: bool,
    /// Optimizer calls PLUS one final independent evaluation of the final point.
    pub evaluations: usize,
    pub active_bounds: Vec<bool>,
}
#[derive(Debug)]
pub enum HestonCalibrationError {
    InvalidInput(&'static str),
    Fourier(FourierError),
    Optimizer(LeastSquaresError<FourierError>),
}
impl fmt::Display for HestonCalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "invalid Heston calibration input: {s}"),
            Self::Fourier(e) => e.fmt(f),
            Self::Optimizer(e) => e.fmt(f),
        }
    }
}
impl Error for HestonCalibrationError {}
impl From<FourierError> for HestonCalibrationError {
    fn from(e: FourierError) -> Self {
        Self::Fourier(e)
    }
}
#[derive(Clone, Debug)]
pub struct HestonCalibrationProblem {
    initial_model: RoughVolatilityModel,
    quotes: Vec<HestonCalibrationQuote>,
    variables: Vec<HestonCalibrationVariable>,
    fourier: HestonFourierConfig,
    /// Each maturity appears once; input order of the quotes is preserved.
    groups: Vec<(f64, Vec<usize>)>,
}
fn parameters(model: &RoughVolatilityModel) -> Result<[f64; 6], FourierError> {
    let m = match model {
        RoughVolatilityModel::RoughHeston(m) => m,
        RoughVolatilityModel::LiftedHeston(m) => m.heston_parameters(),
        _ => return Err(FourierError::UnsupportedModel),
    };
    Ok([
        m.initial_variance(),
        m.mean_reversion(),
        m.long_run_variance(),
        m.vol_of_vol(),
        m.correlation(),
        m.hurst(),
    ])
}
impl HestonCalibrationProblem {
    pub fn new(
        initial_model: RoughVolatilityModel,
        quotes: Vec<HestonCalibrationQuote>,
        variables: Vec<HestonCalibrationVariable>,
        fourier: HestonFourierConfig,
    ) -> Result<Self, HestonCalibrationError> {
        let p = parameters(&initial_model)?;
        if p[0] <= 0.0 {
            return Err(HestonCalibrationError::InvalidInput(
                "positive initial variance required",
            ));
        }
        if variables.is_empty()
            || variables.len() > 6
            || quotes.len() < variables.len()
            || quotes.len() > 4096
        {
            return Err(HestonCalibrationError::InvalidInput(
                "need 1..=6 variables and at least as many quotes, at most 4096",
            ));
        }
        let mut seen = [false; 6];
        for v in &variables {
            let j = v.parameter.index();
            if seen[j] {
                return Err(HestonCalibrationError::InvalidInput(
                    "duplicate calibration variable",
                ));
            }
            seen[j] = true;
            if j == 5 && !matches!(initial_model, RoughVolatilityModel::RoughHeston(_)) {
                return Err(HestonCalibrationError::InvalidInput(
                    "a finite lift has no Hurst-to-kernel mapping",
                ));
            }
            let domain_ok = match j {
                0 => v.lower > 0.0,
                1..=3 => v.lower >= 0.0,
                4 => v.lower >= -1.0 && v.upper <= 1.0,
                5 => v.lower > 0.0 && v.upper <= 0.5,
                _ => false,
            };
            if !domain_ok
                || !v.lower.is_finite()
                || !v.upper.is_finite()
                || v.lower >= v.upper
                || !v.scale.is_finite()
                || v.scale <= 0.0
                || !((v.upper - v.lower) / v.scale).is_finite()
                || p[j] < v.lower
                || p[j] > v.upper
            {
                return Err(HestonCalibrationError::InvalidInput(
                    "variable bounds, scale or initial value",
                ));
            }
        }
        let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
        for (i, q) in quotes.iter().enumerate() {
            if [q.maturity, q.forward, q.strike, q.discount, q.price_scale]
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.0)
                || !q.target_price.is_finite()
            {
                return Err(HestonCalibrationError::InvalidInput(
                    "quote inputs must be finite; maturity/F/K/D/scale positive",
                ));
            }
            let (a, b) = if q.is_call {
                (q.forward, q.strike)
            } else {
                (q.strike, q.forward)
            };
            let lo = q.discount * (a - b).max(0.0);
            let hi = q.discount * a;
            if !lo.is_finite() || !hi.is_finite() || q.target_price < lo || q.target_price > hi {
                return Err(HestonCalibrationError::InvalidInput(
                    "quote violates individual European price bounds",
                ));
            }
            if let Some((_, indices)) = groups.iter_mut().find(|(t, _)| *t == q.maturity) {
                indices.push(i);
            } else {
                groups.push((q.maturity, vec![i]));
            }
        }
        if groups.len() > 64 {
            return Err(HestonCalibrationError::InvalidInput(
                "at most 64 distinct maturities",
            ));
        }
        groups.sort_by(|a, b| a.0.total_cmp(&b.0));
        let problem = Self {
            initial_model,
            quotes,
            variables,
            fourier,
            groups,
        };
        problem.check_work(1)?;
        Ok(problem)
    }
    #[must_use]
    pub fn initial_parameters(&self) -> Vec<f64> {
        let p = parameters(&self.initial_model).expect("validated Heston family");
        self.variables
            .iter()
            .map(|v| p[v.parameter.index()])
            .collect()
    }
    #[must_use]
    pub fn variables(&self) -> &[HestonCalibrationVariable] {
        &self.variables
    }
    #[must_use]
    pub fn quotes(&self) -> &[HestonCalibrationQuote] {
        &self.quotes
    }
    #[must_use]
    pub fn fourier_config(&self) -> HestonFourierConfig {
        self.fourier
    }
    #[must_use]
    pub fn maturity_count(&self) -> usize {
        self.groups.len()
    }
    pub(super) fn check_work(&self, evaluations: usize) -> Result<(), HestonCalibrationError> {
        let steps = self.fourier.time_steps() as u128;
        let nodes = self.fourier.integration_intervals() as u128 + 1;
        let cost = match &self.initial_model {
            RoughVolatilityModel::RoughHeston(_) => steps * steps,
            RoughVolatilityModel::LiftedHeston(m) => steps * m.weights().len() as u128,
            _ => unreachable!(),
        };
        let scalar = self
            .variables
            .iter()
            .any(|v| v.parameter != HestonCalibrationParameter::Hurst);
        let hurst = self
            .variables
            .iter()
            .any(|v| v.parameter == HestonCalibrationParameter::Hurst);
        let multiplier = 1 + 6 * u128::from(scalar) + 2 * u128::from(hurst);
        if cost * nodes * multiplier * self.groups.len() as u128 * evaluations as u128
            > 1_000_000_000_000
        {
            return Err(HestonCalibrationError::InvalidInput(
                "calibration work limit; reduce grid, maturities or evaluation budget",
            ));
        }
        Ok(())
    }
    pub(super) fn model_at(&self, x: &[f64]) -> Result<RoughVolatilityModel, FourierError> {
        if x.len() != self.variables.len() {
            return Err(FourierError::InvalidInput("calibration parameter count"));
        }
        let mut p = parameters(&self.initial_model)?;
        for (&value, v) in x.iter().zip(&self.variables) {
            if !value.is_finite() || value < v.lower || value > v.upper {
                return Err(FourierError::InvalidInput(
                    "calibration point outside bounds",
                ));
            }
            p[v.parameter.index()] = value;
        }
        let model = match &self.initial_model {
            RoughVolatilityModel::RoughHeston(_) => {
                RoughHeston::new(p[5], p[0], p[1], p[2], p[3], p[4]).map(RoughVolatilityModel::from)
            }
            RoughVolatilityModel::LiftedHeston(m) => LiftedHeston::new(
                p[0],
                p[1],
                p[2],
                p[3],
                p[4],
                m.weights().to_vec(),
                m.rates().to_vec(),
            )
            .map(RoughVolatilityModel::from),
            _ => return Err(FourierError::UnsupportedModel),
        };
        model.map_err(|_| FourierError::InvalidInput("calibration model parameters"))
    }
    /// Evaluate all quotes and the analytic residual Jacobian. Points outside
    /// the declared box are errors, not clamped. One cached plan per maturity.
    pub fn evaluate(&self, x: &[f64]) -> Result<HestonCalibrationEvaluation, FourierError> {
        let model = self.model_at(x)?;
        let m = self.quotes.len();
        let n = self.variables.len();
        let mut out = HestonCalibrationEvaluation {
            model_prices: vec![0.0; m],
            scaled_residuals: vec![0.0; m],
            jacobian: vec![0.0; m * n],
            quadrature_differences: vec![0.0; m],
            tail_indicators: vec![0.0; m],
        };
        let scalar = self
            .variables
            .iter()
            .any(|v| v.parameter != HestonCalibrationParameter::Hurst);
        let hurst = self
            .variables
            .iter()
            .any(|v| v.parameter == HestonCalibrationParameter::Hurst);
        for (t, indices) in &self.groups {
            let plan = HestonFourierPlan::compile(model.clone(), *t, self.fourier)?;
            let risk = if scalar {
                Some(plan.parameter_risk_plan()?)
            } else {
                None
            };
            let h_risk = if hurst {
                Some(plan.hurst_risk_plan()?)
            } else {
                None
            };
            for &i in indices {
                let q = self.quotes[i];
                let s = risk
                    .as_ref()
                    .map(|r| r.price(q.forward, q.strike, q.discount))
                    .transpose()?;
                let h = h_risk
                    .as_ref()
                    .map(|r| r.price(q.forward, q.strike, q.discount))
                    .transpose()?;
                let price = if let Some(r) = s {
                    r.price
                } else {
                    h.expect("at least one direction").price
                };
                out.model_prices[i] = if q.is_call { price.call } else { price.put };
                out.scaled_residuals[i] = (out.model_prices[i] - q.target_price) / q.price_scale;
                out.quadrature_differences[i] = price.quadrature_difference;
                out.tail_indicators[i] = price.tail_indicator;
                for (j, v) in self.variables.iter().enumerate() {
                    let d = if v.parameter == HestonCalibrationParameter::Hurst {
                        h.expect("Hurst plan").hurst_sensitivity
                    } else {
                        s.expect("scalar plan").sensitivities.as_array()[v.parameter.index()]
                    };
                    out.jacobian[i * n + j] = d / q.price_scale;
                }
            }
        }
        if out
            .scaled_residuals
            .iter()
            .chain(&out.jacobian)
            .any(|v| !v.is_finite())
        {
            return Err(FourierError::NumericalFailure(
                "scaled calibration residual/Jacobian overflow",
            ));
        }
        Ok(out)
    }
    /// Local box-constrained weighted PRICE fit. The evaluation budget includes
    /// one reserved final re-evaluation. Failure to reach the fit tolerance is
    /// returned as a result, not disguised as successful parameter recovery.
    pub fn calibrate(
        &self,
        options: LeastSquaresOptions,
    ) -> Result<HestonCalibrationResult, HestonCalibrationError> {
        options
            .validate()
            .map_err(HestonCalibrationError::InvalidInput)?;
        if options.max_evaluations < 2 {
            return Err(HestonCalibrationError::InvalidInput(
                "reserve at least two evaluations",
            ));
        }
        self.check_work(options.max_evaluations)?;
        let vars: Vec<_> = self
            .variables
            .iter()
            .map(|v| LeastSquaresVariable {
                lower: v.lower,
                upper: v.upper,
                scale: v.scale,
            })
            .collect();
        let optimization_options = LeastSquaresOptions {
            max_evaluations: options.max_evaluations - 1,
            ..options
        };
        let optimizer = bounded_least_squares(
            &self.initial_parameters(),
            &vars,
            optimization_options,
            |x| {
                let e = self.evaluate(x)?;
                Ok(LeastSquaresEvaluation {
                    residuals: e.scaled_residuals,
                    jacobian: e.jacobian,
                })
            },
        )
        .map_err(HestonCalibrationError::Optimizer)?;
        let model = self.model_at(&optimizer.parameters)?;
        let evaluation = self.evaluate(&optimizer.parameters)?;
        // Do not discard quotes or accept different final evaluation arithmetic.
        if evaluation.scaled_residuals != optimizer.evaluation.residuals
            || evaluation.jacobian != optimizer.evaluation.jacobian
        {
            return Err(HestonCalibrationError::InvalidInput(
                "non-reproducible final calibration evaluation",
            ));
        }
        let fit_achieved = evaluation
            .scaled_residuals
            .iter()
            .all(|r| r.abs() <= options.residual_tolerance);
        let evaluations = optimizer.evaluations + 1;
        let active_bounds = optimizer
            .parameters
            .iter()
            .zip(&self.variables)
            .map(|(&x, v)| x == v.lower || x == v.upper)
            .collect();
        Ok(HestonCalibrationResult {
            model,
            evaluation,
            optimizer,
            fit_achieved,
            evaluations,
            active_bounds,
        })
    }
}
