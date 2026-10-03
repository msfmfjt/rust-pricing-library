//! Exact Black implied-volatility residuals. This is not target-Vega-weighted price fitting.
use super::calibration::{
    HestonCalibrationError, HestonCalibrationProblem, HestonCalibrationQuote,
    HestonCalibrationVariable,
};
use super::{FourierError, HestonFourierConfig};
use crate::market::{ImpliedVarianceSurface, StandardSsvi};
use crate::rough_volatility::RoughVolatilityModel;
use pricing_numerics::least_squares::{
    LeastSquaresEvaluation, LeastSquaresOptions, LeastSquaresResult, LeastSquaresVariable,
    bounded_least_squares,
};
use pricing_numerics::{standard_normal_cdf, standard_normal_pdf};

// Explicit experimental numerical domain; no clipping or Vega floor is applied.
const MAX_STDDEV: f64 = 8.0;
const MIN_NORMALIZED_VEGA: f64 = 1.0e-10;
const IV_TOLERANCE: f64 = 1.0e-12;

/// Positive-forward, positive-maturity Black IV, in absolute annualized volatility units.
/// The residual is (model_IV - target_volatility) / iv_scale. 0.0001 is one IV bp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HestonIvCalibrationQuote {
    pub maturity: f64,
    pub forward: f64,
    pub strike: f64,
    pub discount: f64,
    pub is_call: bool,
    pub target_volatility: f64,
    pub iv_scale: f64,
}
impl HestonIvCalibrationQuote {
    /// Sample the existing validated standard SSVI surface at log(K/F).
    /// Chooses the OTM option side (call at ATM). Forward/dividend conventions
    /// remain the caller's responsibility; no Spot-to-forward mapping is inferred.
    pub fn from_ssvi(
        surface: &StandardSsvi,
        maturity: f64,
        forward: f64,
        strike: f64,
        discount: f64,
        iv_scale: f64,
    ) -> Result<Self, HestonCalibrationError> {
        if [maturity, forward, strike, discount, iv_scale]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(HestonCalibrationError::InvalidInput(
                "positive finite SSVI quote coordinates required",
            ));
        }
        let w = surface
            .total_variance_derivatives(maturity, strike.ln() - forward.ln())
            .map_err(|_| HestonCalibrationError::InvalidInput("invalid standard SSVI query"))?
            .total_variance;
        let q = Self {
            maturity,
            forward,
            strike,
            discount,
            is_call: strike >= forward,
            target_volatility: (w / maturity).sqrt(),
            iv_scale,
        };
        q.to_price_quote()?;
        Ok(q)
    }
    /// Materialize a PRICE quote, without Vega weighting (price_scale=1).
    /// Use HestonIvCalibrationProblem for the exact IV objective instead.
    pub fn to_price_quote(self) -> Result<HestonCalibrationQuote, HestonCalibrationError> {
        if [
            self.maturity,
            self.forward,
            self.strike,
            self.discount,
            self.iv_scale,
            self.target_volatility,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(HestonCalibrationError::InvalidInput(
                "positive finite IV quote inputs required",
            ));
        }
        let b = BlackCoordinates::new(
            self.forward,
            self.strike,
            self.discount,
            self.maturity,
            self.is_call,
        )?;
        let sd = self.target_volatility * b.sqrt_time;
        if !sd.is_finite() || sd <= 0.0 || sd > MAX_STDDEV {
            return Err(HestonCalibrationError::InvalidInput(
                "target total standard deviation must be in (0,8]",
            ));
        }
        let target_price = b.scale * (b.otm(sd) + b.intrinsic);
        // Reject input prices whose time value cannot be reliably represented/inverted.
        let (iv, _) = b.invert(target_price)?;
        if (iv - self.target_volatility).abs() > 1.0e-8 * self.target_volatility.max(1.0) {
            return Err(HestonCalibrationError::InvalidInput(
                "target IV/price round trip is ill-conditioned",
            ));
        }
        Ok(HestonCalibrationQuote {
            maturity: self.maturity,
            forward: self.forward,
            strike: self.strike,
            discount: self.discount,
            is_call: self.is_call,
            target_price,
            price_scale: 1.0,
        })
    }
}

/// One normalized OTM Black calculation, shared by call and put inversion.
struct BlackCoordinates {
    ratio: f64,
    log_ratio: f64,
    scale: f64,
    intrinsic: f64,
    sqrt_time: f64,
}
impl BlackCoordinates {
    fn new(f: f64, k: f64, d: f64, t: f64, is_call: bool) -> Result<Self, FourierError> {
        if [f, k, d, t].iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err(FourierError::InvalidInput(
                "positive finite Black coordinates required",
            ));
        }
        let maximum = f.max(k);
        let ratio = f.min(k) / maximum;
        let scale = d * maximum;
        let intrinsic = if is_call {
            (f - k).max(0.0)
        } else {
            (k - f).max(0.0)
        } / maximum;
        if ratio <= 0.0 || !scale.is_finite() || scale <= 0.0 {
            return Err(FourierError::NumericalFailure(
                "Black normalization overflow/underflow",
            ));
        }
        Ok(Self {
            ratio,
            log_ratio: ratio.ln(),
            scale,
            intrinsic,
            sqrt_time: t.sqrt(),
        })
    }
    fn otm(&self, sd: f64) -> f64 {
        if sd == 0.0 {
            return 0.0;
        }
        let d1 = self.log_ratio / sd + 0.5 * sd;
        self.ratio * standard_normal_cdf(d1) - standard_normal_cdf(d1 - sd)
    }
    fn invert(&self, price: f64) -> Result<(f64, f64), FourierError> {
        let value = price / self.scale - self.intrinsic;
        // Strict interior bounds, before root finding. Never clamp an invalid price.
        if !price.is_finite()
            || !value.is_finite()
            || value <= 32.0 * f64::EPSILON
            || value >= self.ratio
            || value > self.otm(MAX_STDDEV)
        {
            return Err(FourierError::NumericalFailure(
                "Black IV requires resolvable interior price and total stddev <=8",
            ));
        }
        let mut lo = 0.0;
        let mut hi = MAX_STDDEV;
        for _ in 0..96 {
            let mid = 0.5 * (lo + hi);
            if mid == lo || mid == hi {
                break;
            }
            if self.otm(mid) < value {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo <= IV_TOLERANCE * self.sqrt_time {
                break;
            }
        }
        let sd = 0.5 * (lo + hi);
        let iv = sd / self.sqrt_time;
        let d1 = self.log_ratio / sd + 0.5 * sd;
        let normalized_vega = self.ratio * standard_normal_pdf(d1);
        let vega = self.scale * normalized_vega * self.sqrt_time;
        if !iv.is_finite()
            || iv <= 0.0
            || normalized_vega < MIN_NORMALIZED_VEGA
            || !vega.is_finite()
            || vega <= 0.0
            || (self.otm(sd) - value).abs() / (normalized_vega * self.sqrt_time)
                > 1.0e-9 * iv.max(1.0)
        {
            return Err(FourierError::NumericalFailure(
                "Black IV inversion or Vega is ill-conditioned",
            ));
        }
        Ok((iv, vega))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HestonIvCalibrationEvaluation {
    pub model_prices: Vec<f64>,
    pub model_implied_volatilities: Vec<f64>,
    /// Model Black Vega: price per one absolute annualized volatility unit.
    pub model_vegas: Vec<f64>,
    pub scaled_residuals: Vec<f64>,
    /// Row-major d(model IV)/d(parameter)/iv_scale. Variable scale is not included.
    pub jacobian: Vec<f64>,
    /// Unchanged PRICE diagnostics, deliberately not mislabeled as IV errors.
    pub price_quadrature_differences: Vec<f64>,
    pub price_tail_indicators: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct HestonIvCalibrationResult {
    pub model: RoughVolatilityModel,
    pub evaluation: HestonIvCalibrationEvaluation,
    pub optimizer: LeastSquaresResult,
    pub fit_achieved: bool,
    pub evaluations: usize,
    pub active_bounds: Vec<bool>,
}
#[derive(Clone, Debug)]
pub struct HestonIvCalibrationProblem {
    price_problem: HestonCalibrationProblem,
    quotes: Vec<HestonIvCalibrationQuote>,
}
impl HestonIvCalibrationProblem {
    pub fn new(
        initial_model: RoughVolatilityModel,
        quotes: Vec<HestonIvCalibrationQuote>,
        variables: Vec<HestonCalibrationVariable>,
        fourier: HestonFourierConfig,
    ) -> Result<Self, HestonCalibrationError> {
        // Guard allocations and Black inversions before materializing price quotes.
        if quotes.is_empty() || quotes.len() > 4096 {
            return Err(HestonCalibrationError::InvalidInput(
                "need 1..=4096 IV quotes",
            ));
        }
        let prices = quotes
            .iter()
            .map(|q| q.to_price_quote())
            .collect::<Result<Vec<_>, _>>()?;
        let price_problem =
            HestonCalibrationProblem::new(initial_model, prices, variables, fourier)?;
        Ok(Self {
            price_problem,
            quotes,
        })
    }
    #[must_use]
    pub fn initial_parameters(&self) -> Vec<f64> {
        self.price_problem.initial_parameters()
    }
    #[must_use]
    pub fn variables(&self) -> &[HestonCalibrationVariable] {
        self.price_problem.variables()
    }
    #[must_use]
    pub fn quotes(&self) -> &[HestonIvCalibrationQuote] {
        &self.quotes
    }
    #[must_use]
    pub fn maturity_count(&self) -> usize {
        self.price_problem.maturity_count()
    }
    #[must_use]
    pub fn fourier_config(&self) -> HestonFourierConfig {
        self.price_problem.fourier_config()
    }

    pub fn evaluate(&self, x: &[f64]) -> Result<HestonIvCalibrationEvaluation, FourierError> {
        let p = self.price_problem.evaluate(x)?;
        let n = self.variables().len();
        let mut out = HestonIvCalibrationEvaluation {
            model_prices: p.model_prices,
            model_implied_volatilities: Vec::with_capacity(self.quotes.len()),
            model_vegas: Vec::with_capacity(self.quotes.len()),
            scaled_residuals: Vec::with_capacity(self.quotes.len()),
            jacobian: p.jacobian,
            price_quadrature_differences: p.quadrature_differences,
            price_tail_indicators: p.tail_indicators,
        };
        for (i, q) in self.quotes.iter().enumerate() {
            let b = BlackCoordinates::new(q.forward, q.strike, q.discount, q.maturity, q.is_call)?;
            let (iv, vega) = b.invert(out.model_prices[i])?;
            out.model_implied_volatilities.push(iv);
            out.model_vegas.push(vega);
            out.scaled_residuals
                .push((iv - q.target_volatility) / q.iv_scale);
            for entry in &mut out.jacobian[i * n..(i + 1) * n] {
                *entry = (*entry / vega) / q.iv_scale;
            }
        }
        if out
            .scaled_residuals
            .iter()
            .chain(&out.jacobian)
            .any(|x| !x.is_finite())
        {
            return Err(FourierError::NumericalFailure(
                "IV residual/Jacobian overflow",
            ));
        }
        Ok(out)
    }
    /// Local bounded IV fit, with invalid-IV trials rejected and counted. No
    /// quote is dropped; an invalid initial evaluation is an error.
    pub fn calibrate(
        &self,
        options: LeastSquaresOptions,
    ) -> Result<HestonIvCalibrationResult, HestonCalibrationError> {
        options
            .validate()
            .map_err(HestonCalibrationError::InvalidInput)?;
        if options.max_evaluations < 2 {
            return Err(HestonCalibrationError::InvalidInput(
                "reserve at least two evaluations",
            ));
        }
        self.price_problem.check_work(options.max_evaluations)?;
        let vars: Vec<_> = self
            .variables()
            .iter()
            .map(|v| LeastSquaresVariable {
                lower: v.lower,
                upper: v.upper,
                scale: v.scale,
            })
            .collect();
        let optimizer = bounded_least_squares(
            &self.initial_parameters(),
            &vars,
            LeastSquaresOptions {
                max_evaluations: options.max_evaluations - 1,
                ..options
            },
            |x| {
                let e = self.evaluate(x)?;
                Ok(LeastSquaresEvaluation {
                    residuals: e.scaled_residuals,
                    jacobian: e.jacobian,
                })
            },
        )
        .map_err(HestonCalibrationError::Optimizer)?;
        let model = self.price_problem.model_at(&optimizer.parameters)?;
        let evaluation = self.evaluate(&optimizer.parameters)?;
        if evaluation.scaled_residuals != optimizer.evaluation.residuals
            || evaluation.jacobian != optimizer.evaluation.jacobian
        {
            return Err(HestonCalibrationError::InvalidInput(
                "non-reproducible final IV evaluation",
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
            .zip(self.variables())
            .map(|(&x, v)| x == v.lower || x == v.upper)
            .collect();
        Ok(HestonIvCalibrationResult {
            model,
            evaluation,
            optimizer,
            fit_achieved,
            evaluations,
            active_bounds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn black_roundtrip_parity_and_vega() {
        for t in [0.01, 0.25, 1.0, 10.0] {
            for f in [80.0, 100.0, 120.0] {
                for k in [80.0, 100.0, 120.0] {
                    for sigma in [0.2, 0.8] {
                        // Far-OTM low-vega points are intentionally outside the IV domain.
                        if (f64::ln(f / k) / (sigma * f64::sqrt(t))).abs() > 5.0 {
                            continue;
                        }
                        for call in [false, true] {
                            let b = BlackCoordinates::new(f, k, 0.97, t, call).unwrap();
                            let price = b.scale * (b.otm(sigma * t.sqrt()) + b.intrinsic);
                            let (iv, vega) = b.invert(price).unwrap();
                            assert!((iv - sigma).abs() < 2e-8, "{t} {f} {k} {sigma}: {iv}");
                            let d1 = (f / k).ln() / (sigma * t.sqrt()) + 0.5 * sigma * t.sqrt();
                            let expected = 0.97 * f * standard_normal_pdf(d1) * t.sqrt();
                            assert!((vega - expected).abs() < 2e-8 * expected.max(1.0));
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn black_strict_boundaries_and_conditioning() {
        let b = BlackCoordinates::new(100., 100., 1., 1., true).unwrap();
        for price in [-1., 0., 100., 101., f64::NAN, f64::INFINITY] {
            assert!(b.invert(price).is_err());
        }
        assert!(b.invert(99.999999).is_err());
        let wing = BlackCoordinates::new(100., 1000., 1., 0.01, true).unwrap();
        assert!(wing.invert(wing.scale * wing.otm(0.02)).is_err());
        assert!(BlackCoordinates::new(1e308, 1e308, 1e308, 1., true).is_err());
    }
}

#[path = "iv_refinement.rs"]
mod refinement;
pub use refinement::{
    HestonIvGridValidation, HestonIvRefinementOptions, HestonIvRefinementResult,
    HestonIvRefinementStage,
};
