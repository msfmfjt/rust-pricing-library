//! Deterministic European pricing for continuous-time rough/lifted Heston.
//! See docs/models/heston-fourier.md for the transform and numerical contract.
mod riccati;

use crate::Fingerprint;
use crate::models::{RoughHeston, RoughVolatilityModel};
use crate::product::OptionSide;
use pricing_numerics::{Complex64, NeumaierSum, standard_normal_cdf};
use riccati::Kernel;
use std::{error::Error, fmt};

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum FourierError {
    InvalidInput(&'static str),
    UnsupportedModel,
    WorkLimit,
    Numerical(&'static str),
    RiccatiFailure { step: usize, frequency: f64 },
    PriceOutsideBounds { price: f64, lower: f64, upper: f64 },
}
impl fmt::Display for FourierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(name) => write!(f, "invalid Heston Fourier input: {name}"),
            Self::UnsupportedModel => write!(
                f,
                "Fourier pricing supports RoughHeston and LiftedHeston only"
            ),
            Self::WorkLimit => write!(
                f,
                "Heston Fourier work limit exceeded; reduce the numerical grid"
            ),
            Self::Numerical(name) => write!(f, "Heston Fourier numerical failure: {name}"),
            Self::RiccatiFailure { step, frequency } => write!(
                f,
                "Riccati failure at time step {step}, frequency {frequency}"
            ),
            Self::PriceOutsideBounds {
                price,
                lower,
                upper,
            } => write!(
                f,
                "Fourier price {price} outside [{lower}, {upper}]; inspect grid and cutoff"
            ),
        }
    }
}
impl Error for FourierError {}

type Result<T> = std::result::Result<T, FourierError>;

/// Explicit numerical grids, not an accuracy guarantee. Uniform Riccati time
/// nodes and Simpson Fourier nodes; no Monte Carlo sampling error is attached.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FourierConfig {
    pub time_steps: usize,
    pub integration_intervals: usize,
    pub cutoff: f64,
}
impl Default for FourierConfig {
    fn default() -> Self {
        Self {
            time_steps: 512,
            integration_intervals: 512,
            cutoff: 128.0,
        }
    }
}
impl FourierConfig {
    fn validate(self) -> Result<()> {
        if !(2..=8192).contains(&self.time_steps) {
            return Err(FourierError::InvalidInput("time_steps must be in 2..=8192"));
        }
        if !(4..=8192).contains(&self.integration_intervals)
            || !self.integration_intervals.is_multiple_of(2)
        {
            return Err(FourierError::InvalidInput(
                "integration_intervals must be even and in 4..=8192",
            ));
        }
        if !self.cutoff.is_finite() || self.cutoff <= 0.0 || self.cutoff > 10000.0 {
            return Err(FourierError::InvalidInput(
                "cutoff must be finite and in (0,10000]",
            ));
        }
        Ok(())
    }
}

/// Four actual prices, isolating sequential time-grid, integration-grid and
/// cutoff changes. These differences are NOT statistical SEs or error bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FourierRefinement {
    pub base_price: f64,
    pub time_refined_price: f64,
    pub quadrature_refined_price: f64,
    pub extended_cutoff_price: f64,
}
impl FourierRefinement {
    #[must_use]
    pub fn time_change(&self) -> f64 {
        self.time_refined_price - self.base_price
    }
    #[must_use]
    pub fn quadrature_change(&self) -> f64 {
        self.quadrature_refined_price - self.time_refined_price
    }
    #[must_use]
    pub fn cutoff_change(&self) -> f64 {
        self.extended_cutoff_price - self.quadrature_refined_price
    }
}

/// Cached damped transform on one maturity's Fourier grid. Reuse for a strike
/// strip. Inputs are a positive martingale forward and a deterministic payment
/// discount; no automatic cash-dividend, LSV, short-rate or payoff conversion.
#[derive(Clone, Debug)]
pub struct HestonFourierPlan {
    model: RoughVolatilityModel,
    maturity: f64,
    config: FourierConfig,
    kernel: Kernel,
    transforms: Vec<Complex64>,
    fingerprint: Fingerprint,
}
impl HestonFourierPlan {
    pub fn compile(
        model: RoughVolatilityModel,
        maturity: f64,
        config: FourierConfig,
    ) -> Result<Self> {
        config.validate()?;
        if !maturity.is_finite() || maturity < 0.0 {
            return Err(FourierError::InvalidInput("maturity"));
        }
        let work = match &model {
            RoughVolatilityModel::RoughHeston(_) => (config.time_steps as u64).pow(2) / 2,
            RoughVolatilityModel::LiftedHeston(m) => {
                config.time_steps as u64 * m.weights().len() as u64
            }
            _ => return Err(FourierError::UnsupportedModel),
        } * (config.integration_intervals as u64 + 1);
        if work > 2_500_000_000 {
            return Err(FourierError::WorkLimit);
        }
        let kernel = Kernel::new(&model, maturity, config.time_steps)?;
        let mut hash = blake3::Hasher::new();
        hash.update(b"heston-fourier-implicit-hats-lewis-half-moment-control-v1");
        model.fingerprint_into(&mut hash);
        for x in [maturity, config.cutoff] {
            hash.update(&x.to_le_bytes());
        }
        for n in [config.time_steps, config.integration_intervals] {
            hash.update(&(n as u64).to_le_bytes());
        }
        let fingerprint = Fingerprint::from_bytes(*hash.finalize().as_bytes());
        let mut plan = Self {
            model,
            maturity,
            config,
            kernel,
            transforms: Vec::new(),
            fingerprint,
        };
        plan.transforms = (0..=config.integration_intervals)
            .map(|i| {
                plan.transform(
                    0.5,
                    i as f64 * config.cutoff / config.integration_intervals as f64,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(plan)
    }
    fn heston(&self) -> &RoughHeston {
        match &self.model {
            RoughVolatilityModel::RoughHeston(m) => m,
            RoughVolatilityModel::LiftedHeston(m) => m.heston_parameters(),
            _ => unreachable!("validated model variant"),
        }
    }
    #[must_use]
    pub const fn config(&self) -> FourierConfig {
        self.config
    }
    #[must_use]
    pub const fn maturity(&self) -> f64 {
        self.maturity
    }
    #[must_use]
    pub fn model(&self) -> &RoughVolatilityModel {
        &self.model
    }
    #[must_use]
    pub const fn plan_fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }

    /// M(d+iu)=E[exp((d+iu)*log(F_T/F_0))], with d in [0,1].
    /// d=0 is the characteristic function; pricing uses d=1/2. Only the
    /// martingale strip is exposed, not higher moments subject to explosion.
    pub fn transform(&self, damping: f64, frequency: f64) -> Result<Complex64> {
        if !damping.is_finite() || !(0.0..=1.0).contains(&damping) || !frequency.is_finite() {
            return Err(FourierError::InvalidInput("transform damping/frequency"));
        }
        let z = Complex64::new(damping, frequency);
        let m = self.heston();
        // Exact deterministic constant-variance limits, including the absorbing
        // zero-variance process even when the formal vol-of-vol is nonzero.
        let zero = m.initial_variance() == 0.0
            && (m.mean_reversion() == 0.0 || m.long_run_variance() == 0.0);
        if zero || self.maturity == 0.0 {
            return Ok(Complex64::ONE);
        }
        if m.vol_of_vol() == 0.0
            && (m.mean_reversion() == 0.0 || m.long_run_variance() == m.initial_variance())
        {
            let result = ((z * z - z) * (0.5 * m.initial_variance() * self.maturity)).exp();
            return if result.is_finite() {
                Ok(result)
            } else {
                Err(FourierError::Numerical("constant-variance transform"))
            };
        }
        self.kernel.transform(
            m,
            self.maturity / self.config.time_steps as f64,
            self.config.time_steps,
            z,
        )
    }

    /// Deterministic truncated Fourier price. No silent clipping or automatic
    /// claim of convergence; call `refinement` to inspect numerical sensitivity.
    pub fn price(&self, forward: f64, strike: f64, discount: f64, side: OptionSide) -> Result<f64> {
        for (x, name, positive) in [
            (forward, "forward", true),
            (strike, "strike", false),
            (discount, "discount", false),
        ] {
            if !x.is_finite() || x < 0.0 || (positive && x == 0.0) {
                return Err(FourierError::InvalidInput(name));
            }
        }
        if discount == 0.0 {
            return Ok(0.0);
        }
        let sign = if side == OptionSide::Call { 1.0 } else { -1.0 };
        let intrinsic = (sign * (forward - strike)).max(0.0);
        if self.maturity == 0.0 || strike == 0.0 {
            let result = discount * intrinsic;
            return if result.is_finite() {
                Ok(result)
            } else {
                Err(FourierError::Numerical("discounted intrinsic value"))
            };
        }
        // Match M(1/2)=exp(-Q/8), rather than using v0*T. In particular,
        // v0=0 with positive variance immigration still needs a nonzero control.
        let half_moment = self.transforms[0].re;
        if !(0.0 < half_moment && half_moment <= 1.0) {
            return Err(FourierError::Numerical("half-moment control variance"));
        }
        let variance = -8.0 * half_moment.ln();
        if !variance.is_finite() {
            return Err(FourierError::Numerical("Black control variance"));
        }
        let control = black_price(forward, strike, variance, side);
        let log_fk = forward.ln() - strike.ln();
        let step = self.config.cutoff / self.config.integration_intervals as f64;
        let mut integral = NeumaierSum::new();
        for (i, &transform) in self.transforms.iter().enumerate() {
            let u = i as f64 * step;
            let z = Complex64::new(0.5, u);
            let black = ((z * z - z) * (0.5 * variance)).exp();
            let phase = Complex64::new(0.0, u * log_fk).exp();
            let weight = if i == 0 || i == self.config.integration_intervals {
                1.0
            } else if i.is_multiple_of(2) {
                2.0
            } else {
                4.0
            };
            integral.add(weight * (phase * (black - transform)).re / (u * u + 0.25));
        }
        let correction = (forward.sqrt() * strike.sqrt() / std::f64::consts::PI)
            * (step / 3.0)
            * integral.total();
        let price = discount * (control + correction);
        let lower = discount * intrinsic;
        let upper = discount
            * if side == OptionSide::Call {
                forward
            } else {
                strike
            };
        if !price.is_finite() || !upper.is_finite() {
            return Err(FourierError::Numerical("Fourier price"));
        }
        // No lower/upper-bound clamping: an under-resolved inversion must not
        // masquerade as a valid intrinsic price or an accurate zero option.
        if price < lower || price > upper {
            return Err(FourierError::PriceOutsideBounds {
                price,
                lower,
                upper,
            });
        }
        Ok(price)
    }

    /// Compare (N,M,U), (2N,M,U), (2N,2M,U), (2N,4M,2U).
    /// The last comparison holds Fourier mesh fixed while extending the cutoff.
    pub fn refinement(
        &self,
        forward: f64,
        strike: f64,
        discount: f64,
        side: OptionSide,
    ) -> Result<FourierRefinement> {
        let cfg = self.config;
        let time = FourierConfig {
            time_steps: cfg.time_steps * 2,
            ..cfg
        };
        let quadrature = FourierConfig {
            integration_intervals: cfg.integration_intervals * 2,
            ..time
        };
        let cutoff = FourierConfig {
            integration_intervals: cfg.integration_intervals * 4,
            cutoff: cfg.cutoff * 2.0,
            ..time
        };
        // Fail before running expensive partial work when a refinement grid is invalid.
        cutoff.validate()?;
        let base_price = self.price(forward, strike, discount, side)?;
        let eval = |config| {
            Self::compile(self.model.clone(), self.maturity, config)?
                .price(forward, strike, discount, side)
        };
        Ok(FourierRefinement {
            base_price,
            time_refined_price: eval(time)?,
            quadrature_refined_price: eval(quadrature)?,
            extended_cutoff_price: eval(cutoff)?,
        })
    }
}

fn black_price(forward: f64, strike: f64, variance: f64, side: OptionSide) -> f64 {
    if variance == 0.0 {
        return if side == OptionSide::Call {
            (forward - strike).max(0.0)
        } else {
            (strike - forward).max(0.0)
        };
    }
    let sd = variance.sqrt();
    let d1 = (forward.ln() - strike.ln()) / sd + 0.5 * sd;
    let d2 = d1 - sd;
    // Evaluate the OTM leg first to reduce cancellation, then exact parity.
    if forward >= strike {
        let put = strike * standard_normal_cdf(-d2) - forward * standard_normal_cdf(-d1);
        if side == OptionSide::Put {
            put
        } else {
            put + forward - strike
        }
    } else {
        let call = forward * standard_normal_cdf(d1) - strike * standard_normal_cdf(d2);
        if side == OptionSide::Call {
            call
        } else {
            call + strike - forward
        }
    }
}
