//! Additional rough-volatility model definitions; no execution state or calibration.
//!
//! Parameters are under the pricing measure when used by a pricing plan. In
//! particular, historical RFSV estimates are not automatically risk-neutral.

use crate::models::{HullWhiteError, RoughBergomi};
use pricing_numerics::gamma_half_to_two;

pub(crate) fn invalid(field: &'static str) -> HullWhiteError {
    HullWhiteError::InvalidInput { field, index: 0 }
}
fn finite(x: f64, field: &'static str) -> Result<(), HullWhiteError> {
    if x.is_finite() {
        Ok(())
    } else {
        Err(invalid(field))
    }
}
fn nonnegative(x: f64, field: &'static str) -> Result<(), HullWhiteError> {
    finite(x, field)?;
    if x >= 0.0 {
        Ok(())
    } else {
        Err(invalid(field))
    }
}
fn hurst_rho(hurst: f64, correlation: f64) -> Result<(), HullWhiteError> {
    RoughBergomi::new(hurst, 0.0, correlation).map(|_| ())
}

macro_rules! scalar_getters {
    ($($name:ident),+ $(,)?) => { $(
        #[must_use]
        pub const fn $name(&self) -> f64 { self.$name }
    )+ };
}

/// An explicit initial forward-variance curve. Linear interpolation and flat
/// extrapolation apply to knot curves. The exponential form permits the exact
/// classical-SABR boundary without approximating an exponential with knots.
#[derive(Clone, Debug, PartialEq)]
pub struct ForwardVarianceCurve {
    pub(crate) kind: ForwardVarianceKind,
}
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ForwardVarianceKind {
    Knots { times: Vec<f64>, values: Vec<f64> },
    Exponential { initial: f64, growth: f64 },
}
impl ForwardVarianceCurve {
    pub fn constant(variance: f64) -> Result<Self, HullWhiteError> {
        Self::piecewise_linear(vec![0.0], vec![variance])
    }
    pub fn piecewise_linear(times: Vec<f64>, values: Vec<f64>) -> Result<Self, HullWhiteError> {
        if times.is_empty() || times.len() != values.len() || times[0] != 0.0 {
            return Err(invalid("rough_forward_variance_shape"));
        }
        for (i, (&t, &v)) in times.iter().zip(&values).enumerate() {
            nonnegative(t, "rough_forward_variance_time")?;
            nonnegative(v, "rough_forward_variance_value")?;
            if i > 0 && t <= times[i - 1] {
                return Err(invalid("rough_forward_variance_order"));
            }
        }
        Ok(Self {
            kind: ForwardVarianceKind::Knots { times, values },
        })
    }
    pub fn exponential(initial: f64, growth: f64) -> Result<Self, HullWhiteError> {
        nonnegative(initial, "rough_forward_variance_initial")?;
        finite(growth, "rough_forward_variance_growth")?;
        Ok(Self {
            kind: ForwardVarianceKind::Exponential { initial, growth },
        })
    }
    pub fn value(&self, time: f64) -> Result<f64, HullWhiteError> {
        nonnegative(time, "rough_forward_variance_query_time")?;
        let value = match &self.kind {
            ForwardVarianceKind::Exponential { initial, growth } => {
                if *initial == 0.0 {
                    0.0
                } else {
                    let log_value = initial.ln() + growth * time;
                    finite(log_value, "rough_forward_variance_overflow")?;
                    let value = log_value.exp();
                    if value == 0.0 {
                        return Err(invalid("rough_forward_variance_underflow"));
                    }
                    value
                }
            }
            ForwardVarianceKind::Knots { times, values } => {
                let upper = times.partition_point(|&t| t <= time);
                if upper == times.len() {
                    values[upper - 1]
                } else {
                    let lower = upper - 1;
                    let w = (time - times[lower]) / (times[upper] - times[lower]);
                    (1.0 - w) * values[lower] + w * values[upper]
                }
            }
        };
        nonnegative(value, "rough_forward_variance_overflow")?;
        Ok(value)
    }
    pub(crate) fn fingerprint_into(&self, hash: &mut blake3::Hasher) {
        match &self.kind {
            ForwardVarianceKind::Knots { times, values } => {
                hash.update(&[0]);
                hash.update(&(times.len() as u64).to_le_bytes());
                for &x in times.iter().chain(values) {
                    hash.update(&x.to_le_bytes());
                }
            }
            ForwardVarianceKind::Exponential { initial, growth } => {
                hash.update(&[1]);
                hash.update(&initial.to_le_bytes());
                hash.update(&growth.to_le_bytes());
            }
        }
    }
}

/// V = v0 + K * [kappa(theta-V) dt + nu sqrt(V) dW],
/// K(t)=t^(H-1/2)/Gamma(H+1/2). H=1/2 includes ordinary Heston.
#[derive(Clone, Debug, PartialEq)]
pub struct RoughHeston {
    pub(crate) hurst: f64,
    pub(crate) initial_variance: f64,
    pub(crate) mean_reversion: f64,
    pub(crate) long_run_variance: f64,
    pub(crate) vol_of_vol: f64,
    pub(crate) correlation: f64,
}
impl RoughHeston {
    pub fn new(
        hurst: f64,
        initial_variance: f64,
        mean_reversion: f64,
        long_run_variance: f64,
        vol_of_vol: f64,
        correlation: f64,
    ) -> Result<Self, HullWhiteError> {
        hurst_rho(hurst, correlation)?;
        for (x, field) in [
            (initial_variance, "rough_heston_initial_variance"),
            (mean_reversion, "rough_heston_mean_reversion"),
            (long_run_variance, "rough_heston_long_run_variance"),
            (vol_of_vol, "rough_heston_vol_of_vol"),
        ] {
            nonnegative(x, field)?;
        }
        Ok(Self {
            hurst,
            initial_variance,
            mean_reversion,
            long_run_variance,
            vol_of_vol,
            correlation,
        })
    }
    scalar_getters!(
        hurst,
        initial_variance,
        mean_reversion,
        long_run_variance,
        vol_of_vol,
        correlation
    );
}

/// Finite exponential-kernel Heston. All factors share ONE variance Brownian
/// motion; a finite lift is not a mathematically rough process.
#[derive(Clone, Debug, PartialEq)]
pub struct LiftedHeston {
    pub(crate) heston: RoughHeston,
    pub(crate) weights: Vec<f64>,
    pub(crate) rates: Vec<f64>,
}
impl LiftedHeston {
    pub fn new(
        initial_variance: f64,
        mean_reversion: f64,
        long_run_variance: f64,
        vol_of_vol: f64,
        correlation: f64,
        weights: Vec<f64>,
        rates: Vec<f64>,
    ) -> Result<Self, HullWhiteError> {
        let heston = RoughHeston::new(
            0.5,
            initial_variance,
            mean_reversion,
            long_run_variance,
            vol_of_vol,
            correlation,
        )?;
        if weights.is_empty() || weights.len() != rates.len() || weights.len() > 256 {
            return Err(invalid("lifted_heston_factor_count"));
        }
        for (&w, &x) in weights.iter().zip(&rates) {
            nonnegative(w, "lifted_heston_weight")?;
            nonnegative(x, "lifted_heston_rate")?;
            if w == 0.0 {
                return Err(invalid("lifted_heston_weight"));
            }
        }
        Ok(Self {
            heston,
            weights,
            rates,
        })
    }
    #[must_use]
    pub fn heston_parameters(&self) -> &RoughHeston {
        &self.heston
    }
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
    #[must_use]
    pub fn rates(&self) -> &[f64] {
        &self.rates
    }
}

/// Pure-feedback quadratic rough Heston:
/// Z=z0-lambda K*Z dt+lambda eta K*sqrt(V) dW_S; V=a(Z-b)^2+c.
/// The same Brownian motion drives the asset and the feedback state.
#[derive(Clone, Debug, PartialEq)]
pub struct QuadraticRoughHeston {
    pub(crate) hurst: f64,
    pub(crate) initial_state: f64,
    pub(crate) mean_reversion: f64,
    pub(crate) vol_of_vol: f64,
    pub(crate) quadratic: f64,
    pub(crate) shift: f64,
    pub(crate) variance_floor: f64,
}
impl QuadraticRoughHeston {
    pub fn new(
        hurst: f64,
        initial_state: f64,
        mean_reversion: f64,
        vol_of_vol: f64,
        quadratic: f64,
        shift: f64,
        variance_floor: f64,
    ) -> Result<Self, HullWhiteError> {
        hurst_rho(hurst, 1.0)?;
        finite(initial_state, "quadratic_rough_initial_state")?;
        finite(shift, "quadratic_rough_shift")?;
        for (x, field) in [
            (mean_reversion, "quadratic_rough_mean_reversion"),
            (vol_of_vol, "quadratic_rough_vol_of_vol"),
            (quadratic, "quadratic_rough_quadratic"),
            (variance_floor, "quadratic_rough_floor"),
        ] {
            nonnegative(x, field)?;
        }
        Ok(Self {
            hurst,
            initial_state,
            mean_reversion,
            vol_of_vol,
            quadratic,
            shift,
            variance_floor,
        })
    }
    scalar_getters!(
        hurst,
        initial_state,
        mean_reversion,
        vol_of_vol,
        quadratic,
        shift,
        variance_floor
    );
}

/// Shared-driver mixture of lognormal VARIANCES, not a mixture of option prices.
/// eta denotes log-variance volatility. Weights must sum to one.
#[derive(Clone, Debug, PartialEq)]
pub struct MixedRoughBergomi {
    pub(crate) hurst: f64,
    pub(crate) correlation: f64,
    pub(crate) weights: Vec<f64>,
    pub(crate) vol_of_vols: Vec<f64>,
    pub(crate) forward_variance: ForwardVarianceCurve,
}
impl MixedRoughBergomi {
    pub fn new(
        hurst: f64,
        correlation: f64,
        weights: Vec<f64>,
        vol_of_vols: Vec<f64>,
        forward_variance: ForwardVarianceCurve,
    ) -> Result<Self, HullWhiteError> {
        hurst_rho(hurst, correlation)?;
        if weights.is_empty() || weights.len() != vol_of_vols.len() || weights.len() > 256 {
            return Err(invalid("mixed_rough_bergomi_shape"));
        }
        for (&w, &eta) in weights.iter().zip(&vol_of_vols) {
            nonnegative(w, "mixed_rough_bergomi_weight")?;
            nonnegative(eta, "mixed_rough_bergomi_vol_of_vol")?;
        }
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || (total - 1.0).abs() > 1e-12 {
            return Err(invalid("mixed_rough_bergomi_weight_sum"));
        }
        // Only remove roundoff in an already validated convex combination.
        let weights = weights.into_iter().map(|w| w / total).collect();
        Ok(Self {
            hurst,
            correlation,
            weights,
            vol_of_vols,
            forward_variance,
        })
    }
    scalar_getters!(hurst, correlation);
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
    #[must_use]
    pub fn vol_of_vols(&self) -> &[f64] {
        &self.vol_of_vols
    }
    #[must_use]
    pub fn forward_variance(&self) -> &ForwardVarianceCurve {
        &self.forward_variance
    }
}

/// Forward-variance rough SABR: dF=sqrt(xi_t(t))*F^beta*dB.
/// beta=0 is normal (signed F), beta=1 is lognormal. At H=1/2, classical
/// d alpha=nu alpha dW requires eta=2nu and xi0(t)=alpha0^2 exp(nu^2 t).
#[derive(Clone, Debug, PartialEq)]
pub struct RoughSabr {
    pub(crate) hurst: f64,
    pub(crate) vol_of_vol: f64,
    pub(crate) correlation: f64,
    pub(crate) beta: f64,
    pub(crate) forward_variance: ForwardVarianceCurve,
}
impl RoughSabr {
    pub fn new(
        hurst: f64,
        vol_of_vol: f64,
        correlation: f64,
        beta: f64,
        forward_variance: ForwardVarianceCurve,
    ) -> Result<Self, HullWhiteError> {
        hurst_rho(hurst, correlation)?;
        nonnegative(vol_of_vol, "rough_sabr_vol_of_vol")?;
        if !beta.is_finite() || !(0.0..=1.0).contains(&beta) {
            return Err(invalid("rough_sabr_beta"));
        }
        Ok(Self {
            hurst,
            vol_of_vol,
            correlation,
            beta,
            forward_variance,
        })
    }
    scalar_getters!(hurst, vol_of_vol, correlation, beta);
    #[must_use]
    pub fn forward_variance(&self) -> &ForwardVarianceCurve {
        &self.forward_variance
    }
}

/// Stationary fractional-OU log volatility, driven by genuine two-sided fBm.
/// None samples the stationary initial state. Some(x) conditions on X(0)=x
/// ONLY, not the entire observed past. The asset Brownian is independent.
#[derive(Clone, Debug, PartialEq)]
pub struct Rfsv {
    pub(crate) hurst: f64,
    pub(crate) mean_reversion: f64,
    pub(crate) vol_of_log_vol: f64,
    pub(crate) mean_log_vol: f64,
    pub(crate) initial_log_vol: Option<f64>,
}
impl Rfsv {
    pub fn new(
        hurst: f64,
        mean_reversion: f64,
        vol_of_log_vol: f64,
        mean_log_vol: f64,
        initial_log_vol: Option<f64>,
    ) -> Result<Self, HullWhiteError> {
        hurst_rho(hurst, 0.0)?;
        nonnegative(mean_reversion, "rfsv_mean_reversion")?;
        if mean_reversion == 0.0 {
            return Err(invalid("rfsv_mean_reversion"));
        }
        nonnegative(vol_of_log_vol, "rfsv_vol_of_log_vol")?;
        finite(mean_log_vol, "rfsv_mean_log_vol")?;
        if let Some(x) = initial_log_vol {
            finite(x, "rfsv_initial_log_vol")?;
            if vol_of_log_vol == 0.0 && x != mean_log_vol {
                return Err(invalid("rfsv_degenerate_conditioning"));
            }
        }
        Ok(Self {
            hurst,
            mean_reversion,
            vol_of_log_vol,
            mean_log_vol,
            initial_log_vol,
        })
    }
    scalar_getters!(hurst, mean_reversion, vol_of_log_vol, mean_log_vol);
    #[must_use]
    pub const fn initial_log_vol(&self) -> Option<f64> {
        self.initial_log_vol
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RoughVolatilityModel {
    RoughHeston(RoughHeston),
    LiftedHeston(LiftedHeston),
    QuadraticRoughHeston(QuadraticRoughHeston),
    MixedRoughBergomi(MixedRoughBergomi),
    RoughSabr(RoughSabr),
    Rfsv(Rfsv),
}
macro_rules! model_from {
    ($($name:ident),+ $(,)?) => { $(
        impl From<$name> for RoughVolatilityModel {
            fn from(model: $name) -> Self { Self::$name(model) }
        }
    )+ };
}
model_from!(
    RoughHeston,
    LiftedHeston,
    QuadraticRoughHeston,
    MixedRoughBergomi,
    RoughSabr,
    Rfsv
);

impl RoughVolatilityModel {
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::RoughHeston(_) => "rough_heston",
            Self::LiftedHeston(_) => "lifted_heston",
            Self::QuadraticRoughHeston(_) => "quadratic_rough_heston",
            Self::MixedRoughBergomi(_) => "mixed_rough_bergomi",
            Self::RoughSabr(_) => "rough_sabr",
            Self::Rfsv(_) => "rfsv",
        }
    }
    pub(crate) fn fingerprint_into(&self, hash: &mut blake3::Hasher) {
        hash.update(self.name().as_bytes());
        hash.update(&[0]);
        let scalars: Vec<f64> = match self {
            Self::RoughHeston(m) => vec![
                m.hurst,
                m.initial_variance,
                m.mean_reversion,
                m.long_run_variance,
                m.vol_of_vol,
                m.correlation,
            ],
            Self::LiftedHeston(m) => {
                hash.update(&(m.weights.len() as u64).to_le_bytes());
                for &x in m.weights.iter().chain(&m.rates) {
                    hash.update(&x.to_le_bytes());
                }
                vec![
                    m.heston.initial_variance,
                    m.heston.mean_reversion,
                    m.heston.long_run_variance,
                    m.heston.vol_of_vol,
                    m.heston.correlation,
                ]
            }
            Self::QuadraticRoughHeston(m) => vec![
                m.hurst,
                m.initial_state,
                m.mean_reversion,
                m.vol_of_vol,
                m.quadratic,
                m.shift,
                m.variance_floor,
            ],
            Self::MixedRoughBergomi(m) => {
                hash.update(&(m.weights.len() as u64).to_le_bytes());
                for &x in m.weights.iter().chain(&m.vol_of_vols) {
                    hash.update(&x.to_le_bytes());
                }
                m.forward_variance.fingerprint_into(hash);
                vec![m.hurst, m.correlation]
            }
            Self::RoughSabr(m) => {
                m.forward_variance.fingerprint_into(hash);
                vec![m.hurst, m.vol_of_vol, m.correlation, m.beta]
            }
            Self::Rfsv(m) => {
                hash.update(&[u8::from(m.initial_log_vol.is_some())]);
                if let Some(x) = m.initial_log_vol {
                    hash.update(&x.to_le_bytes());
                }
                vec![m.hurst, m.mean_reversion, m.vol_of_log_vol, m.mean_log_vol]
            }
        };
        for x in scalars {
            hash.update(&x.to_le_bytes());
        }
    }
}

impl LiftedHeston {
    /// Abi Jaber's geometric-bin quadrature of the fractional kernel's Laplace
    /// measure. Factor count and ratio determine an approximation, not a
    /// tolerance-certified fit. Rates have inverse-year units when times are years.
    pub fn from_rough(
        model: &RoughHeston,
        factors: usize,
        ratio: f64,
    ) -> Result<Self, HullWhiteError> {
        if factors == 0 || factors > 256 || !ratio.is_finite() || ratio <= 1.0 {
            return Err(invalid("lifted_heston_quadrature"));
        }
        if model.hurst == 0.5 {
            return Self::new(
                model.initial_variance,
                model.mean_reversion,
                model.long_run_variance,
                model.vol_of_vol,
                model.correlation,
                vec![1.0],
                vec![0.0],
            );
        }
        let alpha = model.hurst + 0.5;
        let p = 1.0 - alpha;
        let log_ratio = ratio.ln();
        let denominator = gamma_half_to_two(alpha).ok_or(invalid("lift_gamma"))?
            * gamma_half_to_two(2.0 - alpha).ok_or(invalid("lift_gamma"))?;
        let mass_factor = (p * log_ratio).exp_m1() / denominator;
        let rate_factor =
            p / (1.0 + p) * ((1.0 + p) * log_ratio).exp_m1() / (p * log_ratio).exp_m1();
        let mut weights = Vec::with_capacity(factors);
        let mut rates = Vec::with_capacity(factors);
        for j in 0..factors {
            let log_lower = (j as f64 - factors as f64 / 2.0) * log_ratio;
            weights.push(mass_factor * (p * log_lower).exp());
            rates.push(rate_factor * log_lower.exp());
        }
        Self::new(
            model.initial_variance,
            model.mean_reversion,
            model.long_run_variance,
            model.vol_of_vol,
            model.correlation,
            weights,
            rates,
        )
    }
}
