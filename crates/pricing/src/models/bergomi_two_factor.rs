//! Two-factor Bergomi multiplier and exact joint spot/OU innovations.
use super::Bergomi1Factor;
use crate::core::{CoreError, NonNegativeF64, PositiveF64};
use pricing_numerics::{CorrelationError, CorrelationFactor, CorrelationToleranceConfig};
use std::{error::Error, fmt};

/// Fixed roundoff tolerances for the model's three-driver marginal. Global
/// multi-asset matrices additionally obey the supplied schedule's tolerances.
pub const BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES: CorrelationToleranceConfig =
    CorrelationToleranceConfig {
        symmetry_abs_tol: 0.0,
        diagonal_abs_tol: 0.0,
        psd_abs_tol: 64.0 * f64::EPSILON,
        psd_rel_tol: 0.0,
        zero_pivot_abs_tol: 64.0 * f64::EPSILON,
        zero_pivot_rel_tol: 0.0,
    };

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum BergomiError {
    Core(CoreError),
    Correlation(CorrelationError),
    InvalidKernel,
}
impl fmt::Display for BergomiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(e) => e.fmt(f),
            Self::Correlation(e) => e.fmt(f),
            Self::InvalidKernel => write!(f, "unrepresentable Bergomi OU kernel covariance"),
        }
    }
}
impl Error for BergomiError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Core(e) => Some(e),
            Self::Correlation(e) => Some(e),
            Self::InvalidKernel => None,
        }
    }
}
impl From<CoreError> for BergomiError {
    fn from(e: CoreError) -> Self {
        Self::Core(e)
    }
}
impl From<CorrelationError> for BergomiError {
    fn from(e: CorrelationError) -> Self {
        Self::Correlation(e)
    }
}

/// `a=exp(nu * alpha * ((1-theta)*X1 + theta*X2))`, where
/// `dXj=-kj*Xj*dt+dVj`. Alpha normalizes the instantaneous variance of the
/// weighted OU driver to one. Deterministic exponential normalization cancels
/// in particle-calibrated LSV. Factor order is explicit and never sorted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bergomi2Factor {
    factors: [Bergomi1Factor; 2],
    vol_of_vol: NonNegativeF64,
    mixing_weight: f64,
    factor_correlation: f64,
    weights: [f64; 2],
}
impl Bergomi2Factor {
    pub fn new(
        mean_reversions: [f64; 2],
        vol_of_vol: f64,
        mixing_weight: f64,
        spot_correlations: [f64; 2],
        factor_correlation: f64,
    ) -> Result<Self, BergomiError> {
        if !mixing_weight.is_finite() || !(0.0..=1.0).contains(&mixing_weight) {
            return Err(CoreError::InvalidWeights {
                field: "bergomi_mixing_weight",
            }
            .into());
        }
        let factors = [
            Bergomi1Factor::new(mean_reversions[0], vol_of_vol, spot_correlations[0])?,
            Bergomi1Factor::new(mean_reversions[1], vol_of_vol, spot_correlations[1])?,
        ];
        CorrelationFactor::compile(
            vec![
                vec![1.0, spot_correlations[0], spot_correlations[1]],
                vec![spot_correlations[0], 1.0, factor_correlation],
                vec![spot_correlations[1], factor_correlation, 1.0],
            ],
            BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES,
        )?;
        let theta = mixing_weight;
        // Nonnegative terms avoid catastrophic cancellation near theta=.5,rho=-1.
        let norm2 =
            (1.0 - 2.0 * theta).powi(2) + 2.0 * theta * (1.0 - theta) * (1.0 + factor_correlation);
        let norm = PositiveF64::new(norm2, "bergomi_weighted_driver_variance")?
            .get()
            .sqrt();
        Ok(Self {
            factors,
            vol_of_vol: NonNegativeF64::new(vol_of_vol, "bergomi_vol_of_vol")?,
            mixing_weight,
            factor_correlation,
            weights: [(1.0 - theta) / norm, theta / norm],
        })
    }
    pub fn mean_reversions(self) -> [f64; 2] {
        self.factors.map(Bergomi1Factor::mean_reversion)
    }
    pub fn spot_correlations(self) -> [f64; 2] {
        self.factors.map(Bergomi1Factor::correlation)
    }
    pub const fn vol_of_vol(self) -> f64 {
        self.vol_of_vol.get()
    }
    pub const fn mixing_weight(self) -> f64 {
        self.mixing_weight
    }
    pub const fn factor_correlation(self) -> f64 {
        self.factor_correlation
    }
    pub const fn normalized_weights(self) -> [f64; 2] {
        self.weights
    }
    pub(crate) const fn components(self) -> [Bergomi1Factor; 2] {
        self.factors
    }

    /// Exact covariance of `(Delta W, OU1, OU2)`. All three independent
    /// Gaussian columns are retained, including singular and zero-weight cases.
    pub fn transition(self, dt: f64) -> Result<Bergomi2FactorTransition, BergomiError> {
        let dt = PositiveF64::new(dt, "bergomi_delta_t")?.get();
        let t = [
            self.factors[0].transition(dt)?,
            self.factors[1].transition(dt)?,
        ];
        let k = self.mean_reversions().map(|k| k * dt);
        let r = self.spot_correlations();
        let c1 = r[0] * ou_kernel_correlation(0.0, k[0])?;
        let c2 = r[1] * ou_kernel_correlation(0.0, k[1])?;
        let c12 = self.factor_correlation * ou_kernel_correlation(k[0], k[1])?;
        let c = CorrelationFactor::compile(
            vec![vec![1.0, c1, c2], vec![c1, 1.0, c12], vec![c2, c12, 1.0]],
            BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES,
        )?;
        let scale = [dt.sqrt(), t[0].variance.sqrt(), t[1].variance.sqrt()];
        let mut covariance = [[0.0; 3]; 3];
        let mut lower = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                covariance[i][j] = c.canonical()[3 * i + j] * scale[i] * scale[j];
                lower[i][j] = c.lower()[3 * i + j] * scale[i];
            }
        }
        Ok(Bergomi2FactorTransition {
            decay: [t[0].decay, t[1].decay],
            covariance,
            lower,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bergomi2FactorTransition {
    pub decay: [f64; 2],
    pub covariance: [[f64; 3]; 3],
    pub lower: [[f64; 3]; 3],
}
impl Bergomi2FactorTransition {
    pub fn evolve(self, x: [f64; 2], z: [f64; 3]) -> [f64; 2] {
        std::array::from_fn(|i| {
            self.decay[i] * x[i]
                + self.lower[i + 1][0] * z[0]
                + self.lower[i + 1][1] * z[1]
                + self.lower[i + 1][2] * z[2]
        })
    }
}

/// Normalized integral of two exponential kernels on the unit interval.
pub(crate) fn ou_kernel_correlation(a: f64, b: f64) -> Result<f64, BergomiError> {
    if !a.is_finite()
        || !b.is_finite()
        || !(a + b).is_finite()
        || !(2.0 * a).is_finite()
        || !(2.0 * b).is_finite()
    {
        return Err(BergomiError::InvalidKernel);
    }
    if a == b {
        return Ok(1.0);
    }
    let phi = |x: f64| if x == 0.0 { 1.0 } else { -(-x).exp_m1() / x };
    let value = phi(a + b) / (phi(2.0 * a).sqrt() * phi(2.0 * b).sqrt());
    if !value.is_finite() || value <= 0.0 || value > 1.0 + 32.0 * f64::EPSILON {
        return Err(BergomiError::InvalidKernel);
    }
    Ok(value.min(1.0))
}
