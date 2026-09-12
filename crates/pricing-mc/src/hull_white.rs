//! Equity/Hull–White hybrid simulation and discounted-particle LSV calibration.
//! The simulated equity state is S0*S/F0(t), before proportional dividends.
//! Cash dividends require a different calibration contract and are not accepted
//! by the experimental calibrated facade.

use crate::lsv::{LsvError, LsvLeverageSurface, LsvParticleConfig};
use crate::{LocalVolError, LocalVolTimeGrid, Philox4x32, RandomCoordinate, RandomDomain};
use pricing_market::{ImpliedVarianceSurface, LocalVarianceGrid};
use pricing_models::hull_white::hw_valid;
use pricing_models::{
    Bergomi1Factor, HullWhite1Factor, HullWhiteError, HullWhiteHybridTransition, HybridCorrelation,
};
use pricing_numerics::{NeumaierSum, standard_normal_pdf};
use std::{error::Error, fmt};

pub const HULL_WHITE_EQUITY_SCHEME: &str = "equity-hw1f-joint-gaussian-log-euler-v1";
pub const HULL_WHITE_LSV_CALIBRATION: &str = "lsv-hw-discounted-quartic-centered-rate-v1";

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum HullWhiteMcError {
    Model(HullWhiteError),
    Lsv(LsvError),
    Grid(LocalVolError),
}
impl fmt::Display for HullWhiteMcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(e) => e.fmt(f),
            Self::Lsv(e) => e.fmt(f),
            Self::Grid(e) => e.fmt(f),
        }
    }
}
impl Error for HullWhiteMcError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(match self {
            Self::Model(e) => e,
            Self::Lsv(e) => e,
            Self::Grid(e) => e,
        })
    }
}
impl From<HullWhiteError> for HullWhiteMcError {
    fn from(e: HullWhiteError) -> Self {
        Self::Model(e)
    }
}
impl From<LsvError> for HullWhiteMcError {
    fn from(e: LsvError) -> Self {
        Self::Lsv(e)
    }
}
impl From<LocalVolError> for HullWhiteMcError {
    fn from(e: LocalVolError) -> Self {
        Self::Grid(e)
    }
}
impl From<pricing_market::MarketError> for HullWhiteMcError {
    fn from(e: pricing_market::MarketError) -> Self {
        Self::Model(e.into())
    }
}

fn invalid(field: &'static str, index: usize) -> HullWhiteMcError {
    HullWhiteError::InvalidInput { field, index }.into()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridState {
    pub normalized_equity: f64,
    pub volatility_factor: f64,
    pub rate_factor: f64,
    pub integrated_rate_factor: f64,
}
impl HybridState {
    pub fn initial(spot: f64) -> Result<Self, HullWhiteMcError> {
        hw_valid(spot, "initial_spot", 0, true)?;
        if spot == 0.0 {
            return Err(invalid("initial_spot", 0));
        }
        Ok(Self {
            normalized_equity: spot,
            volatility_factor: 0.0,
            rate_factor: 0.0,
            integrated_rate_factor: 0.0,
        })
    }
}

#[derive(Clone, Debug)]
struct Kernel {
    transition: HullWhiteHybridTransition,
    loading: [[f64; 4]; 4],
    dt: f64,
    integrated_shift: f64,
}
impl Kernel {
    fn new(
        rates: &HullWhite1Factor,
        k: f64,
        corr: HybridCorrelation,
        start: f64,
        end: f64,
    ) -> Result<Self, HullWhiteMcError> {
        let transition = rates.transition(start, end, k, corr)?;
        Ok(Self {
            loading: covariance_loading(transition.covariance)?,
            transition,
            dt: end - start,
            integrated_shift: rates.integrated_shift(start, end)?,
        })
    }
    fn advance(
        &self,
        state: HybridState,
        leverage_squared: f64,
        nu: f64,
        shocks: [f64; 4],
        step: usize,
    ) -> Result<HybridState, HullWhiteMcError> {
        for (i, &z) in shocks.iter().enumerate() {
            hw_valid(z, "shock", i, false)?;
        }
        hw_valid(leverage_squared, "squared_leverage", step, true)?;
        let mut noise = [0.0; 4];
        for (i, value) in noise.iter_mut().enumerate() {
            *value = self.loading[i].iter().zip(shocks).map(|(l, z)| l * z).sum();
        }
        let variance = leverage_squared * (2.0 * nu * state.volatility_factor).exp();
        let integral = self.transition.integral_loading * state.rate_factor + noise[3];
        let next = HybridState {
            normalized_equity: state.normalized_equity
                * (integral + self.integrated_shift - 0.5 * variance * self.dt
                    + variance.sqrt() * noise[0])
                    .exp(),
            volatility_factor: self.transition.vol_decay * state.volatility_factor + noise[1],
            rate_factor: self.transition.rate_decay * state.rate_factor + noise[2],
            integrated_rate_factor: state.integrated_rate_factor + integral,
        };
        if !variance.is_finite()
            || !next.normalized_equity.is_finite()
            || next.normalized_equity <= 0.0
            || !next.volatility_factor.is_finite()
            || !next.rate_factor.is_finite()
            || !next.integrated_rate_factor.is_finite()
        {
            return Err(HullWhiteError::NonPositiveState { step: step + 1 }.into());
        }
        Ok(next)
    }
}

/// PSD Cholesky after normalization to correlation scale. Only round-off sized
/// negative residuals are set to zero; inconsistent singular rows are rejected.
fn covariance_loading(cov: [[f64; 4]; 4]) -> Result<[[f64; 4]; 4], HullWhiteMcError> {
    let scales = std::array::from_fn::<_, 4, _>(|i| cov[i][i].sqrt());
    let mut l = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..=i {
            let c = if scales[i] == 0.0 || scales[j] == 0.0 {
                0.0
            } else {
                cov[i][j] / scales[i] / scales[j]
            };
            let r = c - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if !r.is_finite() {
                return Err(HullWhiteError::InvalidCorrelation.into());
            }
            if i == j {
                if r < -2e-12 {
                    return Err(HullWhiteError::InvalidCorrelation.into());
                }
                l[i][j] = r.max(0.0).sqrt();
            } else if l[j][j] > 1e-14 {
                l[i][j] = r / l[j][j];
            } else if r.abs() > 2e-12 {
                return Err(HullWhiteError::InvalidCorrelation.into());
            }
        }
    }
    for (i, row) in l.iter_mut().enumerate() {
        for value in row.iter_mut().take(i + 1) {
            *value *= scales[i];
        }
    }
    Ok(l)
}

#[derive(Clone, Debug)]
pub enum HybridEquityVolatility {
    BlackScholes(f64),
    BergomiLsv {
        factor: Bergomi1Factor,
        leverage: LsvLeverageSurface,
    },
}

#[derive(Clone, Debug)]
pub struct HullWhiteEquityPlan {
    rates: HullWhite1Factor,
    volatility: HybridEquityVolatility,
    correlation: HybridCorrelation,
    times: Box<[f64]>,
    kernels: Box<[Kernel]>,
}
impl HullWhiteEquityPlan {
    pub fn new(
        rates: HullWhite1Factor,
        volatility: HybridEquityVolatility,
        correlation: HybridCorrelation,
        grid: &LocalVolTimeGrid,
    ) -> Result<Self, HullWhiteMcError> {
        let k = match &volatility {
            HybridEquityVolatility::BlackScholes(sigma) => {
                hw_valid(*sigma, "equity_volatility", 0, true)?;
                0.0
            }
            HybridEquityVolatility::BergomiLsv { factor, leverage } => {
                if factor.correlation() != correlation.equity_vol {
                    return Err(invalid("equity_vol_correlation_mismatch", 0));
                }
                let end = grid.nodes()[grid.nodes().len() - 1];
                if end > leverage.times()[leverage.times().len() - 1] {
                    return Err(invalid("leverage_time_coverage", 0));
                }
                for (i, &t) in leverage
                    .times()
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| **t <= end)
                {
                    if !grid.nodes().contains(&t) {
                        return Err(invalid("missing_leverage_knot", i));
                    }
                }
                factor.mean_reversion()
            }
        };
        let kernels = grid
            .nodes()
            .windows(2)
            .map(|w| Kernel::new(&rates, k, correlation, w[0], w[1]))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            rates,
            volatility,
            correlation,
            times: grid.nodes().into(),
            kernels: kernels.into(),
        })
    }
    #[must_use]
    pub fn times(&self) -> &[f64] {
        &self.times
    }
    #[must_use]
    pub fn rates(&self) -> &HullWhite1Factor {
        &self.rates
    }
    #[must_use]
    pub const fn correlation(&self) -> HybridCorrelation {
        self.correlation
    }
    #[must_use]
    pub fn parameter_fingerprint_bytes(&self) -> Vec<u8> {
        let values = match &self.volatility {
            HybridEquityVolatility::BlackScholes(sigma) => vec![0.0, *sigma],
            HybridEquityVolatility::BergomiLsv { factor, .. } => vec![
                1.0,
                factor.mean_reversion(),
                factor.vol_of_vol(),
                factor.correlation(),
            ],
        };
        values
            .into_iter()
            .flat_map(|v| v.to_bits().to_be_bytes())
            .collect()
    }
    pub fn pseudo_shocks(
        &self,
        seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, HullWhiteMcError> {
        hybrid_shocks(seed, path, self.kernels.len(), domain)
    }
    /// Four factor-major blocks of independent normals. All four blocks must be
    /// sign-reversed for an antithetic path; Brownian bridge acts on each block.
    pub fn evolve_path(
        &self,
        spot: f64,
        shocks: &[f64],
    ) -> Result<Vec<HybridState>, HullWhiteMcError> {
        let n = self.kernels.len();
        if Some(shocks.len()) != n.checked_mul(4) {
            return Err(invalid("shock_count", shocks.len()));
        }
        let mut state = HybridState::initial(spot)?;
        if let HybridEquityVolatility::BergomiLsv { leverage, .. } = &self.volatility
            && spot != leverage.initial_f()
        {
            return Err(invalid("leverage_initial_spot", 0));
        }
        let mut states = vec![state];
        for (i, kernel) in self.kernels.iter().enumerate() {
            let (l2, nu) = match &self.volatility {
                HybridEquityVolatility::BlackScholes(sigma) => (sigma * sigma, 0.0),
                HybridEquityVolatility::BergomiLsv { factor, leverage } => (
                    leverage.squared_leverage_at(self.times[i], state.normalized_equity)?,
                    factor.vol_of_vol(),
                ),
            };
            state = kernel.advance(state, l2, nu, std::array::from_fn(|j| shocks[j * n + i]), i)?;
            states.push(state);
        }
        Ok(states)
    }
}

fn hybrid_shocks(
    seed: u64,
    path: u64,
    steps: usize,
    domain: RandomDomain,
) -> Result<Vec<f64>, HullWhiteMcError> {
    let count = steps
        .checked_mul(4)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| invalid("random_dimension", steps))?;
    let rng = Philox4x32::from_seed(seed);
    Ok((0..count)
        .map(|d| rng.standard_normal(RandomCoordinate::new(path, d, domain)))
        .collect())
}

/// Matching Dupire-variance and T-forward log-density samples. Density means
/// K*C_KK/P(0,T), equivalently f*p_f under the T-forward measure, not Q density.
/// Explicit-grid callers are responsible for the economic consistency of both
/// arrays. A surface constructor computes both from the same smile derivatives.
#[derive(Clone, Debug, PartialEq)]
pub struct HullWhiteLsvTarget {
    grid: LocalVarianceGrid,
    log_densities: Box<[f64]>,
}
impl HullWhiteLsvTarget {
    pub fn flat(
        volatility: f64,
        times: Vec<f64>,
        log_nodes: Vec<f64>,
        floor: f64,
        cap: f64,
    ) -> Result<Self, HullWhiteMcError> {
        hw_valid(volatility, "target_volatility", 0, true)?;
        if volatility == 0.0 {
            return Err(invalid("positive_target_volatility", 0));
        }
        let size = times
            .len()
            .checked_mul(log_nodes.len())
            .ok_or_else(|| invalid("target_shape", 0))?;
        let grid = LocalVarianceGrid::new(
            times,
            log_nodes,
            vec![volatility * volatility; size],
            floor,
            cap,
        )?;
        let densities = grid
            .time_nodes()
            .iter()
            .flat_map(|&t| {
                grid.log_moneyness_nodes().iter().map(move |&k| {
                    if t == 0.0 {
                        0.0
                    } else {
                        let root = volatility * t.sqrt();
                        standard_normal_pdf(-k / root - 0.5 * root) / root
                    }
                })
            })
            .collect();
        Self::new(grid, densities)
    }
    pub fn new(grid: LocalVarianceGrid, log_densities: Vec<f64>) -> Result<Self, HullWhiteMcError> {
        if grid.time_nodes()[0] != 0.0 || grid.values().len() != log_densities.len() {
            return Err(invalid("target_shape", 0));
        }
        for (i, &v) in log_densities.iter().enumerate() {
            hw_valid(v, "forward_log_density", i, true)?;
        }
        if !grid.repairs().is_empty() {
            return Err(invalid("repaired_local_variance_target", 0));
        }
        Ok(Self {
            grid,
            log_densities: log_densities.into(),
        })
    }
    pub fn from_surface(
        surface: &dyn ImpliedVarianceSurface,
        times: Vec<f64>,
        log_nodes: Vec<f64>,
        floor: f64,
        cap: f64,
    ) -> Result<Self, HullWhiteMcError> {
        let grid = LocalVarianceGrid::from_surface(surface, times, log_nodes, floor, cap)?;
        let mut densities = Vec::with_capacity(grid.values().len());
        for &t in grid.time_nodes() {
            for &k in grid.log_moneyness_nodes() {
                densities.push(if t == 0.0 {
                    0.0
                } else {
                    let call = surface.forward_call_evaluation(t, k, 1.0)?;
                    call.strike * call.call_density
                });
            }
        }
        Self::new(grid, densities)
    }
    #[must_use]
    pub const fn grid(&self) -> &LocalVarianceGrid {
        &self.grid
    }
    #[must_use]
    pub fn log_densities(&self) -> &[f64] {
        &self.log_densities
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HullWhiteCalibrationDiagnostics {
    pub time: f64,
    pub minimum_effective_samples: f64,
    pub fallback_nodes: usize,
    pub mean_relative_discount: f64,
    pub mean_discounted_normalized_equity: f64,
    pub maximum_rate_correction: f64,
}

#[derive(Clone, Debug)]
pub struct CalibratedHullWhiteLsv {
    pub surface: LsvLeverageSurface,
    pub diagnostics: Box<[HullWhiteCalibrationDiagnostics]>,
    /// Row-major discounted conditional second moment of exp(nu*X).
    pub conditional_second_moments: Box<[f64]>,
    /// Row-major variance correction subtracted from the deterministic Dupire target.
    pub rate_corrections: Box<[f64]>,
}

pub fn calibrate_hull_white_lsv(
    target: &HullWhiteLsvTarget,
    factor: Bergomi1Factor,
    rates: &HullWhite1Factor,
    correlation: HybridCorrelation,
    initial_spot: f64,
    config: &LsvParticleConfig,
) -> Result<CalibratedHullWhiteLsv, HullWhiteMcError> {
    if config.retain_reverse_trace() {
        return Err(HullWhiteError::Unsupported {
            feature: "hybrid calibration reverse trace (price-only API)",
        }
        .into());
    }
    if factor.correlation() != correlation.equity_vol {
        return Err(invalid("equity_vol_correlation_mismatch", 0));
    }
    let grid = target.grid();
    let times = grid.time_nodes();
    let xs = grid.log_moneyness_nodes();
    let m = xs.len();
    let n = config.particle_count();
    let initial = HybridState::initial(initial_spot)?;
    let mut states = vec![initial; n];
    let kernels = times
        .windows(2)
        .map(|w| Kernel::new(rates, factor.mean_reversion(), correlation, w[0], w[1]))
        .collect::<Result<Vec<_>, _>>()?;
    let rng = Philox4x32::from_seed(config.seed());
    let steps = kernels.len();
    steps
        .checked_mul(4)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| invalid("random_dimension", steps))?;
    let mut values = Vec::with_capacity(grid.values().len());
    let mut moments = Vec::with_capacity(values.capacity());
    let mut corrections = Vec::with_capacity(values.capacity());
    let mut diagnostics = Vec::new();
    for (r, &t) in times.iter().enumerate() {
        let shift = rates.rate_shift(t)?;
        let half_v = 0.5 * rates.integrated_variance(t)?;
        let mut sorted = states
            .iter()
            .map(|s| {
                let weight = (-s.integrated_rate_factor - half_v).exp();
                let a2 = (2.0 * factor.vol_of_vol() * s.volatility_factor).exp();
                (
                    (s.normalized_equity / initial_spot).ln(),
                    weight,
                    weight * (s.rate_factor + shift),
                    a2,
                    s.normalized_equity,
                )
            })
            .collect::<Vec<_>>();
        if sorted.iter().any(|p| {
            !p.1.is_finite() || p.1 <= 0.0 || !p.2.is_finite() || !p.3.is_finite() || p.3 <= 0.0
        }) {
            return Err(invalid("calibration_particle", r));
        }
        sorted.sort_by(|p, q| p.0.total_cmp(&q.0));
        let mut suffix_d = vec![0.0; n + 1];
        let mut suffix_y = vec![0.0; n + 1];
        let (mut sd, mut sy, mut sdf) =
            (NeumaierSum::new(), NeumaierSum::new(), NeumaierSum::new());
        for i in (0..n).rev() {
            sd.add(sorted[i].1);
            sy.add(sorted[i].2);
            sdf.add(sorted[i].1 * sorted[i].4);
            suffix_d[i] = sd.total();
            suffix_y[i] = sy.total();
        }
        let mut row = vec![None; m];
        let analytic = r == 0 || (rates.is_deterministic() && factor.vol_of_vol() == 0.0);
        for (j, &x) in xs.iter().enumerate() {
            if analytic {
                row[j] = Some((1.0, 0.0, n as f64));
                continue;
            }
            let h = config.log_bandwidth();
            let lo = sorted.partition_point(|p| p.0 <= x - h);
            let hi = sorted.partition_point(|p| p.0 < x + h);
            let (mut w, mut w2, mut wa2) =
                (NeumaierSum::new(), NeumaierSum::new(), NeumaierSum::new());
            for p in &sorted[lo..hi] {
                let u = (p.0 - x) / h;
                let kw = (1.0 - u * u).powi(2) * p.1;
                w.add(kw);
                w2.add(kw * kw);
                wa2.add(kw * p.3);
            }
            let ess = w.total() * w.total() / w2.total();
            if !ess.is_finite() || ess < config.minimum_effective_samples() {
                continue;
            }
            let correction = if rates.is_deterministic() {
                0.0
            } else {
                let density = target.log_densities[r * m + j];
                if density == 0.0 {
                    continue;
                }
                let above = sorted.partition_point(|p| p.0 <= x);
                // E[Dbar*(r-f0)]=0 exactly. Empirical centering reduces noise;
                // its finite-population ratio bias is part of this versioned scheme.
                let q = (suffix_y[above] - suffix_d[above] / suffix_d[0] * suffix_y[0]) / n as f64;
                2.0 * q / density
            };
            let second = wa2.total() / w.total();
            if !second.is_finite() || second <= 0.0 || !correction.is_finite() {
                return Err(invalid("conditional_estimator", r * m + j));
            }
            row[j] = Some((second, correction, ess));
        }
        let supported = row
            .iter()
            .enumerate()
            .filter_map(|(j, v)| v.map(|_| j))
            .collect::<Vec<_>>();
        if supported.is_empty() {
            return Err(invalid("no_supported_calibration_node", r));
        }
        let mut fallback = 0;
        let mut max_c: f64 = 0.0;
        let min_ess = supported
            .iter()
            .map(|&j| row[j].unwrap().2)
            .fold(f64::INFINITY, f64::min);
        for j in 0..m {
            let donor = if row[j].is_some() {
                j
            } else {
                fallback += 1;
                *supported
                    .iter()
                    .min_by(|&&p, &&q| {
                        (xs[p] - xs[j])
                            .abs()
                            .total_cmp(&(xs[q] - xs[j]).abs())
                            .then(p.cmp(&q))
                    })
                    .unwrap()
            };
            let (second, correction, _) = row[donor].unwrap();
            let value = (grid.values()[r * m + j] - correction) / second;
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid("non_positive_rate_corrected_variance", r * m + j));
            }
            values.push(value);
            moments.push(second);
            corrections.push(correction);
            max_c = max_c.max(correction.abs());
        }
        diagnostics.push(HullWhiteCalibrationDiagnostics {
            time: t,
            minimum_effective_samples: min_ess,
            fallback_nodes: fallback,
            mean_relative_discount: sd.total() / n as f64,
            mean_discounted_normalized_equity: sdf.total() / n as f64,
            maximum_rate_correction: max_c,
        });
        if r < steps {
            for (p, state) in states.iter_mut().enumerate() {
                let x = (state.normalized_equity / initial_spot).ln();
                let j = xs.partition_point(|v| *v <= x).saturating_sub(1).min(m - 2);
                let w = ((x - xs[j]) / (xs[j + 1] - xs[j])).clamp(0.0, 1.0);
                let l2 = (1.0 - w) * values[r * m + j] + w * values[r * m + j + 1];
                let z = std::array::from_fn(|d| {
                    rng.standard_normal(RandomCoordinate::new(
                        p as u64,
                        (d * steps + r) as u32,
                        RandomDomain::LsvCalibration,
                    ))
                });
                *state = kernels[r].advance(*state, l2, factor.vol_of_vol(), z, r)?;
            }
        }
    }
    Ok(CalibratedHullWhiteLsv {
        surface: LsvLeverageSurface::new(times.to_vec(), xs.to_vec(), values, initial_spot)?,
        diagnostics: diagnostics.into(),
        conditional_second_moments: moments.into(),
        rate_corrections: corrections.into(),
    })
}
