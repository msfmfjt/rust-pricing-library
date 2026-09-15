//! Equity/Hull–White hybrid simulation and discounted-particle LSV calibration.
//! The default equity state is S0*S/F0(t), before proportional dividends.
//! Explicit escrowed cash mode simulates normalized residual equity and
//! calibrates a continuous target coordinate including the stochastic bond reserve.

use crate::mc::lsv::{LsvError, LsvLeverageSurface};

use crate::mc::{LocalVolError, LocalVolTimeGrid, Philox4x32, RandomCoordinate, RandomDomain};

use crate::market::{
    ImpliedVarianceSurface, LocalVarianceGrid, MarketIvSurface, durrleman_density_factor,
};

use crate::models::hull_white::hw_valid;

use crate::models::hull_white_dividends::HullWhiteDividendPlan;

use crate::models::{
    Bergomi1Factor, HullWhite1Factor, HullWhiteError, HullWhiteHybridTransition, HybridCorrelation,
    RoughBergomi,
};

use pricing_numerics::{NeumaierSum, standard_normal_pdf};

use std::{error::Error, fmt};

mod rough;
mod two_factor;
pub use two_factor::{BERGOMI_TWO_FACTOR_HW_SCHEME, Bergomi2FactorHullWhiteDriverPlan};

pub use rough::{
    HybridVolatilityFactor, ROUGH_BERGOMI_SCHEME, ROUGH_CASH_LSV_CALIBRATION,
    ROUGH_LSV_CALIBRATION, RoughBergomiDriverPlan,
};

pub(in crate::engine) mod reverse;

pub use reverse::{
    HULL_WHITE_AAD_METHOD, HullWhiteCalibrationAdjoints, HullWhitePathAdjoints,
    HullWhiteRecordedPath,
};

pub const HULL_WHITE_EQUITY_SCHEME: &str = "equity-hw1f-joint-gaussian-log-euler-v1";

pub const HULL_WHITE_LSV_CALIBRATION: &str = "lsv-hw-discounted-quartic-centered-rate-v1";

pub const HULL_WHITE_CASH_LSV_CALIBRATION: &str = "lsv-hw-escrowed-quadratic-quartic-v1";

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

impl From<crate::market::MarketError> for HullWhiteMcError {
    fn from(e: crate::market::MarketError) -> Self {
        Self::Model(e.into())
    }
}

pub(in crate::engine) fn invalid(field: &'static str, index: usize) -> HullWhiteMcError {
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
pub(in crate::engine) struct Kernel {
    pub(in crate::engine) transition: HullWhiteHybridTransition,
    pub(in crate::engine) loading: [[f64; 4]; 4],
    pub(in crate::engine) dt: f64,
    pub(in crate::engine) integrated_shift: f64,
}

impl Kernel {
    pub(in crate::engine) fn new(
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
    pub(in crate::engine) fn advance(
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
        self.advance_innovations(state, leverage_squared, nu, noise, step)
    }
    pub(in crate::engine) fn advance_innovations(
        &self,
        state: HybridState,
        leverage_squared: f64,
        nu: f64,
        noise: [f64; 4],
        step: usize,
    ) -> Result<HybridState, HullWhiteMcError> {
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
pub(in crate::engine) fn covariance_loading<const N: usize>(
    cov: [[f64; N]; N],
) -> Result<[[f64; N]; N], HullWhiteMcError> {
    let scales = std::array::from_fn::<_, N, _>(|i| cov[i][i].sqrt());
    let mut l = [[0.0; N]; N];
    for i in 0..N {
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
    RoughBergomi {
        factor: RoughBergomi,
        initial_volatility: f64,
    },
    RoughBergomiLsv {
        factor: RoughBergomi,
        leverage: LsvLeverageSurface,
    },
    Bergomi2FactorLsv {
        factor: crate::models::Bergomi2Factor,
        second_vol_rate_correlation: f64,
        leverage: LsvLeverageSurface,
    },
    BergomiLsv {
        factor: Bergomi1Factor,
        leverage: LsvLeverageSurface,
    },
}

impl HybridEquityVolatility {
    pub(in crate::engine) fn lsv(&self) -> Option<(HybridVolatilityFactor, &LsvLeverageSurface)> {
        match self {
            Self::Bergomi2FactorLsv {
                factor,
                leverage,
                second_vol_rate_correlation,
            } => Some((
                HybridVolatilityFactor::BergomiTwoFactor {
                    factor: *factor,
                    second_vol_rate_correlation: *second_vol_rate_correlation,
                },
                leverage,
            )),
            Self::BergomiLsv { factor, leverage } => Some(((*factor).into(), leverage)),
            Self::RoughBergomiLsv { factor, leverage } => Some(((*factor).into(), leverage)),
            _ => None,
        }
    }
    pub(in crate::engine) fn rough(&self) -> Option<RoughBergomi> {
        match self {
            Self::RoughBergomi { factor, .. } | Self::RoughBergomiLsv { factor, .. } => {
                Some(*factor)
            }
            _ => None,
        }
    }
    pub(in crate::engine) fn direct_volatility(&self) -> Option<f64> {
        match self {
            Self::BlackScholes(v)
            | Self::RoughBergomi {
                initial_volatility: v,
                ..
            } => Some(*v),
            _ => None,
        }
    }
    pub(in crate::engine) fn log_vol_coefficient(&self) -> f64 {
        self.lsv().map_or_else(
            || self.rough().map_or(0.0, |v| 0.5 * v.vol_of_vol()),
            |(v, _)| v.vol_of_vol(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct HullWhiteEquityPlan {
    pub(in crate::engine) rates: HullWhite1Factor,
    pub(in crate::engine) volatility: HybridEquityVolatility,
    pub(in crate::engine) correlation: HybridCorrelation,
    pub(in crate::engine) times: Box<[f64]>,
    pub(in crate::engine) kernels: Box<[Kernel]>,
    pub(in crate::engine) dividends: Option<HullWhiteDividendPlan>,
    pub(in crate::engine) rough_driver: Option<RoughBergomiDriverPlan>,
    pub(in crate::engine) two_factor_driver: Option<Bergomi2FactorHullWhiteDriverPlan>,
}

impl HullWhiteEquityPlan {
    pub fn new(
        rates: HullWhite1Factor,
        volatility: HybridEquityVolatility,
        correlation: HybridCorrelation,
        grid: &LocalVolTimeGrid,
    ) -> Result<Self, HullWhiteMcError> {
        if let Some(sigma) = volatility.direct_volatility() {
            hw_valid(sigma, "equity_volatility", 0, true)?;
        }
        if let Some(factor) = volatility.rough()
            && factor.correlation() != correlation.equity_vol
        {
            return Err(invalid("equity_vol_correlation_mismatch", 0));
        }
        let k = if let Some((factor, leverage)) = volatility.lsv() {
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
        } else {
            0.0
        };
        let rough_driver = volatility
            .rough()
            .map(|factor| RoughBergomiDriverPlan::new(factor, &rates, correlation, grid))
            .transpose()?;
        let two_factor_driver = volatility
            .lsv()
            .map(|(f, _)| f.two_factor_driver(&rates, correlation, grid.nodes()))
            .transpose()?
            .flatten();
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
            dividends: None,
            rough_driver,
            two_factor_driver,
        })
    }
    pub fn with_dividends(
        mut self,
        dividends: HullWhiteDividendPlan,
    ) -> Result<Self, HullWhiteMcError> {
        if dividends.rates() != &self.rates {
            return Err(invalid("dividend_rate_model_mismatch", 0));
        }
        if dividends.nodes().len() != self.times.len()
            || dividends
                .nodes()
                .iter()
                .zip(self.times.iter())
                .any(|(n, t)| n.time() != *t)
        {
            return Err(invalid("dividend_plan_time_mismatch", 0));
        }
        self.dividends = Some(dividends);
        Ok(self)
    }
    #[must_use]
    pub fn dividends(&self) -> Option<&HullWhiteDividendPlan> {
        self.dividends.as_ref()
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
            HybridEquityVolatility::RoughBergomi {
                factor,
                initial_volatility,
            } => vec![
                2.0,
                factor.hurst(),
                factor.vol_of_vol(),
                factor.correlation(),
                *initial_volatility,
            ],
            HybridEquityVolatility::RoughBergomiLsv { factor, .. } => vec![
                3.0,
                factor.hurst(),
                factor.vol_of_vol(),
                factor.correlation(),
            ],
            HybridEquityVolatility::Bergomi2FactorLsv {
                factor,
                second_vol_rate_correlation,
                ..
            } => {
                use crate::models::BergomiDynamics;
                let mut values = vec![4.0, *second_vol_rate_correlation];
                values.extend(factor.parameters());
                values
            }
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
    /// Evolve already correlated innovations [dW_S, OU_V1, OU_r, integral_r,
    /// OU_V2]. The final entry is zero/unused for a one-factor or BS asset.
    /// For rough models, entries 1 and 4 are dW_vol and the near-cell integral.
    /// This shares one centered rate path across a multi-asset simulation.
    pub fn evolve_with_innovations(
        &self,
        spot: f64,
        innovations: &[[f64; 5]],
    ) -> Result<Vec<HybridState>, HullWhiteMcError> {
        if innovations.len() != self.kernels.len()
            || innovations.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(invalid("hybrid_external_innovations", innovations.len()));
        }
        if self
            .volatility
            .lsv()
            .is_some_and(|(_, l)| l.initial_f() != spot)
            || self
                .dividends
                .as_ref()
                .is_some_and(|d| d.initial_spot() != spot)
        {
            return Err(invalid("hybrid_external_initial_spot", 0));
        }
        let mut state = HybridState::initial(spot)?;
        let mut states = vec![state];
        let mut x = [0.0; 2];
        let rough = self
            .rough_driver
            .as_ref()
            .map(|driver| {
                let dw: Vec<_> = innovations.iter().map(|z| z[1]).collect();
                let near: Vec<_> = innovations.iter().map(|z| z[4]).collect();
                driver.normalized_with_innovations(&dw, &near)
            })
            .transpose()?;
        for (i, (kernel, noise)) in self.kernels.iter().zip(innovations).enumerate() {
            let l2 = if let Some((_, leverage)) = self.volatility.lsv() {
                leverage.squared_leverage_at(
                    self.times[i],
                    target_state(self.dividends.as_ref(), i, state)?.0,
                )?
            } else {
                self.volatility.direct_volatility().unwrap().powi(2)
            };
            state = kernel.advance_innovations(
                state,
                l2,
                self.volatility.log_vol_coefficient(),
                [noise[0], noise[1], noise[2], noise[3]],
                i,
            )?;
            if let HybridEquityVolatility::Bergomi2FactorLsv { factor, .. } = self.volatility {
                let k = factor.mean_reversions();
                x[0] = (-k[0] * kernel.dt).exp() * x[0] + noise[1];
                x[1] = (-k[1] * kernel.dt).exp() * x[1] + noise[4];
                let w = factor.normalized_weights();
                state.volatility_factor = w[0] * x[0] + w[1] * x[1];
                hw_valid(
                    state.volatility_factor,
                    "hybrid_external_volatility",
                    i,
                    false,
                )?;
            }
            if let Some(values) = &rough {
                state.volatility_factor = values[i + 1];
            }
            states.push(state);
        }
        Ok(states)
    }
    pub fn pseudo_shocks(
        &self,
        seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, HullWhiteMcError> {
        hybrid_shocks(
            seed,
            path,
            self.kernels.len(),
            self.random_factor_count(),
            domain,
        )
    }
    #[must_use]
    pub fn random_factor_count(&self) -> usize {
        if self.rough_driver.is_some() || self.two_factor_driver.is_some() {
            5
        } else {
            4
        }
    }
    #[must_use]
    pub fn is_rough(&self) -> bool {
        self.rough_driver.is_some()
    }
    #[must_use]
    pub fn is_direct_rough(&self) -> bool {
        self.is_rough() && self.volatility.direct_volatility().is_some()
    }
    #[must_use]
    pub fn scheme(&self) -> &'static str {
        if self.two_factor_driver.is_some() {
            BERGOMI_TWO_FACTOR_HW_SCHEME
        } else if self.is_rough() {
            ROUGH_BERGOMI_SCHEME
        } else {
            HULL_WHITE_EQUITY_SCHEME
        }
    }
    #[must_use]
    pub fn calibration_method(&self) -> Option<&'static str> {
        self.volatility
            .lsv()
            .map(|_| match (self.is_rough(), self.dividends.is_some()) {
                (true, true) => ROUGH_CASH_LSV_CALIBRATION,
                (true, false) => ROUGH_LSV_CALIBRATION,
                (false, true) => HULL_WHITE_CASH_LSV_CALIBRATION,
                (false, false) => HULL_WHITE_LSV_CALIBRATION,
            })
    }
    /// Four factor-major blocks (five for rough or two-factor Bergomi). All blocks must be
    /// sign-reversed for an antithetic path; Brownian bridge acts on each block.
    pub fn evolve_path(
        &self,
        spot: f64,
        shocks: &[f64],
    ) -> Result<Vec<HybridState>, HullWhiteMcError> {
        let n = self.kernels.len();
        if Some(shocks.len()) != n.checked_mul(self.random_factor_count()) {
            return Err(invalid("shock_count", shocks.len()));
        }
        let mut state = HybridState::initial(spot)?;
        if self
            .dividends
            .as_ref()
            .is_some_and(|d| d.initial_spot() != spot)
        {
            return Err(invalid("dividend_initial_spot", 0));
        }
        if let Some((_, leverage)) = self.volatility.lsv()
            && spot != leverage.initial_f()
        {
            return Err(invalid("leverage_initial_spot", 0));
        }
        let rough_values = self
            .rough_driver
            .as_ref()
            .map(|driver| driver.normalized(shocks))
            .transpose()?;
        let factor_values = if let Some(driver) = &self.two_factor_driver {
            Some(driver.evolve(shocks)?)
        } else {
            rough_values
        };
        let mut states = vec![state];
        for (i, kernel) in self.kernels.iter().enumerate() {
            let l2 = if let Some((_, leverage)) = self.volatility.lsv() {
                leverage.squared_leverage_at(
                    self.times[i],
                    target_state(self.dividends.as_ref(), i, state)?.0,
                )?
            } else {
                self.volatility.direct_volatility().unwrap().powi(2)
            };
            let nu = self.volatility.log_vol_coefficient();
            state = kernel.advance(state, l2, nu, std::array::from_fn(|j| shocks[j * n + i]), i)?;
            if let Some(values) = &factor_values {
                state.volatility_factor = values[i + 1];
            }
            states.push(state);
        }
        Ok(states)
    }
}

pub(in crate::engine) fn hybrid_shocks(
    seed: u64,
    path: u64,
    steps: usize,
    factors: usize,
    domain: RandomDomain,
) -> Result<Vec<f64>, HullWhiteMcError> {
    let count = steps
        .checked_mul(factors)
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
    pub(in crate::engine) grid: LocalVarianceGrid,
    pub(in crate::engine) log_densities: Box<[f64]>,
    pub(in crate::engine) market_iv: Option<std::sync::Arc<MarketIvSurface>>,
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
            market_iv: None,
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
    /// Retain the exact quote interpolation for VegaKT. Quotes are Black IV in
    /// the target forward coordinate (escrow F for the explicit cash model).
    pub fn from_market_iv(
        surface: MarketIvSurface,
        times: Vec<f64>,
        log_nodes: Vec<f64>,
        floor: f64,
        cap: f64,
    ) -> Result<Self, HullWhiteMcError> {
        let mut target = Self::from_surface(&surface, times, log_nodes, floor, cap)?;
        target.market_iv = Some(std::sync::Arc::new(surface));
        Ok(target)
    }
    #[must_use]
    pub fn market_iv_surface(&self) -> Option<&MarketIvSurface> {
        self.market_iv.as_deref()
    }
    /// Exact transpose of both Dupire variance and K*p_F^T(K). No finite
    /// difference or particle recalibration is performed by this operation.
    pub fn reverse_market_iv(
        &self,
        variance_seeds: &[f64],
        density_seeds: &[f64],
    ) -> Result<Vec<f64>, HullWhiteMcError> {
        let surface = self
            .market_iv
            .as_ref()
            .ok_or_else(|| invalid("market_iv_source_not_retained", 0))?;
        let n = self.grid.values().len();
        if variance_seeds.len() != n
            || density_seeds.len() != n
            || variance_seeds
                .iter()
                .chain(density_seeds)
                .any(|v| !v.is_finite())
        {
            return Err(invalid("market_iv_target_adjoint_shape_or_value", 0));
        }
        let mut out = vec![0.0; surface.implied_volatilities().len()];
        let m = self.grid.log_moneyness_nodes().len();
        for (row, &t) in self.grid.time_nodes().iter().enumerate() {
            let surface_time = if t == 0.0 {
                self.grid.time_nodes()[1]
            } else {
                t
            };
            for (col, &x) in self.grid.log_moneyness_nodes().iter().enumerate() {
                let i = row * m + col;
                let v = surface.total_variance_derivatives(surface_time, x)?;
                let w = v.total_variance;
                let wx = v.log_moneyness_derivative;
                let g = durrleman_density_factor(surface_time, x, v)?;
                let mut g_bar = -variance_seeds[i] * v.time_derivative / (g * g);
                let mut w_bar = 0.0;
                if t > 0.0 {
                    let root = w.sqrt();
                    let d2 = -x / root - 0.5 * root;
                    let normal = standard_normal_pdf(d2) / root;
                    g_bar += density_seeds[i] * normal;
                    w_bar += density_seeds[i]
                        * (normal * g)
                        * (-d2 * (x / (2.0 * w * root) - 1.0 / (4.0 * root)) - 0.5 / w);
                }
                let u = 1.0 - x * wx / (2.0 * w);
                w_bar += g_bar * (u * x * wx / (w * w) + wx * wx / (4.0 * w * w));
                let wx_bar = g_bar * (-u * x / w - 0.5 * wx * (1.0 / w + 0.25));
                surface.transpose_accumulate(
                    surface_time,
                    x,
                    [w_bar, wx_bar, 0.5 * g_bar, variance_seeds[i] / g],
                    &mut out,
                )?;
            }
        }
        Ok(out)
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

pub(in crate::engine) fn target_state(
    dividends: Option<&HullWhiteDividendPlan>,
    index: usize,
    state: HybridState,
) -> Result<(f64, f64), HullWhiteMcError> {
    dividends.map_or(Ok((state.normalized_equity, 0.0)), |d| {
        d.nodes()[index]
            .target_state(state.normalized_equity, state.rate_factor)
            .map_err(Into::into)
    })
}
