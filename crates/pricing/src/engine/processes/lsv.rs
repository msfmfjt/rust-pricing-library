//! Particle-calibrated one- or two-factor Bergomi LSV in the continuous `f` coordinate.
//!
//! Calibration uses a compact quartic kernel in `log(f / f0)`, a versioned
//! variation of SSRN 1885032 (20)-(21). Time interpolation is left-constant;
//! spatial interpolation is linear with flat tails. Pricing paths are independent
//! of calibration particles. Reverse differentiation includes the conditional
//! expectation estimator, rather than freezing it under a Dupire variance bump.

use crate::market::MarketError;
use crate::mc::{LocalVolTimeGrid, Philox4x32, RandomCoordinate, RandomDomain};
use crate::models::{
    Bergomi1Factor, BergomiDynamics, BergomiError, InnovationSource, OrthogonalNormals, OuInnovations,
};
use std::{error::Error, fmt};

#[cfg(test)]
mod innovation_contracts;

pub const BERGOMI_LSV_SCHEME: &str = "bergomi-lsv-log-euler-exact-ou-v1";
pub const BERGOMI_TWO_FACTOR_LSV_SCHEME: &str = "bergomi-two-factor-lsv-log-euler-exact-ou-v1";
pub const LSV_PARTICLE_CALIBRATION: &str = "lsv-quartic-log-f-left-time-v1";
pub const LSV_CALIBRATION_REVERSE: &str = "lsv-discrete-particle-vjp-v1";

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LsvError {
    InvalidInput {
        field: &'static str,
        index: usize,
    },
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    NoSupportedCalibrationNode {
        time_index: usize,
    },
    NonFiniteState {
        time_index: usize,
        path: usize,
    },
    ReverseTraceNotRetained,
    Core(crate::core::CoreError),
    Market(MarketError),
    Bergomi(BergomiError),
}

impl fmt::Display for LsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { field, index } => {
                write!(f, "invalid LSV {field} at index {index}")
            }
            Self::LengthMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "LSV {field}: expected {expected} values, received {actual}"
            ),
            Self::NoSupportedCalibrationNode { time_index } => write!(
                f,
                "no adequately supported LSV calibration node at time index {time_index}"
            ),
            Self::NonFiniteState { time_index, path } => write!(
                f,
                "invalid LSV state at time index {time_index}, path {path}"
            ),
            Self::ReverseTraceNotRetained => write!(
                f,
                "LSV calibration reverse requires retain_reverse_trace=true"
            ),
            Self::Core(e) => e.fmt(f),
            Self::Market(e) => e.fmt(f),
            Self::Bergomi(e) => e.fmt(f),
        }
    }
}
impl Error for LsvError {}
impl From<BergomiError> for LsvError {
    fn from(e: BergomiError) -> Self {
        match e {
            BergomiError::Core(e) => Self::Core(e),
            e => Self::Bergomi(e),
        }
    }
}
/// Rough Bergomi parameters and kernels report Hull–White-family errors; the
/// standalone rough-LSV engine reports them as invalid LSV inputs.
impl From<crate::models::HullWhiteError> for LsvError {
    fn from(e: crate::models::HullWhiteError) -> Self {
        match e {
            crate::models::HullWhiteError::InvalidInput { field, index } => {
                Self::InvalidInput { field, index }
            }
            crate::models::HullWhiteError::Market(e) => Self::Market(e),
            _ => Self::InvalidInput {
                field: "rough_bergomi",
                index: 0,
            },
        }
    }
}

impl From<MarketError> for LsvError {
    fn from(e: MarketError) -> Self {
        Self::Market(e)
    }
}
impl From<crate::core::CoreError> for LsvError {
    fn from(e: crate::core::CoreError) -> Self {
        Self::Core(e)
    }
}

pub(in crate::engine) fn valid(
    value: f64,
    field: &'static str,
    index: usize,
    positive: bool,
) -> Result<(), LsvError> {
    if !value.is_finite() || (positive && value <= 0.0) {
        return Err(LsvError::InvalidInput { field, index });
    }
    Ok(())
}
pub(in crate::engine) fn length(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), LsvError> {
    if expected == actual {
        Ok(())
    } else {
        Err(LsvError::LengthMismatch {
            field,
            expected,
            actual,
        })
    }
}

/// All numerical choices are explicit. The bandwidth is measured in log-f units,
/// is fixed during a sensitivity calculation, and must be refined with N.
#[derive(Clone, Debug, PartialEq)]
pub struct LsvParticleConfig {
    pub(in crate::engine) particle_count: usize,
    pub(in crate::engine) seed: u64,
    pub(in crate::engine) log_bandwidth: f64,
    pub(in crate::engine) minimum_effective_samples: f64,
    pub(in crate::engine) retain_reverse_trace: bool,
}
impl LsvParticleConfig {
    pub fn new(
        particle_count: usize,
        seed: u64,
        log_bandwidth: f64,
        minimum_effective_samples: f64,
        retain_reverse_trace: bool,
    ) -> Result<Self, LsvError> {
        if particle_count < 2 {
            return Err(LsvError::InvalidInput {
                field: "particle_count",
                index: 0,
            });
        }
        valid(log_bandwidth, "log_bandwidth", 0, true)?;
        valid(
            minimum_effective_samples,
            "minimum_effective_samples",
            0,
            true,
        )?;
        if minimum_effective_samples < 1.0 || minimum_effective_samples > particle_count as f64 {
            return Err(LsvError::InvalidInput {
                field: "minimum_effective_samples",
                index: 0,
            });
        }
        Ok(Self {
            particle_count,
            seed,
            log_bandwidth,
            minimum_effective_samples,
            retain_reverse_trace,
        })
    }
    #[must_use]
    pub const fn particle_count(&self) -> usize {
        self.particle_count
    }
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }
    #[must_use]
    pub const fn log_bandwidth(&self) -> f64 {
        self.log_bandwidth
    }
    #[must_use]
    pub const fn minimum_effective_samples(&self) -> f64 {
        self.minimum_effective_samples
    }
    #[must_use]
    pub const fn retain_reverse_trace(&self) -> bool {
        self.retain_reverse_trace
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LsvConditionalMoments {
    pub second: f64,
    pub third: f64,
    pub fourth: f64,
    pub effective_samples: f64,
    /// The donor moment node used for an under-supported tail or interior cell.
    pub source_node: usize,
    pub extrapolated: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LsvCalibrationRowDiagnostics {
    pub time: f64,
    pub minimum_effective_samples: f64,
    pub extrapolated_nodes: usize,
    pub particle_mean_f: f64,
}

/// Immutable squared relative leverage. Values multiply `a^2` in `df/f`.
#[derive(Clone, Debug, PartialEq)]
pub struct LsvLeverageSurface {
    pub(in crate::engine) times: Box<[f64]>,
    pub(in crate::engine) log_nodes: Box<[f64]>,
    pub(in crate::engine) values: Box<[f64]>,
    pub(in crate::engine) initial_f: f64,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::engine) struct Lookup {
    pub(in crate::engine) value: f64,
    pub(in crate::engine) derivative_log_f: f64,
    pub(in crate::engine) left: usize,
    pub(in crate::engine) weight: f64,
}
impl Lookup {
    pub(in crate::engine) fn transpose(self, seed: f64, output: &mut [f64]) {
        output[self.left] += (1.0 - self.weight) * seed;
        output[self.left + 1] += self.weight * seed;
    }
}

impl LsvLeverageSurface {
    pub fn new(
        times: Vec<f64>,
        log_nodes: Vec<f64>,
        values: Vec<f64>,
        initial_f: f64,
    ) -> Result<Self, LsvError> {
        if times.len() < 2 || log_nodes.len() < 2 || times[0].to_bits() != 0.0f64.to_bits() {
            return Err(LsvError::InvalidInput {
                field: "surface_axes",
                index: 0,
            });
        }
        for (field, nodes) in [("times", &times), ("log_nodes", &log_nodes)] {
            for (i, &v) in nodes.iter().enumerate() {
                valid(v, field, i, false)?;
                if i > 0 && v <= nodes[i - 1] {
                    return Err(LsvError::InvalidInput { field, index: i });
                }
            }
        }
        let size = times
            .len()
            .checked_mul(log_nodes.len())
            .ok_or(LsvError::InvalidInput {
                field: "surface_size",
                index: 0,
            })?;
        length("leverage_values", size, values.len())?;
        for (i, &v) in values.iter().enumerate() {
            valid(v, "squared_leverage", i, true)?;
        }
        valid(initial_f, "initial_f", 0, true)?;
        Ok(Self {
            times: times.into_boxed_slice(),
            log_nodes: log_nodes.into_boxed_slice(),
            values: values.into_boxed_slice(),
            initial_f,
        })
    }
    #[must_use]
    pub fn times(&self) -> &[f64] {
        &self.times
    }
    #[must_use]
    pub fn log_nodes(&self) -> &[f64] {
        &self.log_nodes
    }
    #[must_use]
    pub fn squared_leverage(&self) -> &[f64] {
        &self.values
    }
    #[must_use]
    pub const fn initial_f(&self) -> f64 {
        self.initial_f
    }

    pub(in crate::engine) fn lookup(&self, time: f64, f: f64) -> Result<Lookup, LsvError> {
        valid(time, "lookup_time", 0, false)?;
        valid(f, "lookup_f", 0, true)?;
        if time < 0.0 || time > self.times[self.times.len() - 1] {
            return Err(LsvError::InvalidInput {
                field: "lookup_time_coverage",
                index: 0,
            });
        }
        Ok(self.lookup_row(self.row_at(time), (f / self.initial_f).ln()))
    }
    fn row_at(&self, time: f64) -> usize {
        self.times.partition_point(|t| *t <= time).saturating_sub(1)
    }
    pub(in crate::engine) fn lookup_row(&self, row: usize, x: f64) -> Lookup {
        let i = self
            .log_nodes
            .partition_point(|v| *v <= x)
            .saturating_sub(1)
            .min(self.log_nodes.len() - 2);
        self.lookup_cell(row, i, x)
    }
    /// `lookup_row` with the cell found by walking from the previous one. The
    /// cell is the same as `partition_point` finds for finite `x`, so the lookup
    /// is identical; nearby queries along one path cost O(1) instead of O(log n).
    pub(in crate::engine) fn lookup_row_from(
        &self,
        row: usize,
        x: f64,
        hint: &mut usize,
    ) -> Lookup {
        let nodes = &self.log_nodes;
        let mut count = (*hint + 1).min(nodes.len());
        while count < nodes.len() && nodes[count] <= x {
            count += 1;
        }
        while count > 0 && nodes[count - 1] > x {
            count -= 1;
        }
        *hint = count.saturating_sub(1).min(nodes.len() - 2);
        self.lookup_cell(row, *hint, x)
    }
    fn lookup_cell(&self, row: usize, i: usize, x: f64) -> Lookup {
        let n = self.log_nodes.len();
        let w =
            ((x - self.log_nodes[i]) / (self.log_nodes[i + 1] - self.log_nodes[i])).clamp(0.0, 1.0);
        let j = row * n + i;
        let value = (1.0 - w) * self.values[j] + w * self.values[j + 1];
        let derivative_log_f = if x <= self.log_nodes[0] || x >= self.log_nodes[n - 1] {
            0.0
        } else {
            (self.values[j + 1] - self.values[j]) / (self.log_nodes[i + 1] - self.log_nodes[i])
        };
        Lookup {
            value,
            derivative_log_f,
            left: j,
            weight: w,
        }
    }
    pub fn squared_leverage_at(&self, time: f64, f: f64) -> Result<f64, LsvError> {
        Ok(self.lookup(time, f)?.value)
    }
}

#[derive(Clone, Debug)]
pub struct BergomiLsvPlan<F: BergomiDynamics = Bergomi1Factor> {
    pub(in crate::engine) factor: F,
    pub(in crate::engine) surface: LsvLeverageSurface,
    pub(in crate::engine) times: Box<[f64]>,
    pub(in crate::engine) transitions: Box<[F::Transition]>,
    /// Leverage row used by each step, as `LsvLeverageSurface::lookup` resolves
    /// it from the step's start time. Resolved once instead of on every path.
    rows: Box<[usize]>,
}

/// Checks that an execution grid covers the leverage surface's knots inside
/// its horizon, and resolves the leverage row each step starts in. The grid may
/// insert payoff and dividend observations but cannot cross a knot.
pub(in crate::engine) fn step_rows(
    surface: &LsvLeverageSurface,
    times: &[f64],
) -> Result<Box<[usize]>, LsvError> {
    let end = times[times.len() - 1];
    if end > surface.times[surface.times.len() - 1] {
        return Err(LsvError::InvalidInput {
            field: "time_coverage",
            index: 0,
        });
    }
    for (i, &t) in surface.times.iter().enumerate().filter(|(_, t)| **t <= end) {
        if !times.iter().any(|x| x.to_bits() == t.to_bits()) {
            return Err(LsvError::InvalidInput {
                field: "missing_leverage_knot",
                index: i,
            });
        }
    }
    Ok(times[..times.len() - 1]
        .iter()
        .map(|&time| surface.row_at(time))
        .collect())
}

#[derive(Clone, Copy, Debug)]
pub(in crate::engine) struct Step {
    pub(in crate::engine) lookup: Lookup,
    pub(in crate::engine) multiplier_squared: f64,
    pub(in crate::engine) variance: f64,
    pub(in crate::engine) z: f64,
    pub(in crate::engine) dt: f64,
}

#[derive(Clone, Debug)]
pub struct BergomiLsvPath<F: BergomiDynamics = Bergomi1Factor> {
    pub(in crate::engine) states: Box<[f64]>,
    pub(in crate::engine) factors: Box<[F::State]>,
    pub(in crate::engine) steps: Box<[Step]>,
    pub(in crate::engine) transitions: Box<[F::Transition]>,
    pub(in crate::engine) factor: F,
    pub(in crate::engine) value_count: usize,
    innovation_source: InnovationSource,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LsvPathAdjoints<S = f64> {
    pub initial_f: f64,
    pub initial_factor: S,
    pub squared_leverage: Box<[f64]>,
    pub spot_shocks: Box<[f64]>,
    pub orthogonal_shocks: Box<[f64]>,
}

impl<F: BergomiDynamics> BergomiLsvPlan<F> {
    /// The execution grid may insert payoff and dividend observations, but must
    /// contain every leverage knot inside its horizon. It cannot cross a knot.
    pub fn new(
        factor: F,
        surface: LsvLeverageSurface,
        time_grid: &LocalVolTimeGrid,
    ) -> Result<Self, LsvError> {
        let times = time_grid.nodes();
        let rows = step_rows(&surface, times)?;
        let transitions = times
            .windows(2)
            .map(|w| factor.transition(w[1] - w[0]).map_err(Into::into))
            .collect::<Result<Vec<_>, LsvError>>()?;
        Ok(Self {
            factor,
            surface,
            times: times.into(),
            transitions: transitions.into_boxed_slice(),
            rows,
        })
    }
    #[must_use]
    pub fn surface(&self) -> &LsvLeverageSurface {
        &self.surface
    }
    #[must_use]
    pub fn times(&self) -> &[f64] {
        &self.times
    }
    #[must_use]
    pub const fn factor(&self) -> F {
        self.factor
    }

    /// Factor-major layout: first n spot normals, then one n-normal block per
    /// volatility factor. The blocks are independent before the joint OU map.
    /// Keeping spot dimensions first preserves the LV random coordinates at nu=0.
    pub fn pseudo_shocks(
        &self,
        seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, LsvError> {
        lsv_shocks_with_factors(
            seed,
            path,
            self.transitions.len(),
            domain,
            1 + F::FACTOR_COUNT,
        )
    }

    pub fn evolve_path(
        &self,
        initial_f: f64,
        shocks: &[f64],
    ) -> Result<BergomiLsvPath<F>, LsvError> {
        let n = self.transitions.len();
        length("shocks", (1 + F::FACTOR_COUNT) * n, shocks.len())?;
        for (i, &z) in shocks.iter().enumerate() {
            valid(z, "shock", i, false)?;
        }
        self.evolve_with_factor(
            initial_f,
            &shocks[..n],
            |i, x| {
                self.factor.evolve_normals(
                    self.transitions[i],
                    x,
                    shocks[i],
                    OrthogonalNormals::new(&shocks[n + i..], n),
                )
            },
            InnovationSource::IndependentNormals,
        )
    }

    /// Price-only evolution. Writes the same f states as `evolve_path` for the
    /// same shocks, bit for bit, but keeps no reverse-mode step records and
    /// reuses the caller's buffer, so repeated pricing paths do not allocate.
    pub fn evolve_states(
        &self,
        initial_f: f64,
        shocks: &[f64],
        states: &mut Vec<f64>,
    ) -> Result<(), LsvError> {
        let n = self.transitions.len();
        length("shocks", (1 + F::FACTOR_COUNT) * n, shocks.len())?;
        for (i, &z) in shocks.iter().enumerate() {
            valid(z, "shock", i, false)?;
        }
        valid(initial_f, "initial_f", 0, true)?;
        states.clear();
        states.push(initial_f);
        let mut f = initial_f;
        let mut factor_state = F::State::default();
        let mut cell = 0;
        for i in 0..n {
            let dt = self.times[i + 1] - self.times[i];
            let lookup = self.surface.lookup_row_from(
                self.rows[i],
                (f / self.surface.initial_f).ln(),
                &mut cell,
            );
            let variance = lookup.value * self.factor.multiplier_squared(factor_state);
            f = advance(f, variance, dt, shocks[i], i, 0)?;
            factor_state = self.factor.evolve_normals(
                self.transitions[i],
                factor_state,
                shocks[i],
                OrthogonalNormals::new(&shocks[n + i..], n),
            );
            if !F::finite(factor_state) {
                return Err(LsvError::NonFiniteState {
                    time_index: i + 1,
                    path: 0,
                });
            }
            states.push(f);
        }
        Ok(())
    }

    /// Internal joint-driver entry. The global sampler supplies OU innovations
    /// with their full cross-asset covariance, after its independent-factor bridge.
    /// Shock adjoints refer to the spot normal and supplied OU innovations, so
    /// the OU loading is the identity and the spot-to-OU loading is zero.
    pub(in crate::engine) fn evolve_with_ou_innovations(
        &self,
        initial_f: f64,
        spot_shocks: &[f64],
        innovations: &[f64],
    ) -> Result<BergomiLsvPath<F>, LsvError> {
        let n = self.transitions.len();
        length("OU innovations", F::FACTOR_COUNT * n, innovations.len())?;
        for (i, &z) in innovations.iter().enumerate() {
            valid(z, "OU innovation", i, false)?;
        }
        self.evolve_with_factor(
            initial_f,
            spot_shocks,
            |i, x| {
                self.factor.evolve_increments(
                    self.transitions[i],
                    x,
                    OuInnovations::new(&innovations[i..], n),
                )
            },
            InnovationSource::JointOuIncrements,
        )
    }

    fn evolve_with_factor(
        &self,
        initial_f: f64,
        shocks: &[f64],
        next_factor: impl Fn(usize, F::State) -> F::State,
        innovation_source: InnovationSource,
    ) -> Result<BergomiLsvPath<F>, LsvError> {
        let n = self.transitions.len();
        length("spot shocks", n, shocks.len())?;
        valid(initial_f, "initial_f", 0, true)?;
        for (i, &z) in shocks.iter().enumerate() {
            valid(z, "spot shock", i, false)?;
        }
        let mut states = vec![initial_f];
        let mut factors = vec![F::State::default()];
        let mut steps = Vec::with_capacity(n);
        for i in 0..n {
            let dt = self.times[i + 1] - self.times[i];
            let lookup = self.surface.lookup(self.times[i], states[i])?;
            let multiplier_squared = self.factor.multiplier_squared(factors[i]);
            let variance = lookup.value * multiplier_squared;
            let next = advance(states[i], variance, dt, shocks[i], i, 0)?;
            let x = next_factor(i, factors[i]);
            if !F::finite(x) {
                return Err(LsvError::NonFiniteState {
                    time_index: i + 1,
                    path: 0,
                });
            }
            steps.push(Step {
                lookup,
                multiplier_squared,
                variance,
                z: shocks[i],
                dt,
            });
            states.push(next);
            factors.push(x);
        }
        Ok(BergomiLsvPath {
            states: states.into_boxed_slice(),
            factors: factors.into_boxed_slice(),
            steps: steps.into_boxed_slice(),
            transitions: self.transitions.clone(),
            innovation_source,
            factor: self.factor,
            value_count: self.surface.values.len(),
        })
    }
}

pub(in crate::engine) fn advance(
    f: f64,
    variance: f64,
    dt: f64,
    z: f64,
    time_index: usize,
    path: usize,
) -> Result<f64, LsvError> {
    let next = f * (-0.5 * variance * dt + variance.sqrt() * dt.sqrt() * z).exp();
    if !variance.is_finite() || variance <= 0.0 || !next.is_finite() || next <= 0.0 {
        return Err(LsvError::NonFiniteState {
            time_index: time_index + 1,
            path,
        });
    }
    Ok(next)
}

pub(in crate::engine) fn lsv_shocks_with_factors(
    seed: u64,
    path: u64,
    steps: usize,
    domain: RandomDomain,
    factors: usize,
) -> Result<Vec<f64>, LsvError> {
    let count = steps
        .checked_mul(factors)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or(LsvError::InvalidInput {
            field: "random_dimension",
            index: steps,
        })?;
    let rng = Philox4x32::from_seed(seed);
    Ok((0..count)
        .map(|d| rng.standard_normal(RandomCoordinate::new(path, d, domain)))
        .collect())
}

impl<F: BergomiDynamics> BergomiLsvPath<F> {
    #[must_use]
    pub fn states(&self) -> &[f64] {
        &self.states
    }
    #[must_use]
    pub fn factors(&self) -> &[F::State] {
        &self.factors
    }

    /// Price-only callers never allocate adjoint buffers. Seeds may be placed at
    /// any path observation; affine Spot seeds must first be multiplied by B(t).
    pub fn reverse(&self, state_seeds: &[f64]) -> Result<LsvPathAdjoints<F::State>, LsvError> {
        length("state_seeds", self.states.len(), state_seeds.len())?;
        for (i, &s) in state_seeds.iter().enumerate() {
            valid(s, "state_seed", i, false)?;
        }
        let n = self.steps.len();
        let mut fbar = state_seeds[n];
        let mut xbar = F::State::default();
        let mut values = vec![0.0; self.value_count];
        let mut spot = vec![0.0; n];
        let mut orth = vec![0.0; F::FACTOR_COUNT * n];
        for i in (0..n).rev() {
            let c = self.steps[i];
            let exponent_bar = fbar * self.states[i + 1];
            let vbar = exponent_bar * (-0.5 * c.dt + c.dt.sqrt() * c.z / (2.0 * c.variance.sqrt()));
            let leverage_bar = vbar * c.multiplier_squared;
            c.lookup.transpose(leverage_bar, &mut values);
            let factor_bar = self.factor.pullback(
                self.transitions[i],
                xbar,
                vbar,
                c.variance,
                self.innovation_source,
            );
            spot[i] = exponent_bar * c.variance.sqrt() * c.dt.sqrt() + factor_bar.spot;
            for (j, &adjoint) in factor_bar.volatility.as_ref().iter().enumerate() {
                orth[j * n + i] = adjoint;
            }
            xbar = factor_bar.state;
            fbar = state_seeds[i]
                + fbar * self.states[i + 1] / self.states[i]
                + leverage_bar * c.lookup.derivative_log_f / self.states[i];
        }
        for (i, &v) in values
            .iter()
            .chain(spot.iter())
            .chain(orth.iter())
            .chain([fbar].iter())
            .enumerate()
        {
            valid(v, "path_adjoint", i, false)?;
        }
        if !F::finite(xbar) {
            return Err(LsvError::InvalidInput {
                field: "factor_adjoint",
                index: 0,
            });
        }
        Ok(LsvPathAdjoints {
            initial_f: fbar,
            initial_factor: xbar,
            squared_leverage: values.into_boxed_slice(),
            spot_shocks: spot.into_boxed_slice(),
            orthogonal_shocks: orth.into_boxed_slice(),
        })
    }
}
