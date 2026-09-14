//! Particle-calibrated one-factor Bergomi LSV in the continuous `f` coordinate.
//!
//! Calibration uses a compact quartic kernel in `log(f / f0)`, a versioned
//! variation of SSRN 1885032 (20)-(21). Time interpolation is left-constant;
//! spatial interpolation is linear with flat tails. Pricing paths are independent
//! of calibration particles. Reverse differentiation includes the conditional
//! expectation estimator, rather than freezing it under a Dupire variance bump.

use crate::market::MarketError;
use crate::mc::{LocalVolTimeGrid, Philox4x32, RandomCoordinate, RandomDomain};
use crate::models::{Bergomi1Factor, BergomiTransition};
use std::{error::Error, fmt};

pub const BERGOMI_LSV_SCHEME: &str = "bergomi-lsv-log-euler-exact-ou-v1";
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
        }
    }
}
impl Error for LsvError {}
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
        let row = self.times.partition_point(|t| *t <= time).saturating_sub(1);
        Ok(self.lookup_row(row, (f / self.initial_f).ln()))
    }
    pub(in crate::engine) fn lookup_row(&self, row: usize, x: f64) -> Lookup {
        let n = self.log_nodes.len();
        let i = self
            .log_nodes
            .partition_point(|v| *v <= x)
            .saturating_sub(1)
            .min(n - 2);
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
pub struct BergomiLsvPlan {
    pub(in crate::engine) factor: Bergomi1Factor,
    pub(in crate::engine) surface: LsvLeverageSurface,
    pub(in crate::engine) times: Box<[f64]>,
    pub(in crate::engine) transitions: Box<[BergomiTransition]>,
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
pub struct BergomiLsvPath {
    pub(in crate::engine) states: Box<[f64]>,
    pub(in crate::engine) factors: Box<[f64]>,
    pub(in crate::engine) steps: Box<[Step]>,
    pub(in crate::engine) transitions: Box<[BergomiTransition]>,
    pub(in crate::engine) factor: Bergomi1Factor,
    pub(in crate::engine) value_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LsvPathAdjoints {
    pub initial_f: f64,
    pub initial_factor: f64,
    pub squared_leverage: Box<[f64]>,
    pub spot_shocks: Box<[f64]>,
    pub orthogonal_shocks: Box<[f64]>,
}

impl BergomiLsvPlan {
    /// The execution grid may insert payoff and dividend observations, but must
    /// contain every leverage knot inside its horizon. It cannot cross a knot.
    pub fn new(
        factor: Bergomi1Factor,
        surface: LsvLeverageSurface,
        time_grid: &LocalVolTimeGrid,
    ) -> Result<Self, LsvError> {
        let times = time_grid.nodes();
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
        let transitions = times
            .windows(2)
            .map(|w| factor.transition(w[1] - w[0]).map_err(Into::into))
            .collect::<Result<Vec<_>, LsvError>>()?;
        Ok(Self {
            factor,
            surface,
            times: times.into(),
            transitions: transitions.into_boxed_slice(),
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
    pub const fn factor(&self) -> Bergomi1Factor {
        self.factor
    }

    /// Factor-major layout: first n spot normals, then n independent OU normals.
    /// Keeping spot dimensions first preserves the LV random coordinates at nu=0.
    pub fn pseudo_shocks(
        &self,
        seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, LsvError> {
        lsv_shocks(seed, path, self.transitions.len(), domain)
    }

    pub fn evolve_path(&self, initial_f: f64, shocks: &[f64]) -> Result<BergomiLsvPath, LsvError> {
        let n = self.transitions.len();
        length("shocks", 2 * n, shocks.len())?;
        for (i, &z) in shocks.iter().enumerate() {
            valid(z, "shock", i, false)?;
        }
        self.evolve_with_factor(
            initial_f,
            &shocks[..n],
            |i, x| self.transitions[i].evolve(x, shocks[i], shocks[n + i]),
            self.transitions.clone(),
        )
    }

    /// Internal joint-driver entry. The global sampler supplies OU innovations
    /// with their full cross-asset covariance, after its independent-factor bridge.
    /// Here shock adjoints refer to (spot normal, OU innovation), so the second
    /// loading is one and the spot-to-OU loading is zero. Target VJPs are unchanged.
    pub(in crate::engine) fn evolve_with_ou_innovations(
        &self,
        initial_f: f64,
        spot_shocks: &[f64],
        innovations: &[f64],
    ) -> Result<BergomiLsvPath, LsvError> {
        length("OU innovations", self.transitions.len(), innovations.len())?;
        for (i, &z) in innovations.iter().enumerate() {
            valid(z, "OU innovation", i, false)?;
        }
        let mut transitions = self.transitions.clone();
        for t in &mut transitions {
            t.spot_loading = 0.0;
            t.orthogonal_loading = 1.0;
        }
        self.evolve_with_factor(
            initial_f,
            spot_shocks,
            |i, x| self.transitions[i].decay * x + innovations[i],
            transitions,
        )
    }

    fn evolve_with_factor(
        &self,
        initial_f: f64,
        shocks: &[f64],
        next_factor: impl Fn(usize, f64) -> f64,
        reverse_transitions: Box<[BergomiTransition]>,
    ) -> Result<BergomiLsvPath, LsvError> {
        let n = self.transitions.len();
        length("spot shocks", n, shocks.len())?;
        valid(initial_f, "initial_f", 0, true)?;
        for (i, &z) in shocks.iter().enumerate() {
            valid(z, "spot shock", i, false)?;
        }
        let mut states = vec![initial_f];
        let mut factors = vec![0.0];
        let mut steps = Vec::with_capacity(n);
        for i in 0..n {
            let dt = self.times[i + 1] - self.times[i];
            let lookup = self.surface.lookup(self.times[i], states[i])?;
            let multiplier_squared = (2.0 * self.factor.vol_of_vol() * factors[i]).exp();
            let variance = lookup.value * multiplier_squared;
            let next = advance(states[i], variance, dt, shocks[i], i, 0)?;
            let x = next_factor(i, factors[i]);
            if !x.is_finite() {
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
            transitions: reverse_transitions,
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

pub(in crate::engine) fn lsv_shocks(
    seed: u64,
    path: u64,
    steps: usize,
    domain: RandomDomain,
) -> Result<Vec<f64>, LsvError> {
    let count = steps
        .checked_mul(2)
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

impl BergomiLsvPath {
    #[must_use]
    pub fn states(&self) -> &[f64] {
        &self.states
    }
    #[must_use]
    pub fn factors(&self) -> &[f64] {
        &self.factors
    }

    /// Price-only callers never allocate adjoint buffers. Seeds may be placed at
    /// any path observation; affine Spot seeds must first be multiplied by B(t).
    pub fn reverse(&self, state_seeds: &[f64]) -> Result<LsvPathAdjoints, LsvError> {
        length("state_seeds", self.states.len(), state_seeds.len())?;
        for (i, &s) in state_seeds.iter().enumerate() {
            valid(s, "state_seed", i, false)?;
        }
        let n = self.steps.len();
        let mut fbar = state_seeds[n];
        let mut xbar = 0.0;
        let mut values = vec![0.0; self.value_count];
        let mut spot = vec![0.0; n];
        let mut orth = vec![0.0; n];
        for i in (0..n).rev() {
            let c = self.steps[i];
            let exponent_bar = fbar * self.states[i + 1];
            let vbar = exponent_bar * (-0.5 * c.dt + c.dt.sqrt() * c.z / (2.0 * c.variance.sqrt()));
            let leverage_bar = vbar * c.multiplier_squared;
            c.lookup.transpose(leverage_bar, &mut values);
            spot[i] = exponent_bar * c.variance.sqrt() * c.dt.sqrt()
                + xbar * self.transitions[i].spot_loading;
            orth[i] = xbar * self.transitions[i].orthogonal_loading;
            xbar = xbar * self.transitions[i].decay
                + vbar * 2.0 * self.factor.vol_of_vol() * c.variance;
            fbar = state_seeds[i]
                + fbar * self.states[i + 1] / self.states[i]
                + leverage_bar * c.lookup.derivative_log_f / self.states[i];
        }
        for (i, &v) in values
            .iter()
            .chain(spot.iter())
            .chain(orth.iter())
            .chain([fbar, xbar].iter())
            .enumerate()
        {
            valid(v, "path_adjoint", i, false)?;
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
