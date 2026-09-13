//! Particle-calibrated one-factor Bergomi LSV in the continuous `f` coordinate.
//!
//! Calibration uses a compact quartic kernel in `log(f / f0)`, a versioned
//! variation of SSRN 1885032 (20)-(21). Time interpolation is left-constant;
//! spatial interpolation is linear with flat tails. Pricing paths are independent
//! of calibration particles. Reverse differentiation includes the conditional
//! expectation estimator, rather than freezing it under a Dupire variance bump.

use crate::{LocalVolTimeGrid, Philox4x32, RandomCoordinate, RandomDomain};
use pricing_market::{LocalVarianceGrid, MarketError};
use pricing_models::{Bergomi1Factor, BergomiTransition};
use pricing_numerics::NeumaierSum;
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
    Core(pricing_core::CoreError),
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
impl From<pricing_core::CoreError> for LsvError {
    fn from(e: pricing_core::CoreError) -> Self {
        Self::Core(e)
    }
}

fn valid(value: f64, field: &'static str, index: usize, positive: bool) -> Result<(), LsvError> {
    if !value.is_finite() || (positive && value <= 0.0) {
        return Err(LsvError::InvalidInput { field, index });
    }
    Ok(())
}
fn length(field: &'static str, expected: usize, actual: usize) -> Result<(), LsvError> {
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
    particle_count: usize,
    seed: u64,
    log_bandwidth: f64,
    minimum_effective_samples: f64,
    retain_reverse_trace: bool,
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
    times: Box<[f64]>,
    log_nodes: Box<[f64]>,
    values: Box<[f64]>,
    initial_f: f64,
}

#[derive(Clone, Copy, Debug)]
struct Lookup {
    value: f64,
    derivative_log_f: f64,
    left: usize,
    weight: f64,
}
impl Lookup {
    fn transpose(self, seed: f64, output: &mut [f64]) {
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

    fn lookup(&self, time: f64, f: f64) -> Result<Lookup, LsvError> {
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
    fn lookup_row(&self, row: usize, x: f64) -> Lookup {
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
    factor: Bergomi1Factor,
    surface: LsvLeverageSurface,
    times: Box<[f64]>,
    transitions: Box<[BergomiTransition]>,
}

#[derive(Clone, Copy, Debug)]
struct Step {
    lookup: Lookup,
    multiplier_squared: f64,
    variance: f64,
    z: f64,
    dt: f64,
}

#[derive(Clone, Debug)]
pub struct BergomiLsvPath {
    states: Box<[f64]>,
    factors: Box<[f64]>,
    steps: Box<[Step]>,
    transitions: Box<[BergomiTransition]>,
    factor: Bergomi1Factor,
    value_count: usize,
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
        valid(initial_f, "initial_f", 0, true)?;
        for (i, &z) in shocks.iter().enumerate() {
            valid(z, "shock", i, false)?;
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
            let x = self.transitions[i].evolve(factors[i], shocks[i], shocks[n + i]);
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
            transitions: self.transitions.clone(),
            factor: self.factor,
            value_count: self.surface.values.len(),
        })
    }
}

fn advance(
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

fn lsv_shocks(
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

#[derive(Clone, Debug)]
struct TraceRow {
    states: Box<[f64]>,
    multipliers: Box<[f64]>,
    weight_sums: Box<[f64]>,
}

#[derive(Clone, Debug)]
pub struct CalibratedBergomiLsv {
    factor: Bergomi1Factor,
    config: LsvParticleConfig,
    target: LocalVarianceGrid,
    surface: LsvLeverageSurface,
    moments: Box<[LsvConditionalMoments]>,
    diagnostics: Box<[LsvCalibrationRowDiagnostics]>,
    trace: Option<Box<[TraceRow]>>,
}

// The normalization 15/16 cancels in the regression and its derivatives.
fn quartic(u: f64) -> f64 {
    if u.abs() >= 1.0 {
        0.0
    } else {
        (1.0 - u * u).powi(2)
    }
}
fn quartic_derivative(u: f64) -> f64 {
    if u.abs() >= 1.0 {
        0.0
    } else {
        -4.0 * u * (1.0 - u * u)
    }
}

/// Sequential time march; the immutable calibrated object can subsequently be
/// shared by Rayon pricing workers. Fixed particle identity determines reductions.
pub fn calibrate_bergomi_lsv(
    target: &LocalVarianceGrid,
    factor: Bergomi1Factor,
    initial_f: f64,
    config: LsvParticleConfig,
) -> Result<CalibratedBergomiLsv, LsvError> {
    valid(initial_f, "initial_f", 0, true)?;
    if target.time_nodes()[0].to_bits() != 0.0f64.to_bits() {
        return Err(LsvError::InvalidInput {
            field: "initial_calibration_time",
            index: 0,
        });
    }
    let times = target.time_nodes();
    let nodes = target.log_moneyness_nodes();
    let m = nodes.len();
    let nt = times.len();
    let np = config.particle_count;
    let nstep = nt - 1;
    let rng = Philox4x32::from_seed(config.seed);
    let last_dimension = nstep
        .checked_mul(2)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or(LsvError::InvalidInput {
            field: "random_dimension",
            index: nstep,
        })?;
    let _ = last_dimension;
    let mut surface =
        LsvLeverageSurface::new(times.to_vec(), nodes.to_vec(), vec![1.0; nt * m], initial_f)?;
    let mut states = vec![initial_f; np];
    let mut factors = vec![0.0; np];
    let mut moments = Vec::with_capacity(nt * m);
    let mut diagnostics = Vec::with_capacity(nt);
    let mut trace = if config.retain_reverse_trace {
        Some(Vec::with_capacity(nt))
    } else {
        None
    };
    for r in 0..nt {
        let a = factors
            .iter()
            .map(|x| (factor.vol_of_vol() * x).exp())
            .collect::<Vec<_>>();
        for (i, &v) in a.iter().enumerate() {
            valid(v.powi(4), "particle_fourth_moment", i, true)?;
        }
        let mut sorted = (0..np)
            .map(|i| ((states[i] / initial_f).ln(), i))
            .collect::<Vec<_>>();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut row = Vec::with_capacity(m);
        let mut weights = Vec::with_capacity(m);
        for (j, &x) in nodes.iter().enumerate() {
            if r == 0 || factor.vol_of_vol() == 0.0 {
                row.push(LsvConditionalMoments {
                    second: 1.0,
                    third: 1.0,
                    fourth: 1.0,
                    effective_samples: np as f64,
                    source_node: j,
                    extrapolated: false,
                });
                weights.push(np as f64);
                continue;
            }
            let h = config.log_bandwidth;
            let lo = sorted.partition_point(|p| p.0 < x - h);
            let hi = sorted.partition_point(|p| p.0 <= x + h);
            let mut sums = [NeumaierSum::new(); 5];
            for &(xp, i) in &sorted[lo..hi] {
                let w = quartic((xp - x) / h);
                for (s, v) in sums.iter_mut().zip([
                    w,
                    w * w,
                    w * a[i].powi(2),
                    w * a[i].powi(3),
                    w * a[i].powi(4),
                ]) {
                    s.add(v);
                }
            }
            let [w, w2, m2, m3, m4] = sums.map(NeumaierSum::total);
            let ess = if w2 > 0.0 { w * w / w2 } else { 0.0 };
            row.push(LsvConditionalMoments {
                second: if w > 0.0 { m2 / w } else { 1.0 },
                third: if w > 0.0 { m3 / w } else { 1.0 },
                fourth: if w > 0.0 { m4 / w } else { 1.0 },
                effective_samples: ess,
                source_node: j,
                extrapolated: false,
            });
            weights.push(w);
        }
        let supported = (0..m)
            .filter(|&j| row[j].effective_samples >= config.minimum_effective_samples)
            .collect::<Vec<_>>();
        if supported.is_empty() {
            return Err(LsvError::NoSupportedCalibrationNode { time_index: r });
        }
        let mut extrapolated = 0;
        for j in 0..m {
            if row[j].effective_samples < config.minimum_effective_samples {
                let source = *supported
                    .iter()
                    .min_by(|&&a, &&b| {
                        (nodes[a] - nodes[j])
                            .abs()
                            .total_cmp(&(nodes[b] - nodes[j]).abs())
                            .then(a.cmp(&b))
                    })
                    .expect("nonempty supported nodes");
                let original_ess = row[j].effective_samples;
                row[j] = LsvConditionalMoments {
                    effective_samples: original_ess,
                    source_node: source,
                    extrapolated: true,
                    ..row[source]
                };
                extrapolated += 1;
            }
            let v = target.interpolate(times[r], nodes[j])?.value / row[j].second;
            valid(v, "calibrated_squared_leverage", r * m + j, true)?;
            surface.values[r * m + j] = v;
        }
        diagnostics.push(LsvCalibrationRowDiagnostics {
            time: times[r],
            minimum_effective_samples: supported
                .iter()
                .map(|&j| row[j].effective_samples)
                .fold(f64::INFINITY, f64::min),
            extrapolated_nodes: extrapolated,
            particle_mean_f: states.iter().copied().collect::<NeumaierSum>().total() / np as f64,
        });
        if let Some(trace) = &mut trace {
            trace.push(TraceRow {
                states: states.clone().into_boxed_slice(),
                multipliers: a.clone().into_boxed_slice(),
                weight_sums: weights.into_boxed_slice(),
            });
        }
        moments.extend(row);
        if r < nstep {
            let dt = times[r + 1] - times[r];
            let transition = factor.transition(dt)?;
            for i in 0..np {
                let z = rng.standard_normal(RandomCoordinate::new(
                    i as u64,
                    r as u32,
                    RandomDomain::LsvCalibration,
                ));
                let z2 = rng.standard_normal(RandomCoordinate::new(
                    i as u64,
                    (nstep + r) as u32,
                    RandomDomain::LsvCalibration,
                ));
                let lookup = surface.lookup_row(r, (states[i] / initial_f).ln());
                states[i] = advance(states[i], lookup.value * a[i] * a[i], dt, z, r, i)?;
                factors[i] = transition.evolve(factors[i], z, z2);
                if !factors[i].is_finite() {
                    return Err(LsvError::NonFiniteState {
                        time_index: r + 1,
                        path: i,
                    });
                }
            }
        }
    }
    Ok(CalibratedBergomiLsv {
        factor,
        config,
        target: target.clone(),
        surface,
        moments: moments.into_boxed_slice(),
        diagnostics: diagnostics.into_boxed_slice(),
        trace: trace.map(Vec::into_boxed_slice),
    })
}

impl CalibratedBergomiLsv {
    #[must_use]
    pub fn surface(&self) -> &LsvLeverageSurface {
        &self.surface
    }
    #[must_use]
    pub fn target(&self) -> &LocalVarianceGrid {
        &self.target
    }
    #[must_use]
    pub const fn factor(&self) -> Bergomi1Factor {
        self.factor
    }
    #[must_use]
    pub fn config(&self) -> &LsvParticleConfig {
        &self.config
    }
    #[must_use]
    pub fn conditional_moments(&self) -> &[LsvConditionalMoments] {
        &self.moments
    }
    #[must_use]
    pub fn diagnostics(&self) -> &[LsvCalibrationRowDiagnostics] {
        &self.diagnostics
    }

    pub fn pricing_plan(&self, time_grid: &LocalVolTimeGrid) -> Result<BergomiLsvPlan, LsvError> {
        BergomiLsvPlan::new(self.factor, self.surface.clone(), time_grid)
    }

    /// VJP from squared relative leverage to the original target Local variance
    /// grid. Includes the motion of every calibration particle in the kernel
    /// quotient. SV parameters, axes, bandwidth, seed and fallback branches are
    /// held fixed. This is an exact derivative of the finite discrete algorithm;
    /// it is not the infinite-particle continuum result of SSRN 4304114 (4.3).
    pub fn reverse_leverage(&self, leverage_adjoints: &[f64]) -> Result<Vec<f64>, LsvError> {
        length(
            "leverage_adjoints",
            self.surface.values.len(),
            leverage_adjoints.len(),
        )?;
        for (i, &v) in leverage_adjoints.iter().enumerate() {
            valid(v, "leverage_adjoint", i, false)?;
        }
        let trace = self
            .trace
            .as_ref()
            .ok_or(LsvError::ReverseTraceNotRetained)?;
        let nt = self.surface.times.len();
        let m = self.surface.log_nodes.len();
        let np = self.config.particle_count;
        let rng = Philox4x32::from_seed(self.config.seed);
        let mut lbar = leverage_adjoints.to_vec();
        let mut target_bar = vec![0.0; self.target.values().len()];
        let mut state_bar = vec![0.0; np];
        for r in (0..nt).rev() {
            if r + 1 < nt {
                let dt = self.surface.times[r + 1] - self.surface.times[r];
                for (i, bar) in state_bar.iter_mut().enumerate() {
                    let f = trace[r].states[i];
                    let next = trace[r + 1].states[i];
                    let lookup = self
                        .surface
                        .lookup_row(r, (f / self.surface.initial_f).ln());
                    let a2 = trace[r].multipliers[i].powi(2);
                    let q = lookup.value * a2;
                    let z = rng.standard_normal(RandomCoordinate::new(
                        i as u64,
                        r as u32,
                        RandomDomain::LsvCalibration,
                    ));
                    let lb = *bar * next * (-0.5 * dt + dt.sqrt() * z / (2.0 * q.sqrt())) * a2;
                    lookup.transpose(lb, &mut lbar);
                    *bar = *bar * next / f + lb * lookup.derivative_log_f / f;
                }
            }
            let mut moment_bar = vec![0.0; m];
            for j in 0..m {
                let moment = self.moments[r * m + j];
                let interp = self
                    .target
                    .interpolate(self.surface.times[r], self.surface.log_nodes[j])?;
                interp.transpose_accumulate(lbar[r * m + j] / moment.second, &mut target_bar, m);
                moment_bar[moment.source_node] -=
                    lbar[r * m + j] * interp.value / moment.second.powi(2);
            }
            if r == 0 || self.factor.vol_of_vol() == 0.0 {
                continue;
            }
            let h = self.config.log_bandwidth;
            for (j, mb) in moment_bar
                .into_iter()
                .enumerate()
                .filter(|(_, v)| *v != 0.0)
            {
                let sum_w = trace[r].weight_sums[j];
                let m2 = self.moments[r * m + j].second;
                for (i, bar) in state_bar.iter_mut().enumerate() {
                    let f = trace[r].states[i];
                    let u = ((f / self.surface.initial_f).ln() - self.surface.log_nodes[j]) / h;
                    let dw_df = quartic_derivative(u) / (h * f);
                    *bar += mb * dw_df * (trace[r].multipliers[i].powi(2) - m2) / sum_w;
                }
            }
        }
        for (i, &v) in target_bar.iter().enumerate() {
            valid(v, "calibrated_target_adjoint", i, false)?;
        }
        Ok(target_bar)
    }
}
