//! Bergomi particle calibration and its discrete reverse.
//! Particle-calibrated one- or two-factor Bergomi LSV in the continuous `f` coordinate.
//!
//! Calibration uses a compact quartic kernel in `log(f / f0)`, a versioned
//! variation of SSRN 1885032 (20)-(21). Time interpolation is left-constant;
//! spatial interpolation is linear with flat tails. Pricing paths are independent
//! of calibration particles. Reverse differentiation includes the conditional
//! expectation estimator, rather than freezing it under a Dupire variance bump.

use crate::engine::processes::lsv::*;
use crate::market::LocalVarianceGrid;
use crate::mc::{
    DeterministicExecutor, LocalVolTimeGrid, Philox4x32, RandomCoordinate, RandomDomain,
};
use crate::models::{Bergomi1Factor, BergomiDynamics};
use pricing_numerics::NeumaierSum;
use rayon::prelude::*;

#[derive(Clone, Debug)]
struct TraceRow {
    states: Box<[f64]>,
    multipliers: Box<[f64]>,
    weight_sums: Box<[f64]>,
}

#[derive(Clone, Debug)]
pub struct CalibratedBergomiLsv<F: BergomiDynamics = Bergomi1Factor> {
    factor: F,
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
pub fn calibrate_bergomi_lsv<F: BergomiDynamics>(
    target: &LocalVarianceGrid,
    factor: F,
    initial_f: f64,
    config: LsvParticleConfig,
) -> Result<CalibratedBergomiLsv<F>, LsvError> {
    calibrate(target, factor, initial_f, config, None)
}

/// `calibrate_bergomi_lsv` with each time step's particle work spread over the
/// executor's pool. The time march stays sequential. Every particle update and
/// every node's kernel sum is computed exactly as in the sequential calibration
/// and reduced in the same order, so the result is bit-identical for any worker
/// count, and a failure reports the same particle.
pub fn calibrate_bergomi_lsv_parallel<F: BergomiDynamics>(
    target: &LocalVarianceGrid,
    factor: F,
    initial_f: f64,
    config: LsvParticleConfig,
    executor: &DeterministicExecutor,
) -> Result<CalibratedBergomiLsv<F>, LsvError> {
    calibrate(target, factor, initial_f, config, Some(executor))
}

/// One task's slice of particle states, factor states and leverage-cell hints.
type ParticleChunk<'a, F> = (
    usize,
    (
        (&'a mut [f64], &'a mut [<F as BergomiDynamics>::State]),
        &'a mut [usize],
    ),
);

/// Particles per parallel task. Fixed, so task boundaries never depend on the
/// worker count; errors are resolved in task order.
const PARTICLE_CHUNK: usize = 4096;

fn map_ordered<T: Sync, U: Send>(
    executor: Option<&DeterministicExecutor>,
    items: &[T],
    f: impl Fn(&T) -> U + Sync + Send,
) -> Vec<U> {
    match executor {
        None => items.iter().map(f).collect(),
        Some(executor) => executor.install(|| items.par_iter().map(f).collect()),
    }
}

fn calibrate<F: BergomiDynamics>(
    target: &LocalVarianceGrid,
    factor: F,
    initial_f: f64,
    config: LsvParticleConfig,
    executor: Option<&DeterministicExecutor>,
) -> Result<CalibratedBergomiLsv<F>, LsvError> {
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
        .checked_mul(1 + F::FACTOR_COUNT)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or(LsvError::InvalidInput {
            field: "random_dimension",
            index: nstep,
        })?;
    let _ = last_dimension;
    let mut surface =
        LsvLeverageSurface::new(times.to_vec(), nodes.to_vec(), vec![1.0; nt * m], initial_f)?;
    let mut states = vec![initial_f; np];
    let mut factors = vec![F::State::default(); np];
    // Each row is sorted starting from the previous row's order. The comparator
    // is a strict total order (log state, then particle index), so the sorted
    // result does not depend on the starting arrangement; particles move little
    // per step, so the nearly sorted input sorts much faster than a fresh one.
    let mut sorted = (0..np).map(|i| (0.0, i)).collect::<Vec<(f64, usize)>>();
    // Each particle's last leverage cell, a search hint for the next step.
    let mut cells = vec![0usize; np];
    let mut moments = Vec::with_capacity(nt * m);
    let mut diagnostics = Vec::with_capacity(nt);
    let mut trace = if config.retain_reverse_trace {
        Some(Vec::with_capacity(nt))
    } else {
        None
    };
    for r in 0..nt {
        let a = map_ordered(executor, &factors, |&x| factor.multiplier(x));
        for (i, &v) in a.iter().enumerate() {
            valid(v.powi(4), "particle_fourth_moment", i, true)?;
        }
        let key = |entry: &mut (f64, usize)| entry.0 = (states[entry.1] / initial_f).ln();
        let order = |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
        match executor {
            None => {
                sorted.iter_mut().for_each(key);
                sorted.sort_by(order);
            }
            Some(executor) => executor.install(|| {
                sorted.par_iter_mut().for_each(key);
                sorted.par_sort_by(order);
            }),
        }
        let node_moments = |(j, &x): (usize, &f64)| {
            if r == 0 || factor.vol_of_vol() == 0.0 {
                return (
                    LsvConditionalMoments {
                        second: 1.0,
                        third: 1.0,
                        fourth: 1.0,
                        effective_samples: np as f64,
                        source_node: j,
                        extrapolated: false,
                    },
                    np as f64,
                );
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
            (
                LsvConditionalMoments {
                    second: if w > 0.0 { m2 / w } else { 1.0 },
                    third: if w > 0.0 { m3 / w } else { 1.0 },
                    fourth: if w > 0.0 { m4 / w } else { 1.0 },
                    effective_samples: ess,
                    source_node: j,
                    extrapolated: false,
                },
                w,
            )
        };
        let indexed = nodes.iter().enumerate().collect::<Vec<_>>();
        let (mut row, weights): (Vec<_>, Vec<_>) =
            map_ordered(executor, &indexed, |&(j, x)| node_moments((j, x)))
                .into_iter()
                .unzip();
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
            let surface = &surface;
            let a = &a;
            let update = |i: usize,
                          state: &mut f64,
                          factor_state: &mut F::State,
                          cell: &mut usize|
             -> Result<(), LsvError> {
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
                let lookup = surface.lookup_row_from(r, (*state / initial_f).ln(), cell);
                *state = advance(*state, lookup.value * a[i] * a[i], dt, z, r, i)?;
                let z3 = if F::FACTOR_COUNT == 2 {
                    rng.standard_normal(RandomCoordinate::new(
                        i as u64,
                        (2 * nstep + r) as u32,
                        RandomDomain::LsvCalibration,
                    ))
                } else {
                    0.0
                };
                *factor_state = factor.evolve(transition, *factor_state, z, [z2, z3]);
                if !F::finite(*factor_state) {
                    return Err(LsvError::NonFiniteState {
                        time_index: r + 1,
                        path: i,
                    });
                }
                Ok(())
            };
            let chunk = |(c, ((s, f), h)): ParticleChunk<'_, F>| {
                for k in 0..s.len() {
                    update(c * PARTICLE_CHUNK + k, &mut s[k], &mut f[k], &mut h[k])?;
                }
                Ok::<(), LsvError>(())
            };
            let results = match executor {
                None => states
                    .chunks_mut(PARTICLE_CHUNK)
                    .zip(factors.chunks_mut(PARTICLE_CHUNK))
                    .zip(cells.chunks_mut(PARTICLE_CHUNK))
                    .enumerate()
                    .map(chunk)
                    .collect::<Vec<_>>(),
                Some(executor) => executor.install(|| {
                    states
                        .par_chunks_mut(PARTICLE_CHUNK)
                        .zip(factors.par_chunks_mut(PARTICLE_CHUNK))
                        .zip(cells.par_chunks_mut(PARTICLE_CHUNK))
                        .enumerate()
                        .map(chunk)
                        .collect::<Vec<_>>()
                }),
            };
            for result in results {
                result?;
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

impl<F: BergomiDynamics> CalibratedBergomiLsv<F> {
    #[must_use]
    pub fn surface(&self) -> &LsvLeverageSurface {
        &self.surface
    }
    #[must_use]
    pub fn target(&self) -> &LocalVarianceGrid {
        &self.target
    }
    #[must_use]
    pub const fn factor(&self) -> F {
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

    pub fn pricing_plan(
        &self,
        time_grid: &LocalVolTimeGrid,
    ) -> Result<BergomiLsvPlan<F>, LsvError> {
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
