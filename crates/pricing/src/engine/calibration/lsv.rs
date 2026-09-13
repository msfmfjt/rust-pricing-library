//! Bergomi particle calibration and its discrete reverse.
//! Particle-calibrated one-factor Bergomi LSV in the continuous `f` coordinate.
//!
//! Calibration uses a compact quartic kernel in `log(f / f0)`, a versioned
//! variation of SSRN 1885032 (20)-(21). Time interpolation is left-constant;
//! spatial interpolation is linear with flat tails. Pricing paths are independent
//! of calibration particles. Reverse differentiation includes the conditional
//! expectation estimator, rather than freezing it under a Dupire variance bump.

use crate::engine::processes::lsv::*;
use crate::market::LocalVarianceGrid;
use crate::mc::{LocalVolTimeGrid, Philox4x32, RandomCoordinate, RandomDomain};
use crate::models::Bergomi1Factor;
use pricing_numerics::NeumaierSum;

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
