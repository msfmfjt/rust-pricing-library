//! Step-local transition Vega statistics and positive-time Gamma recovery.
//! Direct payoff variance risk is audited separately; its Gamma mapping is unresolved.

use std::sync::Arc;

use crate::MonteCarloError;
use crate::core::FiniteF64;
use crate::market::{ImpliedVarianceSurface, LocalVarianceGrid};
use crate::mc::{
    DeterministicExecutor, DeterministicStatistics, LocalVolLogEulerPlan, LocalVolPath,
};
use crate::risk::{
    AnalyticCallDensityRow, Equation11GammaRow, RiskConfigError, VegaKtDirectVarianceRowEstimate,
    VegaKtStepEstimate, VegaKtStepLayout, VegaKtStepMoment, VegaKtStepRecoveryPlan,
    VegaKtStepRowEstimate, VegaKtStepSample, VegaKtStepSamplingUnit,
    analytic_call_density_row_from_surface, recover_equation11_gamma_row,
};
use pricing_numerics::NeumaierSum;

impl VegaKtStepRecoveryPlan {
    /// Compile density and effective variance at every positive simulation
    /// source time, including inserted events, in the plan's forward coordinate.
    /// The supplied surface must describe that coordinate and the model being
    /// simulated; this constructor does not certify its calibration consistency.
    pub fn new(
        simulation: LocalVolLogEulerPlan,
        grid: LocalVarianceGrid,
        surface: &dyn ImpliedVarianceSurface,
        density_relative_threshold: f64,
    ) -> Result<Self, RiskConfigError> {
        let times = simulation.time_grid().nodes();
        if grid.time_nodes()[0] != 0.0 || grid.time_nodes().last() < times.last() {
            return Err(RiskConfigError::VegaKtStepLayoutMismatch {
                quantity: "variance-grid time coverage",
            });
        }
        let width = grid.log_moneyness_nodes().len();
        let steps = simulation.time_grid().step_count();
        let node_count = steps
            .checked_mul(width)
            .ok_or(RiskConfigError::VegaKtStepShapeOverflow)?;
        let direct_offset = steps
            .checked_mul(3)
            .and_then(|n| n.checked_add(node_count))
            .and_then(|n| n.checked_add(1))
            .ok_or(RiskConfigError::VegaKtStepShapeOverflow)?;
        let direct_totals_offset = times
            .len()
            .checked_mul(width)
            .and_then(|n| n.checked_add(direct_offset))
            .ok_or(RiskConfigError::VegaKtStepShapeOverflow)?;
        let combined_total_index = direct_totals_offset
            .checked_add(times.len())
            .ok_or(RiskConfigError::VegaKtStepShapeOverflow)?;
        let component_count = combined_total_index
            .checked_add(2)
            .ok_or(RiskConfigError::VegaKtStepShapeOverflow)?;
        let mut densities = Vec::with_capacity(steps.saturating_sub(1));
        let mut variances = Vec::with_capacity(node_count);
        for (i, &time) in times[..steps].iter().enumerate() {
            if i > 0 && time + (times[i + 1] - time) != times[i + 1] {
                return Err(RiskConfigError::VegaKtStepLayoutMismatch {
                    quantity: "non-reconstructible interval endpoint",
                });
            }
            for &x in grid.log_moneyness_nodes() {
                variances.push(grid.interpolate(time, x)?.value);
            }
            if i != 0 {
                densities.push(analytic_call_density_row_from_surface(
                    surface,
                    time,
                    simulation.forward_normalizers()[i],
                    grid.log_moneyness_nodes().to_vec(),
                    density_relative_threshold,
                )?);
            }
        }
        // Validate the policy even for a one-step plan with no positive rows.
        crate::risk::active_density_domain(
            grid.log_moneyness_nodes(),
            &vec![1.0; width],
            density_relative_threshold,
        )?;
        Ok(Self {
            layout: Arc::new(VegaKtStepLayout {
                simulation,
                grid,
                densities: densities.into_boxed_slice(),
                variances: variances.into_boxed_slice(),
                node_count,
                direct_offset,
                direct_totals_offset,
                combined_total_index,
                component_count,
            }),
        })
    }

    #[must_use]
    pub fn simulation_plan(&self) -> &LocalVolLogEulerPlan {
        &self.layout.simulation
    }

    #[must_use]
    pub fn variance_grid(&self) -> &LocalVarianceGrid {
        &self.layout.grid
    }

    /// One row per positive source time; the time-zero row is deliberately absent.
    #[must_use]
    pub fn positive_density_rows(&self) -> &[AnalyticCallDensityRow] {
        &self.layout.densities
    }

    /// Build a sample from a path and payoff state seeds. Verifies the cached
    /// transitions against this compiled simulation/grid before reversing them.
    /// Explicit payoff/bridge variance seeds are outside this transition-only API.
    pub fn sample_from_path(
        &self,
        path: &LocalVolPath,
        state_seeds: &[f64],
    ) -> Result<VegaKtStepSample, MonteCarloError> {
        self.validate_path(path)?;
        let width = self.layout.grid.log_moneyness_nodes().len();
        let reverse =
            path.reverse_state_adjoints(state_seeds, self.layout.grid.values().len(), width)?;
        let mut values = vec![0.0; self.layout.component_count];
        for (i, step) in reverse.step_adjoints().iter().enumerate() {
            step.accumulate_local_volatility_nodes(&mut values[i * width..(i + 1) * width])?;
        }
        self.fill_totals(&mut values)?;
        Ok(VegaKtStepSample {
            layout: Arc::clone(&self.layout),
            values: values.into_boxed_slice(),
            sampling_unit: VegaKtStepSamplingUnit::IndependentPath,
        })
    }

    /// State seeds hold effective variances fixed; direct seeds have N+1 entries
    /// at actual nodes. Adds variance/state feedback exactly once. Direct rows
    /// (including the terminal node) remain distinct from transition rows.
    /// Any nonzero direct seed disables Gamma recovery for the resulting estimate,
    /// even when signed samples cancel. Their equation-(11) mapping is unresolved.
    pub fn sample_from_path_with_variance_seeds(
        &self,
        path: &LocalVolPath,
        state_seeds: &[f64],
        variance_seeds: &[f64],
    ) -> Result<VegaKtStepSample, MonteCarloError> {
        self.validate_path(path)?;
        let reverse = self.layout.simulation.reverse_path_with_variance_seeds(
            &self.layout.grid,
            path,
            state_seeds,
            variance_seeds,
        )?;
        let width = self.layout.grid.log_moneyness_nodes().len();
        let mut values = vec![0.0; self.layout.component_count];
        for (i, step) in reverse.transition().step_adjoints().iter().enumerate() {
            step.accumulate_local_volatility_nodes(&mut values[i * width..(i + 1) * width])?;
        }
        for (i, node) in reverse.direct_variance_adjoints().iter().enumerate() {
            let base = self.layout.direct_offset + i * width;
            node.accumulate_local_volatility_nodes(&mut values[base..base + width])?;
        }
        values[self.layout.component_count - 1] = if variance_seeds.iter().any(|v| *v != 0.0) {
            1.0
        } else {
            0.0
        };
        self.fill_totals(&mut values)?;
        Ok(VegaKtStepSample {
            layout: Arc::clone(&self.layout),
            values: values.into_boxed_slice(),
            sampling_unit: VegaKtStepSamplingUnit::IndependentPath,
        })
    }

    fn validate_path(&self, path: &LocalVolPath) -> Result<(), MonteCarloError> {
        let simulation = &self.layout.simulation;
        let times = simulation.time_grid().nodes();
        if path.step_cache().len() + 1 != times.len() {
            return Err(RiskConfigError::VegaKtStepLayoutMismatch {
                quantity: "simulation interval count",
            }
            .into());
        }
        if path.states()[0] != simulation.forward_normalizers()[0] {
            return Err(RiskConfigError::VegaKtStepLayoutMismatch {
                quantity: "initial state versus density forward",
            }
            .into());
        }
        simulation.validate_cached_path(&self.layout.grid, path)?;
        Ok(())
    }

    fn fill_totals(&self, values: &mut [f64]) -> Result<(), RiskConfigError> {
        let width = self.layout.grid.log_moneyness_nodes().len();
        let mut total = NeumaierSum::new();
        for i in 0..self.layout.simulation.time_grid().step_count() {
            let nodes = &values[i * width..(i + 1) * width];
            let row_total = nodes.iter().copied().collect::<NeumaierSum>().total();
            let (retained, excluded) = if i == 0 {
                (0.0, 0.0)
            } else {
                let active = self.layout.densities[i - 1].active_domain();
                let mut kept = NeumaierSum::new();
                let mut dropped = NeumaierSum::new();
                for (j, &value) in nodes.iter().enumerate() {
                    if active.contains_index(j) {
                        kept.add(value);
                    } else {
                        dropped.add(value);
                    }
                }
                (kept.total(), dropped.total())
            };
            let base = self.layout.node_count + 3 * i;
            for (j, value) in [row_total, retained, excluded].into_iter().enumerate() {
                values[base + j] = finite(value, "vega_kt_step_row_sum")?;
            }
            total.add(row_total);
        }
        values[self.layout.direct_offset - 1] = finite(total.total(), "vega_kt_all_step_sum")?;
        for i in 0..self.layout.simulation.time_grid().nodes().len() {
            let base = self.layout.direct_offset + i * width;
            let row_total = values[base..base + width]
                .iter()
                .copied()
                .collect::<NeumaierSum>()
                .total();
            values[self.layout.direct_totals_offset + i] =
                finite(row_total, "vega_kt_direct_row_sum")?;
            total.add(row_total);
        }
        values[self.layout.combined_total_index] = finite(total.total(), "vega_kt_combined_sum")?;
        Ok(())
    }

    /// Fixed-block compensated sums and balanced-tree centered moments over
    /// independent pseudo-MC paths or antithetic pair means. RQMC points are not
    /// independent observations and must not be passed as independent paths.
    /// Row/global totals are reduced as separate per-sample channels, retaining
    /// within-row and across-time covariances in their standard errors.
    pub fn estimate<F>(
        &self,
        executor: &DeterministicExecutor,
        independent_sampling_units: u64,
        sampling_unit: VegaKtStepSamplingUnit,
        evaluate: F,
    ) -> Result<VegaKtStepEstimate, MonteCarloError>
    where
        F: Fn(u64) -> Result<VegaKtStepSample, MonteCarloError> + Send + Sync,
    {
        if independent_sampling_units == 0 {
            return Err(RiskConfigError::VegaKtStepNoSamplingUnits.into());
        }
        let stats = executor.try_map_reduce_statistics_vector(
            independent_sampling_units,
            self.layout.component_count,
            |unit, output| {
                let sample = evaluate(unit)?;
                if !Arc::ptr_eq(&self.layout, &sample.layout) {
                    return Err(MonteCarloError::from(
                        RiskConfigError::VegaKtStepLayoutMismatch {
                            quantity: "sample origin",
                        },
                    ));
                }
                if sample.sampling_unit != sampling_unit {
                    return Err(RiskConfigError::VegaKtStepSamplingUnitMismatch.into());
                }
                output.copy_from_slice(&sample.values);
                Ok(())
            },
        )?;
        self.finish(&stats, independent_sampling_units, sampling_unit)
            .map_err(Into::into)
    }

    fn finish(
        &self,
        stats: &[DeterministicStatistics],
        count: u64,
        sampling_unit: VegaKtStepSamplingUnit,
    ) -> Result<VegaKtStepEstimate, RiskConfigError> {
        let moments: Vec<_> = stats
            .iter()
            .copied()
            .map(|stat| moment(stat, count))
            .collect::<Result<_, _>>()?;
        let times = self.layout.simulation.time_grid().nodes();
        let width = self.layout.grid.log_moneyness_nodes().len();
        let has_unrecovered_direct_variance = moments[self.layout.component_count - 1].mean > 0.0;
        let mut rows = Vec::with_capacity(times.len() - 1);
        for i in 0..times.len() - 1 {
            let nodes = &moments[i * width..(i + 1) * width];
            let (gamma, gamma_standard_errors) = if i == 0 || has_unrecovered_direct_variance {
                (None, None)
            } else {
                let density = &self.layout.densities[i - 1];
                let variances = &self.layout.variances[i * width..(i + 1) * width];
                let dt = times[i + 1] - times[i];
                let means: Vec<_> = nodes.iter().map(|m| m.mean).collect();
                let gamma = recover_equation11_gamma_row(density, &means, variances, dt)?;
                let standard_errors = if count > 1 {
                    let errors: Vec<_> = nodes
                        .iter()
                        .map(|m| m.standard_error.expect("multiple units"))
                        .collect();
                    Some(
                        recover_equation11_gamma_row(density, &errors, variances, dt)?
                            .gamma_values()
                            .to_vec()
                            .into_boxed_slice(),
                    )
                } else {
                    None
                };
                (Some(gamma), standard_errors)
            };
            let base = self.layout.node_count + 3 * i;
            rows.push(VegaKtStepRowEstimate {
                source_maturity: times[i],
                recovered_maturity: times[i + 1],
                forward: self.layout.simulation.forward_normalizers()[i],
                nodes: nodes.to_vec().into_boxed_slice(),
                total: moments[base],
                retained: (i > 0).then_some(moments[base + 1]),
                excluded: (i > 0).then_some(moments[base + 2]),
                gamma,
                gamma_standard_errors,
            });
        }
        Ok(VegaKtStepEstimate {
            independent_sampling_units: count,
            sampling_unit,
            log_moneyness_nodes: self.layout.grid.log_moneyness_nodes().into(),
            rows: rows.into_boxed_slice(),
            total: moments[self.layout.direct_offset - 1],
            direct_rows: times
                .iter()
                .enumerate()
                .map(|(i, &maturity)| {
                    let base = self.layout.direct_offset + i * width;
                    VegaKtDirectVarianceRowEstimate {
                        maturity,
                        nodes: moments[base..base + width].to_vec().into_boxed_slice(),
                        total: moments[self.layout.direct_totals_offset + i],
                    }
                })
                .collect(),
            combined_total: moments[self.layout.combined_total_index],
            has_unrecovered_direct_variance,
        })
    }
}

impl VegaKtStepSample {
    /// Average a primary path and its antithetic mate before moment accumulation.
    /// The caller is responsible for using opposite shocks for the same draw.
    pub fn antithetic_pair(primary: Self, mate: Self) -> Result<Self, RiskConfigError> {
        if !Arc::ptr_eq(&primary.layout, &mate.layout) {
            return Err(RiskConfigError::VegaKtStepLayoutMismatch {
                quantity: "antithetic sample origin",
            });
        }
        if primary.sampling_unit != VegaKtStepSamplingUnit::IndependentPath
            || mate.sampling_unit != VegaKtStepSamplingUnit::IndependentPath
        {
            return Err(RiskConfigError::VegaKtStepSamplingUnitMismatch);
        }
        let mut values = primary.values;
        for (value, mate) in values.iter_mut().zip(mate.values.iter()) {
            *value = finite(0.5 * *value + 0.5 * mate, "vega_kt_step_pair_mean")?;
        }
        Ok(Self {
            layout: primary.layout,
            values,
            sampling_unit: VegaKtStepSamplingUnit::AntitheticPair,
        })
    }

    /// Integrated node Vega, row-major [actual simulation interval][spatial node].
    #[must_use]
    pub fn integrated_node_adjoints(&self) -> &[f64] {
        &self.values[..self.layout.node_count]
    }
}

fn finite(value: f64, label: &'static str) -> Result<f64, RiskConfigError> {
    Ok(FiniteF64::new(value, label)?.get())
}

pub(crate) fn moment(
    stat: DeterministicStatistics,
    count: u64,
) -> Result<VegaKtStepMoment, RiskConfigError> {
    let mean = finite(stat.sum().total() / count as f64, "vega_kt_step_mean")?;
    let sample_variance = stat
        .moments()
        .sample_variance()
        .map(|v| finite(v, "vega_kt_step_sample_variance"))
        .transpose()?;
    let standard_error = sample_variance
        .map(|v| {
            finite(
                v.sqrt() / (count as f64).sqrt(),
                "vega_kt_step_standard_error",
            )
        })
        .transpose()?;
    Ok(VegaKtStepMoment {
        mean,
        sample_variance,
        standard_error,
    })
}

impl VegaKtStepMoment {
    #[must_use]
    pub const fn mean(self) -> f64 {
        self.mean
    }
    #[must_use]
    pub const fn sample_variance(self) -> Option<f64> {
        self.sample_variance
    }
    #[must_use]
    pub const fn standard_error(self) -> Option<f64> {
        self.standard_error
    }
}
impl VegaKtStepEstimate {
    #[must_use]
    pub const fn independent_sampling_units(&self) -> u64 {
        self.independent_sampling_units
    }
    #[must_use]
    pub const fn sampling_unit(&self) -> VegaKtStepSamplingUnit {
        self.sampling_unit
    }
    #[must_use]
    pub fn log_moneyness_nodes(&self) -> &[f64] {
        &self.log_moneyness_nodes
    }
    #[must_use]
    pub fn rows(&self) -> &[VegaKtStepRowEstimate] {
        &self.rows
    }
    /// Transition contribution including direct-seed state feedback, excluding
    /// explicit direct variance deposition. Not reporting-IV Vega.
    #[must_use]
    pub const fn total(&self) -> VegaKtStepMoment {
        self.total
    }
}
impl VegaKtStepRowEstimate {
    #[must_use]
    pub const fn source_maturity(&self) -> f64 {
        self.source_maturity
    }
    #[must_use]
    pub const fn recovered_maturity(&self) -> f64 {
        self.recovered_maturity
    }
    #[must_use]
    pub const fn forward(&self) -> f64 {
        self.forward
    }
    #[must_use]
    pub fn nodes(&self) -> &[VegaKtStepMoment] {
        &self.nodes
    }
    #[must_use]
    pub const fn total(&self) -> VegaKtStepMoment {
        self.total
    }
    #[must_use]
    pub const fn retained(&self) -> Option<VegaKtStepMoment> {
        self.retained
    }
    #[must_use]
    pub const fn excluded(&self) -> Option<VegaKtStepMoment> {
        self.excluded
    }
    #[must_use]
    pub const fn gamma(&self) -> Option<&Equation11GammaRow> {
        self.gamma.as_ref()
    }
    /// Active Gamma nodes only. Conditional MC error excludes density/model error,
    /// equation-(11) time bias, spatial bias and the unresolved initial boundary.
    #[must_use]
    pub fn gamma_standard_errors(&self) -> Option<&[f64]> {
        self.gamma_standard_errors.as_deref()
    }
}

impl VegaKtStepSample {
    /// Row-major N+1 actual-node direct effective-volatility adjoints.
    #[must_use]
    pub fn direct_node_adjoints(&self) -> &[f64] {
        &self.values[self.layout.direct_offset..self.layout.direct_totals_offset]
    }
}
impl VegaKtStepEstimate {
    #[must_use]
    pub fn direct_variance_rows(&self) -> &[VegaKtDirectVarianceRowEstimate] {
        &self.direct_rows
    }
    /// Sum of effective-local-volatility transition and direct contributions.
    /// Includes their covariance. Original-grid nodal Vega uses additional ratios.
    #[must_use]
    pub const fn combined_total(&self) -> VegaKtStepMoment {
        self.combined_total
    }
    /// True if any path supplied a nonzero direct variance seed, even if it
    /// cancels in the sample mean. All Gamma rows are then absent.
    #[must_use]
    pub const fn has_unrecovered_direct_variance(&self) -> bool {
        self.has_unrecovered_direct_variance
    }
}
impl VegaKtDirectVarianceRowEstimate {
    #[must_use]
    pub const fn maturity(&self) -> f64 {
        self.maturity
    }
    #[must_use]
    pub fn nodes(&self) -> &[VegaKtStepMoment] {
        &self.nodes
    }
    #[must_use]
    pub const fn total(&self) -> VegaKtStepMoment {
        self.total
    }
}
