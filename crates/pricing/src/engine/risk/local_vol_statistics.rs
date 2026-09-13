//! mc / early exercise implementation.

use crate::MonteCarloError;
use crate::core::PathIndex;
use crate::engine::plan::simulation::EarlyExerciseRuntime;
use crate::engine::plan::simulation::LocalVolRuntime;
use crate::engine::plan::simulation::PATHWISE_COMPONENTS;
use crate::engine::plan::simulation::SimulationPlan;
use crate::engine::processes::local_vol_valuation::average_local_vol_pathwise;
use crate::engine::sampling::path_normals::local_vol_rqmc_shocks;
use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, LsmNumericalError,
    PseudoMcConfig, RandomDomain, RqmcConfig, RqmcPlan,
};

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_fixed_policy_pseudo_statistics(
        &self,
        engine: PseudoMcConfig,
        early_exercise: &EarlyExerciseRuntime,
        stopping_indices: &[usize],
        local_volatility: &LocalVolRuntime,
    ) -> Result<[DeterministicStatistics; PATHWISE_COMPONENTS], MonteCarloError> {
        let multiplier = if engine.variance_reduction().antithetic() {
            2_usize
        } else {
            1_usize
        };
        let expected = usize::try_from(engine.independent_sampling_units().get())
            .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?
            .checked_mul(multiplier)
            .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
        if stopping_indices.len() != expected {
            return Err(LsmNumericalError::ImmediateValueLengthMismatch {
                expected,
                actual: stopping_indices.len(),
            }
            .into());
        }
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let bump_runtimes = self.local_vol_bump_runtimes()?;
        Ok(executor.try_map_reduce_statistics_array_tiled(
            engine.independent_sampling_units().get(),
            self.aad_tile_policy.resolved_capacity(),
            |sampling_unit| {
                let unit = usize::try_from(sampling_unit)
                    .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
                let base_path = unit
                    .checked_mul(multiplier)
                    .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
                let shocks = local_volatility.plan.path_shocks(
                    engine.master_seed(),
                    sampling_unit,
                    RandomDomain::Valuation,
                    engine.variance_reduction().brownian_bridge(),
                )?;
                let path = PathIndex::new(sampling_unit);
                let primary = self.local_vol_fixed_policy_pathwise_values(
                    local_volatility,
                    bump_runtimes.as_ref(),
                    &shocks,
                    path,
                    self.exercise_cashflow_selection(early_exercise, stopping_indices[base_path])?,
                )?;
                if multiplier == 2 {
                    let mate_shocks = shocks.iter().map(|shock| -*shock).collect::<Vec<_>>();
                    let mate = self.local_vol_fixed_policy_pathwise_values(
                        local_volatility,
                        bump_runtimes.as_ref(),
                        &mate_shocks,
                        path,
                        self.exercise_cashflow_selection(
                            early_exercise,
                            stopping_indices[base_path + 1],
                        )?,
                    )?;
                    Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(
                        average_local_vol_pathwise(primary, mate)?.values,
                    )
                } else {
                    Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary.values)
                }
            },
        )?)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_fixed_policy_rqmc_statistics(
        &self,
        engine: RqmcConfig,
        early_exercise: &EarlyExerciseRuntime,
        stopping_indices: &[usize],
        local_volatility: &LocalVolRuntime,
        qmc: &RqmcPlan,
    ) -> Result<[DeterministicStatistics; PATHWISE_COMPONENTS], MonteCarloError> {
        let multiplier = if engine.variance_reduction().antithetic() {
            2_usize
        } else {
            1_usize
        };
        let points = usize::try_from(engine.points_per_scramble().get())
            .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
        let expected = usize::try_from(engine.scramble_count().get())
            .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?
            .checked_mul(points)
            .and_then(|count| count.checked_mul(multiplier))
            .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
        if stopping_indices.len() != expected {
            return Err(LsmNumericalError::ImmediateValueLengthMismatch {
                expected,
                actual: stopping_indices.len(),
            }
            .into());
        }
        let bridge = if engine.variance_reduction().brownian_bridge() {
            Some(
                BrownianBridgePlan::compile(local_volatility.plan.time_grid().nodes().to_vec(), 1)
                    .map_err(|error| MonteCarloError::LocalVol(error.into()))?,
            )
        } else {
            None
        };
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let bump_runtimes = self.local_vol_bump_runtimes()?;
        let mut replicates: Vec<[f64; PATHWISE_COMPONENTS]> = Vec::with_capacity(
            usize::try_from(engine.scramble_count().get()).expect("u32 fits usize"),
        );
        for scramble in 0..engine.scramble_count().get() {
            let scramble_index = usize::try_from(scramble).expect("u32 fits usize");
            let within = executor.try_map_reduce_statistics_array_tiled(
                engine.points_per_scramble().get(),
                self.aad_tile_policy.resolved_capacity(),
                |point| {
                    let point_index = usize::try_from(point)
                        .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
                    let base_path = scramble_index
                        .checked_mul(points)
                        .and_then(|value| value.checked_add(point_index))
                        .and_then(|value| value.checked_mul(multiplier))
                        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
                    let path = PathIndex::new(
                        u64::from(scramble).saturating_mul(engine.points_per_scramble().get())
                            + point,
                    );
                    let shocks = local_vol_rqmc_shocks(qmc, bridge.as_ref(), scramble, point)?;
                    let primary = self.local_vol_fixed_policy_pathwise_values(
                        local_volatility,
                        bump_runtimes.as_ref(),
                        &shocks,
                        path,
                        self.exercise_cashflow_selection(
                            early_exercise,
                            stopping_indices[base_path],
                        )?,
                    )?;
                    if multiplier == 2 {
                        let mate_shocks = shocks.iter().map(|shock| -*shock).collect::<Vec<_>>();
                        let mate = self.local_vol_fixed_policy_pathwise_values(
                            local_volatility,
                            bump_runtimes.as_ref(),
                            &mate_shocks,
                            path,
                            self.exercise_cashflow_selection(
                                early_exercise,
                                stopping_indices[base_path + 1],
                            )?,
                        )?;
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(
                            average_local_vol_pathwise(primary, mate)?.values,
                        )
                    } else {
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary.values)
                    }
                },
            )?;
            replicates.push(std::array::from_fn(|component| {
                within[component].sum().total() / engine.points_per_scramble().get() as f64
            }));
        }
        Ok(std::array::from_fn(|component| {
            let values = replicates
                .iter()
                .map(|replicate| replicate[component])
                .collect::<Vec<_>>();
            DeterministicStatistics::from_ordered_values_two_pass(&values)
        }))
    }
}
