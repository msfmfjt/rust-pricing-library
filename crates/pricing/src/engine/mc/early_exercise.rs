//! mc / early exercise implementation.

use crate::core::PathIndex;

use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, LsmNumericalError,
    LsmStateVariable, Philox4x32, PseudoMcConfig, RandomDomain, RqmcConfig, RqmcPlan,
    RqmcPlanError,
};

use crate::MonteCarloError;

use crate::engine::plan::simulation::PATHWISE_COMPONENTS;

use crate::engine::plan::simulation::AAD_WORKSPACE_SLOTS;

use crate::engine::plan::simulation::SimulationPlan;

use crate::engine::plan::simulation::LocalVolRuntime;

use crate::engine::plan::simulation::EarlyExerciseRuntime;

use crate::engine::plan::simulation::LsmPathMatrices;

use crate::engine::sampling::path_normals::local_vol_rqmc_shocks;

impl SimulationPlan {
    pub(in crate::engine) fn pseudo_lsm_path_matrices(
        &self,
        engine: PseudoMcConfig,
        domain: RandomDomain,
        early_exercise: &EarlyExerciseRuntime,
    ) -> Result<LsmPathMatrices, MonteCarloError> {
        if let Some(local_volatility) = &self.local_volatility {
            return self.local_vol_pseudo_lsm_path_matrices(
                engine,
                domain,
                early_exercise,
                local_volatility,
            );
        }
        let multiplier = if engine.variance_reduction().antithetic() {
            2_usize
        } else {
            1_usize
        };
        let units = usize::try_from(engine.independent_sampling_units().get())
            .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
        let path_count = units
            .checked_mul(multiplier)
            .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
        let date_count = early_exercise.exercise_dates.len();
        let feature_count = early_exercise.config.state_variables().len();
        let mut immediate_values = zeroed_lsm_values(
            date_count
                .checked_mul(path_count)
                .ok_or(LsmNumericalError::MatrixShapeOverflow)?,
            "American immediate values",
        )?;
        let mut features = zeroed_lsm_values(
            date_count
                .saturating_sub(1)
                .checked_mul(path_count)
                .and_then(|count| count.checked_mul(feature_count))
                .ok_or(LsmNumericalError::MatrixShapeOverflow)?,
            "American state features",
        )?;
        let generator = Philox4x32::from_seed(engine.master_seed());
        for unit in 0..units {
            let normals = self.normals(&generator, unit as u64, domain);
            self.write_lsm_trajectory(
                &normals,
                unit * multiplier,
                path_count,
                early_exercise,
                &mut immediate_values,
                &mut features,
            )?;
            if multiplier == 2 {
                let mate_normals = normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                self.write_lsm_trajectory(
                    &mate_normals,
                    unit * multiplier + 1,
                    path_count,
                    early_exercise,
                    &mut immediate_values,
                    &mut features,
                )?;
            }
        }
        Ok(LsmPathMatrices {
            path_count,
            immediate_values: immediate_values.into_boxed_slice(),
            features: features.into_boxed_slice(),
        })
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn rqmc_lsm_path_matrices(
        &self,
        engine: RqmcConfig,
        early_exercise: &EarlyExerciseRuntime,
    ) -> Result<(LsmPathMatrices, RqmcPlan), MonteCarloError> {
        if let Some(local_volatility) = &self.local_volatility {
            return self.local_vol_rqmc_lsm_path_matrices(engine, early_exercise, local_volatility);
        }
        let effective_dimension = u32::try_from(self.observation_times.len())
            .map_err(|_| RqmcPlanError::TableSizeOverflow)?;
        let qmc = RqmcPlan::compile(engine, effective_dimension)?;
        let multiplier = if engine.variance_reduction().antithetic() {
            2_usize
        } else {
            1_usize
        };
        let scrambles = usize::try_from(engine.scramble_count().get())
            .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
        let points = usize::try_from(engine.points_per_scramble().get())
            .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
        let path_count = scrambles
            .checked_mul(points)
            .and_then(|count| count.checked_mul(multiplier))
            .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
        let date_count = early_exercise.exercise_dates.len();
        let feature_count = early_exercise.config.state_variables().len();
        let mut immediate_values = zeroed_lsm_values(
            date_count
                .checked_mul(path_count)
                .ok_or(LsmNumericalError::MatrixShapeOverflow)?,
            "American RQMC immediate values",
        )?;
        let mut features = zeroed_lsm_values(
            date_count
                .saturating_sub(1)
                .checked_mul(path_count)
                .and_then(|count| count.checked_mul(feature_count))
                .ok_or(LsmNumericalError::MatrixShapeOverflow)?,
            "American RQMC state features",
        )?;
        for scramble in 0..engine.scramble_count().get() {
            for point in 0..engine.points_per_scramble().get() {
                let base_path = (usize::try_from(scramble).expect("u32 fits usize") * points
                    + usize::try_from(point).expect("configured RQMC point count fits usize"))
                    * multiplier;
                let normals = self.rqmc_normals(&qmc, scramble, point)?;
                self.write_lsm_trajectory(
                    &normals,
                    base_path,
                    path_count,
                    early_exercise,
                    &mut immediate_values,
                    &mut features,
                )?;
                if multiplier == 2 {
                    let mate_normals = normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                    self.write_lsm_trajectory(
                        &mate_normals,
                        base_path + 1,
                        path_count,
                        early_exercise,
                        &mut immediate_values,
                        &mut features,
                    )?;
                }
            }
        }
        Ok((
            LsmPathMatrices {
                path_count,
                immediate_values: immediate_values.into_boxed_slice(),
                features: features.into_boxed_slice(),
            },
            qmc,
        ))
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_pseudo_lsm_path_matrices(
        &self,
        engine: PseudoMcConfig,
        domain: RandomDomain,
        early_exercise: &EarlyExerciseRuntime,
        local_volatility: &LocalVolRuntime,
    ) -> Result<LsmPathMatrices, MonteCarloError> {
        let multiplier = if engine.variance_reduction().antithetic() {
            2_usize
        } else {
            1_usize
        };
        let units = usize::try_from(engine.independent_sampling_units().get())
            .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
        let path_count = units
            .checked_mul(multiplier)
            .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
        let date_count = early_exercise.exercise_dates.len();
        let feature_count = early_exercise.config.state_variables().len();
        let mut immediate_values = zeroed_lsm_values(
            date_count
                .checked_mul(path_count)
                .ok_or(LsmNumericalError::MatrixShapeOverflow)?,
            "Local Volatility American immediate values",
        )?;
        let mut features = zeroed_lsm_values(
            date_count
                .saturating_sub(1)
                .checked_mul(path_count)
                .and_then(|count| count.checked_mul(feature_count))
                .ok_or(LsmNumericalError::MatrixShapeOverflow)?,
            "Local Volatility American state features",
        )?;
        for unit in 0..units {
            let shocks = local_volatility.plan.path_shocks(
                engine.master_seed(),
                unit as u64,
                domain,
                engine.variance_reduction().brownian_bridge(),
            )?;
            let path_index = PathIndex::new(unit as u64);
            self.write_local_vol_lsm_trajectory(
                local_volatility,
                &shocks,
                path_index,
                unit * multiplier,
                path_count,
                early_exercise,
                &mut immediate_values,
                &mut features,
            )?;
            if multiplier == 2 {
                let mate_shocks = shocks.iter().map(|shock| -*shock).collect::<Vec<_>>();
                self.write_local_vol_lsm_trajectory(
                    local_volatility,
                    &mate_shocks,
                    path_index,
                    unit * multiplier + 1,
                    path_count,
                    early_exercise,
                    &mut immediate_values,
                    &mut features,
                )?;
            }
        }
        Ok(LsmPathMatrices {
            path_count,
            immediate_values: immediate_values.into_boxed_slice(),
            features: features.into_boxed_slice(),
        })
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_rqmc_lsm_path_matrices(
        &self,
        engine: RqmcConfig,
        early_exercise: &EarlyExerciseRuntime,
        local_volatility: &LocalVolRuntime,
    ) -> Result<(LsmPathMatrices, RqmcPlan), MonteCarloError> {
        let step_count = u32::try_from(local_volatility.plan.time_grid().step_count())
            .map_err(|_| RqmcPlanError::TableSizeOverflow)?;
        let qmc = RqmcPlan::compile(engine, step_count)?;
        let bridge = if engine.variance_reduction().brownian_bridge() {
            Some(
                BrownianBridgePlan::compile(local_volatility.plan.time_grid().nodes().to_vec(), 1)
                    .map_err(|error| MonteCarloError::LocalVol(error.into()))?,
            )
        } else {
            None
        };
        let multiplier = if engine.variance_reduction().antithetic() {
            2_usize
        } else {
            1_usize
        };
        let scrambles = usize::try_from(engine.scramble_count().get())
            .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
        let points = usize::try_from(engine.points_per_scramble().get())
            .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
        let path_count = scrambles
            .checked_mul(points)
            .and_then(|count| count.checked_mul(multiplier))
            .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
        let date_count = early_exercise.exercise_dates.len();
        let feature_count = early_exercise.config.state_variables().len();
        let mut immediate_values = zeroed_lsm_values(
            date_count
                .checked_mul(path_count)
                .ok_or(LsmNumericalError::MatrixShapeOverflow)?,
            "Local Volatility American RQMC immediate values",
        )?;
        let mut features = zeroed_lsm_values(
            date_count
                .saturating_sub(1)
                .checked_mul(path_count)
                .and_then(|count| count.checked_mul(feature_count))
                .ok_or(LsmNumericalError::MatrixShapeOverflow)?,
            "Local Volatility American RQMC state features",
        )?;
        for scramble in 0..engine.scramble_count().get() {
            for point in 0..engine.points_per_scramble().get() {
                let point_index =
                    u64::from(scramble).saturating_mul(engine.points_per_scramble().get()) + point;
                let base_path = (usize::try_from(scramble).expect("u32 fits usize") * points
                    + usize::try_from(point).expect("configured RQMC point count fits usize"))
                    * multiplier;
                let shocks = local_vol_rqmc_shocks(&qmc, bridge.as_ref(), scramble, point)?;
                let path_index = PathIndex::new(point_index);
                self.write_local_vol_lsm_trajectory(
                    local_volatility,
                    &shocks,
                    path_index,
                    base_path,
                    path_count,
                    early_exercise,
                    &mut immediate_values,
                    &mut features,
                )?;
                if multiplier == 2 {
                    let mate_shocks = shocks.iter().map(|shock| -*shock).collect::<Vec<_>>();
                    self.write_local_vol_lsm_trajectory(
                        local_volatility,
                        &mate_shocks,
                        path_index,
                        base_path + 1,
                        path_count,
                        early_exercise,
                        &mut immediate_values,
                        &mut features,
                    )?;
                }
            }
        }
        Ok((
            LsmPathMatrices {
                path_count,
                immediate_values: immediate_values.into_boxed_slice(),
                features: features.into_boxed_slice(),
            },
            qmc,
        ))
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn write_lsm_trajectory(
        &self,
        normals: &[f64],
        path: usize,
        path_count: usize,
        early_exercise: &EarlyExerciseRuntime,
        immediate_values: &mut [f64],
        features: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        let observations = self.path_observations_from_normals(normals, self.spot, self.volatility);
        let outputs = self.payoff_outputs_from_observations(&observations)?;
        if outputs.len() != early_exercise.exercise_dates.len() {
            return Err(LsmNumericalError::ImmediateValueMatrixLengthMismatch {
                expected: early_exercise.exercise_dates.len(),
                actual: outputs.len(),
            }
            .into());
        }
        let feature_count = early_exercise.config.state_variables().len();
        for (date_index, (&immediate_value, &observation_index)) in outputs
            .iter()
            .zip(early_exercise.observation_indices.iter())
            .enumerate()
        {
            immediate_values[date_index * path_count + path] = immediate_value;
            if date_index + 1 == early_exercise.exercise_dates.len() {
                continue;
            }
            for (feature_index, state_variable) in
                early_exercise.config.state_variables().iter().enumerate()
            {
                let feature = match state_variable {
                    LsmStateVariable::Spot => observations[observation_index].post_spot,
                };
                features[(date_index * path_count + path) * feature_count + feature_index] =
                    feature;
            }
        }
        Ok(())
    }
}

impl SimulationPlan {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::engine) fn write_local_vol_lsm_trajectory(
        &self,
        local_volatility: &LocalVolRuntime,
        shocks: &[f64],
        path_index: PathIndex,
        path: usize,
        path_count: usize,
        early_exercise: &EarlyExerciseRuntime,
        immediate_values: &mut [f64],
        features: &mut [f64],
    ) -> Result<(), MonteCarloError> {
        let evolved = if let (Some(dividends), Some(schedule)) = (
            local_volatility.dividends.as_ref(),
            local_volatility.dividend_schedule.as_ref(),
        ) {
            local_volatility.plan.evolve_path_with_dividend_checks(
                &local_volatility.grid,
                self.spot,
                shocks,
                dividends,
                schedule,
                path_index,
            )?
        } else {
            local_volatility
                .plan
                .evolve_path(&local_volatility.grid, self.spot, shocks)?
        };
        let observations =
            self.local_vol_path_observations(local_volatility, &evolved, self.spot)?;
        let outputs = self.payoff_outputs_from_local_vol_observations(&observations)?;
        if outputs.len() != early_exercise.exercise_dates.len() {
            return Err(LsmNumericalError::ImmediateValueMatrixLengthMismatch {
                expected: early_exercise.exercise_dates.len(),
                actual: outputs.len(),
            }
            .into());
        }
        let feature_count = early_exercise.config.state_variables().len();
        for (date_index, (&immediate_value, &observation_index)) in outputs
            .iter()
            .zip(early_exercise.observation_indices.iter())
            .enumerate()
        {
            immediate_values[date_index * path_count + path] = immediate_value;
            if date_index + 1 == early_exercise.exercise_dates.len() {
                continue;
            }
            for (feature_index, state_variable) in
                early_exercise.config.state_variables().iter().enumerate()
            {
                let feature = match state_variable {
                    LsmStateVariable::Spot => observations[observation_index].post_spot,
                };
                features[(date_index * path_count + path) * feature_count + feature_index] =
                    feature;
            }
        }
        Ok(())
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn constant_vol_fixed_policy_pseudo_statistics(
        &self,
        engine: PseudoMcConfig,
        early_exercise: &EarlyExerciseRuntime,
        stopping_indices: &[usize],
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
        let generator = Philox4x32::from_seed(engine.master_seed());
        Ok(executor.try_map_reduce_statistics_array_with_aad_workspace(
            engine.independent_sampling_units().get(),
            self.aad_tile_policy.resolved_capacity(),
            AAD_WORKSPACE_SLOTS,
            |sampling_unit, lane, workspace| {
                let unit = usize::try_from(sampling_unit)
                    .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
                let base_path = unit
                    .checked_mul(multiplier)
                    .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
                let normals = self.normals(&generator, sampling_unit, RandomDomain::Valuation);
                let primary = self.fixed_policy_pathwise_values(
                    &normals,
                    lane,
                    workspace,
                    stopping_indices[base_path],
                    early_exercise,
                )?;
                if multiplier == 2 {
                    let mate_normals = normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                    let mate = self.fixed_policy_pathwise_values(
                        &mate_normals,
                        lane,
                        workspace,
                        stopping_indices[base_path + 1],
                        early_exercise,
                    )?;
                    Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                        |component| (primary[component] + mate[component]) * 0.5,
                    ))
                } else {
                    Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
                }
            },
        )?)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn constant_vol_fixed_policy_rqmc_statistics(
        &self,
        engine: RqmcConfig,
        early_exercise: &EarlyExerciseRuntime,
        stopping_indices: &[usize],
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
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let mut replicates: Vec<[f64; PATHWISE_COMPONENTS]> = Vec::with_capacity(
            usize::try_from(engine.scramble_count().get()).expect("u32 fits usize"),
        );
        for scramble in 0..engine.scramble_count().get() {
            let scramble_index = usize::try_from(scramble).expect("u32 fits usize");
            let within = executor.try_map_reduce_statistics_array_with_aad_workspace(
                engine.points_per_scramble().get(),
                self.aad_tile_policy.resolved_capacity(),
                AAD_WORKSPACE_SLOTS,
                |point, lane, workspace| {
                    let point_index = usize::try_from(point)
                        .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
                    let base_path = scramble_index
                        .checked_mul(points)
                        .and_then(|value| value.checked_add(point_index))
                        .and_then(|value| value.checked_mul(multiplier))
                        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
                    let normals = self.rqmc_normals(qmc, scramble, point)?;
                    let primary = self.fixed_policy_pathwise_values(
                        &normals,
                        lane,
                        workspace,
                        stopping_indices[base_path],
                        early_exercise,
                    )?;
                    if multiplier == 2 {
                        let mate_normals =
                            normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                        let mate = self.fixed_policy_pathwise_values(
                            &mate_normals,
                            lane,
                            workspace,
                            stopping_indices[base_path + 1],
                            early_exercise,
                        )?;
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                            |component| (primary[component] + mate[component]) * 0.5,
                        ))
                    } else {
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
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

pub(in crate::engine) fn zeroed_lsm_values(
    count: usize,
    resource: &'static str,
) -> Result<Vec<f64>, LsmNumericalError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource,
            requested: count,
        })?;
    values.resize(count, 0.0);
    Ok(values)
}

pub(in crate::engine) fn collapse_antithetic_values(
    path_values: &[f64],
    antithetic: bool,
) -> Result<Vec<f64>, LsmNumericalError> {
    if !antithetic {
        let mut values = Vec::new();
        values.try_reserve_exact(path_values.len()).map_err(|_| {
            LsmNumericalError::AllocationFailed {
                resource: "LSM sampling-unit values",
                requested: path_values.len(),
            }
        })?;
        values.extend_from_slice(path_values);
        return Ok(values);
    }
    if !path_values.len().is_multiple_of(2) {
        return Err(LsmNumericalError::MatrixShapeOverflow);
    }
    let pairs = path_values.as_chunks::<2>().0;
    let mut values = Vec::new();
    values
        .try_reserve_exact(pairs.len())
        .map_err(|_| LsmNumericalError::AllocationFailed {
            resource: "LSM antithetic sampling-unit values",
            requested: pairs.len(),
        })?;
    for pair in pairs {
        let value = (pair[0] + pair[1]) * 0.5;
        if !value.is_finite() {
            return Err(LsmNumericalError::NonFiniteIntermediate {
                stage: "antithetic sampling-unit average",
            });
        }
        values.push(value);
    }
    Ok(values)
}

pub(in crate::engine) fn rqmc_replicate_values(
    path_values: &[f64],
    engine: RqmcConfig,
) -> Result<Vec<f64>, LsmNumericalError> {
    let multiplier = if engine.variance_reduction().antithetic() {
        2_usize
    } else {
        1_usize
    };
    let points = usize::try_from(engine.points_per_scramble().get())
        .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
    let paths_per_scramble = points
        .checked_mul(multiplier)
        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
    let scramble_count = usize::try_from(engine.scramble_count().get())
        .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
    let expected = scramble_count
        .checked_mul(paths_per_scramble)
        .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
    if path_values.len() != expected {
        return Err(LsmNumericalError::ImmediateValueLengthMismatch {
            expected,
            actual: path_values.len(),
        });
    }
    let mut replicates = Vec::new();
    replicates.try_reserve_exact(scramble_count).map_err(|_| {
        LsmNumericalError::AllocationFailed {
            resource: "LSM RQMC replicate values",
            requested: scramble_count,
        }
    })?;
    for scramble_values in path_values.chunks_exact(paths_per_scramble) {
        let mut sum = pricing_numerics::NeumaierSum::new();
        if multiplier == 2 {
            for pair in scramble_values.as_chunks::<2>().0 {
                let value = (pair[0] + pair[1]) * 0.5;
                if !value.is_finite() {
                    return Err(LsmNumericalError::NonFiniteIntermediate {
                        stage: "RQMC antithetic point average",
                    });
                }
                sum.add(value);
            }
        } else {
            for &value in scramble_values {
                sum.add(value);
            }
        }
        let replicate = sum.total() / points as f64;
        if !replicate.is_finite() {
            return Err(LsmNumericalError::NonFiniteIntermediate {
                stage: "RQMC replicate average",
            });
        }
        replicates.push(replicate);
    }
    Ok(replicates)
}
