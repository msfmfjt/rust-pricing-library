//! Early-exercise valuation and risk reporting.
use crate::api::mc_result::{
    EarlyExerciseDiagnostics, MonteCarloDiagnostics, MonteCarloPrice, PayoffValuationKind,
};
use crate::core::PathIndex;
use crate::engine::mc::early_exercise::{collapse_antithetic_values, rqmc_replicate_values};
use crate::engine::plan::simulation::{
    EarlyExerciseRuntime, LocalVolRuntime, PATHWISE_COMPONENTS, PRICE, SimulationPlan,
    bucket_sample_capacity,
};
use crate::engine::processes::local_vol_valuation::average_local_vol_pathwise;
use crate::engine::risk::report::{
    empty_risk_diagnostics, estimate_from_statistics, extrapolation_warnings,
};
use crate::engine::sampling::path_normals::local_vol_rqmc_shocks;
use crate::mc::{
    BrownianBridgePlan, DeterministicStatistics, EngineConfig, LsmNumericalError, PseudoMcConfig,
    RandomDomain, RqmcConfig, RqmcPlan, train_exercise_policy, value_exercise_policy,
};
use crate::{Diagnostics, EstimatorKind, MonteCarloError, PricingResult, RiskReport, VegaKtResult};

impl SimulationPlan {
    pub(in crate::engine) fn execute_early_exercise(
        &self,
        early_exercise: &EarlyExerciseRuntime,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        if let (
            EngineConfig::RandomizedQuasiMonteCarlo(training_engine),
            EngineConfig::RandomizedQuasiMonteCarlo(valuation_engine),
        ) = (early_exercise.config.training_engine(), self.engine)
        {
            return self.execute_early_exercise_rqmc(
                early_exercise,
                training_engine,
                valuation_engine,
            );
        }
        let (
            EngineConfig::PseudoMonteCarlo(training_engine),
            EngineConfig::PseudoMonteCarlo(valuation_engine),
        ) = (early_exercise.config.training_engine(), self.engine)
        else {
            return Err(MonteCarloError::UnsupportedEngine);
        };
        let training =
            self.pseudo_lsm_path_matrices(training_engine, RandomDomain::LsmTrain, early_exercise)?;
        let training_metadata = early_exercise
            .config
            .training_metadata(*self.payoff.source_fingerprint().as_bytes())?;
        let fitted = train_exercise_policy(
            &early_exercise.exercise_dates,
            early_exercise.config.basis().clone(),
            &training.features,
            training.path_count,
            &training.immediate_values,
            &early_exercise.discount_factors,
            early_exercise.config.itm_abs_tolerance(),
            early_exercise.config.cpqr_config(),
            early_exercise.config.max_matrix_elements(),
            training_metadata,
        )?;
        let valuation = self.pseudo_lsm_path_matrices(
            valuation_engine,
            RandomDomain::Valuation,
            early_exercise,
        )?;
        let valued = value_exercise_policy(
            fitted.policy(),
            &valuation.features,
            valuation.path_count,
            &valuation.immediate_values,
            &early_exercise.discount_factors,
        )?;
        let antithetic = valuation_engine.variance_reduction().antithetic();
        let unit_values = collapse_antithetic_values(valued.discounted_cashflows(), antithetic)?;
        let statistics = DeterministicStatistics::from_ordered_values_two_pass(&unit_values);
        let independent_units = valuation_engine.independent_sampling_units().get();
        let estimate = estimate_from_statistics(
            statistics,
            independent_units,
            1.0,
            EstimatorKind::PseudoMonteCarlo,
        )?;
        let sampling_variance = statistics.moments().sample_variance().unwrap_or(0.0);
        let estimator_variance = sampling_variance / independent_units as f64;
        let in_sample_discounted = fitted
            .realized_cashflows()
            .iter()
            .zip(fitted.stopping_indices())
            .map(|(cashflow, stopping_index)| {
                cashflow * early_exercise.discount_factors[*stopping_index]
            })
            .collect::<Vec<_>>();
        let in_sample_units = collapse_antithetic_values(
            &in_sample_discounted,
            training_engine.variance_reduction().antithetic(),
        )?;
        let in_sample_value =
            DeterministicStatistics::from_ordered_values_two_pass(&in_sample_units)
                .sum()
                .total()
                / in_sample_units.len() as f64;
        let exercise_probabilities = valued
            .exercise_counts()
            .iter()
            .map(|count| *count as f64 / valuation.path_count as f64)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let early_risk_enabled = self.risk_enabled()
            || self
                .local_volatility
                .as_ref()
                .is_some_and(|local_volatility| local_volatility.vega_kt.is_some());
        let (risks, risk_diagnostics) = if early_risk_enabled {
            let (risk_statistics, vega_kt) = if let Some(local_volatility) = &self.local_volatility
            {
                if local_volatility.vega_kt.is_some() {
                    let (statistics, vega_kt) = self
                        .local_vol_fixed_policy_pseudo_statistics_with_vega_kt(
                            valuation_engine,
                            early_exercise,
                            valued.stopping_indices(),
                            local_volatility,
                        )?;
                    (statistics, Some(vega_kt))
                } else {
                    (
                        self.local_vol_fixed_policy_pseudo_statistics(
                            valuation_engine,
                            early_exercise,
                            valued.stopping_indices(),
                            local_volatility,
                        )?,
                        None,
                    )
                }
            } else {
                (
                    self.constant_vol_fixed_policy_pseudo_statistics(
                        valuation_engine,
                        early_exercise,
                        valued.stopping_indices(),
                    )?,
                    None,
                )
            };
            (
                self.build_risk_report(
                    &risk_statistics,
                    independent_units,
                    EstimatorKind::PseudoMonteCarlo,
                    vega_kt,
                )?,
                if self.local_volatility.is_some() {
                    self.build_local_vol_fixed_policy_risk_diagnostics(valued.policy_fingerprint())
                } else {
                    self.build_fixed_policy_risk_diagnostics(
                        &risk_statistics,
                        independent_units,
                        EstimatorKind::PseudoMonteCarlo,
                        valued.policy_fingerprint(),
                    )?
                },
            )
        } else {
            (
                RiskReport {
                    delta: None,
                    gamma: None,
                    vega: None,
                    vega_kt: None,
                },
                empty_risk_diagnostics(self.smile_dynamics),
            )
        };
        let early_exercise_diagnostics = EarlyExerciseDiagnostics {
            policy_fingerprint: valued.policy_fingerprint(),
            training_random_domain: early_exercise.config.training_random_domain(),
            valuation_random_domain: RandomDomain::Valuation,
            training_direction_checksum: None,
            training_scramble_checksum: None,
            valuation_direction_checksum: None,
            valuation_scramble_checksum: None,
            training_sampling_units: training_engine.independent_sampling_units().get(),
            training_trajectories: training.path_count as u64,
            valuation_sampling_units: independent_units,
            valuation_trajectories: valuation.path_count as u64,
            in_sample_value,
            exercise_dates: early_exercise.exercise_dates.clone(),
            exercise_counts: valued.exercise_counts().into(),
            exercise_probabilities,
            stopping_indices: valued.stopping_indices().into(),
            dividend_collisions: early_exercise.dividend_collisions.clone(),
            regression_diagnostics: fitted.policy().diagnostics().into(),
            policy_basis: fitted.policy().basis().clone(),
            itm_abs_tolerance: fitted.policy().itm_abs_tolerance(),
            cpqr_config: fitted.policy().cpqr_config(),
            max_matrix_elements: fitted.policy().max_matrix_elements(),
            decision_models: fitted.policy().decisions().into(),
        };
        let pricing_result = PricingResult {
            value: estimate,
            risks,
            diagnostics: Diagnostics::new(extrapolation_warnings(
                self.discount_region,
                self.dividend_region,
            )),
            replay: self.replay_metadata(),
        };
        Ok(MonteCarloPrice {
            pricing_result,
            sampling_variance,
            estimator_variance,
            risk_diagnostics,
            independent_sampling_units: independent_units,
            evaluated_paths: valuation_engine.evaluated_paths(),
            diagnostics: MonteCarloDiagnostics {
                master_seed: valuation_engine.master_seed(),
                estimator: EstimatorKind::PseudoMonteCarlo,
                scramble_count: None,
                direction_checksum: None,
                scramble_checksum: None,
                policy_version: self.execution_policy.version(),
                worker_threads: self.execution_policy.worker_threads().get(),
                reduction_block_size: self.execution_policy.reduction_block_size().get(),
                aad_tile_policy_version: self.aad_tile_policy.version(),
                aad_tile_capacity: self.aad_tile_policy.resolved_capacity().get(),
                checkpoint_policy_version: self.checkpoint_policy.version(),
                checkpoint_interval: self.checkpoint_policy.resolved_interval().get(),
                antithetic,
                discount_region: self.discount_region,
                dividend_region: self.dividend_region,
                payoff_fingerprint: self.payoff.tape_fingerprint(),
                valuation_kind: PayoffValuationKind::ExactContractual,
                payoff_smoothing: None,
                path_state: None,
                barrier_bridge: None,
            },
            early_exercise_diagnostics: Some(early_exercise_diagnostics),
        })
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn execute_early_exercise_rqmc(
        &self,
        early_exercise: &EarlyExerciseRuntime,
        training_engine: RqmcConfig,
        valuation_engine: RqmcConfig,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        let (training, training_qmc) =
            self.rqmc_lsm_path_matrices(training_engine, early_exercise)?;
        let training_metadata = early_exercise
            .config
            .training_metadata(*self.payoff.source_fingerprint().as_bytes())?;
        let fitted = train_exercise_policy(
            &early_exercise.exercise_dates,
            early_exercise.config.basis().clone(),
            &training.features,
            training.path_count,
            &training.immediate_values,
            &early_exercise.discount_factors,
            early_exercise.config.itm_abs_tolerance(),
            early_exercise.config.cpqr_config(),
            early_exercise.config.max_matrix_elements(),
            training_metadata,
        )?;
        let (valuation, valuation_qmc) =
            self.rqmc_lsm_path_matrices(valuation_engine, early_exercise)?;
        let valued = value_exercise_policy(
            fitted.policy(),
            &valuation.features,
            valuation.path_count,
            &valuation.immediate_values,
            &early_exercise.discount_factors,
        )?;
        let replicate_values =
            rqmc_replicate_values(valued.discounted_cashflows(), valuation_engine)?;
        let statistics = DeterministicStatistics::from_ordered_values_two_pass(&replicate_values);
        let independent_units = u64::from(valuation_engine.scramble_count().get());
        let estimate = estimate_from_statistics(
            statistics,
            independent_units,
            1.0,
            EstimatorKind::RandomizedQuasiMonteCarlo,
        )?;
        let sampling_variance = statistics.moments().sample_variance().unwrap_or(0.0);
        let estimator_variance = sampling_variance / independent_units as f64;
        let in_sample_discounted = fitted
            .realized_cashflows()
            .iter()
            .zip(fitted.stopping_indices())
            .map(|(cashflow, stopping_index)| {
                cashflow * early_exercise.discount_factors[*stopping_index]
            })
            .collect::<Vec<_>>();
        let in_sample_replicates = rqmc_replicate_values(&in_sample_discounted, training_engine)?;
        let in_sample_value =
            DeterministicStatistics::from_ordered_values_two_pass(&in_sample_replicates)
                .sum()
                .total()
                / in_sample_replicates.len() as f64;
        let exercise_probabilities = valued
            .exercise_counts()
            .iter()
            .map(|count| *count as f64 / valuation.path_count as f64)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let early_risk_enabled = self.risk_enabled()
            || self
                .local_volatility
                .as_ref()
                .is_some_and(|local_volatility| local_volatility.vega_kt.is_some());
        let (risks, risk_diagnostics) = if early_risk_enabled {
            let (risk_statistics, vega_kt) = if let Some(local_volatility) = &self.local_volatility
            {
                if local_volatility.vega_kt.is_some() {
                    let (statistics, vega_kt) = self
                        .local_vol_fixed_policy_rqmc_statistics_with_vega_kt(
                            valuation_engine,
                            early_exercise,
                            valued.stopping_indices(),
                            local_volatility,
                            &valuation_qmc,
                        )?;
                    (statistics, Some(vega_kt))
                } else {
                    (
                        self.local_vol_fixed_policy_rqmc_statistics(
                            valuation_engine,
                            early_exercise,
                            valued.stopping_indices(),
                            local_volatility,
                            &valuation_qmc,
                        )?,
                        None,
                    )
                }
            } else {
                (
                    self.constant_vol_fixed_policy_rqmc_statistics(
                        valuation_engine,
                        early_exercise,
                        valued.stopping_indices(),
                        &valuation_qmc,
                    )?,
                    None,
                )
            };
            (
                self.build_risk_report(
                    &risk_statistics,
                    independent_units,
                    EstimatorKind::RandomizedQuasiMonteCarlo,
                    vega_kt,
                )?,
                if self.local_volatility.is_some() {
                    self.build_local_vol_fixed_policy_risk_diagnostics(valued.policy_fingerprint())
                } else {
                    self.build_fixed_policy_risk_diagnostics(
                        &risk_statistics,
                        independent_units,
                        EstimatorKind::RandomizedQuasiMonteCarlo,
                        valued.policy_fingerprint(),
                    )?
                },
            )
        } else {
            (
                RiskReport {
                    delta: None,
                    gamma: None,
                    vega: None,
                    vega_kt: None,
                },
                empty_risk_diagnostics(self.smile_dynamics),
            )
        };
        let early_exercise_diagnostics = EarlyExerciseDiagnostics {
            policy_fingerprint: valued.policy_fingerprint(),
            training_random_domain: RandomDomain::RqmcScramble,
            valuation_random_domain: RandomDomain::RqmcScramble,
            training_direction_checksum: Some(training_qmc.direction_checksum()),
            training_scramble_checksum: Some(training_qmc.scramble_checksum()),
            valuation_direction_checksum: Some(valuation_qmc.direction_checksum()),
            valuation_scramble_checksum: Some(valuation_qmc.scramble_checksum()),
            training_sampling_units: u64::from(training_engine.scramble_count().get()),
            training_trajectories: training.path_count as u64,
            valuation_sampling_units: independent_units,
            valuation_trajectories: valuation.path_count as u64,
            in_sample_value,
            exercise_dates: early_exercise.exercise_dates.clone(),
            exercise_counts: valued.exercise_counts().into(),
            exercise_probabilities,
            stopping_indices: valued.stopping_indices().into(),
            dividend_collisions: early_exercise.dividend_collisions.clone(),
            regression_diagnostics: fitted.policy().diagnostics().into(),
            policy_basis: fitted.policy().basis().clone(),
            itm_abs_tolerance: fitted.policy().itm_abs_tolerance(),
            cpqr_config: fitted.policy().cpqr_config(),
            max_matrix_elements: fitted.policy().max_matrix_elements(),
            decision_models: fitted.policy().decisions().into(),
        };
        let pricing_result = PricingResult {
            value: estimate,
            risks,
            diagnostics: Diagnostics::new(extrapolation_warnings(
                self.discount_region,
                self.dividend_region,
            )),
            replay: self.replay_metadata(),
        };
        let multiplier = if valuation_engine.variance_reduction().antithetic() {
            2_u128
        } else {
            1_u128
        };
        Ok(MonteCarloPrice {
            pricing_result,
            sampling_variance,
            estimator_variance,
            risk_diagnostics,
            independent_sampling_units: independent_units,
            evaluated_paths: u128::from(valuation_engine.points_per_scramble().get())
                * u128::from(valuation_engine.scramble_count().get())
                * multiplier,
            diagnostics: MonteCarloDiagnostics {
                master_seed: valuation_engine.master_scramble_seed(),
                estimator: EstimatorKind::RandomizedQuasiMonteCarlo,
                scramble_count: Some(valuation_engine.scramble_count().get()),
                direction_checksum: Some(valuation_qmc.direction_checksum()),
                scramble_checksum: Some(valuation_qmc.scramble_checksum()),
                policy_version: self.execution_policy.version(),
                worker_threads: self.execution_policy.worker_threads().get(),
                reduction_block_size: self.execution_policy.reduction_block_size().get(),
                aad_tile_policy_version: self.aad_tile_policy.version(),
                aad_tile_capacity: self.aad_tile_policy.resolved_capacity().get(),
                checkpoint_policy_version: self.checkpoint_policy.version(),
                checkpoint_interval: self.checkpoint_policy.resolved_interval().get(),
                antithetic: valuation_engine.variance_reduction().antithetic(),
                discount_region: self.discount_region,
                dividend_region: self.dividend_region,
                payoff_fingerprint: self.payoff.tape_fingerprint(),
                valuation_kind: PayoffValuationKind::ExactContractual,
                payoff_smoothing: None,
                path_state: None,
                barrier_bridge: None,
            },
            early_exercise_diagnostics: Some(early_exercise_diagnostics),
        })
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_fixed_policy_pseudo_statistics_with_vega_kt(
        &self,
        engine: PseudoMcConfig,
        early_exercise: &EarlyExerciseRuntime,
        stopping_indices: &[usize],
        local_volatility: &LocalVolRuntime,
    ) -> Result<([DeterministicStatistics; PATHWISE_COMPONENTS], VegaKtResult), MonteCarloError>
    {
        let multiplier = if engine.variance_reduction().antithetic() {
            2_usize
        } else {
            1_usize
        };
        let independent_units = engine.independent_sampling_units().get();
        let unit_count = usize::try_from(independent_units)
            .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
        let expected = unit_count
            .checked_mul(multiplier)
            .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
        if stopping_indices.len() != expected {
            return Err(LsmNumericalError::ImmediateValueLengthMismatch {
                expected,
                actual: stopping_indices.len(),
            }
            .into());
        }
        let vega_kt = local_volatility
            .vega_kt
            .as_ref()
            .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?;
        let bucket_count = vega_kt.basis.bucket_count();
        let bump_runtimes = self.local_vol_bump_runtimes()?;
        let mut values = Vec::with_capacity(unit_count);
        let mut price_samples = Vec::with_capacity(unit_count);
        let mut raw_bucket_samples =
            Vec::with_capacity(bucket_sample_capacity(unit_count, bucket_count));
        for sampling_unit in 0..independent_units {
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
            let pathwise = if multiplier == 2 {
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
                average_local_vol_pathwise(primary, mate)?
            } else {
                primary
            };
            price_samples.push(pathwise.values[PRICE]);
            raw_bucket_samples.extend(
                pathwise
                    .raw_buckets
                    .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?,
            );
            values.push(pathwise.values);
        }
        let statistics = std::array::from_fn(|component| {
            let component_values = values
                .iter()
                .map(|pathwise| pathwise[component])
                .collect::<Vec<_>>();
            DeterministicStatistics::from_ordered_values_two_pass(&component_values)
        });
        let report = self.build_vega_kt_result(
            local_volatility,
            &statistics,
            independent_units,
            &price_samples,
            &raw_bucket_samples,
        )?;
        Ok((statistics, report))
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_fixed_policy_rqmc_statistics_with_vega_kt(
        &self,
        engine: RqmcConfig,
        early_exercise: &EarlyExerciseRuntime,
        stopping_indices: &[usize],
        local_volatility: &LocalVolRuntime,
        qmc: &RqmcPlan,
    ) -> Result<([DeterministicStatistics; PATHWISE_COMPONENTS], VegaKtResult), MonteCarloError>
    {
        let multiplier = if engine.variance_reduction().antithetic() {
            2_usize
        } else {
            1_usize
        };
        let points = usize::try_from(engine.points_per_scramble().get())
            .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
        let scramble_count = usize::try_from(engine.scramble_count().get())
            .map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
        let expected = scramble_count
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
        let vega_kt = local_volatility
            .vega_kt
            .as_ref()
            .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?;
        let bucket_count = vega_kt.basis.bucket_count();
        let bridge = if engine.variance_reduction().brownian_bridge() {
            Some(
                BrownianBridgePlan::compile(local_volatility.plan.time_grid().nodes().to_vec(), 1)
                    .map_err(|error| MonteCarloError::LocalVol(error.into()))?,
            )
        } else {
            None
        };
        let bump_runtimes = self.local_vol_bump_runtimes()?;
        let mut replicate_values = Vec::with_capacity(scramble_count);
        let mut price_samples = Vec::with_capacity(scramble_count);
        let mut raw_bucket_samples =
            Vec::with_capacity(bucket_sample_capacity(scramble_count, bucket_count));
        for scramble in 0..engine.scramble_count().get() {
            let scramble_index = usize::try_from(scramble).expect("u32 fits usize");
            let mut component_sums = [pricing_numerics::NeumaierSum::new(); PATHWISE_COMPONENTS];
            let mut bucket_sums = vec![pricing_numerics::NeumaierSum::new(); bucket_count];
            for point in 0..engine.points_per_scramble().get() {
                let point_index =
                    usize::try_from(point).map_err(|_| LsmNumericalError::MatrixShapeOverflow)?;
                let base_path = scramble_index
                    .checked_mul(points)
                    .and_then(|value| value.checked_add(point_index))
                    .and_then(|value| value.checked_mul(multiplier))
                    .ok_or(LsmNumericalError::MatrixShapeOverflow)?;
                let path = PathIndex::new(
                    u64::from(scramble).saturating_mul(engine.points_per_scramble().get()) + point,
                );
                let shocks = local_vol_rqmc_shocks(qmc, bridge.as_ref(), scramble, point)?;
                let primary = self.local_vol_fixed_policy_pathwise_values(
                    local_volatility,
                    bump_runtimes.as_ref(),
                    &shocks,
                    path,
                    self.exercise_cashflow_selection(early_exercise, stopping_indices[base_path])?,
                )?;
                let pathwise = if multiplier == 2 {
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
                    average_local_vol_pathwise(primary, mate)?
                } else {
                    primary
                };
                for (sum, value) in component_sums.iter_mut().zip(pathwise.values) {
                    sum.add(value);
                }
                for (sum, value) in bucket_sums.iter_mut().zip(
                    pathwise
                        .raw_buckets
                        .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?,
                ) {
                    sum.add(value);
                }
            }
            let replicate: [f64; PATHWISE_COMPONENTS] = std::array::from_fn(|component| {
                component_sums[component].total() / engine.points_per_scramble().get() as f64
            });
            price_samples.push(replicate[PRICE]);
            raw_bucket_samples.extend(
                bucket_sums
                    .into_iter()
                    .map(|sum| sum.total() / engine.points_per_scramble().get() as f64),
            );
            replicate_values.push(replicate);
        }
        let statistics = std::array::from_fn(|component| {
            let values = replicate_values
                .iter()
                .map(|replicate| replicate[component])
                .collect::<Vec<_>>();
            DeterministicStatistics::from_ordered_values_two_pass(&values)
        });
        let independent_units = u64::from(engine.scramble_count().get());
        let report = self.build_vega_kt_result(
            local_volatility,
            &statistics,
            independent_units,
            &price_samples,
            &raw_bucket_samples,
        )?;
        Ok((statistics, report))
    }
}
