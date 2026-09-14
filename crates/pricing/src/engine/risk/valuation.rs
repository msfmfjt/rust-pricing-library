//! risk / valuation implementation.

use crate::api::mc_result::MonteCarloDiagnostics;
use crate::api::mc_result::MonteCarloPrice;
use crate::api::mc_result::PayoffValuationKind;
use crate::engine::plan::simulation::AAD_WORKSPACE_SLOTS;
use crate::engine::plan::simulation::PATHWISE_COMPONENTS;
use crate::engine::plan::simulation::PRICE;
use crate::engine::plan::simulation::SimulationPlan;
use crate::engine::risk::report::estimate_from_statistics;
use crate::engine::risk::report::extrapolation_warnings;
use crate::mc::{
    DeterministicExecutor, DeterministicStatistics, EngineConfig, ExecutionPolicy, Philox4x32,
    PseudoMcConfig, RandomDomain, RqmcConfig, RqmcPlan, RqmcPlanError,
};
use crate::{Diagnostics, Estimate, EstimatorKind, MonteCarloError, PricingRequest, PricingResult};

impl SimulationPlan {
    pub fn execute(&self) -> Result<MonteCarloPrice, MonteCarloError> {
        if let Some(early_exercise) = &self.early_exercise {
            return self.execute_early_exercise(early_exercise);
        }
        if self.observation_dates.is_empty() {
            return self.execute_fixed_payoff();
        }
        match self.engine {
            EngineConfig::PseudoMonteCarlo(engine) => self.execute_pseudo(engine),
            EngineConfig::RandomizedQuasiMonteCarlo(engine) => self.execute_rqmc(engine),
        }
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn execute_fixed_payoff(
        &self,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        let value = self.discounted_payoff_from_normals(&[])?;
        let (
            estimator,
            master_seed,
            independent_units,
            evaluated_paths,
            scramble_count,
            antithetic,
        ) = match self.engine {
            EngineConfig::PseudoMonteCarlo(engine) => (
                EstimatorKind::PseudoMonteCarlo,
                engine.master_seed(),
                engine.independent_sampling_units().get(),
                engine.evaluated_paths(),
                None,
                engine.variance_reduction().antithetic(),
            ),
            EngineConfig::RandomizedQuasiMonteCarlo(engine) => {
                let multiplier = if engine.variance_reduction().antithetic() {
                    2_u128
                } else {
                    1_u128
                };
                (
                    EstimatorKind::RandomizedQuasiMonteCarlo,
                    engine.master_scramble_seed(),
                    u64::from(engine.scramble_count().get()),
                    u128::from(engine.points_per_scramble().get())
                        * u128::from(engine.scramble_count().get())
                        * multiplier,
                    Some(engine.scramble_count().get()),
                    engine.variance_reduction().antithetic(),
                )
            }
        };
        let estimate = Estimate::new(value, 0.0, value, value, estimator, independent_units)?;
        let risks = self.fixed_payoff_risk_report(estimator, independent_units)?;
        let risk_diagnostics = self.fixed_payoff_risk_diagnostics(estimator, independent_units)?;
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
            sampling_variance: 0.0,
            estimator_variance: 0.0,
            risk_diagnostics,
            independent_sampling_units: independent_units,
            evaluated_paths,
            diagnostics: MonteCarloDiagnostics {
                master_seed,
                estimator,
                scramble_count,
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
                valuation_kind: if self.payoff_smoothing.is_some() {
                    PayoffValuationKind::SmoothedSurrogate
                } else {
                    PayoffValuationKind::ExactContractual
                },
                payoff_smoothing: self.payoff_smoothing_diagnostics(),
                path_state: self.path_state_diagnostics,
                barrier_bridge: None,
            },
            early_exercise_diagnostics: None,
        })
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn execute_pseudo(
        &self,
        engine: PseudoMcConfig,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        if let Some(local_volatility) = &self.local_volatility {
            return self.execute_local_vol_pseudo(engine, local_volatility);
        }
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let generator = Philox4x32::from_seed(engine.master_seed());
        let antithetic = engine.variance_reduction().antithetic();
        let statistics = if self.risk_enabled() {
            executor.try_map_reduce_statistics_array_with_aad_workspace(
                engine.independent_sampling_units().get(),
                self.aad_tile_policy.resolved_capacity(),
                AAD_WORKSPACE_SLOTS,
                |sampling_unit, lane, workspace| {
                    let normals = self.normals(&generator, sampling_unit, RandomDomain::Valuation);
                    let primary = self.pathwise_values(&normals, lane, workspace)?;
                    if antithetic {
                        let mate_normals =
                            normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                        let mate = self.pathwise_values(&mate_normals, lane, workspace)?;
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                            |component| (primary[component] + mate[component]) * 0.5,
                        ))
                    } else {
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
                    }
                },
            )?
        } else if self.continuous_barrier.is_some() {
            executor.try_map_reduce_statistics_array_tiled(
                engine.independent_sampling_units().get(),
                self.aad_tile_policy.resolved_capacity(),
                |sampling_unit| {
                    let normals = self.normals(&generator, sampling_unit, RandomDomain::Valuation);
                    let primary = self.continuous_barrier_path_values_from_normals(&normals)?;
                    if antithetic {
                        let mate_normals =
                            normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                        let mate =
                            self.continuous_barrier_path_values_from_normals(&mate_normals)?;
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                            |component| (primary[component] + mate[component]) * 0.5,
                        ))
                    } else {
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
                    }
                },
            )?
        } else {
            let price = executor.try_map_reduce_statistics(
                engine.independent_sampling_units().get(),
                |sampling_unit| {
                    let normals = self.normals(&generator, sampling_unit, RandomDomain::Valuation);
                    let primary = self.discounted_payoff_from_normals(&normals)?;
                    if antithetic {
                        let mate_normals =
                            normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                        let mate = self.discounted_payoff_from_normals(&mate_normals)?;
                        Ok::<f64, MonteCarloError>((primary + mate) * 0.5)
                    } else {
                        Ok::<f64, MonteCarloError>(primary)
                    }
                },
            )?;
            let mut values = [DeterministicStatistics::default(); PATHWISE_COMPONENTS];
            values[PRICE] = price;
            values
        };

        let independent_units = engine.independent_sampling_units().get();
        let price = statistics[PRICE].sum().total() / independent_units as f64;
        let sampling_variance = if self.total_variance == 0.0 {
            0.0
        } else {
            statistics[PRICE].moments().sample_variance().ok_or(
                MonteCarloError::InsufficientSamplingUnits {
                    count: independent_units,
                },
            )?
        };
        let estimator_variance = sampling_variance / independent_units as f64;
        let estimate = estimate_from_statistics(
            statistics[PRICE],
            independent_units,
            1.0,
            EstimatorKind::PseudoMonteCarlo,
        )?;
        debug_assert_eq!(estimate.value().get().to_bits(), price.to_bits());
        let risks = self.build_risk_report(
            &statistics,
            independent_units,
            EstimatorKind::PseudoMonteCarlo,
            None,
        )?;
        let risk_diagnostics = self.build_risk_diagnostics(
            &statistics,
            independent_units,
            EstimatorKind::PseudoMonteCarlo,
        )?;
        let warnings = extrapolation_warnings(self.discount_region, self.dividend_region);
        let pricing_result = PricingResult {
            value: estimate,
            risks,
            diagnostics: Diagnostics::new(warnings),
            replay: self.replay_metadata(),
        };
        Ok(MonteCarloPrice {
            pricing_result,
            sampling_variance,
            estimator_variance,
            risk_diagnostics,
            independent_sampling_units: independent_units,
            evaluated_paths: engine.evaluated_paths(),
            diagnostics: MonteCarloDiagnostics {
                master_seed: engine.master_seed(),
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
                valuation_kind: if self.payoff_smoothing.is_some() {
                    PayoffValuationKind::SmoothedSurrogate
                } else {
                    PayoffValuationKind::ExactContractual
                },
                payoff_smoothing: self.payoff_smoothing_diagnostics(),
                path_state: self.path_state_diagnostics,
                barrier_bridge: self.barrier_bridge_diagnostics(&statistics, independent_units),
            },
            early_exercise_diagnostics: None,
        })
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn execute_rqmc(
        &self,
        engine: RqmcConfig,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        if let Some(local_volatility) = &self.local_volatility {
            return self.execute_local_vol_rqmc(engine, local_volatility);
        }
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let qmc = RqmcPlan::compile(
            engine,
            u32::try_from(self.observation_times.len())
                .map_err(|_| MonteCarloError::RqmcPlan(RqmcPlanError::TableSizeOverflow))?,
        )?;
        let antithetic = engine.variance_reduction().antithetic();
        let points = engine.points_per_scramble().get();
        let mut replicate_values: Vec<[f64; PATHWISE_COMPONENTS]> = Vec::with_capacity(
            usize::try_from(engine.scramble_count().get()).expect("u32 fits usize"),
        );
        for scramble in 0..engine.scramble_count().get() {
            let within = if self.risk_enabled() {
                executor.try_map_reduce_statistics_array_with_aad_workspace(
                    points,
                    self.aad_tile_policy.resolved_capacity(),
                    AAD_WORKSPACE_SLOTS,
                    |point, lane, workspace| {
                        let normals = self.rqmc_normals(&qmc, scramble, point)?;
                        let primary = self.pathwise_values(&normals, lane, workspace)?;
                        if antithetic {
                            let mate_normals =
                                normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                            let mate = self.pathwise_values(&mate_normals, lane, workspace)?;
                            Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                                |component| (primary[component] + mate[component]) * 0.5,
                            ))
                        } else {
                            Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
                        }
                    },
                )?
            } else if self.continuous_barrier.is_some() {
                executor.try_map_reduce_statistics_array_tiled(
                    points,
                    self.aad_tile_policy.resolved_capacity(),
                    |point| {
                        let normals = self.rqmc_normals(&qmc, scramble, point)?;
                        let primary = self.continuous_barrier_path_values_from_normals(&normals)?;
                        if antithetic {
                            let mate_normals =
                                normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                            let mate =
                                self.continuous_barrier_path_values_from_normals(&mate_normals)?;
                            Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                                |component| (primary[component] + mate[component]) * 0.5,
                            ))
                        } else {
                            Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
                        }
                    },
                )?
            } else {
                let price = executor.try_map_reduce_statistics(points, |point| {
                    let normals = self.rqmc_normals(&qmc, scramble, point)?;
                    let primary = self.discounted_payoff_from_normals(&normals)?;
                    if antithetic {
                        let mate_normals =
                            normals.iter().map(|normal| -*normal).collect::<Vec<_>>();
                        let mate = self.discounted_payoff_from_normals(&mate_normals)?;
                        Ok::<f64, MonteCarloError>((primary + mate) * 0.5)
                    } else {
                        Ok::<f64, MonteCarloError>(primary)
                    }
                })?;
                let mut values = [DeterministicStatistics::default(); PATHWISE_COMPONENTS];
                values[PRICE] = price;
                values
            };
            replicate_values.push(std::array::from_fn(|component| {
                within[component].sum().total() / points as f64
            }));
        }

        let statistics: [DeterministicStatistics; PATHWISE_COMPONENTS] =
            std::array::from_fn(|component| {
                let values = replicate_values
                    .iter()
                    .map(|replicate| replicate[component])
                    .collect::<Vec<_>>();
                DeterministicStatistics::from_ordered_values_two_pass(&values)
            });
        let independent_units = u64::from(engine.scramble_count().get());
        let sampling_variance = if self.total_variance == 0.0 {
            0.0
        } else {
            statistics[PRICE].moments().sample_variance().ok_or(
                MonteCarloError::InsufficientSamplingUnits {
                    count: independent_units,
                },
            )?
        };
        let estimator_variance = sampling_variance / independent_units as f64;
        let estimate = estimate_from_statistics(
            statistics[PRICE],
            independent_units,
            1.0,
            EstimatorKind::RandomizedQuasiMonteCarlo,
        )?;
        let risks = self.build_risk_report(
            &statistics,
            independent_units,
            EstimatorKind::RandomizedQuasiMonteCarlo,
            None,
        )?;
        let risk_diagnostics = self.build_risk_diagnostics(
            &statistics,
            independent_units,
            EstimatorKind::RandomizedQuasiMonteCarlo,
        )?;
        let pricing_result = PricingResult {
            value: estimate,
            risks,
            diagnostics: Diagnostics::new(extrapolation_warnings(
                self.discount_region,
                self.dividend_region,
            )),
            replay: self.replay_metadata(),
        };
        let multiplier = if antithetic { 2_u128 } else { 1_u128 };
        Ok(MonteCarloPrice {
            pricing_result,
            sampling_variance,
            estimator_variance,
            risk_diagnostics,
            independent_sampling_units: independent_units,
            evaluated_paths: u128::from(points)
                * u128::from(engine.scramble_count().get())
                * multiplier,
            diagnostics: MonteCarloDiagnostics {
                master_seed: engine.master_scramble_seed(),
                estimator: EstimatorKind::RandomizedQuasiMonteCarlo,
                scramble_count: Some(engine.scramble_count().get()),
                direction_checksum: Some(qmc.direction_checksum()),
                scramble_checksum: Some(qmc.scramble_checksum()),
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
                valuation_kind: if self.payoff_smoothing.is_some() {
                    PayoffValuationKind::SmoothedSurrogate
                } else {
                    PayoffValuationKind::ExactContractual
                },
                payoff_smoothing: self.payoff_smoothing_diagnostics(),
                path_state: self.path_state_diagnostics,
                barrier_bridge: self.barrier_bridge_diagnostics(&statistics, independent_units),
            },
            early_exercise_diagnostics: None,
        })
    }
}

pub fn price_pseudo_monte_carlo(
    request: &PricingRequest,
    execution_policy: ExecutionPolicy,
) -> Result<MonteCarloPrice, MonteCarloError> {
    if !matches!(request.engine(), EngineConfig::PseudoMonteCarlo(_)) {
        return Err(MonteCarloError::UnsupportedEngine);
    }
    SimulationPlan::compile(request, execution_policy)?.execute()
}

pub fn price_monte_carlo(
    request: &PricingRequest,
    execution_policy: ExecutionPolicy,
) -> Result<MonteCarloPrice, MonteCarloError> {
    SimulationPlan::compile(request, execution_policy)?.execute()
}
