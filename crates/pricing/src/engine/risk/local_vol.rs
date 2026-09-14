//! risk / local vol implementation.

use crate::api::mc_result::MonteCarloDiagnostics;
use crate::api::mc_result::MonteCarloPrice;
use crate::api::mc_result::PayoffValuationKind;
use crate::core::PathIndex;
use crate::engine::plan::simulation::LocalVolBumpRuntimes;
use crate::engine::plan::simulation::LocalVolRuntime;
use crate::engine::plan::simulation::PATHWISE_COMPONENTS;
use crate::engine::plan::simulation::PRICE;
use crate::engine::plan::simulation::SimulationPlan;
use crate::engine::plan::simulation::VEGA;
use crate::engine::plan::simulation::bucket_sample_capacity;
use crate::engine::processes::local_vol_valuation::average_local_vol_pathwise;
use crate::engine::risk::report::estimate_from_statistics;
use crate::engine::risk::report::extrapolation_warnings;
use crate::engine::sampling::path_normals::local_vol_rqmc_shocks;
use crate::mc::{
    BrownianBridgePlan, DeterministicExecutor, DeterministicStatistics, PseudoMcConfig,
    RandomDomain, RqmcConfig, RqmcPlan, RqmcPlanError,
};
use crate::risk::{
    vega_kt_bucket_estimates, vega_kt_full_bucket_covariance, vega_kt_projection_from_parts,
    vega_kt_report,
};
use crate::{Diagnostics, EstimatorKind, MonteCarloError, PricingResult, VegaKtResult};

impl SimulationPlan {
    pub(in crate::engine) fn execute_local_vol_pseudo(
        &self,
        engine: PseudoMcConfig,
        local_volatility: &LocalVolRuntime,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let antithetic = engine.variance_reduction().antithetic();
        let brownian_bridge = engine.variance_reduction().brownian_bridge();
        let bump_runtimes = self.local_vol_bump_runtimes()?;
        if local_volatility.vega_kt.is_some() {
            return self.execute_local_vol_pseudo_with_vega_kt(
                engine,
                local_volatility,
                bump_runtimes.as_ref(),
            );
        }
        let statistics = executor.try_map_reduce_statistics_array_tiled(
            engine.independent_sampling_units().get(),
            self.aad_tile_policy.resolved_capacity(),
            |sampling_unit| {
                let shocks = local_volatility.plan.path_shocks(
                    engine.master_seed(),
                    sampling_unit,
                    RandomDomain::Valuation,
                    brownian_bridge,
                )?;
                let primary = self.local_vol_pathwise_values(
                    local_volatility,
                    bump_runtimes.as_ref(),
                    &shocks,
                    PathIndex::new(sampling_unit),
                )?;
                if antithetic {
                    let mate_shocks = shocks.iter().map(|shock| -*shock).collect::<Vec<_>>();
                    let mate = self.local_vol_pathwise_values(
                        local_volatility,
                        bump_runtimes.as_ref(),
                        &mate_shocks,
                        PathIndex::new(sampling_unit),
                    )?;
                    Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                        |component| (primary[component] + mate[component]) * 0.5,
                    ))
                } else {
                    Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
                }
            },
        )?;
        let independent_units = engine.independent_sampling_units().get();
        let estimate = estimate_from_statistics(
            statistics[PRICE],
            independent_units,
            1.0,
            EstimatorKind::PseudoMonteCarlo,
        )?;
        let sampling_variance = statistics[PRICE].moments().sample_variance().ok_or(
            MonteCarloError::InsufficientSamplingUnits {
                count: independent_units,
            },
        )?;
        let estimator_variance = sampling_variance / independent_units as f64;
        let pricing_result = PricingResult {
            value: estimate,
            risks: self.build_risk_report(
                &statistics,
                independent_units,
                EstimatorKind::PseudoMonteCarlo,
                None,
            )?,
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
            risk_diagnostics: self.build_local_vol_risk_diagnostics(),
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
    pub(in crate::engine) fn execute_local_vol_pseudo_with_vega_kt(
        &self,
        engine: PseudoMcConfig,
        local_volatility: &LocalVolRuntime,
        bump_runtimes: Option<&LocalVolBumpRuntimes>,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        let vega_kt = local_volatility
            .vega_kt
            .as_ref()
            .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?;
        let antithetic = engine.variance_reduction().antithetic();
        let brownian_bridge = engine.variance_reduction().brownian_bridge();
        let independent_units = engine.independent_sampling_units().get();
        let bucket_count = vega_kt.basis.bucket_count();
        let mut values = Vec::with_capacity(
            usize::try_from(independent_units).expect("sampling-unit count fits usize"),
        );
        let mut price_samples = Vec::with_capacity(values.capacity());
        let mut raw_bucket_samples =
            Vec::with_capacity(bucket_sample_capacity(values.capacity(), bucket_count));
        for sampling_unit in 0..independent_units {
            let shocks = local_volatility.plan.path_shocks(
                engine.master_seed(),
                sampling_unit,
                RandomDomain::Valuation,
                brownian_bridge,
            )?;
            let primary = self.local_vol_pathwise_values_and_buckets(
                local_volatility,
                bump_runtimes,
                &shocks,
                PathIndex::new(sampling_unit),
            )?;
            let pathwise = if antithetic {
                let mate_shocks = shocks.iter().map(|shock| -*shock).collect::<Vec<_>>();
                let mate = self.local_vol_pathwise_values_and_buckets(
                    local_volatility,
                    bump_runtimes,
                    &mate_shocks,
                    PathIndex::new(sampling_unit),
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

        let statistics: [DeterministicStatistics; PATHWISE_COMPONENTS] =
            std::array::from_fn(|component| {
                let component_values = values
                    .iter()
                    .map(|pathwise| pathwise[component])
                    .collect::<Vec<_>>();
                DeterministicStatistics::from_ordered_values_two_pass(&component_values)
            });
        let estimates =
            vega_kt_bucket_estimates(&price_samples, &raw_bucket_samples, bucket_count)?;
        let raw_bucket_means = estimates
            .iter()
            .map(|estimate| estimate.raw_mean())
            .collect::<Vec<_>>();
        let mean_vega = statistics[VEGA].sum().total() / independent_units as f64;
        let bucket_sum = raw_bucket_means
            .iter()
            .copied()
            .collect::<pricing_numerics::NeumaierSum>()
            .total();
        let projection = vega_kt_projection_from_parts(
            raw_bucket_means,
            mean_vega - bucket_sum,
            mean_vega,
            Default::default(),
        )?;
        let full_bucket_covariance = if vega_kt.full_bucket_covariance {
            Some(vega_kt_full_bucket_covariance(
                &raw_bucket_samples,
                bucket_count,
            )?)
        } else {
            None
        };
        let density_row = vega_kt
            .density_rows
            .last()
            .ok_or(MonteCarloError::MismatchedLocalVolatilityReportingBasis)?;
        let vega_kt = VegaKtResult::try_from(&vega_kt_report(
            &vega_kt.basis,
            density_row,
            projection,
            estimates,
            full_bucket_covariance,
        )?)?;
        let estimate = estimate_from_statistics(
            statistics[PRICE],
            independent_units,
            1.0,
            EstimatorKind::PseudoMonteCarlo,
        )?;
        let sampling_variance = statistics[PRICE].moments().sample_variance().ok_or(
            MonteCarloError::InsufficientSamplingUnits {
                count: independent_units,
            },
        )?;
        let estimator_variance = sampling_variance / independent_units as f64;
        let pricing_result = PricingResult {
            value: estimate,
            risks: self.build_risk_report(
                &statistics,
                independent_units,
                EstimatorKind::PseudoMonteCarlo,
                Some(vega_kt),
            )?,
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
            risk_diagnostics: self.build_local_vol_risk_diagnostics(),
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
    pub(in crate::engine) fn execute_local_vol_rqmc(
        &self,
        engine: RqmcConfig,
        local_volatility: &LocalVolRuntime,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        let step_count = u32::try_from(local_volatility.plan.time_grid().step_count())
            .map_err(|_| MonteCarloError::RqmcPlan(RqmcPlanError::TableSizeOverflow))?;
        let qmc = RqmcPlan::compile(engine, step_count)?;
        let bridge = if engine.variance_reduction().brownian_bridge() {
            Some(
                BrownianBridgePlan::compile(local_volatility.plan.time_grid().nodes().to_vec(), 1)
                    .map_err(|error| MonteCarloError::LocalVol(error.into()))?,
            )
        } else {
            None
        };
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let antithetic = engine.variance_reduction().antithetic();
        let points = engine.points_per_scramble().get();
        let bump_runtimes = self.local_vol_bump_runtimes()?;
        if local_volatility.vega_kt.is_some() {
            return self.execute_local_vol_rqmc_with_vega_kt(
                engine,
                local_volatility,
                bump_runtimes.as_ref(),
                &qmc,
                bridge.as_ref(),
            );
        }
        let mut replicate_values: Vec<[f64; PATHWISE_COMPONENTS]> = Vec::with_capacity(
            usize::try_from(engine.scramble_count().get()).expect("u32 fits usize"),
        );
        for scramble in 0..engine.scramble_count().get() {
            let within = executor.try_map_reduce_statistics_array_tiled(
                points,
                self.aad_tile_policy.resolved_capacity(),
                |point| {
                    let path_index =
                        PathIndex::new(u64::from(scramble).saturating_mul(points) + point);
                    let shocks = local_vol_rqmc_shocks(&qmc, bridge.as_ref(), scramble, point)?;
                    let primary = self.local_vol_pathwise_values(
                        local_volatility,
                        bump_runtimes.as_ref(),
                        &shocks,
                        path_index,
                    )?;
                    if antithetic {
                        let mate_shocks = shocks.iter().map(|shock| -*shock).collect::<Vec<_>>();
                        let mate = self.local_vol_pathwise_values(
                            local_volatility,
                            bump_runtimes.as_ref(),
                            &mate_shocks,
                            path_index,
                        )?;
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(std::array::from_fn(
                            |component| (primary[component] + mate[component]) * 0.5,
                        ))
                    } else {
                        Ok::<[f64; PATHWISE_COMPONENTS], MonteCarloError>(primary)
                    }
                },
            )?;
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
        let estimate = estimate_from_statistics(
            statistics[PRICE],
            independent_units,
            1.0,
            EstimatorKind::RandomizedQuasiMonteCarlo,
        )?;
        let sampling_variance = statistics[PRICE].moments().sample_variance().ok_or(
            MonteCarloError::InsufficientSamplingUnits {
                count: independent_units,
            },
        )?;
        let estimator_variance = sampling_variance / independent_units as f64;
        let pricing_result = PricingResult {
            value: estimate,
            risks: self.build_risk_report(
                &statistics,
                independent_units,
                EstimatorKind::RandomizedQuasiMonteCarlo,
                None,
            )?,
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
            risk_diagnostics: self.build_local_vol_risk_diagnostics(),
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

impl SimulationPlan {
    pub(in crate::engine) fn execute_local_vol_rqmc_with_vega_kt(
        &self,
        engine: RqmcConfig,
        local_volatility: &LocalVolRuntime,
        bump_runtimes: Option<&LocalVolBumpRuntimes>,
        qmc: &RqmcPlan,
        bridge: Option<&BrownianBridgePlan>,
    ) -> Result<MonteCarloPrice, MonteCarloError> {
        let vega_kt = local_volatility
            .vega_kt
            .as_ref()
            .ok_or(MonteCarloError::MissingLocalVolatilityReportingBasis)?;
        let antithetic = engine.variance_reduction().antithetic();
        let points = engine.points_per_scramble().get();
        let bucket_count = vega_kt.basis.bucket_count();
        let mut replicate_values: Vec<[f64; PATHWISE_COMPONENTS]> = Vec::with_capacity(
            usize::try_from(engine.scramble_count().get()).expect("u32 fits usize"),
        );
        let mut price_samples = Vec::with_capacity(replicate_values.capacity());
        let mut raw_bucket_samples = Vec::with_capacity(bucket_sample_capacity(
            replicate_values.capacity(),
            bucket_count,
        ));

        for scramble in 0..engine.scramble_count().get() {
            let mut component_sums = [pricing_numerics::NeumaierSum::new(); PATHWISE_COMPONENTS];
            let mut bucket_sums = vec![pricing_numerics::NeumaierSum::new(); bucket_count];
            for point in 0..points {
                let path_index = PathIndex::new(u64::from(scramble).saturating_mul(points) + point);
                let shocks = local_vol_rqmc_shocks(qmc, bridge, scramble, point)?;
                let primary = self.local_vol_pathwise_values_and_buckets(
                    local_volatility,
                    bump_runtimes,
                    &shocks,
                    path_index,
                )?;
                let pathwise = if antithetic {
                    let mate_shocks = shocks.iter().map(|shock| -*shock).collect::<Vec<_>>();
                    let mate = self.local_vol_pathwise_values_and_buckets(
                        local_volatility,
                        bump_runtimes,
                        &mate_shocks,
                        path_index,
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
            let replicate =
                std::array::from_fn(|component| component_sums[component].total() / points as f64);
            price_samples.push(replicate[PRICE]);
            raw_bucket_samples.extend(
                bucket_sums
                    .into_iter()
                    .map(|sum| sum.total() / points as f64),
            );
            replicate_values.push(replicate);
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
        let estimates =
            vega_kt_bucket_estimates(&price_samples, &raw_bucket_samples, bucket_count)?;
        let raw_bucket_means = estimates
            .iter()
            .map(|estimate| estimate.raw_mean())
            .collect::<Vec<_>>();
        let mean_vega = statistics[VEGA].sum().total() / independent_units as f64;
        let bucket_sum = raw_bucket_means
            .iter()
            .copied()
            .collect::<pricing_numerics::NeumaierSum>()
            .total();
        let projection = vega_kt_projection_from_parts(
            raw_bucket_means,
            mean_vega - bucket_sum,
            mean_vega,
            Default::default(),
        )?;
        let full_bucket_covariance = if vega_kt.full_bucket_covariance {
            Some(vega_kt_full_bucket_covariance(
                &raw_bucket_samples,
                bucket_count,
            )?)
        } else {
            None
        };
        let density_row = vega_kt
            .density_rows
            .last()
            .ok_or(MonteCarloError::MismatchedLocalVolatilityReportingBasis)?;
        let vega_kt = VegaKtResult::try_from(&vega_kt_report(
            &vega_kt.basis,
            density_row,
            projection,
            estimates,
            full_bucket_covariance,
        )?)?;
        let estimate = estimate_from_statistics(
            statistics[PRICE],
            independent_units,
            1.0,
            EstimatorKind::RandomizedQuasiMonteCarlo,
        )?;
        let sampling_variance = statistics[PRICE].moments().sample_variance().ok_or(
            MonteCarloError::InsufficientSamplingUnits {
                count: independent_units,
            },
        )?;
        let estimator_variance = sampling_variance / independent_units as f64;
        let pricing_result = PricingResult {
            value: estimate,
            risks: self.build_risk_report(
                &statistics,
                independent_units,
                EstimatorKind::RandomizedQuasiMonteCarlo,
                Some(vega_kt),
            )?,
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
            risk_diagnostics: self.build_local_vol_risk_diagnostics(),
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
