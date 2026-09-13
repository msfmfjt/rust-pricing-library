//! compile / simulation implementation.

use crate::api::mc_result::DEFAULT_VALIDATION_RELATIVE_SPOT_BUMP;
use crate::api::mc_result::DEFAULT_VALIDATION_VOLATILITY_BUMP;
use crate::api::mc_result::PathStateDiagnostics;
use crate::core::{DayCountConvention, PositiveF64};
use crate::engine::aad::{AadTilePolicy, CheckpointPolicy};
use crate::engine::compile::local_vol::compile_local_vol_runtime;
use crate::engine::plan::simulation::ContinuousBarrierRuntime;
use crate::engine::plan::simulation::EarlyExerciseRuntime;
use crate::engine::plan::simulation::LocalVolRuntimeInputs;
use crate::engine::plan::simulation::PLAN_FINGERPRINT_VERSION;
use crate::engine::plan::simulation::SimulationPlan;
use crate::engine::plan::simulation::resolve_spot_bump;
use crate::market::{AffineDividendCoordinate, AffineDividendTransform, DiscountCurve};
use crate::mc::{BarrierBridgeError, EngineConfig, ExecutionPolicy};
use crate::models::ModelSpec;
use crate::product::{
    BarrierDirection, BarrierMonitoring, CompactC2Smoothing, GraphFingerprint, GraphLimitPolicy,
    ProductSpec,
};
use crate::risk::PayoffSmoothing;
use crate::{
    Fingerprint, MigrationProvenance, MonteCarloError, PricingRequest, fingerprint_request,
};

impl SimulationPlan {
    /// The explicit LSV/HW adapters reuse fixed observation payoff graphs.
    /// Exercise policies and continuous barrier bridges need their own model
    /// integration; accepting those requests here would change the payoff.
    pub(crate) fn compile_hybrid_base(
        request: &PricingRequest,
        execution_policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        if matches!(request.product(), ProductSpec::AmericanVanilla(_)) {
            return Err(MonteCarloError::UnsupportedModel {
                model: "early exercise in LSV/Hull-White adapters",
            });
        }
        if matches!(request.product(), ProductSpec::Barrier(b) if b.monitoring() == BarrierMonitoring::Continuous)
        {
            return Err(MonteCarloError::UnsupportedModel {
                model: "continuous Barrier in LSV/Hull-White adapters",
            });
        }
        if request.risk().payoff_smoothing_width_ladder().is_some() {
            return Err(MonteCarloError::UnsupportedRiskForModel {
                model: "smoothing width ladder in LSV/Hull-White adapters",
            });
        }
        Self::compile(request, execution_policy)
    }
}

impl SimulationPlan {
    pub fn compile(
        request: &PricingRequest,
        execution_policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let engine = request.engine();
        let product = request.product();
        let path_state_diagnostics =
            PathStateDiagnostics::from_product(product, request.valuation_date());
        let continuous_barrier_spec = match product {
            ProductSpec::Barrier(barrier)
                if barrier.monitoring() == BarrierMonitoring::Continuous =>
            {
                Some(barrier)
            }
            _ => None,
        };
        let market_forward = request.market().equity().forward();
        let dividend_timeline = market_forward
            .discrete_dividends()
            .map(AffineDividendTransform::event_timeline)
            .transpose()?
            .map_or_else(Vec::new, |entries| entries.into_vec());
        let jump_dates = match product {
            ProductSpec::Barrier(barrier) => barrier
                .monitoring_dates()
                .iter()
                .copied()
                .filter(|date| {
                    let time =
                        DayCountConvention::Act365F.year_fraction(request.valuation_date(), *date);
                    dividend_timeline
                        .iter()
                        .any(|entry| entry.ex_time().to_bits() == time.to_bits())
                })
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        let payoff_graph = match (product, request.risk().payoff_smoothing()) {
            (ProductSpec::Digital(digital), Some(PayoffSmoothing::CompactC2 { half_width })) => {
                digital.smoothed_source_graph(CompactC2Smoothing::from_positive(half_width))?
            }
            (ProductSpec::Barrier(barrier), Some(PayoffSmoothing::CompactC2 { half_width })) => {
                barrier.smoothed_source_graph_with_dividend_jumps(
                    CompactC2Smoothing::from_positive(half_width),
                    &jump_dates,
                )?
            }
            (ProductSpec::Barrier(barrier), None) => {
                barrier.source_graph_with_dividend_jumps(&jump_dates)?
            }
            _ => product.source_graph(request.valuation_date())?,
        };
        let payoff = payoff_graph.compile(GraphLimitPolicy::DEFAULT)?;
        let observations = payoff.terminal_observations();
        for (underlying, _) in &observations {
            if *underlying != product.underlying() {
                return Err(MonteCarloError::UnsupportedObservationUnderlying {
                    product: *underlying,
                    market: product.underlying(),
                });
            }
        }
        let contractual_observation_dates = observations
            .iter()
            .map(|(_, date)| *date)
            .collect::<Vec<_>>();
        let mut observation_points = contractual_observation_dates
            .iter()
            .map(|date| {
                (
                    DayCountConvention::Act365F.year_fraction(request.valuation_date(), *date),
                    Some(*date),
                )
            })
            .collect::<Vec<_>>();
        if let Some(barrier) = continuous_barrier_spec {
            let monitoring_end = DayCountConvention::Act365F.year_fraction(
                request.valuation_date(),
                *barrier
                    .monitoring_dates()
                    .last()
                    .expect("Barrier monitoring dates are non-empty"),
            );
            observation_points
                .retain(|(time, date)| *time > 0.0 || *date == Some(barrier.expiry()));
            for entry in &dividend_timeline {
                let time = entry.ex_time();
                if time > 0.0
                    && time <= monitoring_end
                    && !observation_points
                        .iter()
                        .any(|(point_time, _)| point_time.to_bits() == time.to_bits())
                {
                    observation_points.push((time, None));
                }
            }
            observation_points.sort_by(|left, right| {
                left.0
                    .partial_cmp(&right.0)
                    .expect("observation times are finite")
            });
        }
        let observation_times = observation_points
            .iter()
            .map(|(time, _)| *time)
            .collect::<Vec<_>>();
        let observation_dates = observation_points
            .iter()
            .map(|(_, date)| *date)
            .collect::<Vec<_>>();
        let time = if observations.is_empty() {
            0.0
        } else {
            DayCountConvention::Act365F.year_fraction(request.valuation_date(), product.expiry())
        };
        let payment_time = DayCountConvention::Act365F
            .year_fraction(request.valuation_date(), product.payment_date());
        let forward_evaluation = market_forward.evaluate(time)?;
        let discount_evaluation = market_forward.discount_curve().evaluate(payment_time)?;
        let discount = discount_evaluation.discount;
        let spot = market_forward.spot().get();
        let (volatility, total_variance, local_volatility) = match request.model() {
            ModelSpec::BlackScholes(model) => {
                let volatility = model.volatility().get();
                (volatility, volatility * volatility * time, None)
            }
            ModelSpec::Black76(model) => {
                let volatility = model.volatility().get();
                (volatility, volatility * volatility * time, None)
            }
            ModelSpec::LocalVolatility(model) => {
                let runtime = compile_local_vol_runtime(LocalVolRuntimeInputs {
                    grid: model.local_variance_grid().clone(),
                    reporting_iv_basis: model.reporting_iv_basis(),
                    vega_kt: request.risk().vega_kt(),
                    market_forward,
                    valuation_date: request.valuation_date(),
                    expiry_time: time,
                    event_times: &observation_times,
                })?;
                (0.0, runtime.maximum_total_variance(), Some(runtime))
            }
        };
        if !total_variance.is_finite() {
            return Err(MonteCarloError::NonFiniteTotalVariance {
                bits: total_variance.to_bits(),
            });
        }
        if total_variance > 0.0 {
            let independent_units = match engine {
                EngineConfig::PseudoMonteCarlo(config) => config.independent_sampling_units().get(),
                EngineConfig::RandomizedQuasiMonteCarlo(config) => {
                    u64::from(config.scramble_count().get())
                }
            };
            if independent_units < 2 {
                return Err(MonteCarloError::InsufficientSamplingUnits {
                    count: independent_units,
                });
            }
        }
        let pre_dividend_observations = payoff.pre_dividend_observations();
        let mut observation_forwards = Vec::with_capacity(observation_dates.len());
        let mut observation_affine_coordinates = Vec::with_capacity(observation_dates.len());
        let mut observation_pre_dividend_coordinates = Vec::with_capacity(observation_dates.len());
        for (&date, &observation_time) in observation_dates.iter().zip(&observation_times) {
            let evaluation = market_forward.evaluate(observation_time)?;
            observation_forwards.push(evaluation.forward);
            observation_affine_coordinates.push(evaluation.affine_coordinate);
            let dividend_entry = dividend_timeline
                .iter()
                .find(|entry| entry.ex_time().to_bits() == observation_time.to_bits());
            let needs_pre_dividend = date.is_some_and(|date| {
                pre_dividend_observations.contains(&(product.underlying(), date))
            });
            observation_pre_dividend_coordinates.push(
                dividend_entry
                    .filter(|_| continuous_barrier_spec.is_some() || needs_pre_dividend)
                    .map(|entry| entry.before()),
            );
        }
        if let (Some(barrier), Some(PayoffSmoothing::CompactC2 { half_width })) =
            (continuous_barrier_spec, request.risk().payoff_smoothing())
            && barrier.direction() == BarrierDirection::Up
        {
            let invalid_coordinate = std::iter::once(AffineDividendCoordinate::identity())
                .chain(observation_affine_coordinates.iter().copied())
                .chain(
                    observation_pre_dividend_coordinates
                        .iter()
                        .flatten()
                        .copied(),
                )
                .find_map(|coordinate| {
                    let transformed_spot_barrier = barrier.barrier().get() - coordinate.a() * spot;
                    (half_width.get() >= transformed_spot_barrier)
                        .then_some(transformed_spot_barrier)
                });
            if let Some(transformed_spot_barrier) = invalid_coordinate {
                return Err(BarrierBridgeError::InvalidSmoothedSafeDistance {
                    safe_distance_bits: half_width.get().to_bits(),
                    transformed_spot_barrier_bits: transformed_spot_barrier.to_bits(),
                }
                .into());
            }
        }
        let observation_local_vol_node_indices = local_volatility.as_ref().map(|runtime| {
            observation_times
                .iter()
                .map(|time| {
                    runtime
                        .plan
                        .time_grid()
                        .node_index_for_time(*time)
                        .expect("contractual observations are Local Volatility event nodes")
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        });
        let continuous_barrier = continuous_barrier_spec.map(|barrier| {
            let monitoring_end = DayCountConvention::Act365F.year_fraction(
                request.valuation_date(),
                *barrier
                    .monitoring_dates()
                    .last()
                    .expect("Barrier monitoring dates are non-empty"),
            );
            let bridge_observation_indices = observation_times
                .iter()
                .enumerate()
                .filter_map(|(index, time)| (*time <= monitoring_end).then_some(index))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let expiry_observation_index = observation_dates
                .iter()
                .position(|date| *date == Some(barrier.expiry()))
                .expect("Barrier graph retains expiry");
            ContinuousBarrierRuntime {
                strike: barrier.strike().get(),
                barrier: barrier.barrier().get(),
                notional: barrier.notional().get(),
                rebate: barrier.rebate().map_or(0.0, PositiveF64::get),
                side: barrier.side(),
                direction: barrier.direction(),
                style: barrier.style(),
                monitoring_end_time: monitoring_end,
                bridge_observation_indices,
                expiry_observation_index,
            }
        });
        let early_exercise = match product {
            ProductSpec::AmericanVanilla(american) => {
                let config = request
                    .lsm()
                    .expect("PricingRequest validates American LSM configuration")
                    .clone();
                let mut observation_indices = Vec::with_capacity(american.exercise_dates().len());
                let mut discount_factors = Vec::with_capacity(american.exercise_dates().len());
                let mut dividend_collisions = Vec::with_capacity(american.exercise_dates().len());
                for &date in american.exercise_dates() {
                    let observation_index = observation_dates
                        .iter()
                        .position(|candidate| *candidate == Some(date))
                        .expect("American payoff graph retains every exercise date");
                    let exercise_time =
                        DayCountConvention::Act365F.year_fraction(request.valuation_date(), date);
                    observation_indices.push(observation_index);
                    discount_factors.push(
                        market_forward
                            .discount_curve()
                            .evaluate(exercise_time)?
                            .discount,
                    );
                    dividend_collisions.push(
                        dividend_timeline
                            .iter()
                            .any(|entry| entry.ex_time().to_bits() == exercise_time.to_bits()),
                    );
                }
                Some(EarlyExerciseRuntime {
                    config,
                    exercise_dates: american.exercise_dates().into(),
                    observation_indices: observation_indices.into_boxed_slice(),
                    discount_factors: discount_factors.into_boxed_slice(),
                    dividend_collisions: dividend_collisions.into_boxed_slice(),
                })
            }
            _ => None,
        };
        let request_fingerprint = *fingerprint_request(request)?.as_bytes();
        let request_migration = request
            .wire_migration()
            .cloned()
            .unwrap_or_else(|| MigrationProvenance::current(request_fingerprint));
        let aad_tile_policy = AadTilePolicy::resolve(
            execution_policy.reduction_block_size().get(),
            request.risk().aad_tile_capacity(),
        )?;
        let checkpoint_policy = CheckpointPolicy::resolve(request.risk().checkpoint_interval());
        let plan_fingerprint = build_plan_fingerprint(
            request_fingerprint,
            payoff.tape_fingerprint(),
            execution_policy,
            aad_tile_policy,
            checkpoint_policy,
        );
        if let Some(gamma) = request.risk().gamma() {
            let bump = resolve_spot_bump(gamma, spot);
            if !bump.is_finite() || bump >= spot {
                return Err(MonteCarloError::InvalidGammaBump {
                    spot_bits: spot.to_bits(),
                    bump_bits: bump.to_bits(),
                });
            }
        }
        let validation_spot_bump = request
            .risk()
            .gamma()
            .map_or(spot * DEFAULT_VALIDATION_RELATIVE_SPOT_BUMP, |gamma| {
                resolve_spot_bump(gamma, spot)
            });
        let validation_volatility_bump = if volatility == 0.0 {
            0.0
        } else {
            DEFAULT_VALIDATION_VOLATILITY_BUMP.min(volatility * 0.5)
        };
        let (payoff_smoothing_endpoint_count, payoff_smoothing_dividend_jump_count) =
            match (&continuous_barrier, &local_volatility) {
                (Some(barrier), Some(local_volatility)) => {
                    let node_count = local_volatility
                        .plan
                        .time_grid()
                        .nodes()
                        .iter()
                        .take_while(|time| **time <= barrier.monitoring_end_time)
                        .count();
                    let jump_count = local_volatility
                        .node_pre_dividend_coordinates
                        .iter()
                        .take(node_count)
                        .filter(|coordinate| coordinate.is_some())
                        .count();
                    (
                        u32::try_from(node_count.saturating_sub(jump_count)).unwrap_or(u32::MAX),
                        u32::try_from(jump_count).unwrap_or(u32::MAX),
                    )
                }
                (Some(barrier), None) => {
                    let jump_count = barrier
                        .bridge_observation_indices
                        .iter()
                        .filter(|index| observation_pre_dividend_coordinates[**index].is_some())
                        .count();
                    (
                        u32::try_from(
                            1_usize
                                .saturating_add(barrier.bridge_observation_indices.len())
                                .saturating_sub(jump_count),
                        )
                        .unwrap_or(u32::MAX),
                        u32::try_from(jump_count).unwrap_or(u32::MAX),
                    )
                }
                (None, _) => match product {
                    ProductSpec::Digital(_) => (1, 0),
                    ProductSpec::Barrier(barrier) => (
                        u32::try_from(barrier.monitoring_dates().len()).unwrap_or(u32::MAX),
                        u32::try_from(jump_dates.len()).unwrap_or(u32::MAX),
                    ),
                    _ => (0, 0),
                },
            };
        Ok(Self {
            valuation_date: request.valuation_date(),
            expiry: product.expiry(),
            underlying: product.underlying(),
            time,
            forward: forward_evaluation.forward,
            spot,
            discount,
            volatility,
            total_variance,
            payoff,
            observation_dates: observation_dates.into_boxed_slice(),
            observation_times: observation_times.into_boxed_slice(),
            observation_forwards: observation_forwards.into_boxed_slice(),
            observation_affine_coordinates: observation_affine_coordinates.into_boxed_slice(),
            observation_pre_dividend_coordinates: observation_pre_dividend_coordinates
                .into_boxed_slice(),
            observation_local_vol_node_indices,
            engine,
            execution_policy,
            aad_tile_policy,
            checkpoint_policy,
            request_delta: request.risk().delta(),
            request_gamma: request.risk().gamma(),
            request_vega: request.risk().vega(),
            payoff_smoothing: request.risk().payoff_smoothing(),
            payoff_smoothing_endpoint_count,
            payoff_smoothing_dividend_jump_count,
            path_state_diagnostics,
            smile_dynamics: request.risk().smile_dynamics(),
            validation_spot_bump,
            validation_volatility_bump,
            request_fingerprint,
            request_migration,
            plan_fingerprint,
            discount_region: discount_evaluation.region,
            dividend_region: forward_evaluation.dividend_region,
            market_forward: market_forward.clone(),
            local_volatility,
            continuous_barrier,
            early_exercise,
        })
    }
}

pub(in crate::engine) fn build_plan_fingerprint(
    request_fingerprint: [u8; 32],
    payoff_fingerprint: GraphFingerprint,
    execution_policy: ExecutionPolicy,
    aad_tile_policy: AadTilePolicy,
    checkpoint_policy: CheckpointPolicy,
) -> Fingerprint {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"pricing/plan\0");
    hasher.update(&PLAN_FINGERPRINT_VERSION.to_be_bytes());
    hasher.update(&request_fingerprint);
    hasher.update(payoff_fingerprint.as_bytes());
    hasher.update(&execution_policy.version().to_be_bytes());
    hasher.update(&execution_policy.worker_threads().get().to_be_bytes());
    hasher.update(&execution_policy.reduction_block_size().get().to_be_bytes());
    hasher.update(&aad_tile_policy.version().to_be_bytes());
    hasher.update(&aad_tile_policy.resolved_capacity().get().to_be_bytes());
    hasher.update(&checkpoint_policy.version().to_be_bytes());
    hasher.update(&checkpoint_policy.resolved_interval().get().to_be_bytes());
    Fingerprint::from_bytes(*hasher.finalize().as_bytes())
}
