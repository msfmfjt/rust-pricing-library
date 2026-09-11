use std::path::Path;
use std::sync::Arc;

use pricing::core::{CurrencyId, CurveId, PositiveF64, UnderlyingId};
use pricing::market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
use pricing::mc::{EngineConfig, ExecutionPolicy, PseudoMcConfig, RqmcConfig, VarianceReduction};
use pricing::models::{BlackScholesSpec, ModelSpec};
use pricing::product::{
    ArithmeticAsianSpec, AsianObservation, BarrierDirection, BarrierMonitoring, BarrierSpec,
    BarrierStyle, DigitalPayout, DigitalSpec, FixedLookbackSpec, OptionSide, ProductSpec,
};
use pricing::risk::{GammaConfig, PayoffSmoothing, RiskRequest, SmileDynamics, SpotBump};
use pricing::{
    BarrierHitIndicatorMode, Estimate, MonteCarloPrice, PathStateDiagnostics,
    PayoffSmoothingKernel, PayoffSmoothingWidthUnit, PayoffValuationKind, PricingPlan,
    PricingRequest, RiskMethod, RiskValidation, request_to_json, result_to_json,
};
use serde_json::{Value, json};

const PSEUDO_UNITS: u64 = 4_096;
const RQMC_POINTS: u64 = 512;
const RQMC_SCRAMBLES: u32 = 8;

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .expect("usage: replay_path_dependence <output.json>");
    let policy = ExecutionPolicy::new(2, Some(256)).expect("execution policy");
    let cases = [
        capture(
            "digital_exact_price_only",
            request(digital(), pseudo_engine(), price_only()),
            policy,
        ),
        capture(
            "digital_smoothed_full_risk",
            request(digital(), rqmc_engine(), smoothed_risk(2.0)),
            policy,
        ),
        capture(
            "discrete_barrier_smoothed_full_risk",
            request(discrete_barrier(), pseudo_engine(), smoothed_risk(3.0)),
            policy,
        ),
        capture(
            "continuous_barrier_smoothed_full_risk",
            request(continuous_barrier(), rqmc_engine(), smoothed_risk(2.5)),
            policy,
        ),
        capture(
            "arithmetic_asian_full_risk",
            request(arithmetic_asian(), pseudo_engine(), full_risk()),
            policy,
        ),
        capture(
            "fixed_lookback_full_risk",
            request(fixed_lookback(), rqmc_engine(), full_risk()),
            policy,
        ),
    ];
    let fixture = json!({
        "schema_version": 1,
        "fixture_kind": "path_dependence_replay",
        "platform": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        "cases": cases
    });
    std::fs::write(
        Path::new(&output),
        serde_json::to_string_pretty(&fixture).expect("serialize replay fixture") + "\n",
    )
    .expect("write replay fixture");
}

fn capture(name: &str, request: PricingRequest, policy: ExecutionPolicy) -> Value {
    let request_json: Value = serde_json::from_str(
        &request_to_json(&request).expect("serialize normalized replay request"),
    )
    .expect("request JSON value");
    let plan = PricingPlan::compile(&request, policy).expect("compile replay plan");
    let request_fingerprint = plan.request_fingerprint().to_string();
    let plan_fingerprint = plan.plan_fingerprint().to_string();
    let output = plan.evaluate().expect("evaluate replay plan");
    let result_json: Value = serde_json::from_str(
        &result_to_json(&output.pricing_result).expect("serialize replay result"),
    )
    .expect("result JSON value");
    json!({
        "name": name,
        "request": request_json,
        "plan": {
            "request_fingerprint": request_fingerprint,
            "plan_fingerprint": plan_fingerprint,
            "worker_threads": policy.worker_threads().get(),
            "reduction_block_size": policy.reduction_block_size().get()
        },
        "result": result_json,
        "execution": execution_evidence(&output),
        "path_diagnostics": path_diagnostics(&output)
    })
}

fn execution_evidence(output: &MonteCarloPrice) -> Value {
    let diagnostics = output.diagnostics;
    let methods = output.risk_diagnostics.methods;
    json!({
        "sampling_variance_bits": f64_bits(output.sampling_variance),
        "estimator_variance_bits": f64_bits(output.estimator_variance),
        "independent_sampling_units": output.independent_sampling_units,
        "evaluated_paths": output.evaluated_paths.to_string(),
        "monte_carlo": {
            "master_seed": diagnostics.master_seed.to_string(),
            "estimator": estimator_name(diagnostics.estimator),
            "scramble_count": diagnostics.scramble_count,
            "direction_checksum": diagnostics.direction_checksum.map(|value| hex(&value)),
            "scramble_checksum": diagnostics.scramble_checksum.map(|value| hex(&value)),
            "policy_version": diagnostics.policy_version,
            "worker_threads": diagnostics.worker_threads,
            "reduction_block_size": diagnostics.reduction_block_size,
            "aad_tile_policy_version": diagnostics.aad_tile_policy_version,
            "aad_tile_capacity": diagnostics.aad_tile_capacity,
            "checkpoint_policy_version": diagnostics.checkpoint_policy_version,
            "checkpoint_interval": diagnostics.checkpoint_interval,
            "antithetic": diagnostics.antithetic,
            "discount_region": curve_region_name(diagnostics.discount_region),
            "dividend_region": curve_region_name(diagnostics.dividend_region),
            "payoff_fingerprint": diagnostics.payoff_fingerprint.to_string()
        },
        "risk_methods": {
            "delta": methods.delta.map(risk_method_name),
            "gamma": methods.gamma.map(risk_method_name),
            "vega": methods.vega.map(risk_method_name),
            "smile_dynamics": smile_dynamics_name(methods.smile_dynamics),
            "gamma_spot_bump_bits": methods.gamma_spot_bump.map(f64_bits),
            "validation_spot_bump_bits": methods.validation_spot_bump.map(f64_bits),
            "validation_volatility_bump_bits": methods.validation_volatility_bump.map(f64_bits),
            "bump_policy_version": methods.bump_policy_version
        },
        "risk_validation": {
            "delta": validation_evidence(output.risk_diagnostics.delta_validation),
            "gamma": validation_evidence(output.risk_diagnostics.gamma_validation),
            "vega": validation_evidence(output.risk_diagnostics.vega_validation)
        }
    })
}

fn path_diagnostics(output: &MonteCarloPrice) -> Value {
    let diagnostics = output.diagnostics;
    json!({
        "valuation_kind": match diagnostics.valuation_kind {
            PayoffValuationKind::ExactContractual => "exact_contractual",
            PayoffValuationKind::SmoothedSurrogate => "smoothed_surrogate",
        },
        "payoff_smoothing": diagnostics.payoff_smoothing.map(|smoothing| json!({
            "kernel": match smoothing.kernel {
                PayoffSmoothingKernel::CompactC2 => "compact_c2",
            },
            "policy_version": smoothing.policy_version,
            "half_width_bits": f64_bits(smoothing.half_width.get()),
            "full_transition_width_bits": f64_bits(smoothing.full_transition_width.get()),
            "width_unit": match smoothing.width_unit {
                PayoffSmoothingWidthUnit::Spot => "spot",
            },
            "price_and_greeks_share_payoff": smoothing.price_and_greeks_share_payoff,
            "endpoint_count": smoothing.endpoint_count,
            "dividend_jump_count": smoothing.dividend_jump_count,
        })),
        "path_state": diagnostics.path_state.map(|state| match state {
            PathStateDiagnostics::ArithmeticAsian {
                known_observation_count,
                unknown_observation_count,
                known_weight_sum,
                unknown_weight_sum,
                weighted_known_fixing_sum,
            } => json!({
                "kind": "arithmetic_asian",
                "known_observation_count": known_observation_count,
                "unknown_observation_count": unknown_observation_count,
                "known_weight_sum_bits": f64_bits(known_weight_sum),
                "unknown_weight_sum_bits": f64_bits(unknown_weight_sum),
                "weighted_known_fixing_sum_bits": f64_bits(weighted_known_fixing_sum),
            }),
            PathStateDiagnostics::FixedLookback {
                past_monitoring_count,
                future_monitoring_count,
                historical_extremum,
            } => json!({
                "kind": "fixed_lookback",
                "past_monitoring_count": past_monitoring_count,
                "future_monitoring_count": future_monitoring_count,
                "historical_extremum_bits": historical_extremum.map(f64_bits),
            }),
        }),
        "barrier_bridge": diagnostics.barrier_bridge.map(|bridge| json!({
            "abi": bridge.abi,
            "policy_version": bridge.policy_version,
            "indicator_mode": match bridge.indicator_mode {
                BarrierHitIndicatorMode::Exact => "exact",
                BarrierHitIndicatorMode::CompactC2 => "compact_c2",
            },
            "endpoint_hit_fraction_bits": f64_bits(bridge.endpoint_hit_fraction),
            "dividend_jump_hit_fraction_bits": f64_bits(bridge.dividend_jump_hit_fraction),
            "mean_conditional_bridge_hit_weight_bits":
                f64_bits(bridge.mean_conditional_bridge_hit_weight),
            "mean_interval_count_bits": f64_bits(bridge.mean_interval_count),
            "mean_finite_correction_count_bits": f64_bits(bridge.mean_finite_correction_count),
            "mean_zero_variance_count_bits": f64_bits(bridge.mean_zero_variance_count),
            "mean_survival_underflow_count_bits":
                f64_bits(bridge.mean_survival_underflow_count),
            "mean_certain_survival_count_bits":
                f64_bits(bridge.mean_certain_survival_count),
        })),
    })
}

fn validation_evidence(validation: Option<RiskValidation>) -> Option<Value> {
    validation.map(|value| {
        json!({
            "bump_and_revalue": estimate_evidence(value.bump_and_revalue),
            "bump_minus_primary": estimate_evidence(value.bump_minus_primary)
        })
    })
}

fn estimate_evidence(estimate: Estimate) -> Value {
    json!({
        "value_bits": f64_bits(estimate.value().get()),
        "standard_error_bits": f64_bits(estimate.standard_error().get()),
        "confidence_lower_bits": f64_bits(estimate.confidence_interval().lower().get()),
        "confidence_upper_bits": f64_bits(estimate.confidence_interval().upper().get()),
        "effective_sampling_units": estimate.effective_sampling_units().get()
    })
}

fn request(product: ProductSpec, engine: EngineConfig, risk: RiskRequest) -> PricingRequest {
    let underlying = UnderlyingId::new(1);
    let currency = CurrencyId::new(1);
    PricingRequest::new(
        "2026-09-04".parse().expect("valuation date"),
        product,
        MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100.0, "spot").expect("spot"),
                curve(1, 0.05),
                curve(2, 0.02),
            ),
        )),
        ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
        engine,
        risk,
    )
    .expect("request")
}

fn digital() -> ProductSpec {
    ProductSpec::Digital(
        DigitalSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(1),
            "2027-09-04".parse().expect("expiry"),
            100.0,
            10.0,
            OptionSide::Call,
            DigitalPayout::Cash,
        )
        .expect("Digital"),
    )
}

fn discrete_barrier() -> ProductSpec {
    barrier(BarrierMonitoring::Discrete)
}

fn continuous_barrier() -> ProductSpec {
    barrier(BarrierMonitoring::Continuous)
}

fn barrier(monitoring: BarrierMonitoring) -> ProductSpec {
    ProductSpec::Barrier(
        BarrierSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(1),
            "2027-09-04".parse().expect("expiry"),
            100.0,
            125.0,
            1.0,
            OptionSide::Call,
            BarrierDirection::Up,
            BarrierStyle::KnockOut,
            monitoring,
            vec![
                "2027-03-05".parse().expect("first monitoring"),
                "2027-09-04".parse().expect("expiry monitoring"),
            ],
            Some(2.0),
            "2027-09-04".parse().expect("payment"),
        )
        .expect("Barrier"),
    )
}

fn arithmetic_asian() -> ProductSpec {
    ProductSpec::ArithmeticAsian(
        ArithmeticAsianSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(1),
            100.0,
            1.0,
            OptionSide::Call,
            vec![
                AsianObservation::known("2026-06-04".parse().expect("past observation"), 0.2, 98.0)
                    .expect("known observation"),
                AsianObservation::unknown("2027-03-05".parse().expect("future observation"), 0.3)
                    .expect("first unknown observation"),
                AsianObservation::unknown("2027-09-04".parse().expect("expiry observation"), 0.5)
                    .expect("second unknown observation"),
            ],
            "2027-09-04".parse().expect("payment"),
        )
        .expect("Arithmetic Asian"),
    )
}

fn fixed_lookback() -> ProductSpec {
    ProductSpec::FixedLookback(
        FixedLookbackSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(1),
            100.0,
            1.0,
            OptionSide::Call,
            vec![
                "2026-06-04".parse().expect("past monitoring"),
                "2027-03-05".parse().expect("future monitoring"),
                "2027-09-04".parse().expect("expiry monitoring"),
            ],
            Some(105.0),
            "2027-09-04".parse().expect("payment"),
        )
        .expect("Fixed Lookback"),
    )
}

fn price_only() -> RiskRequest {
    RiskRequest::price_only(SmileDynamics::StickyLogMoneyness)
}

fn full_risk() -> RiskRequest {
    RiskRequest::new(
        true,
        Some(GammaConfig::new(
            SpotBump::relative(0.01).expect("Gamma bump"),
        )),
        true,
        None,
        SmileDynamics::StickyLogMoneyness,
        Some(16),
        Some(128),
    )
    .expect("risk")
}

fn smoothed_risk(half_width: f64) -> RiskRequest {
    full_risk()
        .with_payoff_smoothing(PayoffSmoothing::compact_c2(half_width).expect("payoff smoothing"))
}

fn pseudo_engine() -> EngineConfig {
    EngineConfig::PseudoMonteCarlo(
        PseudoMcConfig::new(
            0x3141_5926_5358_9793,
            PSEUDO_UNITS,
            VarianceReduction::new(true, false),
        )
        .expect("Pseudo-MC engine"),
    )
}

fn rqmc_engine() -> EngineConfig {
    EngineConfig::RandomizedQuasiMonteCarlo(
        RqmcConfig::new(
            RQMC_POINTS,
            RQMC_SCRAMBLES,
            0x2718_2818_2845_9045,
            VarianceReduction::new(true, true),
        )
        .expect("RQMC engine"),
    )
}

fn curve(id: u32, rate: f64) -> Arc<LogLinearDiscountCurve> {
    Arc::new(
        LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, (-rate).exp()])
            .expect("curve"),
    )
}

fn f64_bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn hex(value: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in value {
        output.push(char::from(DIGITS[usize::from(byte >> 4)]));
        output.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    output
}

fn estimator_name(value: pricing::EstimatorKind) -> &'static str {
    match value {
        pricing::EstimatorKind::Analytical => "analytical",
        pricing::EstimatorKind::PseudoMonteCarlo => "pseudo_monte_carlo",
        pricing::EstimatorKind::RandomizedQuasiMonteCarlo => "randomized_quasi_monte_carlo",
    }
}

fn risk_method_name(value: RiskMethod) -> &'static str {
    match value {
        RiskMethod::AadReverse => "aad_reverse",
        RiskMethod::CentralBump => "central_bump",
        RiskMethod::CentralBumpOfAadDelta => "central_bump_of_aad_delta",
    }
}

fn smile_dynamics_name(value: SmileDynamics) -> &'static str {
    match value {
        SmileDynamics::StickyLogMoneyness => "sticky_log_moneyness",
        SmileDynamics::StickyStrike => "sticky_strike",
        SmileDynamics::StickyDelta => "sticky_delta",
    }
}

fn curve_region_name(value: pricing::market::CurveRegion) -> &'static str {
    match value {
        pricing::market::CurveRegion::Pillar => "pillar",
        pricing::market::CurveRegion::Interpolated => "interpolated",
        pricing::market::CurveRegion::RightExtrapolated => "right_extrapolated",
    }
}
