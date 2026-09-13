use std::path::Path;
use std::sync::Arc;

use pricing::core::{CurrencyId, CurveId, PositiveF64, UnderlyingId};
use pricing::market::{
    CurveRegion, EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext,
};
use pricing::mc::{
    CpqrConfig, EngineConfig, ExecutionPolicy, LsmConfig, LsmStateVariable, PolynomialBasisSpec,
    PseudoMcConfig, RqmcConfig, VarianceReduction,
};
use pricing::models::{BlackScholesSpec, ModelSpec};
use pricing::product::{AmericanVanillaSpec, OptionSide, ProductSpec};
use pricing::risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump};
use pricing::{
    Estimate, EstimatorKind, MonteCarloPrice, PricingPlan, PricingRequest, RiskMethod,
    RiskValidation, monte_carlo_result_to_json, request_to_json,
};
use serde_json::{Value, json};

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .expect("usage: replay_early_exercise <output.json>");
    let policy = ExecutionPolicy::new(2, Some(256)).expect("execution policy");
    let cases = [
        capture(
            "pseudo_mc_price_only",
            request(
                pseudo_engine(0x1111, 128),
                pseudo_engine(0x2222, 256),
                price_only(),
            ),
            policy,
        ),
        capture(
            "rqmc_price_only",
            request(
                rqmc_engine(0x3333, 64, 4),
                rqmc_engine(0x4444, 128, 4),
                price_only(),
            ),
            policy,
        ),
        capture(
            "pseudo_mc_fixed_policy_risk",
            request(
                pseudo_engine(0x1111, 128),
                pseudo_engine(0x2222, 256),
                full_risk(),
            ),
            policy,
        ),
        capture(
            "rqmc_fixed_policy_risk",
            request(
                rqmc_engine(0x3333, 64, 4),
                rqmc_engine(0x4444, 128, 4),
                full_risk(),
            ),
            policy,
        ),
    ];
    let fixture = json!({
        "schema_version": 1,
        "fixture_kind": "early_exercise_replay",
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
        &monte_carlo_result_to_json(&output).expect("serialize complete replay result"),
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
        "execution": execution_evidence(&output)
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

fn request(
    training_engine: EngineConfig,
    valuation_engine: EngineConfig,
    risk: RiskRequest,
) -> PricingRequest {
    let underlying = UnderlyingId::new(1);
    let currency = CurrencyId::new(1);
    let exercise_dates = ["2026-12-04", "2027-03-04", "2027-06-04", "2027-09-04"]
        .map(|value| value.parse().expect("exercise date"));
    let lsm = LsmConfig::new(
        training_engine,
        vec![LsmStateVariable::Spot],
        PolynomialBasisSpec::new(1, 3, 8, 8).expect("basis"),
        0.0,
        CpqrConfig::new(1.0e-14, 1.0e-12).expect("CPQR config"),
        1_000_000,
    )
    .expect("LSM config");
    PricingRequest::new_with_lsm(
        "2026-09-04".parse().expect("valuation date"),
        ProductSpec::AmericanVanilla(
            AmericanVanillaSpec::new(
                underlying,
                currency,
                *exercise_dates.last().expect("exercise dates"),
                100.0,
                1.0,
                OptionSide::Put,
                exercise_dates.to_vec(),
            )
            .expect("American product"),
        ),
        MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100.0, "spot").expect("spot"),
                curve(1, 0.05),
                curve(2, 0.0),
            ),
        )),
        ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).expect("model")),
        valuation_engine,
        risk,
        Some(lsm),
    )
    .expect("American request")
}

fn curve(id: u32, rate: f64) -> Arc<LogLinearDiscountCurve> {
    Arc::new(
        LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, (-rate).exp()])
            .expect("curve"),
    )
}

fn pseudo_engine(seed: u64, units: u64) -> EngineConfig {
    EngineConfig::PseudoMonteCarlo(
        PseudoMcConfig::new(seed, units, VarianceReduction::new(true, false))
            .expect("Pseudo-MC engine"),
    )
}

fn rqmc_engine(seed: u64, points: u64, scrambles: u32) -> EngineConfig {
    EngineConfig::RandomizedQuasiMonteCarlo(
        RqmcConfig::new(points, scrambles, seed, VarianceReduction::new(true, true))
            .expect("RQMC engine"),
    )
}

fn price_only() -> RiskRequest {
    RiskRequest::price_only(SmileDynamics::StickyLogMoneyness)
}

fn full_risk() -> RiskRequest {
    RiskRequest::new(
        true,
        Some(GammaConfig::new(
            SpotBump::relative(0.01).expect("gamma bump"),
        )),
        true,
        None,
        SmileDynamics::StickyLogMoneyness,
        Some(13),
        Some(64),
    )
    .expect("risk request")
}

fn f64_bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

const fn estimator_name(estimator: EstimatorKind) -> &'static str {
    match estimator {
        EstimatorKind::PseudoMonteCarlo => "pseudo_monte_carlo",
        EstimatorKind::RandomizedQuasiMonteCarlo => "randomized_quasi_monte_carlo",
        EstimatorKind::Analytical => "analytical",
    }
}

const fn risk_method_name(method: RiskMethod) -> &'static str {
    match method {
        RiskMethod::AadReverse => "aad_reverse",
        RiskMethod::CentralBump => "central_bump",
        RiskMethod::CentralBumpOfAadDelta => "central_bump_of_aad_delta",
    }
}

const fn smile_dynamics_name(dynamics: SmileDynamics) -> &'static str {
    match dynamics {
        SmileDynamics::StickyLogMoneyness => "sticky_log_moneyness",
        SmileDynamics::StickyStrike => "sticky_strike",
        SmileDynamics::StickyDelta => "sticky_delta",
    }
}

const fn curve_region_name(region: CurveRegion) -> &'static str {
    match region {
        CurveRegion::Pillar => "pillar",
        CurveRegion::Interpolated => "interpolated",
        CurveRegion::RightExtrapolated => "right_extrapolated",
    }
}
