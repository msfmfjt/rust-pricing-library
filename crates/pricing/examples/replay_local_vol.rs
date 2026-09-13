use std::path::Path;
use std::sync::Arc;

use pricing::core::{CurrencyId, CurveId, Date, PositiveF64, UnderlyingId};
use pricing::market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
use pricing::mc::{EngineConfig, ExecutionPolicy, PseudoMcConfig, RqmcConfig, VarianceReduction};
use pricing::models::{LocalVolatilityReportingBasis, LocalVolatilitySpec, ModelSpec};
use pricing::product::{EuropeanVanillaSpec, OptionSide, ProductSpec};
use pricing::risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump, VegaKtConfig};
use pricing::{
    Estimate, MonteCarloPrice, PricingPlan, PricingRequest, RiskMethod, RiskValidation,
    request_to_json, result_to_json,
};
use serde_json::{Value, json};

const PSEUDO_UNITS: u64 = 8_192;
const RQMC_POINTS: u64 = 512;
const RQMC_SCRAMBLES: u32 = 8;

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .expect("usage: replay_local_vol <output.json>");
    let policy = ExecutionPolicy::new(2, Some(256)).expect("execution policy");
    let cases = [
        capture(
            "pseudo_mc_price_only",
            request(
                EngineConfig::PseudoMonteCarlo(
                    PseudoMcConfig::new(
                        0x1357_9bdf_2468_ace0,
                        PSEUDO_UNITS,
                        VarianceReduction::new(true, false),
                    )
                    .expect("pseudo engine"),
                ),
                RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            ),
            policy,
        ),
        capture(
            "rqmc_price_only",
            request(
                EngineConfig::RandomizedQuasiMonteCarlo(
                    RqmcConfig::new(
                        RQMC_POINTS,
                        RQMC_SCRAMBLES,
                        0x0246_8ace_1357_9bdf,
                        VarianceReduction::new(true, true),
                    )
                    .expect("RQMC engine"),
                ),
                RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
            ),
            policy,
        ),
        capture(
            "pseudo_mc_delta_gamma_vega_vegakt",
            request(
                EngineConfig::PseudoMonteCarlo(
                    PseudoMcConfig::new(
                        0x1357_9bdf_2468_ace0,
                        PSEUDO_UNITS,
                        VarianceReduction::new(true, false),
                    )
                    .expect("pseudo engine"),
                ),
                risk_request(),
            ),
            policy,
        ),
        capture(
            "rqmc_delta_gamma_vega_vegakt",
            request(
                EngineConfig::RandomizedQuasiMonteCarlo(
                    RqmcConfig::new(
                        RQMC_POINTS,
                        RQMC_SCRAMBLES,
                        0x0246_8ace_1357_9bdf,
                        VarianceReduction::new(true, true),
                    )
                    .expect("RQMC engine"),
                ),
                risk_request(),
            ),
            policy,
        ),
    ];
    let fixture = json!({
        "schema_version": 1,
        "fixture_kind": "local_volatility_replay",
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

fn request(engine: EngineConfig, risk: RiskRequest) -> PricingRequest {
    let underlying = UnderlyingId::new(1);
    let currency = CurrencyId::new(1);
    let model = if risk.vega_kt().is_some() {
        local_volatility_model_with_reporting_basis()
    } else {
        local_volatility_model()
    };
    PricingRequest::new(
        "2026-09-04".parse().expect("valuation date"),
        ProductSpec::EuropeanVanilla(
            EuropeanVanillaSpec::new(
                underlying,
                currency,
                "2027-09-04".parse().expect("expiry"),
                100.0,
                1.0,
                OptionSide::Call,
            )
            .expect("product"),
        ),
        MarketContext::Equity(EquityMarket::new(
            currency,
            EquityForward::new(
                underlying,
                PositiveF64::new(100.0, "spot").expect("spot"),
                curve(1, 0.05),
                curve(2, 0.02),
            ),
        )),
        ModelSpec::LocalVolatility(model),
        engine,
        risk,
    )
    .expect("request")
}

fn local_volatility_model() -> LocalVolatilitySpec {
    LocalVolatilitySpec::from_explicit_grid(
        vec![0.0, 0.5, 1.0],
        vec![-0.3, -0.1, 0.0, 0.2, 0.4],
        vec![
            0.048, 0.042, 0.039, 0.041, 0.047, 0.051, 0.044, 0.040, 0.043, 0.050, 0.056, 0.049,
            0.045, 0.047, 0.053,
        ],
        1.0e-8,
        4.0,
    )
    .expect("local volatility")
}

fn local_volatility_model_with_reporting_basis() -> LocalVolatilitySpec {
    let reporting_maturities = reporting_maturities();
    LocalVolatilitySpec::from_explicit_grid(
        vec![0.0, reporting_maturities[0], reporting_maturities[1]],
        vec![-0.3, -0.1, 0.0, 0.2, 0.4],
        vec![
            0.048, 0.042, 0.039, 0.041, 0.047, 0.051, 0.044, 0.040, 0.043, 0.050, 0.056, 0.049,
            0.045, 0.047, 0.053,
        ],
        1.0e-8,
        4.0,
    )
    .expect("local volatility")
    .with_reporting_iv_basis(reporting_basis())
}

fn risk_request() -> RiskRequest {
    RiskRequest::new(
        true,
        Some(GammaConfig::new(
            SpotBump::relative(0.01).expect("gamma bump"),
        )),
        true,
        Some(
            VegaKtConfig::new(
                vec![
                    "2027-03-05".parse().expect("first VegaKT maturity"),
                    "2027-09-04".parse().expect("second VegaKT maturity"),
                ],
                vec![-0.2, 0.0, 0.2],
                1.0e-8,
                true,
            )
            .expect("VegaKT config"),
        ),
        SmileDynamics::StickyLogMoneyness,
        Some(16),
        Some(256),
    )
    .expect("risk request")
}

fn reporting_basis() -> LocalVolatilityReportingBasis {
    LocalVolatilityReportingBasis::new(
        reporting_maturities(),
        vec![-0.2, 0.0, 0.2],
        vec![0.195, 0.200, 0.207, 0.190, 0.202, 0.215],
    )
    .expect("reporting IV basis")
}

fn reporting_maturities() -> Vec<f64> {
    let valuation_date: Date = "2026-09-04".parse().expect("valuation date");
    ["2027-03-05", "2027-09-04"]
        .into_iter()
        .map(|date| {
            let maturity: Date = date.parse().expect("reporting maturity");
            f64::from(valuation_date.days_until(maturity)) / 365.0
        })
        .collect()
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
