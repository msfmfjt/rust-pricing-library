use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use pricing::core::{CurrencyId, CurveId, PositiveF64, UnderlyingId};
use pricing::market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
use pricing::mc::{EngineConfig, ExecutionPolicy, PseudoMcConfig, VarianceReduction};
use pricing::models::{BlackScholesSpec, ModelSpec};
use pricing::product::{EuropeanVanillaSpec, OptionSide, ProductSpec};
use pricing::risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump};
use pricing::{PricingPlan, PricingRequest};
use serde_json::json;

const SAMPLING_UNITS: u64 = 16_384;
const COMPILE_SAMPLES: usize = 20;
const EVALUATION_SAMPLES: usize = 5;
const SPOT: f64 = 100.0;
const VOLATILITY: f64 = 0.2;
const VALIDATION_SPOT_BUMP: f64 = 1.0;
const VALIDATION_VOLATILITY_BUMP: f64 = 1.0e-4;
const BUMP_PLAN_COUNT: u64 = 5;

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .expect("usage: benchmark_european_bs <output.json>");
    let policy = ExecutionPolicy::new(2, Some(256)).expect("execution policy");
    let price_request = request(
        RiskRequest::price_only(SmileDynamics::StickyLogMoneyness),
        SPOT,
        VOLATILITY,
    );
    let risk_request = request(
        RiskRequest::new(
            true,
            Some(GammaConfig::new(
                SpotBump::relative(0.01).expect("gamma bump"),
            )),
            true,
            None,
            SmileDynamics::StickyLogMoneyness,
            Some(16),
            Some(128),
        )
        .expect("risk request"),
        SPOT,
        VOLATILITY,
    );
    let bump_requests = [
        request(price_only(), SPOT, VOLATILITY),
        request(price_only(), SPOT - VALIDATION_SPOT_BUMP, VOLATILITY),
        request(price_only(), SPOT + VALIDATION_SPOT_BUMP, VOLATILITY),
        request(
            price_only(),
            SPOT,
            VOLATILITY - VALIDATION_VOLATILITY_BUMP,
        ),
        request(
            price_only(),
            SPOT,
            VOLATILITY + VALIDATION_VOLATILITY_BUMP,
        ),
    ];

    let price_plan = PricingPlan::compile(&price_request, policy).expect("price plan");
    let risk_plan = PricingPlan::compile(&risk_request, policy).expect("risk plan");
    let bump_plans: Vec<_> = bump_requests
        .iter()
        .map(|request| {
            PricingPlan::compile(request, policy).expect("compile bump-validation plan")
        })
        .collect();
    black_box(price_plan.evaluate().expect("price warm-up"));
    black_box(risk_plan.evaluate().expect("risk warm-up"));
    for plan in &bump_plans {
        black_box(plan.evaluate().expect("bump-validation warm-up"));
    }

    let compile_price = measure(COMPILE_SAMPLES, None, || {
        black_box(PricingPlan::compile(&price_request, policy).expect("compile price"));
    });
    let evaluate_price = measure(EVALUATION_SAMPLES, Some(SAMPLING_UNITS * 2), || {
        black_box(price_plan.evaluate().expect("evaluate price"));
    });
    let compile_risk = measure(COMPILE_SAMPLES, None, || {
        black_box(PricingPlan::compile(&risk_request, policy).expect("compile risk"));
    });
    let compile_bump = measure(COMPILE_SAMPLES, None, || {
        for request in &bump_requests {
            black_box(PricingPlan::compile(request, policy).expect("compile bump validation"));
        }
    });
    let evaluate_risk = measure(EVALUATION_SAMPLES, Some(SAMPLING_UNITS * 2), || {
        black_box(risk_plan.evaluate().expect("evaluate risk"));
    });
    let evaluate_bump = measure(
        EVALUATION_SAMPLES,
        Some(SAMPLING_UNITS * 2 * BUMP_PLAN_COUNT),
        || {
            for plan in &bump_plans {
                black_box(plan.evaluate().expect("evaluate bump validation"));
            }
        },
    );

    let report = json!({
        "schema_version": 1,
        "benchmark_kind": "rust_european_black_scholes",
        "library_version": pricing::version(),
        "configuration": {
            "engine": "pseudo_monte_carlo",
            "sampling_units": SAMPLING_UNITS,
            "antithetic": true,
            "evaluated_paths": SAMPLING_UNITS * 2,
            "worker_threads": 2,
            "reduction_block_size": 256,
            "compile_samples": COMPILE_SAMPLES,
            "evaluation_samples": EVALUATION_SAMPLES
        },
        "measurements": {
            "compile_price_only": compile_price,
            "evaluate_price_only": evaluate_price,
            "compile_full_risk": compile_risk,
            "evaluate_aad_with_crn_bump_validation": evaluate_risk,
            "compile_crn_bump_validation": compile_bump,
            "evaluate_crn_bump_validation_price_only": evaluate_bump
        },
        "capabilities": {
            "aad_and_bump_timing_separable": false,
            "standalone_bump_timing_available": true,
            "allocation_count_available": false,
            "peak_memory_available_in_process": false
        },
        "notes": [
            "The full-risk kernel computes AAD Delta/Vega, central-bumped AAD Delta Gamma, and CRN bump validations in one execution.",
            "The standalone bump case evaluates base, Spot-down/up, and volatility-down/up Price-only plans with common random numbers.",
            "Results are an optimization baseline and not a latency SLA."
        ]
    });
    std::fs::write(
        Path::new(&output),
        serde_json::to_string_pretty(&report).expect("serialize benchmark") + "\n",
    )
    .expect("write benchmark");
}

fn measure<F>(samples: usize, evaluated_paths: Option<u64>, mut operation: F) -> serde_json::Value
where
    F: FnMut(),
{
    let mut seconds = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        operation();
        seconds.push(start.elapsed().as_secs_f64());
    }
    seconds.sort_by(f64::total_cmp);
    let median_seconds = seconds[samples / 2];
    json!({
        "samples": samples,
        "median_seconds": median_seconds,
        "minimum_seconds": seconds[0],
        "maximum_seconds": seconds[samples - 1],
        "evaluated_paths_per_sample": evaluated_paths,
        "median_paths_per_second": evaluated_paths.map(|paths| paths as f64 / median_seconds)
    })
}

fn price_only() -> RiskRequest {
    RiskRequest::price_only(SmileDynamics::StickyLogMoneyness)
}

fn request(risk: RiskRequest, spot: f64, volatility: f64) -> PricingRequest {
    let underlying = UnderlyingId::new(1);
    let currency = CurrencyId::new(1);
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
                PositiveF64::new(spot, "spot").expect("spot"),
                curve(1, 0.05),
                curve(2, 0.02),
            ),
        )),
        ModelSpec::BlackScholes(BlackScholesSpec::new(volatility).expect("model")),
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x0123_4567_89ab_cdef,
                SAMPLING_UNITS,
                VarianceReduction::new(true, false),
            )
            .expect("engine"),
        ),
        risk,
    )
    .expect("request")
}

fn curve(id: u32, rate: f64) -> Arc<LogLinearDiscountCurve> {
    Arc::new(
        LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, (-rate).exp()])
            .expect("curve"),
    )
}
