use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use pricing::core::{CurrencyId, CurveId, PositiveF64, UnderlyingId};
use pricing::market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
use pricing::mc::{EngineConfig, ExecutionPolicy, PseudoMcConfig, VarianceReduction};
use pricing::models::{BlackScholesSpec, ModelSpec};
use pricing::product::{
    BarrierDirection, BarrierMonitoring, BarrierSpec, BarrierStyle, DigitalPayout, DigitalSpec,
    OptionSide, ProductSpec,
};
use pricing::risk::{
    GammaConfig, PayoffSmoothing, PayoffSmoothingWidthLadder, RiskRequest, SmileDynamics, SpotBump,
};
use pricing::{PricingPlan, PricingRequest};
use serde_json::json;

const SAMPLING_UNITS: u64 = 8_192;
const COMPILE_SAMPLES: usize = 20;
const EVALUATION_SAMPLES: usize = 5;

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .expect("usage: benchmark_path_dependence <output.json>");
    let policy = ExecutionPolicy::new(2, Some(256)).expect("execution policy");
    let exact_request = request(digital(), price_only());
    let smoothed_request = request(digital(), smoothed_risk(2.0));
    let ladder_request = request(
        digital(),
        smoothed_risk(3.0).with_payoff_smoothing_width_ladder(
            PayoffSmoothingWidthLadder::new(vec![4.0, 2.0, 1.0]).expect("width ladder"),
        ),
    );
    let continuous_request = request(continuous_barrier(), full_risk());

    let exact_plan = PricingPlan::compile(&exact_request, policy).expect("exact plan");
    let smoothed_plan = PricingPlan::compile(&smoothed_request, policy).expect("smoothed plan");
    let ladder_plan = PricingPlan::compile(&ladder_request, policy).expect("ladder plan");
    let continuous_plan =
        PricingPlan::compile(&continuous_request, policy).expect("continuous Barrier plan");
    black_box(exact_plan.evaluate().expect("exact warm-up"));
    black_box(smoothed_plan.evaluate().expect("smoothed warm-up"));
    black_box(
        ladder_plan
            .evaluate_width_ladder()
            .expect("width-ladder warm-up"),
    );
    black_box(
        continuous_plan
            .evaluate()
            .expect("continuous Barrier warm-up"),
    );

    let compile_exact = measure(COMPILE_SAMPLES, None, || {
        black_box(PricingPlan::compile(&exact_request, policy).expect("compile exact"));
    });
    let evaluate_exact = measure(EVALUATION_SAMPLES, Some(SAMPLING_UNITS * 2), || {
        black_box(exact_plan.evaluate().expect("evaluate exact"));
    });
    let compile_smoothed = measure(COMPILE_SAMPLES, None, || {
        black_box(PricingPlan::compile(&smoothed_request, policy).expect("compile smoothed"));
    });
    let evaluate_smoothed = measure(EVALUATION_SAMPLES, Some(SAMPLING_UNITS * 2), || {
        black_box(smoothed_plan.evaluate().expect("evaluate smoothed"));
    });
    let compile_ladder = measure(COMPILE_SAMPLES, None, || {
        black_box(PricingPlan::compile(&ladder_request, policy).expect("compile ladder"));
    });
    let evaluate_ladder = measure(EVALUATION_SAMPLES, Some(SAMPLING_UNITS * 2 * 4), || {
        black_box(
            ladder_plan
                .evaluate_width_ladder()
                .expect("evaluate ladder"),
        );
    });
    let compile_continuous = measure(COMPILE_SAMPLES, None, || {
        black_box(
            PricingPlan::compile(&continuous_request, policy).expect("compile continuous Barrier"),
        );
    });
    let evaluate_continuous = measure(EVALUATION_SAMPLES, Some(SAMPLING_UNITS * 2), || {
        black_box(
            continuous_plan
                .evaluate()
                .expect("evaluate continuous Barrier"),
        );
    });

    let report = json!({
        "schema_version": 1,
        "benchmark_kind": "rust_path_dependence",
        "library_version": pricing::version(),
        "configuration": {
            "engine": "pseudo_monte_carlo",
            "sampling_units": SAMPLING_UNITS,
            "antithetic": true,
            "evaluated_paths": SAMPLING_UNITS * 2,
            "worker_threads": 2,
            "reduction_block_size": 256,
            "compile_samples": COMPILE_SAMPLES,
            "evaluation_samples": EVALUATION_SAMPLES,
            "primary_smoothing_half_width": 3.0,
            "width_ladder": [4.0, 2.0, 1.0]
        },
        "measurements": {
            "compile_exact_digital_price": compile_exact,
            "evaluate_exact_digital_price": evaluate_exact,
            "compile_smoothed_digital_full_risk": compile_smoothed,
            "evaluate_smoothed_digital_full_risk": evaluate_smoothed,
            "compile_width_ladder": compile_ladder,
            "evaluate_width_ladder": evaluate_ladder,
            "compile_continuous_barrier_full_risk": compile_continuous,
            "evaluate_continuous_barrier_full_risk": evaluate_continuous
        },
        "capabilities": {
            "continuous_bridge_timing_available": true,
            "exact_and_smoothed_timing_separable": true,
            "peak_memory_available_in_process": false,
            "width_ladder_timing_available": true
        },
        "notes": [
            "Exact Digital Price and explicitly smoothed Digital Price/AAD are timed separately.",
            "The Width ladder evaluates one primary and three caller-ordered widths with common logical random coordinates.",
            "Continuous Barrier timing includes conditional Brownian-bridge survival and full risk validation.",
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

fn request(product: ProductSpec, risk: RiskRequest) -> PricingRequest {
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
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x3141_5926_5358_9793,
                SAMPLING_UNITS,
                VarianceReduction::new(true, false),
            )
            .expect("engine"),
        ),
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

fn continuous_barrier() -> ProductSpec {
    let expiry = "2027-09-04".parse().expect("expiry");
    ProductSpec::Barrier(
        BarrierSpec::new(
            UnderlyingId::new(1),
            CurrencyId::new(1),
            expiry,
            100.0,
            125.0,
            1.0,
            OptionSide::Call,
            BarrierDirection::Up,
            BarrierStyle::KnockOut,
            BarrierMonitoring::Continuous,
            vec!["2027-03-05".parse().expect("monitoring"), expiry],
            Some(2.0),
            expiry,
        )
        .expect("continuous Barrier"),
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

fn curve(id: u32, rate: f64) -> Arc<LogLinearDiscountCurve> {
    Arc::new(
        LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, (-rate).exp()])
            .expect("curve"),
    )
}
