use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use pricing::core::{CurrencyId, CurveId, PositiveF64, UnderlyingId};
use pricing::market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
use pricing::mc::{
    CpqrConfig, EngineConfig, ExecutionPolicy, LsmConfig, LsmStateVariable, PolynomialBasisSpec,
    PseudoMcConfig, VarianceReduction,
};
use pricing::models::{BlackScholesSpec, ModelSpec};
use pricing::product::{AmericanVanillaSpec, OptionSide, ProductSpec};
use pricing::risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump};
use pricing::{PricingPlan, PricingRequest};
use serde_json::json;

const LARGE_UNITS: u64 = 8_192;
const SMALL_UNITS: u64 = 128;
const RISK_UNITS: u64 = 4_096;
const COMPILE_SAMPLES: usize = 20;
const EVALUATION_SAMPLES: usize = 5;

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .expect("usage: benchmark_early_exercise <output.json>");
    let policy = ExecutionPolicy::new(2, Some(256)).expect("execution policy");
    let training_request = request(LARGE_UNITS, SMALL_UNITS, price_only());
    let valuation_request = request(SMALL_UNITS, LARGE_UNITS, price_only());
    let risk_request = request(RISK_UNITS, RISK_UNITS, full_risk());
    let training_plan = PricingPlan::compile(&training_request, policy).expect("training plan");
    let valuation_plan = PricingPlan::compile(&valuation_request, policy).expect("valuation plan");
    let risk_plan = PricingPlan::compile(&risk_request, policy).expect("risk plan");

    black_box(training_plan.evaluate().expect("training warm-up"));
    black_box(valuation_plan.evaluate().expect("valuation warm-up"));
    black_box(risk_plan.evaluate().expect("risk warm-up"));

    let compile_price = measure(COMPILE_SAMPLES, None, || {
        black_box(PricingPlan::compile(&valuation_request, policy).expect("compile Price"));
    });
    let phase_paths = (LARGE_UNITS + SMALL_UNITS) * 2;
    let evaluate_training = measure(EVALUATION_SAMPLES, Some(phase_paths), || {
        black_box(
            training_plan
                .evaluate()
                .expect("evaluate training workload"),
        );
    });
    let evaluate_valuation = measure(EVALUATION_SAMPLES, Some(phase_paths), || {
        black_box(
            valuation_plan
                .evaluate()
                .expect("evaluate valuation workload"),
        );
    });
    let compile_full_risk = measure(COMPILE_SAMPLES, None, || {
        black_box(PricingPlan::compile(&risk_request, policy).expect("compile full risk"));
    });
    let evaluate_full_risk = measure(EVALUATION_SAMPLES, Some(RISK_UNITS * 4), || {
        black_box(risk_plan.evaluate().expect("evaluate full risk"));
    });

    let report = json!({
        "schema_version": 1,
        "benchmark_kind": "rust_early_exercise",
        "library_version": pricing::version(),
        "configuration": {
            "engine": "pseudo_monte_carlo",
            "large_sampling_units": LARGE_UNITS,
            "small_sampling_units": SMALL_UNITS,
            "risk_sampling_units": RISK_UNITS,
            "antithetic": true,
            "worker_threads": 2,
            "reduction_block_size": 256,
            "compile_samples": COMPILE_SAMPLES,
            "evaluation_samples": EVALUATION_SAMPLES,
            "exercise_date_count": 4,
            "basis_max_degree": 3,
            "basis_column_count": 4
        },
        "measurements": {
            "compile_price": compile_price,
            "evaluate_training_dominant_price": evaluate_training,
            "evaluate_valuation_dominant_price": evaluate_valuation,
            "compile_fixed_policy_full_risk": compile_full_risk,
            "evaluate_fixed_policy_aad_with_crn_validation": evaluate_full_risk
        },
        "capabilities": {
            "training_and_valuation_workloads_separable": true,
            "aad_and_crn_validation_timing_separable": false,
            "peak_memory_available_in_process": false
        },
        "process_peak_memory_bytes": null,
        "notes": [
            "The training-dominant workload fixes valuation at 128 sampling units; the valuation-dominant workload fixes training at 128 sampling units.",
            "The fixed-policy risk workload computes AAD Delta/Vega, bumped-AAD Gamma, and common-random-number validations in one execution.",
            "The external benchmark runner records peak resident memory for the complete process.",
            "Results are an optimization baseline and not a latency SLA."
        ]
    });
    std::fs::write(
        Path::new(&output),
        serde_json::to_string_pretty(&report).expect("serialize benchmark report") + "\n",
    )
    .expect("write benchmark report");
}

fn measure<F>(samples: usize, evaluated_paths: Option<u64>, mut operation: F) -> serde_json::Value
where
    F: FnMut(),
{
    let mut durations = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        operation();
        durations.push(started.elapsed().as_secs_f64());
    }
    durations.sort_by(f64::total_cmp);
    let median = durations[durations.len() / 2];
    json!({
        "samples": samples,
        "median_seconds": median,
        "minimum_seconds": durations[0],
        "maximum_seconds": durations[durations.len() - 1],
        "evaluated_paths_per_sample": evaluated_paths,
        "median_paths_per_second": evaluated_paths.map(|paths| paths as f64 / median)
    })
}

fn request(training_units: u64, valuation_units: u64, risk: RiskRequest) -> PricingRequest {
    let underlying = UnderlyingId::new(1);
    let currency = CurrencyId::new(1);
    let exercise_dates = ["2026-12-04", "2027-03-04", "2027-06-04", "2027-09-04"]
        .map(|value| value.parse().expect("exercise date"));
    let training_engine = pseudo_engine(0x1020_3040_5060_7080, training_units);
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
        pseudo_engine(0x0123_4567_89ab_cdef, valuation_units),
        risk,
        Some(lsm),
    )
    .expect("American request")
}

fn pseudo_engine(seed: u64, units: u64) -> EngineConfig {
    EngineConfig::PseudoMonteCarlo(
        PseudoMcConfig::new(seed, units, VarianceReduction::new(true, false))
            .expect("Pseudo-MC engine"),
    )
}

fn curve(id: u32, rate: f64) -> Arc<LogLinearDiscountCurve> {
    Arc::new(
        LogLinearDiscountCurve::new(CurveId::new(id), vec![0.0, 1.0], vec![1.0, (-rate).exp()])
            .expect("curve"),
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
