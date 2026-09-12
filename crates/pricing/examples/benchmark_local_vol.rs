use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use pricing::core::{CurrencyId, CurveId, Date, PositiveF64, UnderlyingId};
use pricing::market::{EquityForward, EquityMarket, LogLinearDiscountCurve, MarketContext};
use pricing::mc::{EngineConfig, ExecutionPolicy, PseudoMcConfig, VarianceReduction};
use pricing::models::{LocalVolatilityReportingBasis, LocalVolatilitySpec, ModelSpec};
use pricing::product::{EuropeanVanillaSpec, OptionSide, ProductSpec};
use pricing::risk::{GammaConfig, RiskRequest, SmileDynamics, SpotBump, VegaKtConfig};
use pricing::{PricingPlan, PricingRequest};
use serde_json::json;

const SAMPLING_UNITS: u64 = 8_192;
const COMPILE_SAMPLES: usize = 20;
const EVALUATION_SAMPLES: usize = 5;
const SPOT: f64 = 100.0;
const SPOT_BUMP: f64 = 1.0;
const LOCAL_VARIANCE_BUMP: f64 = 1.0e-4;
const BUMP_PLAN_COUNT: u64 = 5;

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .expect("usage: benchmark_local_vol <output.json>");
    let policy = ExecutionPolicy::new(2, Some(256)).expect("execution policy");

    let price_request = request(price_only(), SPOT, 0.0, false);
    let local_vega_request = request(local_vega_risk(), SPOT, 0.0, false);
    let vega_kt_request = request(vega_kt_risk(), SPOT, 0.0, true);
    let bump_requests = [
        request(price_only(), SPOT, 0.0, false),
        request(price_only(), SPOT - SPOT_BUMP, 0.0, false),
        request(price_only(), SPOT + SPOT_BUMP, 0.0, false),
        request(price_only(), SPOT, -LOCAL_VARIANCE_BUMP, false),
        request(price_only(), SPOT, LOCAL_VARIANCE_BUMP, false),
    ];

    let price_plan = PricingPlan::compile(&price_request, policy).expect("price plan");
    let local_vega_plan =
        PricingPlan::compile(&local_vega_request, policy).expect("local Vega plan");
    let vega_kt_plan = PricingPlan::compile(&vega_kt_request, policy).expect("VegaKT plan");
    let bump_plans: Vec<_> = bump_requests
        .iter()
        .map(|request| PricingPlan::compile(request, policy).expect("compile bump plan"))
        .collect();

    black_box(price_plan.evaluate().expect("price warm-up"));
    black_box(local_vega_plan.evaluate().expect("local Vega warm-up"));
    black_box(vega_kt_plan.evaluate().expect("VegaKT warm-up"));
    for plan in &bump_plans {
        black_box(plan.evaluate().expect("bump warm-up"));
    }

    let compile_price = measure(COMPILE_SAMPLES, None, || {
        black_box(PricingPlan::compile(&price_request, policy).expect("compile price"));
    });
    let evaluate_price = measure(EVALUATION_SAMPLES, Some(SAMPLING_UNITS * 2), || {
        black_box(price_plan.evaluate().expect("evaluate price"));
    });
    let compile_local_vega = measure(COMPILE_SAMPLES, None, || {
        black_box(PricingPlan::compile(&local_vega_request, policy).expect("compile local Vega"));
    });
    let evaluate_local_vega = measure(EVALUATION_SAMPLES, Some(SAMPLING_UNITS * 2), || {
        black_box(local_vega_plan.evaluate().expect("evaluate local Vega"));
    });
    let compile_vega_kt = measure(COMPILE_SAMPLES, None, || {
        black_box(PricingPlan::compile(&vega_kt_request, policy).expect("compile VegaKT"));
    });
    let evaluate_vega_kt = measure(EVALUATION_SAMPLES, Some(SAMPLING_UNITS * 2), || {
        black_box(vega_kt_plan.evaluate().expect("evaluate VegaKT"));
    });
    let compile_bump = measure(COMPILE_SAMPLES, None, || {
        for request in &bump_requests {
            black_box(PricingPlan::compile(request, policy).expect("compile bump validation"));
        }
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
        "benchmark_kind": "rust_local_volatility_vegakt",
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
            "local_variance_grid_shape": [3, 5],
            "reporting_iv_basis_shape": [2, 3],
            "checkpoint_interval": 16,
            "aad_tile_capacity": 256,
            "vega_kt_covariance_layout": "full_bucket_matrix_row_major"
        },
        "measurements": {
            "compile_price_only": compile_price,
            "evaluate_price_only": evaluate_price,
            "compile_aad_local_vega": compile_local_vega,
            "evaluate_aad_local_vega": evaluate_local_vega,
            "compile_vega_kt_decomposition": compile_vega_kt,
            "evaluate_vega_kt_decomposition": evaluate_vega_kt,
            "compile_selected_crn_bump_validation": compile_bump,
            "evaluate_selected_crn_bump_validation_price_only": evaluate_bump
        },
        "capabilities": {
            "aad_local_vega_timing_available": true,
            "vega_kt_decomposition_timing_available": true,
            "standalone_spot_and_local_variance_bump_timing_available": true,
            "allocation_count_available": false,
            "peak_memory_available_in_process": false
        },
        "notes": [
            "The AAD Local Vega workload computes scalar Local Volatility Vega without VegaKT reporting projection.",
            "The VegaKT workload computes Delta, bumped Gamma, scalar Vega, VegaKT buckets, and full bucket covariance.",
            "The standalone bump case evaluates base, Spot-down/up, and uniform local-variance-down/up Price-only plans with common random numbers.",
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

fn local_vega_risk() -> RiskRequest {
    RiskRequest::new(
        false,
        None,
        true,
        None,
        SmileDynamics::StickyLogMoneyness,
        Some(16),
        Some(256),
    )
    .expect("local Vega risk request")
}

fn vega_kt_risk() -> RiskRequest {
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
    .expect("VegaKT risk request")
}

fn request(
    risk: RiskRequest,
    spot: f64,
    local_variance_shift: f64,
    reporting_basis: bool,
) -> PricingRequest {
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
        ModelSpec::LocalVolatility(local_volatility_model(
            local_variance_shift,
            reporting_basis,
        )),
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(
                0x1357_9bdf_2468_ace0,
                SAMPLING_UNITS,
                VarianceReduction::new(true, false),
            )
            .expect("engine"),
        ),
        risk,
    )
    .expect("request")
}

fn local_volatility_model(local_variance_shift: f64, reporting_basis: bool) -> LocalVolatilitySpec {
    let time_nodes = if reporting_basis {
        let maturities = reporting_maturities();
        vec![0.0, maturities[0], maturities[1]]
    } else {
        vec![0.0, 0.5, 1.0]
    };
    let values = [
        0.048, 0.042, 0.039, 0.041, 0.047, 0.051, 0.044, 0.040, 0.043, 0.050, 0.056, 0.049, 0.045,
        0.047, 0.053,
    ]
    .into_iter()
    .map(|value| value + local_variance_shift)
    .collect();
    let model = LocalVolatilitySpec::from_explicit_grid(
        time_nodes,
        vec![-0.3, -0.1, 0.0, 0.2, 0.4],
        values,
        1.0e-8,
        4.0,
    )
    .expect("local volatility");
    if reporting_basis {
        model.with_reporting_iv_basis(reporting_iv_basis())
    } else {
        model
    }
}

fn reporting_iv_basis() -> LocalVolatilityReportingBasis {
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
