//! Reproducible native-vs-contract timings; run with --release, not as a test.
use pricing::contract::{
    Aggregate, AutocallPeriod, BarrierDirection, Contract, DiscreteBarrierTerms, InstrumentBuilder,
};
use pricing::core::{CurrencyId, Date, UnderlyingId};
use pricing::product::{
    self, BarrierMonitoring, BarrierStyle, CompiledPayoff, GraphLimitPolicy, OptionSide,
    ProductSpec, SourceGraph,
};
use pricing::{JsonLimits, PricingOperation, PricingPlan, PricingRequest, PricingSpecification};
use serde_json::{Value, json};
use std::{hint::black_box, time::Instant};
const PATHS: usize = 100_000;
fn d(s: &str) -> Date {
    s.parse().unwrap()
}
fn measured_compile(mut f: impl FnMut() -> CompiledPayoff) -> f64 {
    let start = Instant::now();
    for _ in 0..200 {
        black_box(f());
    }
    start.elapsed().as_secs_f64() * 1e6 / 200.0
}
fn path_benchmark(p: &CompiledPayoff) -> Value {
    let mut w = p.workspace();
    for spot in [60.0, 110.0, 160.0] {
        p.evaluate_reusing(&mut w, |_, _| Some(spot), |_, _| None)
            .unwrap();
    }
    let before = w.allocated_bytes();
    let mut growth = 0;
    let mut bytes = before;
    let start = Instant::now();
    let mut checksum = 0.0;
    for i in 0..PATHS {
        let spot = 60.0 + (i % 101) as f64;
        let values = p
            .evaluate_reusing(&mut w, |_, _| Some(black_box(spot)), |_, _| None)
            .unwrap();
        checksum += values.iter().sum::<f64>();
        let current = w.allocated_bytes();
        growth += usize::from(current != bytes);
        bytes = current;
    }
    let reused = start.elapsed().as_secs_f64() * 1e9 / PATHS as f64;
    let start = Instant::now();
    for i in 0..PATHS {
        black_box(p.evaluate(|_, _| Some(60.0 + (i % 101) as f64)).unwrap());
    }
    let allocated = start.elapsed().as_secs_f64() * 1e9 / PATHS as f64;
    json!({"reused_ns_per_path":reused, "owned_output_ns_per_path":allocated,
        "workspace_bytes":bytes,"workspace_capacity_growth_after_warmup":growth,
        "state_value_slots":p.opcodes().len(),"observation_slots":p.terminal_observations().len(),
        "checksum":checksum})
}
fn bench(c: &Contract, old: Option<SourceGraph>) -> Value {
    let compile = || {
        c.compile_mc(d("2026-09-04"), vec![UnderlyingId::new(1)])
            .unwrap()
            .groups()[0]
            .payoff()
            .clone()
    };
    let p = compile();
    let mut report =
        json!({"contract_compile_us":measured_compile(compile), "contract":path_benchmark(&p)});
    if let Some(source) = old {
        let native_compile = || source.compile(GraphLimitPolicy::DEFAULT).unwrap();
        let native = native_compile();
        for spot in [60.0, 80.0, 100.0, 120.0, 160.0] {
            let a: f64 = p.evaluate(|_, _| Some(spot)).unwrap().iter().sum();
            let b: f64 = native.evaluate(|_, _| Some(spot)).unwrap().iter().sum();
            assert!((a - b).abs() < 1e-11, "path parity: {a} vs {b}");
        }
        report["native_compile_us"] = json!(measured_compile(native_compile));
        report["native"] = path_benchmark(&native);
    }
    report
}
fn main() {
    let u = UnderlyingId::new(1);
    let ccy = CurrencyId::new(2);
    let expiry = d("2027-09-04");
    let dates = vec![d("2027-03-04"), expiry];
    let b = InstrumentBuilder::new("benchmark", ccy, u, "close").unwrap();
    let vanilla = b
        .european(OptionSide::Call, 100.0, 1.0, expiry, expiry)
        .unwrap();
    let native =
        product::EuropeanVanillaSpec::new(u, ccy, expiry, 100.0, 1.0, OptionSide::Call).unwrap();
    let asian = b
        .asian(
            Aggregate::ArithmeticAverage,
            OptionSide::Call,
            100.0,
            1.0,
            dates.clone(),
            expiry,
        )
        .unwrap();
    let old_asian = product::ArithmeticAsianSpec::new(
        u,
        ccy,
        100.0,
        1.0,
        OptionSide::Call,
        dates
            .iter()
            .map(|t| product::AsianObservation::unknown(*t, 0.5).unwrap())
            .collect(),
        expiry,
    )
    .unwrap();
    let barrier = b
        .discrete_barrier(DiscreteBarrierTerms {
            side: OptionSide::Call,
            strike: 100.0,
            notional: 1.0,
            expiry,
            payment_date: expiry,
            monitoring_dates: dates.clone(),
            direction: BarrierDirection::Up,
            style: BarrierStyle::KnockOut,
            level: 120.0,
            rebate: 3.0,
        })
        .unwrap();
    let old_barrier = product::BarrierSpec::new(
        u,
        ccy,
        expiry,
        100.0,
        120.0,
        1.0,
        OptionSide::Call,
        product::BarrierDirection::Up,
        BarrierStyle::KnockOut,
        BarrierMonitoring::Discrete,
        dates.clone(),
        Some(3.0),
        expiry,
    )
    .unwrap();
    let autocall = b
        .autocall(
            100.0,
            100.0,
            70.0,
            dates
                .iter()
                .map(|t| AutocallPeriod {
                    observation_date: *t,
                    payment_date: *t,
                    call_level: 100.0,
                    coupon_level: 80.0,
                    coupon_rate: 0.05,
                })
                .collect(),
        )
        .unwrap();
    let mut request: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v3/pricing_request.golden.json"
    ))
    .unwrap();
    request["engine"]["independent_sampling_units"] = json!(16384);
    request["risk"]["delta"] = json!(false);
    request["risk"]["vega"] = json!(false);
    let r =
        pricing::parse_request_json(&serde_json::to_vec(&request).unwrap(), JsonLimits::DEFAULT)
            .unwrap();
    let policy = pricing::mc::ExecutionPolicy::new(2, Some(256)).unwrap();
    let mut mc = serde_json::Map::new();
    for (name, contract, product) in [
        (
            "european",
            &vanilla,
            ProductSpec::EuropeanVanilla(native.clone()),
        ),
        (
            "asian",
            &asian,
            ProductSpec::ArithmeticAsian(old_asian.clone()),
        ),
        (
            "barrier",
            &barrier,
            ProductSpec::Barrier(old_barrier.clone()),
        ),
    ] {
        let req = PricingRequest::new(
            r.valuation_date(),
            product,
            r.market().clone(),
            r.model().clone(),
            r.engine(),
            r.risk().clone(),
        )
        .unwrap();
        let source = PricingSpecification::from_request(req.clone(), policy);
        let mut old_times = Vec::new();
        let mut new_times = Vec::new();
        let mut native_task_times = Vec::new();
        let mut result = json!({});
        for iteration in 0..6 {
            let start = Instant::now();
            let old = PricingPlan::compile(&req, policy)
                .unwrap()
                .evaluate()
                .unwrap();
            let old_ms = start.elapsed().as_secs_f64() * 1e3;
            let start = Instant::now();
            let native_task = pricing::PricingTask::new(
                pricing::PricingSource::SingleAsset(source.clone()),
                PricingOperation::Evaluate,
            )
            .unwrap()
            .compile()
            .unwrap()
            .evaluate()
            .unwrap();
            let native_task_ms = start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(
                native_task.report.value(),
                old.pricing_result.value.value().get()
            );
            let start = Instant::now();
            let new = contract
                .compile_mc(r.valuation_date(), vec![u])
                .unwrap()
                .pricing_task(&source, PricingOperation::Evaluate)
                .unwrap()
                .compile()
                .unwrap()
                .evaluate()
                .unwrap();
            let new_ms = start.elapsed().as_secs_f64() * 1e3;
            assert!((old.pricing_result.value.value().get() - new.report.value()).abs() < 1e-11);
            assert!(
                (old.pricing_result.value.standard_error().get() - new.report.standard_error())
                    .abs()
                    < 1e-11
            );
            if iteration > 0 {
                old_times.push(old_ms);
                new_times.push(new_ms);
                native_task_times.push(native_task_ms);
            }
            result = json!({"pv":new.report.value(),"standard_error":new.report.standard_error(),"native_pv":old.pricing_result.value.value().get()});
        }
        native_task_times.sort_by(f64::total_cmp);
        old_times.sort_by(f64::total_cmp);
        new_times.sort_by(f64::total_cmp);
        result["native_total_ms"] = json!(old_times[2]);
        result["native_task_total_ms"] = json!(native_task_times[2]);
        result["contract_total_ms"] = json!(new_times[2]);
        result["timing_repetitions_after_warmup"] = json!(5);
        mc.insert(name.into(), result);
    }
    println!("{}", serde_json::to_string_pretty(&json!({"schema_version":1,"paths":PATHS,"mc_sampling_units":16384,"worker_threads":2,"seed":7,
        "allocation_metric":"Reserved workspace bytes and observed capacity growth; not process-global allocator counts. Model path buffers are owned by existing engines.",
        "european":bench(&vanilla, Some(native.source_graph().unwrap())),
        "asian":bench(&asian, Some(old_asian.source_graph().unwrap())),
        "barrier":bench(&barrier, Some(old_barrier.source_graph().unwrap())),
        "autocall":bench(&autocall, None),"mc":mc})).unwrap());
}
