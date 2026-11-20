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
fn setting(name: &str, default: usize) -> usize {
    let value = std::env::var(name)
        .map(|v| {
            v.parse::<usize>()
                .expect("positive integer benchmark setting")
        })
        .unwrap_or(default);
    assert!(value > 0, "benchmark setting must be positive");
    value
}
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
    let paths = setting("CONTRACT_BENCH_PATHS", 1_000_000);
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
    for i in 0..paths {
        let spot = 60.0 + (i % 101) as f64;
        let values = p
            .evaluate_reusing(&mut w, |_, _| Some(black_box(spot)), |_, _| None)
            .unwrap();
        checksum += values.iter().sum::<f64>();
        let current = w.allocated_bytes();
        growth += usize::from(current != bytes);
        bytes = current;
    }
    let reused = start.elapsed().as_secs_f64() * 1e9 / paths as f64;
    let start = Instant::now();
    for i in 0..paths {
        black_box(p.evaluate(|_, _| Some(60.0 + (i % 101) as f64)).unwrap());
    }
    let allocated = start.elapsed().as_secs_f64() * 1e9 / paths as f64;
    json!({"reused_ns_per_path":reused, "owned_output_ns_per_path":allocated,
        "workspace_bytes":bytes,"workspace_capacity_growth_after_warmup":growth,
        "state_value_slots":p.forward_value_slots(),"ssa_slots":p.opcodes().len(),"evaluation_program_bytes":p.evaluation_program_bytes(),"observation_slots":p.terminal_observations().len(),
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
    let periods = setting("CONTRACT_BENCH_PERIODS", 12);
    assert!(periods <= 12 && 12 % periods == 0, "periods must divide 12");
    let dates: Vec<_> = (1..=periods)
        .map(|i| {
            let month = 9 + i * (12 / periods);
            d(&format!(
                "{}-{:02}-04",
                2026 + (month - 1) / 12,
                (month - 1) % 12 + 1
            ))
        })
        .collect();
    let samples = setting("CONTRACT_BENCH_SAMPLES", 262_144);
    let repetitions = setting("CONTRACT_BENCH_REPETITIONS", 7);
    let workers = setting("CONTRACT_BENCH_WORKERS", 2);
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
            .map(|t| product::AsianObservation::unknown(*t, 1.0 / periods as f64).unwrap())
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
    request["engine"]["independent_sampling_units"] = json!(samples);
    request["risk"]["delta"] = json!(false);
    request["risk"]["vega"] = json!(false);
    let r =
        pricing::parse_request_json(&serde_json::to_vec(&request).unwrap(), JsonLimits::DEFAULT)
            .unwrap();
    let policy = pricing::mc::ExecutionPolicy::new(workers.try_into().unwrap(), Some(256)).unwrap();
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
        let mut paired_ratios = Vec::new();
        let mut result = json!({});
        let mut native_plan = || {
            let old = PricingPlan::compile(&req, policy)
                .unwrap()
                .evaluate()
                .unwrap();
            (
                old.pricing_result.value.value().get(),
                old.pricing_result.value.standard_error().get(),
            )
        };
        let mut native_task = || {
            let old = pricing::PricingTask::new(
                pricing::PricingSource::SingleAsset(source.clone()),
                PricingOperation::Evaluate,
            )
            .unwrap()
            .compile()
            .unwrap()
            .evaluate()
            .unwrap()
            .report;
            (old.value(), old.standard_error())
        };
        let mut contract_task = || {
            let new = contract
                .compile_mc(r.valuation_date(), vec![u])
                .unwrap()
                .pricing_task(&source, PricingOperation::Evaluate)
                .unwrap()
                .compile()
                .unwrap()
                .evaluate()
                .unwrap()
                .report;
            (new.value(), new.standard_error())
        };
        for iteration in 0..=repetitions {
            let mut values = [(0.0, 0.0); 3];
            let mut times = [0.0; 3];
            let runs: [&mut dyn FnMut() -> (f64, f64); 3] =
                [&mut native_plan, &mut native_task, &mut contract_task];
            let order = if iteration % 2 == 0 {
                [0, 1, 2]
            } else {
                [2, 1, 0]
            };
            for index in order {
                let start = Instant::now();
                values[index] = runs[index]();
                times[index] = start.elapsed().as_secs_f64() * 1e3;
            }
            for value in &values[1..] {
                assert!((values[0].0 - value.0).abs() < 1e-11);
                assert!((values[0].1 - value.1).abs() < 1e-11);
            }
            if iteration > 0 {
                old_times.push(times[0]);
                native_task_times.push(times[1]);
                new_times.push(times[2]);
                paired_ratios.push(times[2] / times[1]);
            }
            result = json!({"pv":values[2].0,"standard_error":values[2].1,"native_pv":values[0].0});
        }
        result["native_plan_samples_ms"] = json!(old_times);
        result["native_task_samples_ms"] = json!(native_task_times);
        result["contract_samples_ms"] = json!(new_times);
        result["paired_contract_native_ratios"] = json!(paired_ratios);
        native_task_times.sort_by(f64::total_cmp);
        old_times.sort_by(f64::total_cmp);
        new_times.sort_by(f64::total_cmp);
        paired_ratios.sort_by(f64::total_cmp);
        result["native_total_ms"] = json!(old_times[repetitions / 2]);
        result["native_task_total_ms"] = json!(native_task_times[repetitions / 2]);
        result["contract_total_ms"] = json!(new_times[repetitions / 2]);
        result["median_paired_ratio"] = json!(paired_ratios[repetitions / 2]);
        result["timing_repetitions_after_warmup"] = json!(repetitions);
        mc.insert(name.into(), result);
    }
    println!("{}", serde_json::to_string_pretty(&json!({"schema_version":1,"paths":setting("CONTRACT_BENCH_PATHS", 1_000_000),"periods":periods,"mc_sampling_units":samples,"worker_threads":workers,"seed":7,
        "allocation_metric":"Reserved workspace bytes and observed capacity growth; not process-global allocator counts. Model path buffers are owned by existing engines.",
        "european":bench(&vanilla, Some(native.source_graph().unwrap())),
        "asian":bench(&asian, Some(old_asian.source_graph().unwrap())),
        "barrier":bench(&barrier, Some(old_barrier.source_graph().unwrap())),
        "autocall":bench(&autocall, None),"mc":mc})).unwrap());
}
