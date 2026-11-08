//! Compare dated Contract rights with the same compiled native MC tasks.
use pricing::contract::{
    AdvancedMcSettings, BarrierDirection, ContinuousBarrierTerms, InstrumentBuilder, OptionFeature,
};
use pricing::core::{CurrencyId, Date, UnderlyingId};
use pricing::mc::ExecutionPolicy;
use pricing::product::{
    BarrierDirection as NativeDirection, BarrierMonitoring, BarrierSpec, BarrierStyle,
    GraphLimitPolicy, OptionSide, ProductSpec, SourceGraphBuilder, SourceOpcode,
};
use pricing::{
    JsonLimits, PricingOperation, PricingSpecification, PricingTask, parse_request_json,
};
use serde_json::{Value, json};
use std::time::Instant;

fn compare(contract: PricingTask, native: PricingTask) -> Value {
    assert_eq!(
        contract.fingerprint().unwrap(),
        native.fingerprint().unwrap()
    );
    let plans = [contract.compile().unwrap(), native.compile().unwrap()];
    let expected = plans[1].evaluate().unwrap().report;
    let mut times = [Vec::new(), Vec::new()];
    for iteration in 0..8 {
        for index in [iteration % 2, 1 - iteration % 2] {
            let start = Instant::now();
            let result = plans[index].evaluate().unwrap().report;
            let elapsed = start.elapsed().as_secs_f64();
            assert_eq!(result, expected);
            if iteration != 0 {
                times[index].push(elapsed);
            }
        }
    }
    let median = |values: &[f64]| {
        let mut v = values.to_vec();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    json!({"pv":expected.value(),"standard_error":expected.standard_error(),"contract_seconds":times[0],"native_seconds":times[1],"median_ratio":median(&times[0])/median(&times[1])})
}
fn main() {
    let as_of: Date = "2026-09-04".parse().unwrap();
    let dates: Vec<Date> = ["2027-03-04", "2027-09-04"]
        .map(|d| d.parse().unwrap())
        .to_vec();
    let expiry = dates[1];
    let under = UnderlyingId::new(1);
    let ccy = CurrencyId::new(2);
    let builder = InstrumentBuilder::new("bench", ccy, under, "close").unwrap();
    let request = parse_request_json(
        include_bytes!("../../../fixtures/v3/pricing_request_american.golden.json"),
        JsonLimits::DEFAULT,
    )
    .unwrap();
    let exercise =
        PricingSpecification::from_request(request, ExecutionPolicy::new(2, Some(32)).unwrap());
    let native = PricingTask::new(exercise.clone(), PricingOperation::Evaluate).unwrap();
    let right = |feature, settings| {
        builder
            .vanilla_option(OptionSide::Put, 100., 1., expiry, feature)
            .unwrap()
            .pricing_task(&exercise, PricingOperation::Evaluate, &settings)
            .unwrap()
    };
    let bermudan = compare(
        right(
            OptionFeature::Bermudan {
                dates: dates.clone(),
            },
            Default::default(),
        ),
        native.clone(),
    );
    let american = compare(
        right(
            OptionFeature::American { start: as_of },
            AdvancedMcSettings {
                american_grid: Some(dates),
                ..Default::default()
            },
        ),
        native,
    );
    let native_barrier = BarrierSpec::new(
        under,
        ccy,
        expiry,
        100.,
        85.,
        1.,
        OptionSide::Put,
        NativeDirection::Down,
        BarrierStyle::KnockOut,
        BarrierMonitoring::Continuous,
        vec![expiry],
        Some(2.),
        expiry,
    )
    .unwrap();
    let mut numerics = exercise.numerics().clone();
    numerics.lsm = None;
    let template = PricingSpecification::new(
        as_of,
        ProductSpec::Barrier(native_barrier),
        exercise.market().clone(),
        exercise.model().clone(),
        exercise.engine(),
        exercise.risk().clone(),
        numerics,
    )
    .unwrap();
    let feature = OptionFeature::ContinuousBarrier(ContinuousBarrierTerms {
        start: as_of,
        end: expiry,
        payment_date: expiry,
        direction: BarrierDirection::Down,
        style: BarrierStyle::KnockOut,
        level: 85.,
        rebate: 2.,
    });
    let task = builder
        .vanilla_option(OptionSide::Put, 100., 1., expiry, feature)
        .unwrap()
        .pricing_task(
            &template,
            PricingOperation::Evaluate,
            &AdvancedMcSettings {
                brownian_bridge: true,
                ..Default::default()
            },
        )
        .unwrap();
    let barrier = compare(
        task,
        PricingTask::new(template, PricingOperation::Evaluate).unwrap(),
    );
    let mut graph = SourceGraphBuilder::new();
    let spot = graph
        .push(SourceOpcode::TerminalSpot {
            underlying: under,
            observation_date: expiry,
        })
        .unwrap();
    let one = graph.literal(1.).unwrap();
    let mut sum = spot;
    for _ in 0..128 {
        sum = graph
            .push(SourceOpcode::Add {
                left: sum,
                right: one,
            })
            .unwrap();
    }
    let payoff = graph
        .finish(vec![sum, spot, sum])
        .compile(GraphLimitPolicy::DEFAULT)
        .unwrap();
    let mut workspace = payoff.workspace();
    let bytes = workspace.allocated_bytes();
    for i in 0..10000 {
        let s = f64::from(i);
        assert_eq!(
            payoff
                .evaluate_reusing(&mut workspace, |_, _| Some(s), |_, _| None)
                .unwrap(),
            [s + 128., s, s + 128.]
        );
    }
    println!("{}", serde_json::to_string_pretty(&json!({"warmups":1,"measured_pairs":7,"workers":2,"valuation_units":1024,"training_units":512,"bermudan":bermudan,"american_grid":american,"continuous_barrier":barrier,"forward_state":{"ssa_slots":payoff.opcodes().len(),"registers":payoff.forward_value_slots(),"workspace_bytes":bytes,"capacity_growth":workspace.allocated_bytes()-bytes,"verified_paths":10000}})).unwrap());
}
