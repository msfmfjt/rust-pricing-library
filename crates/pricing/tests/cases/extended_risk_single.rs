use super::*;
use pricing::hull_white::HullWhiteEquityPricingPlan as HwPlan;
use pricing::lsv::{BergomiLsvPricingPlan, LsvLocalVarianceRisk, RoughBergomiLsvPricingPlan};
use pricing::market::{LocalVarianceGrid, MarketIvSurface};
use pricing::mc::hull_white::HullWhiteLsvTarget;
use pricing::models::{BergomiDynamics, HullWhite1Factor, HybridCorrelation};
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::Value;

pub(super) fn quotes() -> Vec<f64> {
    vec![0.238, 0.225, 0.222, 0.244, 0.231, 0.228]
}
pub(super) fn target(values: Vec<f64>, s: Scenario) -> HullWhiteLsvTarget {
    // The single-asset escrowed HW API requires dividend times explicitly.
    let mut times: Vec<_> = (0..=s.steps).map(|i| i as f64 / s.steps as f64).collect();
    times.push(0.37);
    times.sort_by(f64::total_cmp);
    HullWhiteLsvTarget::from_market_iv(
        MarketIvSurface::new(vec![0.5, 1.0], vec![-1.0, 0.0, 1.0], values).unwrap(),
        times,
        vec![-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
        1e-8,
        4.0,
    )
    .unwrap()
}
fn payload(s: Scenario, grid: &LocalVarianceGrid) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["market"]["discount_curve"] =
        json!({"curve_id":10,"times":[0.0,0.4,1.0,2.0],"discount_factors":[1.0,0.985,0.95,0.88]});
    v["market"]["dividend_curve"] =
        json!({"curve_id":11,"times":[0.0,0.7,2.0],"discount_factors":[1.0,0.99,0.96]});
    // Ex-date between uniform steps exercises the escrowed reserve/curve chain.
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.37,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":1.5,"beta":0.02}}
    ]);
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{
        "time_nodes":grid.time_nodes(),"log_forward_moneyness_nodes":grid.log_moneyness_nodes(),
        "shape":[grid.time_nodes().len(),grid.log_moneyness_nodes().len()],"values":grid.values(),"floor":grid.floor(),"cap":grid.cap()}});
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":s.points,
        "scramble_count":s.scrambles,"master_scramble_seed":s.pricing_seed,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    v
}
fn request(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn bump_grid(g: &LocalVarianceGrid, d: &[f64], h: f64) -> LocalVarianceGrid {
    assert_eq!(g.values().len(), d.len());
    LocalVarianceGrid::new(
        g.time_nodes().to_vec(),
        g.log_moneyness_nodes().to_vec(),
        g.values().iter().zip(d).map(|(v, d)| v + h * d).collect(),
        g.floor(),
        g.cap(),
    )
    .unwrap()
}
fn bergomi<F: BergomiDynamics>(
    factor: F,
    s: Scenario,
    g: &LocalVarianceGrid,
    risk: bool,
) -> (f64, Option<LsvLocalVarianceRisk>) {
    let p = BergomiLsvPricingPlan::compile(
        &request(payload(s, g)),
        factor,
        s.particles(risk),
        s.policy(),
    )
    .unwrap();
    let price = p.evaluate().unwrap().value;
    (
        price,
        risk.then(|| p.evaluate_local_variance_risk().unwrap()),
    )
}
fn det(
    f: Factor,
    s: Scenario,
    g: &LocalVarianceGrid,
    risk: bool,
) -> (f64, Option<LsvLocalVarianceRisk>) {
    match f {
        Factor::One => bergomi(f.one(), s, g, risk),
        Factor::Two => bergomi(f.two(), s, g, risk),
        Factor::Rough => {
            let p = RoughBergomiLsvPricingPlan::compile(
                &request(payload(s, g)),
                f.rough(),
                s.particles(risk),
                s.policy(),
            )
            .unwrap();
            (
                p.evaluate().unwrap().value,
                risk.then(|| p.evaluate_local_variance_risk().unwrap()),
            )
        }
    }
}
pub(super) fn deterministic(f: Factor) {
    let mut failures = 0;
    for s in Scenario::all() {
        let t = target(quotes(), s);
        let g = t.grid();
        let (price, risk) = det(f, s, g, true);
        let risk = risk.unwrap();
        assert_eq!(price, risk.price.value);
        assert_eq!(risk.time_nodes.as_ref(), g.time_nodes());
        conditional_errors(risk.standard_errors.as_ref().unwrap(), g.values().len());
        let n = g.values().len();
        // Global, signed term/strike and isolated interior-node perturbations.
        let directions: Vec<_> = (0..4)
            .map(|axis| {
                (0..n)
                    .map(|i| match axis {
                        0 => 1.0,
                        1 => (0.71 * i as f64).cos(),
                        2 => f64::from(i == 3),
                        _ => f64::from(i == (g.time_nodes().len() / 2) * 7 + 3),
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        for (axis, direction) in directions.iter().enumerate() {
            failures += sweep(
                &format!("deterministic_{f:?}"),
                s,
                &format!("effective_variance_direction_{axis}"),
                dot(&risk.node_adjoints, direction),
                |h| det(f, s, &bump_grid(g, direction, h), false).0,
            );
        }
        println!(
            "EXTENDED_RISK_CASE {}",
            json!({"case":format!("deterministic_{f:?}"),"scenario":s,"price":price,
            "node_adjoints":risk.node_adjoints,"conditional_standard_errors":risk.standard_errors,
            "directions":4,"contract":"effective variance; not market IV VegaKT"})
        );
    }
    assert_eq!(failures, 0, "{f:?}: failed derivative sweeps");
}

fn compile_hw(f: Factor, s: Scenario, t: &HullWhiteLsvTarget, v: Value, trace: bool) -> HwPlan {
    let req = request(v);
    let rates = HullWhite1Factor::new(0.2, vec![0.0, 0.45], vec![0.012, 0.0168]).unwrap();
    let corr = HybridCorrelation::new(-0.4, 0.2, -0.05).unwrap();
    match f {
        Factor::One => HwPlan::compile_lsv(
            &req,
            t,
            f.one(),
            rates,
            corr,
            s.particles(trace),
            s.policy(),
        ),
        Factor::Two => HwPlan::compile_lsv_two_factor(
            &req,
            t,
            f.two(),
            rates,
            0.2,
            [-0.05, 0.02],
            s.particles(trace),
            s.policy(),
        ),
        Factor::Rough => HwPlan::compile_rough_lsv(
            &req,
            t,
            f.rough(),
            rates,
            corr,
            s.particles(trace),
            s.policy(),
        ),
    }
    .unwrap()
}
pub(super) fn hybrid(f: Factor) {
    let mut failures = 0;
    for s in Scenario::all() {
        let q = quotes();
        let t = target(q.clone(), s);
        let v = payload(s, t.grid());
        let p = compile_hw(f, s, &t, v.clone(), true);
        let r = p.evaluate_aad().unwrap();
        assert_eq!(r.price.value, p.evaluate().unwrap().value);
        let buckets = r.vega_kt_raw().unwrap();
        assert_eq!(buckets.len(), q.len());
        conditional_errors(r.vega_kt_standard_errors().unwrap(), q.len());
        assert!((r.vega().unwrap() - buckets.iter().sum::<f64>()).abs() < 1e-10);
        for (raw, scaled) in buckets.iter().zip(r.vega_kt_market_scaled().unwrap()) {
            assert_eq!(*raw * 0.01, scaled);
        }
        for (j, aad) in buckets
            .iter()
            .copied()
            .chain(std::iter::once(r.vega().unwrap()))
            .enumerate()
        {
            failures += sweep(
                &format!("hw_{f:?}"),
                s,
                &if j == q.len() {
                    "parallel_market_iv".into()
                } else {
                    format!("market_iv_{j}")
                },
                aad,
                |h| {
                    let qs = q
                        .iter()
                        .enumerate()
                        .map(|(i, x)| x + if i == j || j == q.len() { h } else { 0.0 })
                        .collect();
                    let shifted = target(qs, s);
                    compile_hw(f, s, &shifted, payload(s, shifted.grid()), false)
                        .evaluate()
                        .unwrap()
                        .value
                },
            );
        }
        for (label, bar, index) in [
            ("spot", r.delta(), 0),
            ("discount_curve", r.discount_log_df_adjoints()[2], 2),
            ("dividend_curve", r.dividend_log_df_adjoints()[1], 1),
        ] {
            failures += sweep(&format!("hw_{f:?}"), s, label, bar, |h| {
                let mut b = v.clone();
                if label == "spot" {
                    b["market"]["spot"] = json!(100.0 + h);
                } else {
                    let df = b["market"][label]["discount_factors"][index]
                        .as_f64()
                        .unwrap();
                    b["market"][label]["discount_factors"][index] = json!(df * h.exp());
                }
                compile_hw(f, s, &t, b, false).evaluate().unwrap().value
            });
        }
        // Omitting the density chain must be distinguishable on this nonflat
        // target, independently of the finite-difference agreement.
        let variance_only = t
            .reverse_market_iv(
                r.local_variance_adjoints(),
                &vec![0.0; t.grid().values().len()],
            )
            .unwrap();
        assert!(
            buckets
                .iter()
                .zip(&variance_only)
                .any(|(a, b)| (a - b).abs() > 1e-5)
        );
        println!(
            "EXTENDED_RISK_CASE {}",
            json!({"case":format!("hw_{f:?}"),"scenario":s,
            "price":r.price.value,"market_iv_adjoints":buckets,"variance_only_market_iv_adjoints":variance_only,
            "conditional_standard_errors":r.vega_kt_standard_errors(),"directions":10})
        );
    }
    assert_eq!(failures, 0, "{f:?}: failed derivative sweeps");
}
