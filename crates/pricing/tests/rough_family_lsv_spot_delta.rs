//! Explicit fixed-leverage and sticky-relative-target physical Spot Delta.
use pricing::market::LocalVarianceGrid;
use pricing::mc::lsv::LsvParticleConfig;
use pricing::mc::{ExecutionPolicy, LocalVolTimeGrid, RandomDomain};
use pricing::rough_volatility::*;
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn families() -> Vec<RoughVolatilityModel> {
    let h = RoughHeston::new(0.2, 0.04, 0.7, 0.055, 0.15, -0.6).unwrap();
    let c = ForwardVarianceCurve::constant(0.04).unwrap();
    vec![
        h.clone().into(),
        LiftedHeston::from_rough(&h, 8, 2.5).unwrap().into(),
        QuadraticRoughHeston::new(0.2, 0.1, 0.8, 0.3, 0.25, 0.05, 0.035)
            .unwrap()
            .into(),
        MixedRoughBergomi::new(0.2, -0.6, vec![0.35, 0.65], vec![0.25, 0.6], c.clone())
            .unwrap()
            .into(),
        RoughSabr::new(0.2, 0.4, -0.6, 1.0, c).unwrap().into(),
        Rfsv::new(0.2, 0.7, 0.12, -1.7, None).unwrap().into(),
    ]
}
fn grid() -> LocalVarianceGrid {
    LocalVarianceGrid::new(
        vec![0.0, 0.15, 0.4, 0.7, 1.0],
        vec![-0.6, -0.2, 0.13, 0.45, 0.8],
        (0..25)
            .map(|i| 0.04 + 0.002 * (i % 5) as f64 + 0.001 * (i / 5) as f64)
            .collect(),
        1e-8,
        4.0,
    )
    .unwrap()
}
fn particles(trace: bool) -> LsvParticleConfig {
    LsvParticleConfig::new(128, 429, 0.5, 3.0, trace).unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * (1.0 + b.abs()),
        "{a:.14e} vs {b:.14e}; gap {}",
        (a - b).abs()
    );
}
fn parse(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn payload(lsv: bool, qmc: bool, spot_bump: f64, node_bump: f64) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["market"]["spot"] = json!(100.0 + spot_bump);
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.25,"quote":{"type":"fixed_cash","amount":3.0}},
        {"event_id":2,"ex_time":1.5,"quote":{"type":"fixed_cash","amount":4.0}}]);
    if lsv {
        let g = grid();
        let values = g
            .values()
            .iter()
            .enumerate()
            .map(|(i, v)| v + node_bump * (0.7 * i as f64).cos())
            .collect::<Vec<_>>();
        v["model"] = json!({"type":"local_volatility","local_variance_grid":{
            "time_nodes":g.time_nodes(),"log_forward_moneyness_nodes":g.log_moneyness_nodes(),
            "shape":[5,5],"values":values,"floor":1e-8,"cap":4.0}});
    }
    v["engine"] = if qmc {
        json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":64,"scramble_count":4,
        "master_scramble_seed":819,"variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    } else {
        json!({"type":"pseudo_monte_carlo","independent_sampling_units":128,"master_seed":819,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    };
    v
}

fn build(
    v: Value,
    model: RoughVolatilityModel,
    workers: u32,
    trace: bool,
) -> RoughFamilyLsvPricingPlan {
    RoughFamilyLsvPricingPlan::compile(
        &parse(v),
        model,
        particles(trace),
        ExecutionPolicy::new(workers, Some(64)).unwrap(),
    )
    .unwrap()
}

#[test]
fn sticky_relative_target_delta_matches_full_spot_recompilation_and_recalibration() {
    let mut count = 0;
    for model in families() {
        for qmc in [false, true] {
            let v = payload(true, qmc, 0.0, 0.0);
            let p = build(v.clone(), model.clone(), 1, false);
            let a = p.evaluate_sticky_moneyness_delta().unwrap();
            assert_eq!(a.price, p.evaluate().unwrap());
            assert_eq!(
                a.convention,
                RoughFamilyLsvDeltaConvention::StickyRelativeLocalVariance
            );
            assert!((a.delta_standard_error > 0.0) && a.delta_standard_error.is_finite());
            for e in [1e-4, 5e-5] {
                let up = build(payload(true, qmc, e, 0.0), model.clone(), 1, false);
                let dn = build(payload(true, qmc, -e, 0.0), model.clone(), 1, false);
                for (x, y) in p
                    .calibration()
                    .surface()
                    .squared_leverage()
                    .iter()
                    .zip(up.calibration().surface().squared_leverage())
                {
                    close(*x, *y, 2e-13);
                }
                let fd = (up.evaluate().unwrap().value - dn.evaluate().unwrap().value) / (2.0 * e);
                close(a.delta, fd, 3e-8);
                println!(
                    "LSV_SPOT_RECALIBRATED case={count} bump={e} delta={:.15e} fd={fd:.15e}",
                    a.delta
                );
            }
            count += 1;
        }
    }
}

// Independent European payoff / escrow reconstruction, without the pricing
// adapter's payoff graph or its adjoints. The public finite path/RNG is shared.
fn independent_frozen_units(p: &RoughFamilyLsvPricingPlan, v: Value, shift: f64) -> Vec<f64> {
    use pricing::market::DiscountCurve;
    use pricing::mc::{BrownianBridgePlan, EngineConfig, RqmcPlan, inverse_standard_normal};
    let req = parse(v);
    let market = req.market().equity().forward();
    let spot = market.spot().get();
    let reserve = market
        .discrete_dividends()
        .map_or(0.0, |d| d.initial_reserve());
    let initial = spot + shift * spot / (spot - reserve);
    let t = p.calibration().surface().times();
    let grid = LocalVolTimeGrid::compile(t.to_vec(), 1.0).unwrap();
    let path = p.calibration().pricing_plan(&grid).unwrap();
    let n = t.len() - 1;
    let blocks = match p.calibration().model() {
        RoughVolatilityModel::Rfsv(_) | RoughVolatilityModel::QuadraticRoughHeston(_) => 1,
        _ => 2,
    };
    let fwd = market.evaluate(1.0).unwrap();
    let df = market.discount_curve().discount(1.0).unwrap();
    let coord = fwd.affine_coordinate;
    let sample = |mut z: Vec<f64>, anti: bool, bridge: bool| {
        if bridge {
            let b = BrownianBridgePlan::compile(t.to_vec(), 1).unwrap();
            for zs in z[..blocks * n].chunks_exact_mut(n) {
                let x = b.apply_one_factor(zs).unwrap();
                zs.copy_from_slice(&x);
            }
        }
        let signs = if anti { &[1.0, -1.0][..] } else { &[1.0][..] };
        let mut value = 0.0;
        for sign in signs {
            let zz = z.iter().map(|z| z * sign).collect::<Vec<_>>();
            let f = *path
                .evolve_path(initial, &zz)
                .unwrap()
                .states()
                .last()
                .unwrap();
            let physical = coord.a() * spot + coord.b() * (f * (fwd.forward / spot));
            value += df * (physical - 100.0).max(0.0) / signs.len() as f64;
        }
        value
    };
    match req.engine() {
        EngineConfig::PseudoMonteCarlo(c) => (0..c.independent_sampling_units().get())
            .map(|i| {
                sample(
                    path.pseudo_shocks(c.master_seed(), i, RandomDomain::Valuation),
                    c.variance_reduction().antithetic(),
                    c.variance_reduction().brownian_bridge(),
                )
            })
            .collect(),
        EngineConfig::RandomizedQuasiMonteCarlo(c) => {
            let q = RqmcPlan::compile(c, path.random_dimension()).unwrap();
            (0..c.scramble_count().get())
                .map(|s| {
                    let values = (0..c.points_per_scramble().get())
                        .map(|i| {
                            let z = (0..path.random_dimension())
                                .map(|d| {
                                    inverse_standard_normal(q.uniform(s, i, d).unwrap()).unwrap()
                                })
                                .collect();
                            sample(
                                z,
                                c.variance_reduction().antithetic(),
                                c.variance_reduction().brownian_bridge(),
                            )
                        })
                        .collect::<Vec<_>>();
                    values.iter().sum::<f64>() / values.len() as f64
                })
                .collect()
        }
    }
}
fn mean_se(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let se = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n * (n - 1.0))).sqrt();
    (mean, se)
}

#[test]
fn frozen_leverage_delta_and_se_match_independent_spot_bumps_in_both_estimators() {
    let mut largest_gap: f64 = 0.0;
    let mut largest_convention_difference: f64 = 0.0;
    for model in families() {
        for qmc in [false, true] {
            let v = payload(true, qmc, 0.0, 0.0);
            let p = build(v.clone(), model.clone(), 1, false);
            let a = p.evaluate_frozen_leverage_delta().unwrap();
            assert_eq!(a.price, p.evaluate().unwrap());
            assert_eq!(a.convention, RoughFamilyLsvDeltaConvention::FrozenLeverage);
            let expected_price = mean_se(&independent_frozen_units(&p, v.clone(), 0.0));
            close(a.price.value, expected_price.0, 2e-13);
            close(a.price.standard_error, expected_price.1, 2e-13);
            for e in [1e-4, 5e-5] {
                let up = independent_frozen_units(&p, v.clone(), e);
                let dn = independent_frozen_units(&p, v.clone(), -e);
                let derivatives = up
                    .iter()
                    .zip(dn)
                    .map(|(u, d)| (u - d) / (2.0 * e))
                    .collect::<Vec<_>>();
                let (mean, se) = mean_se(&derivatives);
                close(a.delta, mean, 3e-8);
                close(a.delta_standard_error, se, 3e-8);
                largest_gap = largest_gap.max((a.delta - mean).abs());
            }
            let moving = p.evaluate_sticky_moneyness_delta().unwrap();
            largest_convention_difference =
                largest_convention_difference.max((a.delta - moving.delta).abs());
        }
    }
    assert!(
        largest_convention_difference > 1e-4,
        "must exercise unequal conventions"
    );
    println!(
        "LSV_SPOT_FROZEN max_bump_gap={largest_gap:.15e} max_convention_difference={largest_convention_difference:.15e}"
    );
}

#[test]
fn delta_preserves_price_and_local_risk_across_workers_trace_settings_and_sampling() {
    for model in families() {
        for qmc in [false, true] {
            let v = payload(true, qmc, 0.0, 0.0);
            let p = build(v.clone(), model.clone(), 1, true);
            let before = p.evaluate_local_variance_risk().unwrap();
            for convention in [true, false] {
                let run = |p: &RoughFamilyLsvPricingPlan| {
                    if convention {
                        p.evaluate_frozen_leverage_delta().unwrap()
                    } else {
                        p.evaluate_sticky_moneyness_delta().unwrap()
                    }
                };
                let a = run(&p);
                for (workers, trace) in [(3, true), (1, false)] {
                    let other = run(&build(v.clone(), model.clone(), workers, trace));
                    assert_eq!(a.delta.to_bits(), other.delta.to_bits());
                    assert_eq!(
                        a.delta_standard_error.to_bits(),
                        other.delta_standard_error.to_bits()
                    );
                    assert_eq!(a.price.value.to_bits(), other.price.value.to_bits());
                }
            }
            assert_eq!(before, p.evaluate_local_variance_risk().unwrap());
        }
    }
    // Also exercise neither variance-reduction option, without changing seeds.
    let mut v = payload(true, false, 0.0, 0.0);
    v["engine"]["variance_reduction"] = json!({"antithetic":false,"brownian_bridge":false});
    let p = build(v.clone(), families().remove(0), 1, false);
    let a = p.evaluate_frozen_leverage_delta().unwrap();
    let e = 1e-4;
    let up = independent_frozen_units(&p, v.clone(), e);
    let dn = independent_frozen_units(&p, v, -e);
    let (d, se) = mean_se(
        &up.iter()
            .zip(dn)
            .map(|(u, d)| (u - d) / (2.0 * e))
            .collect::<Vec<_>>(),
    );
    close(a.delta, d, 3e-8);
    close(a.delta_standard_error, se, 3e-8);
}

#[test]
fn initial_spatial_kink_is_not_silently_a_two_sided_frozen_delta() {
    let mut v = payload(true, false, 0.0, 0.0);
    v["model"]["local_variance_grid"]["log_forward_moneyness_nodes"] =
        json!([-0.6, -0.2, 0.0, 0.45, 0.8]);
    v["model"]["local_variance_grid"]["values"] = json!(
        (0..25)
            .map(|i| [0.04, 0.05, 0.04, 0.09, 0.1][i % 5])
            .collect::<Vec<_>>()
    );
    let p = build(v, families().remove(0), 1, false);
    assert!(p.evaluate().is_ok());
    assert!(
        p.evaluate_frozen_leverage_delta()
            .unwrap_err()
            .to_string()
            .contains("rough_lsv_delta_spatial_kink")
    );
    assert!(p.evaluate_sticky_moneyness_delta().is_ok());
}

#[test]
fn zero_time_and_future_cash_with_proportional_dividends_are_differentiated() {
    let mut v = payload(true, true, 0.0, 0.0);
    v["market"]["discrete_dividends"] = json!([
      {"event_id":1,"ex_time":0.0,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":2.0,"beta":0.05}},
      {"event_id":2,"ex_time":0.25,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":3.0,"beta":0.03}},
      {"event_id":3,"ex_time":1.5,"quote":{"type":"fixed_cash","amount":4.0}}]);
    // Use the JSON protocol's canonical field names for proportional dividends.
    for model in families() {
        let p = build(v.clone(), model.clone(), 1, false);
        for e in [1e-4, 5e-5] {
            let mut up = v.clone();
            up["market"]["spot"] = json!(100.0 + e);
            let mut dn = v.clone();
            dn["market"]["spot"] = json!(100.0 - e);
            close(
                p.evaluate_sticky_moneyness_delta().unwrap().delta,
                (build(up, model.clone(), 1, false).evaluate().unwrap().value
                    - build(dn, model.clone(), 1, false).evaluate().unwrap().value)
                    / (2.0 * e),
                3e-8,
            );
            let u = independent_frozen_units(&p, v.clone(), e);
            let d = independent_frozen_units(&p, v.clone(), -e);
            close(
                p.evaluate_frozen_leverage_delta().unwrap().delta,
                (mean_se(&u).0 - mean_se(&d).0) / (2.0 * e),
                3e-8,
            );
        }
    }
}

#[test]
#[ignore = "release numerical acceptance"]
fn six_family_lsv_spot_delta_black_limit_at_two_seeds() {
    let c = ForwardVarianceCurve::constant(0.04).unwrap();
    let h = RoughHeston::new(0.2, 0.04, 0.7, 0.04, 0.0, -0.6).unwrap();
    let models: Vec<RoughVolatilityModel> = vec![
        h.clone().into(),
        LiftedHeston::from_rough(&h, 8, 2.5).unwrap().into(),
        QuadraticRoughHeston::new(0.2, 0.1, 0.8, 0.0, 0.0, 0.05, 0.04)
            .unwrap()
            .into(),
        MixedRoughBergomi::new(0.2, -0.6, vec![1.0], vec![0.0], c.clone())
            .unwrap()
            .into(),
        RoughSabr::new(0.2, 0.0, -0.6, 1.0, c).unwrap().into(),
        Rfsv::new(0.2, 0.7, 0.0, 0.2_f64.ln(), None).unwrap().into(),
    ];
    for (j, m) in models.into_iter().enumerate() {
        for seed in [91, 1973] {
            let mut v = payload(true, true, 0.0, 0.0);
            v["market"]["discrete_dividends"] = json!([]);
            for curve in ["discount_curve", "dividend_curve"] {
                v["market"][curve]["discount_factors"] = json!([1.0, 1.0]);
            }
            v["model"]["local_variance_grid"]["values"] = json!(vec![0.04; 25]);
            v["engine"]["points_per_scramble"] = json!(2048);
            v["engine"]["scramble_count"] = json!(8);
            v["engine"]["master_scramble_seed"] = json!(seed);
            let p = build(v, m.clone(), 1, false);
            let a = p.evaluate_frozen_leverage_delta().unwrap();
            let b = p.evaluate_sticky_moneyness_delta().unwrap();
            for result in [&a, &b] {
                let gap = (result.delta - 0.539827837277029).abs();
                assert!(result.delta_standard_error > 0.0);
                assert!(gap <= 5.0 * result.delta_standard_error + 2e-4);
                assert!(
                    (result.price.value - 7.965567455405804).abs()
                        <= 5.0 * result.price.standard_error + 2e-4
                );
            }
            close(a.delta, b.delta, 2e-13);
            println!(
                "LSV_SPOT_BLACK family={j} seed={seed} price={:.15e} frozen={:.15e} sticky={:.15e} se={:.15e}",
                a.price.value, a.delta, b.delta, a.delta_standard_error
            );
        }
    }
}
