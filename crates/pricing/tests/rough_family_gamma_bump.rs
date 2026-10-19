//! Explicit finite-bump physical Spot Gamma with paired width diagnostics.
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

fn pure(v: Value, model: RoughVolatilityModel, workers: u32) -> RoughVolatilityPricingPlan {
    RoughVolatilityPricingPlan::compile(
        &parse(v),
        model,
        0.25,
        ExecutionPolicy::new(workers, Some(64)).unwrap(),
    )
    .unwrap()
}

#[test]
fn pure_and_sticky_gamma_match_full_physical_spot_recompilation() {
    let mut cases = families();
    for beta in [0.0, 0.5] {
        cases.push(
            RoughSabr::new(
                0.2,
                0.4,
                -0.6,
                beta,
                ForwardVarianceCurve::constant(0.04).unwrap(),
            )
            .unwrap()
            .into(),
        );
    }
    for (j, model) in cases.iter().enumerate() {
        for qmc in [false, true] {
            let h = 1.0;
            let p = pure(payload(false, qmc, 0.0, 0.0), model.clone(), 1);
            let g = p.evaluate_gamma_bump(h).unwrap();
            assert_eq!(g.price, p.evaluate().unwrap());
            for (width, gamma) in [(h, g.gamma), (h / 2.0, g.half_bump_gamma)] {
                let up = pure(payload(false, qmc, width, 0.0), model.clone(), 1)
                    .evaluate()
                    .unwrap()
                    .value;
                let dn = pure(payload(false, qmc, -width, 0.0), model.clone(), 1)
                    .evaluate()
                    .unwrap()
                    .value;
                close(
                    gamma,
                    (up - 2.0 * g.price.value + dn) / (width * width),
                    2e-10,
                );
            }
            assert_eq!(g.payoff_evaluations, 5 * g.price.evaluated_paths);
            let other = pure(payload(false, qmc, 0.0, 0.0), model.clone(), 3)
                .evaluate_gamma_bump(h)
                .unwrap();
            assert_eq!(gamma_numbers(&g), gamma_numbers(&other));
            assert_eq!(g.price.value.to_bits(), other.price.value.to_bits());
            assert_eq!(
                g.price.standard_error.to_bits(),
                other.price.standard_error.to_bits()
            );
            assert_ne!(g.risk_fingerprint, other.risk_fingerprint); // execution policy is part of the parent plan identity
            if j >= 6 {
                continue;
            }
            let l = build(payload(true, qmc, 0.0, 0.0), model.clone(), 1, false);
            let a = l.evaluate_sticky_moneyness_gamma_bump(h).unwrap();
            assert_eq!(a.price, l.evaluate().unwrap());
            for (width, gamma) in [(h, a.gamma), (h / 2.0, a.half_bump_gamma)] {
                let up = build(payload(true, qmc, width, 0.0), model.clone(), 1, false)
                    .evaluate()
                    .unwrap()
                    .value;
                let dn = build(payload(true, qmc, -width, 0.0), model.clone(), 1, false)
                    .evaluate()
                    .unwrap()
                    .value;
                close(
                    gamma,
                    (up - 2.0 * a.price.value + dn) / (width * width),
                    2e-10,
                );
            }
        }
    }
}

#[test]
fn frozen_gamma_and_paired_diagnostics_match_independent_sampling_units() {
    let mut covariance_effect: f64 = 0.0;
    for model in families() {
        for qmc in [false, true] {
            let v = payload(true, qmc, 0.0, 0.0);
            let p = build(v.clone(), model.clone(), 1, false);
            let g = p.evaluate_frozen_leverage_gamma_bump(1.0).unwrap();
            assert_eq!(g.price, p.evaluate().unwrap());
            let values: Vec<_> = [0.0, 1.0, -1.0, 0.5, -0.5]
                .into_iter()
                .map(|s| independent_frozen_units(&p, v.clone(), s))
                .collect();
            let mut full = vec![];
            let mut half = vec![];
            let mut difference = vec![];
            for (i, &base_price) in values[0].iter().enumerate() {
                let a = values[1][i] + values[2][i] - 2.0 * base_price;
                let b = 4.0 * (values[3][i] + values[4][i] - 2.0 * base_price);
                full.push(a);
                half.push(b);
                difference.push(b - a);
            }
            for (units, mean, se) in [
                (&full, g.gamma, g.gamma_standard_error),
                (&half, g.half_bump_gamma, g.half_bump_standard_error),
                (
                    &difference,
                    g.bump_difference,
                    g.bump_difference_standard_error,
                ),
            ] {
                let r = mean_se(units);
                close(mean, r.0, 2e-10);
                close(se, r.1, 2e-10);
            }
            covariance_effect = covariance_effect.max(
                (g.bump_difference_standard_error
                    - g.gamma_standard_error.hypot(g.half_bump_standard_error))
                .abs(),
            );
            let other = build(v, model.clone(), 3, false)
                .evaluate_frozen_leverage_gamma_bump(1.0)
                .unwrap();
            assert_eq!(gamma_numbers(&g), gamma_numbers(&other));
            assert_eq!(g.price.value.to_bits(), other.price.value.to_bits());
            assert_eq!(
                g.price.standard_error.to_bits(),
                other.price.standard_error.to_bits()
            );
            assert_ne!(g.risk_fingerprint, other.risk_fingerprint);
        }
    }
    assert!(
        covariance_effect > 1e-4,
        "must not treat full/half estimates as independent"
    );
}

#[test]
fn widths_boundaries_fingerprints_and_prior_risks_are_preserved() {
    let model = families()[0].clone();
    let p = pure(payload(false, false, 0.0, 0.0), model.clone(), 1);
    let l = build(payload(true, true, 0.0, 0.0), model, 1, true);
    let local_before = l.evaluate_local_variance_risk().unwrap();
    let delta_before = p.evaluate_delta().unwrap();
    for h in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-8, 100.0, 94.0] {
        assert!(p.evaluate_gamma_bump(h).is_err());
        assert!(l.evaluate_frozen_leverage_gamma_bump(h).is_err());
        assert!(l.evaluate_sticky_moneyness_gamma_bump(h).is_err());
    }
    let a = l.evaluate_frozen_leverage_gamma_bump(1.0).unwrap();
    let b = l.evaluate_sticky_moneyness_gamma_bump(1.0).unwrap();
    assert_eq!(a.price, b.price);
    assert_ne!(a.risk_fingerprint, b.risk_fingerprint);
    assert_ne!(
        a.risk_fingerprint,
        l.evaluate_frozen_leverage_gamma_bump(0.5)
            .unwrap()
            .risk_fingerprint
    );
    assert_eq!(p.evaluate_delta().unwrap(), delta_before);
    assert_eq!(l.evaluate_local_variance_risk().unwrap(), local_before);
    close(
        a.half_bump_gamma,
        l.evaluate_frozen_leverage_gamma_bump(0.5).unwrap().gamma,
        2e-13,
    );
}

#[test]
fn time_zero_proportional_and_future_cash_gamma_match_recompilation() {
    let model = families()[3].clone();
    let mut v = payload(false, false, 0.0, 0.0);
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.0,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":2.0,"beta":0.05}},
        {"event_id":2,"ex_time":0.25,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":3.0,"beta":0.07}},
        {"event_id":3,"ex_time":1.5,"quote":{"type":"fixed_cash","amount":4.0}}]);
    let p = pure(v.clone(), model.clone(), 1);
    let g = p.evaluate_gamma_bump(1.0).unwrap();
    v["market"]["spot"] = json!(101.0);
    let up = pure(v.clone(), model.clone(), 1).evaluate().unwrap().value;
    v["market"]["spot"] = json!(99.0);
    let dn = pure(v, model, 1).evaluate().unwrap().value;
    close(g.gamma, up + dn - 2.0 * g.price.value, 2e-10);
}

#[test]
#[ignore = "release numerical acceptance"]
fn six_family_black_gamma_and_finite_width_bias_are_separate() {
    let v0 = 0.04;
    let curve = ForwardVarianceCurve::constant(v0).unwrap();
    let models: Vec<RoughVolatilityModel> = vec![
        RoughHeston::new(0.1, v0, 0.7, v0, 0.0, -0.6)
            .unwrap()
            .into(),
        LiftedHeston::new(
            v0,
            0.7,
            v0,
            0.0,
            -0.6,
            vec![0.2, 0.4, 0.5],
            vec![0.1, 1.0, 8.0],
        )
        .unwrap()
        .into(),
        QuadraticRoughHeston::new(0.2, 0.1, 0.8, 0.3, 0.0, 0.05, v0)
            .unwrap()
            .into(),
        MixedRoughBergomi::new(0.1, -0.6, vec![1.0], vec![0.0], curve.clone())
            .unwrap()
            .into(),
        RoughSabr::new(0.1, 0.0, -0.6, 1.0, curve).unwrap().into(),
        Rfsv::new(0.2, 0.7, 0.0, 0.2_f64.ln(), None).unwrap().into(),
    ];
    // Reference values are computed independently with erfc Black prices in
    // the companion Python test: true Gamma and its finite h/h2 differences.
    let gamma: f64 = 0.01984762737385059;
    let fd_full = 0.019844113046403322;
    let fd_half = 0.019846748725001362;
    assert!((fd_half - gamma).abs() < (fd_full - gamma).abs());
    for (j, model) in models.into_iter().enumerate() {
        for seed in [91, 1973] {
            let mut v = payload(false, true, 0.0, 0.0);
            v["market"]["discrete_dividends"] = json!([]);
            for curve in ["discount_curve", "dividend_curve"] {
                v["market"][curve]["discount_factors"] = json!([1.0, 1.0]);
            }
            v["engine"]["points_per_scramble"] = json!(4096);
            v["engine"]["scramble_count"] = json!(8);
            v["engine"]["master_scramble_seed"] = json!(seed);
            let p = pure(v.clone(), model.clone(), 1);
            let a = p.evaluate_gamma_bump(1.0).unwrap();
            v["model"] = json!({"type":"local_volatility","local_variance_grid":{
                "time_nodes":[0.0,0.25,0.5,0.75,1.0],"log_forward_moneyness_nodes":[-0.4,0.4],
                "shape":[5,2],"values":vec![v0;10],"floor":1e-8,"cap":4.0}});
            let l = build(v, model.clone(), 1, false);
            let b = l.evaluate_frozen_leverage_gamma_bump(1.0).unwrap();
            let c = l.evaluate_sticky_moneyness_gamma_bump(1.0).unwrap();
            for (g, se, r) in [
                (a.gamma, a.gamma_standard_error, fd_full),
                (a.half_bump_gamma, a.half_bump_standard_error, fd_half),
                (b.gamma, b.gamma_standard_error, fd_full),
                (c.half_bump_gamma, c.half_bump_standard_error, fd_half),
            ] {
                assert!(se > 0.0 && se <= 0.002, "se={se}");
                assert!(
                    (g - r).abs() <= 5.0 * se + 2e-4,
                    "gamma={g} ref={r} se={se}"
                );
            }
            close(b.gamma, c.gamma, 2e-10);
            println!(
                "ROUGH_GAMMA_BLACK family={j} seed={seed} gamma={:.15e} half={:.15e} se={:.15e} half_se={:.15e} gap={:.15e} gap_se={:.15e}",
                a.gamma,
                a.half_bump_gamma,
                a.gamma_standard_error,
                a.half_bump_standard_error,
                a.bump_difference,
                a.bump_difference_standard_error
            );
        }
    }
}

fn gamma_numbers<P>(g: &RoughGammaBump<P>) -> [u64; 7] {
    [
        g.gamma,
        g.gamma_standard_error,
        g.half_bump_gamma,
        g.half_bump_standard_error,
        g.bump_difference,
        g.bump_difference_standard_error,
        g.spot_bump,
    ]
    .map(f64::to_bits)
}
