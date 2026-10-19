//! Exact market-IV source binding, discrete chain rule and covariance-aware SE.
use pricing::market::{LocalVarianceGrid, MarketIvSurface};
use pricing::mc::lsv::LsvParticleConfig;
use pricing::mc::{
    BrownianBridgePlan, EngineConfig, ExecutionPolicy, LocalVolTimeGrid, RqmcPlan,
    inverse_standard_normal,
};
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

fn surface(shift: f64, direction: &[f64]) -> MarketIvSurface {
    let times = vec![0.2, 0.6, 1.2];
    let xs = vec![-0.9, -0.45, 0.05, 0.5, 0.9];
    let iv = (0..15)
        .map(|i| {
            let x = xs[i % 5];
            0.2 + 0.002 * (i / 5) as f64 - 0.003 * x + 0.002 * x * x + shift * direction[i]
        })
        .collect();
    MarketIvSurface::new(times, xs, iv).unwrap()
}
fn grid(s: &MarketIvSurface) -> LocalVarianceGrid {
    s.local_variance_grid(
        vec![0.0, 0.13, 0.37, 0.7, 1.0],
        vec![-0.55, -0.13, 0.18, 0.6],
        1e-8,
        4.0,
    )
    .unwrap()
}
fn payload(g: &LocalVarianceGrid, qmc: bool, cash: bool) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{
        "time_nodes":g.time_nodes(), "log_forward_moneyness_nodes":g.log_moneyness_nodes(),
        "shape":[g.time_nodes().len(),g.log_moneyness_nodes().len()],"values":g.values(),"floor":g.floor(),"cap":g.cap()}});
    if cash {
        v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.25,"quote":{"type":"fixed_cash","amount":3.0}},
        {"event_id":2,"ex_time":1.5,"quote":{"type":"fixed_cash","amount":4.0}}]);
    } else {
        for name in ["discount_curve", "dividend_curve"] {
            v["market"][name]["discount_factors"] = json!([1.0, 1.0]);
        }
    }
    v["engine"] = if qmc {
        json!({"type":"randomized_quasi_monte_carlo","master_scramble_seed":819,
        "points_per_scramble":64,"scramble_count":4,"variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    } else {
        json!({"type":"pseudo_monte_carlo","master_seed":819,"independent_sampling_units":128,
        "variance_reduction":{"antithetic":true,"brownian_bridge":false}})
    };
    v
}
fn parse(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn build(
    v: Value,
    model: RoughVolatilityModel,
    threads: u32,
    trace: bool,
) -> RoughFamilyLsvPricingPlan {
    RoughFamilyLsvPricingPlan::compile(
        &parse(v),
        model,
        LsvParticleConfig::new(128, 429, 0.5, 3.0, trace).unwrap(),
        ExecutionPolicy::new(threads, Some(64)).unwrap(),
    )
    .unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * (1.0 + b.abs()),
        "{a:.15e} vs {b:.15e}"
    );
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn moments(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (
        m,
        (v.iter().map(|a| (a - m).powi(2)).sum::<f64>() / (v.len() * (v.len() - 1)) as f64).sqrt(),
    )
}

#[test]
fn dupire_transpose_matches_quote_bumps_and_time_zero_uses_positive_row() {
    let d = (0..15).map(|i| (0.7 * i as f64).cos()).collect::<Vec<_>>();
    let s = surface(0.0, &d);
    // Short/long tails, a maturity knot, interior times, nonuniform spatial nodes.
    let ts = vec![0.0, 0.1, 0.2, 0.43, 0.9, 1.4];
    let xs = vec![-0.7, -0.1, 0.3, 0.7];
    let g = s
        .local_variance_grid(ts.clone(), xs.clone(), 1e-8, 4.0)
        .unwrap();
    let a = (0..g.values().len())
        .map(|i| (0.4 * i as f64).sin())
        .collect::<Vec<_>>();
    let quote = s.local_variance_pullback(&g, &a).unwrap();
    for e in [1e-6, 5e-7] {
        let up = surface(e, &d)
            .local_variance_grid(ts.clone(), xs.clone(), 1e-8, 4.0)
            .unwrap();
        let dn = surface(-e, &d)
            .local_variance_grid(ts.clone(), xs.clone(), 1e-8, 4.0)
            .unwrap();
        close(
            dot(&quote, &d),
            (dot(up.values(), &a) - dot(dn.values(), &a)) / (2.0 * e),
            2e-7,
        );
    }
    let mut initial = vec![0.0; g.values().len()];
    initial[1] = 1.0;
    let mut positive = vec![0.0; g.values().len()];
    positive[xs.len() + 1] = 1.0;
    assert_eq!(
        s.local_variance_pullback(&g, &initial).unwrap(),
        s.local_variance_pullback(&g, &positive).unwrap()
    );
    assert!(s.local_variance_pullback(&g, &[0.0]).is_err());
    initial[0] = f64::NAN;
    assert!(s.local_variance_pullback(&g, &initial).is_err());
}

#[test]
fn six_families_market_iv_risk_matches_full_recalibration_and_preserves_old_risk() {
    let d = (0..15).map(|i| (0.7 * i as f64).cos()).collect::<Vec<_>>();
    for (family, model) in families().into_iter().enumerate() {
        for qmc in [false, true] {
            let s = surface(0.0, &d);
            let g = grid(&s);
            let v = payload(&g, qmc, true);
            let p = build(v.clone(), model.clone(), 1, true);
            let old = p.evaluate_local_variance_risk().unwrap();
            let risk_plan = p.market_iv_risk_plan(s.clone()).unwrap();
            let a = risk_plan.evaluate().unwrap();
            assert_eq!(a.price, p.evaluate().unwrap());
            assert_eq!(a.price, old.price);
            assert_eq!(a.quote_adjoints.len(), 15);
            assert_eq!(a.standard_errors.is_some(), qmc);
            assert_eq!(a.parallel_standard_error.is_some(), qmc);
            assert_eq!(a.risk_fingerprint, risk_plan.risk_fingerprint());
            let expected = s.local_variance_pullback(&g, &old.node_adjoints).unwrap();
            for (a, b) in a.quote_adjoints.iter().zip(expected) {
                close(*a, b, 3e-13);
            }
            close(a.parallel_vega, a.quote_adjoints.iter().sum(), 3e-13);
            let other = build(v, model.clone(), 3, true)
                .market_iv_risk_plan(s)
                .unwrap()
                .evaluate()
                .unwrap();
            assert_eq!(a.quote_adjoints, other.quote_adjoints);
            assert_eq!(a.standard_errors, other.standard_errors);
            assert_eq!(a.parallel_standard_error, other.parallel_standard_error);
            for e in [1e-6, 5e-7] {
                let up = build(
                    payload(&grid(&surface(e, &d)), qmc, true),
                    model.clone(),
                    1,
                    false,
                )
                .evaluate()
                .unwrap();
                let dn = build(
                    payload(&grid(&surface(-e, &d)), qmc, true),
                    model.clone(),
                    1,
                    false,
                )
                .evaluate()
                .unwrap();
                let fd = (up.value - dn.value) / (2.0 * e);
                let got = dot(&a.quote_adjoints, &d);
                close(got, fd, 3e-6);
                println!(
                    "MARKET_IV_CHAIN family={family} qmc={qmc} bump={e} adjoint={got:.15e} fd={fd:.15e}"
                );
            }
            assert_eq!(old, p.evaluate_local_variance_risk().unwrap());
        }
    }
}

#[test]
fn source_mismatch_trace_repairs_and_bound_kinks_are_rejected() {
    let s = surface(0.0, &[0.0; 15]);
    let g = grid(&s);
    let m = families().remove(0);
    let p = build(payload(&g, true, true), m.clone(), 1, true);
    assert!(p.market_iv_risk_plan(surface(0.001, &[1.0; 15])).is_err());
    assert!(
        build(payload(&g, true, true), m, 1, false)
            .market_iv_risk_plan(s.clone())
            .is_err()
    );
    let mut values = g.values().to_vec();
    values[3] = f64::from_bits(values[3].to_bits() + 1);
    let altered = LocalVarianceGrid::new(
        g.time_nodes().to_vec(),
        g.log_moneyness_nodes().to_vec(),
        values,
        1e-8,
        4.0,
    )
    .unwrap();
    assert!(s.validate_local_variance_source(&altered).is_err());
    // Exact equality at a primal cap is not a differentiable strict Dupire node.
    let flat = MarketIvSurface::new(vec![0.25, 1.25], vec![-1.0, 1.0], vec![0.2; 4]).unwrap();
    let fg = flat
        .local_variance_grid(vec![0.0, 0.25, 1.0], vec![-0.5, 0.5], 1e-8, 4.0)
        .unwrap();
    let cap = fg
        .values()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let bound = flat
        .local_variance_grid(
            fg.time_nodes().to_vec(),
            fg.log_moneyness_nodes().to_vec(),
            1e-8,
            cap,
        )
        .unwrap();
    assert!(flat.validate_local_variance_source(&bound).is_err());
    assert!(
        flat.local_variance_grid(vec![0.0, 0.25, 1.0], vec![-0.5, 0.5], 0.1, 4.0)
            .is_err()
    );
    assert!(
        flat.local_variance_grid(vec![0.0, 0.25, 1.0], vec![-1.5, 0.5], 1e-8, 4.0)
            .is_err()
    );
}

#[test]
fn rqmc_quote_and_parallel_errors_keep_cross_node_covariance() {
    let s = surface(0.0, &[0.0; 15]);
    let original = grid(&s);
    let req = parse(payload(&original, true, false));
    let EngineConfig::RandomizedQuasiMonteCarlo(config) = req.engine() else {
        unreachable!()
    };
    let mut covariance_effect = 0.0_f64;
    for model in families() {
        let p = build(payload(&original, true, false), model.clone(), 1, true);
        let got = p
            .market_iv_risk_plan(s.clone())
            .unwrap()
            .evaluate()
            .unwrap();
        let cal = p.calibration();
        let times = cal.surface().times();
        let n = times.len() - 1;
        let path = cal
            .pricing_plan(&LocalVolTimeGrid::compile(times.to_vec(), 1.0).unwrap())
            .unwrap();
        let q = RqmcPlan::compile(config, path.random_dimension()).unwrap();
        let bridge = BrownianBridgePlan::compile(times.to_vec(), 1).unwrap();
        let blocks = match model {
            RoughVolatilityModel::Rfsv(_) | RoughVolatilityModel::QuadraticRoughHeston(_) => 1,
            _ => 2,
        };
        let mut samples = Vec::new();
        for scramble in 0..config.scramble_count().get() {
            let mut lev = vec![0.0; cal.surface().squared_leverage().len()];
            for i in 0..config.points_per_scramble().get() {
                let mut z = (0..path.random_dimension())
                    .map(|d| inverse_standard_normal(q.uniform(scramble, i, d).unwrap()).unwrap())
                    .collect::<Vec<_>>();
                for b in z[..blocks * n].chunks_exact_mut(n) {
                    let v = bridge.apply_one_factor(b).unwrap();
                    b.copy_from_slice(&v);
                }
                for sign in [1.0, -1.0] {
                    let record = path
                        .evolve_path(100.0, &z.iter().map(|v| v * sign).collect::<Vec<_>>())
                        .unwrap();
                    let mut seeds = vec![0.0; times.len()];
                    seeds[n] = if record.states()[n] > 100.0 { 1.0 } else { 0.0 };
                    let a = record.reverse(&seeds).unwrap();
                    for (acc, v) in lev.iter_mut().zip(a.squared_leverage) {
                        *acc += v * 0.5 / config.points_per_scramble().get() as f64;
                    }
                }
            }
            let refined = cal.reverse_leverage(&lev).unwrap();
            let mut a = vec![0.0; original.values().len()];
            let m = original.log_moneyness_nodes().len();
            for (row, &t) in cal.target().time_nodes().iter().enumerate() {
                for (col, &x) in original.log_moneyness_nodes().iter().enumerate() {
                    original.interpolate(t, x).unwrap().transpose_accumulate(
                        refined[row * m + col],
                        &mut a,
                        m,
                    );
                }
            }
            samples.push(s.local_variance_pullback(&original, &a).unwrap());
        }
        for j in 0..15 {
            let (mean, se) = moments(&samples.iter().map(|v| v[j]).collect::<Vec<_>>());
            close(got.quote_adjoints[j], mean, 5e-12);
            close(got.standard_errors.as_ref().unwrap()[j], se, 5e-12);
        }
        let (mean, se) = moments(
            &samples
                .iter()
                .map(|v| v.iter().sum::<f64>())
                .collect::<Vec<_>>(),
        );
        close(got.parallel_vega, mean, 5e-12);
        close(got.parallel_standard_error.unwrap(), se, 5e-12);
        let wrong = (got
            .standard_errors
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v * v)
            .sum::<f64>())
        .sqrt();
        covariance_effect = covariance_effect.max((wrong - se).abs());
    }
    assert!(
        covariance_effect > 1e-4,
        "test must exercise nonzero cross-node covariance"
    );
}

#[test]
#[ignore = "release numerical Black limit"]
fn black_limit_parallel_market_iv_vega() {
    let h = RoughHeston::new(0.2, 0.04, 0.7, 0.04, 0.0, -0.6).unwrap();
    let curve = ForwardVarianceCurve::constant(0.04).unwrap();
    let models: Vec<RoughVolatilityModel> = vec![
        h.clone().into(),
        LiftedHeston::from_rough(&h, 8, 2.5).unwrap().into(),
        QuadraticRoughHeston::new(0.2, 0.1, 0.8, 0.0, 0.0, 0.05, 0.04)
            .unwrap()
            .into(),
        MixedRoughBergomi::new(0.2, -0.6, vec![0.3, 0.7], vec![0.0, 0.0], curve.clone())
            .unwrap()
            .into(),
        RoughSabr::new(0.2, 0.0, -0.6, 1.0, curve).unwrap().into(),
        Rfsv::new(0.2, 0.7, 0.0, 0.2_f64.ln(), None).unwrap().into(),
    ];

    let s = MarketIvSurface::new(vec![0.25, 1.25], vec![-1.0, 0.0, 1.0], vec![0.2; 6]).unwrap();
    let g = s
        .local_variance_grid(
            (0..9).map(|i| i as f64 / 8.0).collect(),
            vec![-1.0, -0.5, 0.0, 0.5, 1.0],
            1e-8,
            4.0,
        )
        .unwrap();
    // Independent analytic constants, not a production Fourier/Black call.
    let (black_price, black_vega) = (7.965567455405804, 39.69525474770118);
    for (family, model) in models.into_iter().enumerate() {
        for seed in [91_u64, 1973] {
            let mut v = payload(&g, true, false);
            v["engine"]["master_scramble_seed"] = json!(seed);
            v["engine"]["points_per_scramble"] = json!(2048);
            v["engine"]["scramble_count"] = json!(8);
            let risk = build(v, model.clone(), 1, true)
                .market_iv_risk_plan(s.clone())
                .unwrap()
                .evaluate()
                .unwrap();
            let se = risk.parallel_standard_error.unwrap();
            assert!(
                (risk.price.value - black_price).abs() <= 5.0 * risk.price.standard_error + 2e-4
            );
            assert!((risk.parallel_vega - black_vega).abs() <= 5.0 * se + 0.02);
            assert!(se > 0.0 && se <= 0.1);
            println!(
                "MARKET_IV_BLACK family={family} seed={seed} price={:.15e} vega={:.15e} se={se:.15e}",
                risk.price.value, risk.parallel_vega
            );
        }
    }
}
