//! Rough/Buehler price-only contracts, causal hybrid history and independent prices.
use pricing::mc::{ExecutionPolicy, LocalVolTimeGrid};
use pricing::models::{Bergomi1Factor, RoughBergomi};
use pricing::risk::{GammaConfig, SpotBump};
use pricing::stochastic_dividends::{
    BuehlerDividendModel as Dividend, StochasticDividendPathPlan as Path,
    StochasticDividendPricingPlan as Plan,
};
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn request(points: u64, events: bool) -> PricingRequest {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":points,
        "scramble_count":8,"master_scramble_seed":612,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    v["market"]["discrete_dividends"] = if events {
        json!([
        {"event_id":1,"ex_time":0.5,"quote":{"type":"fixed_cash","amount":5.0}},
        {"event_id":2,"ex_time":1.0,"quote":{"type":"fixed_cash","amount":3.0}},
        {"event_id":3,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":25.0}}])
    } else {
        json!([{"event_id":1,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":25.0}}])
    };
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn dividend() -> Dividend {
    Dividend::new(0.7, 0.6, 0.35, -0.25).unwrap()
}
fn factor(h: f64, eta: f64) -> RoughBergomi {
    RoughBergomi::new(h, eta, -0.4).unwrap()
}
fn policy(workers: u32) -> ExecutionPolicy {
    ExecutionPolicy::new(workers, Some(64)).unwrap()
}

#[test]
fn zero_eta_projects_to_bs_and_brownian_boundary_matches_zero_reversion_bergomi() {
    let r = request(16, true);
    let m = r.market().equity().forward();
    let grid = LocalVolTimeGrid::compile(vec![0.13, 0.5, 1.0], 1.0).unwrap();
    let z4 = [
        0.2, -0.6, 1.1, 0.7, 0.4, -0.8, 0.3, -0.2, -0.3, 0.7, -0.5, 0.9,
    ];
    let z2: Vec<_> = z4
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|z| [z[0], z[1]])
        .collect();
    let z3: Vec<_> = z4
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|z| [z[0], z[1], z[2]])
        .collect();
    let bs = Path::compile(m, dividend(), 0.2, &grid).unwrap();
    for h in [0.03, 0.1, 0.5] {
        let rough =
            Path::compile_rough_bergomi(m, dividend(), 0.2, factor(h, 0.0), 0.15, &grid).unwrap();
        assert_eq!(rough.random_factor_count(), 4);
        assert_eq!(rough.random_dimension(), 12);
        assert_eq!(
            bs.evolve_path(&z2).unwrap(),
            rough.evolve_path(&z4).unwrap()
        );
    }
    // Rough eta is LOG-VARIANCE vol-of-vol; 1F nu is LOG-VOLATILITY vol-of-vol.
    let rough =
        Path::compile_rough_bergomi(m, dividend(), 0.2, factor(0.5, 0.6), 0.15, &grid).unwrap();
    let one = Path::compile_bergomi(
        m,
        dividend(),
        0.2,
        Bergomi1Factor::new(0.0, 0.3, -0.4).unwrap(),
        0.15,
        &grid,
    )
    .unwrap();
    for (a, b) in rough
        .evolve_path(&z4)
        .unwrap()
        .iter()
        .zip(one.evolve_path(&z3).unwrap())
    {
        assert!((a.equity() - b.equity()).abs() < 2e-14);
        assert!((a.dividend() - b.dividend()).abs() < 2e-14);
    }
    let mut changed = z4;
    for j in [3, 7, 11] {
        changed[j] += 13.0;
    }
    assert_eq!(
        rough.evolve_path(&z4).unwrap(),
        rough.evolve_path(&changed).unwrap()
    );
}

#[test]
fn irregular_history_cash_events_and_no_future_lookahead_match_independent_construction() {
    let r = request(16, true);
    let m = r.market().equity().forward();
    let grid = LocalVolTimeGrid::compile(vec![0.13, 0.5, 1.0], 1.0).unwrap();
    let plan =
        Path::compile_rough_bergomi(m, dividend(), 0.2, factor(0.1, 0.6), 0.15, &grid).unwrap();
    let normals = [
        0.2, -0.6, 1.1, 0.7, 0.4, -0.8, 0.3, -0.2, -0.3, 0.7, -0.5, 0.9,
    ];
    let actual = plan.evolve_path(&normals).unwrap();
    let times: [f64; 4] = [0.0, 0.13, 0.5, 1.0];
    let (h, eta, kd, alpha, nu_d, sd, sv, dv) = (
        0.1_f64, 0.6_f64, 0.7_f64, 0.6_f64, 0.35_f64, -0.25_f64, -0.4_f64, 0.15_f64,
    );
    let q = (1.0 - sd * sd).sqrt();
    let lv = (dv - sd * sv) / q;
    let last = (1.0 - sv * sv - lv * lv).sqrt();
    let mut dw = Vec::new();
    let mut near = Vec::new();
    let mut f = 1.0;
    let mut y = 1.0;
    for (i, z) in normals.as_chunks::<4>().0.iter().enumerate() {
        let dt = times[i + 1] - times[i];
        let mut x = 0.0;
        let mut variance = 0.0;
        if i > 0 {
            x = near[i - 1];
            variance = (times[i] - times[i - 1]).powf(2.0 * h);
            for (j, &past_dw) in dw.iter().enumerate().take(i - 1) {
                let p = h + 0.5;
                let w = (2.0 * h).sqrt()
                    * ((times[i] - times[j]).powf(p) - (times[i] - times[j + 1]).powf(p))
                    / (p * (times[j + 1] - times[j]));
                x += w * past_dw;
                variance += w * w * (times[j + 1] - times[j]);
            }
        }
        let sigma = 0.2 * (0.5 * eta * x - 0.25 * eta * eta * variance).exp();
        let a = (-kd * dt / 2.0).exp();
        let b = 1.0 - a;
        let half = a * y + b * (alpha * f + 1.0 - alpha);
        f *= (-0.5 * sigma * sigma * dt + sigma * dt.sqrt() * z[0]).exp();
        y = a * half * (-0.5 * nu_d * nu_d * dt + nu_d * dt.sqrt() * (sd * z[0] + q * z[1])).exp()
            + b * (alpha * f + 1.0 - alpha);
        assert!((actual[i + 1].equity() - f).abs() < 3e-14);
        assert!((actual[i + 1].dividend() - y).abs() < 3e-14);
        let increment = dt.sqrt() * (sv * z[0] + lv * z[1] + last * z[2]);
        dw.push(increment);
        near.push(
            (2.0 * h).sqrt() * dt.powf(h - 0.5) / (h + 0.5) * increment
                + dt.powf(h) * (1.0 - 2.0 * h / (h + 0.5).powi(2)).sqrt() * z[3],
        );
        let g = |t: f64| (0.98_f64 / 0.95).powf(t);
        let events = [(0.5, 5.0), (1.0, 3.0), (1.4, 25.0)];
        let residual = 100.0 - events.iter().map(|(t, c)| c / g(*t)).sum::<f64>();
        let t = times[i + 1];
        let mut post = g(t) * residual * f;
        let mut cash = 0.0;
        for (ex, amount) in events {
            if ex > t {
                let w = (-kd * (ex - t)).exp();
                post += g(t) / g(ex) * amount * (w * y + (1.0 - w) * (alpha * f + 1.0 - alpha));
            }
            if ex == t {
                cash += amount * y;
            }
        }
        let observed = plan.nodes()[i + 1].spots(actual[i + 1]).unwrap();
        assert!((observed.0 - post).abs() < 1e-12);
        if cash > 0.0 {
            assert!((observed.1.unwrap() - post - cash).abs() < 1e-12);
        }
    }
    let mut future = normals;
    for z in &mut future[8..] {
        *z += 4.0;
    }
    assert_eq!(&actual[..3], &plan.evolve_path(&future).unwrap()[..3]);
}

#[test]
fn independent_two_step_price_oracle_and_metadata() {
    let r = request(8192, false);
    // Independent math/NumPy quadrature order 24; order 20/24 gap < 2e-7.
    for (h, expected) in [(0.1, 7.7467332042848), (0.5, 7.765235145769844)] {
        let plan =
            Plan::compile_rough_bergomi(&r, dividend(), factor(h, 0.6), 0.15, 0.5, policy(2))
                .unwrap();
        let p = plan.evaluate().unwrap();
        println!(
            "H={h}: price={} se={} independent_reference={expected}",
            p.value, p.standard_error
        );
        assert!((p.value - expected).abs() < 6.0 * p.standard_error + 0.002);
        assert_eq!(p.independent_sampling_units, 8);
        assert_eq!(p.evaluated_paths, 131072);
        assert_eq!(p.scheme, plan.scheme());
        assert_eq!(p.plan_fingerprint, plan.plan_fingerprint());
    }
}

#[test]
fn full_psd_inputs_history_limit_and_unsupported_risk_fail_explicitly() {
    let r = request(16, false);
    let bad_d = Dividend::new(0.7, 0.6, 0.0, 0.9).unwrap();
    let bad_v = RoughBergomi::new(0.1, 0.0, 0.9).unwrap();
    assert!(Plan::compile_rough_bergomi(&r, bad_d, bad_v, -0.9, 0.5, policy(1)).is_err());
    for bad in [f64::NAN, f64::INFINITY, 1.01] {
        assert!(
            Plan::compile_rough_bergomi(&r, dividend(), factor(0.1, 0.6), bad, 0.5, policy(1))
                .is_err()
        );
    }
    let grid = LocalVolTimeGrid::compile(vec![1.0], 1.0).unwrap();
    let path = Path::compile_rough_bergomi(
        r.market().equity().forward(),
        dividend(),
        0.2,
        factor(0.1, 0.6),
        0.15,
        &grid,
    )
    .unwrap();
    assert!(path.evolve_path(&[0.0; 3]).is_err());
    assert!(path.evolve_path(&[f64::NAN; 4]).is_err());
    let huge = LocalVolTimeGrid::compile(vec![1.0], 1.0 / 4097.0).unwrap();
    assert!(
        Path::compile_rough_bergomi(
            r.market().equity().forward(),
            dividend(),
            0.2,
            factor(0.1, 0.6),
            0.15,
            &huge
        )
        .is_err()
    );
    let singular = Plan::compile_rough_bergomi(
        &r,
        Dividend::new(0.7, 0.6, 0.35, 1.0).unwrap(),
        RoughBergomi::new(0.1, 0.6, 1.0).unwrap(),
        1.0,
        0.5,
        policy(1),
    )
    .unwrap();
    let price = singular.evaluate().unwrap();
    assert!(singular.evaluate_aad().is_err());
    assert!(singular.evaluate_bergomi_aad().is_err());
    assert!(singular.evaluate_correlation_aad().is_err());
    assert!(
        singular
            .evaluate_gamma(GammaConfig::new(SpotBump::absolute(1.0).unwrap()))
            .is_err()
    );
    assert_eq!(price, singular.evaluate().unwrap());
}

#[test]
fn exact_worker_replay_and_all_rough_parameters_change_fingerprint() {
    let r = request(128, false);
    let make = |h, eta, sv, dv, w| {
        Plan::compile_rough_bergomi(
            &r,
            dividend(),
            RoughBergomi::new(h, eta, sv).unwrap(),
            dv,
            0.125,
            policy(w),
        )
        .unwrap()
    };
    let a = make(0.1, 0.6, -0.4, 0.15, 1);
    let b = make(0.1, 0.6, -0.4, 0.15, 3);
    let x = a.evaluate().unwrap();
    let y = b.evaluate().unwrap();
    assert_eq!(x.value, y.value);
    assert_eq!(x.standard_error, y.standard_error);
    assert_eq!(x, a.evaluate().unwrap());
    for p in [
        make(0.2, 0.6, -0.4, 0.15, 1),
        make(0.1, 0.7, -0.4, 0.15, 1),
        make(0.1, 0.6, -0.3, 0.15, 1),
        make(0.1, 0.6, -0.4, 0.2, 1),
    ] {
        assert_ne!(a.plan_fingerprint(), p.plan_fingerprint());
    }
    assert_ne!(
        make(0.1, 0.0, -0.4, 0.15, 1).plan_fingerprint(),
        make(0.1, 0.0, -0.4, 0.2, 1).plan_fingerprint()
    );
}
