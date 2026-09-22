use pricing::mc::{ExecutionPolicy, LocalVolTimeGrid};
use pricing::stochastic_dividends::{
    BuehlerDividendModel as Model, BuehlerDividendState as State,
    StochasticDividendPathPlan as PathPlan, StochasticDividendPricingPlan as PricePlan,
};
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance,
        "{a} != {b}, tolerance {tolerance}"
    );
}
fn model(kappa: f64, alpha: f64, nu: f64, rho: f64) -> Model {
    Model::new(kappa, alpha, nu, rho).unwrap()
}
fn policy(workers: u32) -> ExecutionPolicy {
    ExecutionPolicy::new(workers, Some(64)).unwrap()
}
fn payload(qmc: bool, dividends: &[(f64, f64)]) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    if qmc {
        v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":2048,
            "scramble_count":8,"master_scramble_seed":612,
            "variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    }
    v["market"]["discrete_dividends"] = json!(dividends.iter().enumerate().map(|(i, &(t, d))|
        json!({"event_id":i+1,"ex_time":t,"quote":{"type":"fixed_cash","amount":d}})
    ).collect::<Vec<_>>());
    v
}
fn request(v: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}

#[test]
fn validation_and_positive_split_boundaries_are_explicit() {
    for bad in [f64::NAN, f64::INFINITY, -0.1] {
        assert!(Model::new(bad, 0.5, 0.2, 0.0).is_err());
        assert!(Model::new(1.0, 0.5, bad, 0.0).is_err());
    }
    for bad in [f64::NAN, -0.1, 1.1] {
        assert!(Model::new(1.0, bad, 0.2, 0.0).is_err());
    }
    for bad in [f64::NAN, -1.1, 1.1] {
        assert!(Model::new(1.0, 0.5, 0.2, bad).is_err());
    }
    assert!(State::new(0.0, 1.0).is_err());
    assert!(State::new(1.0, f64::INFINITY).is_err());
    let state = State::new(1.3, 0.7).unwrap();
    for rho in [-1.0, 1.0] {
        let m = model(0.0, 0.5, 0.4, rho);
        assert_eq!(m.evolve(state, 0.2, 0.0, [2.0, -3.0]).unwrap(), state);
        let next = m.evolve(state, 0.2, 0.7, [0.8, 1.2]).unwrap();
        close(
            next.equity(),
            1.3 * (-0.5_f64 * 0.04 * 0.7 + 0.2 * 0.7_f64.sqrt() * 0.8).exp(),
            1e-15,
        );
        close(
            next.dividend(),
            0.7 * (-0.5_f64 * 0.16 * 0.7 + 0.4 * 0.7_f64.sqrt() * rho * 0.8).exp(),
            1e-15,
        );
        assert!(m.evolve(state, 0.2, 0.1, [f64::NAN, 0.0]).is_err());
        assert!(m.evolve(state, 1000.0, 1.0, [0.0; 2]).is_err());
    }
    let fixed = model(2.0, 0.0, 0.0, -0.7);
    let mut state = State::initial();
    for i in 0..32 {
        state = fixed
            .evolve(state, 0.3, 0.1, [0.1 * (i as f64 - 10.0), 0.7])
            .unwrap();
        assert_eq!(state.dividend(), 1.0);
    }
    let linked = model(2.0, 1.0, 0.0, 0.0)
        .evolve(State::initial(), 0.3, 0.5, [1.0, 0.0])
        .unwrap();
    assert_ne!(linked.dividend(), 1.0); // nu=0 alone is not the fixed-cash limit.
}

#[test]
fn conditional_cash_forecasts_and_cross_moments_match_independent_quadrature() {
    // Gaussian quadrature checks the actual nonlinear production step against
    // analytically integrated split moments, at arbitrary positive states.
    const Z: [f64; 12] = [
        -5.500901704467748,
        -4.2718258479322815,
        -3.2237098287700974,
        -2.2594644510007993,
        -1.340375197151617,
        -0.444403001944139,
        0.444403001944139,
        1.340375197151617,
        2.2594644510007993,
        3.2237098287700974,
        4.2718258479322815,
        5.500901704467748,
    ];
    const W: [f64; 12] = [
        1.4999271676371695e-07,
        4.8371849225906334e-05,
        0.002203380687533198,
        0.02911668791236418,
        0.14696704804532998,
        0.32166436151283,
        0.32166436151283,
        0.14696704804532998,
        0.02911668791236418,
        0.002203380687533198,
        4.8371849225906334e-05,
        1.4999271676371695e-07,
    ];
    let m = model(1.4, 0.6, 0.45, -0.35);
    let sigma: f64 = 0.2;
    let dt: f64 = 0.3;
    let a = (-0.5 * m.mean_reversion() * dt).exp();
    let b = 1.0 - a;
    let cross = (sigma * m.dividend_volatility() * m.equity_dividend_correlation() * dt).exp();
    for (f, y) in [(1.0, 1.0), (0.7, 1.3), (2.0, 0.4)] {
        let state = State::new(f, y).unwrap();
        let target = m.equity_linkage() * f + 1.0 - m.equity_linkage();
        let h = a * y + b * target;
        let ff = f * f * (sigma * sigma * dt).exp();
        let fm = f * cross;
        let mm = (m.dividend_volatility().powi(2) * dt).exp();
        let c = b * (1.0 - m.equity_linkage());
        let d = b * m.equity_linkage();
        let expected = [
            f,
            a * h + d * f + c,
            ff,
            a * h * fm + d * ff + c * f,
            a * a * h * h * mm
                + d * d * ff
                + c * c
                + 2.0 * a * h * d * fm
                + 2.0 * a * h * c
                + 2.0 * d * c * f,
        ];
        let mut moments = [0.0; 5];
        let mut future_forecast = 0.0;
        for i in 0..12 {
            for j in 0..12 {
                let next = m.evolve(state, sigma, dt, [Z[i], Z[j]]).unwrap();
                let f = next.equity();
                let y = next.dividend();
                for (sum, value) in moments.iter_mut().zip([f, y, f * f, f * y, y * y]) {
                    *sum += W[i] * W[j] * value;
                }
                future_forecast += W[i] * W[j] * m.expected_factor(next, 0.7).unwrap();
            }
        }
        for (actual, expected) in moments.into_iter().zip(expected) {
            close(actual, expected, 3e-13);
        }
        close(moments[1], m.expected_factor(state, dt).unwrap(), 3e-13);
        close(
            future_forecast,
            m.expected_factor(state, 1.0).unwrap(),
            3e-13,
        );
    }
}

#[test]
fn reserve_uses_full_schedule_observed_state_and_post_event_spot() {
    let r = request(payload(false, &[(0.5, 6.0), (1.0, 4.0), (1.4, 3.0)]));
    let grid = LocalVolTimeGrid::compile(vec![0.5, 1.0], 0.5).unwrap();
    let path = PathPlan::compile(
        r.market().equity().forward(),
        model(0.7, 0.6, 0.4, -0.2),
        0.2,
        &grid,
    )
    .unwrap();
    let growth = |t: f64| (0.98_f64 / 0.95).powf(t);
    close(
        path.risky_spot(),
        100.0 - 6.0 / growth(0.5) - 4.0 / growth(1.0) - 3.0 / growth(1.4),
        3e-14,
    );
    assert_eq!(*path.times().last().unwrap(), 1.0); // Funding does not extend paths to 1.4.
    close(
        path.nodes()[0].spots(State::initial()).unwrap().0,
        100.0,
        3e-14,
    );
    for &(f, y) in &[(0.8, 1.5), (0.01, 100.0)] {
        let state = State::new(f, y).unwrap();
        for (index, cash) in [(1, 6.0), (2, 4.0)] {
            let (post, pre) = path.nodes()[index].spots(state).unwrap();
            close(pre.unwrap() - post, cash * y, 2e-12);
            close(
                path.nodes()[index].cash_paid(state).unwrap(),
                cash * y,
                1e-14,
            );
            assert!(post > 0.0);
        }
    }
    let fixed = PathPlan::compile(
        r.market().equity().forward(),
        model(0.7, 0.0, 0.0, 0.2),
        0.2,
        &grid,
    )
    .unwrap();
    for (node, &time) in fixed.nodes().iter().zip(fixed.times()) {
        let f = 0.85;
        let reserve = [(0.5, 6.0), (1.0, 4.0), (1.4, 3.0)]
            .iter()
            .filter(|(t, _)| *t > time)
            .map(|(t, d)| d / growth(*t))
            .sum::<f64>();
        close(
            node.spots(State::new(f, 1.0).unwrap()).unwrap().0,
            growth(time) * (fixed.risky_spot() * f + reserve),
            3e-14,
        );
    }
    let absent = LocalVolTimeGrid::compile(vec![1.0], 1.0).unwrap();
    assert!(
        PathPlan::compile(
            r.market().equity().forward(),
            model(0.7, 0.6, 0.4, -0.2),
            0.2,
            &absent
        )
        .is_err()
    );
    assert!(path.evolve_path(&[0.0; 3]).is_err());
}

#[test]
fn public_prices_match_independent_conditional_black_and_fixed_cash_limits() {
    // kappa=0: terminal physical stock is a sum of two correlated lognormals.
    // Reference integrates a conditional Black formula with adaptive quadrature,
    // not the production sampler, bridge, state update or payoff compiler.
    let r = request(payload(true, &[(1.4, 25.0)]));
    let p = PricePlan::compile_bs(&r, model(0.0, 0.6, 0.45, -0.35), 1.0, policy(1)).unwrap();
    let result = p.evaluate().unwrap();
    close(
        result.value,
        7.653276188575835,
        6.0 * result.standard_error + 0.002,
    );
    assert_eq!(result.independent_sampling_units, 8);
    assert_eq!(result.evaluated_paths, 32768);
    assert_eq!(p.random_factor_count(), 2);
    assert_eq!(p.evaluate().unwrap(), result);
    let parallel = PricePlan::compile_bs(&r, model(0.0, 0.6, 0.45, -0.35), 1.0, policy(3))
        .unwrap()
        .evaluate()
        .unwrap();
    assert_eq!(parallel.value, result.value);
    assert_eq!(parallel.standard_error, result.standard_error);
    // Fingerprints include execution policy; worker changes need not hash equally.
    let mut v = payload(true, &[(1.0, 10.0)]);
    v["model"]["volatility"] = json!(0.0);
    v["product"]["strike"] = json!(80.0);
    let deterministic =
        PricePlan::compile_bs(&request(v), model(2.0, 0.0, 0.0, 0.2), 0.125, policy(1))
            .unwrap()
            .evaluate()
            .unwrap();
    close(
        deterministic.value,
        0.95 * (100.0 * 0.98 / 0.95 - 10.0 - 80.0),
        1e-13,
    );
    close(deterministic.standard_error, 0.0, 1e-13);
}

#[test]
fn pseudo_mc_replay_and_fingerprints_preserve_policy_and_model_identity() {
    let r = request(payload(false, &[(0.5, 6.0), (1.4, 3.0)]));
    let m = model(0.7, 0.6, 0.4, -0.2);
    let p = PricePlan::compile_bs(&r, m, 0.125, policy(1)).unwrap();
    let q = PricePlan::compile_bs(&r, m, 0.125, policy(3)).unwrap();
    let a = p.evaluate().unwrap();
    let b = q.evaluate().unwrap();
    assert_eq!(a.value, b.value);
    assert_eq!(a.standard_error, b.standard_error);
    assert_eq!(a.independent_sampling_units, 1024);
    assert_eq!(a.evaluated_paths, 2048);
    let changed = PricePlan::compile_bs(&r, model(0.7, 0.6, 0.41, -0.2), 0.125, policy(1)).unwrap();
    assert_ne!(p.plan_fingerprint(), changed.plan_fingerprint());
    let refined = PricePlan::compile_bs(&r, m, 0.0625, policy(1)).unwrap();
    assert_ne!(p.plan_fingerprint(), refined.plan_fingerprint());
}

#[test]
fn unsupported_cash_mixtures_and_risk_requests_fail_before_sampling() {
    let m = model(0.7, 0.6, 0.4, -0.2);
    let mut v = payload(false, &[(0.5, 6.0)]);
    v["risk"]["delta"] = json!(true);
    assert!(PricePlan::compile_bs(&request(v), m, 0.125, policy(1)).is_err());
    let mut v = payload(false, &[(0.5, 6.0)]);
    v["market"]["discrete_dividends"][0]["quote"] =
        json!({"type":"fixed_cash_and_proportional","fixed_cash":6.0,"beta":0.05});
    assert!(PricePlan::compile_bs(&request(v), m, 0.125, policy(1)).is_err());
}

#[test]
fn shared_barrier_payoff_observes_pre_and_post_dividend_states() {
    let mut v = payload(false, &[(1.0, 10.0)]);
    v["schema_version"] = json!(2);
    v["model"]["volatility"] = json!(0.0);
    v["product"] = json!({"type":"barrier","underlying_id":1,"currency_id":2,
        "expiry":"2027-09-04","strike":80.0,"barrier":100.0,"notional":1.0,
        "side":{"type":"call"},"direction":{"type":"up"},"style":{"type":"knock_out"},
        "monitoring":{"type":"discrete"},"monitoring_dates":["2027-09-04"],
        "payment_date":"2027-09-04"});
    let m = model(1.0, 0.0, 0.0, 0.0);
    // Pre-event S is above 100 and post-event S is below 100. The jump's
    // pre-state must still knock out; terminal post-state alone would not.
    let p = PricePlan::compile_bs(&request(v.clone()), m, 0.125, policy(1)).unwrap();
    close(p.evaluate().unwrap().value, 0.0, 0.0);
    v["product"]["monitoring"] = json!({"type":"continuous"});
    assert!(PricePlan::compile_bs(&request(v), m, 0.125, policy(1)).is_err());
}
