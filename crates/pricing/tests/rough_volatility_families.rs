use pricing::mc::{ExecutionPolicy, RandomDomain};
use pricing::models::RoughBergomi;
use pricing::rough_volatility::*;
use pricing::stochastic_volatility::StochasticVolatilityPricingPlan;
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
    );
}
fn curve() -> ForwardVarianceCurve {
    ForwardVarianceCurve::constant(0.04).unwrap()
}
fn models(stochastic: bool) -> Vec<RoughVolatilityModel> {
    let eta = if stochastic { 0.3 } else { 0.0 };
    let heston = RoughHeston::new(0.1, 0.04, 1.2, 0.04, eta, -0.6).unwrap();
    vec![
        heston.clone().into(),
        LiftedHeston::from_rough(&heston, 12, 2.5).unwrap().into(),
        QuadraticRoughHeston::new(
            0.1,
            0.1,
            1.2,
            eta,
            if stochastic { 0.3 } else { 0.0 },
            0.1,
            0.04,
        )
        .unwrap()
        .into(),
        MixedRoughBergomi::new(0.1, -0.6, vec![0.3, 0.7], vec![eta, 2.0 * eta], curve())
            .unwrap()
            .into(),
        RoughSabr::new(0.1, eta, -0.6, 1.0, curve()).unwrap().into(),
        Rfsv::new(0.1, 1.2, eta / 5.0, 0.2_f64.ln(), Some(0.2_f64.ln()))
            .unwrap()
            .into(),
    ]
}
fn grid() -> Vec<f64> {
    vec![0.0, 0.13, 0.5, 1.0]
}
fn shocks(plan: &RoughVolatilityPathPlan) -> Vec<f64> {
    (0..plan.random_dimension())
        .map(|i| (1.7 * i as f64).sin())
        .collect()
}
fn policy(workers: u32) -> ExecutionPolicy {
    ExecutionPolicy::new(workers, Some(64)).unwrap()
}
fn payload(qmc: bool, cash: bool, antithetic: bool, bridge: bool) -> Value {
    let mut value: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v2/pricing_request.golden.json"
    ))
    .unwrap();
    value["engine"] = if qmc {
        json!({"type":"randomized_quasi_monte_carlo", "points_per_scramble":64,
            "scramble_count":4, "master_scramble_seed":612,
            "variance_reduction":{"antithetic":antithetic,"brownian_bridge":bridge}})
    } else {
        json!({"type":"pseudo_monte_carlo", "independent_sampling_units":64,
            "master_seed":612, "variance_reduction":{"antithetic":antithetic,"brownian_bridge":bridge}})
    };
    if cash {
        value["market"]["discrete_dividends"] = json!([
            {"event_id":1,"ex_time":0.35,"quote":{"type":"fixed_cash_and_proportional","fixed_cash":6.0,"beta":0.05}},
            {"event_id":2,"ex_time":1.4,"quote":{"type":"fixed_cash","amount":3.0}}
        ]);
    }
    value
}
fn request(value: Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(&value).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn references() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/reference.json"
    ))
    .unwrap()
}

#[test]
fn rejects_invalid_parameters_grids_and_random_inputs() {
    assert!(RoughHeston::new(0.0, 0.04, 1.0, 0.04, 0.2, 0.0).is_err());
    assert!(RoughHeston::new(0.1, -0.04, 1.0, 0.04, 0.2, 0.0).is_err());
    assert!(LiftedHeston::new(0.04, 1.0, 0.04, 0.2, 0.0, vec![1.0], vec![]).is_err());
    assert!(QuadraticRoughHeston::new(0.1, 0.0, 1.0, 0.2, -1.0, 0.0, 0.04).is_err());
    assert!(MixedRoughBergomi::new(0.1, 0.0, vec![0.2], vec![1.0], curve()).is_err());
    assert!(MixedRoughBergomi::new(0.1, 0.0, vec![f64::NAN], vec![1.0], curve()).is_err());
    assert!(RoughSabr::new(0.1, 0.2, -0.6, 1.1, curve()).is_err());
    assert!(Rfsv::new(0.1, 0.0, 0.2, -1.0, None).is_err());
    assert!(Rfsv::new(0.1, 1.0, 0.0, -1.0, Some(-2.0)).is_err());
    for model in models(true) {
        assert!(RoughVolatilityPathPlan::compile(model.clone(), vec![0.0, 0.0, 1.0]).is_err());
        assert!(RoughVolatilityPathPlan::compile(model.clone(), vec![0.0, f64::INFINITY]).is_err());
        let limit = RoughVolatilityPathPlan::maximum_time_steps(&model);
        assert!(
            RoughVolatilityPathPlan::compile(
                model.clone(),
                (0..limit + 2).map(|i| i as f64).collect()
            )
            .is_err()
        );
        let plan = RoughVolatilityPathPlan::compile(model, grid()).unwrap();
        assert!(plan.evolve_path(100.0, &[]).is_err());
        assert!(
            plan.evolve_path(100.0, &vec![f64::NAN; plan.random_dimension() as usize])
                .is_err()
        );
        assert!(plan.evolve_path(-1.0, &shocks(&plan)).is_err());
    }
}

#[test]
fn forward_variance_curve_contract_and_overflow() {
    let c = ForwardVarianceCurve::piecewise_linear(vec![0.0, 0.5, 1.0], vec![0.04, 0.09, 0.16])
        .unwrap();
    close(c.value(0.25).unwrap(), 0.065, 1e-15);
    assert_eq!(c.value(3.0).unwrap(), 0.16);
    assert!(c.value(-0.1).is_err());
    assert!(ForwardVarianceCurve::piecewise_linear(vec![0.1], vec![0.04]).is_err());
    let e = ForwardVarianceCurve::exponential(0.04, 0.25).unwrap();
    close(e.value(2.0).unwrap(), 0.04 * 0.5_f64.exp(), 2e-16);
    assert!(e.value(1e100).is_err());
    assert!(
        ForwardVarianceCurve::exponential(0.04, -1.0)
            .unwrap()
            .value(1e100)
            .is_err()
    );
    assert_eq!(
        ForwardVarianceCurve::exponential(0.0, f64::MAX)
            .unwrap()
            .value(2.0)
            .unwrap(),
        0.0
    );
}

#[test]
fn six_families_have_the_same_constant_variance_path_limit() {
    for model in models(false) {
        let plan = RoughVolatilityPathPlan::compile(model, grid()).unwrap();
        let z = shocks(&plan);
        let path = plan.evolve_path(100.0, &z).unwrap();
        let mut expected = 100.0;
        for (j, pair) in grid().windows(2).enumerate() {
            let dt = pair[1] - pair[0];
            expected *= (-0.5 * 0.04 * dt + 0.2 * dt.sqrt() * z[j]).exp();
            close(path.forwards[j + 1], expected, 2e-12);
        }
        for v in path.variances {
            close(v, 0.04, 1e-15);
        }
        assert_eq!(path.negative_variance_nodes, 0);
        assert_eq!(path.absorbed_forward_steps, 0);
    }
}

#[test]
fn heston_and_lift_brownian_boundaries_match_independent_full_truncation() {
    let model = RoughHeston::new(0.5, 0.04, 1.2, 0.06, 0.7, -0.5).unwrap();
    let rough = RoughVolatilityPathPlan::compile(model.clone().into(), grid()).unwrap();
    let lift = RoughVolatilityPathPlan::compile(
        LiftedHeston::from_rough(&model, 12, 2.5).unwrap().into(),
        grid(),
    )
    .unwrap();
    let normals = [0.2, -0.4, 0.7, -2.0, 0.1, -0.3, 0.4, -0.5, 0.6];
    let paths = [
        rough.evolve_path(100.0, &normals).unwrap(),
        lift.evolve_path(100.0, &normals[..6]).unwrap(),
    ];
    let mut raw = 0.04_f64;
    let mut f = 100.0;
    let mut negatives = 0;
    for (j, pair) in grid().windows(2).enumerate() {
        let dt = pair[1] - pair[0];
        let v = raw.max(0.0);
        f *= (-0.5 * v * dt + v.sqrt() * dt.sqrt() * normals[j]).exp();
        let dw = dt.sqrt() * (-0.5 * normals[j] + 0.75_f64.sqrt() * normals[3 + j]);
        raw += 1.2 * (0.06 - v) * dt + 0.7 * v.sqrt() * dw;
        negatives += u64::from(raw < 0.0);
        for path in &paths {
            close(path.latent_states[j + 1], raw, 2e-14);
            close(path.variances[j + 1], raw.max(0.0), 2e-14);
            close(path.forwards[j + 1], f, 3e-12);
        }
    }
    assert!(negatives > 0);
    for path in paths {
        assert_eq!(path.negative_variance_nodes, negatives);
    }
}

#[test]
fn rough_heston_near_cell_matches_independent_reference_values() {
    for row in references()["heston_one_step"].as_array().unwrap() {
        let v = |key: &str| row[key].as_f64().unwrap();
        let model = RoughHeston::new(
            v("hurst"),
            v("initial_variance"),
            v("mean_reversion"),
            v("long_run_variance"),
            v("vol_of_vol"),
            v("correlation"),
        )
        .unwrap();
        let plan = RoughVolatilityPathPlan::compile(model.into(), vec![0.0, v("dt")]).unwrap();
        let z: Vec<f64> = row["normals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|z| z.as_f64().unwrap())
            .collect();
        let path = plan.evolve_path(100.0, &z).unwrap();
        close(path.latent_states[1], v("raw_variance"), 3e-13);
        close(path.forwards[1], v("forward"), 2e-12);
    }
}

#[test]
fn quadratic_brownian_feedback_matches_scalar_recursion() {
    let model = QuadraticRoughHeston::new(0.5, 0.1, 1.2, 0.7, 0.3, 0.05, 0.01).unwrap();
    let plan = RoughVolatilityPathPlan::compile(model.into(), grid()).unwrap();
    assert_eq!(plan.random_dimension(), 6);
    let z = shocks(&plan);
    let path = plan.evolve_path(100.0, &z).unwrap();
    let mut latent = 0.1_f64;
    for (j, pair) in grid().windows(2).enumerate() {
        let dt = pair[1] - pair[0];
        let variance = 0.3 * (latent - 0.05).powi(2) + 0.01;
        latent += -1.2 * latent * dt + 1.2 * 0.7 * variance.sqrt() * dt.sqrt() * z[j];
        close(path.latent_states[j + 1], latent, 3e-14);
        close(
            path.variances[j + 1],
            0.3 * (latent - 0.05).powi(2) + 0.01,
            3e-14,
        );
    }
}

#[test]
fn mixed_single_component_equals_sabr_beta_one_and_centering_uses_actual_variance() {
    let m = MixedRoughBergomi::new(0.1, -0.6, vec![1.0], vec![0.7], curve()).unwrap();
    let a = RoughVolatilityPathPlan::compile(m.into(), grid()).unwrap();
    let b = RoughVolatilityPathPlan::compile(
        RoughSabr::new(0.1, 0.7, -0.6, 1.0, curve()).unwrap().into(),
        grid(),
    )
    .unwrap();
    let z = shocks(&a);
    assert_eq!(
        a.evolve_path(100.0, &z).unwrap(),
        b.evolve_path(100.0, &z).unwrap()
    );
    let dimension = a.random_dimension() as usize;
    let zero = a.evolve_path(100.0, &vec![0.0; dimension]).unwrap();
    let mut variance = [0.0; 4];
    for d in 0..dimension {
        let mut basis = vec![0.0; dimension];
        basis[d] = 1.0;
        let path = a.evolve_path(100.0, &basis).unwrap();
        for (v, x) in variance.iter_mut().zip(path.latent_states) {
            *v += x * x;
        }
    }
    for (i, &v) in variance.iter().enumerate() {
        close(
            (zero.variances[i] / 0.04).ln(),
            -0.5 * 0.7_f64.powi(2) * v,
            3e-14,
        );
    }
}

#[test]
fn classical_sabr_limit_normal_forward_and_absorbing_cev() {
    let nu = 0.3_f64;
    let c = ForwardVarianceCurve::exponential(0.04, nu * nu).unwrap();
    let plan = RoughVolatilityPathPlan::compile(
        RoughSabr::new(0.5, 2.0 * nu, -0.6, 1.0, c).unwrap().into(),
        grid(),
    )
    .unwrap();
    let z = shocks(&plan);
    let path = plan.evolve_path(100.0, &z).unwrap();
    let mut w = 0.0;
    for (j, pair) in grid().windows(2).enumerate() {
        w += (pair[1] - pair[0]).sqrt() * (-0.6 * z[j] + 0.8 * z[3 + j]);
        let alpha = 0.2 * (nu * w - 0.5 * nu * nu * pair[1]).exp();
        close(path.variances[j + 1].sqrt(), alpha, 3e-15);
    }
    let normal = RoughVolatilityPathPlan::compile(
        RoughSabr::new(
            0.1,
            0.0,
            0.0,
            0.0,
            ForwardVarianceCurve::constant(4.0).unwrap(),
        )
        .unwrap()
        .into(),
        vec![0.0, 1.0],
    )
    .unwrap();
    close(
        normal
            .evolve_path(-1.0, &[-2.0, 0.0, 0.0])
            .unwrap()
            .forwards[1],
        -5.0,
        1e-14,
    );
    let cev = RoughVolatilityPathPlan::compile(
        RoughSabr::new(
            0.1,
            0.0,
            0.0,
            0.5,
            ForwardVarianceCurve::constant(4.0).unwrap(),
        )
        .unwrap()
        .into(),
        vec![0.0, 1.0, 2.0],
    )
    .unwrap();
    let path = cev
        .evolve_path(1.0, &[-2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        .unwrap();
    assert_eq!(path.forwards, vec![1.0, 0.0, 0.0]);
    assert_eq!(path.absorbed_forward_steps, 1);
}

#[test]
fn rfsv_brownian_boundary_matches_exact_stationary_ou_with_conditional_initial_state() {
    for fixed in [None, Some(-1.3)] {
        let plan = RoughVolatilityPathPlan::compile(
            Rfsv::new(0.5, 0.7, 0.3, -1.5, fixed).unwrap().into(),
            grid(),
        )
        .unwrap();
        let z = shocks(&plan);
        let path = plan.evolve_path(100.0, &z).unwrap();
        let sd = (0.3_f64.powi(2) / (2.0 * 0.7)).sqrt();
        let mut x = fixed.unwrap_or(-1.5 + sd * z[3]);
        close(path.latent_states[0], x, 1e-14);
        for (j, pair) in grid().windows(2).enumerate() {
            let decay = (-0.7 * (pair[1] - pair[0])).exp();
            x = -1.5 + decay * (x + 1.5) + sd * (1.0 - decay * decay).sqrt() * z[4 + j];
            close(path.latent_states[j + 1], x, 2e-13);
        }
    }
}

#[test]
fn rfsv_covariance_matches_spectral_reference_including_negative_tail() {
    for row in references()["rfsv_correlations"].as_array().unwrap() {
        let h = row["hurst"].as_f64().unwrap();
        let lag = row["lag"].as_f64().unwrap();
        let expected = row["correlation"].as_f64().unwrap();
        let plan = RoughVolatilityPathPlan::compile(
            Rfsv::new(h, lag / 0.1, 0.1, -5.0, None).unwrap().into(),
            vec![0.0, 0.1],
        )
        .unwrap();
        let path = plan.evolve_path(100.0, &[0.0, 1.0, 0.0]).unwrap();
        let actual = (path.latent_states[1] + 5.0) / (path.latent_states[0] + 5.0);
        close(actual, expected, 2e-10);
    }
}

#[test]
fn every_asset_step_is_predictable_and_seed_replay_is_exact() {
    for model in models(true) {
        let plan = RoughVolatilityPathPlan::compile(model, grid()).unwrap();
        let mut z = plan.pseudo_shocks(91, 7, RandomDomain::Valuation);
        assert_eq!(z, plan.pseudo_shocks(91, 7, RandomDomain::Valuation));
        let original = plan.evolve_path(100.0, &z).unwrap();
        *z.last_mut().unwrap() += 0.7;
        let changed = plan.evolve_path(100.0, &z).unwrap();
        assert_eq!(
            original.forwards,
            changed.forwards,
            "{} looks ahead",
            plan.model().name()
        );
        assert_ne!(original.latent_states.last(), changed.latent_states.last());
    }
}

#[test]
fn six_pricing_adapters_preserve_sampling_unit_counts_and_thread_replay() {
    for model in models(true) {
        for qmc in [false, true] {
            for antithetic in [false, true] {
                for bridge in [false, true] {
                    let request = request(payload(qmc, true, antithetic, bridge));
                    let p = RoughVolatilityPricingPlan::compile(
                        &request,
                        model.clone(),
                        0.25,
                        policy(1),
                    )
                    .unwrap();
                    let replay = RoughVolatilityPricingPlan::compile(
                        &request,
                        model.clone(),
                        0.25,
                        policy(3),
                    )
                    .unwrap();
                    let a = p.evaluate().unwrap();
                    let b = replay.evaluate().unwrap();
                    // Execution-policy metadata belongs in the plan identity;
                    // only numerical estimates and sampling counts replay across workers.
                    assert_eq!(a.value.to_bits(), b.value.to_bits());
                    assert_eq!(a.standard_error.to_bits(), b.standard_error.to_bits());
                    assert_eq!(a.independent_sampling_units, b.independent_sampling_units);
                    assert_eq!(a.evaluated_paths, b.evaluated_paths);
                    assert_eq!(a.scheme, b.scheme);
                    assert_eq!(a.calibration_method, b.calibration_method);
                    assert_eq!(a.calibration_seed, b.calibration_seed);
                    assert_eq!(a.cash_dividend_model, b.cash_dividend_model);
                    assert_eq!(a.plan_fingerprint, p.plan_fingerprint());
                    assert_eq!(b.plan_fingerprint, replay.plan_fingerprint());
                    assert_ne!(a.plan_fingerprint, b.plan_fingerprint);
                    assert_eq!(a, p.evaluate().unwrap());
                    assert!(a.value > 0.0 && a.standard_error > 0.0);
                    assert_eq!(a.independent_sampling_units, if qmc { 4 } else { 64 });
                    assert_eq!(
                        a.evaluated_paths,
                        (if qmc { 256 } else { 64 }) * (if antithetic { 2 } else { 1 })
                    );
                    assert!(p.risky_spot() < 100.0);
                    assert!(p.time_nodes().contains(&0.35));
                    assert_eq!(p.time_nodes().last().copied(), Some(1.0));
                }
            }
        }
    }
}

#[test]
fn constant_limits_with_affine_dividends_match_existing_pricer() {
    for qmc in [false, true] {
        for bridge in [false, true] {
            let r = request(payload(qmc, true, true, bridge));
            let reference = StochasticVolatilityPricingPlan::compile_rough_bergomi(
                &r,
                RoughBergomi::new(0.1, 0.0, -0.6).unwrap(),
                0.25,
                policy(1),
            )
            .unwrap()
            .evaluate()
            .unwrap();
            for model in models(false) {
                let actual = RoughVolatilityPricingPlan::compile(&r, model, 0.25, policy(1))
                    .unwrap()
                    .evaluate()
                    .unwrap();
                close(actual.value, reference.value, 3e-11);
                close(actual.standard_error, reference.standard_error, 3e-11);
            }
        }
    }
}

#[test]
fn constant_limits_match_independent_black_price() {
    let mut v = payload(true, false, true, true);
    v["engine"]["points_per_scramble"] = json!(2048);
    v["engine"]["scramble_count"] = json!(8);
    let r = request(v);
    let reference = references()["black_call"].as_f64().unwrap();
    for model in models(false) {
        let price = RoughVolatilityPricingPlan::compile(&r, model, 0.25, policy(1))
            .unwrap()
            .evaluate()
            .unwrap();
        close(price.value, reference, 6.0 * price.standard_error + 1e-6);
        assert!(price.standard_error < 0.02);
    }
}

#[test]
fn explicit_price_only_contract_and_fingerprint_separation() {
    let r = request(payload(false, false, true, false));
    let mut fingerprints = std::collections::HashSet::new();
    for model in models(true) {
        let a = RoughVolatilityPricingPlan::compile(&r, model.clone(), 0.25, policy(1)).unwrap();
        let b = RoughVolatilityPricingPlan::compile(&r, model, 0.125, policy(1)).unwrap();
        assert_ne!(a.plan_fingerprint(), b.plan_fingerprint());
        assert!(fingerprints.insert(a.plan_fingerprint()));
    }
    let mut v = payload(false, false, true, false);
    v["risk"]["delta"] = json!(true);
    let r = request(v);
    for model in models(true) {
        assert!(RoughVolatilityPricingPlan::compile(&r, model, 0.25, policy(1)).is_err());
    }
}

#[test]
fn positive_variance_arithmetic_does_not_silently_become_zero() {
    for model in [
        RoughSabr::new(0.1, f64::MAX, 0.0, 1.0, curve())
            .unwrap()
            .into(),
        MixedRoughBergomi::new(0.1, 0.0, vec![1.0], vec![f64::MAX], curve())
            .unwrap()
            .into(),
        Rfsv::new(0.1, 1.0, 0.0, -f64::MAX, None).unwrap().into(),
    ] {
        let plan = RoughVolatilityPathPlan::compile(model, vec![0.0, 1.0]).unwrap();
        assert!(
            plan.evolve_path(100.0, &vec![0.0; plan.random_dimension() as usize])
                .is_err()
        );
    }
}
