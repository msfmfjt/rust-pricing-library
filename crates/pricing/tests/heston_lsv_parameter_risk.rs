//! Full particle re-calibration derivatives at a fixed relative LV target.
use pricing::market::LocalVarianceGrid;
use pricing::mc::ExecutionPolicy;
use pricing::mc::lsv::LsvParticleConfig;
use pricing::rough_volatility::*;
use pricing::{JsonLimits, PricingRequest, parse_request_json};
use serde_json::{Value, json};

fn model(lift: bool, h: f64, p: [f64; 5]) -> RoughVolatilityModel {
    if lift {
        LiftedHeston::new(
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            vec![0.2, 0.8, 1.3],
            vec![0.0, 0.7, 12.0],
        )
        .unwrap()
        .into()
    } else {
        RoughHeston::new(h, p[0], p[1], p[2], p[3], p[4])
            .unwrap()
            .into()
    }
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
fn close(a: f64, b: f64, t: f64) {
    assert!(
        (a - b).abs() <= t * (1.0 + b.abs()),
        "{a:.15e} vs {b:.15e} diff {:.6e}",
        (a - b).abs()
    );
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn payload(qmc: bool) -> Value {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/v1/pricing_request.golden.json"
    ))
    .unwrap();
    v["market"]["spot"] = json!(100.0);
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.25,"quote":{"type":"fixed_cash","amount":3.0}},
        {"event_id":2,"ex_time":1.5,"quote":{"type":"fixed_cash","amount":4.0}}]);
    let g = grid();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{
        "time_nodes":g.time_nodes(),"log_forward_moneyness_nodes":g.log_moneyness_nodes(),
        "shape":[5,5],"values":g.values(),"floor":1e-8,"cap":4.0}});
    v["engine"] = if qmc {
        json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":64,"scramble_count":4,
        "master_scramble_seed":819,"variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    } else {
        json!({"type":"pseudo_monte_carlo","independent_sampling_units":128,"master_seed":819,
        "variance_reduction":{"antithetic":true,"brownian_bridge":true}})
    };
    v
}
fn request(qmc: bool) -> PricingRequest {
    parse_request_json(
        &serde_json::to_vec(&payload(qmc)).unwrap(),
        JsonLimits::DEFAULT,
    )
    .unwrap()
}
fn plan(
    req: &PricingRequest,
    lift: bool,
    h: f64,
    p: [f64; 5],
    threads: u32,
    trace: bool,
) -> RoughFamilyLsvPricingPlan {
    RoughFamilyLsvPricingPlan::compile(
        req,
        model(lift, h, p),
        particles(trace),
        ExecutionPolicy::new(threads, None).unwrap(),
    )
    .unwrap()
}

#[test]
fn calibration_reverse_covers_all_parameters_and_donor_nodes() {
    let g = grid();
    let p = [0.04, 0.7, 0.055, 0.15, -0.6];
    let seeds = (0..25).map(|i| (i as f64 * 0.4).cos()).collect::<Vec<_>>();
    let mut count = 0;
    let mut maximum = 0.0_f64;
    for (lift, h) in [(false, 0.1), (false, 0.3), (false, 0.5), (true, 0.2)] {
        let build = |pp, hh| {
            calibrate_rough_family_lsv(&g, model(lift, hh, pp), 100.0, particles(true)).unwrap()
        };
        let c = build(p, h);
        let a = c.reverse_heston_parameters(&seeds, !lift).unwrap();
        assert!(c.conditional_moments().iter().any(|m| m.extrapolated));
        let topology = |c: &CalibratedRoughFamilyLsv| {
            c.conditional_moments()
                .iter()
                .map(|m| (m.source_node, m.extrapolated))
                .collect::<Vec<_>>()
        };
        for (j, &adj) in a.iter().enumerate() {
            for eps in [2e-7, 1e-7] {
                let scenario = |shift: f64| {
                    let mut pp = p;
                    let mut hh = h;
                    if j == 5 {
                        hh += shift;
                    } else {
                        pp[j] += shift;
                    }
                    build(pp, hh)
                };
                let dn = scenario(-eps);
                let up = if j == 5 && h == 0.5 {
                    scenario(-2.0 * eps)
                } else {
                    scenario(eps)
                };
                assert_eq!(topology(&c), topology(&dn));
                assert_eq!(topology(&c), topology(&up));
                let f = |c: &CalibratedRoughFamilyLsv| dot(c.surface().squared_leverage(), &seeds);
                let fd = if j == 5 && h == 0.5 {
                    (3.0 * f(&c) - 4.0 * f(&dn) + f(&up)) / (2.0 * eps)
                } else {
                    (f(&up) - f(&dn)) / (2.0 * eps)
                };
                close(adj, fd, 3e-5);
                count += 1;
                maximum = maximum.max((adj - fd).abs());
            }
        }
    }
    println!("calibration_vjp comparisons={count} max_gap={maximum:.15e}");
}
#[test]
fn total_risk_matches_full_model_recompilation_and_recalibration() {
    let p = [0.04, 0.7, 0.055, 0.15, -0.6];
    let mut count = 0;
    let mut maximum = 0.0_f64;
    for (lift, h) in [(false, 0.1), (false, 0.3), (false, 0.5), (true, 0.2)] {
        for qmc in [false, true] {
            let req = request(qmc);
            let base = plan(&req, lift, h, p, 1, true);
            let before = base.evaluate_local_variance_risk().unwrap();
            let a = base.evaluate_heston_parameter_risk(!lift).unwrap();
            assert_eq!(a.price, base.evaluate().unwrap());
            assert_eq!(before, base.evaluate_local_variance_risk().unwrap());
            assert_eq!(a.standard_errors.is_some(), qmc);
            for (j, &adj) in a.parameter_adjoints.iter().enumerate() {
                close(adj, a.direct_adjoints[j] + a.calibration_adjoints[j], 1e-12);
                for eps in [1e-6, 5e-7] {
                    let price = |shift: f64| {
                        let mut pp = p;
                        let mut hh = h;
                        if j == 5 {
                            hh += shift;
                        } else {
                            pp[j] += shift;
                        }
                        plan(&req, lift, hh, pp, 1, false).evaluate().unwrap().value
                    };
                    let fd = if j == 5 && h == 0.5 {
                        (3.0 * a.price.value - 4.0 * price(-eps) + price(-2.0 * eps)) / (2.0 * eps)
                    } else {
                        (price(eps) - price(-eps)) / (2.0 * eps)
                    };
                    close(adj, fd, 3e-5);
                    count += 1;
                    maximum = maximum.max((adj - fd).abs());
                }
            }
            let b = plan(&req, lift, h, p, 3, true)
                .evaluate_heston_parameter_risk(!lift)
                .unwrap();
            assert_eq!(a.parameter_adjoints, b.parameter_adjoints);
            assert_eq!(a.standard_errors, b.standard_errors);
            assert_ne!(a.price.plan_fingerprint, b.price.plan_fingerprint);
            if !lift {
                let scalars = base.evaluate_heston_parameter_risk(false).unwrap();
                assert_eq!(&a.parameter_adjoints[..5], &*scalars.parameter_adjoints);
                assert_ne!(a.risk_fingerprint, scalars.risk_fingerprint);
            }
        }
    }
    println!("full_recalibration comparisons={count} max_gap={maximum:.15e}");
}
#[test]
fn deterministic_variance_risk_cancels_through_recalibration() {
    // At nu=0 the same time-dependent V_r affects every calibration and
    // valuation path: L_r^2=a_r/V_r. All deterministic variance-curve changes
    // cancel EXACTLY in the finite scheme when both grids coincide.
    let p = [0.04, 0.7, 0.055, 0.0, -0.6];
    for (lift, h) in [(false, 0.1), (false, 0.3), (false, 0.5), (true, 0.2)] {
        let req = request(true);
        let a = plan(&req, lift, h, p, 1, true)
            .evaluate_heston_parameter_risk(!lift)
            .unwrap();
        assert!(a.direct_adjoints[0].abs() > 1.0);
        for j in (0..a.parameter_adjoints.len()).filter(|&j| j != 3) {
            close(a.parameter_adjoints[j], 0.0, 2e-10);
            assert!(a.standard_errors.as_ref().unwrap()[j] < 2e-10);
        }
    }
}
#[test]
fn unsupported_domains_and_missing_traces_fail_explicitly() {
    let req = request(false);
    let p = [0.04, 0.7, 0.055, 0.15, -0.6];
    assert!(
        plan(&req, false, 0.2, p, 1, false)
            .evaluate_heston_parameter_risk(false)
            .is_err()
    );
    assert!(
        plan(&req, true, 0.2, p, 1, true)
            .evaluate_heston_parameter_risk(true)
            .is_err()
    );
    let c =
        calibrate_rough_family_lsv(&grid(), model(false, 0.2, p), 100.0, particles(true)).unwrap();
    assert!(c.reverse_heston_parameters(&[], false).is_err());
    assert!(c.reverse_heston_parameters(&[f64::NAN; 25], false).is_err());
    let bad = MixedRoughBergomi::new(
        0.2,
        -0.6,
        vec![1.0],
        vec![0.3],
        ForwardVarianceCurve::constant(0.04).unwrap(),
    )
    .unwrap();
    let c = calibrate_rough_family_lsv(&grid(), bad.into(), 100.0, particles(true)).unwrap();
    assert!(c.reverse_heston_parameters(&[0.0; 25], false).is_err());
}

#[test]
#[ignore = "release statistical Black-price and fixed-target zero-risk panel"]
fn black_price_and_zero_deterministic_parameter_risk() {
    // Fixed before running: variance .04, S=K=100, one year, unit curves,
    // seeds91/1973, 8x2048 RQMC antithetic/bridge, 128 calibration particles.
    // Price: |error| <= 5 SE + 2e-4. Deterministic-parameter cancellation:
    // |total| <= 1e-8; it is an exact finite-scheme identity, not a CI.
    let mut v = payload(true);
    v["market"]["discrete_dividends"] = json!([]);
    v["market"]["discount_curve"]["discount_factors"] = json!([1.0, 1.0]);
    v["market"]["dividend_curve"]["discount_factors"] = json!([1.0, 1.0]);
    v["model"]["local_variance_grid"] = json!({"time_nodes":[0.,0.25,0.5,0.75,1.],
        "log_forward_moneyness_nodes":[-0.4,0.4],"shape":[5,2],"values":vec![0.04;10],"floor":1e-8,"cap":4.0});
    v["engine"]["points_per_scramble"] = json!(2048);
    v["engine"]["scramble_count"] = json!(8);
    for seed in [91, 1973] {
        v["engine"]["master_scramble_seed"] = json!(seed);
        let req =
            parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
        for (lift, h) in [(false, 0.1), (false, 0.3), (false, 0.5), (true, 0.2)] {
            let r = plan(&req, lift, h, [0.04, 0.7, 0.055, 0.0, -0.6], 1, true)
                .evaluate_heston_parameter_risk(!lift)
                .unwrap();
            let err = (r.price.value - 7.965_567_455_405_804).abs();
            assert!(
                err <= 5.0 * r.price.standard_error + 2e-4,
                "price {} SE {}",
                r.price.value,
                r.price.standard_error
            );
            let max = r
                .parameter_adjoints
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != 3)
                .map(|(_, v)| v.abs())
                .fold(0., f64::max);
            assert!(max <= 1e-8);
            println!(
                "black_fixed_target lift={lift} h={h} seed={seed} price={:.15e} se={:.15e} direct_v0={:.15e} calibration_v0={:.15e} max_cancellation={max:.15e}",
                r.price.value,
                r.price.standard_error,
                r.direct_adjoints[0],
                r.calibration_adjoints[0]
            );
        }
    }
}
