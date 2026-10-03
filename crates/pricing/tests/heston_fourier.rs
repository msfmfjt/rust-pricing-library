use pricing::product::OptionSide::{Call, Put};
use pricing::rough_volatility::{
    FourierConfig, FourierError, HestonFourierPlan, LiftedHeston, Rfsv, RoughHeston,
    RoughVolatilityModel,
};
use pricing_numerics::Complex64 as C;
use serde_json::Value;

fn rough(h: f64) -> RoughHeston {
    RoughHeston::new(h, 0.04, 0.7, 0.055, 0.18, -0.65).unwrap()
}
fn lifted() -> LiftedHeston {
    LiftedHeston::new(
        0.04,
        0.7,
        0.055,
        0.18,
        -0.65,
        vec![0.6, 1.1, 0.8],
        vec![0.2, 1.5, 12.0],
    )
    .unwrap()
}
fn config(n: usize) -> FourierConfig {
    FourierConfig {
        time_steps: n,
        integration_intervals: 256,
        cutoff: 128.0,
    }
}
fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/fourier.json"
    ))
    .unwrap()
}
fn model(name: &str) -> RoughVolatilityModel {
    match name {
        "classical" => rough(0.5).into(),
        "lifted" => lifted().into(),
        "rough_0.1" => rough(0.1).into(),
        "rough_0.3" => rough(0.3).into(),
        _ => panic!("unknown reference family"),
    }
}

#[test]
fn exact_constant_variance_absorbing_and_expiry_limits() {
    for h in [0.01, 0.1, 0.3, 0.5] {
        let m = RoughHeston::new(h, 0.04, 0.7, 0.04, 0.0, -0.65).unwrap();
        let plan = HestonFourierPlan::compile(m.into(), 1.0, config(16)).unwrap();
        assert!((plan.price(100.0, 100.0, 0.97, Call).unwrap() - 7.72660043174363).abs() < 3e-13);
        for (d, u) in [(0.0, 2.0), (0.5, 3.0), (1.0, -0.25)] {
            let z = C::new(d, u);
            assert_eq!(plan.transform(d, u).unwrap(), ((z * z - z) * 0.02).exp());
        }
        let zero = RoughHeston::new(h, 0.0, 0.7, 0.0, 2.0, 1.0).unwrap();
        let zero = HestonFourierPlan::compile(zero.into(), 1.0, config(16)).unwrap();
        assert_eq!(zero.price(100.0, 90.0, 0.97, Call).unwrap(), 9.7);
        assert_eq!(zero.price(100.0, 90.0, 0.97, Put).unwrap(), 0.0);
        let expiry = HestonFourierPlan::compile(rough(h).into(), 0.0, config(16)).unwrap();
        assert_eq!(expiry.price(100.0, 105.0, 0.97, Put).unwrap(), 4.85);
    }
}

#[test]
fn independent_continuous_transform_references_and_time_refinement() {
    let fixture = fixture();
    let tolerance = fixture["protocol"]["transform_tolerance"].as_f64().unwrap();
    for family in ["classical", "lifted", "rough_0.1", "rough_0.3"] {
        let time = if family.starts_with("rough_") {
            0.1
        } else {
            1.0
        };
        let fine = HestonFourierPlan::compile(
            model(family),
            time,
            FourierConfig {
                time_steps: 1024,
                integration_intervals: 4,
                cutoff: 1.0,
            },
        )
        .unwrap();
        let coarse = HestonFourierPlan::compile(
            model(family),
            time,
            FourierConfig {
                time_steps: 128,
                integration_intervals: 4,
                cutoff: 1.0,
            },
        )
        .unwrap();
        for row in fixture["transforms"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["family"] == family)
        {
            let d = row["damping"].as_f64().unwrap();
            let u = row["frequency"].as_f64().unwrap();
            let reference = C::new(row["real"].as_f64().unwrap(), row["imag"].as_f64().unwrap());
            let actual = fine.transform(d, u).unwrap();
            let err = (actual - reference).norm();
            let coarse_err = (coarse.transform(d, u).unwrap() - reference).norm();
            eprintln!(
                "TRANSFORM {family} d={d} u={u} fine_error={err:.12e} coarse_error={coarse_err:.12e}"
            );
            assert!(err < tolerance, "{family}: {err}");
            if u != 0.0 {
                assert!(
                    err < coarse_err,
                    "refinement {family} {u}: {err} {coarse_err}"
                );
            }
            assert_eq!(fine.transform(d, -u).unwrap(), actual.conjugate());
        }
    }
}

#[test]
#[ignore = "continuous-time price accuracy panel; run explicitly in release"]
fn independent_continuous_european_prices_and_parity() {
    let fixture = fixture();
    let tolerance = fixture["protocol"]["price_tolerance"].as_f64().unwrap();
    let cfg = FourierConfig {
        time_steps: 1024,
        integration_intervals: 1024,
        cutoff: 160.0,
    };
    for family in ["classical", "lifted", "rough_0.1", "rough_0.3"] {
        let plan = HestonFourierPlan::compile(model(family), 1.0, cfg).unwrap();
        for row in fixture["prices"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["family"] == family)
        {
            let f = row["forward"].as_f64().unwrap();
            let k = row["strike"].as_f64().unwrap();
            let d = row["discount"].as_f64().unwrap();
            let actual = plan.price(f, k, d, Call).unwrap();
            let reference = row["call"].as_f64().unwrap();
            let error = (actual - reference).abs();
            eprintln!(
                "PRICE {family} K={k} actual={actual:.12} reference={reference:.12} error={error:.12e}"
            );
            assert!(error < tolerance, "{family}: {error}");
            let put = plan.price(f, k, d, Put).unwrap();
            assert!((actual - put - d * (f - k)).abs() < 2e-13);
            assert_eq!(plan.price(f, k, 0.0, Call).unwrap(), 0.0);
            assert_eq!(plan.price(f, 0.0, d, Call).unwrap(), f * d);
        }
    }
}

#[test]
fn zero_rate_lift_boundary_and_shared_factor_splitting() {
    let r = rough(0.5);
    let one = LiftedHeston::new(0.04, 0.7, 0.055, 0.18, -0.65, vec![1.0], vec![0.0]).unwrap();
    let split = LiftedHeston::new(
        0.04,
        0.7,
        0.055,
        0.18,
        -0.65,
        vec![0.25, 0.75],
        vec![0.0, 0.0],
    )
    .unwrap();
    let a = HestonFourierPlan::compile(r.into(), 1.0, config(256)).unwrap();
    let b = HestonFourierPlan::compile(one.into(), 1.0, config(256)).unwrap();
    let c = HestonFourierPlan::compile(split.into(), 1.0, config(256)).unwrap();
    for u in [0.0, 0.25, 1.0, 10.0, 100.0] {
        let z = a.transform(0.5, u).unwrap();
        assert_eq!(z, b.transform(0.5, u).unwrap());
        assert!((z - c.transform(0.5, u).unwrap()).norm() < 2e-13);
    }
    assert_ne!(a.plan_fingerprint(), b.plan_fingerprint());
}

#[test]
fn deterministic_mean_reverting_variance_and_stiff_lift() {
    // Independent solution of ordinary deterministic mean reversion.
    let v0 = 0.09;
    let k: f64 = 1.3;
    let theta = 0.025;
    let t: f64 = 0.7;
    let iv = theta * t + (v0 - theta) * (1.0 - (-k * t).exp()) / k;
    let m = RoughHeston::new(0.5, v0, k, theta, 0.0, 0.9).unwrap();
    let p = HestonFourierPlan::compile(m.into(), t, config(1024)).unwrap();
    let z = C::new(0.5, 3.0);
    assert!((p.transform(z.re, z.im).unwrap() - ((z * z - z) * (0.5 * iv)).exp()).norm() < 2e-8);
    // High-rate modes are integrated exponentially, not by an explicit ODE step.
    let m = LiftedHeston::new(
        0.04,
        0.7,
        0.055,
        0.18,
        -0.65,
        vec![0.6, 1.1, 0.8],
        vec![0.0, 1e8, 1e100],
    )
    .unwrap();
    let p = HestonFourierPlan::compile(m.into(), 1.0, config(128)).unwrap();
    assert!(p.price(100.0, 100.0, 1.0, Call).unwrap() > 0.0);
}

#[test]
fn refinement_diagnostics_isolate_the_three_grid_changes() {
    let cfg = FourierConfig {
        time_steps: 32,
        integration_intervals: 64,
        cutoff: 64.0,
    };
    let p = HestonFourierPlan::compile(rough(0.5).into(), 1.0, cfg).unwrap();
    let r = p.refinement(100.0, 105.0, 0.97, Call).unwrap();
    assert_eq!(r.base_price, p.price(100.0, 105.0, 0.97, Call).unwrap());
    let configs = [
        FourierConfig {
            time_steps: 64,
            ..cfg
        },
        FourierConfig {
            time_steps: 64,
            integration_intervals: 128,
            ..cfg
        },
        FourierConfig {
            time_steps: 64,
            integration_intervals: 256,
            cutoff: 128.0,
        },
    ];
    for (cfg, expected) in configs.into_iter().zip([
        r.time_refined_price,
        r.quadrature_refined_price,
        r.extended_cutoff_price,
    ]) {
        let q = HestonFourierPlan::compile(rough(0.5).into(), 1.0, cfg).unwrap();
        assert_eq!(q.price(100.0, 105.0, 0.97, Call).unwrap(), expected);
        assert_ne!(p.plan_fingerprint(), q.plan_fingerprint());
    }
    assert_eq!(r.time_change(), r.time_refined_price - r.base_price);
    assert_eq!(
        r.quadrature_change(),
        r.quadrature_refined_price - r.time_refined_price
    );
    assert_eq!(
        r.cutoff_change(),
        r.extended_cutoff_price - r.quadrature_refined_price
    );
}

#[test]
fn invalid_inputs_and_unsupported_models_do_not_fall_back_to_mc() {
    for cfg in [
        FourierConfig {
            time_steps: 0,
            ..config(32)
        },
        FourierConfig {
            integration_intervals: 7,
            ..config(32)
        },
        FourierConfig {
            cutoff: f64::NAN,
            ..config(32)
        },
    ] {
        assert!(HestonFourierPlan::compile(rough(0.1).into(), 1.0, cfg).is_err());
    }
    for t in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(HestonFourierPlan::compile(rough(0.1).into(), t, config(32)).is_err());
    }
    let unsupported = Rfsv::new(0.1, 1.0, 0.2, -1.6, None).unwrap();
    assert_eq!(
        HestonFourierPlan::compile(unsupported.into(), 1.0, config(32)).unwrap_err(),
        FourierError::UnsupportedModel
    );
    let huge = FourierConfig {
        time_steps: 8192,
        integration_intervals: 8192,
        cutoff: 128.0,
    };
    assert_eq!(
        HestonFourierPlan::compile(rough(0.1).into(), 1.0, huge).unwrap_err(),
        FourierError::WorkLimit
    );
    let p = HestonFourierPlan::compile(rough(0.5).into(), 1.0, config(32)).unwrap();
    for (d, u) in [
        (-0.1, 0.0),
        (1.1, 0.0),
        (f64::NAN, 1.0),
        (0.5, f64::INFINITY),
    ] {
        assert!(p.transform(d, u).is_err());
    }
    for (f, k, d) in [
        (0.0, 100.0, 1.0),
        (100.0, -1.0, 1.0),
        (100.0, 100.0, -1.0),
        (f64::NAN, 1.0, 1.0),
    ] {
        assert!(p.price(f, k, d, Call).is_err());
    }
}
