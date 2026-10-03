//! Continuous-time transform and inversion checks; finite MC comparisons retain
//! time/truncation bias and are not mathematical error bounds.
use pricing::rough_volatility::*;
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/fourier.json"
    ))
    .unwrap()
}
fn rough(h: f64) -> RoughHeston {
    RoughHeston::new(h, 0.04, 0.7, 0.055, 0.18, -0.65).unwrap()
}
fn lift() -> LiftedHeston {
    LiftedHeston::new(
        0.04,
        0.7,
        0.055,
        0.18,
        -0.65,
        vec![0.2, 0.4, 0.5],
        vec![0.1, 1.0, 8.0],
    )
    .unwrap()
}
fn cfg(n: usize, m: usize, u: f64) -> HestonFourierConfig {
    HestonFourierConfig::new(n, m, u).unwrap()
}
fn complex(v: &Value) -> Complex64 {
    Complex64::new(v[0].as_f64().unwrap(), v[1].as_f64().unwrap())
}
fn close(a: f64, b: f64, t: f64) {
    assert!(
        (a - b).abs() <= t,
        "a={a:.15} b={b:.15} gap={:.5e} tolerance={t}",
        (a - b).abs()
    );
}

#[test]
fn input_and_model_rejections() {
    for n in [0, 1, 8193, usize::MAX] {
        assert!(HestonFourierConfig::new(n, 32, 64.0).is_err());
    }
    for m in [0, 7, 9, 8193, usize::MAX] {
        assert!(HestonFourierConfig::new(32, m, 64.0).is_err());
    }
    for u in [0.0, -1.0, f64::NAN, f64::INFINITY, 10001.0] {
        assert!(HestonFourierConfig::new(32, 32, u).is_err());
    }
    for t in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(HestonFourierPlan::compile(rough(0.1).into(), t, cfg(32, 32, 64.0)).is_err());
    }
    assert!(HestonFourierPlan::compile(rough(0.1).into(), 1.0, cfg(8192, 8192, 64.0)).is_err());
    let curve = ForwardVarianceCurve::constant(0.04).unwrap();
    for m in [
        RoughSabr::new(0.1, 0.2, -0.6, 1.0, curve.clone())
            .unwrap()
            .into(),
        MixedRoughBergomi::new(0.1, -0.6, vec![1.0], vec![0.2], curve)
            .unwrap()
            .into(),
        QuadraticRoughHeston::new(0.1, 0.1, 0.7, 0.2, 0.2, 0.1, 0.04)
            .unwrap()
            .into(),
        Rfsv::new(0.1, 0.7, 0.2, -1.5, None).unwrap().into(),
    ] {
        assert_eq!(
            HestonFourierPlan::compile(m, 1.0, cfg(32, 32, 64.0)).unwrap_err(),
            FourierError::UnsupportedModel
        );
    }
    let plan = HestonFourierPlan::compile(rough(0.1).into(), 1.0, cfg(128, 256, 64.0)).unwrap();
    for z in [
        Complex64::new(-0.1, 0.0),
        Complex64::new(1.1, 0.0),
        Complex64::new(0.5, f64::NAN),
        Complex64::new(0.5, 1e7),
    ] {
        assert!(plan.log_transform(z).is_err());
    }
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(plan.price(bad, 100.0, 0.97).is_err());
        assert!(plan.price(100.0, bad, 0.97).is_err());
        assert!(plan.price(100.0, 100.0, bad).is_err());
    }
}

#[test]
fn transform_invariants_and_black_limit() {
    for m in [
        rough(0.1).into(),
        rough(0.3).into(),
        rough(0.5).into(),
        lift().into(),
    ] {
        let p = HestonFourierPlan::compile(m, 1.0, cfg(128, 128, 32.0)).unwrap();
        assert_eq!(p.log_transform(Complex64::ZERO).unwrap(), Complex64::ZERO);
        assert_eq!(p.log_transform(Complex64::ONE).unwrap(), Complex64::ZERO);
        for u in [0.1, 2.0, 20.0] {
            let z = p.characteristic_function(u).unwrap();
            let nz = p.characteristic_function(-u).unwrap();
            close((z.conj() - nz).abs(), 0.0, 1e-13);
            assert!(z.abs() <= 1.0 + 1e-12);
        }
        for k in [70.0, 100.0, 150.0] {
            let v = p.price(100.0, k, 0.97).unwrap();
            close(v.call - v.put, 0.97 * (100.0 - k), 2e-14);
        }
    }
    for h in [0.01, 0.1, 0.3, 0.5] {
        let m = RoughHeston::new(h, 0.04, 0.0, 0.055, 0.0, -0.65).unwrap();
        let p = HestonFourierPlan::compile(m.into(), 1.0, cfg(32, 128, 32.0)).unwrap();
        let v = p.price(100.0, 100.0, 1.0).unwrap();
        close(v.call, 7.965567455405804, 2e-12);
        close(v.tail_indicator, 0.0, 1e-12);
        close(v.quadrature_difference, 0.0, 1e-12);
    }
    // Nonconstant deterministic variance: exercise the exactly linear Riccati
    // branch, not only the flat Black case, against the elementary OU integral.
    let linear = RoughHeston::new(0.5, 0.04, 0.7, 0.055, 0.0, -0.65).unwrap();
    let p = HestonFourierPlan::compile(linear.into(), 1.0, cfg(1024, 8, 4.0)).unwrap();
    let integrated = 0.055 + (0.04 - 0.055) * (-(-0.7_f64).exp_m1()) / 0.7;
    for z in [Complex64::new(0.5, 0.7), Complex64::new(0.0, 2.0)] {
        close(
            (p.log_transform(z).unwrap() - (z * z - z) * (0.5 * integrated)).abs(),
            0.0,
            2e-9,
        );
    }
    for h in [0.1, 0.3, 0.5] {
        let zero_nu = RoughHeston::new(h, 0.04, 0.7, 0.055, 0.0, -0.65).unwrap();
        let small_nu = RoughHeston::new(h, 0.04, 0.7, 0.055, 1e-10, -0.65).unwrap();
        let a = HestonFourierPlan::compile(zero_nu.into(), 1.0, cfg(128, 8, 4.0)).unwrap();
        let b = HestonFourierPlan::compile(small_nu.into(), 1.0, cfg(128, 8, 4.0)).unwrap();
        close(
            (a.log_transform(Complex64::new(0.5, 2.0)).unwrap()
                - b.log_transform(Complex64::new(0.5, 2.0)).unwrap())
            .abs(),
            0.0,
            1e-10,
        );
    }
    let zero = HestonFourierPlan::compile(rough(0.1).into(), 0.0, cfg(32, 32, 32.0)).unwrap();
    close(zero.price(100.0, 80.0, 0.9).unwrap().call, 18.0, 0.0);
}

#[test]
fn fractional_transform_matches_independent_high_precision_series() {
    let r = reference();
    let tolerance = r["protocol"]["rough_cf_abs_tolerance"].as_f64().unwrap();
    for h in [0.1, 0.3] {
        for t in [0.25, 1.0] {
            // A small Fourier grid suffices here; this test examines independently
            // requested transforms, not prices from this coarse inversion grid.
            let p = HestonFourierPlan::compile(rough(h).into(), t, cfg(2048, 8, 4.0)).unwrap();
            for row in r["rough_transforms"].as_array().unwrap() {
                if row["hurst"].as_f64().unwrap() != h || row["maturity"].as_f64().unwrap() != t {
                    continue;
                }
                let got = p.log_transform(complex(&row["exponent"])).unwrap();
                let expected = complex(&row["log_transform"]);
                close((got - expected).abs(), 0.0, tolerance);
            }
        }
    }
}

#[test]
fn markov_transforms_and_prices_match_independent_odes_and_p1p2() {
    let r = reference();
    for t in [0.25, 1.0] {
        for (family, model, key) in [
            ("heston", rough(0.5).into(), "heston_log"),
            ("lift", lift().into(), "lift_log"),
        ] {
            let p = HestonFourierPlan::compile(model, t, cfg(1024, 1024, 192.0)).unwrap();
            for row in r["markov_transforms"].as_array().unwrap() {
                if row["maturity"].as_f64().unwrap() != t {
                    continue;
                }
                let gap = (p.log_transform(complex(&row["exponent"])).unwrap()
                    - complex(&row[key]))
                .abs();
                close(
                    gap,
                    0.0,
                    r["protocol"]["markov_cf_abs_tolerance"].as_f64().unwrap(),
                );
            }
            for row in r["prices"].as_array().unwrap() {
                if row["family"] != family || row["maturity"].as_f64().unwrap() != t {
                    continue;
                }
                let got = p
                    .price(100.0, row["strike"].as_f64().unwrap(), 0.97)
                    .unwrap();
                let expected = row["call"].as_f64().unwrap();
                println!(
                    "oracle family={family} T={t} K={} price={:.12} reference={expected:.12} qdiff={:.3e} tail={:.3e}",
                    row["strike"], got.call, got.quadrature_difference, got.tail_indicator
                );
                close(
                    got.call,
                    expected,
                    r["protocol"]["price_abs_tolerance"].as_f64().unwrap(),
                );
                assert!(got.quadrature_difference < 1e-5 && got.tail_indicator < 1e-5);
            }
        }
    }
}

#[test]
fn heston_boundary_factor_splitting_and_stiff_lift() {
    let h = rough(0.5);
    let p = HestonFourierPlan::compile(h.clone().into(), 1.0, cfg(128, 128, 32.0)).unwrap();
    let l = LiftedHeston::from_rough(&h, 20, 2.5).unwrap();
    let q = HestonFourierPlan::compile(l.into(), 1.0, cfg(128, 128, 32.0)).unwrap();
    let split = LiftedHeston::new(
        0.04,
        0.7,
        0.055,
        0.18,
        -0.65,
        vec![0.2, 0.8],
        vec![0.0, 0.0],
    )
    .unwrap();
    let s = HestonFourierPlan::compile(split.into(), 1.0, cfg(128, 128, 32.0)).unwrap();
    for k in [80.0, 100.0, 120.0] {
        close(
            p.price(100.0, k, 0.97).unwrap().call,
            q.price(100.0, k, 0.97).unwrap().call,
            1e-10,
        );
        close(
            p.price(100.0, k, 0.97).unwrap().call,
            s.price(100.0, k, 0.97).unwrap().call,
            1e-10,
        );
    }
    let stiff = LiftedHeston::from_rough(&rough(0.1), 20, 2.5).unwrap();
    assert!(*stiff.rates().last().unwrap() > 1000.0);
    let plan = HestonFourierPlan::compile(stiff.into(), 1.0, cfg(512, 512, 128.0)).unwrap();
    assert!(plan.price(100.0, 100.0, 1.0).unwrap().call > 0.0);
}

#[test]
#[ignore = "independent Fourier refinement and public Monte Carlo panel; release CI"]
fn fourier_refinement_and_continuous_time_mc_comparison() {
    use pricing::mc::ExecutionPolicy;
    use pricing::{JsonLimits, parse_request_json};
    use serde_json::json;
    let r = reference();
    let protocol = &r["protocol"];
    let models: Vec<RoughVolatilityModel> = vec![
        rough(0.1).into(),
        rough(0.3).into(),
        lift().into(),
        LiftedHeston::from_rough(&rough(0.1), 20, 2.5)
            .unwrap()
            .into(),
    ];
    let mut failures = Vec::new();
    for (i, m) in models.into_iter().enumerate() {
        let base = HestonFourierPlan::compile(m.clone(), 1.0, cfg(1024, 512, 128.0)).unwrap();
        let time = HestonFourierPlan::compile(m.clone(), 1.0, cfg(2048, 512, 128.0)).unwrap();
        let freq = HestonFourierPlan::compile(m.clone(), 1.0, cfg(2048, 1024, 256.0)).unwrap();
        for k in [80.0, 100.0, 120.0] {
            let a = base.price(100.0, k, 1.0).unwrap();
            let b = time.price(100.0, k, 1.0).unwrap();
            let c = freq.price(100.0, k, 1.0).unwrap();
            println!(
                "refinement model={i} K={k} price={:.12} time_diff={:.9} cutoff_diff={:.9} qdiff={:.3e} tail={:.3e}",
                c.call,
                (a.call - b.call).abs(),
                (b.call - c.call).abs(),
                c.quadrature_difference,
                c.tail_indicator
            );
            close(
                a.call,
                b.call,
                protocol["refinement_abs_tolerance"].as_f64().unwrap(),
            );
            close(
                b.call,
                c.call,
                protocol["refinement_abs_tolerance"].as_f64().unwrap(),
            );
            assert!(c.quadrature_difference < 1e-5 && c.tail_indicator < 1e-5);
        }
        let expected = freq.price(100.0, 100.0, 1.0).unwrap().call;
        // The 512-step H=0.1 experiment is retained as a failed coarse-grid
        // diagnostic. Refine only that model's time grid, not the error budget.
        let mc_steps = if i == 0 {
            protocol["mc_rough_h01_steps"].as_f64().unwrap()
        } else {
            protocol["mc_steps"].as_f64().unwrap()
        };
        for seed in protocol["mc_seed"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
        {
            let mut request: Value = serde_json::from_str(include_str!(
                "../../../fixtures/v2/pricing_request.golden.json"
            ))
            .unwrap();
            request["market"]["discount_curve"]["discount_factors"] = json!([1.0, 1.0]);
            request["market"]["dividend_curve"]["discount_factors"] = json!([1.0, 1.0]);
            request["product"]["strike"] = json!(100.0);
            request["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":protocol["mc_points"],"scramble_count":protocol["mc_scrambles"],"master_scramble_seed":seed,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
            let req =
                parse_request_json(&serde_json::to_vec(&request).unwrap(), JsonLimits::DEFAULT)
                    .unwrap();
            let plan = RoughVolatilityPricingPlan::compile(
                &req,
                m.clone(),
                1.0 / mc_steps,
                ExecutionPolicy::new(4, Some(256)).unwrap(),
            )
            .unwrap();
            let result = plan.evaluate().unwrap();
            let gap = result.value - expected;
            let bound = gap.abs() + 4.0 * result.standard_error;
            println!(
                "mc model={i} seed={seed} steps={mc_steps} price={:.12} fourier={expected:.12} gap={gap:.9} se={:.9} bound={bound:.9} paths={}",
                result.value, result.standard_error, result.evaluated_paths
            );
            if bound > protocol["mc_combined_budget"].as_f64().unwrap()
                || result.standard_error <= 0.0
                || result.standard_error > protocol["mc_se_cap"].as_f64().unwrap()
            {
                failures.push((i, seed, gap, result.standard_error));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "finite MC grid comparison failures: {failures:?}"
    );
}
