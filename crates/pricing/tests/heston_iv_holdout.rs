//! Holdout validation never fits and excludes economically equivalent training sites.
use pricing::rough_volatility::*;

fn config() -> HestonFourierConfig {
    HestonFourierConfig::new(16, 128, 64.).unwrap()
}
fn policy() -> HestonIvRefinementOptions {
    HestonIvRefinementOptions {
        max_stages: 1,
        ..Default::default()
    }
}
fn quote(t: f64, k: f64, target: f64) -> HestonIvCalibrationQuote {
    HestonIvCalibrationQuote {
        maturity: t,
        forward: 100.,
        strike: k,
        discount: 0.97,
        is_call: k >= 100.,
        target_volatility: target,
        iv_scale: 1.,
    }
}
fn variables() -> Vec<HestonCalibrationVariable> {
    vec![HestonCalibrationVariable {
        parameter: HestonCalibrationParameter::InitialVariance,
        lower: 0.01,
        upper: 0.16,
        scale: 0.04,
    }]
}
fn model(random: bool, lift: bool) -> RoughVolatilityModel {
    let (k, nu) = if random { (0.7, 0.18) } else { (0., 0.) };
    if lift {
        LiftedHeston::new(
            0.04,
            k,
            0.055,
            nu,
            -0.65,
            vec![0.2, 0.4, 0.5],
            vec![0.1, 1., 8.],
        )
        .unwrap()
        .into()
    } else {
        RoughHeston::new(0.1, 0.04, k, 0.055, nu, -0.65)
            .unwrap()
            .into()
    }
}
fn problem(random: bool, lift: bool) -> HestonIvCalibrationProblem {
    HestonIvCalibrationProblem::new(
        model(random, lift),
        vec![quote(1., 100., 0.22)],
        variables(),
        config(),
    )
    .unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a:0.16e} vs {b:0.16e}");
}

#[test]
fn constant_variance_holdout_is_frozen_and_reports_absolute_iv() {
    let p = problem(false, false);
    let qs = vec![
        quote(0.5, 95., 0.25),
        quote(1., 110., 0.3),
        quote(0.5, 105., 0.25),
    ];
    let r = p.validate_holdout(&[0.0625], &qs, policy()).unwrap();
    assert_eq!(r.plan_compilations, 10); // Holdout maturities, not the training count.
    assert_eq!(r.model_implied_volatilities.len(), 3);
    for row in &r.model_implied_volatilities {
        for &v in row {
            close(v, 0.25, 2e-12);
        }
    }
    close(r.max_abs_iv_residual, 0.05, 2e-12);
    assert!(r.grid_stable && !r.fit_within_tolerance && !r.accepted);
    assert_eq!(p.initial_parameters(), vec![0.04]);
    assert_eq!(p.quotes(), &[quote(1., 100., 0.22)]);
    let mut changed = qs.clone();
    for q in &mut changed {
        q.iv_scale = 1e8;
    }
    assert_eq!(
        r,
        p.validate_holdout(&[0.0625], &changed, policy()).unwrap()
    );
    changed.reverse();
    let reversed = p.validate_holdout(&[0.0625], &changed, policy()).unwrap();
    let mut rows = r.iv_residuals.clone();
    rows.reverse();
    assert_eq!(rows, reversed.iv_residuals);
}

#[test]
fn training_overlap_and_duplicate_holdout_sites_are_rejected() {
    let p = problem(false, false);
    for mut q in [quote(1., 100., 0.2), quote(1. + f64::EPSILON, 100., 0.3)] {
        q.is_call = false;
        q.forward = 200.;
        q.strike = 200.;
        q.discount = 0.8;
        q.iv_scale = 2.;
        assert!(p.validate_holdout(&[0.04], &[q], policy()).is_err());
    }
    let first = quote(0.5, 95., 0.2);
    let mut second = first;
    second.forward *= 3.;
    second.strike *= 3.;
    second.is_call = !first.is_call;
    second.target_volatility = 0.3;
    second.discount = 0.7;
    assert!(
        p.validate_holdout(&[0.04], &[first, second], policy())
            .is_err()
    );
    // Distinct, even if close, sites outside the documented exclusion tolerance remain usable.
    assert!(
        p.validate_holdout(&[0.04], &[quote(1. + 1e-9, 100., 0.2)], policy())
            .is_ok()
    );
    // Non-ATM equivalent points are excluded too; absolute strike alone is not the key.
    let shifted = HestonIvCalibrationProblem::new(
        model(false, false),
        vec![quote(1., 110., 0.2)],
        variables(),
        config(),
    )
    .unwrap();
    let mut equivalent = quote(1., 220., 0.2);
    equivalent.forward = 200.;
    assert!(
        shifted
            .validate_holdout(&[0.04], &[equivalent], policy())
            .is_err()
    );
}

#[test]
fn every_grid_matches_public_repricing_for_both_families() {
    let qs = vec![
        quote(0.5, 95., 0.21),
        quote(0.75, 105., 0.19),
        quote(0.5, 105., 0.2),
    ];
    for lift in [false, true] {
        let p = problem(true, lift);
        let x = [0.055];
        let before = p.evaluate(&x).unwrap();
        let r = p.validate_holdout(&x, &qs, policy()).unwrap();
        assert_eq!(p.evaluate(&x).unwrap(), before);
        for (column, &c) in r.configurations.iter().enumerate() {
            let other =
                HestonIvCalibrationProblem::new(model(true, lift), qs.clone(), variables(), c)
                    .unwrap();
            let e = other.evaluate(&x).unwrap();
            for (i, &v) in e.model_implied_volatilities.iter().enumerate() {
                close(v, r.model_implied_volatilities[i][column], 0.);
                close(r.iv_residuals[i][column], v - qs[i].target_volatility, 0.);
                if column > 0 {
                    close(
                        r.iv_differences[i][column - 1],
                        v - r.model_implied_volatilities[i][0],
                        0.,
                    );
                }
            }
        }
        assert_eq!(r.plan_compilations, 10);
        assert!(r.max_abs_iv_difference > 0.); // Nonvacuous random-volatility grid diagnostics.
    }
}

#[test]
fn holdout_failure_cannot_change_a_fit_or_start_a_new_calibration() {
    let p = problem(false, false);
    let opts = LeastSquaresOptions {
        max_iterations: 40,
        max_evaluations: 60,
        ..Default::default()
    };
    let fit = p.calibrate(opts).unwrap();
    assert!(fit.fit_achieved);
    let train = p
        .validate_grid(&fit.optimizer.parameters, policy())
        .unwrap();
    assert!(train.accepted);
    let hold = p
        .validate_holdout(
            &fit.optimizer.parameters,
            &[quote(0.5, 100., 0.4)],
            policy(),
        )
        .unwrap();
    assert!(!hold.accepted);
    assert!(hold.grid_stable);
    let again = p.calibrate(opts).unwrap();
    assert_eq!(fit.optimizer.parameters, again.optimizer.parameters);
    assert_eq!(fit.evaluation, again.evaluation);
    assert_eq!(fit.evaluations, again.evaluations);
    assert_eq!(
        train,
        p.validate_grid(&fit.optimizer.parameters, policy())
            .unwrap()
    );
    assert_eq!(p.initial_parameters(), vec![0.04]);
}

#[test]
fn holdout_validation_rejects_invalid_inputs_and_never_drops_quotes() {
    let p = problem(false, false);
    let q = quote(0.5, 100., 0.2);
    assert!(p.validate_holdout(&[0.04], &[], policy()).is_err());
    assert!(
        p.validate_holdout(&[0.04], &vec![q; 4097], policy())
            .is_err()
    );
    for x in [vec![], vec![f64::NAN], vec![0.001], vec![0.04, 0.04]] {
        assert!(p.validate_holdout(&x, &[q], policy()).is_err());
    }
    for value in [0., -1., f64::NAN, f64::INFINITY] {
        for column in 0..6 {
            let mut bad = q;
            match column {
                0 => bad.maturity = value,
                1 => bad.forward = value,
                2 => bad.strike = value,
                3 => bad.discount = value,
                4 => bad.target_volatility = value,
                _ => bad.iv_scale = value,
            }
            assert!(p.validate_holdout(&[0.04], &[q, bad], policy()).is_err());
        }
    }
    let many: Vec<_> = (1..=65)
        .map(|i| quote(0.1 + i as f64 / 100., 100., 0.2))
        .collect();
    assert!(p.validate_holdout(&[0.04], &many, policy()).is_err());
    let mut bad_policy = policy();
    bad_policy.fit_tolerance = f64::NAN;
    assert!(p.validate_holdout(&[0.04], &[q], bad_policy).is_err());
    // Strict IV conditioning rejects an otherwise valid extreme-wing holdout, not zero residual.
    assert!(
        p.validate_holdout(&[0.04], &[quote(0.001, 1000., 0.2)], policy())
            .is_err()
    );
}

#[test]
#[ignore = "calibrate on training sites then validate disjoint retained quote panels"]
fn independent_and_ssvi_holdout_panels() {
    use HestonCalibrationParameter::*;
    use serde_json::Value;
    let f: Value = serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/iv-holdout.json"
    ))
    .unwrap();
    let parent: Value = serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/iv-calibration.json"
    ))
    .unwrap();
    let a = &f["protocol"];
    let config = HestonFourierConfig::new(
        a["time_steps"].as_u64().unwrap() as usize,
        a["integration_intervals"].as_u64().unwrap() as usize,
        a["cutoff"].as_f64().unwrap(),
    )
    .unwrap();
    let opts = LeastSquaresOptions {
        max_iterations: a["max_iterations"].as_u64().unwrap() as usize,
        max_evaluations: a["max_evaluations"].as_u64().unwrap() as usize,
        residual_tolerance: a["residual_tolerance"].as_f64().unwrap(),
        ..Default::default()
    };
    let mut failures = Vec::new();
    let mut rows = 0;
    let mut panels = 0;
    let mut fits = 0;
    for family in ["heston", "lift", "ssvi"] {
        let train: Vec<_> = if family == "ssvi" {
            use pricing::market::*;
            let surface = StandardSsvi::new(
                ThetaPchip::new(vec![0.25, 0.75, 1.5], vec![0.01, 0.03, 0.06], 0.04).unwrap(),
                -0.5,
                PhiSpec::PowerLaw {
                    eta: 0.35,
                    gamma: 0.5,
                },
                SurfaceValidationTolerance::local_vol_vegakt_v1(),
            )
            .unwrap();
            [0.25, 0.75, 1.5]
                .into_iter()
                .flat_map(|t| [85., 100., 115.].into_iter().map(move |k| (t, k)))
                .map(|(t, k)| {
                    HestonIvCalibrationQuote::from_ssvi(&surface, t, 100., k, 0.97, 1.).unwrap()
                })
                .collect()
        } else {
            parent["markov_quotes"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|q| q["family"] == family)
                .map(|r| {
                    quote(
                        r["maturity"].as_f64().unwrap(),
                        r["strike"].as_f64().unwrap(),
                        r["target_volatility"].as_f64().unwrap(),
                    )
                })
                .collect()
        };
        for (start_idx, start) in f["starts"].as_array().unwrap().iter().enumerate() {
            let x: Vec<_> = start
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let initial = if family == "lift" {
                LiftedHeston::new(
                    x[0],
                    x[1],
                    x[2],
                    x[3],
                    x[4],
                    vec![0.2, 0.4, 0.5],
                    vec![0.1, 1., 8.],
                )
                .unwrap()
                .into()
            } else {
                RoughHeston::new(
                    if family == "heston" { 0.5 } else { x[5] },
                    x[0],
                    x[1],
                    x[2],
                    x[3],
                    x[4],
                )
                .unwrap()
                .into()
            };
            let variables = [
                InitialVariance,
                MeanReversion,
                LongRunVariance,
                VolOfVol,
                Correlation,
                Hurst,
            ]
            .into_iter()
            .take(if family == "ssvi" { 6 } else { 5 })
            .zip(f["bounds"].as_array().unwrap())
            .map(|(parameter, b)| HestonCalibrationVariable {
                parameter,
                lower: b[0].as_f64().unwrap(),
                upper: b[1].as_f64().unwrap(),
                scale: b[2].as_f64().unwrap(),
            })
            .collect();
            let p =
                HestonIvCalibrationProblem::new(initial, train.clone(), variables, config).unwrap();
            // The optimization never sees holdout targets, coordinates or acceptance tolerances.
            let fit = p.calibrate(opts).unwrap();
            fits += 1;
            println!(
                "HOLDOUT_FIT family={family} start={start_idx} fit={} evaluations={} parameters={:?}",
                fit.fit_achieved, fit.evaluations, fit.optimizer.parameters
            );
            if family != "ssvi" && !fit.fit_achieved {
                failures.push(format!("training fit {family} {start_idx}"));
            }
            let before = p.evaluate(&fit.optimizer.parameters).unwrap();
            for region in ["interpolation", "extrapolation"] {
                let qs: Vec<_> = f["rows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|q| q["family"] == family && q["region"] == region)
                    .map(|r| {
                        quote(
                            r["maturity"].as_f64().unwrap(),
                            r["strike"].as_f64().unwrap(),
                            r["target_volatility"].as_f64().unwrap(),
                        )
                    })
                    .collect();
                if qs.is_empty() {
                    assert!(family != "ssvi" && region == "extrapolation");
                    continue;
                }
                let policy = HestonIvRefinementOptions {
                    fit_tolerance: a[format!("{region}_fit_tolerance")].as_f64().unwrap(),
                    grid_tolerance: a["grid_tolerance"].as_f64().unwrap(),
                    max_stages: 1,
                };
                let report = p
                    .validate_holdout(&fit.optimizer.parameters, &qs, policy)
                    .unwrap();
                panels += 1;
                println!(
                    "HOLDOUT_PANEL family={family} start={start_idx} region={region} quotes={} accepted={} max_residual={:0.12e} max_difference={:0.12e} plans={}",
                    qs.len(),
                    report.accepted,
                    report.max_abs_iv_residual,
                    report.max_abs_iv_difference,
                    report.plan_compilations
                );
                for (i, q) in qs.iter().enumerate() {
                    rows += 1;
                    println!(
                        "HOLDOUT_ROW family={family} start={start_idx} region={region} t={} k={} residuals={:?} differences={:?}",
                        q.maturity, q.strike, report.iv_residuals[i], report.iv_differences[i]
                    );
                }
                if !report.accepted {
                    failures.push(format!(
                        "{family} {start_idx} {region}: residual {} difference {}",
                        report.max_abs_iv_residual, report.max_abs_iv_difference
                    ));
                }
            }
            assert_eq!(p.evaluate(&fit.optimizer.parameters).unwrap(), before);
        }
    }
    assert_eq!((fits, panels, rows), (6, 8, 100));
    assert!(
        failures.is_empty(),
        "holdout criteria failed (all rows retained): {failures:?}"
    );
}
