//! Finite-grid validation semantics, immutable warm-start stages and independent IV targets.
use pricing::rough_volatility::*;
use serde_json::Value;

fn cfg() -> HestonFourierConfig {
    HestonFourierConfig::new(16, 128, 64.).unwrap()
}
fn policy(stages: usize) -> HestonIvRefinementOptions {
    HestonIvRefinementOptions {
        max_stages: stages,
        ..Default::default()
    }
}
fn options() -> LeastSquaresOptions {
    LeastSquaresOptions {
        max_iterations: 40,
        max_evaluations: 60,
        ..Default::default()
    }
}
fn model(v0: f64, random: bool, lift: bool) -> RoughVolatilityModel {
    let (kappa, nu) = if random { (0.7, 0.18) } else { (0., 0.) };
    if lift {
        LiftedHeston::new(
            v0,
            kappa,
            0.055,
            nu,
            -0.65,
            vec![0.2, 0.4, 0.5],
            vec![0.1, 1., 8.],
        )
        .unwrap()
        .into()
    } else {
        RoughHeston::new(0.1, v0, kappa, 0.055, nu, -0.65)
            .unwrap()
            .into()
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
fn q(t: f64, k: f64, target: f64) -> HestonIvCalibrationQuote {
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
fn problem(
    m: RoughVolatilityModel,
    quotes: Vec<HestonIvCalibrationQuote>,
    c: HestonFourierConfig,
) -> HestonIvCalibrationProblem {
    HestonIvCalibrationProblem::new(m, quotes, variables(), c).unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a:.15e} vs {b:.15e}, tol {tol}");
}
fn check_report(r: &HestonIvGridValidation, quotes: &[HestonIvCalibrationQuote]) {
    assert_eq!(r.model_implied_volatilities.len(), quotes.len());
    let mut max_r = 0f64;
    let mut max_d = 0f64;
    for (i, q) in quotes.iter().enumerate() {
        for j in 0..5 {
            close(
                r.iv_residuals[i][j],
                r.model_implied_volatilities[i][j] - q.target_volatility,
                0.,
            );
            max_r = max_r.max(r.iv_residuals[i][j].abs());
        }
        for j in 0..4 {
            close(
                r.iv_differences[i][j],
                r.model_implied_volatilities[i][j + 1] - r.model_implied_volatilities[i][0],
                0.,
            );
            max_d = max_d.max(r.iv_differences[i][j].abs());
        }
    }
    close(r.max_abs_iv_residual, max_r, 0.);
    close(r.max_abs_iv_difference, max_d, 0.);
    assert_eq!(r.fit_within_tolerance, max_r <= r.fit_tolerance);
    assert_eq!(r.grid_stable, max_d <= r.grid_tolerance);
    assert_eq!(r.accepted, r.fit_within_tolerance && r.grid_stable);
}
#[test]
fn exact_black_validation_is_unscaled_preserves_order_and_does_not_hide_mismatch() {
    let mut quotes = vec![
        q(1., 100., 0.2),
        q(0.25, 90., 0.2),
        q(1., 110., 0.201),
        q(1., 100., 0.2),
    ];
    for (q, s) in quotes.iter_mut().zip([1e-3, 2., 100., 0.1]) {
        q.iv_scale = s;
    }
    let p = problem(model(0.04, false, false), quotes.clone(), cfg());
    let r = p.validate_grid(&[0.04], policy(2)).unwrap();
    assert_eq!(r.plan_compilations, 10);
    check_report(&r, &quotes);
    for row in &r.model_implied_volatilities {
        for v in row {
            close(*v, 0.2, 2e-12);
        }
    }
    close(r.max_abs_iv_residual, 0.001, 2e-12);
    assert!(!r.fit_within_tolerance && r.grid_stable && !r.accepted);
    let altered = problem(
        model(0.04, false, false),
        quotes
            .iter()
            .map(|q| HestonIvCalibrationQuote { iv_scale: 1., ..*q })
            .collect(),
        cfg(),
    );
    assert_eq!(r, altered.validate_grid(&[0.04], policy(1)).unwrap());
}
#[test]
fn all_probes_match_separate_public_evaluations_at_the_same_point() {
    for lift in [false, true] {
        let quotes = vec![q(1., 100., 0.2), q(0.25, 90., 0.23), q(1., 110., 0.19)];
        let p = problem(model(0.04, true, lift), quotes.clone(), cfg());
        let r = p.validate_grid(&[0.045], policy(1)).unwrap();
        assert_eq!(
            r.configurations
                .map(|c| (c.time_steps(), c.integration_intervals(), c.cutoff())),
            [
                (16, 128, 64.),
                (32, 128, 64.),
                (16, 256, 64.),
                (16, 256, 128.),
                (32, 512, 128.)
            ]
        );
        for (j, &c) in r.configurations.iter().enumerate() {
            let e = problem(model(0.04, true, lift), quotes.clone(), c)
                .evaluate(&[0.045])
                .unwrap();
            for (i, &iv) in e.model_implied_volatilities.iter().enumerate() {
                close(r.model_implied_volatilities[i][j], iv, 0.);
            }
        }
        check_report(&r, &quotes);
        assert!(r.iv_differences.iter().any(|row| row[0].abs() > 1e-8));
        assert_eq!(p.initial_parameters(), vec![0.04]);
    }
}
#[test]
fn accepted_fit_stops_early_without_mutating_the_problem() {
    let p = problem(
        model(0.04, false, false),
        vec![q(0.25, 100., 0.22), q(1., 110., 0.22)],
        cfg(),
    );
    let ordinary = p.calibrate(options()).unwrap();
    let r = p.calibrate_refined(options(), policy(2)).unwrap();
    assert!(r.accepted);
    assert_eq!(r.stages.len(), 1);
    assert_eq!(
        ordinary.optimizer.parameters,
        r.stages[0].calibration.optimizer.parameters
    );
    assert_eq!(ordinary.evaluation, r.stages[0].calibration.evaluation);
    assert_eq!(r.optimizer_evaluations, ordinary.evaluations);
    assert_eq!(r.validation_plan_compilations, 10);
    close(
        r.stages[0].calibration.optimizer.parameters[0],
        0.22f64.powi(2),
        2e-7,
    );
    assert_eq!(p.initial_parameters(), vec![0.04]);
}
#[test]
fn unattainable_fit_exhausts_stages_and_records_exact_warm_start() {
    // Identical contracts with conflicting IV targets. Analytic LS minimum is sigma=.25.
    let p = problem(
        model(0.04, false, true),
        vec![q(1., 100., 0.2), q(1., 100., 0.3)],
        cfg(),
    );
    let r = p.calibrate_refined(options(), policy(2)).unwrap();
    assert!(!r.accepted);
    assert_eq!(r.stages.len(), 2);
    assert_eq!(
        r.stages[1].initial_parameters,
        r.stages[0].calibration.optimizer.parameters
    );
    assert_eq!(
        r.stages[1].validation.configurations[0],
        r.stages[0].validation.configurations[4]
    );
    assert_eq!(
        r.optimizer_evaluations,
        r.stages
            .iter()
            .map(|s| s.calibration.evaluations)
            .sum::<usize>()
    );
    assert_eq!(r.validation_plan_compilations, 10);
    for s in &r.stages {
        close(s.calibration.optimizer.parameters[0], 0.0625, 2e-7);
        assert!(
            !s.calibration.fit_achieved
                && !s.validation.fit_within_tolerance
                && s.validation.grid_stable
        );
        check_report(&s.validation, p.quotes());
    }
}
#[test]
fn a_coarse_grid_fit_is_not_automatically_accepted() {
    let c = HestonFourierConfig::new(8, 128, 64.).unwrap();
    let tmp = problem(model(0.04, true, false), vec![q(1., 100., 0.2)], c);
    let iv = tmp.evaluate(&[0.04]).unwrap().model_implied_volatilities[0];
    let p = problem(model(0.04, true, false), vec![q(1., 100., iv)], c);
    let r = p
        .calibrate_refined(
            options(),
            HestonIvRefinementOptions {
                fit_tolerance: 1.,
                grid_tolerance: 1e-14,
                max_stages: 1,
            },
        )
        .unwrap();
    assert!(r.stages[0].calibration.fit_achieved);
    assert!(r.stages[0].validation.fit_within_tolerance);
    assert!(!r.stages[0].validation.grid_stable && !r.accepted);
}
#[test]
fn invalid_inputs_future_grids_and_work_budgets_are_rejected() {
    let p = problem(model(0.04, false, false), vec![q(1., 100., 0.2)], cfg());
    for bad in [
        HestonIvRefinementOptions {
            fit_tolerance: f64::NAN,
            ..policy(1)
        },
        HestonIvRefinementOptions {
            grid_tolerance: 0.,
            ..policy(1)
        },
        policy(0),
        policy(4),
    ] {
        assert!(p.validate_grid(&[0.04], bad).is_err());
        assert!(p.calibrate_refined(options(), bad).is_err());
    }
    for x in [vec![], vec![f64::NAN], vec![0.001]] {
        assert!(p.validate_grid(&x, policy(1)).is_err());
    }
    let p = problem(
        model(0.04, false, false),
        vec![q(1., 100., 0.2)],
        HestonFourierConfig::new(32, 2048, 64.).unwrap(),
    );
    // Even an exact initial fit cannot evade preflight of the impossible second joint grid.
    assert!(p.calibrate_refined(options(), policy(2)).is_err());
    let p = problem(
        model(0.04, false, false),
        vec![q(1., 100., 0.2)],
        HestonFourierConfig::new(1024, 512, 64.).unwrap(),
    );
    assert!(
        p.calibrate_refined(
            LeastSquaresOptions {
                max_evaluations: 10001,
                ..options()
            },
            policy(1)
        )
        .is_err()
    );
    assert!(
        p.calibrate_refined(
            LeastSquaresOptions {
                max_evaluations: 1,
                ..options()
            },
            policy(1)
        )
        .is_err()
    );
}

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/iv-refinement.json"
    ))
    .unwrap()
}
#[test]
#[ignore = "multistage calibration and finite-grid numerical acceptance"]
fn independent_targets_and_ssvi_pass_finite_grid_policy() {
    let f = fixture();
    let parent: Value = serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/iv-calibration.json"
    ))
    .unwrap();
    let a = &f["protocol"];
    let c = HestonFourierConfig::new(
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
    let pol = HestonIvRefinementOptions {
        fit_tolerance: a["fit_tolerance"].as_f64().unwrap(),
        grid_tolerance: a["grid_tolerance"].as_f64().unwrap(),
        max_stages: a["max_stages"].as_u64().unwrap() as usize,
    };
    use HestonCalibrationParameter::*;
    for family in ["heston", "lift", "ssvi", "ssvi_tight"] {
        let quotes: Vec<_> = if family.starts_with("ssvi") {
            use pricing::market::*;
            let surf = StandardSsvi::new(
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
                    HestonIvCalibrationQuote::from_ssvi(&surf, t, 100., k, 0.97, 1.).unwrap()
                })
                .collect()
        } else {
            parent["markov_quotes"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| r["family"] == family)
                .map(|r| {
                    q(
                        r["maturity"].as_f64().unwrap(),
                        r["strike"].as_f64().unwrap(),
                        r["target_volatility"].as_f64().unwrap(),
                    )
                })
                .collect()
        };
        for (start_idx, start) in f["starts"].as_array().unwrap().iter().enumerate() {
            let p: Vec<f64> = start
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let m = if family == "lift" {
                LiftedHeston::new(
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    p[4],
                    vec![0.2, 0.4, 0.5],
                    vec![0.1, 1., 8.],
                )
                .unwrap()
                .into()
            } else {
                RoughHeston::new(
                    if family == "heston" { 0.5 } else { p[5] },
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    p[4],
                )
                .unwrap()
                .into()
            };
            let vars = [
                InitialVariance,
                MeanReversion,
                LongRunVariance,
                VolOfVol,
                Correlation,
                Hurst,
            ]
            .into_iter()
            .take(if family.starts_with("ssvi") { 6 } else { 5 })
            .zip(f["bounds"].as_array().unwrap())
            .map(|(parameter, row)| HestonCalibrationVariable {
                parameter,
                lower: row[0].as_f64().unwrap(),
                upper: row[1].as_f64().unwrap(),
                scale: row[2].as_f64().unwrap(),
            })
            .collect();
            let initial_grid = if family == "ssvi_tight" {
                HestonFourierConfig::new(
                    a["warm_start_time_steps"].as_u64().unwrap() as usize,
                    c.integration_intervals(),
                    c.cutoff(),
                )
                .unwrap()
            } else {
                c
            };
            let problem =
                HestonIvCalibrationProblem::new(m, quotes.clone(), vars, initial_grid).unwrap();
            let selected_policy = if family == "ssvi_tight" {
                HestonIvRefinementOptions {
                    grid_tolerance: a["warm_start_grid_tolerance"].as_f64().unwrap(),
                    ..pol
                }
            } else {
                pol
            };
            let out = problem.calibrate_refined(opts, selected_policy).unwrap();
            if family == "ssvi_tight" {
                assert_eq!(out.stages.len(), 2);
                assert!(!out.stages[0].validation.grid_stable);
                assert_eq!(
                    out.stages[1].initial_parameters,
                    out.stages[0].calibration.optimizer.parameters
                );
                assert_eq!(
                    out.stages[1].validation.configurations[0],
                    out.stages[0].validation.configurations[4]
                );
            }
            for (level, s) in out.stages.iter().enumerate() {
                check_report(&s.validation, &quotes);
                println!(
                    "IV_GRID family={family} start={start_idx} stage={level} optimizer_fit={} accepted={} evaluations={} max_residual={:.12e} max_difference={:.12e}",
                    s.calibration.fit_achieved,
                    s.validation.accepted,
                    s.calibration.evaluations,
                    s.validation.max_abs_iv_residual,
                    s.validation.max_abs_iv_difference
                );
                for (i, row) in s.validation.iv_differences.iter().enumerate() {
                    println!(
                        "IV_PROBE family={family} start={start_idx} stage={level} quote={i} time={:.12e} frequency={:.12e} cutoff={:.12e} joint={:.12e}",
                        row[0], row[1], row[2], row[3]
                    );
                }
            }
            assert!(out.accepted, "{family} start {start_idx}");
        }
    }
}
