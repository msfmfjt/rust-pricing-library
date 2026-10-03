//! Calibration contracts and independent-price/synthetic finite-grid panels.
use pricing::rough_volatility::*;
const P: [f64; 6] = [0.04, 0.7, 0.055, 0.18, -0.65, 0.1];
fn rough(p: [f64; 6]) -> RoughVolatilityModel {
    RoughHeston::new(p[5], p[0], p[1], p[2], p[3], p[4])
        .unwrap()
        .into()
}
fn lift(p: [f64; 6]) -> RoughVolatilityModel {
    LiftedHeston::new(
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        vec![0.2, 0.4, 0.5],
        vec![0.1, 1.0, 8.0],
    )
    .unwrap()
    .into()
}
fn config() -> HestonFourierConfig {
    HestonFourierConfig::new(32, 128, 48.0).unwrap()
}
fn variables() -> Vec<HestonCalibrationVariable> {
    use HestonCalibrationParameter::*;
    [
        InitialVariance,
        MeanReversion,
        LongRunVariance,
        VolOfVol,
        Correlation,
        Hurst,
    ]
    .into_iter()
    .zip([
        (0.01, 0.1, 0.04),
        (0.1, 2.0, 0.7),
        (0.015, 0.12, 0.05),
        (0.03, 0.4, 0.2),
        (-0.95, -0.05, 0.5),
        (0.02, 0.49, 0.2),
    ])
    .map(
        |(parameter, (lower, upper, scale))| HestonCalibrationVariable {
            parameter,
            lower,
            upper,
            scale,
        },
    )
    .collect()
}
fn quotes(model: RoughVolatilityModel, cfg: HestonFourierConfig) -> Vec<HestonCalibrationQuote> {
    let mut quotes = Vec::new();
    for t in [0.25, 0.75, 1.5] {
        let p = HestonFourierPlan::compile(model.clone(), t, cfg).unwrap();
        for k in [85.0, 100.0, 115.0] {
            let is_call = k >= 100.0;
            let price = p.price(100.0, k, 0.97).unwrap();
            quotes.push(HestonCalibrationQuote {
                maturity: t,
                forward: 100.0,
                strike: k,
                discount: 0.97,
                is_call,
                target_price: if is_call { price.call } else { price.put },
                price_scale: 0.1,
            });
        }
    }
    quotes
}
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} vs {b}, tol={tol}");
}
#[test]
fn analytic_jacobian_preserves_quote_order_scaling_and_fixed_parameters() {
    let v = variables();
    let mut q = quotes(rough(P), config());
    q.reverse();
    for (i, q) in q.iter_mut().enumerate() {
        q.price_scale = 0.1 + i as f64 * 0.02;
    }
    let prob = HestonCalibrationProblem::new(rough(P), q.clone(), v, config()).unwrap();
    assert_eq!(prob.maturity_count(), 3);
    let x = prob.initial_parameters();
    let base = prob.evaluate(&x).unwrap();
    assert!(base.scaled_residuals.iter().all(|r| *r == 0.0));
    for j in 0..6 {
        for d in [1e-5, 5e-6] {
            let mut up = x.clone();
            let mut down = x.clone();
            up[j] += d;
            down[j] -= d;
            let a = prob.evaluate(&up).unwrap();
            let b = prob.evaluate(&down).unwrap();
            for i in 0..q.len() {
                close(
                    base.jacobian[i * 6 + j],
                    (a.scaled_residuals[i] - b.scaled_residuals[i]) / (2.0 * d),
                    3e-4,
                );
            }
        }
    }
    let prob =
        HestonCalibrationProblem::new(lift(P), q, variables()[..5].to_vec(), config()).unwrap();
    let r = prob
        .calibrate(LeastSquaresOptions {
            max_evaluations: 2,
            ..Default::default()
        })
        .unwrap();
    assert!(!r.fit_achieved);
    assert_eq!(
        r.optimizer.termination,
        LeastSquaresTermination::MaxEvaluations
    );
    assert_eq!(r.evaluations, 2);
    if let RoughVolatilityModel::LiftedHeston(m) = r.model {
        assert_eq!(m.weights(), [0.2, 0.4, 0.5]);
        assert_eq!(m.rates(), [0.1, 1.0, 8.0]);
    } else {
        panic!("wrong family");
    }
}
#[test]
fn black_variance_recovery_and_infeasible_fit_status() {
    let mut p = P;
    p[1] = 0.0;
    p[3] = 0.0;
    let q = quotes(rough(p), config());
    let mut initial = p;
    initial[0] = 0.06;
    let v = vec![variables()[0]];
    let prob =
        HestonCalibrationProblem::new(rough(initial), q.clone(), v.clone(), config()).unwrap();
    let r = prob.calibrate(LeastSquaresOptions::default()).unwrap();
    assert!(r.fit_achieved);
    close(r.optimizer.parameters[0], p[0], 1e-8);
    assert_eq!(
        r.optimizer.accepted_objectives[0],
        0.5 * prob
            .evaluate(&[0.06])
            .unwrap()
            .scaled_residuals
            .iter()
            .map(|r| r * r)
            .sum::<f64>()
    );
    assert!(
        r.optimizer
            .accepted_objectives
            .windows(2)
            .all(|w| w[1] < w[0])
    );
    if let RoughVolatilityModel::RoughHeston(m) = r.model {
        assert_eq!(m.mean_reversion(), 0.0);
        assert_eq!(m.vol_of_vol(), 0.0);
        assert_eq!(m.hurst(), P[5]);
    } else {
        panic!();
    }
    // Truth is outside the fitted variance interval. A stationary bound is NOT a fit.
    let mut bad = v;
    bad[0].lower = 0.05;
    let prob = HestonCalibrationProblem::new(rough(initial), q, bad, config()).unwrap();
    let r = prob.calibrate(LeastSquaresOptions::default()).unwrap();
    assert!(!r.fit_achieved);
    assert_eq!(r.active_bounds, [true]);
    close(r.optimizer.parameters[0], 0.05, 0.0);
    assert_eq!(
        r.optimizer.termination,
        LeastSquaresTermination::ProjectedGradientTolerance
    );
}
#[test]
fn invalid_inputs_and_work_budget_are_explicit() {
    let q = quotes(rough(P), config());
    let vars = variables();
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut qs = q.clone();
        qs[0].price_scale = bad;
        assert!(HestonCalibrationProblem::new(rough(P), qs, vars.clone(), config()).is_err());
    }
    let mut dup = vars.clone();
    dup[1] = dup[0];
    assert!(HestonCalibrationProblem::new(rough(P), q.clone(), dup, config()).is_err());
    assert!(HestonCalibrationProblem::new(lift(P), q.clone(), vars.clone(), config()).is_err());
    let mut bad = q.clone();
    bad[0].target_price = -1.0;
    assert!(HestonCalibrationProblem::new(rough(P), bad, vars.clone(), config()).is_err());
    let mut bad = vars.clone();
    bad[0].lower = 0.0;
    assert!(HestonCalibrationProblem::new(rough(P), q.clone(), bad, config()).is_err());
    let p = HestonCalibrationProblem::new(rough(P), q.clone(), vars.clone(), config()).unwrap();
    assert!(p.evaluate(&[]).is_err());
    assert!(p.evaluate(&[f64::NAN; 6]).is_err());
    assert!(
        p.calibrate(LeastSquaresOptions {
            max_evaluations: 1,
            ..Default::default()
        })
        .is_err()
    );
    let p = HestonCalibrationProblem::new(
        rough(P),
        q,
        vars,
        HestonFourierConfig::new(512, 512, 128.0).unwrap(),
    )
    .unwrap();
    assert!(
        p.calibrate(LeastSquaresOptions {
            max_evaluations: 10_000,
            ..Default::default()
        })
        .is_err()
    );
}
#[test]
fn selected_hurst_recovery_and_repeatability() {
    let q = quotes(rough(P), config());
    let mut initial = P;
    initial[5] = 0.22;
    let p =
        HestonCalibrationProblem::new(rough(initial), q, vec![variables()[5]], config()).unwrap();
    let options = LeastSquaresOptions {
        residual_tolerance: 1e-6,
        ..Default::default()
    };
    let a = p.calibrate(options).unwrap();
    let b = p.calibrate(options).unwrap();
    assert!(a.fit_achieved);
    close(a.optimizer.parameters[0], 0.1, 1e-5);
    assert_eq!(a.evaluation, b.evaluation);
    assert_eq!(a.optimizer, b.optimizer);
}
#[test]
#[ignore = "multi-parameter calibration numerical acceptance panel"]
fn independent_markov_prices_and_rough_synthetic_surface_fits() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/fourier.json"
    ))
    .unwrap();
    let protocol: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/calibration.json"
    ))
    .unwrap();
    let grid = |name: &str| {
        let g = &protocol[name];
        HestonFourierConfig::new(
            g["time_steps"].as_u64().unwrap() as usize,
            g["integration_intervals"].as_u64().unwrap() as usize,
            g["cutoff"].as_f64().unwrap(),
        )
        .unwrap()
    };
    let cfg = grid("fit_grid");
    for family in ["heston", "lift", "rough"] {
        for start in 0..2 {
            let mut initial = P;
            initial[5] = if family == "heston" { 0.5 } else { 0.16 };
            initial[0] = if start == 0 { 0.045 } else { 0.035 };
            initial[1] = if start == 0 { 0.85 } else { 0.55 };
            initial[2] = if start == 0 { 0.06 } else { 0.05 };
            initial[3] = if start == 0 { 0.21 } else { 0.15 };
            initial[4] = if start == 0 { -0.55 } else { -0.75 };
            let model = if family == "lift" {
                lift(initial)
            } else {
                rough(initial)
            };
            let qs = if family == "rough" {
                quotes(rough(P), cfg)
            } else {
                fixture["prices"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|r| r["family"] == family)
                    .map(|r| HestonCalibrationQuote {
                        maturity: r["maturity"].as_f64().unwrap(),
                        forward: 100.0,
                        strike: r["strike"].as_f64().unwrap(),
                        discount: 0.97,
                        is_call: true,
                        target_price: r["call"].as_f64().unwrap(),
                        price_scale: 1.0,
                    })
                    .collect()
            };
            let vs = if family == "rough" {
                variables()
            } else {
                variables()[..5].to_vec()
            };
            let problem = HestonCalibrationProblem::new(model, qs, vs, cfg).unwrap();
            let r = problem
                .calibrate(LeastSquaresOptions {
                    max_iterations: protocol["max_iterations"].as_u64().unwrap() as usize,
                    max_evaluations: protocol["max_evaluations"].as_u64().unwrap() as usize,
                    residual_tolerance: protocol["residual_tolerance"].as_f64().unwrap(),
                    ..Default::default()
                })
                .unwrap();
            let err = r
                .evaluation
                .scaled_residuals
                .iter()
                .fold(0.0_f64, |a, b| a.max(b.abs()));
            println!(
                "CALIBRATION family={family} start={start} fit={} status={:?} evaluations={} objective={:.12e} max_scaled_residual={:.12e} parameters={:?}",
                r.fit_achieved,
                r.optimizer.termination,
                r.evaluations,
                r.optimizer.objective,
                err,
                r.optimizer.parameters
            );
            assert!(
                r.fit_achieved,
                "fit failed: {family} {start}, {err}, {:?}",
                r.optimizer.termination
            );
            assert!(r.evaluation.model_prices.iter().all(|p| p.is_finite()));
            // Independent final reprice at a finer grid. This measures a finite
            // numerical shift, not continuum/model/identifiability uncertainty.
            for q in problem.quotes() {
                let fine =
                    HestonFourierPlan::compile(r.model.clone(), q.maturity, grid("check_grid"))
                        .unwrap()
                        .price(q.forward, q.strike, q.discount)
                        .unwrap();
                let actual = if q.is_call { fine.call } else { fine.put };
                println!(
                    "REPRICE family={family} start={start} t={} k={} gap={:.12e}",
                    q.maturity,
                    q.strike,
                    actual - q.target_price
                );
                assert!(
                    (actual - q.target_price).abs() <= protocol["reprice_budget"].as_f64().unwrap()
                );
            }
        }
    }
}
