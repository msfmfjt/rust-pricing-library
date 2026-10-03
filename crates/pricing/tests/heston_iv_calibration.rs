//! Exact-IV objective, independent target inputs and model-mismatch-aware calibration.
use pricing::market::{PhiSpec, StandardSsvi, SurfaceValidationTolerance, ThetaPchip};
use pricing::rough_volatility::*;
use serde_json::Value;
const P: [f64; 6] = [0.04, 0.7, 0.055, 0.18, -0.65, 0.1];
fn model(p: [f64; 6], family: &str) -> RoughVolatilityModel {
    if family == "lift" {
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
    }
}
fn vars(n: usize) -> Vec<HestonCalibrationVariable> {
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
    .take(n)
    .zip([
        (0.01, 0.1, 0.04),
        (0.1, 2., 0.7),
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
fn cfg() -> HestonFourierConfig {
    HestonFourierConfig::new(32, 256, 64.).unwrap()
}
fn q(t: f64, k: f64, vol: f64) -> HestonIvCalibrationQuote {
    HestonIvCalibrationQuote {
        maturity: t,
        forward: 100.,
        strike: k,
        discount: 0.97,
        is_call: k >= 100.,
        target_volatility: vol,
        iv_scale: 1.,
    }
}
fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/iv-calibration.json"
    ))
    .unwrap()
}
fn num(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap()
}
fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a:0.14e} vs {b:0.14e} tol={tol}");
}
#[test]
fn black_input_and_deterministic_iv_jacobian_match_independent_references() {
    let f = fixture();
    for row in f["black"].as_array().unwrap() {
        for call in [false, true] {
            let quote = HestonIvCalibrationQuote {
                maturity: num(row, "maturity"),
                forward: num(row, "forward"),
                strike: num(row, "strike"),
                discount: num(row, "discount"),
                is_call: call,
                target_volatility: num(row, "volatility"),
                iv_scale: 0.02,
            };
            let price = quote.to_price_quote().unwrap();
            close(
                price.target_price,
                num(row, if call { "call" } else { "put" }),
                2e-12,
            );
            let p = [quote.target_volatility.powi(2), 0., 0.04, 0., 0., 0.3];
            let problem = HestonIvCalibrationProblem::new(
                model(p, "rough"),
                vec![quote],
                vec![HestonCalibrationVariable {
                    parameter: HestonCalibrationParameter::InitialVariance,
                    lower: 0.001,
                    upper: 0.3,
                    scale: 0.04,
                }],
                cfg(),
            )
            .unwrap();
            let e = problem.evaluate(&[p[0]]).unwrap();
            close(
                e.model_implied_volatilities[0],
                quote.target_volatility,
                2e-8,
            );
            close(e.model_vegas[0], num(row, "vega"), 2e-8);
            close(
                e.jacobian[0],
                1. / (2. * quote.target_volatility * quote.iv_scale),
                2e-6,
            );
        }
    }
}
#[test]
fn iv_jacobian_is_model_vega_chain_rule_not_target_vega_weighting() {
    for family in ["rough", "lift"] {
        let mut quotes = vec![
            q(0.25, 90., 0.3),
            q(1., 110., 0.3),
            q(0.25, 100., 0.3),
            q(1., 100., 0.3),
            q(0.75, 85., 0.3),
            q(0.75, 115., 0.3),
        ];
        for (i, q) in quotes.iter_mut().enumerate() {
            q.iv_scale = 0.1 + 0.01 * i as f64;
        }
        let mut v = vars(if family == "rough" { 6 } else { 5 });
        v.reverse();
        let m = model(P, family);
        let pp = HestonCalibrationProblem::new(
            m.clone(),
            quotes.iter().map(|q| q.to_price_quote().unwrap()).collect(),
            v.clone(),
            cfg(),
        )
        .unwrap();
        let ip = HestonIvCalibrationProblem::new(m, quotes.clone(), v, cfg()).unwrap();
        let x = ip.initial_parameters();
        let e = ip.evaluate(&x).unwrap();
        let p = pp.evaluate(&x).unwrap();
        assert_eq!(e.model_prices, p.model_prices);
        assert_eq!(e.price_tail_indicators, p.tail_indicators);
        assert_eq!(ip.maturity_count(), 3);
        for (i, quote) in quotes.iter().enumerate() {
            close(
                e.scaled_residuals[i],
                (e.model_implied_volatilities[i] - 0.3) / quote.iv_scale,
                1e-14,
            );
            assert!(
                (e.scaled_residuals[i]
                    - (p.model_prices[i] - pp.quotes()[i].target_price)
                        / (e.model_vegas[i] * quote.iv_scale))
                    .abs()
                    > 1e-5
            );
            for j in 0..x.len() {
                close(
                    e.jacobian[i * x.len() + j] * e.model_vegas[i] * quote.iv_scale,
                    p.jacobian[i * x.len() + j],
                    1e-10,
                );
            }
        }
        for j in 0..x.len() {
            for h in [1e-5, 5e-6] {
                let mut up = x.clone();
                let mut down = x.clone();
                up[j] += h;
                down[j] -= h;
                let a = ip.evaluate(&up).unwrap();
                let b = ip.evaluate(&down).unwrap();
                for i in 0..quotes.len() {
                    close(
                        e.jacobian[i * x.len() + j],
                        (a.scaled_residuals[i] - b.scaled_residuals[i]) / (2. * h),
                        3e-5,
                    );
                }
            }
        }
    }
}
#[test]
fn ssvi_reuses_validated_surface_and_forward_moneyness() {
    let f = fixture();
    for kind in ["power_law", "heston_like"] {
        let phi = if kind == "power_law" {
            PhiSpec::PowerLaw {
                eta: 0.35,
                gamma: 0.5,
            }
        } else {
            PhiSpec::HestonLike { lambda: 1. }
        };
        let s = StandardSsvi::new(
            ThetaPchip::new(vec![0.25, 0.75, 1.5], vec![0.012, 0.035, 0.072], 0.048).unwrap(),
            -0.5,
            phi,
            SurfaceValidationTolerance::local_vol_vegakt_v1(),
        )
        .unwrap();
        for row in f["ssvi_quotes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["kind"] == kind)
        {
            let q = HestonIvCalibrationQuote::from_ssvi(
                &s,
                num(row, "maturity"),
                num(row, "forward"),
                num(row, "strike"),
                num(row, "discount"),
                0.001,
            )
            .unwrap();
            close(q.target_volatility, num(row, "target_volatility"), 2e-12);
            assert_eq!(q.is_call, row["is_call"].as_bool().unwrap());
            let scaled = HestonIvCalibrationQuote::from_ssvi(
                &s,
                q.maturity,
                q.forward * 2.,
                q.strike * 2.,
                0.5,
                q.iv_scale,
            )
            .unwrap();
            close(scaled.target_volatility, q.target_volatility, 2e-14);
        }
    }
}
#[test]
fn invalid_iv_inputs_and_nonfit_stops_are_explicit() {
    let base = q(1., 100., 0.2);
    let m = model([0.06, 0., 0.04, 0., 0., 0.1], "rough");
    let v = vec![vars(1)[0]];
    for bad in [
        HestonIvCalibrationQuote {
            maturity: 0.,
            ..base
        },
        HestonIvCalibrationQuote {
            target_volatility: 0.,
            ..base
        },
        HestonIvCalibrationQuote {
            target_volatility: 9.,
            ..base
        },
        HestonIvCalibrationQuote {
            iv_scale: 0.,
            ..base
        },
        HestonIvCalibrationQuote {
            target_volatility: f64::NAN,
            ..base
        },
        HestonIvCalibrationQuote {
            forward: 0.,
            ..base
        },
        HestonIvCalibrationQuote {
            strike: 1e100,
            ..base
        },
    ] {
        assert!(HestonIvCalibrationProblem::new(m.clone(), vec![bad], v.clone(), cfg()).is_err());
    }
    let ivp = HestonIvCalibrationProblem::new(
        m.clone(),
        vec![base],
        vec![HestonCalibrationVariable {
            lower: 0.05,
            ..v[0]
        }],
        cfg(),
    )
    .unwrap();
    let r = ivp.calibrate(LeastSquaresOptions::default()).unwrap();
    assert!(!r.fit_achieved);
    assert!(r.active_bounds[0]);
    close(r.optimizer.parameters[0], 0.05, 0.);
    let ivp = HestonIvCalibrationProblem::new(m, vec![base], v.clone(), cfg()).unwrap();
    let r = ivp
        .calibrate(LeastSquaresOptions {
            max_evaluations: 2,
            ..LeastSquaresOptions::default()
        })
        .unwrap();
    assert!(!r.fit_achieved);
    let low = HestonIvCalibrationProblem::new(
        model([1e-12, 0., 0.04, 0., 0., 0.1], "rough"),
        vec![q(1., 120., 0.2)],
        vec![HestonCalibrationVariable {
            lower: 1e-14,
            ..v[0]
        }],
        cfg(),
    )
    .unwrap();
    assert!(low.evaluate(&low.initial_parameters()).is_err());
    assert!(low.calibrate(LeastSquaresOptions::default()).is_err());
}

#[test]
#[ignore = "nondegenerate multi-expiry IV fits and SSVI misspecification; explicit release CI"]
fn numerical_iv_fits_and_ssvi_repricing() {
    let f = fixture();
    let p = &f["protocol"];
    let c = HestonFourierConfig::new(
        p["time_steps"].as_u64().unwrap() as usize,
        p["integration_intervals"].as_u64().unwrap() as usize,
        num(p, "cutoff"),
    )
    .unwrap();
    let fine = HestonFourierConfig::new(
        p["fine_time_steps"].as_u64().unwrap() as usize,
        p["fine_integration_intervals"].as_u64().unwrap() as usize,
        num(p, "fine_cutoff"),
    )
    .unwrap();
    let options = LeastSquaresOptions {
        max_iterations: 60,
        max_evaluations: 100,
        residual_tolerance: 2e-6,
        ..LeastSquaresOptions::default()
    };
    for family in ["heston", "lift", "rough", "ssvi"] {
        let n = if ["rough", "ssvi"].contains(&family) {
            6
        } else {
            5
        };
        let family_model = if family == "ssvi" { "rough" } else { family };
        let quotes = if ["heston", "lift"].contains(&family) {
            f["markov_quotes"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| r["family"] == family)
                .map(|row| HestonIvCalibrationQuote {
                    is_call: true,
                    ..q(
                        num(row, "maturity"),
                        num(row, "strike"),
                        num(row, "target_volatility"),
                    )
                })
                .collect::<Vec<_>>()
        } else {
            let s = StandardSsvi::new(
                ThetaPchip::new(vec![0.25, 0.75, 1.5], vec![0.01, 0.03, 0.06], 0.04).unwrap(),
                -0.5,
                PhiSpec::PowerLaw {
                    eta: 0.35,
                    gamma: 0.5,
                },
                SurfaceValidationTolerance::local_vol_vegakt_v1(),
            )
            .unwrap();
            let mut qs = Vec::new();
            for t in [0.25, 0.75, 1.5] {
                for k in [85., 100., 115.] {
                    qs.push(if family == "ssvi" {
                        HestonIvCalibrationQuote::from_ssvi(&s, t, 100., k, 0.97, 1.).unwrap()
                    } else {
                        q(t, k, 0.2)
                    });
                }
            }
            if family == "rough" {
                let e = HestonIvCalibrationProblem::new(model(P, "rough"), qs.clone(), vars(6), c)
                    .unwrap()
                    .evaluate(&P)
                    .unwrap();
                for (q, iv) in qs.iter_mut().zip(e.model_implied_volatilities) {
                    q.target_volatility = iv;
                }
            }
            qs
        };
        for (start, start_json) in p["starts"].as_array().unwrap().iter().enumerate() {
            let mut pp = [0.; 6];
            for (x, value) in pp.iter_mut().zip(start_json.as_array().unwrap()) {
                *x = value.as_f64().unwrap();
            }
            let problem = HestonIvCalibrationProblem::new(
                model(pp, family_model),
                quotes.clone(),
                vars(n),
                c,
            )
            .unwrap();
            let initial = problem.evaluate(&problem.initial_parameters()).unwrap();
            let obj0 = initial
                .scaled_residuals
                .iter()
                .map(|x| x * x / 2.)
                .sum::<f64>();
            let r = problem.calibrate(options).unwrap();
            let max_iv = r
                .evaluation
                .scaled_residuals
                .iter()
                .map(|x| x.abs())
                .fold(0., f64::max);
            let check =
                HestonIvCalibrationProblem::new(r.model.clone(), quotes.clone(), vars(n), fine)
                    .unwrap()
                    .evaluate(&r.optimizer.parameters)
                    .unwrap();
            let max_fine = check
                .scaled_residuals
                .iter()
                .map(|x| x.abs())
                .fold(0., f64::max);
            println!(
                "IV_FIT family={family} start={start} fit={} termination={:?} evaluations={} objective0={obj0:0.12e} objective={:0.12e} max_iv={max_iv:0.12e} fine_max_iv={max_fine:0.12e} parameters={:?}",
                r.fit_achieved,
                r.optimizer.termination,
                r.evaluations,
                r.optimizer.objective,
                r.optimizer.parameters
            );
            if family == "ssvi" {
                assert!(r.optimizer.objective <= obj0 * num(p, "ssvi_objective_ratio"));
                assert!(max_fine <= num(p, "ssvi_max_iv_residual"));
            } else {
                assert!(r.fit_achieved);
                assert!(max_fine <= num(p, "fine_iv_budget"));
            }
            assert_eq!(r.fit_achieved, max_iv <= options.residual_tolerance);
            for (i, (a, b)) in r
                .evaluation
                .model_implied_volatilities
                .iter()
                .zip(&check.model_implied_volatilities)
                .enumerate()
            {
                println!(
                    "IV_REPRICE family={family} start={start} row={i} target={:0.12e} fitted={a:0.12e} fine={b:0.12e}",
                    quotes[i].target_volatility
                );
            }
        }
    }
}
