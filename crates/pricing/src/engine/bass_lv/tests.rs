use super::*;

fn marginal(t: f64, vol: f64) -> BassMarginal {
    BassMarginal::lognormal(t, 100.0, vol, 1601, 7.0).unwrap()
}
fn model() -> BassLvModel {
    BassLvModel::calibrate(
        100.0,
        vec![marginal(0.5, 0.2), marginal(1.0, 0.2), marginal(2.0, 0.2)],
        BassLvConfig::default(),
    )
    .unwrap()
}

#[test]
fn gaussian_convolution_matches_closed_form_and_preserves_constants() {
    let x: Vec<_> = (0..401).map(|j| -10.0 + j as f64 * 0.05).collect();
    let constant = Table {
        x: x.clone(),
        y: vec![3.0; x.len()],
    };
    for w in [-30.0, -9.0, 0.0, 8.0, 30.0] {
        let (v, d) = constant.heat_value_slope(w, 0.7);
        assert!((v - 3.0).abs() < 2e-13, "{w} {v}");
        assert_eq!(d, 0.0);
    }
    let ramp = Table {
        x: x.clone(),
        y: x.iter().map(|w| w.max(0.0)).collect(),
    };
    let heated = ramp.heated(0.7);
    for w in [-2.0, 0.0, 1.5] {
        let sd = 0.7_f64.sqrt();
        let expected = w * cdf(w / sd) + sd * pdf(w / sd);
        assert!((ramp.heat_value_slope(w, 0.7).0 - expected).abs() < 1e-12);
        assert!((heated.at(w) - expected).abs() < 1e-12);
    }
}

#[test]
fn tabulated_distribution_has_exact_moments_calls_and_inverse() {
    let m = BassMarginal::new(1.0, vec![50.0, 100.0, 150.0], vec![0.0, 0.5, 1.0]).unwrap();
    assert_eq!(m.mean(), 100.0);
    assert!((m.variance() - 10000.0 / 12.0).abs() < 1e-10);
    assert_eq!(m.call_price(100.0).unwrap(), 12.5);
    assert_eq!(m.call_price(0.0).unwrap(), 100.0);
    for p in [0.0, 0.1, 0.8, 1.0] {
        assert!((m.cdf(m.quantile(p).unwrap()) - p).abs() < 1e-14);
    }
    assert!(m.quantile(f64::NAN).is_err());
}

#[test]
fn convex_order_checks_between_knots_even_when_variance_increases() {
    let spots = vec![50.0, 70.0, 90.0, 110.0, 130.0, 150.0];
    let a = BassMarginal::new(1.0, spots.clone(), vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).unwrap();
    let b =
        BassMarginal::new(2.0, spots.clone(), vec![0.0, 0.225, 0.35, 0.65, 0.775, 1.0]).unwrap();
    assert!((a.mean() - b.mean()).abs() < 1e-12);
    assert!(b.variance() > a.variance());
    for strike in spots {
        assert!(b.call_price(strike).unwrap() - a.call_price(strike).unwrap() > -1e-12);
    }
    assert!(matches!(
        BassLvModel::calibrate(100.0, vec![a, b], BassLvConfig::default()),
        Err(BassError::ConvexOrder { strike, .. }) if (strike - 100.0).abs() < 1e-10
    ));
}

#[test]
fn black_scholes_mapping_volatility_and_boundary_continuity() {
    let m = model();
    eprintln!(
        "diagnostics {:?}; initial error {}",
        m.diagnostics(),
        m.initial_spot_error()
    );
    for t in [0.1, 0.4, 0.5, 0.8, 1.0, 1.7] {
        for s in [80.0, 100.0, 120.0] {
            let vol = m.local_volatility(t, s).unwrap();
            assert!((vol - 0.2).abs() < 0.0015, "t={t} s={s} vol={vol}");
        }
    }
    for pair in m.intervals.windows(2) {
        for w in [-0.7, 0.0, 0.9] {
            let s = pair[0].terminal.at(w);
            let reset = pair[1].initial.inverse(s).unwrap();
            assert!((pair[1].initial.at(reset) - s).abs() < 1e-12);
        }
    }
    let plan = m.compile_simulation(vec![0.0, 0.25, 0.75, 2.0]).unwrap();
    assert_eq!(plan.time_nodes(), &[0.0, 0.25, 0.5, 0.75, 1.0, 2.0]);
    let z = [0.2, -0.4, 0.7, -0.3, 0.1];
    let path = plan.path_from_normals(&z).unwrap();
    let mut b = 0.0;
    let mut expected = vec![100.0];
    for (j, t) in plan.time_nodes().windows(2).enumerate() {
        b += (t[1] - t[0]).sqrt() * z[j];
        if [0.25, 0.75, 2.0].contains(&t[1]) {
            expected.push(100.0 * (0.2 * b - 0.02 * t[1]).exp());
        }
    }
    for (s, e) in path.iter().zip(expected) {
        assert!((s - e).abs() < 0.03, "{s} {e}");
    }
}

#[test]
fn monte_carlo_recovers_marginals_and_martingale() {
    let m = model();
    let plan = m.compile_simulation(vec![0.5, 1.0, 2.0]).unwrap();
    let paths = plan.sample_paths(60_000, 73021).unwrap();
    for (i, margin) in m.marginals().iter().enumerate() {
        let mean = paths.iter().map(|s| s[i]).sum::<f64>() / paths.len() as f64;
        assert!(
            (mean - 100.0).abs() < 5.0 * (margin.variance() / paths.len() as f64).sqrt() + 0.02
        );
        for k in [80.0, 100.0, 120.0] {
            let values: Vec<_> = paths.iter().map(|s| (s[i] - k).max(0.0)).collect();
            let price = values.iter().sum::<f64>() / values.len() as f64;
            let se = (values.iter().map(|v| (v - price).powi(2)).sum::<f64>()
                / ((values.len() - 1) * values.len()) as f64)
                .sqrt();
            let target = margin.call_price(k).unwrap();
            assert!(
                (price - target).abs() < 5.0 * se + 0.02,
                "t={} k={k} price={price} target={target} se={se}",
                margin.expiry()
            );
        }
    }
    let call = plan.price_european(100.0, true, 5000, 8, 0.97).unwrap();
    let again = plan.price_european(100.0, true, 5000, 8, 0.97).unwrap();
    assert_eq!(call.price, again.price);
    assert_eq!(call.standard_error, again.standard_error);
}

fn uniform(t: f64, width: f64) -> BassMarginal {
    BassMarginal::new(
        t,
        vec![100.0 - width, 100.0, 100.0 + width],
        vec![0.0, 0.5, 1.0],
    )
    .unwrap()
}

#[test]
fn non_lognormal_marginals_are_fitted() {
    // A skewed density, followed by a mean-preserving dilation.
    let a = BassMarginal::new(
        0.5,
        vec![70.0, 90.0, 110.0, 150.0],
        vec![0.0, 0.25, 0.75, 1.0],
    )
    .unwrap();
    let spot = a.mean();
    let b = BassMarginal::new(
        1.5,
        a.spots().iter().map(|s| spot + 1.5 * (s - spot)).collect(),
        a.probabilities().to_vec(),
    )
    .unwrap();
    let m = BassLvModel::calibrate(
        spot,
        vec![a, b],
        BassLvConfig {
            cdf_tolerance: 3e-6,
            ..Default::default()
        },
    )
    .unwrap();
    let p = m.compile_simulation(vec![0.5, 1.5]).unwrap();
    for (i, margin) in m.marginals().iter().enumerate() {
        for k in [85.0, 100.0, 120.0] {
            let e = p.price(40_000, 191, 1.0, |s| (s[i] - k).max(0.0)).unwrap();
            assert!(
                (e.price - margin.call_price(k).unwrap()).abs() < 5.0 * e.standard_error + 0.025
            );
        }
    }
}

#[test]
fn rejects_invalid_inputs_and_unconverged_calibration() {
    assert!(BassMarginal::new(1.0, vec![1.0, 2.0, 3.0], vec![0.0, 0.8, 0.7]).is_err());
    assert!(
        BassLvModel::calibrate(101.0, vec![uniform(1.0, 20.0)], BassLvConfig::default()).is_err()
    );
    assert!(matches!(
        BassLvModel::calibrate(
            100.0,
            vec![uniform(1.0, 30.0), uniform(2.0, 20.0)],
            BassLvConfig::default()
        ),
        Err(BassError::ConvexOrder { .. })
    ));
    assert!(
        BassLvModel::calibrate(
            100.0,
            vec![uniform(1.0, 20.0), uniform(2.0, 20.0)],
            BassLvConfig::default()
        )
        .is_err()
    );
    let result = BassLvModel::calibrate(
        100.0,
        vec![uniform(1.0, 20.0), uniform(2.0, 40.0)],
        BassLvConfig {
            max_iterations: 1,
            cdf_tolerance: 1e-12,
            ..Default::default()
        },
    );
    assert!(
        matches!(result, Err(BassError::Calibration { .. })),
        "{result:?}"
    );
    let m =
        BassLvModel::calibrate(100.0, vec![marginal(1.0, 0.2)], BassLvConfig::default()).unwrap();
    assert!(m.compile_simulation(vec![0.5, 0.5]).is_err());
    assert!(m.compile_simulation(vec![1.1]).is_err());
    let p = m.compile_simulation(vec![1.0]).unwrap();
    assert!(p.path_from_normals(&[]).is_err());
    assert!(p.path_from_normals(&[f64::NAN]).is_err());
    assert!(p.path_from_normals(&[100.0]).is_err());
    assert!(p.price_european(100.0, true, 1, 0, 1.0).is_err());
    assert!(p.price(10, 0, 1.0, |_| f64::NAN).is_err());
    assert!(matches!(
        BassLvModel::calibrate(
            100.0,
            vec![marginal(1.0, 0.2)],
            BassLvConfig {
                grid_width: 4.0,
                ..Default::default()
            }
        ),
        Err(BassError::GridTooNarrow { .. })
    ));
}

fn propagated_cdf_error(model: &BassLvModel) -> f64 {
    let mut previous: Option<Table> = None;
    let mut maximum: f64 = 0.0;
    for (i, interval) in model.intervals.iter().enumerate() {
        let x = &interval.terminal.x;
        let end_cdf = if let Some(prev) = previous {
            let old = &model.intervals[i - 1].terminal;
            let start_cdf = Table {
                x: x.clone(),
                y: interval
                    .initial
                    .y
                    .iter()
                    .map(|s| {
                        if *s <= old.y[0] {
                            0.0
                        } else if *s >= *old.y.last().unwrap() {
                            1.0
                        } else {
                            prev.at(old.inverse(*s).unwrap())
                        }
                    })
                    .collect(),
            };
            start_cdf.heated(interval.end - model.intervals[i - 1].end)
        } else {
            Table {
                x: x.clone(),
                y: x.iter()
                    .map(|w| cdf((w - model.initial_w) / interval.end.sqrt()))
                    .collect(),
            }
        };
        for p in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let s = model.marginals[i].quantile(p).unwrap();
            let actual = end_cdf.at(interval.terminal.inverse(s).unwrap());
            maximum = maximum.max((actual - p).abs());
        }
        previous = Some(end_cdf);
    }
    maximum
}

#[test]
fn deterministic_distribution_propagation_refines_to_target() {
    let a = BassMarginal::new(
        0.5,
        vec![70.0, 90.0, 110.0, 150.0],
        vec![0.0, 0.25, 0.75, 1.0],
    )
    .unwrap();
    let spot = a.mean();
    let b = BassMarginal::new(
        1.5,
        a.spots().iter().map(|s| spot + 1.5 * (s - spot)).collect(),
        a.probabilities().to_vec(),
    )
    .unwrap();
    let mut errors = Vec::new();
    for grid_points in [401, 801, 1601] {
        let m = BassLvModel::calibrate(
            spot,
            vec![a.clone(), b.clone()],
            BassLvConfig {
                grid_points,
                cdf_tolerance: 3e-6,
                ..Default::default()
            },
        )
        .unwrap();
        errors.push(propagated_cdf_error(&m));
    }
    eprintln!("skewed marginal propagated CDF errors (401/801/1601): {errors:?}");
    // Quantile-map kinks at the sparse density jumps give first-order CDF
    // convergence. Require < 0.1 percentage point and > 2x refinement gain.
    assert!(errors[2] < 1e-3);
    assert!(errors[2] < errors[0] * 0.5);
}

#[test]
fn heat_mapping_satisfies_conditional_martingale() {
    let m = model();
    for interval in &m.intervals {
        let dt = interval.end * 0.15;
        let remainder = interval.end * 0.1;
        for w in [-0.5, 0.0, 0.7] {
            let conditional: f64 = (-500..=500)
                .map(|j| {
                    let z = j as f64 * 0.02;
                    interval
                        .terminal
                        .heat_value_slope(w + dt.sqrt() * z, remainder)
                        .0
                        * pdf(z)
                        * 0.02
                })
                .sum();
            let initial = interval.terminal.heat_value_slope(w, dt + remainder).0;
            assert!(
                (conditional - initial).abs() < 1e-9,
                "{conditional} {initial}"
            );
        }
    }
}
