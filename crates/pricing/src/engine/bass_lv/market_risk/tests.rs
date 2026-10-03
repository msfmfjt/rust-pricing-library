use super::*;

fn model(grid_points: usize) -> BassMarketIvModel {
    BassMarketIvModel::calibrate(
        100.0,
        MarketIvSurface::new(vec![0.5, 1.0], vec![-2.0, 0.0, 2.0], vec![0.2; 6]).unwrap(),
        (0..=800).map(|i| -2.0 + 4.0 * i as f64 / 800.0).collect(),
        BassLvConfig {
            grid_points,
            cdf_tolerance: 1e-7,
            ..Default::default()
        },
        BassSurfaceProjectionConfig::default(),
    )
    .unwrap()
}

#[test]
fn quote_risk_matches_independent_recalibration_and_paired_statistics() {
    let m = model(401);
    let h = 1e-4;
    let r = m.compile_vega_kt(vec![0.25, 0.75, 1.0], h).unwrap();
    let a = r.price_asian(100.0, true, 700, 18, 0.97).unwrap();
    let base = r
        .simulation
        .price_asian(100.0, true, 700, 18, 0.97)
        .unwrap();
    assert_eq!(a.estimate.price, base.price);
    assert_eq!(a.estimate.standard_error, base.standard_error);
    assert_eq!(a.method(), BASS_VEGA_KT_METHOD);
    assert_eq!(a.maturity_nodes, vec![0.5, 1.0]);
    assert_eq!(a.log_moneyness_nodes, vec![-2.0, 0.0, 2.0]);
    assert_eq!(r.diagnostics.len(), 14);
    assert_eq!(r.diagnostics[12].quote_index, None);
    assert_eq!(r.diagnostics[1].shift, -h);
    assert!(
        r.diagnostics
            .iter()
            .flat_map(|d| &d.calibration)
            .all(|d| d.cdf_residual <= 1e-7)
    );
    let mut path_sums = vec![0.0; 700];
    for j in 0..6 {
        let mut shifts = vec![0.0; 6];
        shifts[j] = h;
        let up = m
            .bumped(&shifts)
            .unwrap()
            .model
            .compile_simulation(vec![0.25, 0.75, 1.0])
            .unwrap()
            .sample_paths(700, 18)
            .unwrap();
        shifts[j] = -h;
        let down = m
            .bumped(&shifts)
            .unwrap()
            .model
            .compile_simulation(vec![0.25, 0.75, 1.0])
            .unwrap()
            .sample_paths(700, 18)
            .unwrap();
        let pay = |s: &Vec<f64>| 0.97 * (s.iter().sum::<f64>() / 3.0 - 100.0).max(0.0);
        let differences: Vec<_> = up
            .iter()
            .zip(&down)
            .map(|(u, d)| (pay(u) - pay(d)) / (2.0 * h))
            .collect();
        for (s, v) in path_sums.iter_mut().zip(&differences) {
            *s += v;
        }
        let (mean, se) = stats(&differences);
        assert!((a.sensitivities[j] - mean).abs() < 1e-10);
        assert!((a.standard_errors[j] - se).abs() < 1e-10);
        assert_eq!(a.vega_per_vol_point()[j], 0.01 * a.sensitivities[j]);
        assert_eq!(
            a.standard_errors_per_vol_point()[j],
            0.01 * a.standard_errors[j]
        );
    }
    let (sum, se) = stats(&path_sums);
    assert!((a.bucket_sum - sum).abs() < 1e-10);
    assert!((a.bucket_sum_standard_error - se).abs() < 1e-10);
    assert!((a.bucket_sum - a.sensitivities.iter().sum::<f64>()).abs() < 1e-10);
    let discounted = r.price_asian(100.0, true, 700, 18, 1.0).unwrap();
    assert!((a.parallel_sensitivity - 0.97 * discounted.parallel_sensitivity).abs() < 1e-9);
    let replay = r.price_asian(100.0, true, 700, 18, 0.97).unwrap();
    assert_eq!(a.sensitivities, replay.sensitivities);
    assert_eq!(a.parallel_standard_error, replay.parallel_standard_error);
}
fn stats(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let se = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n * (n - 1.0))).sqrt();
    (mean, se)
}

#[test]
fn skew_smile_terminal_quote_vega_matches_market_call_oracle() {
    let m = BassMarketIvModel::calibrate(
        100.0,
        MarketIvSurface::new(
            vec![0.5, 1.0],
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![0.24, 0.22, 0.2, 0.18, 0.16, 0.24, 0.22, 0.2, 0.18, 0.16],
        )
        .unwrap(),
        (0..=1600).map(|i| -2.0 + 4.0 * i as f64 / 1600.0).collect(),
        BassLvConfig {
            grid_points: 801,
            cdf_tolerance: 1e-7,
            ..Default::default()
        },
        BassSurfaceProjectionConfig::default(),
    )
    .unwrap();
    let a = m
        .compile_vega_kt(vec![1.0], 1e-4)
        .unwrap()
        .price_european(100.0, true, 60_000, 107, 1.0)
        .unwrap();
    // At a source quote strike/maturity, Black's call uses that single IV.
    // The Bass dynamics and all intermediate recalibrations must recover it.
    for j in 0..10 {
        let expected = if j == 7 { 100.0 * pdf(0.1) } else { 0.0 };
        assert!(
            (a.sensitivities[j] - expected).abs() < 5.0 * a.standard_errors[j] + 0.06,
            "j={j}: {} +/- {}, expected {expected}",
            a.sensitivities[j],
            a.standard_errors[j]
        );
    }
}

#[test]
fn black_scholes_vega_bucket_sum_parallel_bump_and_causality() {
    let m = model(801);
    let r = m.compile_vega_kt(vec![1.0], 1e-4).unwrap();
    let a = r.price_european(100.0, true, 60_000, 948, 1.0).unwrap();
    let reference = 100.0 * pdf(0.1);
    eprintln!(
        "BS vega {} +/- {}, reference {reference}, sum {} +/- {}",
        a.parallel_sensitivity,
        a.parallel_standard_error,
        a.bucket_sum,
        a.bucket_sum_standard_error
    );
    assert!((a.parallel_sensitivity - reference).abs() < 5.0 * a.parallel_standard_error + 0.04);
    assert!((a.bucket_sum - a.parallel_sensitivity).abs() < 0.04);
    // Earlier-expiry vanilla prices depend on earlier marginals only.
    let early = m
        .compile_vega_kt(vec![0.5], 1e-4)
        .unwrap()
        .price_european(100.0, true, 1000, 3, 1.0)
        .unwrap();
    assert_eq!(&early.sensitivities[3..], &[0.0; 3]);
    assert_eq!(&early.standard_errors[3..], &[0.0; 3]);
    let linear = r.price(60_000, 948, 1.0, |s| s[0]).unwrap();
    assert!(linear.parallel_sensitivity.abs() < 5.0 * linear.parallel_standard_error + 0.04);
    let put = r.price_european(100.0, false, 60_000, 948, 1.0).unwrap();
    assert!(
        (a.parallel_sensitivity - put.parallel_sensitivity - linear.parallel_sensitivity).abs()
            < 1e-8
    );
}

#[test]
fn bump_and_grid_refinement_and_invalid_scenarios() {
    let m = model(401);
    let mut values = Vec::new();
    for h in [2e-4, 1e-4, 5e-5] {
        let a = m
            .compile_vega_kt(vec![0.25, 0.75, 1.0], h)
            .unwrap()
            .price_asian(100.0, true, 3000, 42, 1.0)
            .unwrap();
        eprintln!(
            "h={h}: {:?}, parallel {}",
            a.sensitivities, a.parallel_sensitivity
        );
        values.push(a);
    }
    for j in 0..6 {
        assert!((values[1].sensitivities[j] - values[2].sensitivities[j]).abs() < 0.02);
    }
    let fine = model(801)
        .compile_vega_kt(vec![0.25, 0.75, 1.0], 1e-4)
        .unwrap()
        .price_asian(100.0, true, 3000, 42, 1.0)
        .unwrap();
    for j in 0..6 {
        assert!((values[1].sensitivities[j] - fine.sensitivities[j]).abs() < 0.08);
    }
    for h in [0.0, -1e-4, f64::NAN, 0.2, 1e-30] {
        assert!(m.compile_vega_kt(vec![1.0], h).is_err());
    }
    assert!(m.compile_vega_kt(vec![1.1], 1e-4).is_err());
    assert!(m.bumped(&[0.0]).is_err());
    assert!(m.bumped(&[f64::NAN; 6]).is_err());
    // A large positive node bump can violate calendar order/density. Do not
    // silently reduce the step or return a partial risk vector.
    let e = m.compile_vega_kt(vec![1.0], 0.15).unwrap_err().to_string();
    assert!(e.contains("quote[") && e.contains("shift"), "{e}");
    let r = m.compile_vega_kt(vec![0.0, 0.5], 1e-4).unwrap();
    assert!(r.price_asian(100.0, true, 1, 0, 1.0).is_err());
    assert!(r.price_european(f64::NAN, true, 2, 0, 1.0).is_err());
    assert!(r.price_european(100.0, true, 2, 0, 0.0).is_err());
    assert!(r.price(2, 0, 1.0, |_| f64::NAN).is_err());
    let fixed = r.price(10, 0, 1.0, |s| s[0]).unwrap();
    assert_eq!(fixed.sensitivities, vec![0.0; 6]);
}
