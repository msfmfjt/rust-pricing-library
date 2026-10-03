use super::*;
use crate::bass_lv::{BassMappingBump, BassSurfaceProjectionConfig};
use crate::market::{EssviSlice, EssviSurface, ImpliedVarianceSurface, SurfaceValidationTolerance};

fn surface() -> EssviSurface {
    EssviSurface::new(
        vec![
            EssviSlice::new(0.5, 0.02, 0.06, -0.025).unwrap(),
            EssviSlice::new(1.0, 0.04, 0.09, -0.035).unwrap(),
            EssviSlice::new(2.0, 0.08, 0.13, -0.045).unwrap(),
        ],
        0.04,
        SurfaceValidationTolerance::local_vol_vegakt_v1(),
    )
    .unwrap()
}
fn model() -> BassLvModel {
    BassLvModel::calibrate(
        100.0,
        [0.5, 1.0, 2.0]
            .iter()
            .map(|t| BassMarginal::lognormal(*t, 100.0, 0.2, 1601, 7.0).unwrap())
            .collect(),
        BassLvConfig::default(),
    )
    .unwrap()
}
fn risk(m: &BassLvModel) -> BassMappingRiskPlan {
    m.compile_mapping_risk(
        vec![0.0, 0.3, 0.5, 0.8, 1.4, 2.0],
        (0..3)
            .flat_map(|i| {
                [-0.4, 0.6].map(|c| BassMappingBump::new(i, c - 0.8, c, c + 0.8).unwrap())
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn surface_projection_uses_skew_derivative_and_preserves_calls() {
    let s = surface();
    let nodes: Vec<_> = (0..1601).map(|j| -4.0 + 8.0 * j as f64 / 1600.0).collect();
    let mut marginals = Vec::new();
    for t in [0.5, 1.0, 2.0] {
        let p = BassMarginal::from_surface(
            t,
            100.0,
            &s,
            &nodes,
            BassSurfaceProjectionConfig::default(),
        )
        .unwrap();
        eprintln!("surface t={t} diagnostics {:?}", p.diagnostics);
        assert!((p.marginal.mean() - 100.0).abs() < 1e-10);
        assert!(p.diagnostics.max_call_price_error < 0.002);
        for k in [-0.3_f64, 0.0, 0.2] {
            let h = 1e-4;
            let strike = 100.0 * k.exp();
            let up = s
                .forward_call_evaluation(t, ((strike + h) / 100.0).ln(), 100.0)
                .unwrap()
                .undiscounted_price;
            let dn = s
                .forward_call_evaluation(t, ((strike - h) / 100.0).ln(), 100.0)
                .unwrap()
                .undiscounted_price;
            let expected = 1.0 + (up - dn) / (2.0 * h);
            assert!((p.marginal.cdf(strike) - expected).abs() < 3e-5);
        }
        marginals.push(p.marginal);
    }
    let m = BassLvModel::calibrate(100.0, marginals, BassLvConfig::default()).unwrap();
    let p = m
        .compile_simulation(vec![0.5, 1.0, 2.0])
        .unwrap()
        .price_european(100.0, true, 40_000, 919, 1.0)
        .unwrap();
    let reference = s
        .forward_call_evaluation(2.0, 0.0, 100.0)
        .unwrap()
        .undiscounted_price;
    assert!((p.price - reference).abs() < 5.0 * p.standard_error + 0.01);
}

#[test]
fn surface_projection_refines_and_rejects_truncation() {
    let s = surface();
    let mut errors = Vec::new();
    for n in [401, 801, 1601] {
        let nodes: Vec<_> = (0..n)
            .map(|j| -4.0 + 8.0 * j as f64 / (n - 1) as f64)
            .collect();
        let p = BassMarginal::from_surface(
            1.0,
            100.0,
            &s,
            &nodes,
            BassSurfaceProjectionConfig {
                relative_mean_tolerance: 1e-3,
                ..Default::default()
            },
        )
        .unwrap();
        errors.push(p.diagnostics.max_call_price_error);
    }
    eprintln!("surface projection call errors {errors:?}");
    assert!(errors[2] < errors[0] * 0.2);
    let narrow: Vec<_> = (0..101).map(|j| -0.2 + 0.4 * j as f64 / 100.0).collect();
    assert!(BassMarginal::from_surface(1.0, 100.0, &s, &narrow, Default::default()).is_err());
}

#[test]
fn path_reverse_matches_fixed_spot_map_bumps_across_resets() {
    let m = model();
    let r = risk(&m);
    let z: Vec<_> = (0..r.simulation().normal_count())
        .map(|i| 0.3 * (i as f64 - 2.0))
        .collect();
    let partials = [0.7, 0.2, -0.4, 0.6, 1.1, -0.1];
    let actual = r.path_sensitivities(&z, &partials).unwrap();
    for epsilon in [1e-4, 3e-5] {
        for j in 0..actual.len() {
            let mut b = vec![0.0; actual.len()];
            b[j] = epsilon;
            let up = r
                .bumped_simulation(&b)
                .unwrap()
                .path_from_normals(&z)
                .unwrap();
            b[j] = -epsilon;
            let dn = r
                .bumped_simulation(&b)
                .unwrap()
                .path_from_normals(&z)
                .unwrap();
            let fd: f64 = up
                .iter()
                .zip(dn)
                .zip(partials)
                .map(|((u, d), p)| p * (u - d) / (2.0 * epsilon))
                .sum();
            assert!(
                (actual[j] - fd).abs() < 2e-7,
                "j={j} aad={} fd={fd}",
                actual[j]
            );
            assert_eq!(up[0], 100.0);
        }
    }
}

#[test]
fn mapping_risk_mc_has_price_parity_and_matches_crn() {
    let m = model();
    let r = risk(&m);
    let n = 12_000;
    let seed = 271;
    for asian in [false, true] {
        let aad = if asian {
            r.price_asian(100.0, true, n, seed, 0.97)
        } else {
            r.price_european(100.0, true, n, seed, 0.97)
        }
        .unwrap();
        let base = if asian {
            r.simulation().price_asian(100.0, true, n, seed, 0.97)
        } else {
            r.simulation().price_european(100.0, true, n, seed, 0.97)
        }
        .unwrap();
        assert_eq!(base.price, aad.estimate.price);
        assert_eq!(base.standard_error, aad.estimate.standard_error);
        for j in 0..aad.sensitivities.len() {
            let eps = 1e-4;
            let mut b = vec![0.0; aad.sensitivities.len()];
            b[j] = eps;
            let up = r.bumped_simulation(&b).unwrap();
            b[j] = -eps;
            let dn = r.bumped_simulation(&b).unwrap();
            let up = if asian {
                up.price_asian(100.0, true, n, seed, 0.97)
            } else {
                up.price_european(100.0, true, n, seed, 0.97)
            }
            .unwrap();
            let dn = if asian {
                dn.price_asian(100.0, true, n, seed, 0.97)
            } else {
                dn.price_european(100.0, true, n, seed, 0.97)
            }
            .unwrap();
            let fd = (up.price - dn.price) / (2.0 * eps);
            assert!(
                (aad.sensitivities[j] - fd).abs() < 3e-5,
                "asian={asian} j={j} aad={} fd={fd}",
                aad.sensitivities[j]
            );
            assert!(aad.standard_errors[j].is_finite() && aad.standard_errors[j] >= 0.0);
        }
    }
}

#[test]
fn deterministic_vanilla_risk_matches_bumps_and_monte_carlo() {
    let m = model();
    let r = risk(&m);
    for expiry in [0, 1, 2] {
        let a = r.vanilla_call(expiry, 100.0, 0.98).unwrap();
        let reference = m.marginals()[expiry].call_price(100.0).unwrap() * 0.98;
        assert!(
            (a.price - reference).abs() < 0.005,
            "{} vs {reference}",
            a.price
        );
        for j in 0..r.bumps().len() {
            let eps = 1e-5;
            let mut b = vec![0.0; r.bumps().len()];
            b[j] = eps;
            let up = r.bumped_vanilla_call(&b, expiry, 100.0, 0.98).unwrap();
            b[j] = -eps;
            let dn = r.bumped_vanilla_call(&b, expiry, 100.0, 0.98).unwrap();
            let fd = (up - dn) / (2.0 * eps);
            assert!(
                (a.sensitivities[j] - fd).abs() < 3e-6,
                "expiry={expiry} j={j} aad={} fd={fd}",
                a.sensitivities[j]
            );
            if r.bumps()[j].interval() > expiry {
                assert_eq!(a.sensitivities[j], 0.0);
            }
        }
    }
    let a = r.vanilla_call(2, 100.0, 1.0).unwrap();
    let mc = r.price_european(100.0, true, 80_000, 763, 1.0).unwrap();
    for (j, v) in a.sensitivities.iter().enumerate() {
        assert!((v - mc.sensitivities[j]).abs() < 5.0 * mc.standard_errors[j] + 0.002);
    }
    let linear = r
        .price(80_000, 763, 1.0, |s, d| {
            d[s.len() - 1] = 1.0;
            s[s.len() - 1]
        })
        .unwrap();
    for (j, v) in linear.sensitivities.iter().enumerate() {
        assert!(v.abs() < 5.0 * linear.standard_errors[j] + 0.001);
    }
}

#[test]
fn mapping_risk_rejects_invalid_basis_amplitudes_and_partials() {
    assert!(BassMappingBump::new(0, 0.0, 0.0, 1.0).is_err());
    let m = model();
    assert!(m.compile_mapping_risk(vec![1.0], vec![]).is_err());
    for bump in [
        BassMappingBump::new(0, -100.0, 0.0, 100.0).unwrap(),
        BassMappingBump::new(0, 0.003, 0.004, 0.005).unwrap(),
    ] {
        assert!(m.compile_mapping_risk(vec![1.0], vec![bump]).is_err());
    }
    assert!(
        m.compile_mapping_risk(
            vec![1.0],
            vec![BassMappingBump::new(5, -1.0, 0.0, 1.0).unwrap()]
        )
        .is_err()
    );
    let r = risk(&m);
    assert!(r.bumped_simulation(&[]).is_err());
    assert!(r.bumped_simulation(&vec![1000.0; r.bumps().len()]).is_err());
    assert!(
        r.path_sensitivities(&vec![0.0; r.simulation().normal_count()], &[1.0])
            .is_err()
    );
    assert!(r.vanilla_call(3, 100.0, 1.0).is_err());
    assert!(r.price_european(100.0, true, 1, 1, 1.0).is_err());
    assert!(
        r.price(2, 1, 1.0, |_, d| {
            d[0] = f64::NAN;
            0.0
        })
        .is_err()
    );
}
