use super::*;
use crate::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

#[test]
fn mixed_total_scramble_errors_preserve_direct_calibration_covariance() {
    mixed_scramble_covariance(false);
}
#[test]
fn mixed_hurst_total_scramble_errors_preserve_direct_calibration_covariance() {
    mixed_scramble_covariance(true);
}
fn mixed_scramble_covariance(include_hurst: bool) {
    let mut v: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/v1/pricing_request.golden.json"
    )))
    .unwrap();
    v["model"] = json!({"type":"local_volatility","local_variance_grid":{
        "time_nodes":[0.,0.2,0.5,1.],"log_forward_moneyness_nodes":[-0.4,-0.1,0.2,0.4],
        "values":(0..16).map(|i|0.04+0.001*i as f64).collect::<Vec<_>>(),"shape":[4,4],"floor":1e-8,"cap":4.0}});
    v["engine"] = json!({"type":"randomized_quasi_monte_carlo","points_per_scramble":32,"scramble_count":8,
        "master_scramble_seed":191,"variance_reduction":{"antithetic":true,"brownian_bridge":true}});
    let req = parse_request_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let model = crate::models::MixedRoughBergomi::new(
        0.2,
        -0.6,
        vec![0.35, 0.65],
        vec![0.3, 0.8],
        crate::models::ForwardVarianceCurve::constant(0.04).unwrap(),
    )
    .unwrap();
    let p = RoughFamilyLsvPricingPlan::compile(
        &req,
        model.into(),
        LsvParticleConfig::new(128, 429, 0.5, 3., true).unwrap(),
        ExecutionPolicy::new(1, None).unwrap(),
    )
    .unwrap();
    let risk = if include_hurst {
        p.evaluate_mixed_bergomi_parameter_risk_with_hurst()
    } else {
        p.evaluate_mixed_bergomi_parameter_risk()
    }
    .unwrap();
    let width = risk.parameter_names.len();
    let core = &p.core;
    let EngineConfig::RandomizedQuasiMonteCarlo(c) = core.engine else {
        unreachable!()
    };
    let dim = core.path_plan.random_dimension();
    let qmc = RqmcPlan::compile(c, dim).unwrap();
    let bridge = core.bridge(c.variance_reduction()).unwrap();
    let compile = if include_hurst {
        crate::engine::processes::rough_volatility::MixedBergomiVarianceRiskPlan::compile_with_hurst
    } else {
        crate::engine::processes::rough_volatility::MixedBergomiVarianceRiskPlan::compile
    };
    let driver = compile(
        core.path_plan.model().clone(),
        core.path_plan.times().to_vec(),
    )
    .unwrap();
    let mut total_rows = Vec::new();
    let mut direct_rows = Vec::new();
    let mut cal_rows = Vec::new();
    for scramble in 0..8 {
        let mut direct = vec![0.; width];
        let mut lb = vec![0.; core.calibration.surface().squared_leverage().len()];
        for i in 0..32 {
            let z = (0..dim)
                .map(|d| inverse_standard_normal(qmc.uniform(scramble, i, d).unwrap()).unwrap())
                .collect();
            let z = core.apply_bridge(z, bridge.as_ref()).unwrap();
            for sign in [1., -1.] {
                let z = z.iter().map(|v| sign * v).collect::<Vec<_>>();
                let path = core.path_plan.evolve_path(core.base.spot, &z).unwrap();
                let (_, s) = core
                    .base
                    .lsv_payoff(
                        path.states(),
                        PathIndex::new(u64::from(scramble) * 32 + i),
                        true,
                    )
                    .unwrap();
                let (l, var) = path.reverse_variances(&s.unwrap()).unwrap();
                let a = driver.reverse(&z, &var).unwrap();
                for (x, y) in direct.iter_mut().zip(a) {
                    *x += y / 64.;
                }
                for (x, y) in lb.iter_mut().zip(l.squared_leverage) {
                    *x += y / 64.;
                }
            }
        }
        let cal = if include_hurst {
            core.calibration
                .reverse_mixed_bergomi_parameters_with_hurst(&lb)
        } else {
            core.calibration.reverse_mixed_bergomi_parameters(&lb)
        }
        .unwrap();
        total_rows.push(
            direct
                .iter()
                .zip(&cal)
                .map(|(a, b)| a + b)
                .collect::<Vec<_>>(),
        );
        direct_rows.push(direct);
        cal_rows.push(cal);
    }
    // Independent plain two-pass statistics, not the production reducer.
    let moments = |rows: &Vec<Vec<f64>>, j: usize| {
        let mean = rows.iter().map(|r| r[j]).sum::<f64>() / 8.;
        let se = (rows.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / (8. * 7.)).sqrt();
        (mean, se)
    };
    let mut covariance_effect = 0.0_f64;
    for j in 0..width {
        let (m, se) = moments(&total_rows, j);
        assert!((m - risk.parameter_adjoints[j]).abs() < 1e-10);
        assert!((se - risk.standard_errors.as_ref().unwrap()[j]).abs() < 1e-10);
        let wrong = (moments(&direct_rows, j).1.powi(2) + moments(&cal_rows, j).1.powi(2)).sqrt();
        covariance_effect = covariance_effect.max((wrong - se).abs());
        if include_hurst && j == width - 1 {
            assert!((wrong - se).abs() > 1e-8);
        }
    }
    assert!(
        covariance_effect > 1e-3,
        "must detect incorrect independent-SE combination"
    );
}

#[test]
fn mixed_variance_reverse_does_not_evolve_unused_asset() {
    use crate::engine::processes::rough_volatility::{
        MixedBergomiVarianceRiskPlan, RoughVolatilityPathPlan,
    };
    let m = crate::models::MixedRoughBergomi::new(
        0.2,
        0.,
        vec![1.],
        vec![0.],
        crate::models::ForwardVarianceCurve::constant(0.04).unwrap(),
    )
    .unwrap();
    let p = RoughVolatilityPathPlan::compile(m.clone().into(), vec![0., 1.]).unwrap();
    let z = [5000., 0., 0.];
    assert!(p.evolve_path(100., &z).is_err());
    let r = MixedBergomiVarianceRiskPlan::compile(m.into(), vec![0., 1.]).unwrap();
    assert_eq!(r.reverse(&z, &[0., 1.]).unwrap(), vec![0., 0.]);
}
