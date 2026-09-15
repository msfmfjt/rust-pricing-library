use super::*;
use pricing::models::RoughBergomi;

fn config(h: f64, eta: f64, rho: f64, seed: u64, trace: bool) -> MultiAssetBergomiLsvConfig {
    MultiAssetRoughLsvConfig {
        factor: RoughBergomi::new(h, eta, rho).unwrap(),
        particles: LsvParticleConfig::new(512, seed, 0.8, 2.0, trace).unwrap(),
    }
    .into()
}
fn rough_case() -> Case {
    let mut c = Case::new();
    c.configs = vec![
        Some(config(0.08, 0.5, -0.65, 401, true)),
        Some(config(0.33, 0.6, -0.35, 402, true)),
    ];
    c.rate_correlations = vec![0.2, -0.1, -0.05, 0.03];
    c.full = Some(vec![vec![
        vec![1.0, 0.4, -0.65, 0.1, 0.2],
        vec![0.4, 1.0, -0.15, -0.35, -0.1],
        vec![-0.65, -0.15, 1.0, 0.3, -0.05],
        vec![0.1, -0.35, 0.3, 1.0, 0.03],
        vec![0.2, -0.1, -0.05, 0.03, 1.0],
    ]]);
    c
}
fn quotes(values: Vec<f64>) -> HullWhiteLsvTarget {
    HullWhiteLsvTarget::from_market_iv(
        MarketIvSurface::new(vec![0.5, 1.0], vec![-0.5, 0.0, 0.5], values).unwrap(),
        vec![0.0, 0.5, 1.0],
        vec![-0.35, 0.0, 0.35],
        1e-8,
        4.0,
    )
    .unwrap()
}

#[derive(Clone, Copy)]
struct Kernel {
    brownian: usize,
    power: f64,
    coefficient: f64,
    decay: f64,
    rate: bool,
    bond: bool,
}
fn kernels(c: &Case) -> Vec<Kernel> {
    let plain = |brownian, decay| Kernel {
        brownian,
        decay,
        power: 0.0,
        coefficient: 1.0,
        rate: false,
        bond: false,
    };
    let mut rows = vec![plain(0, 0.0), plain(1, 0.0)];
    let mut rough = Vec::new();
    for cfg in c.configs.iter().flatten() {
        let index = rows.len();
        match cfg {
            MultiAssetBergomiLsvConfig::OneFactor(c) => {
                rows.push(plain(index, c.factor.mean_reversion()))
            }
            MultiAssetBergomiLsvConfig::TwoFactor(c) => {
                for k in c.factor.mean_reversions() {
                    rows.push(plain(rows.len(), k));
                }
            }
            MultiAssetBergomiLsvConfig::Rough(c) => {
                rows.push(plain(index, 0.0));
                rough.push(Kernel {
                    brownian: index,
                    power: c.factor.hurst() - 0.5,
                    coefficient: (2.0 * c.factor.hurst()).sqrt(),
                    decay: 0.0,
                    rate: false,
                    bond: false,
                });
            }
        }
    }
    let index = rows.len();
    rows.push(Kernel {
        rate: true,
        ..plain(index, c.rates.mean_reversion())
    });
    rows.push(Kernel {
        rate: true,
        bond: true,
        ..plain(index, 0.0)
    });
    rows.extend(rough);
    rows
}
fn independent_covariance(c: &Case, a: Kernel, b: Kernel, start: f64, end: f64) -> f64 {
    let power = 1.0 + a.power + b.power;
    let mr = c.rates.mean_reversion();
    let mut total = 0.0;
    for (i, &left) in c.rates.volatility_times().iter().enumerate() {
        let lo = start.max(left);
        let hi = end.min(
            c.rates
                .volatility_times()
                .get(i + 1)
                .copied()
                .unwrap_or(end),
        );
        if hi <= lo {
            continue;
        }
        let sigma = c.rates.volatilities()[i];
        // y=u^power cancels the singular product of power kernels analytically.
        let integral = quad(
            |y| {
                let u = y.powf(1.0 / power);
                let bond = if mr == 0.0 {
                    u
                } else {
                    -(-mr * u).exp_m1() / mr
                };
                (-(a.decay + b.decay) * u).exp()
                    * if a.bond || b.bond {
                        bond.powi(i32::from(a.bond) + i32::from(b.bond))
                    } else {
                        1.0
                    }
            },
            (end - hi).powf(power),
            (end - lo).powf(power),
        );
        total += integral / power
            * a.coefficient
            * b.coefficient
            * sigma.powi(i32::from(a.rate) + i32::from(b.rate));
    }
    total
}

#[test]
fn rough_joint_covariances_match_independent_power_ou_rate_quadrature() {
    for mixed in [false, true] {
        let mut c = rough_case();
        if mixed {
            c.configs[1] = two_factor_configs(0.3)[1].clone();
            c.rate_correlations.push(-0.04);
            c.full = None;
        }
        c.step = 0.3;
        c.targets = vec![Some(quotes(vec![0.28; 6])); 2];
        c.correlation = CorrelationTermStructure::new(
            vec![u(1), u(2)],
            vec![
                (today(), vec![vec![1.0, 0.4], vec![0.4, 1.0]]),
                (d("2026-07-02"), vec![vec![1.0, 0.2], vec![0.2, 1.0]]),
            ],
            tol(),
        )
        .unwrap();
        if let Some(full) = &mut c.full {
            let mut next = full[0].clone();
            next[0][1] = 0.2;
            next[1][0] = 0.2;
            next[2][3] = -0.1;
            next[3][2] = -0.1;
            full.push(next);
        }
        for mean_reversion in [0.0, 0.13, 5.0] {
            c.rates =
                HullWhite1Factor::new(mean_reversion, vec![0.0, 0.37], vec![0.004, 0.006]).unwrap();
            let p = c.compile(1).unwrap();
            assert_eq!(p.random_factor_count(), 8);
            let kernels = kernels(&c);
            for (step, times) in p.time_nodes().windows(2).enumerate() {
                let entry = usize::from(
                    times[0] >= DayCountConvention::Act365F.year_fraction(today(), d("2026-07-02")),
                );
                let brownian = &p.lsv_driver_correlations()[entry];
                let cov = &p.lsv_transition_covariances()[step];
                for (i, &a) in kernels.iter().enumerate() {
                    for (j, &b) in kernels.iter().enumerate() {
                        let expected = brownian.canonical()
                            [a.brownian * brownian.dimension() + b.brownian]
                            * independent_covariance(&c, a, b, times[0], times[1]);
                        near(cov[i][j], expected, 2e-11 * (1.0 + expected.abs()));
                    }
                }
            }
        }
    }
}

#[test]
fn rough_recalibrated_paired_targets_spot_gamma_curves_and_worker_replay() {
    check_paired_spot_gamma_curve_risks(rough_case());
}

#[test]
fn rough_vega_kt_all_quotes_mixed_factors_mc_errors_and_trace_contract() {
    let mut c = rough_case();
    c.full = None;
    c.configs[1] = two_factor_configs(0.3)[1].clone();
    c.rate_correlations.push(-0.04);
    c.targets = vec![Some(quotes(vec![0.28; 6])); 2];
    c.step = 0.25;
    let p = c.compile(1).unwrap();
    let r = p.evaluate_aad(MultiAssetRiskConfig::default()).unwrap();
    assert_eq!(p.random_factor_count(), 8);
    assert_eq!(r.price.value(), p.evaluate().unwrap().price.value());
    for asset in 0..2 {
        let risk = r.risks[asset].hull_white_lsv.as_ref().unwrap();
        assert_eq!(
            r.risks[asset]
                .lsv_local_variance
                .as_ref()
                .unwrap()
                .time_nodes,
            p.time_nodes()
        );
        for j in 0..6 {
            let bumped = |h| {
                let mut b = c.clone();
                let mut values = vec![0.28; 6];
                values[j] += h;
                b.targets[asset] = Some(quotes(values));
                b.value()
            };
            fd_close(
                risk.vega_kt_raw.as_ref().unwrap()[j],
                (bumped(1e-7) - bumped(-1e-7)) / 2e-7,
            );
        }
        near(
            risk.parallel_vega.unwrap(),
            risk.vega_kt_raw.as_ref().unwrap().iter().sum(),
            1e-12,
        );
        assert!(risk.parallel_vega_standard_error.unwrap().is_finite());
    }
    c.engine = EngineConfig::PseudoMonteCarlo(
        PseudoMcConfig::new(511, 128, VarianceReduction::new(true, true)).unwrap(),
    );
    let r = c
        .compile(1)
        .unwrap()
        .evaluate_aad(MultiAssetRiskConfig::default())
        .unwrap();
    assert!(
        r.risks[0]
            .hull_white_lsv
            .as_ref()
            .unwrap()
            .vega_kt_standard_errors
            .is_none()
    );
    assert!(
        r.risks[0]
            .lsv_local_variance
            .as_ref()
            .unwrap()
            .standard_errors
            .is_none()
    );
    c.configs[0] = Some(config(0.08, 0.5, -0.65, 401, false));
    let p = c.compile(1).unwrap();
    assert!(p.evaluate().is_ok());
    assert!(p.evaluate_aad(MultiAssetRiskConfig::default()).is_err());
    c.targets[1] = None;
    c.configs[1] = None;
    c.rate_correlations.truncate(3);
    let p = c.compile(1).unwrap();
    assert_eq!(p.random_factor_count(), 6);
    assert!(p.evaluate().is_ok());
}

#[test]
fn rough_brownian_boundary_zero_eta_and_singular_drivers_are_explicit() {
    let mut c = rough_case();
    c.full = None;
    c.rates = HullWhite1Factor::new(0.0, vec![0.0], vec![0.0]).unwrap();
    c.configs = vec![
        Some(config(0.08, 0.0, -0.65, 401, true)),
        Some(config(0.33, 0.0, -0.35, 402, true)),
    ];
    let r = c
        .compile(1)
        .unwrap()
        .evaluate_aad(MultiAssetRiskConfig::default())
        .unwrap();
    let direct = MultiAssetPricingPlan::compile(
        today(),
        c.product.clone(),
        c.markets.clone(),
        vec![bs(0.27), bs(0.31)],
        c.correlation.clone(),
        c.engine,
        ExecutionPolicy::new(1, Some(128)).unwrap(),
        0.5,
    )
    .unwrap()
    .evaluate()
    .unwrap();
    near(r.price.value().get(), direct.price.value().get(), 1e-12);
    c.rates = HullWhite1Factor::new(0.1, vec![0.0], vec![0.001]).unwrap();
    c.correlation = corr(2, 1.0);
    c.full = Some(vec![vec![vec![1.0; 5]; 5]]);
    c.rate_correlations = vec![1.0; 4];
    for hs in [[0.01, 0.3], [0.5, 0.5], [0.499999999999, 0.5]] {
        c.configs = vec![
            Some(config(hs[0], 0.0, 1.0, 401, true)),
            Some(config(hs[1], 0.0, 1.0, 402, true)),
        ];
        let p = c.compile(1).unwrap();
        assert_eq!(p.random_factor_count(), 8);
        assert_eq!(p.lsv_driver_correlations()[0].diagnostics().rank, 1);
        assert!(p.evaluate_aad(MultiAssetRiskConfig::default()).is_ok());
    }
    let mut c = rough_case();
    c.full = None;
    for i in 0..2 {
        c.configs[i] = Some(config(0.5, 0.4, [-0.65, -0.35][i], 401 + i as u64, true));
    }
    let rough = c.compile(1).unwrap().evaluate().unwrap();
    for i in 0..2 {
        c.configs[i] = Some(lsv_config(0.0, 0.2, [-0.65, -0.35][i], 401 + i as u64, true).into());
    }
    let brownian = c.compile(1).unwrap().evaluate().unwrap();
    near(
        rough.price.value().get(),
        brownian.price.value().get(),
        3e-12,
    );
}
