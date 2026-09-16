use super::*;
use pricing::mc::hull_white::HullWhiteLsvTarget;
use pricing::models::{HullWhite1Factor, RoughBergomi};

#[derive(Clone)]
struct JointCase {
    base: Case,
    config: MultiAssetBergomiLsvConfig,
    asset: HullWhiteLsvTarget,
    basket: HullWhiteLsvTarget,
    hw: bool,
    rate_vol: f64,
    second: Option<Vec<Vec<Vec<f64>>>>,
}
impl JointCase {
    fn new(mode: usize, hw: bool) -> Self {
        let mut base = Case::new();
        let times = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let xs = vec![-0.6, 0.0, 0.6];
        let asset = HullWhiteLsvTarget::flat(0.28, times.clone(), xs.clone(), 1e-6, 1.0).unwrap();
        let basket = HullWhiteLsvTarget::flat(0.235, times, xs, 1e-6, 1.0).unwrap();
        base.config.target = basket.grid().clone();
        let particles = LsvParticleConfig::new(512, 318, 0.8, 2.0, true).unwrap();
        let config = match mode {
            0 => MultiAssetLsvConfig {
                factor: Bergomi1Factor::new(0.7, 0.4, -0.35).unwrap(),
                particles,
            }
            .into(),
            1 => MultiAssetLsv2FactorConfig {
                factor: Bergomi2Factor::new([0.4, 2.0], 0.4, 0.35, [-0.35, -0.2], 0.25).unwrap(),
                particles,
            }
            .into(),
            _ => MultiAssetRoughLsvConfig {
                factor: RoughBergomi::new(if mode == 2 { 0.12 } else { 0.5 }, 0.5, -0.35).unwrap(),
                particles,
            }
            .into(),
        };
        Self {
            base,
            config,
            asset,
            basket,
            hw,
            rate_vol: 0.012,
            second: None,
        }
    }
    fn compile(&self, workers: u32) -> Result<MultiAssetPricingPlan, MultiAssetError> {
        let c = &self.base;
        let g = self.asset.grid();
        let models = vec![
            ModelSpec::LocalVolatility(
                LocalVolatilitySpec::from_explicit_grid(
                    g.time_nodes().to_vec(),
                    g.log_moneyness_nodes().to_vec(),
                    g.values().to_vec(),
                    g.floor(),
                    g.cap(),
                )
                .unwrap(),
            ),
            c.models[1].clone(),
        ];
        let mut config = c.config.clone();
        config.target = self.basket.grid().clone();
        let mut rho = vec![0.1, -0.05];
        rho.extend(vec![0.02; self.config.factor_count()]);
        MultiAssetPricingPlan::compile_with_joint_local_correlation(
            today(),
            c.product.clone(),
            c.markets.clone(),
            models,
            c.first.clone(),
            c.engine,
            ExecutionPolicy::new(workers, Some(128)).unwrap(),
            c.step,
            vec![Some(self.config.clone()), None],
            None,
            self.hw.then(|| MultiAssetHullWhiteConfig {
                rate_model: HullWhite1Factor::new(
                    0.13,
                    vec![0.0, 0.37],
                    vec![self.rate_vol, self.rate_vol * 1.2],
                )
                .unwrap(),
                rate_correlations: rho,
                lsv_targets: vec![Some(self.asset.clone()), None],
            }),
            config,
            LocalCorrelationExtensions {
                second_driver_correlations: self.second.clone(),
                hull_white_target: self.hw.then(|| self.basket.clone()),
            },
        )
    }
    fn fixed(&self) -> MultiAssetPricingPlan {
        let c = &self.base;
        let g = self.asset.grid();
        let models = vec![
            ModelSpec::LocalVolatility(
                LocalVolatilitySpec::from_explicit_grid(
                    g.time_nodes().to_vec(),
                    g.log_moneyness_nodes().to_vec(),
                    g.values().to_vec(),
                    g.floor(),
                    g.cap(),
                )
                .unwrap(),
            ),
            c.models[1].clone(),
        ];
        let mut rho = vec![0.1, -0.05];
        rho.extend(vec![0.02; self.config.factor_count()]);
        let execution = ExecutionPolicy::new(1, Some(128)).unwrap();
        if self.hw {
            MultiAssetPricingPlan::compile_with_hull_white(
                today(),
                c.product.clone(),
                c.markets.clone(),
                models,
                c.first.clone(),
                c.engine,
                execution,
                c.step,
                vec![Some(self.config.clone()), None],
                None,
                MultiAssetHullWhiteConfig {
                    rate_model: HullWhite1Factor::new(
                        0.13,
                        vec![0.0, 0.37],
                        vec![self.rate_vol, self.rate_vol * 1.2],
                    )
                    .unwrap(),
                    rate_correlations: rho,
                    lsv_targets: vec![Some(self.asset.clone()), None],
                },
            )
            .unwrap()
        } else {
            MultiAssetPricingPlan::compile_with_bergomi_lsv(
                today(),
                c.product.clone(),
                c.markets.clone(),
                models,
                c.first.clone(),
                c.engine,
                execution,
                c.step,
                vec![Some(self.config.clone()), None],
                None,
            )
            .unwrap()
        }
    }
    fn value(&self) -> f64 {
        self.compile(1)
            .unwrap()
            .evaluate()
            .unwrap()
            .price
            .value()
            .get()
    }
    fn bump(&self, basket: bool, density: bool, k: usize, h: f64) -> Self {
        let mut c = self.clone();
        let target = if basket { &mut c.basket } else { &mut c.asset };
        let g = target.grid();
        let mut v = g.values().to_vec();
        let mut p = target.log_densities().to_vec();
        if density {
            p[k] += h;
        } else {
            v[k] += h;
        }
        *target = HullWhiteLsvTarget::new(
            LocalVarianceGrid::new(
                g.time_nodes().to_vec(),
                g.log_moneyness_nodes().to_vec(),
                v,
                g.floor(),
                g.cap(),
            )
            .unwrap(),
            p,
        )
        .unwrap();
        c
    }
}
#[test]
fn joint_local_correlation_recalibrated_lsv_hw_and_rough_vjp() {
    for (mode, hw) in [
        (0, false),
        (1, false),
        (0, true),
        (1, true),
        (2, true),
        (3, true),
    ] {
        let c = JointCase::new(mode, hw);
        let p = c.compile(1).unwrap();
        let out = p.evaluate_aad(MultiAssetRiskConfig::default()).unwrap();
        let risk = out.local_correlation_risk.as_ref().unwrap();
        assert_eq!(out.price, p.evaluate().unwrap().price);
        assert_eq!(
            p.random_factor_count(),
            2 * (2 + c.config.factor_count() + usize::from(hw) * 2 + usize::from(mode >= 2))
        );
        for basket in [true, false] {
            for density in [false, true] {
                if density && !hw {
                    continue;
                }
                for k in [1, 7] {
                    if density && k == 1 {
                        continue;
                    }
                    let adj = match (basket, density) {
                        (true, false) => risk.basket_variance_adjoints[k],
                        (false, false) => risk.asset_adjoints[0][k],
                        (true, true) => {
                            risk.basket_hull_white
                                .as_ref()
                                .unwrap()
                                .forward_log_density_adjoints[k]
                        }
                        (false, true) => {
                            risk.asset_hull_white[0]
                                .as_ref()
                                .unwrap()
                                .forward_log_density_adjoints[k]
                        }
                    };
                    let h = 1e-7;
                    let expected = (c.bump(basket, density, k, h).value()
                        - c.bump(basket, density, k, -h).value())
                        / (2.0 * h);
                    assert!(
                        (adj - expected).abs() < 7e-5 * (1.0 + expected.abs()),
                        "mode={mode} hw={hw} basket={basket} density={density} node={k}: aad {adj} fd {expected}"
                    );
                }
            }
        }
        let bump = |h| {
            let mut b = c.clone();
            b.base.models[1] = bs(0.3 + h);
            b.value()
        };
        fd(risk.asset_adjoints[1][0], (bump(1e-7) - bump(-1e-7)) / 2e-7);
        let replay = c
            .compile(3)
            .unwrap()
            .evaluate_aad(MultiAssetRiskConfig::default())
            .unwrap();
        assert_eq!(risk, replay.local_correlation_risk.as_ref().unwrap());
        assert_eq!(out.fingerprint, replay.fingerprint);
    }
}
#[test]
fn joint_local_correlation_endpoints_preserve_marginals_and_validate_full_matrix() {
    let mut c = JointCase::new(1, true);
    let p = c.compile(1).unwrap();
    let cal = p.local_correlation_calibration().unwrap();
    for t in [0.0, 0.5, 1.0] {
        for x in [-0.5, 0.0, 0.5] {
            let r = cal.driver_correlation_at(t, x).unwrap();
            CorrelationFactor::compile(r.clone(), tol()).unwrap();
            for (i, j, v) in [
                (0, 2, -0.35),
                (0, 3, -0.2),
                (2, 3, 0.25),
                (0, 4, 0.1),
                (1, 4, -0.05),
                (2, 4, 0.02),
                (3, 4, 0.02),
            ] {
                near(r[i][j], v, 1e-13);
            }
            near(r[0][1], cal.correlation_at(t, x).unwrap()[0][1], 1e-13);
        }
    }
    assert_eq!(p.lsv_transition_covariances()[0].len(), 6);
    let full = vec![
        vec![1.0, 0.95, -0.35, -0.2, 0.1],
        vec![0.95, 1.0, -0.3325, -0.19, -0.05],
        vec![-0.35, -0.3325, 1.0, 0.25, 0.02],
        vec![-0.2, -0.19, 0.25, 1.0, 0.02],
        vec![0.1, -0.05, 0.02, 0.02, 1.0],
    ];
    c.second = Some(vec![full.clone()]);
    let explicit = c.compile(1).unwrap();
    let mut bad = full.clone();
    bad[0][2] = -0.34;
    bad[2][0] = -0.34;
    c.second = Some(vec![bad]);
    assert!(c.compile(1).is_err());
    let mut alternate = full;
    alternate[1][2] += 0.01;
    alternate[2][1] += 0.01;
    c.second = Some(vec![alternate]);
    assert_ne!(explicit.fingerprint(), c.compile(1).unwrap().fingerprint());
    c.hw = false;
    assert!(c.compile(1).is_err()); // full endpoint still includes rate
}
#[test]
fn joint_local_correlation_hw_cash_dividends_discounting_and_curve_risk() {
    let mut c = JointCase::new(2, true);
    c.base.markets[0] = market(
        1,
        100.0,
        0.025,
        0.01,
        vec![
            DividendEvent::new(
                EventId::new(1),
                0.5,
                DividendQuote::fixed_cash_and_proportional(2.0, 0.03, EventId::new(1)).unwrap(),
            )
            .unwrap(),
        ],
    );
    c.base.product = MultiAssetProduct::basket(
        CurrencyId::new(1),
        vec![
            BasketComponent {
                underlying: u(1),
                weight: 0.6,
                scale: 1.0,
            },
            BasketComponent {
                underlying: u(2),
                weight: 0.4,
                scale: 1.0,
            },
        ],
        OptionSide::Call,
        96.0,
        1.0,
        expiry(),
        d("2027-02-01"),
        Some(CompactC2Smoothing::new(3.0).unwrap()),
    )
    .unwrap();
    let out = c
        .compile(1)
        .unwrap()
        .evaluate_aad(MultiAssetRiskConfig {
            gamma_relative_bump: Some(0.001),
        })
        .unwrap();
    let risk = out.local_correlation_risk.as_ref().unwrap();
    fd(
        risk.basket_variance_adjoints[7],
        (c.bump(true, false, 7, 1e-7).value() - c.bump(true, false, 7, -1e-7).value()) / 2e-7,
    );
    for i in 0..2 {
        let bump = |h| {
            let mut b = c.clone();
            b.base.markets[i] = bump_market(
                &c.base.markets[i],
                c.base.markets[i].forward().spot().get() + h,
            );
            b.value()
        };
        fd(
            out.risks[i].delta.value().get(),
            (bump(1e-4) - bump(-1e-4)) / 2e-4,
        );
    }
    let shifted = |h: f64| {
        let mut b = c.clone();
        b.base.markets = b
            .base
            .markets
            .iter()
            .enumerate()
            .map(|(i, m)| {
                market(
                    i as u32 + 1,
                    m.forward().spot().get(),
                    0.025 + h,
                    0.01 * (i + 1) as f64,
                    if i == 0 {
                        vec![
                            DividendEvent::new(
                                EventId::new(1),
                                0.5,
                                DividendQuote::fixed_cash_and_proportional(
                                    2.0,
                                    0.03,
                                    EventId::new(1),
                                )
                                .unwrap(),
                            )
                            .unwrap(),
                        ]
                    } else {
                        vec![]
                    },
                )
            })
            .collect();
        b.value()
    };
    let cr = out.hull_white_curve_risk.as_ref().unwrap();
    let parallel: f64 = cr
        .discount_log_df_adjoints
        .iter()
        .zip(&cr.discount_time_nodes)
        .map(|(a, t)| -t * a.value().get())
        .sum();
    fd(parallel, (shifted(1e-6) - shifted(-1e-6)) / 2e-6);
}

#[test]
fn joint_local_correlation_fixed_endpoint_replays_existing_joint_processes() {
    for (mode, hw, rate_vol) in [
        (0, false, 0.0),
        (1, false, 0.0),
        (0, true, 0.012),
        (1, true, 0.012),
        (2, true, 0.012),
        (3, true, 0.012),
        (2, true, 0.0),
    ] {
        let mut c = JointCase::new(mode, hw);
        c.rate_vol = rate_vol;
        c.base.engine = pseudo();
        c.base.config.second_correlation = c.base.first.clone();
        let plan = c.compile(1).unwrap();
        assert!(
            plan.local_correlation_calibration()
                .unwrap()
                .mixing_coefficients()
                .iter()
                .all(|v| *v == 0.0)
        );
        let actual = plan.evaluate().unwrap();
        let fixed = c.fixed().evaluate().unwrap();
        near(actual.price.value().get(), fixed.price.value().get(), 2e-11);
        near(
            actual.price.standard_error().get(),
            fixed.price.standard_error().get(),
            2e-11,
        );
    }
}

#[test]
fn joint_local_correlation_multiple_volatility_assets_and_density_fallback() {
    let mut c = JointCase::new(1, true);
    let g = c.asset.grid();
    let model = ModelSpec::LocalVolatility(
        LocalVolatilitySpec::from_explicit_grid(
            g.time_nodes().to_vec(),
            g.log_moneyness_nodes().to_vec(),
            g.values().to_vec(),
            g.floor(),
            g.cap(),
        )
        .unwrap(),
    );
    let configs = vec![
        Some(c.config.clone()),
        Some(
            MultiAssetRoughLsvConfig {
                factor: RoughBergomi::new(0.2, 0.45, -0.3).unwrap(),
                particles: LsvParticleConfig::new(512, 319, 0.8, 2.0, true).unwrap(),
            }
            .into(),
        ),
    ];
    let compile = |basket: HullWhiteLsvTarget| {
        let mut lc = c.base.config.clone();
        lc.target = basket.grid().clone();
        MultiAssetPricingPlan::compile_with_joint_local_correlation(
            today(),
            c.base.product.clone(),
            c.base.markets.clone(),
            vec![model.clone(), model.clone()],
            c.base.first.clone(),
            c.base.engine,
            ExecutionPolicy::new(1, Some(128)).unwrap(),
            c.base.step,
            configs.clone(),
            None,
            Some(MultiAssetHullWhiteConfig {
                rate_model: HullWhite1Factor::new(0.13, vec![0.0], vec![0.012]).unwrap(),
                rate_correlations: vec![0.04; 5],
                lsv_targets: vec![Some(c.asset.clone()); 2],
            }),
            lc,
            LocalCorrelationExtensions {
                hull_white_target: Some(basket),
                second_driver_correlations: None,
            },
        )
        .unwrap()
    };
    let p = compile(c.basket.clone());
    assert_eq!(p.random_factor_count(), 16);
    let risk = p
        .evaluate_aad(MultiAssetRiskConfig::default())
        .unwrap()
        .local_correlation_risk
        .unwrap();
    let expected = (compile(c.bump(true, false, 7, 1e-7).basket)
        .evaluate()
        .unwrap()
        .price
        .value()
        .get()
        - compile(c.bump(true, false, 7, -1e-7).basket)
            .evaluate()
            .unwrap()
            .price
            .value()
            .get())
        / 2e-7;
    fd(risk.basket_variance_adjoints[7], expected);
    assert!(risk.asset_hull_white.iter().all(Option::is_some));
    for t in [0.0, 0.5] {
        let m = p
            .local_correlation_calibration()
            .unwrap()
            .driver_correlation_at(t, 0.0)
            .unwrap();
        near(m[0][2], -0.35, 1e-13);
        near(m[0][3], -0.2, 1e-13);
        near(m[1][4], -0.3, 1e-13);
    }
    let mut density = c.basket.log_densities().to_vec();
    density[6] = 0.0;
    c.basket = HullWhiteLsvTarget::new(c.basket.grid().clone(), density).unwrap();
    let p = c.compile(1).unwrap();
    let cal = p.local_correlation_calibration().unwrap();
    assert!(cal.diagnostics()[6].fallback);
    let risk = p
        .evaluate_aad(MultiAssetRiskConfig::default())
        .unwrap()
        .local_correlation_risk
        .unwrap();
    assert_eq!(
        risk.basket_hull_white.unwrap().forward_log_density_adjoints[6],
        0.0
    );
}
