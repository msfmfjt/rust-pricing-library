use super::*;
use pricing::market::{CorrelationFactor, LocalVarianceGrid};

#[derive(Clone)]
struct Case {
    config: LocalCorrelationConfig,
    models: Vec<ModelSpec>,
    markets: Vec<EquityMarket>,
    product: MultiAssetProduct,
    first: CorrelationTermStructure,
    step: f64,
    engine: EngineConfig,
}
impl Case {
    fn new() -> Self {
        Self {
            config: LocalCorrelationConfig {
                basket_weights: vec![0.6, 0.4],
                target: LocalVarianceGrid::new(
                    vec![0.0, 0.5, 1.0],
                    vec![-0.6, 0.0, 0.6],
                    vec![
                        0.059, 0.053, 0.049, 0.061, 0.055, 0.050, 0.063, 0.057, 0.052,
                    ],
                    1e-6,
                    1.0,
                )
                .unwrap(),
                second_correlation: corr(2, 0.95),
                particles: LsvParticleConfig::new(512, 8401, 0.7, 2.0, true).unwrap(),
                feasibility: LocalCorrelationFeasibility::ProjectAndReport,
                minimum_variance_span: 1e-12,
            },
            models: vec![
                lv(vec![
                    0.082, 0.070, 0.065, 0.086, 0.074, 0.068, 0.090, 0.078, 0.072,
                ]),
                bs(0.3),
            ],
            markets: vec![
                market(1, 100.0, 0.025, 0.01, vec![]),
                market(2, 90.0, 0.025, 0.02, vec![]),
            ],
            product: basket(&[0.6, 0.4], 96.0, Some(3.0)),
            first: corr(2, -0.3),
            step: 0.25,
            engine: rqmc(128, true),
        }
    }
    fn compile(&self, workers: u32) -> Result<MultiAssetPricingPlan, MultiAssetError> {
        MultiAssetPricingPlan::compile_with_local_correlation(
            today(),
            self.product.clone(),
            self.markets.clone(),
            self.models.clone(),
            self.first.clone(),
            self.engine,
            ExecutionPolicy::new(workers, Some(128)).unwrap(),
            self.step,
            self.config.clone(),
        )
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
    fn basket_bump(&self, j: usize, h: f64) -> Self {
        let mut c = self.clone();
        let g = &self.config.target;
        let mut v = g.values().to_vec();
        v[j] += h;
        c.config.target = LocalVarianceGrid::new(
            g.time_nodes().to_vec(),
            g.log_moneyness_nodes().to_vec(),
            v,
            g.floor(),
            g.cap(),
        )
        .unwrap();
        c
    }
}
fn fd(actual: f64, expected: f64) {
    near(actual, expected, 4e-5 * (1.0 + expected.abs()));
}

#[test]
fn local_correlation_recalibrated_basket_and_all_constituent_risks() {
    let mut c = Case::new();
    c.markets = vec![
        market(
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
        ),
        market(2, 90.0, 0.025, 0.02, vec![]),
    ];
    let p = c.compile(1).unwrap();
    let result = p
        .evaluate_aad(MultiAssetRiskConfig {
            gamma_relative_bump: Some(0.001),
        })
        .unwrap();
    assert_eq!(result.price.value(), p.evaluate().unwrap().price.value());
    let risk = result.local_correlation_risk.as_ref().unwrap();
    for j in 0..9 {
        fd(
            risk.basket_variance_adjoints[j],
            (c.basket_bump(j, 1e-7).value() - c.basket_bump(j, -1e-7).value()) / 2e-7,
        );
        let bump = |h| {
            let mut b = c.clone();
            if let ModelSpec::LocalVolatility(v) = &c.models[0] {
                let mut values = v.local_variance_grid().values().to_vec();
                values[j] += h;
                b.models[0] = lv(values);
            }
            b.value()
        };
        fd(risk.asset_adjoints[0][j], (bump(1e-7) - bump(-1e-7)) / 2e-7);
    }
    let bump = |h| {
        let mut b = c.clone();
        b.models[1] = bs(0.3 + h);
        b.value()
    };
    fd(risk.asset_adjoints[1][0], (bump(1e-7) - bump(-1e-7)) / 2e-7);
    assert!(
        risk.basket_standard_errors
            .as_ref()
            .unwrap()
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    );
    assert!(
        result
            .risks
            .iter()
            .all(|r| r.bs_vega.is_none() && r.local_variance.is_empty())
    );
    for j in 0..2 {
        let h = c.markets[j].forward().spot().get() * 0.001;
        let bumped = |h| {
            let mut b = c.clone();
            b.markets[j] = bump_market(&c.markets[j], c.markets[j].forward().spot().get() + h);
            b
        };
        fd(
            result.risks[j].delta.value().get(),
            (bumped(1e-4).value() - bumped(-1e-4).value()) / 2e-4,
        );
        let up = bumped(h)
            .compile(1)
            .unwrap()
            .evaluate_aad(MultiAssetRiskConfig::default())
            .unwrap();
        let down = bumped(-h)
            .compile(1)
            .unwrap()
            .evaluate_aad(MultiAssetRiskConfig::default())
            .unwrap();
        for i in 0..2 {
            near(
                result.gamma[i][j].value().get(),
                (up.risks[i].delta.value().get() - down.risks[i].delta.value().get()) / (2.0 * h),
                1e-10,
            );
        }
    }
    let replay = c
        .compile(3)
        .unwrap()
        .evaluate_aad(MultiAssetRiskConfig {
            gamma_relative_bump: Some(0.001),
        })
        .unwrap();
    assert_eq!(result.price, replay.price);
    assert_eq!(result.local_correlation_risk, replay.local_correlation_risk);
    assert_eq!(result.gamma, replay.gamma);
    assert_eq!(result.fingerprint, replay.fingerprint);
}

#[test]
fn local_correlation_psd_projection_support_and_input_contracts() {
    let mut c = Case::new();
    let p = c.compile(1).unwrap();
    let cal = p.local_correlation_calibration().unwrap();
    assert_eq!(p.random_factor_count(), 4);
    let s0 = 0.07_f64.sqrt();
    let a = 0.6 * s0;
    let b = 0.4 * 0.3;
    let q0 = a * a + b * b - 0.6 * a * b;
    let q1 = a * a + b * b + 1.9 * a * b;
    let lambda = (0.053 - q0) / (q1 - q0);
    near(cal.mixing_coefficients()[1], lambda, 1e-14);
    for t in [0.0, 0.3, 0.75, 1.0] {
        for x in [-9.0, -0.3, 0.0, 0.4, 9.0] {
            let r = cal.correlation_at(t, x).unwrap();
            CorrelationFactor::compile(r.clone(), tol()).unwrap();
            near(r[0][0], 1.0, 1e-15);
            assert!((-0.3..=0.95).contains(&r[0][1]));
        }
    }
    for (index, d) in cal.diagnostics().iter().enumerate() {
        near(
            d.attained_variance,
            d.endpoint_variances[0]
                + cal.mixing_coefficients()[index]
                    * (d.endpoint_variances[1] - d.endpoint_variances[0]),
            1e-14,
        );
        if !d.projected {
            near(d.attained_variance, d.target_variance, 1e-12);
        }
    }
    c.config.target = LocalVarianceGrid::new(
        vec![0.0, 1.0],
        vec![-0.6, 0.0, 0.6],
        vec![0.9; 6],
        1e-6,
        1.0,
    )
    .unwrap();
    c.config.feasibility = LocalCorrelationFeasibility::Reject;
    assert!(c.compile(1).is_err());
    c.config.feasibility = LocalCorrelationFeasibility::ProjectAndReport;
    let p = c.compile(1).unwrap();
    let cal = p.local_correlation_calibration().unwrap();
    assert!(cal.diagnostics().iter().all(|d| d.projected));
    assert!(cal.mixing_coefficients().iter().all(|&v| v == 1.0));
    assert!(p.evaluate_aad(MultiAssetRiskConfig::default()).is_ok());
    for weights in [vec![0.6], vec![-0.2, 1.2], vec![0.0, 1.0], vec![0.6, 0.5]] {
        let mut b = c.clone();
        b.config.basket_weights = weights;
        assert!(b.compile(1).is_err());
    }
    c.config.second_correlation = CorrelationTermStructure::new(
        vec![u(1), u(2)],
        vec![
            (today(), vec![vec![1.0, 0.9], vec![0.9, 1.0]]),
            (d("2030-01-01"), vec![vec![1.0, 0.8], vec![0.8, 1.0]]),
        ],
        tol(),
    )
    .unwrap();
    assert!(c.compile(1).is_err());
}

#[test]
fn local_correlation_mc_trace_and_identical_endpoint_black_scholes_limit() {
    let mut c = Case::new();
    c.engine = pseudo();
    let risk = c
        .compile(1)
        .unwrap()
        .evaluate_aad(MultiAssetRiskConfig::default())
        .unwrap()
        .local_correlation_risk
        .unwrap();
    assert!(risk.basket_standard_errors.is_none() && risk.asset_standard_errors.is_none());
    c.config.particles = LsvParticleConfig::new(512, 8401, 0.7, 2.0, false).unwrap();
    let p = c.compile(1).unwrap();
    assert!(p.evaluate().is_ok());
    assert!(p.evaluate_aad(MultiAssetRiskConfig::default()).is_err());
    c.models = vec![bs(0.2), bs(0.2)];
    c.first = corr(2, 1.0);
    c.config.second_correlation = corr(2, 1.0);
    c.config.target = LocalVarianceGrid::new(
        vec![0.0, 0.5, 1.0],
        vec![-0.6, 0.0, 0.6],
        vec![0.04; 9],
        1e-6,
        1.0,
    )
    .unwrap();
    c.config.feasibility = LocalCorrelationFeasibility::Reject;
    c.engine = rqmc(4096, true);
    c.product = basket(&[0.6, 0.4], 96.0, None);
    let p = c.compile(1).unwrap();
    assert!(
        p.local_correlation_calibration()
            .unwrap()
            .diagnostics()
            .iter()
            .all(|d| d.unidentifiable)
    );
    let forward = 0.6 * 100.0 * 0.015_f64.exp() + 0.4 * 90.0 * 0.005_f64.exp();
    let d1 = ((forward / 96.0).ln() + 0.02) / 0.2;
    statistical(
        p.evaluate().unwrap().price,
        (-0.025_f64).exp() * (forward * cdf(d1) - 96.0 * cdf(d1 - 0.2)),
        0.002,
    );
}

#[test]
fn local_correlation_donor_reverse_and_exact_active_set_transition() {
    let mut c = Case::new();
    c.config.target = LocalVarianceGrid::new(
        vec![0.0, 0.5, 1.0],
        vec![-4.0, 0.0, 4.0],
        vec![
            0.059, 0.053, 0.049, 0.061, 0.055, 0.050, 0.063, 0.057, 0.052,
        ],
        1e-6,
        1.0,
    )
    .unwrap();
    c.config.particles = LsvParticleConfig::new(512, 8401, 0.3, 4.0, true).unwrap();
    let p = c.compile(1).unwrap();
    let cal = p.local_correlation_calibration().unwrap();
    assert!(cal.diagnostics()[3..].iter().any(|d| d.fallback));
    for d in cal.diagnostics().iter().filter(|d| d.fallback) {
        assert_eq!(d.source_node, 1);
        assert_eq!(d.effective_samples, 0.0);
    }
    let risk = p
        .evaluate_aad(MultiAssetRiskConfig::default())
        .unwrap()
        .local_correlation_risk
        .unwrap();
    for j in 0..9 {
        fd(
            risk.basket_variance_adjoints[j],
            (c.basket_bump(j, 1e-7).value() - c.basket_bump(j, -1e-7).value()) / 2e-7,
        );
    }
    let bump = |h| {
        let mut b = c.clone();
        b.models[1] = bs(0.3 + h);
        b.value()
    };
    fd(risk.asset_adjoints[1][0], (bump(1e-7) - bump(-1e-7)) / 2e-7);
    let endpoint = cal.diagnostics()[0].endpoint_variances[0];
    c.config.target = LocalVarianceGrid::new(
        vec![0.0, 1.0],
        vec![-4.0, 0.0, 4.0],
        vec![endpoint; 6],
        1e-6,
        1.0,
    )
    .unwrap();
    let p = c.compile(1).unwrap();
    assert_eq!(
        p.local_correlation_calibration().unwrap().diagnostics()[1].raw_mixing,
        0.0
    );
    assert!(p.evaluate().is_ok());
    assert!(p.evaluate_aad(MultiAssetRiskConfig::default()).is_err());
}

#[test]
fn local_correlation_three_assets_dated_psd_endpoints_and_marginals() {
    let mut c = Case::new();
    c.models = vec![bs(0.3); 3];
    c.markets = (1..=3)
        .map(|i| market(i, 100.0, 0.0, 0.0, vec![]))
        .collect();
    c.config.basket_weights = vec![0.2, 0.3, 0.5];
    c.config.target = LocalVarianceGrid::new(
        vec![0.0, 0.5, 1.0],
        vec![-0.6, 0.0, 0.6],
        vec![0.07; 9],
        1e-6,
        1.0,
    )
    .unwrap();
    let a = vec![
        vec![1.0, 0.1, -0.1],
        vec![0.1, 1.0, 0.2],
        vec![-0.1, 0.2, 1.0],
    ];
    let b = vec![
        vec![1.0, 0.9, 0.8],
        vec![0.9, 1.0, 0.85],
        vec![0.8, 0.85, 1.0],
    ];
    let change = d("2026-07-02");
    let schedule = |first, second| {
        CorrelationTermStructure::new(
            vec![u(1), u(2), u(3)],
            vec![(today(), first), (change, second)],
            tol(),
        )
        .unwrap()
    };
    // Swap endpoints at the date: the algorithm must handle a negative variance span.
    c.first = schedule(a.clone(), b.clone());
    c.config.second_correlation = schedule(b.clone(), a.clone());
    c.engine = rqmc(2048, false);
    c.product = basket(&[1.0, 0.0, 0.0], 100.0, None);
    let p = c.compile(1).unwrap();
    let cal = p.local_correlation_calibration().unwrap();
    assert_eq!(p.random_factor_count(), 6);
    let change_time = DayCountConvention::Act365F.year_fraction(today(), change);
    assert!(cal.time_nodes().contains(&change_time));
    for t in [0.0, 0.25, change_time, 0.75] {
        for x in [-9.0, -0.2, 0.0, 0.3, 9.0] {
            let r = cal.correlation_at(t, x).unwrap();
            CorrelationFactor::compile(r.clone(), tol()).unwrap();
            let (lo, hi) = if t < change_time { (&a, &b) } else { (&b, &a) };
            let lambda = (r[0][1] - lo[0][1]) / (hi[0][1] - lo[0][1]);
            near(
                r[0][2],
                (1.0 - lambda) * lo[0][2] + lambda * hi[0][2],
                1e-14,
            );
            near(
                r[1][2],
                (1.0 - lambda) * lo[1][2] + lambda * hi[1][2],
                1e-14,
            );
        }
    }
    statistical(
        p.evaluate().unwrap().price,
        100.0 * (cdf(0.15) - cdf(-0.15)),
        0.005,
    );
    c.product = basket(&[0.2, 0.3, 0.5], 100.0, Some(3.0));
    c.engine = rqmc(128, true);
    let risk = c
        .compile(1)
        .unwrap()
        .evaluate_aad(MultiAssetRiskConfig::default())
        .unwrap()
        .local_correlation_risk
        .unwrap();
    for j in [1, 4] {
        fd(
            risk.basket_variance_adjoints[j],
            (c.basket_bump(j, 1e-7).value() - c.basket_bump(j, -1e-7).value()) / 2e-7,
        );
    }
}
