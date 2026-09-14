use super::*;
use pricing::market::{LocalVarianceGrid, MarketIvSurface};
use pricing::mc::hull_white::HullWhiteLsvTarget;
use pricing::models::HullWhite1Factor;

#[derive(Clone)]
struct Case {
    product: MultiAssetProduct,
    markets: Vec<EquityMarket>,
    targets: Vec<Option<HullWhiteLsvTarget>>,
    configs: Vec<Option<MultiAssetBergomiLsvConfig>>,
    correlation: CorrelationTermStructure,
    rates: HullWhite1Factor,
    rate_correlations: Vec<f64>,
    full: Option<Vec<Vec<Vec<f64>>>>,
    engine: EngineConfig,
    step: f64,
}
impl Case {
    fn new() -> Self {
        let rho = vec![0.2, -0.1, -0.05, 0.02, 0.03, -0.04];
        let mut full = two_factor_full();
        for (i, row) in full.iter_mut().enumerate() {
            row.push(rho[i]);
        }
        let mut last = rho.clone();
        last.push(1.0);
        full.push(last);
        Self {
            product: basket(&[0.6, 0.4], 96.0, Some(3.0)),
            markets: vec![
                market(1, 100.0, 0.025, 0.01, vec![]),
                market(2, 90.0, 0.025, 0.02, vec![]),
            ],
            targets: [0.27, 0.31]
                .into_iter()
                .map(|v| {
                    Some(
                        HullWhiteLsvTarget::flat(
                            v,
                            vec![0.0, 0.5, 1.0],
                            vec![-0.35, 0.0, 0.35],
                            1e-8,
                            4.0,
                        )
                        .unwrap(),
                    )
                })
                .collect(),
            configs: two_factor_configs(0.3),
            correlation: corr(2, 0.4),
            rates: HullWhite1Factor::new(0.13, vec![0.0, 0.37], vec![0.004, 0.006]).unwrap(),
            rate_correlations: rho,
            full: Some(vec![full]),
            engine: rqmc(128, true),
            step: 0.5,
        }
    }
    fn compile(&self, workers: u32) -> Result<MultiAssetPricingPlan, MultiAssetError> {
        let models = self
            .targets
            .iter()
            .enumerate()
            .map(|(i, t)| match t {
                None => bs([0.24, 0.3][i]),
                Some(t) => {
                    let g = t.grid();
                    ModelSpec::LocalVolatility(
                        LocalVolatilitySpec::from_explicit_grid(
                            g.time_nodes().to_vec(),
                            g.log_moneyness_nodes().to_vec(),
                            g.values().to_vec(),
                            g.floor(),
                            g.cap(),
                        )
                        .unwrap(),
                    )
                }
            })
            .collect();
        MultiAssetPricingPlan::compile_with_hull_white(
            today(),
            self.product.clone(),
            self.markets.clone(),
            models,
            self.correlation.clone(),
            self.engine,
            ExecutionPolicy::new(workers, Some(128)).unwrap(),
            self.step,
            self.configs.clone(),
            self.full.clone(),
            MultiAssetHullWhiteConfig {
                rate_model: self.rates.clone(),
                rate_correlations: self.rate_correlations.clone(),
                lsv_targets: self.targets.clone(),
            },
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
    fn target_bump(&self, asset: usize, index: usize, amount: f64, density: bool) -> Self {
        let mut c = self.clone();
        let t = c.targets[asset].as_ref().unwrap();
        let g = t.grid();
        let mut v = g.values().to_vec();
        let mut p = t.log_densities().to_vec();
        if density {
            p[index] += amount;
        } else {
            v[index] += amount;
        }
        c.targets[asset] = Some(
            HullWhiteLsvTarget::new(
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
            .unwrap(),
        );
        c
    }
}
fn quad(f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let n = 2048;
    let h = (hi - lo) / n as f64;
    let mut s = f(lo) + f(hi);
    for i in 1..n {
        s += f(lo + h * i as f64) * if i % 2 == 0 { 2.0 } else { 4.0 };
    }
    s * h / 3.0
}
fn fd_close(a: f64, b: f64) {
    near(a, b, 3e-6 + 4e-5 * b.abs());
}

#[test]
fn hw_shared_rate_exchange_with_payment_lag_matches_independent_gaussian_integral() {
    let mut c = Case::new();
    c.targets = vec![None, None];
    c.configs = vec![None, None];
    c.rate_correlations = vec![0.3, -0.1];
    c.full = None;
    c.step = 0.25;
    c.rates = HullWhite1Factor::new(0.13, vec![0.0, 0.37], vec![0.015, 0.022]).unwrap();
    c.engine = rqmc(4096, true);
    for payment in [expiry(), d("2027-04-01")] {
        c.product = MultiAssetProduct::basket(
            CurrencyId::new(1),
            vec![
                BasketComponent {
                    underlying: u(1),
                    weight: 1.0,
                    scale: 1.0,
                },
                BasketComponent {
                    underlying: u(2),
                    weight: -1.0,
                    scale: 1.0,
                },
            ],
            OptionSide::Call,
            0.0,
            1.0,
            expiry(),
            payment,
            None,
        )
        .unwrap();
        let t = DayCountConvention::Act365F.year_fraction(today(), payment);
        let b = -(-0.13 * (t - 1.0)).exp_m1() / 0.13;
        let ix = quad(
            |s| {
                let z = 1.0 - s;
                0.015f64.powi(2) * (-0.13 * z).exp() * (-(-0.13 * z).exp_m1()) / 0.13
            },
            0.0,
            0.37,
        ) + quad(
            |s| {
                let z = 1.0 - s;
                0.022f64.powi(2) * (-0.13 * z).exp() * (-(-0.13 * z).exp_m1()) / 0.13
            },
            0.37,
            1.0,
        );
        let wx = quad(|s| 0.015 * (-0.13 * (1.0 - s)).exp(), 0.0, 0.37)
            + quad(|s| 0.022 * (-0.13 * (1.0 - s)).exp(), 0.37, 1.0);
        let (v1, v2, rho) = (0.24f64, 0.3f64, 0.4f64);
        let f1 = 100.0 * (0.025f64 - 0.01).exp();
        let f2 = 90.0 * (0.025f64 - 0.02).exp();
        let variance = v1 * v1 + v2 * v2 - 2.0 * rho * v1 * v2;
        let mean = (f1 / f2).ln() - 0.5 * (v1 * v1 - v2 * v2);
        let shift = b * (v1 * 0.3 - v2 * (-0.1)) * wx;
        let expected = (-0.025 * t).exp()
            * (f1
                * (-b * (ix + v1 * 0.3 * wx)).exp()
                * cdf((mean + v1 * v1 - rho * v1 * v2 - shift) / variance.sqrt())
                - f2 * (-b * (ix + v2 * (-0.1) * wx)).exp()
                    * cdf((mean + rho * v1 * v2 - v2 * v2 - shift) / variance.sqrt()));
        let p = c.compile(1).unwrap();
        assert_eq!(p.random_factor_count(), 4);
        statistical(p.evaluate().unwrap().price, expected, 0.001);
    }
}

#[test]
fn hw_eight_coordinate_covariance_matches_quadrature_and_checks_future_marginals() {
    let mut c = Case::new();
    let mut later = c.full.as_ref().unwrap()[0].clone();
    later[0][1] = 0.25;
    later[1][0] = 0.25;
    c.correlation = CorrelationTermStructure::new(
        vec![u(1), u(2)],
        vec![
            (today(), vec![vec![1.0, 0.4], vec![0.4, 1.0]]),
            (d("2026-07-02"), vec![vec![1.0, 0.25], vec![0.25, 1.0]]),
        ],
        tol(),
    )
    .unwrap();
    // July 2 is not exactly 0.5 ACT/365F: quote provenance regenerates marginals.
    let surface =
        MarketIvSurface::new(vec![0.5, 1.0], vec![-0.5, 0.0, 0.5], vec![0.28; 6]).unwrap();
    c.targets = vec![
        Some(
            HullWhiteLsvTarget::from_market_iv(
                surface,
                vec![0.0, 0.5, 1.0],
                vec![-0.35, 0.0, 0.35],
                1e-8,
                4.0
            )
            .unwrap()
        );
        2
    ];
    c.full.as_mut().unwrap().push(later);
    let p = c.compile(1).unwrap();
    assert_eq!(p.random_factor_count(), 8);
    let ks = [0.0, 0.0, 4.0, 0.35, 2.2, 0.12];
    for (s, actual) in p.lsv_transition_covariances().iter().enumerate() {
        let lo = p.time_nodes()[s];
        let end = p.time_nodes()[s + 1];
        let entry = p.correlation_entry_indices()[s];
        let matrix = &c.full.as_ref().unwrap()[entry];
        let kernel = |i: usize, t: f64, sigma: f64| {
            if i < 6 {
                (-ks[i] * (end - t)).exp()
            } else if i == 6 {
                sigma * (-0.13 * (end - t)).exp()
            } else {
                sigma * (-(-0.13 * (end - t)).exp_m1()) / 0.13
            }
        };
        for i in 0..8 {
            for j in 0..8 {
                let rho = matrix[i.min(6)][j.min(6)];
                let mut expected = 0.0;
                for (a, b, sigma) in [(lo, end.min(0.37), 0.004), (lo.max(0.37), end, 0.006)] {
                    if b > a {
                        expected += rho * quad(|t| kernel(i, t, sigma) * kernel(j, t, sigma), a, b);
                    }
                }
                near(actual[i][j], expected, 2e-11 * (1.0 + expected.abs()));
            }
        }
    }
    let mut bad = c.clone();
    let m = &mut bad.full.as_mut().unwrap()[1];
    m[0][6] += 0.01;
    m[6][0] += 0.01;
    assert!(bad.compile(1).is_err());
    let mut bad = c.clone();
    bad.rate_correlations[0] = f64::NAN;
    assert!(bad.compile(1).is_err());
    let mut bad = c.clone();
    bad.rate_correlations.pop();
    assert!(bad.compile(1).is_err());
    let mut bad = c.clone();
    bad.targets[0] = None;
    assert!(bad.compile(1).is_err());
    let mut bad = Case::new();
    bad.step = 0.25;
    assert!(bad.compile(1).is_err());
}

#[test]
fn hw_two_factor_all_paired_buckets_spot_gamma_and_curves_match_recalibration() {
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
                    DividendQuote::FixedCashAndProportional {
                        fixed_cash: 2.0,
                        beta: 0.02,
                    },
                )
                .unwrap(),
            ],
        ),
        market(
            2,
            90.0,
            0.025,
            0.02,
            vec![DividendEvent::new(EventId::new(2), 0.5, DividendQuote::FixedCash(1.0)).unwrap()],
        ),
    ];
    let p = c.compile(1).unwrap();
    let cfg = MultiAssetRiskConfig {
        gamma_relative_bump: Some(0.001),
    };
    let risk = p.evaluate_aad(cfg).unwrap();
    assert_eq!(p.evaluate().unwrap().price.value(), risk.price.value());
    let replay = c.compile(3).unwrap().evaluate_aad(cfg).unwrap();
    assert_eq!(risk.risks, replay.risks);
    assert_eq!(risk.gamma, replay.gamma);
    assert_eq!(risk.hull_white_curve_risk, replay.hull_white_curve_risk);
    assert!(
        p.hull_white_lsv_calibrations()
            .iter()
            .flatten()
            .any(|c| c.rate_corrections.iter().any(|v| v.abs() > 1e-7))
    );
    for asset in 0..2 {
        let r = &risk.risks[asset];
        let bars = &r.lsv_local_variance.as_ref().unwrap().node_adjoints;
        let densities = &r
            .hull_white_lsv
            .as_ref()
            .unwrap()
            .forward_log_density_adjoints;
        assert!(densities.iter().any(|v| v.abs() > 1e-8));
        assert!(densities[..3].iter().all(|v| *v == 0.0));
        for density in [false, true] {
            for j in if density { 3..9 } else { 0..9 } {
                let h = 1e-7;
                let fd = (c.target_bump(asset, j, h, density).value()
                    - c.target_bump(asset, j, -h, density).value())
                    / (2.0 * h);
                fd_close(if density { densities[j] } else { bars[j] }, fd);
            }
        }
        let spot = c.markets[asset].forward().spot().get();
        let h = spot * 0.001;
        let mut up = c.clone();
        up.markets[asset] = bump_market(&c.markets[asset], spot + h);
        let mut down = c.clone();
        down.markets[asset] = bump_market(&c.markets[asset], spot - h);
        let ru = up
            .compile(1)
            .unwrap()
            .evaluate_aad(MultiAssetRiskConfig::default())
            .unwrap();
        let rd = down
            .compile(1)
            .unwrap()
            .evaluate_aad(MultiAssetRiskConfig::default())
            .unwrap();
        for i in 0..2 {
            fd_close(
                risk.gamma[i][asset].value().get(),
                (ru.risks[i].delta.value().get() - rd.risks[i].delta.value().get()) / (2.0 * h),
            );
        }
        let h = 1e-4;
        up.markets[asset] = bump_market(&c.markets[asset], spot + h);
        down.markets[asset] = bump_market(&c.markets[asset], spot - h);
        fd_close(
            r.delta.value().get(),
            (up.value() - down.value()) / (2.0 * h),
        );
    }
    let curves = risk.hull_white_curve_risk.as_ref().unwrap();
    for q_asset in [None, Some(0), Some(1)] {
        let bump = |amount: f64| {
            let mut b = c.clone();
            for (i, m) in b.markets.iter_mut().enumerate() {
                let f = c.markets[i].forward();
                let r = 0.025 + if q_asset.is_none() { amount } else { 0.0 };
                let q = [0.01, 0.02][i] + if q_asset == Some(i) { amount } else { 0.0 };
                *m = market(
                    i as u32 + 1,
                    f.spot().get(),
                    r,
                    q,
                    f.discrete_dividends()
                        .unwrap()
                        .events()
                        .iter()
                        .map(|e| {
                            DividendEvent::new(
                                e.event(),
                                e.ex_time(),
                                DividendQuote::FixedCashAndProportional {
                                    fixed_cash: e.fixed_cash(),
                                    beta: e.beta(),
                                },
                            )
                            .unwrap()
                        })
                        .collect(),
                );
            }
            b.value()
        };
        let h = 1e-7;
        let fd = (bump(h) - bump(-h)) / (2.0 * h);
        let bar = q_asset.map_or(curves.discount_log_df_adjoints[1], |i| {
            curves.dividend_log_df_adjoints[i][1]
        });
        fd_close(-2.0 * bar.value().get(), fd);
    }
}

#[test]
fn hw_zero_rate_limit_and_rank_deficient_drivers_preserve_prices_and_risks() {
    let mut c = Case::new();
    c.rates = HullWhite1Factor::new(0.0, vec![0.0], vec![0.0]).unwrap();
    c.configs = two_factor_configs(0.0);
    let p = c.compile(1).unwrap();
    let risk = p.evaluate_aad(MultiAssetRiskConfig::default()).unwrap();
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
    .evaluate_aad(MultiAssetRiskConfig::default())
    .unwrap();
    near(risk.price.value().get(), direct.price.value().get(), 1e-12);
    for i in 0..2 {
        near(
            risk.risks[i].delta.value().get(),
            direct.risks[i].delta.value().get(),
            1e-12,
        );
        assert!(
            risk.risks[i]
                .hull_white_lsv
                .as_ref()
                .unwrap()
                .forward_log_density_adjoints
                .iter()
                .all(|v| *v == 0.0)
        );
    }
    // Rank-one Brownian matrix, but differing kernels need additional exact normals.
    c.configs = vec![
        Some(
            MultiAssetLsv2FactorConfig {
                factor: Bergomi2Factor::new([1.0, 0.15], 0.0, 0.4, [1.0, 1.0], 1.0).unwrap(),
                particles: LsvParticleConfig::new(512, 401, 0.8, 2.0, true).unwrap()
            }
            .into()
        );
        2
    ];
    c.correlation = corr(2, 1.0);
    c.rate_correlations = vec![1.0; 6];
    c.full = Some(vec![vec![vec![1.0; 7]; 7]]);
    c.rates = HullWhite1Factor::new(0.1, vec![0.0], vec![0.001]).unwrap();
    let p = c.compile(1).unwrap();
    assert_eq!(p.random_factor_count(), 8);
    assert_eq!(p.lsv_driver_correlations()[0].diagnostics().rank, 1);
    assert!(
        p.evaluate_aad(MultiAssetRiskConfig::default())
            .unwrap()
            .price
            .value()
            .get()
            .is_finite()
    );
}

#[test]
fn hw_quote_refinement_vega_kt_mc_trace_and_mixed_factors_are_explicit() {
    let mut c = Case::new();
    c.full = None;
    c.step = 0.25;
    let quote = |v: Vec<f64>| {
        HullWhiteLsvTarget::from_market_iv(
            MarketIvSurface::new(vec![0.5, 1.0], vec![-0.5, 0.0, 0.5], v).unwrap(),
            vec![0.0, 0.5, 1.0],
            vec![-0.35, 0.0, 0.35],
            1e-8,
            4.0,
        )
        .unwrap()
    };
    c.targets = vec![Some(quote(vec![0.28; 6])); 2];
    c.configs[1] = Some(lsv_config(1.3, 0.25, -0.35, 402, true).into());
    c.rate_correlations = vec![0.2, -0.1, -0.05, 0.02, 0.03];
    let p = c.compile(1).unwrap();
    assert_eq!(p.random_factor_count(), 7);
    assert_eq!(p.lsv_volatility_factor_counts(), vec![2, 1]);
    let r = p.evaluate_aad(MultiAssetRiskConfig::default()).unwrap();
    for i in 0..2 {
        let risk = r.risks[i].hull_white_lsv.as_ref().unwrap();
        assert_eq!(
            r.risks[i].lsv_local_variance.as_ref().unwrap().time_nodes,
            vec![0.0, 0.25, 0.5, 0.75, 1.0]
        );
        for j in 0..6 {
            let bumped = |h| {
                let mut b = c.clone();
                let mut q = vec![0.28; 6];
                q[j] += h;
                b.targets[i] = Some(quote(q));
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
    c.configs[0] = Some(
        MultiAssetLsv2FactorConfig {
            factor: Bergomi2Factor::new([4.0, 0.35], 0.3, 0.3, [-0.65, -0.25], 0.5).unwrap(),
            particles: LsvParticleConfig::new(512, 401, 0.8, 2.0, false).unwrap(),
        }
        .into(),
    );
    let p = c.compile(1).unwrap();
    assert!(p.evaluate().is_ok());
    assert!(p.evaluate_aad(MultiAssetRiskConfig::default()).is_err());
    c.targets[1] = None;
    c.configs[1] = None;
    c.rate_correlations.pop();
    let p = c.compile(1).unwrap();
    assert_eq!(p.lsv_volatility_factor_counts(), vec![2, 0]);
    assert!(p.evaluate().is_ok());
}
