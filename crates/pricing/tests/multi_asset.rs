use pricing::Estimate;
use pricing::core::{
    CurrencyId, CurveId, Date, DayCountConvention, EventId, PositiveF64, UnderlyingId,
};
use pricing::market::{
    DividendEvent, DividendQuote, EquityForward, EquityMarket, LogLinearDiscountCurve,
};
use pricing::mc::{EngineConfig, ExecutionPolicy, PseudoMcConfig, RqmcConfig, VarianceReduction};
use pricing::models::{BlackScholesSpec, LocalVolatilitySpec, ModelSpec};
use pricing::multi_asset::*;
use pricing::product::{CompactC2Smoothing, OptionSide, SourceGraphBuilder, SourceOpcode};
use pricing_numerics::{standard_normal_cdf as cdf, standard_normal_pdf as pdf};
use std::sync::Arc;
fn d(s: &str) -> Date {
    s.parse().unwrap()
}
fn today() -> Date {
    d("2026-01-01")
}
fn expiry() -> Date {
    d("2027-01-01")
}
fn u(i: u32) -> UnderlyingId {
    UnderlyingId::new(i)
}
fn tol() -> CorrelationToleranceConfig {
    CorrelationToleranceConfig {
        symmetry_abs_tol: 1e-12,
        diagonal_abs_tol: 1e-12,
        psd_abs_tol: 1e-12,
        psd_rel_tol: 1e-12,
        zero_pivot_abs_tol: 1e-12,
        zero_pivot_rel_tol: 1e-12,
    }
}
fn curve(id: u32, r: f64) -> Arc<LogLinearDiscountCurve> {
    Arc::new(
        LogLinearDiscountCurve::new(
            CurveId::new(id),
            vec![0.0, 2.0],
            vec![1.0, (-2.0 * r).exp()],
        )
        .unwrap(),
    )
}
fn market(i: u32, spot: f64, r: f64, q: f64, divs: Vec<DividendEvent>) -> EquityMarket {
    let f = EquityForward::with_discrete_dividends(
        u(i),
        PositiveF64::new(spot, "spot").unwrap(),
        curve(1, r),
        curve(100 + i, q),
        divs,
    )
    .unwrap();
    EquityMarket::new(CurrencyId::new(1), f)
}
fn bs(v: f64) -> ModelSpec {
    ModelSpec::BlackScholes(BlackScholesSpec::new(v).unwrap())
}
fn lv(values: Vec<f64>) -> ModelSpec {
    ModelSpec::LocalVolatility(
        LocalVolatilitySpec::from_explicit_grid(
            vec![0.0, 0.5, 1.0],
            vec![-0.8, 0.0, 0.8],
            values,
            1e-5,
            2.0,
        )
        .unwrap(),
    )
}
fn rqmc(points: u64, bridge: bool) -> EngineConfig {
    EngineConfig::RandomizedQuasiMonteCarlo(
        RqmcConfig::new(points, 8, 702, VarianceReduction::new(true, bridge)).unwrap(),
    )
}
fn pseudo() -> EngineConfig {
    EngineConfig::PseudoMonteCarlo(
        PseudoMcConfig::new(511, 4096, VarianceReduction::new(true, true)).unwrap(),
    )
}
fn corr(n: usize, rho: f64) -> CorrelationTermStructure {
    CorrelationTermStructure::new(
        (1..=n as u32).map(u).collect(),
        vec![(
            today(),
            (0..n)
                .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
                .collect(),
        )],
        tol(),
    )
    .unwrap()
}
fn compile(
    product: MultiAssetProduct,
    markets: Vec<EquityMarket>,
    models: Vec<ModelSpec>,
    c: CorrelationTermStructure,
    e: EngineConfig,
    workers: u32,
) -> MultiAssetPricingPlan {
    MultiAssetPricingPlan::compile(
        today(),
        product,
        markets,
        models,
        c,
        e,
        ExecutionPolicy::new(workers, Some(256)).unwrap(),
        0.25,
    )
    .unwrap()
}
fn basket(weights: &[f64], strike: f64, smooth: Option<f64>) -> MultiAssetProduct {
    MultiAssetProduct::basket(
        CurrencyId::new(1),
        weights
            .iter()
            .enumerate()
            .map(|(i, &weight)| BasketComponent {
                underlying: u(i as u32 + 1),
                weight,
                scale: 1.0,
            })
            .collect(),
        OptionSide::Call,
        strike,
        1.0,
        expiry(),
        expiry(),
        smooth.map(|h| CompactC2Smoothing::new(h).unwrap()),
    )
    .unwrap()
}
fn near(x: f64, y: f64, t: f64) {
    assert!((x - y).abs() <= t, "{x} vs {y}, tolerance {t}");
}
fn statistical(e: Estimate, expected: f64, floor: f64) {
    near(
        e.value().get(),
        expected,
        7.0 * e.standard_error().get() + floor,
    );
}
fn bump_market(m: &EquityMarket, spot: f64) -> EquityMarket {
    EquityMarket::new(
        m.currency(),
        m.forward()
            .with_spot(PositiveF64::new(spot, "spot").unwrap())
            .unwrap(),
    )
}

#[test]
fn one_asset_and_perfectly_correlated_duplicates_match_black_scholes() {
    let m = market(1, 100.0, 0.03, 0.01, vec![]);
    let v = 0.2;
    let d1 = ((100f64 / 105.0).ln() + 0.02 + 0.5 * v * v) / v;
    let d2 = d1 - v;
    let p = compile(
        basket(&[1.0], 105.0, None),
        vec![m],
        vec![bs(v)],
        corr(1, 0.0),
        rqmc(8192, true),
        2,
    );
    let r = p
        .evaluate_aad(MultiAssetRiskConfig {
            gamma_relative_bump: Some(0.01),
        })
        .unwrap();
    statistical(
        r.price,
        100.0 * (-0.01f64).exp() * cdf(d1) - 105.0 * (-0.03f64).exp() * cdf(d2),
        0.002,
    );
    statistical(r.risks[0].delta, (-0.01f64).exp() * cdf(d1), 0.0003);
    statistical(
        r.risks[0].bs_vega.unwrap(),
        100.0 * (-0.01f64).exp() * pdf(d1),
        0.004,
    );
    statistical(
        r.gamma[0][0],
        (-0.01f64).exp() * pdf(d1) / (100.0 * v),
        0.0003,
    );
    let duplicated = compile(
        basket(&[0.5, 0.5], 105.0, None),
        vec![
            market(1, 100.0, 0.03, 0.01, vec![]),
            market(2, 100.0, 0.03, 0.01, vec![]),
        ],
        vec![bs(v), bs(v)],
        corr(2, 1.0),
        rqmc(8192, true),
        1,
    )
    .evaluate_aad(MultiAssetRiskConfig::default())
    .unwrap();
    near(duplicated.price.value().get(), r.price.value().get(), 1e-12);
    near(
        duplicated.risks[0].delta.value().get() + duplicated.risks[1].delta.value().get(),
        r.risks[0].delta.value().get(),
        1e-13,
    );
}
#[test]
fn time_dependent_correlation_matches_margrabe_with_and_without_bridge() {
    let change = d("2026-07-02");
    let t = DayCountConvention::Act365F.year_fraction(today(), change);
    let c = CorrelationTermStructure::new(
        vec![u(1), u(2)],
        vec![
            (d("2025-12-01"), vec![vec![1.0, 1.0], vec![1.0, 1.0]]),
            (change, vec![vec![1.0, -0.5], vec![-0.5, 1.0]]),
        ],
        tol(),
    )
    .unwrap();
    let variance = 0.2f64.powi(2) + 0.3f64.powi(2) - 2.0 * 0.2 * 0.3 * (t - 0.5 * (1.0 - t));
    let sigma = variance.sqrt();
    let d1 = ((100f64 / 95.0).ln() + (-0.01 + 0.02) + variance / 2.0) / sigma;
    let expected = 100.0 * (-0.01f64).exp() * cdf(d1) - 95.0 * (-0.02f64).exp() * cdf(d1 - sigma);
    for bridge in [false, true] {
        let p = compile(
            basket(&[1.0, -1.0], 0.0, None),
            vec![
                market(1, 100.0, 0.04, 0.01, vec![]),
                market(2, 95.0, 0.04, 0.02, vec![]),
            ],
            vec![bs(0.2), bs(0.3)],
            c.clone(),
            rqmc(8192, bridge),
            2,
        );
        let node = p.time_nodes().iter().position(|x| *x == t).unwrap();
        assert_eq!(p.correlation_entry_indices()[node - 1], 0);
        assert_eq!(p.correlation_entry_indices()[node], 1);
        statistical(p.evaluate().unwrap().price, expected, 0.006);
    }
}
#[test]
fn first_order_and_cross_gamma_follow_cash_fixed_spot_bumps() {
    let div = DividendEvent::new(EventId::new(1), 0.4, DividendQuote::FixedCash(3.0)).unwrap();
    let markets = vec![
        market(1, 100.0, 0.02, 0.01, vec![div]),
        market(2, 95.0, 0.02, 0.0, vec![]),
    ];
    let models = vec![bs(0.22), bs(0.3)];
    let prod = basket(&[0.6, 0.4], 98.0, Some(3.0));
    let engine = rqmc(1024, true);
    let p = compile(
        prod.clone(),
        markets.clone(),
        models.clone(),
        corr(2, 0.4),
        engine,
        2,
    );
    let base = p
        .evaluate_aad(MultiAssetRiskConfig {
            gamma_relative_bump: Some(0.002),
        })
        .unwrap();
    assert_eq!(p.evaluate().unwrap().price, base.price);
    for j in 0..2 {
        let mut up = markets.clone();
        let mut down = markets.clone();
        let spot = markets[j].forward().spot().get();
        let h = 0.002 * spot;
        up[j] = bump_market(&markets[j], spot + h);
        down[j] = bump_market(&markets[j], spot - h);
        let plus = compile(prod.clone(), up, models.clone(), corr(2, 0.4), engine, 1)
            .evaluate_aad(MultiAssetRiskConfig::default())
            .unwrap();
        let minus = compile(prod.clone(), down, models.clone(), corr(2, 0.4), engine, 1)
            .evaluate_aad(MultiAssetRiskConfig::default())
            .unwrap();
        for i in 0..2 {
            near(
                base.gamma[i][j].value().get(),
                (plus.risks[i].delta.value().get() - minus.risks[i].delta.value().get())
                    / (2.0 * h),
                1e-11,
            );
        }
        near(
            base.risks[j].delta.value().get(),
            (plus.price.value().get() - minus.price.value().get()) / (2.0 * h),
            2e-4,
        );
        let mut up = models.clone();
        let mut down = models.clone();
        let vol = if j == 0 { 0.22 } else { 0.3 };
        up[j] = bs(vol + 1e-5);
        down[j] = bs(vol - 1e-5);
        let pu = compile(prod.clone(), markets.clone(), up, corr(2, 0.4), engine, 1)
            .evaluate()
            .unwrap()
            .price
            .value()
            .get();
        let pd = compile(prod.clone(), markets.clone(), down, corr(2, 0.4), engine, 1)
            .evaluate()
            .unwrap()
            .price
            .value()
            .get();
        near(
            base.risks[j].bs_vega.unwrap().value().get(),
            (pu - pd) / 2e-5,
            2e-5,
        );
    }
}
#[test]
fn mixed_bs_lv_reverses_all_surface_nodes_and_preserves_spot_grid_anchor() {
    let values = vec![0.06, 0.04, 0.05, 0.055, 0.045, 0.06, 0.065, 0.055, 0.07];
    let models = vec![lv(values.clone()), bs(0.3)];
    let markets = vec![
        market(1, 100.0, 0.03, 0.01, vec![]),
        market(2, 90.0, 0.03, 0.02, vec![]),
    ];
    let prod = basket(&[0.4, 0.6], 93.0, Some(3.0));
    let e = rqmc(256, true);
    let risk = compile(
        prod.clone(),
        markets.clone(),
        models.clone(),
        corr(2, -0.3),
        e,
        2,
    )
    .evaluate_aad(MultiAssetRiskConfig::default())
    .unwrap();
    assert!(risk.risks[0].bs_vega.is_none());
    assert_eq!(risk.risks[0].local_variance.len(), 9);
    for k in 0..9 {
        let mut up = values.clone();
        let mut down = values.clone();
        up[k] += 1e-6;
        down[k] -= 1e-6;
        let pu = compile(
            prod.clone(),
            markets.clone(),
            vec![lv(up), bs(0.3)],
            corr(2, -0.3),
            e,
            1,
        )
        .evaluate()
        .unwrap()
        .price
        .value()
        .get();
        let pd = compile(
            prod.clone(),
            markets.clone(),
            vec![lv(down), bs(0.3)],
            corr(2, -0.3),
            e,
            1,
        )
        .evaluate()
        .unwrap()
        .price
        .value()
        .get();
        near(
            risk.risks[0].local_variance[k].value().get(),
            (pu - pd) / 2e-6,
            1e-4,
        );
    }
    let mut up = markets.clone();
    let mut down = markets.clone();
    up[0] = bump_market(&markets[0], 100.001);
    down[0] = bump_market(&markets[0], 99.999);
    let pu = compile(prod.clone(), up, models.clone(), corr(2, -0.3), e, 1)
        .evaluate()
        .unwrap()
        .price
        .value()
        .get();
    let pd = compile(prod, down, models, corr(2, -0.3), e, 1)
        .evaluate()
        .unwrap()
        .price
        .value()
        .get();
    near(risk.risks[0].delta.value().get(), (pu - pd) / 0.002, 1e-6);
}
#[test]
fn flat_local_volatility_reproduces_bs_pathwise_price_and_vega_chain_rule() {
    let m = vec![
        market(1, 100.0, 0.02, 0.01, vec![]),
        market(2, 90.0, 0.02, 0.0, vec![]),
    ];
    let p = basket(&[0.5, 0.5], 95.0, Some(2.0));
    let e = rqmc(256, true);
    let b = compile(
        p.clone(),
        m.clone(),
        vec![bs(0.2), bs(0.3)],
        corr(2, 0.7),
        e,
        1,
    )
    .evaluate_aad(MultiAssetRiskConfig::default())
    .unwrap();
    let l = compile(p, m, vec![lv(vec![0.04; 9]), bs(0.3)], corr(2, 0.7), e, 2)
        .evaluate_aad(MultiAssetRiskConfig::default())
        .unwrap();
    near(b.price.value().get(), l.price.value().get(), 1e-12);
    near(
        b.risks[0].delta.value().get(),
        l.risks[0].delta.value().get(),
        1e-12,
    );
    near(
        b.risks[0].bs_vega.unwrap().value().get(),
        l.risks[0]
            .local_variance
            .iter()
            .map(|x| x.value().get() * 0.4)
            .sum(),
        1e-10,
    );
}
#[test]
fn three_assets_and_cross_gamma_have_independent_product_moment_reference() {
    let mut b = SourceGraphBuilder::new();
    let mut nodes = Vec::new();
    for i in 1..=3 {
        nodes.push(
            b.push(SourceOpcode::TerminalSpot {
                underlying: u(i),
                observation_date: expiry(),
            })
            .unwrap(),
        );
    }
    let ab = b
        .push(SourceOpcode::Multiply {
            left: nodes[0],
            right: nodes[1],
        })
        .unwrap();
    let abc = b
        .push(SourceOpcode::Multiply {
            left: ab,
            right: nodes[2],
        })
        .unwrap();
    let product = MultiAssetProduct::from_graph(
        CurrencyId::new(1),
        vec![u(1), u(2), u(3)],
        b.finish(vec![abc]),
        vec![expiry()],
    )
    .unwrap();
    let markets = (1..=3).map(|i| market(i, 1.0, 0.0, 0.0, vec![])).collect();
    let risk = compile(
        product,
        markets,
        vec![bs(0.2), bs(0.3), bs(0.4)],
        corr(3, 0.3),
        rqmc(4096, true),
        3,
    )
    .evaluate_aad(MultiAssetRiskConfig {
        gamma_relative_bump: Some(0.001),
    })
    .unwrap();
    let expected = (0.3f64 * (0.2 * 0.3 + 0.2 * 0.4 + 0.3 * 0.4)).exp();
    statistical(risk.price, expected, 2e-4);
    for i in 0..3 {
        statistical(risk.risks[i].delta, expected, 2e-4);
        for j in 0..3 {
            if i == j {
                near(risk.gamma[i][j].value().get(), 0.0, 1e-12);
            } else {
                statistical(risk.gamma[i][j], expected, 2e-4);
            }
        }
    }
}
#[test]
fn independent_dividend_dates_reconstruct_after_expiry_jumps_and_cash_carry() {
    let m = vec![
        market(
            1,
            100.0,
            0.05,
            0.01,
            vec![DividendEvent::new(EventId::new(1), 0.4, DividendQuote::FixedCash(4.0)).unwrap()],
        ),
        market(
            2,
            80.0,
            0.05,
            0.02,
            vec![
                DividendEvent::new(
                    EventId::new(2),
                    1.0,
                    DividendQuote::FixedCashAndProportional {
                        fixed_cash: 2.0,
                        beta: 0.1,
                    },
                )
                .unwrap(),
            ],
        ),
    ];
    let p = compile(
        basket(&[1.0, 1.0], 0.0, None),
        m,
        vec![bs(0.0), bs(0.0)],
        corr(2, 0.5),
        pseudo(),
        1,
    );
    assert!(p.time_nodes().contains(&0.4));
    let expected = ((100.0 * (0.04f64).exp() - 4.0 * (0.04f64 * 0.6).exp())
        + 0.9 * 80.0 * (0.03f64).exp()
        - 2.0)
        * (-0.05f64).exp();
    near(p.evaluate().unwrap().price.value().get(), expected, 1e-11);
}
#[test]
fn replay_is_independent_of_workers_and_rqmc_uncertainty_uses_scrambles() {
    for e in [pseudo(), rqmc(128, true)] {
        let p = basket(&[0.5, 0.5], 95.0, Some(2.0));
        let m = vec![
            market(1, 100.0, 0.01, 0.0, vec![]),
            market(2, 90.0, 0.01, 0.0, vec![]),
        ];
        let models = vec![bs(0.2), bs(0.3)];
        let a = compile(p.clone(), m.clone(), models.clone(), corr(2, 0.2), e, 1);
        let b = compile(p, m, models, corr(2, 0.2), e, 4);
        assert_eq!(a.fingerprint(), b.fingerprint());
        let config = MultiAssetRiskConfig {
            gamma_relative_bump: Some(0.01),
        };
        let r = a.evaluate_aad(config).unwrap();
        let s = b.evaluate_aad(config).unwrap();
        assert_eq!(r.price, s.price);
        assert_eq!(r.risks, s.risks);
        assert_eq!(r.gamma, s.gamma);
        assert_eq!(r.fingerprint, s.fingerprint);
        assert_eq!(
            r.price.effective_sampling_units().get(),
            if matches!(e, EngineConfig::PseudoMonteCarlo(_)) {
                4096
            } else {
                8
            }
        );
        assert_eq!(
            r.evaluated_paths,
            if matches!(e, EngineConfig::PseudoMonteCarlo(_)) {
                8192
            } else {
                2048
            }
        );
    }
}
fn autocall(memory: bool, termination: MemoryTermination) -> AutocallSpec {
    AutocallSpec {
        currency: CurrencyId::new(1),
        components: vec![
            WorstOfComponent {
                underlying: u(1),
                reference_level: 100.0,
            },
            WorstOfComponent {
                underlying: u(2),
                reference_level: 80.0,
            },
        ],
        observations: vec![d("2026-04-01"), d("2026-07-01"), expiry()]
            .into_iter()
            .map(|date| AutocallObservation {
                date,
                payment_date: date,
                coupon_amount: 5.0,
                coupon_level: 0.9,
                call_level: Some(1.1),
                call_coupon_amount: 1.0,
            })
            .collect(),
        notional: 100.0,
        final_barrier: 0.7,
        maturity: expiry(),
        maturity_payment: expiry(),
        memory,
        on_autocall: termination,
        on_maturity: termination,
    }
}
#[test]
fn autocall_exact_cashflows_memory_termination_and_inclusive_barriers() {
    use pricing::product::GraphLimitPolicy;
    for (memory, expected) in [(true, 116.0), (false, 111.0)] {
        let spec = autocall(memory, MemoryTermination::Forfeit);
        let p = spec.product(None).unwrap();
        let tape = p
            .source_graph()
            .compile(GraphLimitPolicy::default())
            .unwrap();
        let values = tape
            .evaluate(|id, date| {
                let performance = if date == spec.observations[0].date {
                    0.8
                } else if date == spec.observations[1].date {
                    1.0
                } else {
                    1.2
                };
                Some(performance * if id == u(1) { 100.0 } else { 80.0 })
            })
            .unwrap();
        near(values.iter().sum(), expected, 1e-12);
    }
    for (policy, expected) in [
        (MemoryTermination::Forfeit, 101.0),
        (MemoryTermination::Pay, 106.0),
    ] {
        let mut spec = autocall(true, policy);
        spec.observations[0].call_level = Some(0.8);
        let tape = spec
            .product(None)
            .unwrap()
            .source_graph()
            .compile(GraphLimitPolicy::default())
            .unwrap();
        let values = tape
            .evaluate(|id, _| Some(0.8 * if id == u(1) { 100.0 } else { 80.0 }))
            .unwrap();
        near(values.iter().sum(), expected, 1e-12);
        assert!(values[1..].iter().all(|x| *x == 0.0));
    }
    for (performance, expected) in [(0.6, 75.0), (0.7, 115.0), (0.9, 115.0)] {
        let mut spec = autocall(true, MemoryTermination::Pay);
        for o in &mut spec.observations {
            o.call_level = None;
        }
        let tape = spec
            .product(None)
            .unwrap()
            .source_graph()
            .compile(GraphLimitPolicy::default())
            .unwrap();
        let values = tape
            .evaluate(|id, _| Some(performance * if id == u(1) { 100.0 } else { 80.0 }))
            .unwrap();
        near(values.iter().sum(), expected, 1e-12);
    }
}
#[test]
fn smoothed_autocall_aad_matches_crn_price_bumps_and_exact_aad_is_rejected() {
    let markets = vec![
        market(1, 100.0, 0.02, 0.0, vec![]),
        market(2, 80.0, 0.02, 0.0, vec![]),
    ];
    let models = vec![bs(0.2), bs(0.25)];
    let e = rqmc(256, true);
    let spec = autocall(true, MemoryTermination::Pay);
    let exact = compile(
        spec.product(None).unwrap(),
        markets.clone(),
        models.clone(),
        corr(2, 0.4),
        e,
        1,
    );
    assert!(exact.evaluate().is_ok());
    assert!(exact.evaluate_aad(MultiAssetRiskConfig::default()).is_err());
    let prod = spec
        .product(Some(CompactC2Smoothing::new(0.05).unwrap()))
        .unwrap();
    let p = compile(
        prod.clone(),
        markets.clone(),
        models.clone(),
        corr(2, 0.4),
        e,
        1,
    );
    let risk = p.evaluate_aad(MultiAssetRiskConfig::default()).unwrap();
    assert_eq!(p.evaluate().unwrap().price, risk.price);
    for j in 0..2 {
        let mut up = markets.clone();
        let mut down = markets.clone();
        let s = markets[j].forward().spot().get();
        up[j] = bump_market(&markets[j], s + 0.001);
        down[j] = bump_market(&markets[j], s - 0.001);
        let pu = compile(prod.clone(), up, models.clone(), corr(2, 0.4), e, 1)
            .evaluate()
            .unwrap()
            .price
            .value()
            .get();
        let pd = compile(prod.clone(), down, models.clone(), corr(2, 0.4), e, 1)
            .evaluate()
            .unwrap()
            .price
            .value()
            .get();
        near(risk.risks[j].delta.value().get(), (pu - pd) / 0.002, 1e-5);
    }
}
#[test]
fn worst_of_references_stay_contractual_under_spot_bumps() {
    let components = vec![
        WorstOfComponent {
            underlying: u(1),
            reference_level: 100.0,
        },
        WorstOfComponent {
            underlying: u(2),
            reference_level: 200.0,
        },
    ];
    let prod = MultiAssetProduct::worst_of(
        CurrencyId::new(1),
        components,
        OptionSide::Call,
        0.5,
        100.0,
        expiry(),
        expiry(),
        None,
    )
    .unwrap();
    let m = vec![
        market(1, 80.0, 0.0, 0.0, vec![]),
        market(2, 190.0, 0.0, 0.0, vec![]),
    ];
    let p = compile(
        prod.clone(),
        m.clone(),
        vec![bs(0.0), bs(0.0)],
        corr(2, 0.0),
        pseudo(),
        1,
    )
    .evaluate_aad(MultiAssetRiskConfig::default())
    .unwrap();
    near(p.price.value().get(), 30.0, 1e-12);
    near(p.risks[0].delta.value().get(), 1.0, 1e-12);
    near(p.risks[1].delta.value().get(), 0.0, 1e-12);
    let mut bumped = m;
    bumped[0] = bump_market(&bumped[0], 81.0);
    let q = compile(
        prod,
        bumped,
        vec![bs(0.0), bs(0.0)],
        corr(2, 0.0),
        pseudo(),
        1,
    )
    .evaluate()
    .unwrap();
    near(q.price.value().get(), 31.0, 1e-12);
}
#[test]
fn validates_dimensions_order_dates_causality_currency_and_nonpositive_paths() {
    let m = vec![
        market(1, 100.0, 0.0, 0.0, vec![]),
        market(2, 100.0, 0.0, 0.0, vec![]),
    ];
    let models = vec![bs(0.2), bs(0.2)];
    let attempt = |p, markets, models, c| {
        MultiAssetPricingPlan::compile(
            today(),
            p,
            markets,
            models,
            c,
            pseudo(),
            ExecutionPolicy::new(1, None).unwrap(),
            0.25,
        )
    };
    let mut swapped = m.clone();
    swapped.swap(0, 1);
    assert!(
        attempt(
            basket(&[0.5, 0.5], 100.0, None),
            swapped,
            models.clone(),
            corr(2, 0.0)
        )
        .is_err()
    );
    assert!(
        attempt(
            basket(&[0.5, 0.5], 100.0, None),
            m.clone(),
            vec![bs(0.2)],
            corr(2, 0.0)
        )
        .is_err()
    );
    let future = CorrelationTermStructure::new(
        vec![u(1), u(2)],
        vec![(d("2026-01-02"), vec![vec![1.0, 0.0], vec![0.0, 1.0]])],
        tol(),
    )
    .unwrap();
    assert!(
        attempt(
            basket(&[0.5, 0.5], 100.0, None),
            m.clone(),
            models.clone(),
            future
        )
        .is_err()
    );
    let mut different = m.clone();
    different[1] = market(2, 100.0, 0.01, 0.0, vec![]);
    assert!(
        attempt(
            basket(&[0.5, 0.5], 100.0, None),
            different,
            models.clone(),
            corr(2, 0.0)
        )
        .is_err()
    );
    assert!(
        CorrelationTermStructure::new(
            vec![u(1), u(1)],
            vec![(today(), vec![vec![1.0, 0.0], vec![0.0, 1.0]])],
            tol()
        )
        .is_err()
    );
    let mut b = SourceGraphBuilder::new();
    let node = b
        .push(SourceOpcode::TerminalSpot {
            underlying: u(1),
            observation_date: expiry(),
        })
        .unwrap();
    let p = MultiAssetProduct::from_graph(
        CurrencyId::new(1),
        vec![u(1), u(2)],
        b.finish(vec![node]),
        vec![d("2026-06-01")],
    )
    .unwrap();
    assert!(attempt(p, m.clone(), models, corr(2, 0.0)).is_err());
    let div = DividendEvent::new(EventId::new(1), 0.5, DividendQuote::FixedCash(99.0)).unwrap();
    let dangerous = vec![
        market(1, 100.0, 0.0, 0.0, vec![div]),
        market(2, 100.0, 0.0, 0.0, vec![]),
    ];
    let p = attempt(
        basket(&[1.0, 1.0], 0.0, None),
        dangerous,
        vec![bs(0.5), bs(0.2)],
        corr(2, 0.0),
    )
    .unwrap();
    assert!(p.evaluate().is_err());
}

#[test]
fn pre_and_post_observations_share_a_state_on_each_assets_ex_date() {
    let mut graph = SourceGraphBuilder::new();
    let mut total = graph.literal(0.0).unwrap();
    for i in 1..=2 {
        let pre = graph
            .push(SourceOpcode::PreDividendSpot {
                underlying: u(i),
                observation_date: expiry(),
            })
            .unwrap();
        let post = graph
            .push(SourceOpcode::TerminalSpot {
                underlying: u(i),
                observation_date: expiry(),
            })
            .unwrap();
        let paid = graph
            .push(SourceOpcode::Subtract {
                left: pre,
                right: post,
            })
            .unwrap();
        total = graph
            .push(SourceOpcode::Add {
                left: total,
                right: paid,
            })
            .unwrap();
    }
    let product = MultiAssetProduct::from_graph(
        CurrencyId::new(1),
        vec![u(1), u(2)],
        graph.finish(vec![total]),
        vec![expiry()],
    )
    .unwrap();
    let markets = vec![
        market(
            1,
            100.0,
            0.0,
            0.0,
            vec![
                DividendEvent::new(
                    EventId::new(1),
                    1.0,
                    DividendQuote::FixedCashAndProportional {
                        fixed_cash: 2.0,
                        beta: 0.1,
                    },
                )
                .unwrap(),
            ],
        ),
        market(
            2,
            80.0,
            0.0,
            0.0,
            vec![DividendEvent::new(EventId::new(2), 1.0, DividendQuote::FixedCash(3.0)).unwrap()],
        ),
    ];
    let result = compile(
        product,
        markets,
        vec![bs(0.2), bs(0.3)],
        corr(2, 0.4),
        rqmc(2048, true),
        2,
    )
    .evaluate_aad(MultiAssetRiskConfig::default())
    .unwrap();
    statistical(result.price, 15.0, 0.001);
    statistical(result.risks[0].delta, 0.1, 0.00001);
    near(result.risks[1].delta.value().get(), 0.0, 1e-14);
    near(result.risks[1].bs_vega.unwrap().value().get(), 0.0, 1e-12);
}

#[test]
fn future_cash_does_not_reserve_spot_or_extend_the_simulated_horizon() {
    let markets = vec![
        market(1, 100.0, 0.02, 0.01, vec![]),
        market(2, 90.0, 0.02, 0.0, vec![]),
    ];
    let mut future = markets.clone();
    future[0] = market(
        1,
        100.0,
        0.02,
        0.01,
        vec![DividendEvent::new(EventId::new(1), 1.5, DividendQuote::FixedCash(20.0)).unwrap()],
    );
    let product = basket(&[0.5, 0.5], 95.0, Some(2.0));
    let first = compile(
        product.clone(),
        markets,
        vec![bs(0.2), bs(0.3)],
        corr(2, 0.3),
        rqmc(128, true),
        1,
    );
    let second = compile(
        product,
        future,
        vec![bs(0.2), bs(0.3)],
        corr(2, 0.3),
        rqmc(128, true),
        2,
    );
    assert_eq!(first.time_nodes(), second.time_nodes());
    assert_eq!(second.time_nodes().last(), Some(&1.0));
    let a = first.evaluate_aad(MultiAssetRiskConfig::default()).unwrap();
    let b = second
        .evaluate_aad(MultiAssetRiskConfig::default())
        .unwrap();
    assert_eq!(a.price, b.price);
    assert_eq!(a.risks, b.risks);
}
