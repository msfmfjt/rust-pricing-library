use super::*;
use pricing::mc::hull_white::HullWhiteLsvTarget;
use pricing::models::{HullWhite1Factor, RoughBergomi};

fn particles() -> LsvParticleConfig {
    LsvParticleConfig::new(64, 718, 0.5, 2.0, true).unwrap()
}

fn one_factor() -> MultiAssetLsvConfig {
    MultiAssetLsvConfig {
        factor: Bergomi1Factor::new(0.7, 0.2, -0.4).unwrap(),
        particles: particles(),
    }
}

fn target(volatility: f64) -> HullWhiteLsvTarget {
    HullWhiteLsvTarget::flat(
        volatility,
        vec![0.0, 0.5, 1.0],
        vec![-0.8, 0.0, 0.8],
        1e-5,
        2.0,
    )
    .unwrap()
}

fn hw(target: Option<HullWhiteLsvTarget>, factors: usize) -> MultiAssetHullWhiteConfig {
    MultiAssetHullWhiteConfig {
        rate_model: HullWhite1Factor::new(0.1, vec![0.0], vec![0.0]).unwrap(),
        rate_correlations: vec![0.0; 1 + factors],
        lsv_targets: vec![target],
    }
}

fn configured(
    markets: Vec<EquityMarket>,
    models: Vec<ModelSpec>,
    configs: Vec<Option<MultiAssetBergomiLsvConfig>>,
    rates: Option<MultiAssetHullWhiteConfig>,
) -> Result<MultiAssetPricingPlan, MultiAssetError> {
    let policy = ExecutionPolicy::new(1, Some(16)).unwrap();
    match rates {
        Some(rates) => MultiAssetPricingPlan::compile_with_hull_white(
            today(),
            basket(&[1.0], 100.0, Some(3.0)),
            markets,
            models,
            corr(1, 0.0),
            rqmc(16, true),
            policy,
            0.5,
            configs,
            None,
            rates,
        ),
        None => MultiAssetPricingPlan::compile_with_bergomi_lsv(
            today(),
            basket(&[1.0], 100.0, Some(3.0)),
            markets,
            models,
            corr(1, 0.0),
            rqmc(16, true),
            policy,
            0.5,
            configs,
            None,
        ),
    }
}

fn invalid(result: Result<MultiAssetPricingPlan, MultiAssetError>, expected: &str) {
    assert!(matches!(result, Err(MultiAssetError::Invalid(message)) if message == expected));
}

#[test]
fn legacy_multi_asset_adapters_preserve_prices_risks_and_fingerprints() {
    let markets = vec![
        market(1, 100.0, 0.03, 0.01, vec![]),
        market(2, 90.0, 0.03, 0.02, vec![]),
    ];
    let models = vec![lv(vec![0.04; 9]), bs(0.25)];
    let product = basket(&[0.6, 0.4], 95.0, Some(3.0));
    let policy = ExecutionPolicy::new(1, Some(16)).unwrap();
    for engine in [
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(83, 32, VarianceReduction::new(true, true)).unwrap(),
        ),
        rqmc(16, true),
    ] {
        for calibrated in [false, true] {
            let configs = vec![calibrated.then(one_factor), None];
            let legacy = MultiAssetPricingPlan::compile_with_lsv(
                today(),
                product.clone(),
                markets.clone(),
                models.clone(),
                corr(2, 0.2),
                engine,
                policy,
                0.5,
                configs.clone(),
                None,
            )
            .unwrap();
            let mixed = MultiAssetPricingPlan::compile_with_bergomi_lsv(
                today(),
                product.clone(),
                markets.clone(),
                models.clone(),
                corr(2, 0.2),
                engine,
                policy,
                0.5,
                configs.into_iter().map(|c| c.map(Into::into)).collect(),
                None,
            )
            .unwrap();
            assert_eq!(legacy.fingerprint(), mixed.fingerprint());
            assert_eq!(legacy.time_nodes(), mixed.time_nodes());
            assert_eq!(legacy.random_factor_count(), 2 + usize::from(calibrated));
            assert_eq!(legacy.evaluate().unwrap(), mixed.evaluate().unwrap());
            assert_eq!(
                legacy
                    .evaluate_aad(MultiAssetRiskConfig::default())
                    .unwrap(),
                mixed.evaluate_aad(MultiAssetRiskConfig::default()).unwrap()
            );
            if !calibrated {
                let direct = MultiAssetPricingPlan::compile(
                    today(),
                    product.clone(),
                    markets.clone(),
                    models.clone(),
                    corr(2, 0.2),
                    engine,
                    policy,
                    0.5,
                )
                .unwrap();
                assert_eq!(direct.evaluate().unwrap(), mixed.evaluate().unwrap());
            }
        }
    }
}

#[test]
fn composition_rejections_preserve_request_product_and_target_order() {
    let markets = || vec![market(1, 100.0, 0.03, 0.01, vec![])];
    let rough: MultiAssetBergomiLsvConfig = MultiAssetRoughLsvConfig {
        factor: RoughBergomi::new(0.2, 0.3, -0.4).unwrap(),
        particles: particles(),
    }
    .into();
    invalid(
        configured(vec![], vec![bs(0.2)], vec![], None),
        "market/model dimensions differ",
    );
    invalid(
        configured(markets(), vec![bs(0.2)], vec![], None),
        "LSV configuration count must equal asset count",
    );
    // Rough without HW wins over both an unsuitable target model and a wrong
    // underlying order. With HW, the target-model check is the first failure.
    let wrong_order = || vec![market(2, 100.0, 0.03, 0.01, vec![])];
    invalid(
        configured(
            wrong_order(),
            vec![bs(0.2)],
            vec![Some(rough.clone())],
            None,
        ),
        "rough-LSV requires the shared HW adapter and paired targets; use a zero-volatility HW model for deterministic rates",
    );
    invalid(
        configured(
            wrong_order(),
            vec![bs(0.2)],
            vec![Some(rough)],
            Some(hw(None, 0)),
        ),
        "each LSV asset requires a LocalVolatility target model",
    );
    invalid(
        configured(
            wrong_order(),
            vec![lv(vec![0.04; 9])],
            vec![Some(one_factor().into())],
            Some(hw(None, 0)),
        ),
        "product, market and correlation underlying order must match",
    );
    invalid(
        configured(
            markets(),
            vec![lv(vec![0.04; 9])],
            vec![Some(one_factor().into())],
            Some(hw(None, 0)),
        ),
        "HW target/rate-correlation dimensions or values are invalid",
    );
    invalid(
        configured(
            markets(),
            vec![lv(vec![0.04; 9])],
            vec![Some(one_factor().into())],
            Some(hw(None, 1)),
        ),
        "every HW LSV asset requires a paired variance/density target",
    );
    invalid(
        configured(
            markets(),
            vec![lv(vec![0.04; 9])],
            vec![Some(one_factor().into())],
            Some(hw(Some(target(0.3)), 1)),
        ),
        "HW paired target and model grid must match",
    );
    invalid(
        configured(
            markets(),
            vec![bs(0.2)],
            vec![None],
            Some(hw(Some(target(0.2)), 0)),
        ),
        "HW target requires an LSV configuration",
    );
    invalid(
        configured(
            markets(),
            vec![lv(vec![0.04; 9])],
            vec![None],
            Some(hw(None, 0)),
        ),
        "LocalVolatility under HW requires an LSV configuration and paired target; use zero vol-of-vol for the local-vol limit",
    );
}

#[test]
fn zero_rate_volatility_keeps_hw_and_rough_coordinates() {
    for (config, count) in [
        (None, 3),
        (Some(one_factor().into()), 4),
        (
            Some(
                MultiAssetLsv2FactorConfig {
                    factor: Bergomi2Factor::new([3.0, 0.2], 0.3, 0.4, [-0.4, -0.2], 0.3).unwrap(),
                    particles: particles(),
                }
                .into(),
            ),
            5,
        ),
        (
            Some(
                MultiAssetRoughLsvConfig {
                    factor: RoughBergomi::new(0.2, 0.3, -0.4).unwrap(),
                    particles: particles(),
                }
                .into(),
            ),
            5,
        ),
    ] {
        let paired = config.as_ref().map(|_| target(0.2));
        let model = paired.as_ref().map_or_else(
            || bs(0.2),
            |t| {
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
            },
        );
        let factors = config
            .as_ref()
            .map_or(0, MultiAssetBergomiLsvConfig::factor_count);
        let plan = configured(
            vec![market(1, 100.0, 0.03, 0.01, vec![])],
            vec![model],
            vec![config],
            Some(hw(paired, factors)),
        )
        .unwrap();
        assert_eq!(plan.random_factor_count(), count);
        assert_eq!(plan.time_nodes(), [0.0, 0.5, 1.0]);
        let price = plan.evaluate().unwrap();
        let risk = plan.evaluate_aad(MultiAssetRiskConfig::default()).unwrap();
        assert_eq!(price.price, risk.price);
        assert!(price.price.value().get().is_finite());
    }
}
