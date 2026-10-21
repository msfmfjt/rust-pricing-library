use pricing::core::{CurrencyId, CurveId, Date, PositiveF64, UnderlyingId};
use pricing::market::*;
use pricing::mc::{
    EngineConfig, ExecutionPolicy, PseudoMcConfig, RqmcConfig, VarianceReduction,
    lsv::LsvParticleConfig,
};
use pricing::models::*;
use pricing::multi_asset::*;
use pricing::product::{CompactC2Smoothing, OptionSide};
use pricing::*;
use serde_json::{Value, json};
use std::sync::Arc;
fn policy() -> ExecutionPolicy {
    ExecutionPolicy::new(1, Some(16)).unwrap()
}
fn source() -> PricingSpecification {
    let mut v: Value = serde_json::from_str(include_str!(
        "../../../fixtures/specification-v1/black_scholes.golden.json"
    ))
    .unwrap();
    v["engine"]["independent_sampling_units"] = json!(32);
    v["numerics"]["worker_threads"] = json!(1);
    v["numerics"]["reduction_block_size"] = json!(16);
    parse_specification_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn dividend_source() -> PricingSpecification {
    let mut v: Value = serde_json::from_str(&specification_to_json(&source()).unwrap()).unwrap();
    v["market"]["discrete_dividends"] = json!([
        {"event_id":1,"ex_time":0.25,"quote":{"type":"fixed_cash","amount":2.0}},
        {"event_id":2,"ex_time":0.75,"quote":{"type":"fixed_cash","amount":3.0}}
    ]);
    parse_specification_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap()
}
fn replay(t: &PricingTask) -> PricingTaskResult {
    let json = pricing_task_to_json(t).unwrap();
    let restored = parse_pricing_task_json(json.as_bytes(), JsonLimits::DEFAULT).unwrap();
    assert_eq!(json, pricing_task_to_json(&restored).unwrap());
    assert_eq!(t.fingerprint().unwrap(), restored.fingerprint().unwrap());
    let p = t.compile().unwrap();
    let q = restored.compile().unwrap();
    assert_eq!(p.plan_fingerprint(), q.plan_fingerprint());
    assert_eq!(p.capability().status, CapabilityStatus::Supported);
    let r = p.evaluate().unwrap();
    let replayed = q.evaluate().unwrap();
    assert_eq!(r, replayed);
    let saved = valuation_report_to_json(&r.report).unwrap();
    assert_eq!(
        r.report,
        parse_valuation_report_json(saved.as_bytes(), JsonLimits::DEFAULT).unwrap()
    );
    r
}
fn factor(i: usize) -> VolatilityFactorSpec {
    match i {
        0 => VolatilityFactorSpec::Bergomi(Bergomi1Factor::new(0.7, 0.2, -0.3).unwrap()),
        1 => VolatilityFactorSpec::BergomiTwoFactor(
            Bergomi2Factor::new([0.4, 1.2], 0.2, 0.4, [-0.2, -0.3], 0.1).unwrap(),
        ),
        _ => VolatilityFactorSpec::RoughBergomi(RoughBergomi::new(0.2, 0.15, -0.3).unwrap()),
    }
}
fn particles(retain: bool) -> LsvParticleConfig {
    LsvParticleConfig::new(32, 74, 0.5, 2.0, retain).unwrap()
}
fn lv() -> LocalVolatilitySpec {
    LocalVolatilitySpec::from_explicit_grid(
        vec![0.0, 0.5, 1.0],
        vec![-0.8, 0.0, 0.8],
        vec![0.04; 9],
        1e-8,
        4.0,
    )
    .unwrap()
}
#[test]
fn standard_report_keeps_native_risks_and_wire_domains() {
    let mut v: Value = serde_json::from_str(&specification_to_json(&source()).unwrap()).unwrap();
    v["risk"]["delta"] = json!(true);
    v["risk"]["vega"] = json!(true);
    let s =
        parse_specification_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let old = s.compile().unwrap().evaluate().unwrap();
    let t = PricingTask::new(s, PricingOperation::Evaluate).unwrap();
    let r = replay(&t);
    assert_eq!(r.native_result, PricingTaskOutput::SingleAsset(old));
    assert_eq!(r.report.risks().len(), 4);
    assert!(
        r.report
            .risks()
            .iter()
            .any(|v| v.unit == SensitivityUnit::CurrencyPerVolatilityPoint)
    );
    assert!(
        parse_request_json(
            pricing_task_to_json(&t).unwrap().as_bytes(),
            JsonLimits::DEFAULT
        )
        .is_err()
    );
}
#[test]
fn hybrid_and_lsv_risk_contracts_preserve_missing_uncertainty() {
    for i in 0..3 {
        let s = source()
            .with_model(
                PricingModelSpec::new(
                    EquityModelSpec::StochasticVolatility {
                        initial_volatility: 0.2,
                        factor: factor(i),
                    },
                    RateModelSpec::Deterministic,
                ),
                Some(0.5),
            )
            .unwrap();
        let r = replay(
            &PricingTask::new(
                s,
                PricingOperation::Aad {
                    gamma_relative_bump: None,
                },
            )
            .unwrap(),
        );
        assert!(!r.report.risks().is_empty());
        assert!(r.report.risks().iter().all(|r| r.standard_errors.is_none()));
        let s = source()
            .with_model(
                PricingModelSpec::new(
                    EquityModelSpec::LocalStochasticVolatility {
                        target: lv(),
                        factor: factor(i),
                        particles: particles(true),
                        forward_density: None,
                    },
                    RateModelSpec::Deterministic,
                ),
                None,
            )
            .unwrap();
        let r = replay(&PricingTask::new(s, PricingOperation::LocalVarianceRisk).unwrap());
        assert_eq!(
            r.report.uncertainty_scope(),
            UncertaintyScope::ConditionalOnCalibration
        );
        assert_eq!(r.report.calibration_seeds(), &[74]);
        assert!(r.report.risks()[0].standard_errors.is_none());
    }
}
fn basket_model(rqmc: bool) -> MultiAssetSpecification {
    let date: Date = "2026-01-01".parse().unwrap();
    let expiry: Date = "2027-01-01".parse().unwrap();
    let curve = Arc::new(
        LogLinearDiscountCurve::new(CurveId::new(1), vec![0.0, 2.0], vec![1.0, 0.96]).unwrap(),
    );
    let markets = (1..=2)
        .map(|i| {
            EquityMarket::new(
                CurrencyId::new(1),
                EquityForward::new(
                    UnderlyingId::new(i),
                    PositiveF64::new(100.0, "spot").unwrap(),
                    curve.clone(),
                    curve.clone(),
                ),
            )
        })
        .collect();
    let tol = CorrelationToleranceConfig {
        symmetry_abs_tol: 1e-12,
        diagonal_abs_tol: 1e-12,
        psd_abs_tol: 1e-12,
        psd_rel_tol: 1e-12,
        zero_pivot_abs_tol: 1e-12,
        zero_pivot_rel_tol: 1e-12,
    };
    let correlation = CorrelationTermStructure::new(
        vec![UnderlyingId::new(1), UnderlyingId::new(2)],
        vec![(date, vec![vec![1.0, 0.2], vec![0.2 + 1e-14, 1.0]])],
        tol,
    )
    .unwrap();
    let model = MultiAssetModelSpec::new(
        vec![ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).unwrap()); 2],
        correlation,
    );
    let product = MultiAssetProduct::basket(
        CurrencyId::new(1),
        (1..=2)
            .map(|i| BasketComponent {
                underlying: UnderlyingId::new(i),
                weight: 0.5,
                scale: 1.0,
            })
            .collect(),
        OptionSide::Call,
        100.0,
        1.0,
        expiry,
        expiry,
        Some(CompactC2Smoothing::new(1.0).unwrap()),
    )
    .unwrap()
    .to_product_graph()
    .unwrap();
    let engine = if rqmc {
        EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(16, 4, 82, VarianceReduction::new(true, false)).unwrap(),
        )
    } else {
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(82, 32, VarianceReduction::new(true, false)).unwrap(),
        )
    };
    MultiAssetSpecification::new(date, product, markets, model, engine, policy(), 0.5).unwrap()
}
#[test]
fn basket_full_source_matches_native_aad_and_gamma() {
    for rqmc in [false, true] {
        let s = basket_model(rqmc);
        let m = s.model();
        let native = MultiAssetPricingPlan::compile(
            s.valuation_date(),
            MultiAssetProduct::from_product_graph(s.product().clone()),
            s.markets().to_vec(),
            m.models.clone(),
            m.correlation.clone(),
            s.engine(),
            s.execution(),
            s.maximum_step(),
        )
        .unwrap();
        let config = MultiAssetRiskConfig {
            gamma_relative_bump: Some(0.01),
        };
        let expected = native.evaluate_aad(config).unwrap();
        let t = PricingTask::new(
            s,
            PricingOperation::Aad {
                gamma_relative_bump: config.gamma_relative_bump,
            },
        )
        .unwrap();
        let r = replay(&t);
        assert_eq!(
            r.native_result,
            PricingTaskOutput::MultiAsset(Box::new(expected))
        );
        assert_eq!(
            r.report
                .risks()
                .iter()
                .filter(|r| r.name == "gamma")
                .count(),
            4
        );
    }
}
#[test]
fn stochastic_dividend_all_native_price_families_replay() {
    let d = BuehlerDividendModel::new(0.5, 0.2, 0.1, 0.1).unwrap();
    for i in 0..3 {
        for calibrated in [false, true] {
            let f = factor(i);
            let n = if i == 1 { 2 } else { 1 };
            let equity = if calibrated {
                EquityModelSpec::LocalStochasticVolatility {
                    target: lv(),
                    factor: f.clone(),
                    particles: particles(true),
                    forward_density: None,
                }
            } else {
                EquityModelSpec::StochasticVolatility {
                    initial_volatility: 0.2,
                    factor: f.clone(),
                }
            };
            let base = dividend_source()
                .with_model(
                    PricingModelSpec::new(equity, RateModelSpec::Deterministic),
                    if calibrated { None } else { Some(0.5) },
                )
                .unwrap();
            let s = StochasticDividendSpecification::new(base, d, vec![0.0; n], None, 0.5).unwrap();
            replay(&PricingTask::new(s.clone(), PricingOperation::Evaluate).unwrap());
            let r = replay(
                &PricingTask::new(
                    s,
                    if calibrated {
                        PricingOperation::LocalVarianceRisk
                    } else {
                        PricingOperation::Aad {
                            gamma_relative_bump: None,
                        }
                    },
                )
                .unwrap(),
            );
            assert_eq!(
                r.report.uncertainty_scope(),
                if calibrated {
                    UncertaintyScope::ConditionalOnCalibration
                } else {
                    UncertaintyScope::PricingSampling
                }
            );
        }
    }
    for hybrid in [false, true] {
        let mut base = dividend_source();
        if hybrid {
            let m = base.model().equity.clone();
            base = base
                .with_model(
                    PricingModelSpec::new(
                        m,
                        RateModelSpec::HullWhite {
                            model: HullWhite1Factor::new(0.1, vec![0.0], vec![0.005]).unwrap(),
                            equity_rate_correlation: 0.1,
                            vol_rate_correlations: vec![],
                        },
                    ),
                    Some(0.5),
                )
                .unwrap();
        }
        let request = PricingRequest::new(
            base.valuation_date(),
            base.product().clone(),
            base.market().clone(),
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).unwrap()),
            base.engine(),
            base.risk().clone(),
        )
        .unwrap();
        let expected = if let RateModelSpec::HullWhite {
            model,
            equity_rate_correlation,
            ..
        } = &base.model().rates
        {
            pricing::stochastic_dividends::StochasticDividendHullWhitePricingPlan::compile_bs(
                &request,
                d,
                model.clone(),
                *equity_rate_correlation,
                0.0,
                0.5,
                policy(),
            )
            .unwrap()
            .evaluate_aad()
            .unwrap()
        } else {
            pricing::stochastic_dividends::StochasticDividendPricingPlan::compile_bs(
                &request,
                d,
                0.5,
                policy(),
            )
            .unwrap()
            .evaluate_aad()
            .unwrap()
        };
        let s = StochasticDividendSpecification::new(
            base,
            d,
            vec![],
            if hybrid { Some(0.0) } else { None },
            0.5,
        )
        .unwrap();
        let actual = replay(
            &PricingTask::new(
                s,
                PricingOperation::Aad {
                    gamma_relative_bump: None,
                },
            )
            .unwrap(),
        );
        assert_eq!(
            actual.native_result,
            PricingTaskOutput::StochasticDividendRisk(Box::new(expected))
        );
    }
}

fn rich_basket() -> MultiAssetSpecification {
    use pricing::mc::hull_white::HullWhiteLsvTarget;
    let s = extended_basket(0, true, true, true);
    let mut m = s.model().clone();
    let target = HullWhiteLsvTarget::from_market_iv(
        MarketIvSurface::new(vec![0.5, 1.0], vec![-0.8, 0.0, 0.8], vec![0.2; 6]).unwrap(),
        vec![0.0, 0.5, 1.0],
        vec![-0.8, 0.0, 0.8],
        1e-8,
        4.0,
    )
    .unwrap();
    let g = target.grid();
    m.models[0] = ModelSpec::LocalVolatility(
        LocalVolatilitySpec::from_explicit_grid(
            g.time_nodes().to_vec(),
            g.log_moneyness_nodes().to_vec(),
            g.values().to_vec(),
            g.floor(),
            g.cap(),
        )
        .unwrap(),
    );
    m.hull_white.as_mut().unwrap().lsv_targets[0] = Some(target);
    let raw = m.correlation.entries()[0].1.raw();
    m.driver_correlations = Some(vec![vec![
        vec![1.0, raw[1], -0.3, 0.0],
        vec![raw[2], 1.0, -0.1, 0.0],
        vec![-0.3, -0.1, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ]]);
    m.local_correlation
        .as_mut()
        .unwrap()
        .1
        .second_driver_correlations = Some(vec![vec![
        vec![1.0, 0.9, -0.3, 0.0],
        vec![0.9, 1.0, -0.27, 0.0],
        vec![-0.3, -0.27, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ]]);
    MultiAssetSpecification::new(
        s.valuation_date(),
        s.product().clone(),
        s.markets().to_vec(),
        m,
        s.engine(),
        s.execution(),
        s.maximum_step(),
    )
    .unwrap()
}

#[test]
fn full_driver_endpoints_and_market_iv_provenance_replay() {
    let task = PricingTask::new(
        rich_basket(),
        PricingOperation::Aad {
            gamma_relative_bump: None,
        },
    )
    .unwrap();
    let result = replay(&task);
    assert!(
        result
            .report
            .risks()
            .iter()
            .any(|r| r.name == "joint_asset_hw.market_iv.raw")
    );
    let original: Value = serde_json::from_str(&pricing_task_to_json(&task).unwrap()).unwrap();
    for pointer in [
        "/source/driver_correlations/0/1/2",
        "/source/local_correlation/second_driver_correlations/0/1/2",
    ] {
        let mut v = original.clone();
        *v.pointer_mut(pointer).unwrap() = json!(-0.2);
        let changed =
            parse_pricing_task_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).unwrap();
        assert_ne!(task.fingerprint().unwrap(), changed.fingerprint().unwrap());
        // Structural source inspection cannot assert a covariance is usable.
        assert_eq!(
            changed
                .source()
                .capability(PricingOperation::Evaluate)
                .status,
            CapabilityStatus::Conditional
        );
        assert!(changed.compile().is_err());
    }
}
#[test]
fn capability_reports_unavailable_reverse_without_calibrating() {
    let s = source()
        .with_model(
            PricingModelSpec::new(
                EquityModelSpec::LocalStochasticVolatility {
                    target: lv(),
                    factor: factor(0),
                    particles: particles(false),
                    forward_density: None,
                },
                RateModelSpec::Deterministic,
            ),
            None,
        )
        .unwrap();
    assert_eq!(
        s.capability(PricingOperation::LocalVarianceRisk).reason,
        Some(CapabilityReason::ReverseTraceRequired)
    );
    assert!(PricingTask::new(s.clone(), PricingOperation::LocalVarianceRisk).is_err());
    assert!(
        PricingTask::new(s, PricingOperation::Evaluate)
            .unwrap()
            .compile()
            .is_ok()
    );
    assert_eq!(
        source()
            .capability(PricingOperation::Aad {
                gamma_relative_bump: None
            })
            .status,
        CapabilityStatus::Unsupported
    );
}
#[test]
fn strict_task_and_report_readers_reject_corruption() {
    let t = PricingTask::new(basket_model(false), PricingOperation::Evaluate).unwrap();
    let original: Value = serde_json::from_str(&pricing_task_to_json(&t).unwrap()).unwrap();
    for (pointer, value) in [
        ("/source/numerics/maximum_step", json!(0)),
        ("/source/lsv", json!([])),
        ("/source/markets/1/underlying_id", json!(77)),
        ("/source/extra", json!(true)),
        ("/operation/type", json!("delta")),
        ("/schema_version", json!(2)),
    ] {
        let mut v = original.clone();
        if pointer == "/source/extra" {
            v["source"]["extra"] = value;
        } else {
            *v.pointer_mut(pointer).unwrap() = value;
        }
        assert!(
            parse_pricing_task_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).is_err(),
            "{pointer}"
        );
    }
    let text = pricing_task_to_json(&t).unwrap();
    let duplicate = text.replacen('{', "{\"schema_version\":1,", 1);
    assert!(parse_pricing_task_json(duplicate.as_bytes(), JsonLimits::DEFAULT).is_err());
    let r = replay(&t);
    let mut v: Value = serde_json::from_str(&valuation_report_to_json(&r.report).unwrap()).unwrap();
    v["standard_error"] = json!(-1);
    assert!(
        parse_valuation_report_json(&serde_json::to_vec(&v).unwrap(), JsonLimits::DEFAULT).is_err()
    );
    let mut limits = JsonLimits::DEFAULT;
    limits.max_input_bytes = 8;
    assert!(parse_pricing_task_json(text.as_bytes(), limits).is_err());
}

fn extended_basket(mode: usize, hybrid: bool, joint: bool, rqmc: bool) -> MultiAssetSpecification {
    use pricing::mc::hull_white::HullWhiteLsvTarget;
    let s = basket_model(rqmc);
    let mut m = s.model().clone();
    m.models[0] = ModelSpec::LocalVolatility(lv());
    m.lsv[0] = Some(match factor(mode) {
        VolatilityFactorSpec::Bergomi(factor) => MultiAssetLsvConfig {
            factor,
            particles: particles(true),
        }
        .into(),
        VolatilityFactorSpec::BergomiTwoFactor(factor) => MultiAssetLsv2FactorConfig {
            factor,
            particles: particles(true),
        }
        .into(),
        VolatilityFactorSpec::RoughBergomi(factor) => MultiAssetRoughLsvConfig {
            factor,
            particles: particles(true),
        }
        .into(),
        _ => unreachable!(),
    });
    if hybrid {
        let target =
            HullWhiteLsvTarget::flat(0.2, vec![0.0, 0.5, 1.0], vec![-0.8, 0.0, 0.8], 1e-8, 4.0)
                .unwrap();
        let grid = target.grid();
        m.models[0] = ModelSpec::LocalVolatility(
            LocalVolatilitySpec::from_explicit_grid(
                grid.time_nodes().to_vec(),
                grid.log_moneyness_nodes().to_vec(),
                grid.values().to_vec(),
                grid.floor(),
                grid.cap(),
            )
            .unwrap(),
        );
        m.hull_white = Some(MultiAssetHullWhiteConfig {
            rate_model: HullWhite1Factor::new(0.1, vec![0.0, 0.4], vec![0.005, 0.006]).unwrap(),
            rate_correlations: vec![0.0; 2 + m.lsv[0].as_ref().unwrap().factor_count()],
            lsv_targets: vec![Some(target), None],
        });
    }
    if joint {
        let hw_target = hybrid.then(|| {
            HullWhiteLsvTarget::flat(
                0.03_f64.sqrt(),
                vec![0.0, 0.5, 1.0],
                vec![-0.8, 0.0, 0.8],
                1e-8,
                4.0,
            )
            .unwrap()
        });
        let second = CorrelationTermStructure::new(
            m.correlation.underlyings().to_vec(),
            vec![(s.valuation_date(), vec![vec![1.0, 0.9], vec![0.9, 1.0]])],
            m.correlation.tolerances(),
        )
        .unwrap();
        m.local_correlation = Some((
            LocalCorrelationConfig {
                basket_weights: vec![0.5, 0.5],
                target: hw_target
                    .as_ref()
                    .map(|t| t.grid().clone())
                    .unwrap_or_else(|| {
                        LocalVarianceGrid::new(
                            vec![0.0, 0.5, 1.0],
                            vec![-0.8, 0.0, 0.8],
                            vec![0.03; 9],
                            1e-8,
                            4.0,
                        )
                        .unwrap()
                    }),
                second_correlation: second,
                particles: LsvParticleConfig::new(64, 513, 0.8, 2.0, true).unwrap(),
                feasibility: LocalCorrelationFeasibility::ProjectAndReport,
                minimum_variance_span: 1e-12,
            },
            LocalCorrelationExtensions {
                second_driver_correlations: None,
                hull_white_target: hw_target,
            },
        ));
    }
    MultiAssetSpecification::new(
        s.valuation_date(),
        s.product().clone(),
        s.markets().to_vec(),
        m,
        s.engine(),
        s.execution(),
        s.maximum_step(),
    )
    .unwrap()
}
#[test]
fn mixed_lsv_shared_rates_and_joint_calibration_are_complete_sources() {
    for (mode, hybrid, joint) in [
        (0, false, false),
        (1, false, false),
        (0, true, false),
        (1, true, false),
        (2, true, false),
        (0, false, true),
        (0, true, true),
        (2, true, true),
    ] {
        for rqmc in [false, true] {
            let s = extended_basket(mode, hybrid, joint, rqmc);
            let m = s.model();
            let native = if let Some((local, ext)) = &m.local_correlation {
                MultiAssetPricingPlan::compile_with_joint_local_correlation(
                    s.valuation_date(),
                    MultiAssetProduct::from_product_graph(s.product().clone()),
                    s.markets().to_vec(),
                    m.models.clone(),
                    m.correlation.clone(),
                    s.engine(),
                    s.execution(),
                    s.maximum_step(),
                    m.lsv.clone(),
                    m.driver_correlations.clone(),
                    m.hull_white.clone(),
                    local.clone(),
                    ext.clone(),
                )
                .unwrap()
            } else if let Some(hw) = &m.hull_white {
                MultiAssetPricingPlan::compile_with_hull_white(
                    s.valuation_date(),
                    MultiAssetProduct::from_product_graph(s.product().clone()),
                    s.markets().to_vec(),
                    m.models.clone(),
                    m.correlation.clone(),
                    s.engine(),
                    s.execution(),
                    s.maximum_step(),
                    m.lsv.clone(),
                    m.driver_correlations.clone(),
                    hw.clone(),
                )
                .unwrap()
            } else {
                MultiAssetPricingPlan::compile_with_bergomi_lsv(
                    s.valuation_date(),
                    MultiAssetProduct::from_product_graph(s.product().clone()),
                    s.markets().to_vec(),
                    m.models.clone(),
                    m.correlation.clone(),
                    s.engine(),
                    s.execution(),
                    s.maximum_step(),
                    m.lsv.clone(),
                    m.driver_correlations.clone(),
                )
                .unwrap()
            };
            let expected = native
                .evaluate_aad(MultiAssetRiskConfig::default())
                .unwrap();
            let result = replay(
                &PricingTask::new(
                    s,
                    PricingOperation::Aad {
                        gamma_relative_bump: None,
                    },
                )
                .unwrap(),
            );
            assert_eq!(
                result.native_result,
                PricingTaskOutput::MultiAsset(Box::new(expected)),
                "mode={mode} hw={hybrid} joint={joint} rqmc={rqmc}"
            );
            assert_eq!(
                result.report.uncertainty_scope(),
                UncertaintyScope::ConditionalOnCalibration
            );
        }
    }
}
#[test]
fn fingerprint_covers_execution_calibration_correlations_and_operation() {
    let t = PricingTask::new(
        extended_basket(0, true, true, false),
        PricingOperation::Evaluate,
    )
    .unwrap();
    let original: Value = serde_json::from_str(&pricing_task_to_json(&t).unwrap()).unwrap();
    for (pointer, value) in [
        ("/source/numerics/worker_threads", json!(2)),
        ("/source/numerics/maximum_step", json!(0.25)),
        ("/source/lsv/0/value/particles/seed", json!(75)),
        ("/source/hull_white/rate_correlations/0", json!(0.1)),
        ("/source/local_correlation/particles/seed", json!(515)),
        ("/source/local_correlation/target/values/0", json!(0.031)),
        ("/source/markets/0/spot", json!(101.0)),
    ] {
        let mut changed = original.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        let changed =
            parse_pricing_task_json(&serde_json::to_vec(&changed).unwrap(), JsonLimits::DEFAULT)
                .unwrap();
        assert_ne!(
            t.fingerprint().unwrap(),
            changed.fingerprint().unwrap(),
            "{pointer}"
        );
    }
    let other = PricingTask::new(
        t.source().clone(),
        PricingOperation::Aad {
            gamma_relative_bump: None,
        },
    )
    .unwrap();
    assert_ne!(t.fingerprint().unwrap(), other.fingerprint().unwrap());
}
