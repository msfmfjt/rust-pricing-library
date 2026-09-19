use super::*;
use pricing::core::{CurrencyId, CurveId, PositiveF64, UnderlyingId};
use pricing::market::{EquityForward, EquityMarket, LogLinearDiscountCurve};
use pricing::models::{
    Bergomi2Factor, BlackScholesSpec, HullWhite1Factor, LocalVolatilitySpec, ModelSpec,
    RoughBergomi,
};
use pricing::multi_asset::*;
use pricing::product::OptionSide;
use std::sync::Arc;

fn correlation(rho: f64) -> CorrelationTermStructure {
    CorrelationTermStructure::new(
        vec![UnderlyingId::new(1), UnderlyingId::new(2)],
        vec![(today(), vec![vec![1.0, rho], vec![rho, 1.0]])],
        CorrelationToleranceConfig {
            symmetry_abs_tol: 1e-12,
            diagonal_abs_tol: 1e-12,
            psd_abs_tol: 1e-12,
            psd_rel_tol: 1e-12,
            zero_pivot_abs_tol: 1e-12,
            zero_pivot_rel_tol: 1e-12,
        },
    )
    .unwrap()
}
fn market(i: u32) -> EquityMarket {
    let curve = |id, r: f64| {
        Arc::new(
            LogLinearDiscountCurve::new(
                CurveId::new(id),
                vec![0.0, 3.0],
                vec![1.0, (-3.0 * r).exp()],
            )
            .unwrap(),
        )
    };
    EquityMarket::new(
        CurrencyId::new(1),
        EquityForward::with_discrete_dividends(
            UnderlyingId::new(i),
            PositiveF64::new(100.0, "spot").unwrap(),
            curve(1, RATE),
            curve(100 + i, DIVIDEND_RATE),
            vec![],
        )
        .unwrap(),
    )
}

#[derive(Clone, Copy, Debug, Serialize)]
enum Case {
    Bs,
    Two,
    RoughHw,
}

fn compile(
    case: Case,
    x: f64,
    marginal: bool,
    res: Resolution,
    seed: u64,
) -> MultiAssetPricingPlan {
    let expiry = "2027-01-01";
    let times: Vec<_> = (0..=res.steps)
        .map(|i| i as f64 / res.steps as f64)
        .collect();
    let xs: Vec<_> = (0..=80).map(|i| -0.8 + i as f64 * 0.02).collect();
    let asset = HullWhiteLsvTarget::flat(0.28, times.clone(), xs.clone(), 1e-8, 4.0).unwrap();
    let basket = HullWhiteLsvTarget::flat(0.235, times, xs, 1e-8, 4.0).unwrap();
    let bs = || ModelSpec::BlackScholes(BlackScholesSpec::new(0.28).unwrap());
    let lv = || {
        let g = asset.grid();
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
    };
    let product = MultiAssetProduct::basket(
        CurrencyId::new(1),
        (1..=2)
            .map(|i| BasketComponent {
                underlying: UnderlyingId::new(i),
                weight: if marginal {
                    if i == 1 { 1.0 } else { 0.0 }
                } else {
                    0.5
                },
                scale: 1.0,
            })
            .collect(),
        if x < 0.0 {
            OptionSide::Put
        } else {
            OptionSide::Call
        },
        100.0 * (RATE - DIVIDEND_RATE + x).exp(),
        1.0,
        date(expiry),
        date(expiry),
        None,
    )
    .unwrap();
    let config = LocalCorrelationConfig {
        basket_weights: vec![0.5, 0.5],
        target: basket.grid().clone(),
        second_correlation: correlation(0.95),
        particles: res.particles(seed),
        feasibility: LocalCorrelationFeasibility::ProjectAndReport,
        minimum_variance_span: 1e-12,
    };
    if matches!(case, Case::Bs) {
        MultiAssetPricingPlan::compile_with_local_correlation(
            today(),
            product,
            vec![market(1), market(2)],
            vec![bs(), bs()],
            correlation(-0.3),
            res.engine(seed),
            res.policy(),
            1.0 / res.steps as f64,
            config,
        )
        .unwrap()
    } else {
        let lsv: MultiAssetBergomiLsvConfig = match case {
            Case::Two => MultiAssetLsv2FactorConfig {
                factor: Bergomi2Factor::new([0.4, 2.0], 0.4, 0.35, [-0.35, -0.2], 0.25).unwrap(),
                particles: res.particles(seed),
            }
            .into(),
            Case::RoughHw => MultiAssetRoughLsvConfig {
                factor: RoughBergomi::new(0.12, 0.5, -0.35).unwrap(),
                particles: res.particles(seed),
            }
            .into(),
            Case::Bs => unreachable!(),
        };
        let hw = matches!(case, Case::RoughHw);
        let mut rate_correlations = vec![0.1, -0.05];
        rate_correlations.extend(vec![0.02; lsv.factor_count()]);
        MultiAssetPricingPlan::compile_with_joint_local_correlation(
            today(),
            product,
            vec![market(1), market(2)],
            vec![lv(), bs()],
            correlation(-0.3),
            res.engine(seed),
            res.policy(),
            1.0 / res.steps as f64,
            vec![Some(lsv), None],
            None,
            hw.then(|| MultiAssetHullWhiteConfig {
                rate_model: HullWhite1Factor::new(0.13, vec![0.0, 0.375], vec![0.012, 0.0144])
                    .unwrap(),
                rate_correlations,
                lsv_targets: vec![Some(asset), None],
            }),
            config,
            LocalCorrelationExtensions {
                second_driver_correlations: None,
                hull_white_target: hw.then_some(basket),
            },
        )
        .unwrap()
    }
}

#[test]
#[ignore = "release-mode multi-seed extended-model acceptance"]
fn local_correlation_basket_and_constituent_repricing() {
    for case in [Case::Bs, Case::Two, Case::RoughHw] {
        for marginal in [false, true] {
            let quotes:Vec<_> = STRIKES.into_iter().map(|x| {
                let runs = SEEDS.into_iter().map(|seed| {
                    let plan = compile(case,x,marginal,FINE,seed);
                    let cal = plan.local_correlation_calibration().unwrap();
                    let m = cal.log_nodes().len();
                    let row = cal.time_nodes().len()-1;
                    let right = cal.log_nodes().partition_point(|&node|node<x).min(m-1);
                    let left = right.saturating_sub(1);
                    for j in left..=right {
                        let d = &cal.diagnostics()[row*m+j];
                        assert!(!d.projected && !d.unidentifiable && !d.fallback && d.effective_samples>=100.0,
                            "{case:?}, seed={seed}, x={x}: unsupported evaluation node {d:?}");
                    }
                    println!("EXTENDED_LOCAL_CORRELATION {}",json!({"case":case,"marginal":marginal,"seed":seed,"x":x,"global_projected_nodes":cal.diagnostics().iter().filter(|d|d.projected).count(),"global_fallback_nodes":cal.diagnostics().iter().filter(|d|d.fallback).count(),"total_nodes":cal.diagnostics().len()}));
                    let p = plan.evaluate().unwrap().price;
                    PriceRun {calibration_seed:seed,pricing_seed:FINE.pricing_seed(seed),value:p.value().get(),conditional_se:p.standard_error().get()}
                }).collect();
                summarize(runs,100.0*(RATE-DIVIDEND_RATE).exp(),1.0,x,if marginal {0.28} else {0.235})
            }).collect();
            report(
                json!({"multi_asset":case,"marginal":marginal,"expiry":"2027-01-01"}),
                FINE,
                &quotes,
                LSV_BUDGET,
            );
        }
    }
}
