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

fn product(expiry: &str, x: f64, marginal: bool) -> MultiAssetProduct {
    let t = time(expiry);
    MultiAssetProduct::basket(
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
        100.0 * ((RATE - DIVIDEND_RATE) * t + x).exp(),
        1.0,
        date(expiry),
        date(expiry),
        None,
    )
    .unwrap()
}

fn compile(
    case: Case,
    x: f64,
    marginal: bool,
    res: Resolution,
    seed: u64,
    stress: bool,
    maximum_step_override: Option<f64>,
) -> MultiAssetPricingPlan {
    let expiry = if stress { "2029-01-01" } else { "2027-01-01" };
    let t = time(expiry);
    let maximum_step = maximum_step_override.unwrap_or(t / res.steps as f64);
    let times = grid_times(t, res.steps);
    let xs: Vec<_> = (0..=80).map(|i| -0.8 + i as f64 * 0.02).collect();
    let asset = if stress && matches!(case, Case::RoughHw) {
        // The joint HW grid inserts common refinement dates. Keep
        // quotes so paired variance/density targets can regenerate there.
        let quote_times = vec![time("2026-07-02"), 1.0, 2.0, t];
        let quote_xs = vec![-1.2, -0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8, 1.2];
        let ivs = vec![0.28; quote_times.len() * quote_xs.len()];
        HullWhiteLsvTarget::from_market_iv(
            MarketIvSurface::new(quote_times, quote_xs, ivs).unwrap(),
            times.clone(),
            xs.clone(),
            1e-8,
            4.0,
        )
        .unwrap()
    } else {
        HullWhiteLsvTarget::flat(0.28, times.clone(), xs.clone(), 1e-8, 4.0).unwrap()
    };
    let basket = if stress {
        Smile::BasketStress.target(expiry, res)
    } else {
        HullWhiteLsvTarget::flat(0.235, times, xs, 1e-8, 4.0).unwrap()
    };
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
    let product = product(expiry, x, marginal);
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
            maximum_step,
            config,
        )
        .unwrap()
    } else {
        let lsv: MultiAssetBergomiLsvConfig = match case {
            Case::Two => MultiAssetLsv2FactorConfig {
                factor: Bergomi2Factor::new(
                    [0.4, 2.0],
                    // The three-year basket still exceeds the sampling
                    // budgets at nu=0.6 (as did nu=0.8). Preserve the failing
                    // gate and its measured evidence; do not relax the budget.
                    if stress { 0.6 } else { 0.4 },
                    0.35,
                    [-0.35, -0.2],
                    0.25,
                )
                .unwrap(),
                particles: res.particles(seed),
            }
            .into(),
            Case::RoughHw => MultiAssetRoughLsvConfig {
                factor: RoughBergomi::new(0.12, if stress { 0.9 } else { 0.5 }, -0.35).unwrap(),
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
            maximum_step,
            vec![Some(lsv), None],
            None,
            hw.then(|| MultiAssetHullWhiteConfig {
                rate_model: HullWhite1Factor::new(0.13, vec![0.0, 0.375 * t], vec![0.012, 0.0144])
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

fn panel(case: Case, marginal: bool, res: Resolution, stress: bool) -> Vec<QuoteReport> {
    let expiry = if stress { "2029-01-01" } else { "2027-01-01" };
    let t = time(expiry);
    let strikes = if stress { WINGS } else { STRIKES };
    strikes.into_iter().map(|x| {
        let runs = SEEDS.into_iter().map(|s| {
            let seed = s + res.seed_offset;
            let plan = compile(case, x, marginal, res, seed, stress, None);
            let cal = plan.local_correlation_calibration().unwrap();
            let m = cal.log_nodes().len();
            let row = cal.time_nodes().len()-1;
            let nodes: Vec<_> = bracket(cal.log_nodes(),x).map(|j| {
                let d = &cal.diagnostics()[row*m+j];
                json!({"x":cal.log_nodes()[j],"ess":d.effective_samples,"fallback":d.fallback,
                    "projected":d.projected,"unidentifiable":d.unidentifiable,"source_node":d.source_node,
                    "target_variance":d.target_variance,"attained_variance":d.attained_variance,
                    "raw_mixing":d.raw_mixing})
            }).collect();
            println!("EXTENDED_LOCAL_CORRELATION {}", json!({"case":case,"marginal":marginal,"stress":stress,
                "resolution":res,"seed":seed,"x":x,"terminal_quote_nodes":nodes,
                "global_projected_nodes":cal.diagnostics().iter().filter(|d|d.projected).count(),
                "global_unidentifiable_nodes":cal.diagnostics().iter().filter(|d|d.unidentifiable).count(),
                "global_fallback_nodes":cal.diagnostics().iter().filter(|d|d.fallback).count(),"total_nodes":cal.diagnostics().len()}));
            for j in bracket(cal.log_nodes(),x) {
                let d = &cal.diagnostics()[row*m+j];
                assert!(!d.projected && !d.unidentifiable && !d.fallback && d.effective_samples.is_finite() && d.effective_samples>=100.0,
                    "{case:?}, seed={seed}, x={x}: unsupported evaluation node {d:?}");
            }
            let p = plan.evaluate().unwrap().price;
            PriceRun {calibration_seed:seed,pricing_seed:res.pricing_seed(seed),value:p.value().get(),conditional_se:p.standard_error().get()}
        }).collect();
        let target_iv = if marginal {0.28} else if stress {Smile::BasketStress.iv(t,x)} else {0.235};
        summarize(runs,100.0*((RATE-DIVIDEND_RATE)*t).exp(),t,x,target_iv)
    }).collect()
}

fn refinement_case(case: Case, marginal: bool) {
    let label = json!({"multi_asset":case,"marginal":marginal,"expiry":"2027-01-01"});
    let base = panel(case, marginal, FINE, false);
    report(
        json!({"refinement":"base","case":label}),
        FINE,
        &base,
        LSV_BUDGET,
    );
    for (axis, res) in refinements() {
        let refined = panel(case, marginal, res, false);
        report(
            json!({"refinement":axis,"case":label}),
            res,
            &refined,
            LSV_BUDGET,
        );
        compare_refinement(label.clone(), axis, &base, &refined);
    }
}

fn stress_case(case: Case, marginal: bool) {
    let res = match case {
        Case::Bs => STRESS,
        Case::Two => Resolution {
            particles: 131_072,
            points: 32_768,
            ..STRESS
        },
        Case::RoughHw => Resolution {
            points: 16_384,
            ..STRESS
        },
    };
    report_at(
        json!({"stress":true,"multi_asset":case,"marginal":marginal,"expiry":"2029-01-01"}),
        res,
        &panel(case, marginal, res, true),
        &WINGS,
        LSV_BUDGET,
    );
}

macro_rules! model_cases {
    ($refinement:ident, $stress:ident, $case:expr, $marginal:expr) => {
        #[test]
        #[ignore = "release-mode multi-seed extended-model acceptance"]
        fn $refinement() {
            refinement_case($case, $marginal);
        }
        #[test]
        #[ignore = "release-mode multi-seed extended-model acceptance"]
        fn $stress() {
            stress_case($case, $marginal);
        }
    };
}
model_cases!(
    bs_basket_refinement_consistency,
    bs_basket_stress_repricing,
    Case::Bs,
    false
);
model_cases!(
    bs_constituent_refinement_consistency,
    bs_constituent_stress_repricing,
    Case::Bs,
    true
);
model_cases!(
    two_factor_basket_refinement_consistency,
    two_factor_basket_stress_repricing,
    Case::Two,
    false
);
model_cases!(
    two_factor_constituent_refinement_consistency,
    two_factor_constituent_stress_repricing,
    Case::Two,
    true
);
model_cases!(
    rough_hw_basket_refinement_consistency,
    rough_hw_basket_stress_repricing,
    Case::RoughHw,
    false
);
model_cases!(
    rough_hw_constituent_refinement_consistency,
    rough_hw_constituent_stress_repricing,
    Case::RoughHw,
    true
);

#[test]
fn joint_hw_quote_targets_cover_inserted_common_dates() {
    let res = Resolution {
        particles: 2048,
        steps: 4,
        points: 16,
        scrambles: 3,
        bandwidth: 0.1,
        ..STRESS
    };
    let plan = compile(
        Case::RoughHw,
        0.0,
        false,
        res,
        SEEDS[0],
        true,
        Some(time("2029-01-01") / (2 * res.steps) as f64),
    );
    let times = plan.local_correlation_calibration().unwrap().time_nodes();
    assert!(times.len() > res.steps + 1);
    let requested = grid_times(time("2029-01-01"), res.steps);
    assert!(times.iter().any(|t| !requested.contains(t)));
}

#[test]
fn infeasible_basket_skew_is_reported_and_rejected() {
    let expiry = "2029-01-01";
    let res = Resolution {
        particles: 2048,
        steps: 4,
        points: 16,
        scrambles: 3,
        ..STRESS
    };
    let target = Smile::InfeasibleBasketStress.target(expiry, res);
    let compile = |feasibility| {
        MultiAssetPricingPlan::compile_with_local_correlation(
            today(),
            product(expiry, 0.0, false),
            vec![market(1), market(2)],
            vec![ModelSpec::BlackScholes(BlackScholesSpec::new(0.28).unwrap()); 2],
            correlation(-0.3),
            res.engine(SEEDS[0]),
            res.policy(),
            time(expiry) / res.steps as f64,
            LocalCorrelationConfig {
                basket_weights: vec![0.5, 0.5],
                target: target.grid().clone(),
                second_correlation: correlation(0.95),
                particles: res.particles(SEEDS[0]),
                feasibility,
                minimum_variance_span: 1e-12,
            },
        )
    };
    let plan = compile(LocalCorrelationFeasibility::ProjectAndReport).unwrap();
    let cal = plan.local_correlation_calibration().unwrap();
    // All particles start at the same point, so the initial attainable
    // interval is analytic and cannot be repaired by increasing the sample.
    let lower = 0.28_f64.powi(2) * (1.0 - 0.3) / 2.0;
    let upper = 0.28_f64.powi(2) * (1.0 + 0.95) / 2.0;
    let row = &cal.diagnostics()[..cal.log_nodes().len()];
    // For BS/BS, positive asset-value weights sum to one. Even unequal
    // composition and correlation 1 cannot exceed the constituent variance.
    let universal_upper = 0.28_f64.powi(2);
    let beyond_universal = row
        .iter()
        .filter(|d| d.target_variance > universal_upper)
        .count();
    assert!(beyond_universal > 0);
    let projected: Vec<_> = row
        .iter()
        .enumerate()
        .filter(|(_, d)| d.projected)
        .collect();
    assert!(!projected.is_empty());
    for d in row {
        assert!((d.endpoint_variances[0] - lower).abs() < 1e-12);
        assert!((d.endpoint_variances[1] - upper).abs() < 1e-12);
        assert_eq!(
            d.projected,
            d.target_variance < lower || d.target_variance > upper
        );
        assert!((d.attained_variance - d.target_variance.clamp(lower, upper)).abs() < 1e-12);
    }
    println!(
        "EXTENDED_INFEASIBLE_TARGET {}",
        json!({"target":"InfeasibleBasketStress",
        "initial_variance_interval":[lower,upper],"projected_initial_nodes":projected.len(),
        "bs_variance_ceiling":universal_upper,"nodes_above_bs_ceiling":beyond_universal,
        "projected_log_nodes":projected.iter().map(|(j,_)|cal.log_nodes()[*j]).collect::<Vec<_>>()})
    );
    let error = compile(LocalCorrelationFeasibility::Reject).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("infeasible local correlation target"),
        "{error}"
    );
}
