use super::*;
use pricing::core::{CurrencyId, CurveId, Date, EventId, PositiveF64, UnderlyingId};
use pricing::market::{
    DividendEvent, DividendQuote, EquityForward, EquityMarket, LocalVarianceGrid,
    LogLinearDiscountCurve,
};
use pricing::mc::hull_white::HullWhiteLsvTarget;
use pricing::models::{BlackScholesSpec, HullWhite1Factor, LocalVolatilitySpec, ModelSpec};
use pricing::multi_asset::*;
use pricing::product::multi_asset::{BasketComponent, MultiAssetProduct};
use pricing::product::{CompactC2Smoothing, OptionSide};
use std::sync::Arc;

fn date(s: &str) -> Date {
    s.parse().unwrap()
}
fn correlation(rho: f64) -> CorrelationTermStructure {
    CorrelationTermStructure::new(
        vec![UnderlyingId::new(1), UnderlyingId::new(2)],
        vec![(date("2026-01-01"), vec![vec![1.0, rho], vec![rho, 1.0]])],
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
#[derive(Clone)]
struct Case {
    scenario: Scenario,
    factors: [Factor; 2],
    joint: bool,
    hw: bool,
    targets: Vec<HullWhiteLsvTarget>,
    basket: HullWhiteLsvTarget,
    spots: [f64; 2],
    discount_shift: f64,
    dividend_shift: f64,
    bs_sigma: f64,
}
impl Case {
    fn quotes(asset: usize) -> Vec<f64> {
        single::quotes()
            .iter()
            .map(|v| v + [0.045, 0.065][asset])
            .collect()
    }
    fn new(s: Scenario, f: Factor, joint: bool, hw: bool) -> Self {
        Self {
            scenario: s,
            factors: [
                f,
                if matches!(f, Factor::Rough) {
                    Factor::Two
                } else {
                    Factor::One
                },
            ],
            joint,
            hw,
            targets: (0..2).map(|i| single::target(Self::quotes(i), s)).collect(),
            basket: single::target(single::quotes(), s),
            spots: [100.0, 90.0],
            discount_shift: 0.0,
            dividend_shift: 0.0,
            bs_sigma: 0.29,
        }
    }
    fn compile(&self, trace: bool) -> MultiAssetPricingPlan {
        let s = self.scenario;
        let discount = Arc::new(
            LogLinearDiscountCurve::new(
                CurveId::new(1),
                vec![0.0, 0.4, 1.0, 2.0],
                [0.0_f64, 0.4, 1.0, 2.0]
                    .into_iter()
                    .enumerate()
                    .map(|(i, t)| {
                        (-0.025 * t + if i == 2 { self.discount_shift } else { 0.0 }).exp()
                    })
                    .collect(),
            )
            .unwrap(),
        );
        let markets = (0..2)
            .map(|i| {
                let dividend = Arc::new(
                    LogLinearDiscountCurve::new(
                        CurveId::new(100 + i as u32),
                        vec![0.0, 0.7, 2.0],
                        [0.0_f64, 0.7, 2.0]
                            .into_iter()
                            .enumerate()
                            .map(|(j, t)| {
                                (-[0.01, 0.02][i] * t
                                    + if i == 0 && j == 1 {
                                        self.dividend_shift
                                    } else {
                                        0.0
                                    })
                                .exp()
                            })
                            .collect(),
                    )
                    .unwrap(),
                );
                let dividends = if i == 0 {
                    vec![
                        DividendEvent::new(
                            EventId::new(1),
                            0.37,
                            DividendQuote::fixed_cash_and_proportional(1.5, 0.02, EventId::new(1))
                                .unwrap(),
                        )
                        .unwrap(),
                    ]
                } else {
                    vec![]
                };
                EquityMarket::new(
                    CurrencyId::new(1),
                    EquityForward::with_discrete_dividends(
                        UnderlyingId::new(i as u32 + 1),
                        PositiveF64::new(self.spots[i], "spot").unwrap(),
                        discount.clone(),
                        dividend,
                        dividends,
                    )
                    .unwrap(),
                )
            })
            .collect();
        let models = self
            .targets
            .iter()
            .enumerate()
            .map(|(i, t)| {
                if self.joint && i == 1 {
                    return ModelSpec::BlackScholes(BlackScholesSpec::new(self.bs_sigma).unwrap());
                }
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
            })
            .collect();
        let configs: Vec<_> = (0..2)
            .map(|i| {
                if self.joint && i == 1 {
                    None
                } else {
                    let mut cal = s;
                    cal.calibration_seed += i as u64 * 101;
                    Some(self.factors[i].multi(cal, trace))
                }
            })
            .collect();
        let mut rate_correlations = vec![0.1, -0.05];
        for cfg in configs.iter().flatten() {
            rate_correlations.extend(vec![0.02; cfg.factor_count()]);
        }
        let hw = MultiAssetHullWhiteConfig {
            rate_model: HullWhite1Factor::new(0.13, vec![0.0, 0.37], vec![0.008, 0.01]).unwrap(),
            rate_correlations,
            lsv_targets: self
                .targets
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    if self.joint && i == 1 {
                        None
                    } else {
                        Some(t.clone())
                    }
                })
                .collect(),
        };
        let product = MultiAssetProduct::basket(
            CurrencyId::new(1),
            (0..2)
                .map(|i| BasketComponent {
                    underlying: UnderlyingId::new(i + 1),
                    weight: [0.6, 0.4][i as usize],
                    scale: 1.0,
                })
                .collect(),
            OptionSide::Call,
            96.0,
            1.0,
            date("2027-01-01"),
            date("2027-01-01"),
            Some(CompactC2Smoothing::new(3.0).unwrap()),
        )
        .unwrap();
        if self.joint {
            MultiAssetPricingPlan::compile_with_joint_local_correlation(
                date("2026-01-01"),
                product,
                markets,
                models,
                correlation(-0.3),
                s.engine(),
                s.policy(),
                1.0 / s.steps as f64,
                configs,
                None,
                self.hw.then_some(hw),
                LocalCorrelationConfig {
                    basket_weights: vec![0.6, 0.4],
                    target: self.basket.grid().clone(),
                    second_correlation: correlation(0.95),
                    particles: s.particles(trace),
                    feasibility: LocalCorrelationFeasibility::ProjectAndReport,
                    minimum_variance_span: 1e-12,
                },
                LocalCorrelationExtensions {
                    second_driver_correlations: None,
                    hull_white_target: self.hw.then(|| self.basket.clone()),
                },
            )
            .unwrap()
        } else {
            MultiAssetPricingPlan::compile_with_hull_white(
                date("2026-01-01"),
                product,
                markets,
                models,
                correlation(0.4),
                s.engine(),
                s.policy(),
                1.0 / s.steps as f64,
                configs,
                None,
                hw,
            )
            .unwrap()
        }
    }
    fn price(&self) -> f64 {
        self.compile(false).evaluate().unwrap().price.value().get()
    }
}
fn market_sweeps(c: &Case, r: &MultiAssetPrice, label: &str) -> usize {
    let mut failures = 0;
    for asset in 0..2 {
        failures += sweep(
            label,
            c.scenario,
            &format!("spot_{asset}"),
            r.risks[asset].delta.value().get(),
            |h| {
                let mut b = c.clone();
                b.spots[asset] += h;
                b.price()
            },
        );
    }
    if c.hw {
        let curve = r.hull_white_curve_risk.as_ref().unwrap();
        for (name, bar) in [
            (
                "discount_log_df_1y",
                curve.discount_log_df_adjoints[2].value().get(),
            ),
            (
                "asset_0_dividend_log_df",
                curve.dividend_log_df_adjoints[0][1].value().get(),
            ),
        ] {
            failures += sweep(label, c.scenario, name, bar, |h| {
                let mut b = c.clone();
                if name == "discount_log_df_1y" {
                    b.discount_shift += h;
                } else {
                    b.dividend_shift += h;
                }
                b.price()
            });
        }
    }
    failures
}
fn quote_sweeps(
    c: &Case,
    label: &str,
    basket: bool,
    asset: usize,
    risk: &MultiAssetHullWhiteLsvRisk,
) -> usize {
    let q = if basket {
        single::quotes()
    } else {
        Case::quotes(asset)
    };
    let raw = risk.vega_kt_raw.as_ref().unwrap();
    assert_eq!(raw.len(), q.len());
    conditional_errors(risk.vega_kt_standard_errors.as_ref().unwrap(), q.len());
    assert!((risk.parallel_vega.unwrap() - raw.iter().sum::<f64>()).abs() < 1e-10);
    for (raw, scaled) in raw.iter().zip(risk.vega_kt_market_scaled.as_ref().unwrap()) {
        assert_eq!(*raw * 0.01, *scaled);
    }
    let mut failures = 0;
    for (j, aad) in raw
        .iter()
        .copied()
        .chain(std::iter::once(risk.parallel_vega.unwrap()))
        .enumerate()
    {
        failures += sweep(
            label,
            c.scenario,
            &format!(
                "{}_market_iv_{}",
                if basket {
                    "basket".into()
                } else {
                    format!("asset_{asset}")
                },
                if j == q.len() {
                    "parallel".into()
                } else {
                    j.to_string()
                }
            ),
            aad,
            |h| {
                let mut b = c.clone();
                let shifted = single::target(
                    q.iter()
                        .enumerate()
                        .map(|(i, v)| v + if i == j || j == q.len() { h } else { 0.0 })
                        .collect(),
                    c.scenario,
                );
                if basket {
                    b.basket = shifted;
                } else {
                    b.targets[asset] = shifted;
                }
                b.price()
            },
        );
    }
    failures
}
pub(super) fn fixed(rough: bool) {
    let f = if rough { Factor::Rough } else { Factor::Two };
    let label = format!("mixed_hw_{f:?}");
    let mut failures = 0;
    for s in Scenario::all() {
        let c = Case::new(s, f, false, true);
        let p = c.compile(true);
        let r = p.evaluate_aad(MultiAssetRiskConfig::default()).unwrap();
        assert_eq!(r.price, p.evaluate().unwrap().price);
        for asset in 0..2 {
            failures += quote_sweeps(
                &c,
                &label,
                false,
                asset,
                r.risks[asset].hull_white_lsv.as_ref().unwrap(),
            );
        }
        failures += market_sweeps(&c, &r, &label);
        println!(
            "EXTENDED_RISK_CASE {}",
            json!({"case":label,"scenario":s,"price":r.price.value().get(),
            "market_iv_adjoints":r.risks.iter().map(|a|&a.hull_white_lsv.as_ref().unwrap().vega_kt_raw).collect::<Vec<_>>(),
            "conditional_standard_errors":r.risks.iter().map(|a|&a.hull_white_lsv.as_ref().unwrap().vega_kt_standard_errors).collect::<Vec<_>>(),
            "directions":18})
        );
    }
    assert_eq!(failures, 0, "{label}: failed derivative sweeps");
}
pub(super) fn joint(f: Factor, hw: bool) {
    let label = format!("joint_{f:?}_hw_{hw}");
    let mut failures = 0;
    for s in Scenario::all() {
        let c = Case::new(s, f, true, hw);
        let p = c.compile(true);
        let r = p.evaluate_aad(MultiAssetRiskConfig::default()).unwrap();
        assert_eq!(r.price, p.evaluate().unwrap().price);
        let risk = r.local_correlation_risk.as_ref().unwrap();
        let mut directions = 0;
        if hw {
            failures += quote_sweeps(
                &c,
                &label,
                true,
                0,
                risk.basket_hull_white.as_ref().unwrap(),
            );
            failures += quote_sweeps(
                &c,
                &label,
                false,
                0,
                risk.asset_hull_white[0].as_ref().unwrap(),
            );
            directions += 14;
        } else {
            for basket in [true, false] {
                let t = if basket { &c.basket } else { &c.targets[0] };
                let g = t.grid();
                let adj = if basket {
                    &risk.basket_variance_adjoints
                } else {
                    &risk.asset_adjoints[0]
                };
                for axis in 0..2 {
                    let d: Vec<_> = (0..g.values().len())
                        .map(|i| {
                            if axis == 0 {
                                1.0
                            } else {
                                (0.71 * i as f64).cos()
                            }
                        })
                        .collect();
                    failures += sweep(
                        &label,
                        s,
                        &format!(
                            "{}_variance_direction_{axis}",
                            if basket { "basket" } else { "asset_0" }
                        ),
                        dot(adj, &d),
                        |h| {
                            let mut b = c.clone();
                            let shifted = HullWhiteLsvTarget::new(
                                LocalVarianceGrid::new(
                                    g.time_nodes().to_vec(),
                                    g.log_moneyness_nodes().to_vec(),
                                    g.values().iter().zip(&d).map(|(v, d)| v + h * d).collect(),
                                    g.floor(),
                                    g.cap(),
                                )
                                .unwrap(),
                                t.log_densities().to_vec(),
                            )
                            .unwrap();
                            if basket {
                                b.basket = shifted;
                            } else {
                                b.targets[0] = shifted;
                            }
                            b.price()
                        },
                    );
                    directions += 1;
                }
            }
        }
        failures += sweep(
            &label,
            s,
            "asset_1_bs_sigma",
            risk.asset_adjoints[1][0],
            |h| {
                let mut b = c.clone();
                b.bs_sigma += h;
                b.price()
            },
        );
        failures += market_sweeps(&c, &r, &label);
        directions += if hw { 5 } else { 3 };
        let diagnostics = p.local_correlation_calibration().unwrap().diagnostics();
        println!(
            "EXTENDED_RISK_CASE {}",
            json!({"case":label,"scenario":s,"price":r.price.value().get(),
            "projected_nodes":diagnostics.iter().filter(|d|d.projected).count(),
            "fallback_nodes":diagnostics.iter().filter(|d|d.fallback).count(),
            "total_nodes":diagnostics.len(),"directions":directions,
            "basket_adjoints":risk.basket_variance_adjoints,"asset_0_adjoints":risk.asset_adjoints[0]})
        );
    }
    assert_eq!(failures, 0, "{label}: failed derivative sweeps");
}
