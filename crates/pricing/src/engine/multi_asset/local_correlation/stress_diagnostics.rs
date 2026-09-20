//! Manual attribution study, not an acceptance gate. Retain the legacy
//! coordinate order only here for comparison with corrected production sampling.
//! Reuse each calibration and common pricing scrambles across seeds.
use super::*;
use crate::core::{CurrencyId, CurveId, DayCountConvention, PositiveF64};
use crate::market::{
    CorrelationToleranceConfig, EquityMarket, LogLinearDiscountCurve, MarketIvSurface,
};
use crate::mc::hull_white::HullWhiteLsvTarget;
use crate::mc::{DeterministicExecutor, RqmcConfig, VarianceReduction, inverse_standard_normal};
use crate::models::{Bergomi2Factor, BlackScholesSpec, LocalVolatilitySpec};
use crate::product::OptionSide;
use crate::product::multi_asset::{BasketComponent, MultiAssetProduct};
use pricing_numerics::{standard_normal_cdf, standard_normal_pdf};
use serde_json::json;
use std::sync::Arc;

const CALIBRATION_SEEDS: [u64; 3] = [41709, 42903, 44001];
const PRICING_SEED: u64 = 41709 ^ 0xd1b5_4a32_d192_ed03;
const POINTS: u64 = 8192;
const PREFIX: u64 = 2048;
const SCRAMBLES: u32 = 8;
const METRICS: usize = 18;
const XS: [f64; 3] = [-0.36, 0.0, 0.36];

fn expiry_time() -> f64 {
    DayCountConvention::Act365F
        .year_fraction("2026-01-01".parse().unwrap(), "2029-01-01".parse().unwrap())
}

fn correlation(rho: f64) -> CorrelationTermStructure {
    CorrelationTermStructure::new(
        vec![UnderlyingId::new(1), UnderlyingId::new(2)],
        vec![(
            "2026-01-01".parse().unwrap(),
            vec![vec![1.0, rho], vec![rho, 1.0]],
        )],
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

fn plan(seed: u64) -> MultiAssetPricingPlan {
    let t = expiry_time();
    let times: Vec<_> = (0..=192)
        .map(|i| if i == 192 { t } else { t * i as f64 / 192.0 })
        .collect();
    let xs: Vec<_> = (0..=80).map(|i| -0.8 + i as f64 * 0.02).collect();
    let target = HullWhiteLsvTarget::flat(0.28, times.clone(), xs.clone(), 1e-8, 4.0).unwrap();
    let qtimes = vec![
        DayCountConvention::Act365F
            .year_fraction("2026-01-01".parse().unwrap(), "2026-07-02".parse().unwrap()),
        1.0,
        2.0,
        t,
    ];
    let qxs = vec![-1.2, -0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8, 1.2];
    let ivs = qtimes
        .iter()
        .flat_map(|_| qxs.iter().map(|&x| basket_iv(x)))
        .collect();
    let basket = HullWhiteLsvTarget::from_market_iv(
        MarketIvSurface::new(qtimes, qxs, ivs).unwrap(),
        times,
        xs,
        1e-8,
        4.0,
    )
    .unwrap();
    let market = |i| {
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
                curve(1, 0.03),
                curve(100 + i, 0.01),
                vec![],
            )
            .unwrap(),
        )
    };
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
        100.0 * (0.02 * t + 0.36).exp(),
        1.0,
        "2029-01-01".parse().unwrap(),
        "2029-01-01".parse().unwrap(),
        None,
    )
    .unwrap();
    let g = target.grid();
    let lv = ModelSpec::LocalVolatility(
        LocalVolatilitySpec::from_explicit_grid(
            g.time_nodes().to_vec(),
            g.log_moneyness_nodes().to_vec(),
            g.values().to_vec(),
            g.floor(),
            g.cap(),
        )
        .unwrap(),
    );
    let particles = LsvParticleConfig::new(131_072, seed, 0.035, 20.0, false).unwrap();
    MultiAssetPricingPlan::compile_with_joint_local_correlation(
        "2026-01-01".parse().unwrap(),
        product,
        vec![market(1), market(2)],
        vec![
            lv,
            ModelSpec::BlackScholes(BlackScholesSpec::new(0.28).unwrap()),
        ],
        correlation(-0.3),
        EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(
                POINTS,
                SCRAMBLES,
                PRICING_SEED,
                VarianceReduction::new(true, true),
            )
            .unwrap(),
        ),
        ExecutionPolicy::new(2, Some(256)).unwrap(),
        t / 192.0,
        vec![
            Some(
                MultiAssetLsv2FactorConfig {
                    factor: Bergomi2Factor::new([0.4, 2.0], 0.6, 0.35, [-0.35, -0.2], 0.25)
                        .unwrap(),
                    particles: particles.clone(),
                }
                .into(),
            ),
            None,
        ],
        None,
        None,
        LocalCorrelationConfig {
            basket_weights: vec![0.5, 0.5],
            target: basket.grid().clone(),
            second_correlation: correlation(0.95),
            particles,
            feasibility: LocalCorrelationFeasibility::ProjectAndReport,
            minimum_variance_span: 1e-12,
        },
        LocalCorrelationExtensions::default(),
    )
    .unwrap()
}

fn basket_iv(x: f64) -> f64 {
    (0.235_f64.powi(2) * (1.0 - 0.15 * x + 0.025 * x * x)).sqrt()
}

fn black(x: f64, iv: f64) -> (f64, f64, f64) {
    let t = expiry_time();
    let scale = 100.0 * (-0.01 * t).exp();
    let v = iv * t.sqrt();
    let d1 = -x / v + 0.5 * v;
    let d2 = d1 - v;
    let value = if x < 0.0 {
        x.exp() * standard_normal_cdf(-d2) - standard_normal_cdf(-d1)
    } else {
        standard_normal_cdf(d1) - x.exp() * standard_normal_cdf(d2)
    };
    (
        scale * value,
        scale * standard_normal_pdf(d1) * t.sqrt(),
        standard_normal_cdf(d1),
    )
}

fn iv_error(price: f64, x: f64, target: f64) -> f64 {
    let (mut lo, mut hi) = (1e-8, 3.0);
    for _ in 0..70 {
        let mid = 0.5 * (lo + hi);
        if black(x, mid).0 < price {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi) - target) * 1e4
}

fn mean_se(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let se = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n * (n - 1.0))).sqrt();
    (mean, se)
}

fn study_shocks(
    plan: &MultiAssetPricingPlan,
    scramble: u32,
    point: u64,
    rank_major: bool,
) -> Vec<Vec<f64>> {
    let factors = plan.random_factor_count();
    let steps = plan.times.len() - 1;
    (0..factors)
        .map(|factor| {
            let values: Vec<_> = (0..steps)
                .map(|rank| {
                    inverse_standard_normal(
                        plan.qmc
                            .as_ref()
                            .unwrap()
                            .uniform(
                                scramble,
                                point,
                                (if rank_major {
                                    rank * factors + factor
                                } else {
                                    factor * steps + rank
                                }) as u32,
                            )
                            .unwrap(),
                    )
                    .unwrap()
                })
                .collect();
            plan.bridge
                .as_ref()
                .unwrap()
                .apply_one_factor(&values)
                .unwrap()
        })
        .collect()
}

fn samples(cal: &LocalCorrelationCalibration, shocks: &[Vec<f64>]) -> [f64; METRICS] {
    let path = cal.evolve(shocks).unwrap();
    let last = path.states.last().unwrap();
    let b = 0.5 * (last[0] + last[1]);
    let scale = 100.0 * (-0.01 * expiry_time()).exp();
    let mut out = [0.0; METRICS];
    for (kind, m) in [b, last[0], last[1]].into_iter().enumerate() {
        for (k, x) in XS.into_iter().enumerate() {
            out[kind * 3 + k] = scale
                * if x < 0.0 {
                    (x.exp() - m).max(0.0)
                } else {
                    (m - x.exp()).max(0.0)
                };
        }
    }
    out[9] = last[0];
    out[10] = last[1];
    out[11] = b;
    out[12] = path.boundary_counts[0];
    out[13] = path.states[..path.states.len() - 1]
        .iter()
        .filter(|s| {
            let x = cal.basket(s).ln();
            x < -0.8 || x > 0.8
        })
        .count() as f64;
    out[14] = f64::from(out[12] > 0.0);
    out[15] = f64::from(out[13] > 0.0);
    for (k, x) in [0.0, 0.36].into_iter().enumerate() {
        out[16 + k] = out[1 + k] - black(x, basket_iv(x)).2 * scale * (b - 1.0);
    }
    out
}

#[test]
#[ignore = "manual release-mode fixed-calibration/common-scramble attribution study"]
fn two_factor_stress_sampling_attribution() {
    for seed in CALIBRATION_SEEDS {
        let plan = plan(seed);
        let cal = plan.local_correlation.as_ref().unwrap();
        println!(
            "STRESS_CALIBRATION {}",
            json!({"seed":seed,"pricing_seed":PRICING_SEED,"particles":131072,"steps":plan.times.len()-1,"factors":plan.random_factor_count(),"projected_nodes":cal.diagnostics.iter().filter(|d| d.projected).count(),"fallback_nodes":cal.diagnostics.iter().filter(|d| d.fallback).count(),"terminal_particle_means":cal.particle_means.last()})
        );
        for rank_major in [false, true] {
            let layout = if rank_major {
                "bridge_rank_major"
            } else {
                "legacy_factor_major"
            };
            let executor = DeterministicExecutor::new(plan.execution).unwrap();
            let mut full = Vec::new();
            let mut prefix = Vec::new();
            for scramble in 0..SCRAMBLES {
                let stats = executor
                    .try_map_reduce_statistics_vector(POINTS, 2 * METRICS, |point, output| {
                        let mut z = study_shocks(&plan, scramble, point, rank_major);
                        let a = samples(cal, &z);
                        for row in &mut z {
                            for v in row {
                                *v = -*v;
                            }
                        }
                        let b = samples(cal, &z);
                        for k in 0..METRICS {
                            let value = 0.5 * (a[k] + b[k]);
                            output[k] = value;
                            output[METRICS + k] = if point < PREFIX { value } else { 0.0 };
                        }
                        Ok::<(), E>(())
                    })
                    .unwrap();
                let means: Vec<_> = stats[..METRICS]
                    .iter()
                    .map(|s| s.sum().total() / POINTS as f64)
                    .collect();
                let small: Vec<_> = stats[METRICS..]
                    .iter()
                    .map(|s| s.sum().total() / PREFIX as f64)
                    .collect();
                println!(
                    "STRESS_SCRAMBLE {}",
                    json!({"seed":seed,"layout":layout,"scramble":scramble,"points":POINTS,"means":means,"prefix_points":PREFIX,"prefix_means":small})
                );
                full.push(means);
                prefix.push(small);
            }
            for (points, means) in [(PREFIX, &prefix), (POINTS, &full)] {
                let stats: Vec<_> = (0..METRICS)
                    .map(|k| mean_se(&means.iter().map(|row| row[k]).collect::<Vec<_>>()))
                    .collect();
                let quotes:Vec<_>=(0..3).flat_map(|kind| XS.into_iter().enumerate().map(move |(k,x)|(kind,k,x))).map(|(kind,k,x)| {
                    let target=if kind==0 {basket_iv(x)} else {0.28};let (mean,se)=stats[3*kind+k];
                    json!({"kind":(["basket","lsv_constituent","bs_constituent"][kind]),"x":x,"price":mean,"conditional_se":se,"iv_error_bp":iv_error(mean,x,target),"pricing_se_bp":se/black(x,target).1*1e4})
                }).collect();
                println!(
                    "STRESS_ATTRIBUTION {}",
                    json!({"seed":seed,"pricing_seed":PRICING_SEED,"layout":layout,"points":points,"scrambles":SCRAMBLES,"quotes":quotes,"metric_mean_se":stats,"scramble_means":means})
                );
            }
            if rank_major && seed == CALIBRATION_SEEDS[0] {
                // Confirm that the corrected diagnostic is the public price,
                // including its antithetic and between-scramble SE conventions.
                let price = plan.evaluate().unwrap().price;
                let (mean, se) = mean_se(&full.iter().map(|v| v[2]).collect::<Vec<_>>());
                assert!((mean - price.value().get()).abs() < 1e-10);
                assert!((se - price.standard_error().get()).abs() < 1e-10);
                println!(
                    "STRESS_PUBLIC_REPLAY {}",
                    json!({"layout":layout,"price":price.value().get(),"conditional_se":price.standard_error().get()})
                );
            }
        }
    }
}
