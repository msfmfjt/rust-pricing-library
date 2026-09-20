use super::*;
use crate::core::{CurrencyId, CurveId, PositiveF64};
use crate::market::{CorrelationToleranceConfig, EquityMarket, LogLinearDiscountCurve};
use crate::mc::{PseudoMcConfig, RqmcConfig, VarianceReduction};
use crate::models::BlackScholesSpec;
use crate::product::OptionSide;
use crate::product::multi_asset::{BasketComponent, MultiAssetProduct};
use std::sync::Arc;

fn plan(engine: EngineConfig) -> MultiAssetPricingPlan {
    let today: Date = "2026-01-01".parse().unwrap();
    let expiry: Date = "2027-01-01".parse().unwrap();
    let curve = Arc::new(
        LogLinearDiscountCurve::new(CurveId::new(1), vec![0.0, 1.0], vec![1.0, 1.0]).unwrap(),
    );
    let markets = (1..=3)
        .map(|i| {
            EquityMarket::new(
                CurrencyId::new(1),
                EquityForward::with_discrete_dividends(
                    UnderlyingId::new(i),
                    PositiveF64::new(100.0, "spot").unwrap(),
                    curve.clone(),
                    curve.clone(),
                    vec![],
                )
                .unwrap(),
            )
        })
        .collect();
    let product = MultiAssetProduct::basket(
        CurrencyId::new(1),
        (1..=3)
            .map(|i| BasketComponent {
                underlying: UnderlyingId::new(i),
                weight: 1.0 / 3.0,
                scale: 1.0,
            })
            .collect(),
        OptionSide::Call,
        100.0,
        1.0,
        expiry,
        expiry,
        None,
    )
    .unwrap();
    let correlation = CorrelationTermStructure::new(
        (1..=3).map(UnderlyingId::new).collect(),
        vec![(
            today,
            (0..3)
                .map(|i| (0..3).map(|j| f64::from(i == j)).collect())
                .collect(),
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
    .unwrap();
    MultiAssetPricingPlan::compile(
        today,
        product,
        markets,
        vec![ModelSpec::BlackScholes(BlackScholesSpec::new(0.2).unwrap()); 3],
        correlation,
        engine,
        ExecutionPolicy::new(1, Some(4)).unwrap(),
        0.25,
    )
    .unwrap()
}

#[test]
fn every_factor_terminal_uses_a_leading_sobol_coordinate() {
    let p = plan(EngineConfig::RandomizedQuasiMonteCarlo(
        RqmcConfig::new(16, 3, 702, VarianceReduction::new(true, true)).unwrap(),
    ));
    // Identity correlation makes the integrated output shocks the independent
    // terminal Brownian values. All three must use the first three dimensions.
    for scramble in 0..3 {
        for point in [0, 1, 7, 15] {
            let shocks = p.shocks(Some(scramble), point).unwrap();
            for (factor, z) in shocks.iter().enumerate() {
                let terminal = z
                    .iter()
                    .zip(p.times.windows(2))
                    .map(|(z, w)| z * (w[1] - w[0]).sqrt())
                    .sum::<f64>();
                let expected = inverse_standard_normal(
                    p.qmc
                        .as_ref()
                        .unwrap()
                        .uniform(scramble, point, factor as u32)
                        .unwrap(),
                )
                .unwrap();
                assert!(
                    (terminal - expected).abs() < 2e-14,
                    "factor {factor}: {terminal} != {expected}"
                );
            }
        }
    }
}

#[test]
fn pseudo_and_unbridged_qmc_retain_factor_major_coordinates() {
    for engine in [
        EngineConfig::PseudoMonteCarlo(
            PseudoMcConfig::new(511, 16, VarianceReduction::new(true, true)).unwrap(),
        ),
        EngineConfig::RandomizedQuasiMonteCarlo(
            RqmcConfig::new(16, 3, 702, VarianceReduction::new(true, false)).unwrap(),
        ),
    ] {
        let p = plan(engine);
        let actual = p.shocks(Some(1), 7).unwrap();
        for (factor, z) in actual.iter().enumerate() {
            let mut expected: Vec<_> = (0..4)
                .map(|step| {
                    let dimension = (factor * 4 + step) as u32;
                    match engine {
                        EngineConfig::PseudoMonteCarlo(c) => {
                            Philox4x32::from_seed(c.master_seed()).standard_normal(
                                RandomCoordinate::new(7, dimension, RandomDomain::Valuation),
                            )
                        }
                        EngineConfig::RandomizedQuasiMonteCarlo(_) => inverse_standard_normal(
                            p.qmc.as_ref().unwrap().uniform(1, 7, dimension).unwrap(),
                        )
                        .unwrap(),
                    }
                })
                .collect();
            if let Some(bridge) = &p.bridge {
                expected = bridge.apply_one_factor(&expected).unwrap();
            }
            assert_eq!(z, &expected);
        }
    }
}
