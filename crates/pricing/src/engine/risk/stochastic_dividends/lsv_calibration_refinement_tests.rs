//! Full production recalibration with independent outer calibration replicates.
//! Valuation paths are coupled; the production calibration coordinate layout
//! is unchanged and is not a common-Brownian coupling across different grids.

use super::lsv_refinement_tests::{
    Coupling, antithetic_estimates, calibrated_with, mean_se, payload,
};
use super::*;
use crate::models::RoughBergomi;
use crate::{JsonLimits, parse_request_json};
use serde_json::json;

#[derive(Debug)]
struct Ensemble {
    mean: f64,
    // SE of independent replicate means; includes calibration and valuation.
    total_se: f64,
    // Conditional valuation SE of their average, reported separately.
    conditional_valuation_se: f64,
}

fn ensemble(replicates: &[(f64, f64)]) -> Ensemble {
    assert!(replicates.len() >= 2);
    assert!(
        replicates
            .iter()
            .all(|(m, se)| m.is_finite() && se.is_finite() && *se >= 0.0)
    );
    let (mean, total_se) = mean_se(&replicates.iter().map(|x| x.0).collect::<Vec<_>>());
    let conditional_valuation_se =
        replicates.iter().map(|x| x.1 * x.1).sum::<f64>().sqrt() / replicates.len() as f64;
    Ensemble {
        mean,
        total_se,
        conditional_valuation_se,
    }
}

#[test]
fn calibration_ensemble_se_uses_replicates_without_double_counting_inner_noise() {
    // Independent replicate observations [1,3], [3,5], [5,7] give means 2,4,6
    // and conditional SE 1 each. The outer SE is sqrt(4/3), not the inner
    // sqrt(1/3), nor sqrt(5/3) from adding that variance a second time.
    let result = ensemble(&[(2.0, 1.0), (4.0, 1.0), (6.0, 1.0)]);
    assert_eq!(result.mean, 4.0);
    assert!((result.total_se - (4.0_f64 / 3.0).sqrt()).abs() < 1e-14);
    assert!((result.conditional_valuation_se - (1.0_f64 / 3.0).sqrt()).abs() < 1e-14);
    let duplicated = [(2.0, 1.0), (4.0, 1.0), (6.0, 1.0)].repeat(2);
    let more = ensemble(&duplicated);
    assert!((more.total_se - (8.0_f64 / 15.0).sqrt()).abs() < 1e-14);
    assert!((more.conditional_valuation_se - (1.0_f64 / 6.0).sqrt()).abs() < 1e-14);
}

#[test]
fn recalibrated_grid_and_particle_comparisons_vanish_in_exact_constant_volatility_limit() {
    let mut value = payload();
    value["model"]["local_variance_grid"]["values"] = json!(vec![0.04; 9]);
    let request =
        parse_request_json(&serde_json::to_vec(&value).unwrap(), JsonLimits::DEFAULT).unwrap();
    let mut plans = Vec::new();
    for seed in [42, 193] {
        for particles in [32, 64] {
            for steps in [8, 16] {
                let plan = StochasticDividendPricingPlan::compile_rough_bergomi_lsv(
                    &request,
                    BuehlerDividendModel::new(0.0, 0.6, 0.35, -0.25).unwrap(),
                    RoughBergomi::new(0.1, 0.0, -0.4).unwrap(),
                    0.15,
                    LsvParticleConfig::new(particles, seed, 0.35, 5.0, false).unwrap(),
                    1.0 / steps as f64,
                    ExecutionPolicy::new(1, Some(32)).unwrap(),
                )
                .unwrap();
                assert_eq!(plan.time_nodes().len(), steps + 1);
                assert!(
                    plan.lsv_squared_leverage()
                        .unwrap()
                        .iter()
                        .all(|x| (x - 0.04).abs() < 2e-14)
                );
                plans.push(plan);
            }
        }
    }
    let rng = Philox4x32::from_seed(900_001);
    let coupling = Coupling::new(0.1, 2);
    let mut exercised = false;
    for p in 0..32 {
        let z = (0..64)
            .map(|d| rng.standard_normal(RandomCoordinate::new(p, d, RandomDomain::Valuation)))
            .collect::<Vec<_>>();
        let extra = (64..72)
            .map(|d| rng.standard_normal(RandomCoordinate::new(p, d, RandomDomain::Valuation)))
            .collect::<Vec<_>>();
        let coarse = coupling.coarsen(&z, &extra);
        let reference = antithetic_estimates(&plans[1].path, &z);
        exercised |= reference[0] > 0.0 && reference[1] > 0.0;
        for plan in &plans {
            let shocks = if plan.time_nodes().len() == 9 {
                &coarse
            } else {
                &z
            };
            for (actual, expected) in antithetic_estimates(&plan.path, shocks)
                .into_iter()
                .zip(reference)
            {
                assert!((actual - expected).abs() < 3e-11, "{actual} != {expected}");
            }
        }
    }
    assert!(exercised);
}

#[test]
#[ignore = "release-mode full recalibration and particle ensemble refinement"]
fn rough_lsv_recalibrated_grid_and_particle_ensemble_refinement() {
    const PARTICLES: [usize; 3] = [256, 1024, 4096];
    const CALIBRATION_SEEDS: [u64; 8] = [42, 193, 617, 877, 1973, 4179, 6127, 9109];
    const UNITS: u64 = 16384;
    // Preserve the initial eight replicates; add independent calibration
    // repetitions because their variance dominates conditional valuation SE.
    let calibration_seeds = CALIBRATION_SEEDS
        .into_iter()
        .chain((0..24).map(|i| 100_003 + 997 * i))
        .collect::<Vec<_>>();
    // Plans are (N=256 coarse/fine), (N=1024 coarse/fine), (N=4096 coarse/fine).
    // The final two comparisons isolate the particle count on the fine grid.
    const CONTRASTS: [(usize, usize, &str); 5] = [
        (0, 1, "grid_256"),
        (2, 3, "grid_1024"),
        (4, 5, "grid_4096"),
        (1, 3, "particles_256_1024"),
        (3, 5, "particles_1024_4096"),
    ];
    let mut failures = Vec::new();
    for h in [0.1, 0.3] {
        let coupling = Coupling::new(h, 2);
        let mut replicates = vec![[Vec::new(), Vec::new()]; CONTRASTS.len()];
        let mut first_surface = None;
        for (replicate, &calibration_seed) in calibration_seeds.iter().enumerate() {
            let valuation_seed = 900_001 + 97 * replicate as u64;
            let mut plans = Vec::new();
            for particles in PARTICLES {
                for steps in [64, 128] {
                    let plan = calibrated_with(h, steps, particles, calibration_seed);
                    assert_eq!(plan.time_nodes().len(), steps + 1);
                    assert_eq!(plan.lsv_time_nodes().unwrap().len(), steps + 1);
                    plans.push(plan);
                }
            }
            // This is full recalibration, not repeated valuation of one surface.
            assert_ne!(
                plans[1].lsv_squared_leverage(),
                plans[3].lsv_squared_leverage()
            );
            assert_ne!(
                plans[3].lsv_squared_leverage(),
                plans[5].lsv_squared_leverage()
            );
            let surface = plans[5].lsv_squared_leverage().unwrap();
            if let Some(first) = &first_surface {
                assert_ne!(surface, first);
            } else {
                first_surface = Some(surface.to_vec());
            }
            let rng = Philox4x32::from_seed(valuation_seed);
            let mut differences = vec![
                [
                    Vec::with_capacity(UNITS as usize),
                    Vec::with_capacity(UNITS as usize)
                ];
                CONTRASTS.len()
            ];
            for p in 0..UNITS {
                let normal =
                    |d| rng.standard_normal(RandomCoordinate::new(p, d, RandomDomain::Valuation));
                let z = (0..512).map(normal).collect::<Vec<_>>();
                let extra = (512..576).map(normal).collect::<Vec<_>>();
                let coarse = coupling.coarsen(&z, &extra);
                let estimates = plans
                    .iter()
                    .enumerate()
                    .map(|(i, plan)| {
                        antithetic_estimates(&plan.path, if i % 2 == 0 { &coarse } else { &z })
                    })
                    .collect::<Vec<_>>();
                for (contrast, &(a, b, _)) in CONTRASTS.iter().enumerate() {
                    for j in 0..2 {
                        differences[contrast][j].push(estimates[a][j] - estimates[b][j]);
                    }
                }
            }
            for (contrast, &(_, _, label)) in CONTRASTS.iter().enumerate() {
                for (j, quantity) in ["price", "delta"].iter().enumerate() {
                    let (gap, se) = mean_se(&differences[contrast][j]);
                    replicates[contrast][j].push((gap, se));
                    println!(
                        "{}",
                        json!({"kind":"replicate", "hurst":h, "contrast":label,
                        "quantity":quantity, "calibration_seed":calibration_seed, "valuation_seed":valuation_seed,
                        "antithetic_units":UNITS, "paired_difference":gap, "conditional_valuation_se":se})
                    );
                }
            }
        }
        for (contrast, &(_, _, label)) in CONTRASTS.iter().enumerate() {
            for (j, quantity) in ["price", "delta"].iter().enumerate() {
                let summary = ensemble(&replicates[contrast][j]);
                println!(
                    "{}",
                    json!({"kind":"ensemble", "hurst":h, "contrast":label,
                    "quantity":quantity, "calibration_replicates":calibration_seeds.len(),
                    "antithetic_units_per_replicate":UNITS, "mean_difference":summary.mean,
                    "total_replication_se":summary.total_se,
                    "conditional_valuation_se":summary.conditional_valuation_se,
                    "calibration_coupling":"unchanged_production_coordinates",
                    "scope":"calibration_and_valuation_replication"})
                );
                assert!(summary.total_se > 1e-8 && summary.conditional_valuation_se > 1e-8);
                if contrast == 2 || contrast == 4 {
                    let budget = [0.05, 0.005][j];
                    let passes = summary.mean.abs() + 4.0 * summary.total_se < budget
                        && summary.total_se < [0.01, 0.001][j];
                    if !passes {
                        failures.push(format!(
                            "H={h}, {label}, {quantity}: {summary:?}, budget={budget}"
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
