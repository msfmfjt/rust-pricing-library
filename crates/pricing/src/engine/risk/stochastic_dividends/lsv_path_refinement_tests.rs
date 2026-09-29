//! Conditional rough-LSV refinement of fixed contractual observations.
//! Payoffs and physical-Spot derivatives are reconstructed without production
//! payoff/reverse helpers. Barrier smoothing is held fixed across all grids.

use super::lsv_refinement_tests::{Coupling, mean_se, payload};
use super::*;
use crate::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

pub(super) const EXPIRY: f64 = 364.0 / 365.0;
pub(super) const FIXING: f64 = 182.0 / 365.0;
pub(super) const PAYMENT: f64 = 456.0 / 365.0;
const WIDTH: f64 = 8.0;
const LEVELS: [usize; 4] = [16, 32, 64, 128];
const LABELS: [&str; 4] = [
    "asian_price",
    "asian_delta",
    "barrier_price",
    "barrier_delta",
];

#[derive(Clone, Copy)]
pub(super) enum Contract {
    Asian,
    Barrier,
}

pub(super) fn request_value(contract: Contract) -> Value {
    let mut v = payload();
    v["model"]["local_variance_grid"]["time_nodes"] = json!([0.0, FIXING, EXPIRY]);
    v["market"]["discrete_dividends"][0]["ex_time"] = json!(FIXING);
    v["market"]["discrete_dividends"][1]["ex_time"] = json!(EXPIRY);
    v["engine"]["independent_sampling_units"] = json!(64);
    v["engine"]["variance_reduction"]["antithetic"] = json!(true);
    v["product"] = match contract {
        Contract::Asian => json!({"type":"arithmetic_asian", "underlying_id":1,
            "currency_id":2, "strike":95.0, "notional":1.0, "side":{"type":"call"},
            "observations":[
                {"date":"2026-08-05", "weight":0.2, "value":{"type":"known", "fixing":102.0}},
                {"date":"2027-03-05", "weight":0.3, "value":{"type":"unknown"}},
                {"date":"2027-09-03", "weight":0.5, "value":{"type":"unknown"}}
            ], "payment_date":"2027-12-04"}),
        Contract::Barrier => json!({"type":"barrier", "underlying_id":1,
            "currency_id":2, "expiry":"2027-09-03", "strike":80.0, "barrier":105.0,
            "notional":1.0, "side":{"type":"call"}, "direction":{"type":"up"},
            "style":{"type":"knock_in"}, "monitoring":{"type":"discrete"},
            "monitoring_dates":["2027-03-05", "2027-09-03"], "payment_date":"2027-12-04"}),
    };
    if matches!(contract, Contract::Barrier) {
        v["risk"]["payoff_smoothing"] = json!({"type":"compact_c2", "half_width":WIDTH});
    }
    v
}

pub(super) fn compile(v: &Value, h: f64, eta: f64, kappa: f64) -> StochasticDividendPricingPlan {
    let request = parse_request_json(&serde_json::to_vec(v).unwrap(), JsonLimits::DEFAULT).unwrap();
    let plan = StochasticDividendPricingPlan::compile_rough_bergomi_lsv(
        &request,
        BuehlerDividendModel::new(kappa, 0.6, 0.35, -0.25).unwrap(),
        RoughBergomi::new(h, eta, -0.4).unwrap(),
        0.15,
        LsvParticleConfig::new(512, 42, 0.35, 5.0, false).unwrap(),
        EXPIRY / 16.0,
        ExecutionPolicy::new(1, Some(32)).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.time_nodes().len(), 17);
    assert_eq!(*plan.time_nodes().last().unwrap(), EXPIRY);
    assert_eq!(plan.payment_time, PAYMENT);
    plan
}

pub(super) fn paths(
    base: &StochasticDividendPricingPlan,
    h: f64,
    eta: f64,
    kappa: f64,
) -> Vec<StochasticDividendPathPlan> {
    let surface = base.path.lsv_surface().unwrap();
    LEVELS
        .iter()
        .map(|&steps| {
            // Preserve the production calibration knots bit-for-bit. Computing
            // i*T/n afresh can move a knot by one ulp and cross a surface row.
            let ratio = steps / 16;
            let times = base
                .time_nodes()
                .windows(2)
                .flat_map(|w| {
                    (1..=ratio).map(move |j| {
                        if j == ratio {
                            w[1]
                        } else {
                            w[0] + (w[1] - w[0]) * (j as f64 / ratio as f64)
                        }
                    })
                })
                .collect();
            let grid = LocalVolTimeGrid::compile(times, 1.0).unwrap();
            let path = StochasticDividendPathPlan::compile(
                &base.market,
                BuehlerDividendModel::new(kappa, 0.6, 0.35, -0.25).unwrap(),
                0.0,
                &grid,
            )
            .unwrap()
            .with_rough_bergomi_lsv(
                RoughBergomi::new(h, eta, -0.4).unwrap(),
                0.15,
                surface.clone(),
            )
            .unwrap();
            assert_eq!(path.times().len(), steps + 1);
            assert_eq!(path.times()[steps / 2], FIXING);
            assert_eq!(path.times()[steps], EXPIRY);
            for (i, &time) in path.times().iter().enumerate() {
                assert!((time - EXPIRY * (i as f64 / steps as f64)).abs() < 1e-15);
            }
            assert_eq!(
                path.lsv_surface().unwrap().squared_leverage(),
                surface.squared_leverage()
            );
            path
        })
        .collect()
}

fn indicator(x: f64) -> (f64, f64) {
    let t = (x / WIDTH).clamp(-1.0, 1.0);
    (
        0.5 + 15.0 * t / 16.0 - 5.0 * t.powi(3) / 8.0 + 3.0 * t.powi(5) / 16.0,
        15.0 * (1.0 - t * t).powi(2) / (16.0 * WIDTH),
    )
}

fn positive_part(x: f64) -> f64 {
    if x.abs() >= WIDTH {
        x.max(0.0)
    } else {
        let t = x / WIDTH;
        WIDTH * (5.0 + 16.0 * t + 15.0 * t * t - 5.0 * t.powi(4) + t.powi(6)) / 32.0
    }
}

fn estimates(path: &StochasticDividendPathPlan, z: &[f64], include_pre_cash: bool) -> [f64; 4] {
    let states = path.evolve_path(z).unwrap();
    let at = |time: f64| {
        let i = path
            .times()
            .binary_search_by(|t| t.total_cmp(&time))
            .unwrap();
        let [a, b, c] = path.nodes()[i].coefficients();
        let state = states[i];
        // Normalized f/Y and the cash jump are invariant under physical Spot
        // with leverage re-anchored to funded residual equity.
        (
            a * state.equity() + b * state.dividend() + c,
            (0.98_f64 / 0.95).powf(time) * state.equity(),
            state.dividend(),
        )
    };
    let mid = at(FIXING);
    let end = at(EXPIRY);
    let discount = 0.95_f64.powf(PAYMENT);
    let average = 0.2 * 102.0 + 0.3 * mid.0 + 0.5 * end.0;
    let asian_delta = if average > 95.0 {
        0.3 * mid.1 + 0.5 * end.1
    } else {
        0.0
    };
    let mut survival = 1.0;
    let mut survival_delta = 0.0;
    for (state, cash) in [(mid, 5.0), (end, 3.0)] {
        let score = state.0 - 105.0
            + if include_pre_cash {
                positive_part(cash * state.2)
            } else {
                0.0
            };
        let (hit, slope) = indicator(score);
        survival_delta = survival_delta * (1.0 - hit) - survival * slope * state.1;
        survival *= 1.0 - hit;
    }
    let vanilla = (end.0 - 80.0).max(0.0);
    let vanilla_delta = if end.0 > 80.0 { end.1 } else { 0.0 };
    [
        discount * (average - 95.0).max(0.0),
        discount * asian_delta,
        discount * vanilla * (1.0 - survival),
        discount * (vanilla_delta * (1.0 - survival) - vanilla * survival_delta),
    ]
}

fn antithetic(path: &StochasticDividendPathPlan, z: &[f64], pre_cash: bool) -> [f64; 4] {
    let a = estimates(path, z, pre_cash);
    let b = estimates(path, &z.iter().map(|x| -x).collect::<Vec<_>>(), pre_cash);
    std::array::from_fn(|i| 0.5 * (a[i] + b[i]))
}

#[test]
fn rough_path_refinement_observations_match_public_price_and_delta() {
    for h in [0.1, 0.3] {
        for (contract, offset) in [(Contract::Asian, 0), (Contract::Barrier, 2)] {
            let base = compile(&request_value(contract), h, 0.6, 0.7);
            for path in paths(&base, h, 0.6, 0.7) {
                // Keep the original contract and calibration while checking the
                // public payoff/reverse adapter on every refined execution grid.
                let mut plan = base.clone();
                plan.path = path;
                let rng = Philox4x32::from_seed(193);
                let mut values = [Vec::new(), Vec::new()];
                let mut cash_witness = 0.0;
                for p in 0..64 {
                    let z = (0..plan.path.random_dimension())
                        .map(|d| {
                            rng.standard_normal(RandomCoordinate::new(
                                p,
                                d,
                                RandomDomain::Valuation,
                            ))
                        })
                        .collect::<Vec<_>>();
                    let sample = antithetic(&plan.path, &z, true);
                    let wrong = antithetic(&plan.path, &z, false);
                    cash_witness += sample[2] - wrong[2];
                    for j in 0..2 {
                        values[j].push(sample[offset + j]);
                    }
                }
                assert!(
                    cash_witness > 1.0,
                    "panel must exercise both sides of cash jumps"
                );
                let risk = plan.evaluate_lsv_spot_risk().unwrap();
                for (samples, (actual, se)) in values.iter().zip([
                    (risk.price.value, risk.price.standard_error),
                    (risk.delta, risk.delta_standard_error),
                ]) {
                    let reference = mean_se(samples);
                    assert!(reference.0 > 0.0 && reference.1 > 0.0);
                    assert!((reference.0 - actual).abs() < 2e-10);
                    assert!((reference.1 - se).abs() < 2e-10);
                }
            }
        }
    }
}

#[test]
fn rough_path_refinement_delta_matches_recalibrated_spot_bumps() {
    const BUMP: f64 = 1e-4;
    for h in [0.1, 0.3] {
        for contract in [Contract::Asian, Contract::Barrier] {
            let mut v = request_value(contract);
            let base = compile(&v, h, 0.6, 0.7);
            let delta = base.evaluate_lsv_spot_risk().unwrap().delta;
            v["market"]["spot"] = json!(100.0 + BUMP);
            let up = compile(&v, h, 0.6, 0.7).evaluate().unwrap().value;
            v["market"]["spot"] = json!(100.0 - BUMP);
            let down = compile(&v, h, 0.6, 0.7).evaluate().unwrap().value;
            let central = (up - down) / (2.0 * BUMP);
            assert!((central - delta).abs() < 2e-7, "{central} != {delta}");
        }
    }
}

#[test]
fn rough_path_refinement_is_exact_at_contract_dates_in_constant_volatility_limit() {
    let mut v = request_value(Contract::Asian);
    v["model"]["local_variance_grid"]["values"] = json!(vec![0.04; 9]);
    let base = compile(&v, 0.1, 0.0, 0.0);
    let paths = paths(&base, 0.1, 0.0, 0.0);
    let fine = paths.last().unwrap();
    let rng = Philox4x32::from_seed(193);
    let couplings = LEVELS[..3]
        .iter()
        .map(|n| Coupling::new(0.1, 128 / n))
        .collect::<Vec<_>>();
    let mut nonzero = [false; 4];
    for p in 0..64 {
        let normal = |d| rng.standard_normal(RandomCoordinate::new(p, d, RandomDomain::Valuation));
        let z = (0..fine.random_dimension()).map(normal).collect::<Vec<_>>();
        let reference = antithetic(fine, &z, true);
        for j in 0..4 {
            nonzero[j] |= reference[j] > 0.0;
        }
        for (i, coupling) in couplings.iter().enumerate() {
            let extra = (0..LEVELS[i])
                .map(|d| normal(fine.random_dimension() + d as u32))
                .collect::<Vec<_>>();
            let coarse = coupling.coarsen(&z, &extra);
            for (actual, expected) in antithetic(&paths[i], &coarse, true).iter().zip(reference) {
                assert!((actual - expected).abs() < 3e-10, "{actual} != {expected}");
            }
        }
    }
    assert!(nonzero.into_iter().all(|x| x));
}

#[test]
#[ignore = "release-mode fixed-observation rough-LSV Asian and Barrier refinement"]
fn rough_lsv_asian_and_barrier_fixed_surface_refinement() {
    const UNITS: u64 = 262144;
    let mut failures = Vec::new();
    for h in [0.1, 0.3] {
        let base = compile(&request_value(Contract::Asian), h, 0.6, 0.7);
        let paths = paths(&base, h, 0.6, 0.7);
        let fine = paths.last().unwrap();
        let couplings = LEVELS[..3]
            .iter()
            .map(|n| Coupling::new(h, 128 / n))
            .collect::<Vec<_>>();
        for seed in [193, 877] {
            let rng = Philox4x32::from_seed(seed);
            let mut differences = vec![vec![Vec::new(); 4]; 3];
            let mut coarse_values = vec![vec![Vec::new(); 4]; 3];
            let mut fine_values = vec![Vec::new(); 4];
            for p in 0..UNITS {
                let normal =
                    |d| rng.standard_normal(RandomCoordinate::new(p, d, RandomDomain::Valuation));
                let z = (0..fine.random_dimension()).map(normal).collect::<Vec<_>>();
                let reference = antithetic(fine, &z, true);
                for j in 0..4 {
                    fine_values[j].push(reference[j]);
                }
                for (level, coupling) in couplings.iter().enumerate() {
                    let extra = (0..LEVELS[level])
                        .map(|i| normal(fine.random_dimension() + i as u32))
                        .collect::<Vec<_>>();
                    let coarse = coupling.coarsen(&z, &extra);
                    let sample = antithetic(&paths[level], &coarse, true);
                    for j in 0..4 {
                        coarse_values[level][j].push(sample[j]);
                        differences[level][j].push(sample[j] - reference[j]);
                    }
                }
            }
            for (level, &steps) in LEVELS[..3].iter().enumerate() {
                for (j, label) in LABELS.iter().enumerate() {
                    let (gap, se) = mean_se(&differences[level][j]);
                    let coarse = mean_se(&coarse_values[level][j]);
                    let reference = mean_se(&fine_values[j]);
                    let unpaired = coarse.1.hypot(reference.1);
                    println!(
                        "{}",
                        json!({"hurst":h, "seed":seed, "steps":steps,
                        "reference_steps":128, "quantity":label, "antithetic_units":UNITS,
                        "coarse":coarse.0, "reference":reference.0, "paired_difference":gap,
                        "paired_se":se, "unpaired_se":unpaired, "expiry":EXPIRY,
                        "fixing_time":FIXING, "payment_time":PAYMENT, "barrier_smoothing_width":WIDTH,
                        "calibration_steps":16, "calibration_particles":512, "calibration_seed":42,
                        "scope":"fixed_surface_fixed_contract_observations"})
                    );
                    assert!(
                        [gap, se, coarse.0, reference.0, unpaired]
                            .into_iter()
                            .all(f64::is_finite)
                    );
                    // All levels must benefit from coupling. The stronger
                    // 25% reduction requirement applies to the acceptance
                    // level, not to far-coarse unsmoothed Asian Delta.
                    let ratio_limit = if level == 2 { 0.75 } else { 1.0 };
                    if se >= ratio_limit * unpaired {
                        failures.push(format!("H={h}, seed={seed}, steps={steps}, {label}: paired SE={se}, unpaired SE={unpaired}"));
                    }
                    if level == 2
                        && (gap.abs() + 4.0 * se >= [0.05, 0.005][j % 2]
                            || se >= [0.01, 0.001][j % 2])
                    {
                        failures.push(format!("H={h}, seed={seed}, {label}: gap={gap}, SE={se}"));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
