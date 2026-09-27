//! Reconstruct sampling units from primal paths without production payoff,
//! finite-bump sampling, reduction or standard-error helpers.

use super::*;
use crate::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

type Plan = StochasticDividendPricingPlan;
const PARAMETERS: [f64; 4] = [0.1, 0.6, -0.4, 0.15];
const BUMPS: [f64; 4] = [0.01, 0.02, 0.02, 0.02];

fn request(rqmc: bool, antithetic: bool, bridge: bool, seed: u64) -> PricingRequest {
    let mut value: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/v1/pricing_request.golden.json"
    )))
    .unwrap();
    value["model"] = json!({"type": "local_volatility", "local_variance_grid": {
        "time_nodes": [0.0, 0.5, 1.0], "log_forward_moneyness_nodes": [-0.5, 0.0, 0.5],
        "shape": [3, 3], "values": vec![0.04; 9], "floor": 1e-8, "cap": 4.0
    }});
    value["market"]["discrete_dividends"] = json!([
        {"event_id": 1, "ex_time": 0.5, "quote": {"type": "fixed_cash", "amount": 6.0}},
        {"event_id": 2, "ex_time": 1.0, "quote": {"type": "fixed_cash", "amount": 2.0}},
        {"event_id": 3, "ex_time": 1.4, "quote": {"type": "fixed_cash", "amount": 3.0}}
    ]);
    value["engine"] = if rqmc {
        json!({"type": "randomized_quasi_monte_carlo", "points_per_scramble": 16,
            "scramble_count": 4, "master_scramble_seed": seed})
    } else {
        json!({"type": "pseudo_monte_carlo", "independent_sampling_units": 64,
            "master_seed": seed})
    };
    value["engine"]["variance_reduction"] =
        json!({"antithetic": antithetic, "brownian_bridge": bridge});
    parse_request_json(&serde_json::to_vec(&value).unwrap(), JsonLimits::DEFAULT).unwrap()
}

fn compile(request: &PricingRequest, parameters: [f64; 4]) -> Plan {
    Plan::compile_rough_bergomi_lsv(
        request,
        BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap(),
        RoughBergomi::new(parameters[0], parameters[1], parameters[2]).unwrap(),
        parameters[3],
        LsvParticleConfig::new(64, 42, 0.35, 5.0, false).unwrap(),
        0.25,
        ExecutionPolicy::new(1, Some(16)).unwrap(),
    )
    .unwrap()
}

fn terminal_call(plan: &Plan, shocks: &[f64]) -> f64 {
    let states = plan.path.evolve_path(shocks).unwrap();
    let state = *states.last().unwrap();
    let node = plan.path.nodes().last().unwrap();
    assert_eq!(node.time(), 1.0);
    let [a, b, c] = node.coefficients();
    // The European call observes post-cash stock, including cash at expiry.
    // Fixed contract: one-year maturity/payment, strike 100, unit notional.
    let stock = a * state.equity() + b * state.dividend() + c;
    0.95 * (stock - 100.0).max(0.0)
}

fn payoffs(plans: &[Plan], mut normals: Vec<f64>, vr: VarianceReduction) -> Vec<f64> {
    if vr.brownian_bridge() {
        let bridge = BrownianBridgePlan::compile(plans[0].time_nodes().to_vec(), 1).unwrap();
        let factors = plans[0].random_factor_count();
        for factor in 0..factors {
            let input = normals
                .iter()
                .skip(factor)
                .step_by(factors)
                .copied()
                .collect::<Vec<_>>();
            for (step, value) in bridge
                .apply_one_factor(&input)
                .unwrap()
                .into_iter()
                .enumerate()
            {
                normals[step * factors + factor] = value;
            }
        }
    }
    plans
        .iter()
        .map(|plan| {
            let positive = terminal_call(plan, &normals);
            if vr.antithetic() {
                let negative = normals.iter().map(|z| -z).collect::<Vec<_>>();
                (positive + terminal_call(plan, &negative)) / 2.0
            } else {
                positive
            }
        })
        .collect()
}

fn independent_units(plans: &[Plan]) -> Vec<Vec<f64>> {
    let dimension = plans[0].path.random_dimension();
    match plans[0].engine {
        EngineConfig::PseudoMonteCarlo(config) => {
            let rng = Philox4x32::from_seed(config.master_seed());
            (0..config.independent_sampling_units().get())
                .map(|p| {
                    let z = (0..dimension)
                        .map(|d| {
                            rng.standard_normal(RandomCoordinate::new(
                                p,
                                d,
                                RandomDomain::Valuation,
                            ))
                        })
                        .collect();
                    payoffs(plans, z, config.variance_reduction())
                })
                .collect()
        }
        EngineConfig::RandomizedQuasiMonteCarlo(config) => {
            let qmc = RqmcPlan::compile(config, dimension).unwrap();
            (0..config.scramble_count().get())
                .map(|scramble| {
                    let mut sums = vec![0.0; plans.len()];
                    for p in 0..config.points_per_scramble().get() {
                        let z = (0..dimension)
                            .map(|d| {
                                inverse_standard_normal(qmc.uniform(scramble, p, d).unwrap())
                                    .unwrap()
                            })
                            .collect();
                        for (sum, value) in
                            sums.iter_mut()
                                .zip(payoffs(plans, z, config.variance_reduction()))
                        {
                            *sum += value;
                        }
                    }
                    for sum in &mut sums {
                        *sum /= config.points_per_scramble().get() as f64;
                    }
                    sums
                })
                .collect()
        }
    }
}

fn mean_and_error(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let squared_deviations = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
    (mean, (squared_deviations / (n * (n - 1.0))).sqrt())
}

fn close(label: &str, actual: f64, expected: f64) {
    let tolerance = 2e-10 + 1e-11 * expected.abs();
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: {actual} != {expected}"
    );
}

fn check_engine(rqmc: bool) {
    for seed in [91, 1973] {
        for antithetic in [false, true] {
            for bridge in [false, true] {
                let request = request(rqmc, antithetic, bridge, seed);
                let mut plans = vec![compile(&request, PARAMETERS)];
                for coordinate in 0..4 {
                    for sign in [-1.0, 1.0] {
                        let mut parameters = PARAMETERS;
                        parameters[coordinate] += sign * BUMPS[coordinate];
                        plans.push(compile(&request, parameters));
                    }
                }
                let units = independent_units(&plans);
                let base = &plans[0];
                let parameters = base
                    .evaluate_lsv_rough_bergomi_parameter_risk(BUMPS[0], BUMPS[1])
                    .unwrap();
                let correlations = base
                    .evaluate_lsv_rough_bergomi_correlation_risk(BUMPS[2], BUMPS[3])
                    .unwrap();
                let price = base.evaluate().unwrap();
                assert_eq!(parameters.price, price);
                assert_eq!(correlations.price, price);
                assert_eq!(price.independent_sampling_units as usize, units.len());
                assert_eq!(price.evaluated_paths, if antithetic { 128 } else { 64 });
                let baseline = units.iter().map(|row| row[0]).collect::<Vec<_>>();
                let (mean, error) = mean_and_error(&baseline);
                close("price", price.value, mean);
                close("price SE", price.standard_error, error);
                for (coordinate, (&derivative, &standard_error)) in parameters
                    .derivatives
                    .iter()
                    .chain(correlations.derivatives.iter())
                    .zip(
                        parameters
                            .standard_errors
                            .iter()
                            .chain(correlations.standard_errors.iter()),
                    )
                    .enumerate()
                {
                    // Pair before computing variance. RQMC rows are independent
                    // scramble means; MC rows already average antithetic paths.
                    let paired = units
                        .iter()
                        .map(|row| {
                            (row[2 + 2 * coordinate] - row[1 + 2 * coordinate])
                                / (2.0 * BUMPS[coordinate])
                        })
                        .collect::<Vec<_>>();
                    let (mean, error) = mean_and_error(&paired);
                    close("derivative", derivative, mean);
                    close("paired derivative SE", standard_error, error);
                    assert!(error > 1e-8, "oracle must exercise a nonzero SE");

                    let up = units
                        .iter()
                        .map(|row| row[2 + 2 * coordinate])
                        .collect::<Vec<_>>();
                    let down = units
                        .iter()
                        .map(|row| row[1 + 2 * coordinate])
                        .collect::<Vec<_>>();
                    let unpaired_error = mean_and_error(&up).1.hypot(mean_and_error(&down).1)
                        / (2.0 * BUMPS[coordinate]);
                    assert!(
                        (unpaired_error - error).abs() > 1e-6,
                        "oracle must distinguish paired from independent scenario errors"
                    );
                }
            }
        }
    }
}

#[test]
fn rough_lsv_mc_errors_match_independent_antithetic_units() {
    check_engine(false);
}

#[test]
fn rough_lsv_rqmc_errors_match_independent_scramble_means() {
    check_engine(true);
}
