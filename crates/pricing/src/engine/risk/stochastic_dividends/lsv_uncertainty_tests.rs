//! Reconstruct sampling units from primal paths without production payoff,
//! finite-bump sampling, reduction or standard-error helpers.

use super::*;
use crate::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

type Plan = StochasticDividendPricingPlan;
const PARAMETERS: [f64; 4] = [0.1, 0.6, -0.4, 0.15];
const BUMPS: [f64; 4] = [0.01, 0.02, 0.02, 0.02];
const FIRST_FIXING: f64 = 182.0 / 365.0;
const DELAYED_PAYMENT: f64 = 456.0 / 365.0;
const SMOOTHING_WIDTH: f64 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Contract {
    European,
    Asian,
    Barrier,
}

fn payload(contract: Contract, rqmc: bool, antithetic: bool, bridge: bool, seed: u64) -> Value {
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
    if contract != Contract::European {
        value["market"]["discrete_dividends"][0]["ex_time"] = json!(FIRST_FIXING);
    }
    match contract {
        Contract::European => {}
        Contract::Asian => {
            value["product"] = json!({"type": "arithmetic_asian", "underlying_id": 1,
                "currency_id": 2, "strike": 95.0, "notional": 1.0, "side": {"type": "call"},
                "observations": [
                    {"date": "2026-08-05", "weight": 0.2, "value": {"type": "known", "fixing": 102.0}},
                    {"date": "2027-03-05", "weight": 0.3, "value": {"type": "unknown"}},
                    {"date": "2027-09-04", "weight": 0.5, "value": {"type": "unknown"}}
                ], "payment_date": "2027-12-04"});
        }
        Contract::Barrier => {
            value["product"] = json!({"type": "barrier", "underlying_id": 1,
                "currency_id": 2, "expiry": "2027-09-04", "strike": 80.0, "barrier": 105.0,
                "notional": 1.0, "side": {"type": "call"}, "direction": {"type": "up"},
                "style": {"type": "knock_in"}, "monitoring": {"type": "discrete"},
                "monitoring_dates": ["2027-03-05", "2027-09-04"], "payment_date": "2027-12-04"});
            value["risk"]["payoff_smoothing"] =
                json!({"type": "compact_c2", "half_width": SMOOTHING_WIDTH});
        }
    }
    value
}

fn request(value: &Value) -> PricingRequest {
    parse_request_json(&serde_json::to_vec(value).unwrap(), JsonLimits::DEFAULT).unwrap()
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

fn smooth_indicator(x: f64) -> f64 {
    let t = (x / SMOOTHING_WIDTH).clamp(-1.0, 1.0);
    0.5 + 15.0 * t / 16.0 - 5.0 * t.powi(3) / 8.0 + 3.0 * t.powi(5) / 16.0
}

fn smooth_positive_part(x: f64) -> f64 {
    if x.abs() >= SMOOTHING_WIDTH {
        return x.max(0.0);
    }
    let t = x / SMOOTHING_WIDTH;
    // Integral of the centered quintic above; independent of the production
    // CompactC2Smoothing helper and its shifted-coordinate Horner expression.
    SMOOTHING_WIDTH * (5.0 + 16.0 * t + 15.0 * t * t - 5.0 * t.powi(4) + t.powi(6)) / 32.0
}

fn contract_payoff(contract: Contract, plan: &Plan, shocks: &[f64], include_pre_cash: bool) -> f64 {
    let states = plan.path.evolve_path(shocks).unwrap();
    assert_eq!(*plan.time_nodes().first().unwrap(), 0.0);
    assert_eq!(*plan.time_nodes().last().unwrap(), 1.0);
    let at = |time: f64| {
        let index = plan
            .time_nodes()
            .binary_search_by(|t| t.total_cmp(&time))
            .unwrap();
        let [a, b, c] = plan.path.nodes()[index].coefficients();
        let state = states[index];
        (
            a * state.equity() + b * state.dividend() + c,
            state.dividend(),
        )
    };
    let terminal = at(1.0).0;
    let delayed_discount = 0.95_f64.powf(DELAYED_PAYMENT);
    match contract {
        Contract::European => 0.95 * (terminal - 100.0).max(0.0),
        Contract::Asian => {
            let average = 0.2 * 102.0 + 0.3 * at(FIRST_FIXING).0 + 0.5 * terminal;
            delayed_discount * (average - 95.0).max(0.0)
        }
        Contract::Barrier => {
            let mut survival = 1.0;
            for (time, cash_mean) in [(FIRST_FIXING, 6.0), (1.0, 2.0)] {
                let (post, dividend) = at(time);
                let distance = post - 105.0;
                let score = if include_pre_cash {
                    distance + smooth_positive_part(cash_mean * dividend)
                } else {
                    distance
                };
                survival *= 1.0 - smooth_indicator(score);
            }
            delayed_discount * (terminal - 80.0).max(0.0) * (1.0 - survival)
        }
    }
}

fn payoffs(
    contract: Contract,
    plans: &[Plan],
    mut normals: Vec<f64>,
    vr: VarianceReduction,
) -> Vec<f64> {
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
    let mut values = plans
        .iter()
        .map(|plan| {
            let positive = contract_payoff(contract, plan, &normals, true);
            if vr.antithetic() {
                let negative = normals.iter().map(|z| -z).collect::<Vec<_>>();
                (positive + contract_payoff(contract, plan, &negative, true)) / 2.0
            } else {
                positive
            }
        })
        .collect::<Vec<_>>();
    if contract == Contract::Barrier {
        // Retain a deliberately wrong post-cash-only reference to show that
        // the sampled panel actually exercises pre-cash jump observations.
        let mut wrong = contract_payoff(contract, &plans[0], &normals, false);
        if vr.antithetic() {
            let negative = normals.iter().map(|z| -z).collect::<Vec<_>>();
            wrong = (wrong + contract_payoff(contract, &plans[0], &negative, false)) / 2.0;
        }
        values.push(wrong);
    }
    values
}

fn independent_units(contract: Contract, plans: &[Plan]) -> Vec<Vec<f64>> {
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
                    payoffs(contract, plans, z, config.variance_reduction())
                })
                .collect()
        }
        EngineConfig::RandomizedQuasiMonteCarlo(config) => {
            let qmc = RqmcPlan::compile(config, dimension).unwrap();
            (0..config.scramble_count().get())
                .map(|scramble| {
                    let mut sums =
                        vec![0.0; plans.len() + usize::from(contract == Contract::Barrier)];
                    for p in 0..config.points_per_scramble().get() {
                        let z = (0..dimension)
                            .map(|d| {
                                inverse_standard_normal(qmc.uniform(scramble, p, d).unwrap())
                                    .unwrap()
                            })
                            .collect();
                        for (sum, value) in sums.iter_mut().zip(payoffs(
                            contract,
                            plans,
                            z,
                            config.variance_reduction(),
                        )) {
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

fn check_engine(contract: Contract, rqmc: bool) {
    for seed in [91, 1973] {
        for antithetic in [false, true] {
            for bridge in [false, true] {
                let context = format!(
                    "{contract:?}, rqmc={rqmc}, seed={seed}, antithetic={antithetic}, bridge={bridge}"
                );
                let compare = |label: &str, actual, expected| {
                    close(&format!("{context}, {label}"), actual, expected);
                };
                let request = request(&payload(contract, rqmc, antithetic, bridge, seed));
                let mut plans = vec![compile(&request, PARAMETERS)];
                for coordinate in 0..4 {
                    for sign in [-1.0, 1.0] {
                        let mut parameters = PARAMETERS;
                        parameters[coordinate] += sign * BUMPS[coordinate];
                        plans.push(compile(&request, parameters));
                    }
                }
                let units = independent_units(contract, &plans);
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
                compare("price", price.value, mean);
                compare("price SE", price.standard_error, error);
                if contract != Contract::European {
                    let wrong_payment_price = mean * 0.95 / 0.95_f64.powf(DELAYED_PAYMENT);
                    assert!(
                        (mean - wrong_payment_price).abs() > 1e-6,
                        "panel must distinguish payment from fixing discounting"
                    );
                }
                if contract == Contract::Barrier {
                    let wrong = units
                        .iter()
                        .map(|row| *row.last().unwrap())
                        .collect::<Vec<_>>();
                    assert!(
                        (mean - mean_and_error(&wrong).0).abs() > 1e-6,
                        "panel must distinguish pre/post-cash from post-only monitoring"
                    );
                }
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
                    compare(&format!("derivative[{coordinate}]"), derivative, mean);
                    compare(
                        &format!("paired derivative SE[{coordinate}]"),
                        standard_error,
                        error,
                    );
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
    check_engine(Contract::European, false);
}

#[test]
fn rough_lsv_rqmc_errors_match_independent_scramble_means() {
    check_engine(Contract::European, true);
}

#[test]
fn rough_lsv_asian_mc_errors_include_known_fixings_and_delayed_payment() {
    check_engine(Contract::Asian, false);
}

#[test]
fn rough_lsv_asian_rqmc_errors_include_known_fixings_and_delayed_payment() {
    check_engine(Contract::Asian, true);
}

#[test]
fn rough_lsv_barrier_mc_errors_include_both_sides_of_cash_jumps() {
    check_engine(Contract::Barrier, false);
}

#[test]
fn rough_lsv_barrier_rqmc_errors_include_both_sides_of_cash_jumps() {
    check_engine(Contract::Barrier, true);
}

#[test]
fn rough_lsv_unsmoothed_barrier_risk_is_rejected_without_affecting_price() {
    let mut value = payload(Contract::Barrier, false, true, true, 91);
    value["risk"]
        .as_object_mut()
        .unwrap()
        .remove("payoff_smoothing");
    let plan = compile(&request(&value), PARAMETERS);
    let before = plan.evaluate().unwrap();
    assert!(
        plan.evaluate_lsv_rough_bergomi_parameter_risk(BUMPS[0], BUMPS[1])
            .is_err()
    );
    assert!(
        plan.evaluate_lsv_rough_bergomi_correlation_risk(BUMPS[2], BUMPS[3])
            .is_err()
    );
    assert_eq!(plan.evaluate().unwrap(), before);
}
