//! Factor quadrature error at a FIXED time grid. The reference is the infinite
//! Laplace measure of the semi-implicit scheme, not continuous-time rough Heston.
//! Production paths, kernel factory and random layout are not changed.

use pricing::mc::{ExecutionPolicy, RandomDomain};
use pricing::rough_volatility::{
    LiftedHeston, RoughHeston, RoughVolatilityPathPlan, RoughVolatilityPricingPlan,
};
use pricing::{JsonLimits, parse_request_json};
use serde_json::{Value, json};

const FACTORS: [usize; 6] = [8, 16, 32, 64, 128, 256];
const STRIKES: [f64; 3] = [80.0, 100.0, 120.0];
const STEPS: usize = 64;
const UNITS: usize = 8192;
const PRICE_BUDGET: f64 = 0.025;
const SE_CAP: f64 = 0.002;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/rough-volatility/lifted-factor-prices.json"
    ))
    .unwrap()
}
fn heston(h: f64) -> RoughHeston {
    RoughHeston::new(h, 0.04, 0.7, 0.055, 0.18, -0.65).unwrap()
}
fn lift(h: f64, n: usize) -> LiftedHeston {
    LiftedHeston::from_rough(&heston(h), n, (3.0 / (n as f64).sqrt()).exp()).unwrap()
}
fn grid(t: f64, n: usize) -> Vec<f64> {
    (0..=n).map(|j| t * j as f64 / n as f64).collect()
}

// Exact elimination of all finite factors, also on NONUNIFORM grids.
// G[i][j] = sum w_k product_{ell=j}^{i-1} (1+x_k dt_ell)^(-1).
// It deliberately does not use the production factor recurrence or summation.
fn finite_kernel(model: &LiftedHeston, times: &[f64]) -> Vec<Vec<f64>> {
    let n = times.len() - 1;
    let mut g = vec![vec![0.0; n]; n + 1];
    for (i, row) in g.iter_mut().enumerate().skip(1) {
        for (j, entry) in row.iter_mut().enumerate().take(i) {
            *entry = model
                .weights()
                .iter()
                .zip(model.rates())
                .map(|(&w, &x)| {
                    (j..i).fold(w, |value, ell| {
                        value / (1.0 + x * (times[ell + 1] - times[ell]))
                    })
                })
                .sum();
        }
    }
    g
}

// The closed Laplace integral on a UNIFORM grid is
// g_l = dt^(alpha-1) Gamma(l+alpha-1)/(Gamma(alpha) Gamma(l)).
// g_1=dt^(alpha-1); g_(l+1)/g_l=(l+alpha-1)/l avoids a Gamma dependency.
fn infinite_kernel(h: f64, dt: f64, n: usize) -> Vec<Vec<f64>> {
    let alpha = h + 0.5;
    let mut q = vec![0.0; n + 1];
    q[1] = dt.powf(alpha - 1.0);
    for lag in 1..n {
        q[lag + 1] = q[lag] * (lag as f64 + alpha - 1.0) / lag as f64;
    }
    (0..=n)
        .map(|i| (0..n).map(|j| if j < i { q[i - j] } else { 0.0 }).collect())
        .collect()
}

#[derive(Debug)]
struct ReferencePath {
    forwards: Vec<f64>,
    variance: Vec<f64>,
    raw: Vec<f64>,
    negative: u64,
}
fn reference_path(h: &RoughHeston, times: &[f64], g: &[Vec<f64>], z: &[f64]) -> ReferencePath {
    let n = times.len() - 1;
    assert_eq!(z.len(), 2 * n);
    let mut increments = vec![0.0; n];
    let mut forwards = vec![100.0];
    let mut variance = vec![h.initial_variance()];
    let mut raw = vec![h.initial_variance()];
    let mut negative = 0;
    for j in 0..n {
        let dt = times[j + 1] - times[j];
        let vol = variance[j].sqrt();
        // LEFT-node variance keeps the return exponential adapted.
        forwards.push(forwards[j] * (-0.5 * variance[j] * dt + vol * dt.sqrt() * z[j]).exp());
        let dw = dt.sqrt()
            * (h.correlation() * z[j] + (1.0 - h.correlation().powi(2)).sqrt() * z[n + j]);
        increments[j] = h.mean_reversion() * (h.long_run_variance() - variance[j]) * dt
            + h.vol_of_vol() * vol * dw;
        let value =
            h.initial_variance() + (0..=j).map(|k| g[j + 1][k] * increments[k]).sum::<f64>();
        assert!(value.is_finite());
        raw.push(value);
        negative += u64::from(value < 0.0);
        variance.push(value.max(0.0));
    }
    ReferencePath {
        forwards,
        variance,
        raw,
        negative,
    }
}
fn stats(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (variance / n).sqrt())
}
fn close(a: f64, b: f64, tolerance: f64) {
    assert!(a.is_finite() && b.is_finite());
    assert!(
        (a - b).abs() <= tolerance * b.abs().max(1.0),
        "{a} versus {b}"
    );
}

#[test]
fn fixed_protocol_and_rational_kernels_match_independent_laplace_integrals() {
    let data = fixture();
    assert_eq!(
        data["scope"],
        "fixed-grid-semi-implicit-infinite-factor-limit"
    );
    assert_eq!(
        data["protocol"],
        json!({
            "hursts":[0.1,0.3],"maturities":[0.25,1.0],"steps":STEPS,
            "factors":FACTORS,"ratio_rule":"exp(3/sqrt(factors))","seeds":[91,1973],
            "independent_antithetic_units":UNITS,"strikes":STRIKES,"spot":100.0,
            "initial_variance":0.04,"mean_reversion":0.7,"long_run_variance":0.055,
            "vol_of_vol":0.18,"correlation":-0.65,"final_factors":256,
            "paired_price_budget":PRICE_BUDGET,"paired_se_cap":SE_CAP,
            "forward_sampling_sigmas":5.0,"forward_se_cap":0.15
        })
    );
    assert_eq!(data["kernels"].as_array().unwrap().len(), 4);
    for case in data["kernels"].as_array().unwrap() {
        let h = case["hurst"].as_f64().unwrap();
        let t = case["maturity"].as_f64().unwrap();
        let times = grid(t, STEPS);
        let infinite = infinite_kernel(h, t / STEPS as f64, STEPS);
        for (i, lag) in case["lags"].as_array().unwrap().iter().enumerate() {
            let lag = lag.as_u64().unwrap() as usize;
            close(
                infinite[lag][0],
                case["infinite_kernel"][i].as_f64().unwrap(),
                3e-12,
            );
        }
        assert_eq!(
            case["finite_kernels"].as_array().unwrap().len(),
            FACTORS.len()
        );
        for row in case["finite_kernels"].as_array().unwrap() {
            let count = row["factors"].as_u64().unwrap() as usize;
            let finite = finite_kernel(&lift(h, count), &times);
            for (i, lag) in case["lags"].as_array().unwrap().iter().enumerate() {
                close(
                    finite[lag.as_u64().unwrap() as usize][0],
                    row["values"][i].as_f64().unwrap(),
                    3e-12,
                );
            }
        }
    }
}

#[test]
fn eliminated_factor_reference_matches_full_paths_including_truncation() {
    let times = vec![0.0, 0.003, 0.009, 0.05, 0.2, 0.65, 1.0];
    let mut negative_seen = false;
    for h in [0.1, 0.3] {
        for count in [8, 64, 256] {
            let model = lift(h, count);
            let kernel = finite_kernel(&model, &times);
            let plan = RoughVolatilityPathPlan::compile(model.into(), times.clone()).unwrap();
            for unit in 0..64 {
                let z = plan.pseudo_shocks(91, unit, RandomDomain::Valuation);
                let expected = reference_path(&heston(h), &times, &kernel, &z);
                let actual = plan.evolve_path(100.0, &z).unwrap();
                assert_eq!(actual.negative_variance_nodes, expected.negative);
                negative_seen |= expected.negative > 0;
                for i in 0..times.len() {
                    close(actual.forwards[i], expected.forwards[i], 2e-12);
                    close(actual.latent_states[i], expected.raw[i], 2e-13);
                    close(actual.variances[i], expected.variance[i], 2e-13);
                }
            }
        }
    }
    assert!(negative_seen, "truncation coverage must not be vacuous");
}

#[test]
fn brownian_boundary_and_factor_split_preserve_paths() {
    let times = grid(1.0, 32);
    let h = heston(0.5);
    let ordinary = LiftedHeston::from_rough(&h, 256, 1.2).unwrap();
    let one = RoughVolatilityPathPlan::compile(ordinary.clone().into(), times.clone()).unwrap();
    let split = LiftedHeston::new(
        0.04,
        0.7,
        0.055,
        0.18,
        -0.65,
        vec![0.125, 0.25, 0.625],
        vec![0.0; 3],
    )
    .unwrap();
    let split = RoughVolatilityPathPlan::compile(split.into(), times.clone()).unwrap();
    let limit = infinite_kernel(0.5, 1.0 / 32.0, 32);
    for unit in 0..64 {
        let z = one.pseudo_shocks(1973, unit, RandomDomain::Valuation);
        let a = one.evolve_path(100.0, &z).unwrap();
        let b = split.evolve_path(100.0, &z).unwrap();
        let c = reference_path(&h, &times, &limit, &z);
        for i in 0..times.len() {
            close(a.forwards[i], b.forwards[i], 2e-13);
            close(a.forwards[i], c.forwards[i], 2e-13);
        }
    }
    assert_eq!(ordinary.weights(), &[1.0]);
    assert_eq!(ordinary.rates(), &[0.0]);
}

#[test]
fn public_prices_and_errors_match_eliminated_factor_reference() {
    for n in [8, 64] {
        for count in [8, 256] {
            let model = lift(0.1, count);
            let times = grid(1.0, n);
            let kernel = finite_kernel(&model, &times);
            for antithetic in [false, true] {
                for strike in STRIKES {
                    let mut request: Value = serde_json::from_str(include_str!(
                        "../../../fixtures/v2/pricing_request.golden.json"
                    ))
                    .unwrap();
                    request["product"]["strike"] = json!(strike);
                    request["market"]["discount_curve"]["discount_factors"] = json!([1.0, 1.0]);
                    request["market"]["dividend_curve"]["discount_factors"] = json!([1.0, 1.0]);
                    request["engine"] = json!({"type":"pseudo_monte_carlo","master_seed":91,
                        "independent_sampling_units":64,"variance_reduction":{
                            "antithetic":antithetic,"brownian_bridge":false}});
                    let request = parse_request_json(
                        &serde_json::to_vec(&request).unwrap(),
                        JsonLimits::DEFAULT,
                    )
                    .unwrap();
                    let plan = RoughVolatilityPricingPlan::compile(
                        &request,
                        model.clone().into(),
                        1.0 / n as f64,
                        ExecutionPolicy::new(2, Some(32)).unwrap(),
                    )
                    .unwrap();
                    assert_eq!(plan.time_nodes(), times);
                    let values: Vec<f64> = (0..64)
                        .map(|unit| {
                            let mut z =
                                plan.path_plan()
                                    .pseudo_shocks(91, unit, RandomDomain::Valuation);
                            let a = reference_path(&heston(0.1), &times, &kernel, &z);
                            let plus = (a.forwards[n] - strike).max(0.0);
                            if !antithetic {
                                return plus;
                            }
                            for x in &mut z {
                                *x = -*x;
                            }
                            let b = reference_path(&heston(0.1), &times, &kernel, &z);
                            0.5 * (plus + (b.forwards[n] - strike).max(0.0))
                        })
                        .collect();
                    let (mean, se) = stats(&values);
                    let actual = plan.evaluate().unwrap();
                    assert!(se > 0.0);
                    close(actual.value, mean, 5e-12);
                    close(actual.standard_error, se, 5e-12);
                    assert_eq!(actual.independent_sampling_units, 64);
                    assert_eq!(actual.evaluated_paths, if antithetic { 128 } else { 64 });
                }
            }
        }
    }
}

// Each sample is one antithetic UNIT, not two independent observations.
#[derive(Clone, Default)]
struct Sample {
    prices: [[f64; 3]; 7],
    forward: [f64; 7],
    negative_nodes: [u64; 7],
}

#[test]
#[ignore = "factor-price numerical acceptance; execute explicitly in release"]
fn fixed_grid_factor_prices_approach_infinite_measure_reference() {
    let mut failures = Vec::new();
    for h in [0.1, 0.3] {
        for maturity in [0.25, 1.0] {
            let times = grid(maturity, STEPS);
            let target = infinite_kernel(h, maturity / STEPS as f64, STEPS);
            let plans: Vec<_> = FACTORS
                .iter()
                .map(|&count| {
                    RoughVolatilityPathPlan::compile(lift(h, count).into(), times.clone()).unwrap()
                })
                .collect();
            for seed in [91, 1973] {
                let mut samples = vec![Sample::default(); UNITS];
                let chunk_size = UNITS / 4;
                std::thread::scope(|scope| {
                    for (chunk_id, chunk) in samples.chunks_mut(chunk_size).enumerate() {
                        let times = &times;
                        let target = &target;
                        let plans = &plans;
                        scope.spawn(move || {
                            for (offset, sample) in chunk.iter_mut().enumerate() {
                                let unit = (chunk_id * chunk_size + offset) as u64;
                                let mut z =
                                    plans[0].pseudo_shocks(seed, unit, RandomDomain::Valuation);
                                for _ in 0..2 {
                                    let reference = reference_path(&heston(h), times, target, &z);
                                    let mut terminals = [0.0; 7];
                                    terminals[0] = reference.forwards[STEPS];
                                    sample.negative_nodes[0] += reference.negative;
                                    for (i, plan) in plans.iter().enumerate() {
                                        let path = plan.evolve_path(100.0, &z).unwrap();
                                        terminals[i + 1] = path.forwards[STEPS];
                                        sample.negative_nodes[i + 1] +=
                                            path.negative_variance_nodes;
                                    }
                                    for (i, &terminal) in terminals.iter().enumerate() {
                                        sample.forward[i] += 0.5 * terminal;
                                        for (j, &strike) in STRIKES.iter().enumerate() {
                                            sample.prices[i][j] +=
                                                0.5 * (terminal - strike).max(0.0);
                                        }
                                    }
                                    for x in &mut z {
                                        *x = -*x;
                                    }
                                }
                            }
                        });
                    }
                });
                for index in 0..7 {
                    let values: Vec<_> = samples.iter().map(|s| s.forward[index]).collect();
                    let (mean, se) = stats(&values);
                    let negative: u64 = samples.iter().map(|s| s.negative_nodes[index]).sum();
                    let fraction = negative as f64 / (2 * UNITS * STEPS) as f64;
                    let count = if index == 0 { 0 } else { FACTORS[index - 1] };
                    println!(
                        "LIFT_FORWARD H={h} T={maturity} seed={seed} factors={count} units={UNITS} mean={mean:.12} se={se:.12} negative_fraction={fraction:.12}"
                    );
                    if !(se > 0.0 && se <= 0.15 && (mean - 100.0).abs() <= 5.0 * se) {
                        failures.push(format!("forward H={h} T={maturity} seed={seed} factors={count}: mean={mean},se={se}"));
                    }
                }
                for (j, &strike) in STRIKES.iter().enumerate() {
                    let target_values: Vec<_> = samples.iter().map(|s| s.prices[0][j]).collect();
                    let (reference_price, reference_se) = stats(&target_values);
                    for (i, &count) in FACTORS.iter().enumerate() {
                        let model_values: Vec<_> =
                            samples.iter().map(|s| s.prices[i + 1][j]).collect();
                        let differences: Vec<_> = samples
                            .iter()
                            .map(|s| s.prices[i + 1][j] - s.prices[0][j])
                            .collect();
                        let (price, price_se) = stats(&model_values);
                        let (gap, se) = stats(&differences);
                        let bound = gap.abs() + 4.0 * se;
                        println!(
                            "LIFT_PRICE H={h} T={maturity} seed={seed} factors={count} strike={strike} steps={STEPS} units={UNITS} reference={reference_price:.12} reference_se={reference_se:.12} price={price:.12} price_se={price_se:.12} gap={gap:.12} paired_se={se:.12} bound={bound:.12}"
                        );
                        assert!(gap.is_finite() && se.is_finite() && se > 0.0);
                        if count == 256 && (bound > PRICE_BUDGET || se > SE_CAP) {
                            failures.push(format!("price H={h} T={maturity} seed={seed} K={strike}: bound={bound},se={se}"));
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "fixed factor-price/moment budgets failed: {failures:#?}"
    );
}
