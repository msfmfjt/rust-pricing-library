//! Smoothing-width comparisons at one fixed calibration and execution grid.
//! The hard payoff is a price reference, never a pathwise Delta reference.

use super::lsv_path_refinement_tests::{
    Contract, EXPIRY, FIXING, PAYMENT, compile, paths, request_value,
};
use super::lsv_refinement_tests::mean_se;
use super::*;
use serde_json::{Value, json};

const WIDTHS: [f64; 5] = [8.0, 4.0, 2.0, 1.0, 0.5];
const BUMPS: [f64; 2] = [0.5, 0.25];

#[derive(Clone, Copy)]
struct Observation {
    post: f64,
    spot_derivative: f64,
    cash_jump: f64,
}

fn observations(path: &StochasticDividendPathPlan, z: &[f64]) -> [Observation; 2] {
    let states = path.evolve_path(z).unwrap();
    std::array::from_fn(|j| {
        let time = [FIXING, EXPIRY][j];
        let i = path
            .times()
            .binary_search_by(|t| t.total_cmp(&time))
            .unwrap();
        let [a, b, c] = path.nodes()[i].coefficients();
        Observation {
            post: a * states[i].equity() + b * states[i].dividend() + c,
            spot_derivative: (0.98_f64 / 0.95).powf(time) * states[i].equity(),
            cash_jump: [5.0, 3.0][j] * states[i].dividend(),
        }
    })
}

fn indicator(x: f64, width: f64) -> (f64, f64) {
    let t = (x / width).clamp(-1.0, 1.0);
    (
        0.5 + 15.0 * t / 16.0 - 5.0 * t.powi(3) / 8.0 + 3.0 * t.powi(5) / 16.0,
        15.0 * (1.0 - t * t).powi(2) / (16.0 * width),
    )
}

fn positive_part(x: f64, width: f64) -> f64 {
    if x.abs() >= width {
        x.max(0.0)
    } else {
        let t = x / width;
        width * (5.0 + 16.0 * t + 15.0 * t * t - 5.0 * t.powi(4) + t.powi(6)) / 32.0
    }
}

fn smooth(obs: &[Observation; 2], width: f64, spot_shift: f64) -> [f64; 2] {
    let mut survival = 1.0;
    let mut derivative = 0.0;
    for o in obs {
        let score =
            o.post + spot_shift * o.spot_derivative - 105.0 + positive_part(o.cash_jump, width);
        let (hit, slope) = indicator(score, width);
        derivative = derivative * (1.0 - hit) - survival * slope * o.spot_derivative;
        survival *= 1.0 - hit;
    }
    let intrinsic = obs[1].post + spot_shift * obs[1].spot_derivative - 80.0;
    let vanilla = intrinsic.max(0.0);
    let delta = if intrinsic > 0.0 {
        obs[1].spot_derivative
    } else {
        0.0
    };
    let discount = 0.95_f64.powf(PAYMENT);
    [
        discount * vanilla * (1.0 - survival),
        discount * (delta * (1.0 - survival) - vanilla * derivative),
    ]
}

fn hard_price(obs: &[Observation; 2], spot_shift: f64) -> f64 {
    let touched = obs
        .iter()
        .any(|o| o.post + spot_shift * o.spot_derivative + o.cash_jump.max(0.0) >= 105.0);
    if touched {
        0.95_f64.powf(PAYMENT) * (obs[1].post + spot_shift * obs[1].spot_derivative - 80.0).max(0.0)
    } else {
        0.0
    }
}

fn set_width(value: &mut Value, width: Option<f64>) {
    if let Some(width) = width {
        value["risk"]["payoff_smoothing"] = json!({"type":"compact_c2", "half_width":width});
    } else {
        value["risk"]
            .as_object_mut()
            .unwrap()
            .remove("payoff_smoothing");
    }
}

// One independent observation is an antithetic pair. Evolve each sign once
// and reuse its states across every width, hard payoff and finite Spot bump.
fn unit(path: &StochasticDividendPathPlan, z: &[f64]) -> (Vec<[f64; 2]>, f64, [f64; 2]) {
    let plus = observations(path, z);
    let minus = observations(path, &z.iter().map(|x| -x).collect::<Vec<_>>());
    let values = WIDTHS
        .iter()
        .map(|&width| {
            let a = smooth(&plus, width, 0.0);
            let b = smooth(&minus, width, 0.0);
            std::array::from_fn(|j| (a[j] + b[j]) * 0.5)
        })
        .collect();
    let hard = 0.5 * (hard_price(&plus, 0.0) + hard_price(&minus, 0.0));
    let finite_differences = BUMPS.map(|b| {
        (hard_price(&plus, b) - hard_price(&plus, -b) + hard_price(&minus, b)
            - hard_price(&minus, -b))
            / (4.0 * b)
    });
    (values, hard, finite_differences)
}

#[test]
fn barrier_width_payoffs_and_errors_match_public_estimators() {
    for h in [0.1, 0.3] {
        let mut v = request_value(Contract::Barrier);
        let base = compile(&v, h, 0.6, 0.7);
        let path = paths(&base, h, 0.6, 0.7).pop().unwrap();
        let rng = Philox4x32::from_seed(193);
        let samples = (0..64)
            .map(|p| {
                let z = (0..path.random_dimension())
                    .map(|d| {
                        rng.standard_normal(RandomCoordinate::new(p, d, RandomDomain::Valuation))
                    })
                    .collect::<Vec<_>>();
                unit(&path, &z)
            })
            .collect::<Vec<_>>();
        for (i, width) in WIDTHS.into_iter().map(Some).chain([None]).enumerate() {
            set_width(&mut v, width);
            let mut plan = compile(&v, h, 0.6, 0.7);
            assert_eq!(plan.lsv_squared_leverage(), base.lsv_squared_leverage());
            plan.path = path.clone();
            let actual = if width.is_some() {
                let risk = plan.evaluate_lsv_spot_risk().unwrap();
                vec![
                    (risk.price.value, risk.price.standard_error),
                    (risk.delta, risk.delta_standard_error),
                ]
            } else {
                let price = plan.evaluate().unwrap();
                assert!(plan.evaluate_lsv_spot_risk().is_err());
                assert_eq!(price, plan.evaluate().unwrap());
                vec![(price.value, price.standard_error)]
            };
            for (j, (mean, se)) in actual.into_iter().enumerate() {
                let values = samples
                    .iter()
                    .map(|s| if width.is_some() { s.0[i][j] } else { s.1 })
                    .collect::<Vec<_>>();
                let expected = mean_se(&values);
                assert!(expected.0 > 0.0 && expected.1 > 0.0);
                assert!((mean - expected.0).abs() < 2e-10);
                assert!((se - expected.1).abs() < 2e-10);
            }
        }
    }
}

#[test]
fn barrier_width_derivatives_and_exterior_hard_payoff_are_consistent() {
    for width in WIDTHS {
        for cash in [-2.0, 0.0, 3.0] {
            let obs = [
                Observation {
                    // Exercise the interior transition at every width and
                    // for either sign of the jump, including zero.
                    post: 105.0 - positive_part(cash, width) + 0.2 * width,
                    spot_derivative: 0.9,
                    cash_jump: cash,
                },
                Observation {
                    post: 94.0,
                    spot_derivative: 1.1,
                    cash_jump: 3.0,
                },
            ];
            let bump = 1e-5;
            assert!(smooth(&obs, width, 0.0)[1] > 0.0);
            let central =
                (smooth(&obs, width, bump)[0] - smooth(&obs, width, -bump)[0]) / (2.0 * bump);
            assert!((central - smooth(&obs, width, 0.0)[1]).abs() < 2e-8);
        }
        for stock in [70.0, 130.0] {
            let obs = [Observation {
                post: stock,
                spot_derivative: 1.0,
                cash_jump: 3.0,
            }; 2];
            assert_eq!(smooth(&obs, width, 0.0)[0], hard_price(&obs, 0.0));
        }
    }
}

#[test]
fn hard_barrier_spot_shifts_match_full_recalibration() {
    for h in [0.1, 0.3] {
        let mut v = request_value(Contract::Barrier);
        set_width(&mut v, None);
        let base = compile(&v, h, 0.6, 0.7);
        let rng = Philox4x32::from_seed(193);
        for shift in [-0.5, -0.25, 0.25, 0.5] {
            v["market"]["spot"] = json!(100.0 + shift);
            let plan = compile(&v, h, 0.6, 0.7);
            let expected = (0..64)
                .map(|p| {
                    let z = (0..base.path.random_dimension())
                        .map(|d| {
                            rng.standard_normal(RandomCoordinate::new(
                                p,
                                d,
                                RandomDomain::Valuation,
                            ))
                        })
                        .collect::<Vec<_>>();
                    let a = observations(&base.path, &z);
                    let b = observations(&base.path, &z.iter().map(|x| -x).collect::<Vec<_>>());
                    (hard_price(&a, shift) + hard_price(&b, shift)) * 0.5
                })
                .collect::<Vec<_>>();
            let actual = plan.evaluate().unwrap();
            let expected = mean_se(&expected);
            assert!((actual.value - expected.0).abs() < 2e-10);
            assert!((actual.standard_error - expected.1).abs() < 2e-10);
        }
    }
}

#[test]
#[ignore = "release-mode rough-LSV Barrier smoothing width comparison"]
fn rough_lsv_barrier_smoothing_width_ladder() {
    const UNITS: u64 = 1_048_576;
    let mut failures = Vec::new();
    for h in [0.1, 0.3] {
        let base = compile(&request_value(Contract::Barrier), h, 0.6, 0.7);
        let path = paths(&base, h, 0.6, 0.7).pop().unwrap();
        for seed in [193, 877] {
            let rng = Philox4x32::from_seed(seed);
            let mut estimates = vec![[Vec::new(), Vec::new()]; WIDTHS.len()];
            let mut price_gaps = vec![Vec::new(); WIDTHS.len()];
            let mut delta_gaps = vec![Vec::new(); WIDTHS.len() - 1];
            let mut hard = Vec::new();
            let mut bumps = [Vec::new(), Vec::new()];
            let mut bump_gaps = [Vec::new(), Vec::new()];
            for p in 0..UNITS {
                let z = (0..path.random_dimension())
                    .map(|d| {
                        rng.standard_normal(RandomCoordinate::new(p, d, RandomDomain::Valuation))
                    })
                    .collect::<Vec<_>>();
                let (values, exact, fd) = unit(&path, &z);
                hard.push(exact);
                for (i, value) in values.iter().enumerate() {
                    for j in 0..2 {
                        estimates[i][j].push(value[j]);
                    }
                    price_gaps[i].push(value[0] - exact);
                    if i > 0 {
                        delta_gaps[i - 1].push(value[1] - values[i - 1][1]);
                    }
                }
                for j in 0..2 {
                    bumps[j].push(fd[j]);
                    bump_gaps[j].push(values.last().unwrap()[1] - fd[j]);
                }
            }
            let exact = mean_se(&hard);
            for (i, &width) in WIDTHS.iter().enumerate() {
                let price = mean_se(&estimates[i][0]);
                let delta = mean_se(&estimates[i][1]);
                let price_gap = mean_se(&price_gaps[i]);
                let delta_gap = if i > 0 {
                    Some(mean_se(&delta_gaps[i - 1]))
                } else {
                    None
                };
                assert!(
                    [price.0, price.1, delta.0, delta.1, price_gap.0, price_gap.1]
                        .into_iter()
                        .all(f64::is_finite)
                );
                println!(
                    "{}",
                    json!({"kind":"width", "hurst":h, "seed":seed,
                    "antithetic_units":UNITS, "steps":128, "half_width":width,
                    "price":price.0, "price_se":price.1, "delta":delta.0, "delta_se":delta.1,
                    "hard_price":exact.0, "hard_price_se":exact.1,
                    "price_minus_hard":price_gap.0, "price_gap_se":price_gap.1,
                    "previous_width":i.checked_sub(1).map(|j| WIDTHS[j]),
                    "delta_minus_previous":delta_gap.map(|x| x.0), "delta_gap_se":delta_gap.map(|x| x.1),
                    "calibration_steps":16,"calibration_particles":512,"calibration_seed":42,
                    "scope":"fixed_grid_fixed_surface_smoothing"})
                );
                if i == WIDTHS.len() - 1 {
                    let d = delta_gap.unwrap();
                    let passes = price_gap.0.abs() + 4.0 * price_gap.1 < 0.02
                        && price_gap.1 < 0.004
                        && d.0.abs() + 4.0 * d.1 < 0.01
                        && d.1 < 0.002;
                    if !passes {
                        failures.push(format!(
                            "H={h}, seed={seed}: price gap={price_gap:?}, delta gap={d:?}"
                        ));
                    }
                }
            }
            for (j, bump) in BUMPS.iter().enumerate() {
                let fd = mean_se(&bumps[j]);
                let gap = mean_se(&bump_gaps[j]);
                assert!([fd.0, fd.1, gap.0, gap.1].into_iter().all(f64::is_finite));
                println!(
                    "{}",
                    json!({"kind":"hard_finite_bump", "hurst":h, "seed":seed,
                    "steps":128, "antithetic_units":UNITS, "spot_bump":bump,
                    "finite_difference":fd.0, "finite_difference_se":fd.1,
                    "smoothing_half_width":0.5, "smoothed_delta_minus_finite_difference":gap.0,
                    "paired_gap_se":gap.1, "scope":"finite_bump_diagnostic_not_exact_delta"})
                );
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
