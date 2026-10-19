use pricing::market::{
    ImpliedVarianceSurface, LocalVarianceGrid, MarketError, ThetaRegion, TotalVarianceDerivatives,
};
use pricing::mc::{
    LocalVolError, LocalVolLogEulerPlan, LocalVolPath, LocalVolTimeGrid, RandomDomain,
};
use pricing::risk::{analytic_call_density_row_from_surface, recover_equation11_gamma_row};
use serde_json::Value;

const FIXTURE: &str = include_str!("../../../fixtures/local-vol/step-adjoints-v0.1.json");

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn close(actual: f64, expected: f64, relative: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= relative * expected.abs(),
        "actual={actual:.17e}, expected={expected:.17e}"
    );
}

fn inputs(
    case: &Value,
) -> (
    LocalVolLogEulerPlan,
    LocalVarianceGrid,
    LocalVolPath,
    Vec<f64>,
) {
    let plan = LocalVolLogEulerPlan::new(
        LocalVolTimeGrid::compile(numbers(&case["simulation_times"]), 1.).unwrap(),
        numbers(&case["forwards"]),
    )
    .unwrap();
    let grid = LocalVarianceGrid::new(
        numbers(&case["grid_times"]),
        numbers(&case["nodes"]),
        numbers(&case["variances"]),
        1e-6,
        1.,
    )
    .unwrap();
    let path = plan
        .evolve_path(
            &grid,
            case["initial"].as_f64().unwrap(),
            &numbers(&case["shocks"]),
        )
        .unwrap();
    let states = path.states();
    let mut seeds: Vec<_> = numbers(&case["quadratic_weights"])
        .iter()
        .zip(states)
        .map(|(w, s)| 2. * w * s)
        .collect();
    let cross = case["cross_weight"].as_f64().unwrap();
    seeds[2] += cross * states[states.len() - 1];
    *seeds.last_mut().unwrap() += cross * states[2];
    (plan, grid, path, seeds)
}

#[test]
fn step_and_grid_adjoints_match_136_independent_decimal_price_bumps() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let mut comparisons = 0;
    let mut max_relative: f64 = 0.;
    for case in fixture["cases"].as_array().unwrap() {
        let (plan, grid, path, seeds) = inputs(case);
        let reverse = path
            .reverse_state_adjoints(&seeds, grid.values().len(), 3)
            .unwrap();
        let expected = &case["expected"];
        for (&actual, reference) in path.states().iter().zip(numbers(&expected["states"])) {
            close(actual, reference, 1e-13);
        }
        assert_eq!(reverse.step_adjoints().len(), plan.time_grid().step_count());
        assert!(plan.time_grid().step_count() > grid.time_nodes().len());
        for (i, step) in reverse.step_adjoints().iter().enumerate() {
            let mut row = vec![0.; 3];
            step.accumulate_local_volatility_nodes(&mut row).unwrap();
            let actual = [
                vec![
                    step.local_variance_adjoint(),
                    step.local_volatility_adjoint(),
                ],
                row,
            ]
            .concat();
            let expected = [
                vec![
                    expected["step_variance"][i].as_f64().unwrap(),
                    expected["step_volatility"][i].as_f64().unwrap(),
                ],
                numbers(&expected["step_hat_volatility"][i]),
            ]
            .concat();
            for (actual, expected) in actual.into_iter().zip(expected) {
                close(
                    actual,
                    expected,
                    fixture["relative_tolerance"].as_f64().unwrap(),
                );
                if expected != 0. {
                    max_relative = max_relative.max((actual / expected - 1.).abs());
                }
                comparisons += 1;
            }
        }
        for (&actual, expected) in reverse
            .local_variance_value_adjoints()
            .iter()
            .zip(numbers(&expected["grid_variance"]))
        {
            close(actual, expected, 1e-9);
            if expected != 0. {
                max_relative = max_relative.max((actual / expected - 1.).abs());
            }
            comparisons += 1;
        }
    }
    assert_eq!(comparisons, 136);
    println!("Decimal price derivatives={comparisons}, maximum_relative_error={max_relative:.9e}");
}

#[test]
fn stored_steps_reconstruct_original_grid_adjoints_bitwise() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let (_, grid, path, seeds) = inputs(case);
        let reverse = path
            .reverse_state_adjoints(&seeds, grid.values().len(), 3)
            .unwrap();
        let mut reconstructed = vec![0.; grid.values().len()];
        for (cache, step) in path.step_cache().iter().zip(reverse.step_adjoints()).rev() {
            cache.interpolation.transpose_accumulate(
                step.local_variance_adjoint(),
                &mut reconstructed,
                3,
            );
        }
        for (&a, &b) in reconstructed
            .iter()
            .zip(reverse.local_variance_value_adjoints())
        {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[test]
fn nonflat_grid_volatility_conversion_requires_the_effective_volatility_ratio() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let (_, grid, path, seeds) = inputs(&fixture["cases"][0]);
    let reverse = path
        .reverse_state_adjoints(&seeds, grid.values().len(), 3)
        .unwrap();
    let mut correct = [0.; 9];
    let mut naive = [0.; 9];
    for (cache, step) in path.step_cache().iter().zip(reverse.step_adjoints()).rev() {
        let mut row = vec![0.; 3];
        step.accumulate_local_volatility_nodes(&mut row).unwrap();
        for (offset, tw) in [
            (0, 1. - cache.interpolation.time_weight),
            (1, cache.interpolation.time_weight),
        ] {
            for (j, &adjoint) in row.iter().enumerate() {
                let index = (cache.interpolation.lower_time_index + offset) * 3 + j;
                correct[index] +=
                    adjoint * tw * (grid.values()[index] / cache.local_variance).sqrt();
                naive[index] += adjoint * tw;
            }
        }
    }
    let expected: Vec<_> = reverse
        .local_variance_value_adjoints()
        .iter()
        .zip(grid.values())
        .map(|(a, v)| 2. * v.sqrt() * a)
        .collect();
    for (&actual, &reference) in correct.iter().zip(&expected) {
        close(actual, reference, 1e-12);
    }
    assert!(naive.iter().zip(expected).any(|(a, b)| (a - b).abs() > 1.));
}

#[test]
fn steps_keep_initial_risk_and_respect_fixing_causality_and_signed_seeds() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let (plan, grid, path, _) = inputs(&fixture["cases"][0]);
    let mut seeds = vec![0.; path.states().len()];
    seeds[2] = 1.;
    let reverse = path
        .reverse_state_adjoints(&seeds, grid.values().len(), 3)
        .unwrap();
    let signed = path
        .reverse_state_adjoints(
            &seeds.iter().map(|s| -3.25 * s).collect::<Vec<_>>(),
            grid.values().len(),
            3,
        )
        .unwrap();
    assert_eq!(plan.time_grid().nodes()[0], 0.);
    assert_ne!(reverse.step_adjoints()[0].local_volatility_adjoint(), 0.);
    for (i, (step, scaled)) in reverse
        .step_adjoints()
        .iter()
        .zip(signed.step_adjoints())
        .enumerate()
    {
        close(
            scaled.local_volatility_adjoint(),
            -3.25 * step.local_volatility_adjoint(),
            1e-13,
        );
        if i >= 2 {
            assert_eq!(step.local_volatility_adjoint(), 0.);
        }
    }
}

#[test]
fn reverse_and_recovery_row_reject_bad_shapes_and_nonfinite_values() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let (_, grid, path, seeds) = inputs(&fixture["cases"][0]);
    let reverse = path
        .reverse_state_adjoints(&seeds, grid.values().len(), 3)
        .unwrap();
    let mut short_row = vec![1., 2.];
    assert!(
        reverse.step_adjoints()[0]
            .accumulate_local_volatility_nodes(&mut short_row)
            .is_err()
    );
    assert_eq!(short_row, vec![1., 2.]);
    for bad in [f64::NAN, f64::INFINITY, f64::MAX] {
        let mut bad_seeds = seeds.clone();
        *bad_seeds.last_mut().unwrap() = bad;
        assert!(matches!(
            path.reverse_state_adjoints(&bad_seeds, 9, 3),
            Err(LocalVolError::NonFiniteAdjoint { .. })
        ));
    }
    // Incompatible spatial counts fail instead of indexing a neighbouring row.
    assert!(path.reverse_state_adjoints(&seeds, 9, 2).is_err());
    let scaled = path
        .reverse_state_adjoints(&seeds.iter().map(|s| s * 1e300).collect::<Vec<_>>(), 9, 3)
        .unwrap();
    let mut row = vec![f64::MAX; 3];
    assert!(
        scaled.step_adjoints()[0]
            .accumulate_local_volatility_nodes(&mut row)
            .is_err()
    );
    assert_eq!(row, vec![f64::MAX; 3]);
}

struct Flat;
impl ImpliedVarianceSurface for Flat {
    fn total_variance_derivatives(
        &self,
        time: f64,
        _: f64,
    ) -> Result<TotalVarianceDerivatives, MarketError> {
        Ok(TotalVarianceDerivatives {
            total_variance: 0.04 * time,
            log_moneyness_derivative: 0.,
            log_moneyness_second_derivative: 0.,
            time_derivative: 0.04,
            theta: 0.04 * time,
            theta_derivative: 0.04,
            theta_region: ThetaRegion::Interpolated,
        })
    }
}

// Independent forward-only price with an additive volatility bump to ONE
// simulation interval. The original grid is queried again after the bump.
fn bumped_price(
    plan: &LocalVolLogEulerPlan,
    grid: &LocalVarianceGrid,
    shocks: &[f64],
    interval: usize,
    bump: f64,
) -> f64 {
    let mut state: f64 = 100.;
    for (i, &shock) in shocks.iter().enumerate() {
        let time = plan.time_grid().nodes()[i];
        let dt = plan.time_grid().nodes()[i + 1] - time;
        let variance = grid.interpolate(time, (state / 100.).ln()).unwrap().value;
        let sigma = variance.sqrt() + if i == interval { bump } else { 0. };
        state *= (-0.5 * sigma * sigma * dt + sigma * dt.sqrt() * shock).exp();
    }
    0.5 * state * state
}

#[test]
fn event_inserted_step_means_feed_positive_time_gamma_recovery_and_match_crn() {
    let nodes = vec![-0.8, -0.4, 0., 0.4, 0.8];
    let grid =
        LocalVarianceGrid::new(vec![0., 0.4, 1.], nodes.clone(), vec![0.04; 15], 1e-5, 1.).unwrap();
    let plan = LocalVolLogEulerPlan::with_constant_forward(
        LocalVolTimeGrid::compile(vec![0., 0.1, 0.23, 0.4, 0.7, 1.], 1.).unwrap(),
        100.,
    )
    .unwrap();
    let mut rows = vec![vec![0.; nodes.len()]; 5];
    let mut crn = [0.; 5];
    let count = 4096;
    for sample in 0..count {
        let shocks = plan
            .path_shocks(41, sample, RandomDomain::Valuation, false)
            .unwrap();
        let path = plan.evolve_path(&grid, 100., &shocks).unwrap();
        let reverse = path
            .reverse_terminal(*path.states().last().unwrap(), 15, 5)
            .unwrap();
        for (i, step) in reverse.step_adjoints().iter().enumerate() {
            step.accumulate_local_volatility_nodes(&mut rows[i])
                .unwrap();
            crn[i] += (bumped_price(&plan, &grid, &shocks, i, 1e-5)
                - bumped_price(&plan, &grid, &shocks, i, -1e-5))
                / 2e-5;
        }
    }
    for (i, row) in rows.iter_mut().enumerate() {
        for adjoint in row.iter_mut() {
            *adjoint /= count as f64;
        }
        crn[i] /= count as f64;
        let sum: f64 = row.iter().sum();
        close(sum, crn[i], 2e-7);
        let time = plan.time_grid().nodes()[i];
        let dt = plan.time_grid().nodes()[i + 1] - time;
        if i == 0 {
            assert!(
                analytic_call_density_row_from_surface(&Flat, time, 100., nodes.clone(), 1e-12)
                    .is_err()
            );
            assert_ne!(sum, 0.); // Retained, not falsely divided by a time-zero density.
        } else {
            let density =
                analytic_call_density_row_from_surface(&Flat, time, 100., nodes.clone(), 1e-12)
                    .unwrap();
            let recovered = recover_equation11_gamma_row(&density, row, &[0.04; 5], dt).unwrap();
            assert_eq!(recovered.source_maturity().get(), time);
            assert_eq!(recovered.recovered_maturity().get(), time + dt);
            close(recovered.input_local_vega(), crn[i], 2e-7);
            assert!(
                recovered
                    .gamma_values()
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
        println!(
            "step={i}, source={time}, dt={dt}, mean_AD={sum:.12}, CRN={:.12}",
            crn[i]
        );
    }
}
