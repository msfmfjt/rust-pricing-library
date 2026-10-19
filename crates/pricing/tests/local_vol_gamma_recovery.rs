use pricing::market::{
    ImpliedVarianceSurface, LocalVarianceGrid, MarketError, ThetaRegion, TotalVarianceDerivatives,
};
use pricing::mc::{LocalVolLogEulerPlan, LocalVolTimeGrid, RandomDomain};
use pricing::risk::{
    AnalyticCallDensityRow, RiskConfigError, analytic_call_density_row_from_surface,
    recover_equation11_gamma_row,
};
use serde_json::Value;
use std::f64::consts::PI;

const FIXTURE: &str = include_str!("../../../fixtures/local-vol/gamma-recovery-v0.1.json");

struct Flat(f64);

impl ImpliedVarianceSurface for Flat {
    fn total_variance_derivatives(
        &self,
        time: f64,
        _: f64,
    ) -> Result<TotalVarianceDerivatives, MarketError> {
        let variance = self.0 * self.0;
        Ok(TotalVarianceDerivatives {
            total_variance: variance * time,
            log_moneyness_derivative: 0.0,
            log_moneyness_second_derivative: 0.0,
            time_derivative: variance,
            theta: variance * time,
            theta_derivative: variance,
            theta_region: ThetaRegion::Interpolated,
        })
    }
}

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

fn gamma(spot: f64, strike: f64, variance: f64) -> f64 {
    let sd = variance.sqrt();
    let d1 = ((spot / strike).ln() + variance * 0.5) / sd;
    (-0.5 * d1 * d1).exp() / ((2.0 * PI).sqrt() * spot * sd)
}

fn row(
    nodes: Vec<f64>,
    time: f64,
    forward: f64,
    sigma: f64,
    threshold: f64,
) -> AnalyticCallDensityRow {
    analytic_call_density_row_from_surface(&Flat(sigma), time, forward, nodes, threshold).unwrap()
}

// Analytic conditional call vega for a bump to just one future interval.
// This is separately checked against Decimal price differences in the fixture.
fn conditional_node_adjoints(
    density: &AnalyticCallDensityRow,
    sigma: f64,
    expiry: f64,
    strike: f64,
    dt: f64,
) -> Vec<f64> {
    let nodes = density.log_moneyness_nodes();
    nodes
        .iter()
        .copied()
        .enumerate()
        .map(|(i, x)| {
            let width = if i == 0 {
                nodes[1] - x
            } else if i + 1 == nodes.len() {
                x - nodes[i - 1]
            } else {
                nodes[i + 1] - nodes[i - 1]
            };
            let s = density.forward().get() * x.exp();
            let remaining_variance = sigma * sigma * (expiry - density.maturity().get());
            let d1 = ((s / strike).ln() + remaining_variance * 0.5) / remaining_variance.sqrt();
            let conditional_vega = s * (-0.5 * d1 * d1).exp() / (2.0 * PI).sqrt() * sigma * dt
                / remaining_variance.sqrt();
            0.5 * width * s * density.call_densities()[i] * conditional_vega
        })
        .collect()
}

#[test]
fn recovery_matches_independent_conditional_price_bumps_on_nonuniform_grid() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let nodes = numbers(&fixture["log_moneyness_nodes"]);
    let sigma = fixture["volatility"].as_f64().unwrap();
    let time = fixture["source_maturity"].as_f64().unwrap();
    let dt = fixture["delta_t"].as_f64().unwrap();
    let density = row(
        nodes,
        time,
        fixture["forward"].as_f64().unwrap(),
        sigma,
        1e-12,
    );
    for case in fixture["cases"].as_array().unwrap() {
        let adjoints = numbers(&case["integrated_local_vol_adjoints"]);
        let recovered = recover_equation11_gamma_row(
            &density,
            &adjoints,
            &vec![sigma * sigma; adjoints.len()],
            dt,
        )
        .unwrap();
        assert_eq!(recovered.source_maturity().get(), time);
        assert_eq!(recovered.recovered_maturity().get(), time + dt);
        assert_eq!(recovered.delta_t().get(), dt);
        assert_eq!(recovered.policy_label(), "equation_11_first_order_v1");
        assert_eq!(recovered.truncation_order(), "O(delta_t_k)");
        assert_eq!(recovered.excluded_local_vega(), 0.0);
        for (actual, expected) in recovered
            .gamma_values()
            .iter()
            .zip(numbers(&fixture["current_time_gamma"]))
        {
            close(
                *actual,
                expected,
                fixture["gamma_relative_tolerance"].as_f64().unwrap(),
            );
        }
        let expected = conditional_node_adjoints(&density, sigma, 1.0, 105.0, dt);
        for (actual, expected) in adjoints.iter().zip(expected) {
            close(*actual, expected, 1e-7);
        }
    }
}

#[test]
fn recovery_and_propagation_show_first_order_time_error_for_a_vanilla() {
    let sigma = 0.2;
    let time = 0.4;
    let nodes = (0..=2048).map(|i| -1.0 + f64::from(i) / 1024.0).collect();
    let density = row(nodes, time, 100.0, sigma, 1e-14);
    let mut errors = Vec::new();
    for dt in [0.0625, 0.03125, 0.015625, 0.0078125] {
        let adjoints = conditional_node_adjoints(&density, sigma, 1.0, 100.0, dt);
        let recovered = recover_equation11_gamma_row(
            &density,
            &adjoints,
            &vec![sigma * sigma; adjoints.len()],
            dt,
        )
        .unwrap();
        let atm = recovered
            .strike_nodes()
            .iter()
            .position(|k| *k == 100.0)
            .unwrap();
        // Equation (11) labels the estimate as next-time Gamma. The price-bump
        // oracle instead gives current-time Gamma, hence a real O(dt) defect.
        let next_reference = gamma(100.0, 100.0, sigma * sigma * (1.0 - time - dt));
        let recovery_error = (recovered.gamma_values()[atm] - next_reference).abs();
        let propagated = recovered.propagate_to_source(100.0, sigma * sigma).unwrap();
        let current_reference = gamma(100.0, 100.0, sigma * sigma * (1.0 - time));
        let propagation_error = (propagated - current_reference).abs();
        println!(
            "dt={dt}: recovery_error={recovery_error:.9e}, propagation_error={propagation_error:.9e}"
        );
        errors.push((recovery_error, propagation_error));
    }
    for pair in errors.windows(2) {
        for (coarse, fine) in [(pair[0].0, pair[1].0), (pair[0].1, pair[1].1)] {
            let order = (coarse / fine).log2();
            assert!((0.85..1.15).contains(&order), "observed time order={order}");
        }
    }
}

#[test]
fn propagation_of_recovered_rows_converges_under_strike_refinement() {
    let mut errors = Vec::new();
    let dt = 0.0625;
    for intervals in [128, 256, 512] {
        let nodes = (0..=intervals)
            .map(|i| -1.0 + 2.0 * f64::from(i) / f64::from(intervals))
            .collect();
        let density = row(nodes, 0.4, 100.0, 0.2, 1e-14);
        let adjoints = conditional_node_adjoints(&density, 0.2, 1.0, 100.0, dt);
        let recovered =
            recover_equation11_gamma_row(&density, &adjoints, &vec![0.04; adjoints.len()], dt)
                .unwrap();
        // Independent GBM semigroup: applying the Gamma kernel adds variance.
        let expected = gamma(100.0, 100.0, 0.04 * (0.6 + dt));
        let actual = recovered.propagate_to_source(100.0, 0.04).unwrap();
        let error = (actual - expected).abs();
        println!("intervals={intervals}: spatial_error={error:.9e}");
        errors.push(error);
    }
    for pair in errors.windows(2) {
        let order = (pair[0] / pair[1]).log2();
        assert!(
            (1.8..2.2).contains(&order),
            "observed spatial order={order}"
        );
    }
}

#[test]
fn active_domain_keeps_original_hat_areas_and_signed_exclusions() {
    let density = row(vec![-1.0, -0.15, 0.0, 0.18, 1.0], 0.5, 100.0, 0.2, 0.05);
    let recovered =
        recover_equation11_gamma_row(&density, &[-7.0, 1.0, -2.0, 3.0, 5.0], &[0.04; 5], 0.1)
            .unwrap();
    assert_eq!(recovered.active_domain().start_index(), 1);
    assert_eq!(recovered.active_domain().end_index(), 3);
    assert_eq!(recovered.input_local_vega(), 0.0);
    assert_eq!(recovered.retained_local_vega(), 2.0);
    assert_eq!(recovered.excluded_local_vega(), -2.0);
    assert_eq!(
        recovered.excluded_probability_mass(),
        density.excluded_probability_mass()
    );
    let k = recovered.strike_nodes()[0];
    let reconstructed =
        recovered.gamma_values()[0] * k.powi(3) * density.call_densities()[1] * 0.2 * 0.1 * 0.5;
    close(reconstructed, 1.0, 1e-13); // Original hat area is 0.5, not the trimmed 0.075.
    assert!(recovered.gamma_values()[1] < 0.0);
}

#[test]
fn recovery_and_transition_preserve_currency_units_and_signed_scaling() {
    let nodes = vec![-0.3, -0.1, 0.0, 0.12, 0.3];
    let density = row(nodes.clone(), 0.4, 100.0, 0.2, 1e-12);
    let base = [1.0, 2.0, -1.0, 3.0, 0.5];
    let recovered = recover_equation11_gamma_row(&density, &base, &[0.04; 5], 0.1).unwrap();
    for scale in [1e-70, 1.0, 1e70] {
        let scaled_density = row(nodes.clone(), 0.4, 100.0 * scale, 0.2, 1e-12);
        let adjoints = base.map(|a| -2.0 * scale * a);
        let scaled =
            recover_equation11_gamma_row(&scaled_density, &adjoints, &[0.04; 5], 0.1).unwrap();
        for (actual, original) in scaled.gamma_values().iter().zip(recovered.gamma_values()) {
            close(*actual, -2.0 * original / scale, 1e-12);
        }
        close(
            scaled.propagate_to_source(100.0 * scale, 0.04).unwrap(),
            -2.0 * recovered.propagate_to_source(100.0, 0.04).unwrap() / scale,
            1e-10,
        );
    }
}

#[test]
fn recovery_rejects_bad_shape_values_time_and_single_node_transition() {
    let density = row(vec![-1.0, 0.0, 1.0], 0.5, 100.0, 0.2, 0.5);
    assert!(matches!(
        recover_equation11_gamma_row(&density, &[1.0], &[0.04; 3], 0.1),
        Err(RiskConfigError::VegaKtRecoveryLengthMismatch {
            quantity: "node adjoint",
            ..
        })
    ));
    assert!(matches!(
        recover_equation11_gamma_row(&density, &[1.0; 3], &[0.04], 0.1),
        Err(RiskConfigError::VegaKtRecoveryLengthMismatch {
            quantity: "local variance",
            ..
        })
    ));
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            recover_equation11_gamma_row(&density, &[1.0; 3], &[bad, 0.04, 0.04], 0.1).is_err()
        );
        assert!(recover_equation11_gamma_row(&density, &[1.0; 3], &[0.04; 3], bad).is_err());
    }
    assert!(
        recover_equation11_gamma_row(&density, &[f64::NAN, 1.0, 1.0], &[0.04; 3], 0.1).is_err()
    );
    assert!(matches!(
        recover_equation11_gamma_row(&density, &[1.0; 3], &[0.04; 3], 1e-30),
        Err(RiskConfigError::InvalidVegaKtRecoveryInterval { .. })
    ));
    assert!(
        analytic_call_density_row_from_surface(&Flat(0.2), 0.0, 100.0, vec![-0.1, 0.1], 1e-12)
            .is_err()
    );
    let recovered = recover_equation11_gamma_row(&density, &[1.0; 3], &[0.04; 3], 0.1).unwrap();
    assert_eq!(recovered.gamma_values().len(), 1);
    assert!(matches!(
        recovered.propagate_to_source(100.0, 0.04),
        Err(RiskConfigError::TooFewVegaKtTransitionStrikes { count: 1 })
    ));
}

#[test]
fn actual_native_path_adjoints_feed_recovery_and_match_a_row_crn_bump() {
    let times = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let nodes = vec![-0.5, -0.2, 0.0, 0.2, 0.5];
    let plan = LocalVolLogEulerPlan::with_constant_forward(
        LocalVolTimeGrid::compile(times.clone(), 0.25).unwrap(),
        100.0,
    )
    .unwrap();
    let make_grid = |bump: f64| {
        let mut values = vec![0.04; times.len() * nodes.len()];
        values[nodes.len()..2 * nodes.len()].fill((0.2 + bump).powi(2));
        LocalVarianceGrid::new(times.clone(), nodes.clone(), values, 1e-5, 1.0).unwrap()
    };
    let grid = make_grid(0.0);
    let up = make_grid(1e-5);
    let down = make_grid(-1e-5);
    let mut adjoints = vec![0.0; nodes.len()];
    let mut bump_sum = 0.0;
    let paths = 4096;
    for sample in 0..paths {
        let shocks = plan
            .path_shocks(41, sample, RandomDomain::Valuation, false)
            .unwrap();
        let path = plan.evolve_path(&grid, 100.0, &shocks).unwrap();
        let terminal = *path.states().last().unwrap();
        let reverse = path
            .reverse_terminal(2.0 * terminal, grid.values().len(), nodes.len())
            .unwrap();
        for (a, v) in adjoints
            .iter_mut()
            .zip(&reverse.local_variance_value_adjoints()[nodes.len()..2 * nodes.len()])
        {
            *a += 0.4 * v / paths as f64;
        }
        let p_up = *plan
            .evolve_path(&up, 100.0, &shocks)
            .unwrap()
            .states()
            .last()
            .unwrap();
        let p_down = *plan
            .evolve_path(&down, 100.0, &shocks)
            .unwrap()
            .states()
            .last()
            .unwrap();
        bump_sum += (p_up.powi(2) - p_down.powi(2)) / 2e-5 / paths as f64;
    }
    let density = row(nodes, 0.25, 100.0, 0.2, 1e-12);
    let recovered = recover_equation11_gamma_row(&density, &adjoints, &[0.04; 5], 0.25).unwrap();
    close(recovered.input_local_vega(), bump_sum, 2e-7);
    assert!(
        recovered
            .propagate_to_source(100.0, 0.04)
            .unwrap()
            .is_finite()
    );
    println!(
        "native row AD={}, CRN={bump_sum}",
        recovered.input_local_vega()
    );
}
