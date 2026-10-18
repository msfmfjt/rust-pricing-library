use pricing::market::{ImpliedVarianceSurface, MarketError, ThetaRegion, TotalVarianceDerivatives};
use pricing::risk::{
    Equation11GammaRow, ReportingIvBasis, RiskConfigError, analytic_call_density_row_from_surface,
    integrate_local_gamma_total_variance_transition as total_gamma,
    integrate_piecewise_linear_local_gamma_transition as frozen_gamma, mass_lumped_hat_areas,
    project_vanilla_hedge_measure_to_reporting_iv as project, recover_equation11_gamma_row,
};
use serde_json::Value;
use std::f64::consts::PI;

const FIXTURE: &str = include_str!("../../../fixtures/local-vol/vegakt-paper-v0.1.json");

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
        "actual={actual:0.17e}, expected={expected:0.17e}"
    );
}

#[test]
fn total_spot_derivative_matches_independent_payoff_price_differences() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    for case in fixture["gamma_cases"].as_array().unwrap() {
        let actual = total_gamma(
            case["spot"].as_f64().unwrap(),
            case["variance"].as_f64().unwrap(),
            case["variance_first"].as_f64().unwrap(),
            case["variance_second"].as_f64().unwrap(),
            &numbers(&case["strikes"]),
            &numbers(&case["gamma"]),
        )
        .unwrap();
        let expected = case["expected_price_gamma"].as_f64().unwrap();
        println!(
            "{} relative_error={:0.9e}",
            case["id"],
            (actual / expected - 1.0).abs()
        );
        close(
            actual,
            expected,
            fixture["relative_tolerance"].as_f64().unwrap(),
        );
    }
}

#[test]
fn zero_variance_derivatives_preserve_frozen_operator_bitwise() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/gamma-numerics-v0.1.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let s = case["spot"].as_f64().unwrap();
        let v = case["variance"].as_f64().unwrap();
        let strikes = numbers(&case["strikes"]);
        let gamma = numbers(&case["gamma"]);
        assert_eq!(
            total_gamma(s, v, 0., 0., &strikes, &gamma)
                .unwrap()
                .to_bits(),
            frozen_gamma(s, v, &strikes, &gamma).unwrap().to_bits()
        );
    }
}

#[test]
fn curvature_can_make_the_total_kernel_signed_and_cannot_be_frozen() {
    let value = total_gamma(100., 0.04, 0., -0.0005, &[70., 130.], &[1., 1.]).unwrap();
    // d2/dS2 E[S_next^2/2] for v(S)=0.04-0.00025*(S-100)^2.
    close(value, -1.5 * 0.04_f64.exp(), 1e-13);
    assert!(frozen_gamma(100., 0.04, &[70., 130.], &[1., 1.]).unwrap() > 0.);
}

#[test]
fn total_operator_is_invariant_under_strike_unit_changes() {
    let expected = total_gamma(100., 0.04, 0.0004, -0.000003, &[90., 110.], &[1., 2.]).unwrap();
    for scale in [1e-70, 1e70] {
        close(
            total_gamma(
                100. * scale,
                0.04,
                0.0004 / scale,
                -0.000003 / scale / scale,
                &[90. * scale, 110. * scale],
                &[1., 2.],
            )
            .unwrap(),
            expected,
            1e-12,
        );
    }
}

#[test]
fn invalid_variance_jets_and_shapes_are_rejected() {
    for (v, first, second) in [
        (0., 0., 0.),
        (0.04, f64::NAN, 0.),
        (0.04, 0., f64::INFINITY),
    ] {
        assert!(total_gamma(100., v, first, second, &[90., 110.], &[1., 2.]).is_err());
    }
    assert!(total_gamma(100., 0.04, 0.0001, 0., &[90., 110.], &[1.]).is_err());
    assert!(total_gamma(100., 0.04, 0.0001, 0., &[90., 110.], &[1., f64::NAN]).is_err());
}

#[test]
fn remote_short_step_cells_do_not_request_unbounded_quadrature() {
    // The nonzero Gamma is nearly a billion normal standard deviations away.
    // Its mass is below binary64 range; quadrature panel counts must not scale
    // with that distance (roughly 400 million panels for these finite inputs).
    assert_eq!(
        total_gamma(100., 1e-20, 1e-23, 0., &[110., 110.000000005], &[0., 1.]).unwrap(),
        0.
    );
}

fn basis(fixture: &Value) -> ReportingIvBasis {
    let b = &fixture["projection_basis"];
    ReportingIvBasis::new(
        numbers(&b["maturities"]),
        numbers(&b["log_moneyness_nodes"]),
        numbers(&b["implied_volatilities"]),
    )
    .unwrap()
}

fn vega(spot: f64, time: f64, x: f64, sigma: f64) -> f64 {
    let sd = sigma * time.sqrt();
    let d1 = -x / sd + sd / 2.;
    spot * (-d1 * d1 / 2.).exp() / (2. * PI).sqrt() * time.sqrt()
}

#[test]
fn equation_nine_matches_independent_quoted_iv_price_bumps() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let basis = basis(&fixture);
    for case in fixture["projection_cases"].as_array().unwrap() {
        let time = case["maturity"].as_f64().unwrap();
        let projection = project(
            &basis,
            time,
            100.,
            &numbers(&case["strikes"]),
            &numbers(&case["hedge_weights"]),
        )
        .unwrap();
        for (i, (&risk, expected)) in projection
            .position_risk()
            .raw_buckets()
            .iter()
            .zip(numbers(&case["expected_raw_buckets"]))
            .enumerate()
        {
            close(risk, expected, 1e-9);
            let node_time = basis.maturity_nodes()[i / 3];
            let node_vega = vega(
                100.,
                node_time,
                basis.log_moneyness_nodes()[i % 3],
                basis.implied_volatilities()[i],
            );
            close(
                -projection.hedge_quantities()[i] * node_vega,
                expected,
                1e-9,
            );
        }
        assert_eq!(projection.position_risk().signed_residual(), 0.);
        assert!(projection.position_risk().reconciles());
        if case["id"] == "unit_call_at_node" {
            close(projection.hedge_quantities()[4], -1., 1e-14);
        }
        if case["id"] == "edge_calls" {
            assert_eq!(
                projection
                    .position_risk()
                    .reporting_stats()
                    .left_edge_count(),
                1
            );
            assert_eq!(
                projection
                    .position_risk()
                    .reporting_stats()
                    .right_edge_count(),
                1
            );
        }
    }
}

#[test]
fn off_node_hedges_require_the_nodal_vega_ratio() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let projection = project(&basis(&fixture), 0.5, 100., &[112.], &[-1.]).unwrap();
    let hat = (112.0_f64 / 100.).ln() / 0.3;
    // Assigning plain interpolation weights would get the hedge materially wrong.
    assert!((projection.hedge_quantities()[2] + hat).abs() > 0.1);
}

#[test]
fn projection_validates_time_weights_and_vanishing_nodal_vega() {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    let basis = basis(&fixture);
    assert!(matches!(
        project(&basis, 0.75, 100., &[100.], &[-1.]),
        Err(RiskConfigError::VegaKtProjectionMaturityNotNode { .. })
    ));
    assert!(project(&basis, 1., 100., &[100.], &[]).is_err());
    assert!(project(&basis, 1., 100., &[100.], &[f64::NAN]).is_err());
    assert!(project(&basis, 1., 100., &[0.], &[-1.]).is_err());
    let tail = ReportingIvBasis::new(vec![0.5, 1.], vec![0., 10.], vec![0.2; 4]).unwrap();
    // A material off-node risk must not be divided by a nodal Vega that underflows.
    assert!(project(&tail, 1., 100., &[112.], &[-1.]).is_err());
    // Unused zero-risk nodes do not require a nonzero Vega.
    assert!(project(&tail, 1., 100., &[100.], &[-1.]).is_ok());
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

fn gamma_row(time: f64, dt: f64, gamma: f64) -> Equation11GammaRow {
    gamma_row_on_nodes(time, dt, gamma, vec![-0.2, 0., 0.2])
}

fn gamma_row_on_nodes(time: f64, dt: f64, gamma: f64, nodes: Vec<f64>) -> Equation11GammaRow {
    let density = analytic_call_density_row_from_surface(&Flat, time, 100., nodes, 1e-200).unwrap();
    let areas = mass_lumped_hat_areas(density.log_moneyness_nodes()).unwrap();
    let adjoints: Vec<_> = density
        .log_moneyness_nodes()
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let strike = 100. * x.exp();
            gamma * strike.powi(3) * density.call_densities()[i] * 0.2 * dt * areas[i]
        })
        .collect();
    recover_equation11_gamma_row(&density, &adjoints, &vec![0.04; adjoints.len()], dt).unwrap()
}

#[test]
fn adjacent_gamma_rows_compose_with_hedge_sign_and_terminal_boundary() {
    let current = gamma_row(0.25, 0.25, 0.5);
    let next = gamma_row(0.5, 0.25, 1.);
    let hedge = current
        .vanilla_hedge_density_from_next(&next, &[[0.04, 0.0004, 0.000002]; 3])
        .unwrap();
    for (&actual, &spot) in hedge.iter().zip(current.strike_nodes()) {
        let expected = 0.04_f64.exp()
            * (1. + 2. * spot * 0.0004 + 0.5 * spot * spot * (0.0004 * 0.0004 + 0.000002))
            - 0.5;
        close(actual, expected, 1e-13);
    }
    for value in next.terminal_vanilla_hedge_density() {
        close(value, -1., 1e-14);
    }
    assert!(matches!(
        next.vanilla_hedge_density_from_next(&current, &[[0.04, 0., 0.]; 3]),
        Err(RiskConfigError::VegaKtGammaRowTimeMismatch { .. })
    ));
    assert!(current.vanilla_hedge_density_from_next(&next, &[]).is_err());
}

#[test]
fn gamma_to_hedge_to_iv_composition_localizes_a_quadratic_at_expiry() {
    // Synthetic exact Gamma rows isolate equations (4)/(9), not equation (11)'s
    // time error or the unknown initial row. E[S_T^2/2] = S0^2*exp(sigma^2*T)/2.
    let basis = ReportingIvBasis::new(vec![0.5, 1.], vec![-2., 0., 2.], vec![0.2; 6]).unwrap();
    let current = gamma_row(0.25, 0.25, 0.02_f64.exp());
    let expected_vega = 10000. * 0.2 * 0.04_f64.exp();
    let mut errors = Vec::new();
    for intervals in [128, 256, 512] {
        let nodes: Vec<_> = (0..=intervals)
            .map(|i| 0.2_f64.ln() + f64::from(i) / f64::from(intervals) * 25.0_f64.ln())
            .collect();
        let terminal = gamma_row_on_nodes(0.5, 0.5, 1., nodes);
        let intermediate_hedge = current
            .vanilla_hedge_density_from_next(&terminal, &[[0.02, 0., 0.]; 3])
            .unwrap();
        assert!(intermediate_hedge.iter().all(|value| value.abs() < 1e-13));
        let strikes = terminal.strike_nodes();
        let density = terminal.terminal_vanilla_hedge_density();
        let weights: Vec<_> = density
            .iter()
            .enumerate()
            .map(|(i, value)| {
                let left = if i == 0 { strikes[i] } else { strikes[i - 1] };
                let right = if i + 1 == strikes.len() {
                    strikes[i]
                } else {
                    strikes[i + 1]
                };
                value * (right - left) / 2.
            })
            .collect();
        let projection = project(&basis, 1., 100., strikes, &weights).unwrap();
        assert_eq!(&projection.position_risk().raw_buckets()[..3], &[0.; 3]);
        let error = (projection.position_risk().scalar_vega() - expected_vega).abs();
        println!("composed quadratic intervals={intervals}, absolute_vega_error={error:.9e}");
        errors.push(error);
    }
    for pair in errors.windows(2) {
        let order = (pair[0] / pair[1]).log2();
        assert!((1.9..2.1).contains(&order));
    }
    assert!(errors[2] / expected_vega < 1e-5);
}
