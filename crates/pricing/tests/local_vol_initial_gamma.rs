use pricing::MonteCarloError;
use pricing::market::{
    ImpliedVarianceSurface, LocalVarianceGrid, MarketError, ThetaRegion, TotalVarianceDerivatives,
};
use pricing::mc::{
    DeterministicExecutor, ExecutionPolicy, LocalVolLogEulerPlan, LocalVolTimeGrid, RandomDomain,
};
use pricing::risk::{
    ReportingIvBasis, VegaKtGammaBumpEstimate, VegaKtInitialGammaPlan, VegaKtInitialGammaSample,
    VegaKtStepMoment, VegaKtStepSamplingUnit, analytic_call_density_row_from_surface,
    mass_lumped_hat_areas, project_vanilla_hedge_measure_to_reporting_iv,
    recover_equation11_gamma_row,
};
use serde_json::Value;
const FIXTURE: &str = include_str!("../../../fixtures/local-vol/initial-gamma-v0.1.json");
fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn fixture() -> Value {
    serde_json::from_str(FIXTURE).unwrap()
}
fn executor(n: u32) -> DeterministicExecutor {
    DeterministicExecutor::new(ExecutionPolicy::new(n, Some(2)).unwrap()).unwrap()
}
fn process(times: Vec<f64>) -> LocalVolLogEulerPlan {
    LocalVolLogEulerPlan::with_constant_forward(LocalVolTimeGrid::compile(times, 1.).unwrap(), 100.)
        .unwrap()
}
fn grid(times: Vec<f64>, variances: Vec<f64>) -> LocalVarianceGrid {
    LocalVarianceGrid::new(times, vec![-0.4, 0., 0.4], variances, 1e-6, 1.).unwrap()
}
fn flat_plan(strikes: Vec<f64>, bump: f64) -> VegaKtInitialGammaPlan {
    VegaKtInitialGammaPlan::new(
        process(vec![0., 0.1, 0.4]),
        grid(vec![0., 0.4], vec![0.04; 6]),
        strikes,
        bump,
    )
    .unwrap()
}
fn setup(c: &Value) -> VegaKtInitialGammaPlan {
    VegaKtInitialGammaPlan::new(
        process(numbers(&c["times"])),
        grid(
            numbers(&c["grid_times"]),
            numbers(&c["row_variances"])
                .iter()
                .flat_map(|v| [*v; 3])
                .collect(),
        ),
        numbers(&c["strikes"]),
        c["relative_bump"].as_f64().unwrap(),
    )
    .unwrap()
}
fn payoff(c: &Value, s: &[f64]) -> Result<f64, MonteCarloError> {
    Ok(c["discount"].as_f64().unwrap()
        * (c["quadratic"].as_f64().unwrap() * s[s.len() - 1].powi(2)
            + c["sextic"].as_f64().unwrap() * s[s.len() - 1].powi(6)
            + c["cross"].as_f64().unwrap() * s[1] * s[3]
            + 0.2 * s[1]
            + 7.))
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        a.is_finite() && (a - b).abs() <= tol,
        "actual={a:0.17e}, reference={b:0.17e}, tolerance={tol:0.3e}"
    );
}
fn check_moment(a: VegaKtStepMoment, b: &Value, difference: bool) {
    let mean = b["mean"].as_f64().unwrap();
    let var = b["sample_variance"].as_f64().unwrap();
    let se = b["standard_error"].as_f64().unwrap();
    if difference {
        close(a.mean(), mean, 1e-9);
        close(a.sample_variance().unwrap().sqrt(), var.sqrt(), 1e-9);
        close(a.standard_error().unwrap(), se, 1e-9);
    } else {
        close(a.mean(), mean, 1e-8 * mean.abs());
        close(a.sample_variance().unwrap(), var, 1e-8 * var.abs());
        close(a.standard_error().unwrap(), se, 1e-8 * se.abs());
    }
}
fn check_bump(a: VegaKtGammaBumpEstimate, b: &[Value]) {
    check_moment(a.coarse(), &b[0], false);
    check_moment(a.fine(), &b[1], false);
    check_moment(a.difference(), &b[2], true);
}

#[test]
fn conditional_replay_preserves_suffix_states_bitwise_and_keeps_first_state_exact() {
    let p = process(vec![0., 0.1, 0.23, 0.4]);
    let g = grid(vec![0., 0.4], vec![0.06, 0.04, 0.05, 0.08, 0.05, 0.09]);
    let shocks = [0.3, -0.5, 0.7];
    let full = p.evolve_path(&g, 100., &shocks).unwrap();
    let continued = p
        .continue_states_from_first_node(&g, 100., full.states()[1], &shocks[1..])
        .unwrap();
    assert_eq!(
        full.states()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        continued.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let fixed = p
        .continue_states_from_first_node(&g, 100., 95.123, &shocks[1..])
        .unwrap();
    assert_eq!(fixed[1].to_bits(), 95.123_f64.to_bits());
    assert!(
        p.continue_states_from_first_node(&g, 100., 95., &[0.2])
            .is_err()
    );
    assert!(
        p.continue_states_from_first_node(&g, 100., 95., &[f64::NAN, 0.2])
            .is_err()
    );
    assert!(
        p.continue_states_from_first_node(&g, 100., 0., &shocks[1..])
            .is_err()
    );
}

#[test]
fn conditional_gamma_and_initial_vega_moments_match_independent_decimal_prices() {
    let f = fixture();
    let c = &f["case"];
    let p = setup(c);
    let shocks: Vec<_> = f["shocks"]
        .as_array()
        .unwrap()
        .iter()
        .map(numbers)
        .collect();
    for (k, kind) in [
        VegaKtStepSamplingUnit::IndependentPath,
        VegaKtStepSamplingUnit::AntitheticPair,
    ]
    .into_iter()
    .enumerate()
    {
        let eval = |i: u64| {
            let primary = p.sample_from_shocks(&shocks[i as usize], |s| payoff(c, s))?;
            if kind == VegaKtStepSamplingUnit::IndependentPath {
                Ok(primary)
            } else {
                Ok(VegaKtInitialGammaSample::antithetic_pair(
                    primary,
                    p.sample_from_shocks(
                        &shocks[i as usize].iter().map(|v| -v).collect::<Vec<_>>(),
                        |s| payoff(c, s),
                    )?,
                )?)
            }
        };
        let a = p.estimate(&executor(1), 3, kind, eval).unwrap();
        let b = p.estimate(&executor(4), 3, kind, eval).unwrap();
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        let reference = f["estimates"][k].as_array().unwrap();
        for (i, &node) in a.nodes().iter().enumerate() {
            check_bump(node, &reference[3 * i..3 * i + 3]);
        }
        let n = 3 * a.nodes().len();
        check_bump(a.first_step_vega(), &reference[n..n + 3]);
        for i in 0..2 {
            check_moment(a.tail_vegas()[i], &reference[n + 3 + i], false);
            close(
                a.kernel_tail_masses()[i],
                f["kernel_tail_masses"][i].as_f64().unwrap(),
                1e-12,
            );
        }
        assert_eq!(a.independent_sampling_units(), 3);
        assert_eq!(a.maturity(), 0.1);
        assert_eq!(
            a.policy_label(),
            "initial_gamma_conditional_fd_diagnostic_v1"
        );
    }
}

#[test]
fn sixth_power_payoff_shows_fourth_order_bump_bias_without_hiding_it_as_sampling_error() {
    let mut errors = Vec::new();
    for bump in [0.04, 0.02, 0.01] {
        let p = flat_plan(vec![80., 100., 120.], bump);
        let r = p
            .estimate(
                &executor(1),
                1,
                VegaKtStepSamplingUnit::IndependentPath,
                |_| p.sample_from_shocks(&[0.3], |s| Ok(1e-10 * s[2].powi(6))),
            )
            .unwrap();
        let multiplier = (-0.5 * 0.04 * 0.3 + 0.04_f64.sqrt() * 0.3_f64.sqrt() * 0.3).exp();
        let gamma = 30. * 1e-10 * 100_f64.powi(4) * multiplier.powi(6);
        let node = r.nodes()[1];
        errors.push((node.coarse().mean() - gamma).abs());
        assert!(node.fine().standard_error().is_none());
        assert!(node.difference().mean() > 0.);
        close(
            node.fine().mean() - node.coarse().mean(),
            node.difference().mean(),
            0.,
        );
    }
    for pair in errors.windows(2) {
        assert!(
            (3.9..4.1).contains(&(pair[0] / pair[1]).log2()),
            "{errors:?}"
        );
    }
}

#[test]
fn first_vega_reconstruction_matches_conditional_analytic_price_derivative_and_native_ad() {
    let p = flat_plan(vec![60., 80., 100., 120., 160.], 0.01);
    let units = 4096;
    let r = p
        .estimate(
            &executor(4),
            units,
            VegaKtStepSamplingUnit::AntitheticPair,
            |i| {
                let g = p
                    .simulation_plan()
                    .path_shocks(41, i, RandomDomain::Valuation, false)?;
                Ok(VegaKtInitialGammaSample::antithetic_pair(
                    p.sample_from_shocks(&g[1..], |s| Ok(0.5 * s[2] * s[2]))?,
                    p.sample_from_shocks(&[-g[1]], |s| Ok(0.5 * s[2] * s[2]))?,
                )?)
            },
        )
        .unwrap();
    let expected_gamma = (0.04_f64 * 0.3).exp();
    let expected_vega = 10000. * 0.2 * 0.1 * (0.04_f64 * 0.4).exp();
    for node in r.nodes() {
        assert!(
            (node.fine().mean() - expected_gamma).abs()
                < 6. * node.fine().standard_error().unwrap()
        );
    }
    let reconstructed = r.first_step_vega().fine();
    assert!(
        (reconstructed.mean() - expected_vega).abs() < 6. * reconstructed.standard_error().unwrap()
    );
    // Every strike shares the same quadratic Gamma random variable. Its total
    // SE must retain perfect spatial covariance rather than add node variances.
    close(
        reconstructed.standard_error().unwrap(),
        r.nodes()[2].fine().standard_error().unwrap() * 200. * 0.004_f64.exp(),
        1e-7,
    );
    let ad = executor(4)
        .try_map_reduce_statistics_vector(units, 1, |i, out| {
            let g = p
                .simulation_plan()
                .path_shocks(41, i, RandomDomain::Valuation, false)?;
            let mut v = 0.;
            for sign in [-1., 1.] {
                let shocks: Vec<_> = g.iter().map(|z| sign * z).collect();
                let path = p
                    .simulation_plan()
                    .evolve_path(p.variance_grid(), 100., &shocks)?;
                v += 0.5
                    * path
                        .reverse_terminal(path.states()[2], 6, 3)?
                        .step_adjoints()[0]
                        .local_volatility_adjoint();
            }
            out[0] = v;
            Ok::<_, MonteCarloError>(())
        })
        .unwrap();
    let mean = ad[0].sum().total() / units as f64;
    let se = (ad[0].moments().sample_variance().unwrap() / units as f64).sqrt();
    assert!(
        (mean - reconstructed.mean()).abs() < 6. * (se + reconstructed.standard_error().unwrap())
    );
    println!(
        "initial Vega: reconstruction={}, SE={}, native AD={mean}, AD SE={se}, analytic={expected_vega}",
        reconstructed.mean(),
        reconstructed.standard_error().unwrap()
    );
}

struct Flat;
impl ImpliedVarianceSurface for Flat {
    fn total_variance_derivatives(
        &self,
        t: f64,
        _: f64,
    ) -> Result<TotalVarianceDerivatives, MarketError> {
        Ok(TotalVarianceDerivatives {
            total_variance: 0.04 * t,
            log_moneyness_derivative: 0.,
            log_moneyness_second_derivative: 0.,
            time_derivative: 0.04,
            theta: 0.04 * t,
            theta_derivative: 0.04,
            theta_region: ThetaRegion::Interpolated,
        })
    }
}
#[test]
fn initial_gamma_connects_to_equation_four_and_nine_without_density_at_zero() {
    let p = flat_plan(vec![60., 80., 100., 120., 160.], 0.01);
    let units = 4096;
    let r = p
        .estimate(
            &executor(2),
            units,
            VegaKtStepSamplingUnit::AntitheticPair,
            |i| {
                let g = p
                    .simulation_plan()
                    .path_shocks(7, i, RandomDomain::Valuation, false)?;
                Ok(VegaKtInitialGammaSample::antithetic_pair(
                    p.sample_from_shocks(&g[1..], |s| Ok(0.5 * s[2].powi(2)))?,
                    p.sample_from_shocks(&[-g[1]], |s| Ok(0.5 * s[2].powi(2)))?,
                )?)
            },
        )
        .unwrap();
    let xs = vec![-0.4, 0., 0.4];
    let density =
        analytic_call_density_row_from_surface(&Flat, 0.1, 100., xs.clone(), 1e-12).unwrap();
    let areas = mass_lumped_hat_areas(&xs).unwrap();
    let vega: Vec<_> = xs
        .iter()
        .enumerate()
        .map(|(j, x)| areas[j] * (100. * x.exp()).powi(3) * density.call_densities()[j] * 0.2 * 0.3)
        .collect();
    let next = recover_equation11_gamma_row(&density, &vega, &[0.04; 3], 0.3).unwrap();
    let hedge = r
        .vanilla_hedge_density_from_next(&next, &[[0.012, 0., 0.]; 5])
        .unwrap();
    for (h, n) in hedge.iter().zip(r.nodes()) {
        assert!(h.abs() < 6. * n.fine().standard_error().unwrap());
    }
    let weights: Vec<_> = hedge
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let lo = r.strike_nodes()[i.saturating_sub(1)];
            let hi = r.strike_nodes()[(i + 1).min(4)];
            h * (hi - lo) / 2.
        })
        .collect();
    let basis = ReportingIvBasis::new(vec![0.1, 0.4], vec![-0.8, 0., 0.8], vec![0.2; 6]).unwrap();
    let projection = project_vanilla_hedge_measure_to_reporting_iv(
        &basis,
        0.1,
        100.,
        r.strike_nodes(),
        &weights,
    )
    .unwrap();
    assert_eq!(&projection.position_risk().raw_buckets()[3..], &[0.; 3]);
    assert!(
        r.vanilla_hedge_density_from_next(&next, &[[0.012, 0., 0.]; 4])
            .is_err()
    );
    let wrong_density =
        analytic_call_density_row_from_surface(&Flat, 0.2, 100., xs, 1e-12).unwrap();
    let wrong = recover_equation11_gamma_row(&wrong_density, &vega, &[0.04; 3], 0.2).unwrap();
    assert!(
        r.vanilla_hedge_density_from_next(&wrong, &[[0.012, 0., 0.]; 5])
            .is_err()
    );
}

#[test]
fn tail_contributions_are_reported_and_a_narrow_domain_is_not_called_zero_error() {
    let p = flat_plan(vec![99., 100., 101.], 0.01);
    let r = p
        .estimate(
            &executor(1),
            1,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| p.sample_from_shocks(&[0.], |s| Ok(0.5 * s[2].powi(2))),
        )
        .unwrap();
    assert!(r.kernel_tail_masses().iter().sum::<f64>() > 0.8);
    assert!(
        r.tail_vegas().iter().map(|v| v.mean()).sum::<f64>()
            > 0.8 * r.first_step_vega().fine().mean()
    );
}

#[test]
fn invalid_scope_stencils_and_callback_values_fail() {
    let base = process(vec![0., 0.1, 0.4]);
    let g = grid(vec![0., 0.4], vec![0.04; 6]);
    assert!(
        VegaKtInitialGammaPlan::new(process(vec![0., 0.4]), g.clone(), vec![90., 110.], 0.01)
            .is_err()
    );
    let changing =
        LocalVolLogEulerPlan::new(base.time_grid().clone(), vec![100., 101., 102.]).unwrap();
    assert!(VegaKtInitialGammaPlan::new(changing, g.clone(), vec![90., 110.], 0.01).is_err());
    assert!(
        VegaKtInitialGammaPlan::new(
            base.clone(),
            grid(vec![0., 0.4], vec![0.05, 0.04, 0.05, 0.04, 0.04, 0.04]),
            vec![90., 110.],
            0.01
        )
        .is_err()
    );
    for bump in [0., -0.01, 0.25, f64::NAN, f64::EPSILON / 10.] {
        assert!(
            VegaKtInitialGammaPlan::new(base.clone(), g.clone(), vec![90., 110.], bump).is_err()
        );
    }
    for strikes in [vec![100.], vec![110., 90.], vec![90., 90.], vec![0., 100.]] {
        assert!(VegaKtInitialGammaPlan::new(base.clone(), g.clone(), strikes, 0.01).is_err());
    }
    let p = flat_plan(vec![90., 110.], 0.01);
    assert!(p.sample_from_shocks(&[0.], |_| Ok(f64::NAN)).is_err());
    assert!(p.sample_from_shocks(&[], |_| Ok(0.)).is_err());
}

#[test]
fn sample_origin_pair_semantics_and_empty_estimates_are_checked() {
    let p = flat_plan(vec![90., 110.], 0.01);
    let other = flat_plan(vec![90., 110.], 0.01);
    let sample = p.sample_from_shocks(&[0.2], |s| Ok(s[2].powi(2))).unwrap();
    let foreign = other
        .sample_from_shocks(&[0.2], |s| Ok(s[2].powi(2)))
        .unwrap();
    assert!(VegaKtInitialGammaSample::antithetic_pair(sample.clone(), foreign.clone()).is_err());
    assert!(
        p.estimate(
            &executor(1),
            1,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| Ok(foreign.clone())
        )
        .is_err()
    );
    assert!(
        p.estimate(
            &executor(1),
            1,
            VegaKtStepSamplingUnit::AntitheticPair,
            |_| Ok(sample.clone())
        )
        .is_err()
    );
    assert!(
        p.estimate(
            &executor(1),
            0,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| unreachable!()
        )
        .is_err()
    );
    let pair = VegaKtInitialGammaSample::antithetic_pair(sample.clone(), sample.clone()).unwrap();
    assert!(VegaKtInitialGammaSample::antithetic_pair(pair, sample).is_err());
}

#[test]
fn vanilla_kink_is_validated_in_expectation_against_black_scholes_gamma_and_initial_vega() {
    let strikes: Vec<_> = (0..=32)
        .map(|j| 100. * (-0.5 + f64::from(j) / 32.).exp())
        .collect();
    let p = flat_plan(strikes, 0.02);
    let units = 8192;
    let r = p
        .estimate(
            &executor(4),
            units,
            VegaKtStepSamplingUnit::AntitheticPair,
            |i| {
                let g = p
                    .simulation_plan()
                    .path_shocks(97, i, RandomDomain::Valuation, false)?;
                Ok(VegaKtInitialGammaSample::antithetic_pair(
                    p.sample_from_shocks(&g[1..], |s| Ok((s[2] - 100.).max(0.)))?,
                    p.sample_from_shocks(&[-g[1]], |s| Ok((s[2] - 100.).max(0.)))?,
                )?)
            },
        )
        .unwrap();
    let sd = (0.04_f64 * 0.3).sqrt();
    for j in [12, 16, 20] {
        let s = r.strike_nodes()[j];
        let d1 = (s / 100.).ln() / sd + 0.5 * sd;
        let analytic = pricing_numerics::standard_normal_pdf(d1) / (s * sd);
        let gamma = r.nodes()[j].fine();
        assert!((gamma.mean() - analytic).abs() < 6. * gamma.standard_error().unwrap());
    }
    let total_sd = (0.04_f64 * 0.4).sqrt();
    let analytic =
        100. * pricing_numerics::standard_normal_pdf(0.5 * total_sd) * 0.2 * 0.1 / total_sd;
    let vega = r.first_step_vega().fine();
    // A 32-cell linear reconstruction adds deterministic spatial bias; require
    // both a 6-SE comparison and an absolute 2% accuracy check for this case.
    assert!((vega.mean() - analytic).abs() < 6. * vega.standard_error().unwrap());
    assert!((vega.mean() / analytic - 1.).abs() < 0.02);
    println!(
        "vanilla initial Vega: reconstruction={}, SE={}, analytic={analytic}, bump change={}",
        vega.mean(),
        vega.standard_error().unwrap(),
        r.first_step_vega().difference().mean()
    );
}
