use pricing::MonteCarloError;
use pricing::market::{
    ImpliedVarianceSurface, LocalVarianceGrid, MarketError, ThetaRegion, TotalVarianceDerivatives,
};
use pricing::mc::{
    DeterministicExecutor, ExecutionPolicy, LocalVolLogEulerPlan, LocalVolTimeGrid, RandomDomain,
};
use pricing::risk::{
    RiskConfigError, VegaKtStepMoment, VegaKtStepRecoveryPlan, VegaKtStepSample,
    VegaKtStepSamplingUnit,
};
use serde_json::Value;

const FIXTURE: &str = include_str!("../../../fixtures/local-vol/step-recovery-v0.1.json");
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
fn plan() -> VegaKtStepRecoveryPlan {
    let f = fixture();
    let b = &f["base"];
    let simulation = LocalVolLogEulerPlan::new(
        LocalVolTimeGrid::compile(numbers(&b["simulation_times"]), 1.).unwrap(),
        numbers(&b["forwards"]),
    )
    .unwrap();
    let grid = LocalVarianceGrid::new(
        numbers(&b["grid_times"]),
        numbers(&b["nodes"]),
        numbers(&b["variances"]),
        1e-6,
        1.,
    )
    .unwrap();
    VegaKtStepRecoveryPlan::new(simulation, grid, &Flat, 0.05).unwrap()
}
fn executor(workers: u32, block: u64) -> DeterministicExecutor {
    DeterministicExecutor::new(ExecutionPolicy::new(workers, Some(block)).unwrap()).unwrap()
}
fn sample(
    plan: &VegaKtStepRecoveryPlan,
    shocks: &[f64],
    scale: f64,
) -> Result<VegaKtStepSample, MonteCarloError> {
    let path = plan
        .simulation_plan()
        .evolve_path(plan.variance_grid(), 100., shocks)?;
    let s = path.states();
    let mut seeds = vec![
        0.,
        0.3 * s[1],
        0.1 * s[5],
        -0.16 * s[3],
        0.,
        0.5 * s[5] + 0.1 * s[2],
    ];
    for seed in &mut seeds {
        *seed *= scale;
    }
    plan.sample_from_path(&path, &seeds)
}
fn unit_sample(
    plan: &VegaKtStepRecoveryPlan,
    shocks: &[f64],
    kind: VegaKtStepSamplingUnit,
) -> Result<VegaKtStepSample, MonteCarloError> {
    let primary = sample(plan, shocks, 1.)?;
    if kind == VegaKtStepSamplingUnit::IndependentPath {
        Ok(primary)
    } else {
        Ok(VegaKtStepSample::antithetic_pair(
            primary,
            sample(plan, &shocks.iter().map(|g| -g).collect::<Vec<_>>(), 1.)?,
        )?)
    }
}
fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= tolerance * expected.abs(),
        "actual={actual:.17e}, expected={expected:.17e}"
    );
}
fn check_moment(actual: VegaKtStepMoment, expected: &Value) {
    close(actual.mean(), expected["mean"].as_f64().unwrap(), 1e-9);
    close(
        actual.sample_variance().unwrap(),
        expected["sample_variance"].as_f64().unwrap(),
        1e-9,
    );
    close(
        actual.standard_error().unwrap(),
        expected["standard_error"].as_f64().unwrap(),
        1e-9,
    );
}

#[test]
fn path_and_pair_statistics_and_gamma_errors_match_independent_price_references() {
    let f = fixture();
    let plan = plan();
    let shocks: Vec<_> = f["primary_shocks"]
        .as_array()
        .unwrap()
        .iter()
        .map(numbers)
        .collect();
    for (i, kind) in [
        VegaKtStepSamplingUnit::IndependentPath,
        VegaKtStepSamplingUnit::AntitheticPair,
    ]
    .into_iter()
    .enumerate()
    {
        let estimate = plan
            .estimate(&executor(3, 2), 5, kind, |j| {
                unit_sample(&plan, &shocks[j as usize], kind)
            })
            .unwrap();
        let reference = &f["estimates"][i];
        assert_eq!(estimate.independent_sampling_units(), 5);
        assert_eq!(estimate.sampling_unit(), kind);
        assert_eq!(estimate.rows().len(), 5);
        check_moment(estimate.total(), &reference["total"]);
        for (step, row) in estimate.rows().iter().enumerate() {
            let expected = &reference["rows"][step];
            assert_eq!(
                row.source_maturity(),
                plan.simulation_plan().time_grid().nodes()[step]
            );
            assert_eq!(
                row.recovered_maturity(),
                plan.simulation_plan().time_grid().nodes()[step + 1]
            );
            assert_eq!(row.forward(), 100.);
            check_moment(row.total(), &expected["total"]);
            for (j, &node) in row.nodes().iter().enumerate() {
                check_moment(node, &expected["nodes"][j]);
            }
            if step == 0 {
                assert!(row.gamma().is_none() && row.gamma_standard_errors().is_none());
                assert!(row.retained().is_none() && row.excluded().is_none());
                assert_ne!(row.total().mean(), 0.);
            } else {
                check_moment(row.retained().unwrap(), &expected["retained"]);
                check_moment(row.excluded().unwrap(), &expected["excluded"]);
                let gamma = row.gamma().unwrap();
                assert_eq!(gamma.recovered_maturity().get(), row.recovered_maturity());
                let indices: Vec<_> = (gamma.active_domain().start_index()
                    ..=gamma.active_domain().end_index())
                    .collect();
                assert_eq!(
                    indices,
                    expected["active_indices"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect::<Vec<_>>()
                );
                for (j, &g) in gamma.gamma_values().iter().enumerate() {
                    close(g, expected["gamma"][j]["mean"].as_f64().unwrap(), 1e-9);
                    close(
                        row.gamma_standard_errors().unwrap()[j],
                        expected["gamma"][j]["standard_error"].as_f64().unwrap(),
                        1e-9,
                    );
                }
                close(gamma.input_local_vega(), row.total().mean(), 1e-12);
                close(
                    gamma.retained_local_vega(),
                    row.retained().unwrap().mean(),
                    1e-12,
                );
                close(
                    gamma.excluded_local_vega(),
                    row.excluded().unwrap().mean(),
                    1e-12,
                );
            }
        }
        println!(
            "{kind:?}: total_mean={}, total_SE={}",
            estimate.total().mean(),
            estimate.total().standard_error().unwrap()
        );
    }
}

#[test]
fn worker_counts_preserve_bitwise_results_including_an_odd_final_block() {
    let f = fixture();
    let plan = plan();
    let shocks: Vec<_> = f["primary_shocks"]
        .as_array()
        .unwrap()
        .iter()
        .map(numbers)
        .collect();
    let kind = VegaKtStepSamplingUnit::AntitheticPair;
    let serial = plan
        .estimate(&executor(1, 2), 5, kind, |j| {
            unit_sample(&plan, &shocks[j as usize], kind)
        })
        .unwrap();
    let cloned = plan.clone();
    let parallel = cloned
        .estimate(&executor(4, 2), 5, kind, |j| {
            unit_sample(&plan, &shocks[j as usize], kind)
        })
        .unwrap();
    assert_eq!(serial, parallel);
    let moment_bits = |m: VegaKtStepMoment| {
        (
            m.mean().to_bits(),
            m.sample_variance().map(f64::to_bits),
            m.standard_error().map(f64::to_bits),
        )
    };
    assert_eq!(moment_bits(serial.total()), moment_bits(parallel.total()));
    for (left, right) in serial.rows().iter().zip(parallel.rows()) {
        let row_bits = |row: &pricing::risk::VegaKtStepRowEstimate| {
            row.nodes()
                .iter()
                .copied()
                .chain([row.total()])
                .chain(row.retained())
                .chain(row.excluded())
                .map(moment_bits)
                .collect::<Vec<_>>()
        };
        assert_eq!(row_bits(left), row_bits(right));
        if let (Some(a), Some(b)) = (left.gamma(), right.gamma()) {
            for (a, b) in a.gamma_values().iter().zip(b.gamma_values()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
        if let (Some(a), Some(b)) = (left.gamma_standard_errors(), right.gamma_standard_errors()) {
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
}

#[test]
fn total_uncertainty_keeps_cross_node_and_cross_time_covariance() {
    let plan = plan();
    let f = fixture();
    let shocks: Vec<_> = f["primary_shocks"]
        .as_array()
        .unwrap()
        .iter()
        .map(numbers)
        .collect();
    let result = plan
        .estimate(
            &executor(2, 2),
            5,
            VegaKtStepSamplingUnit::IndependentPath,
            |j| sample(&plan, &shocks[j as usize], 1.),
        )
        .unwrap();
    let naive_time: f64 = result
        .rows()
        .iter()
        .map(|r| r.total().sample_variance().unwrap())
        .sum();
    let actual = result.total().sample_variance().unwrap();
    assert!((naive_time - actual).abs() > 0.1 * actual);
    let row = &result.rows()[2];
    let naive_nodes: f64 = row
        .nodes()
        .iter()
        .map(|n| n.sample_variance().unwrap())
        .sum();
    assert!(
        (naive_nodes - row.total().sample_variance().unwrap()).abs()
            > 0.1 * row.total().sample_variance().unwrap()
    );
}

#[test]
fn one_sampling_unit_has_no_invented_standard_error_and_zero_units_fail() {
    let plan = plan();
    let result = plan
        .estimate(
            &executor(1, 2),
            1,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| sample(&plan, &[0.2; 5], 1.),
        )
        .unwrap();
    assert!(result.total().standard_error().is_none());
    for row in result.rows() {
        assert!(row.total().sample_variance().is_none() && row.gamma_standard_errors().is_none());
        assert!(row.nodes().iter().all(|n| n.standard_error().is_none()));
    }
    assert!(matches!(
        plan.estimate(
            &executor(1, 2),
            0,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| unreachable!()
        ),
        Err(MonteCarloError::RiskConfig(
            RiskConfigError::VegaKtStepNoSamplingUnits
        ))
    ));
}

#[test]
fn sample_origins_and_sampling_unit_semantics_cannot_be_mixed() {
    let plan = plan();
    let other = VegaKtStepRecoveryPlan::new(
        plan.simulation_plan().clone(),
        plan.variance_grid().clone(),
        &Flat,
        0.1,
    )
    .unwrap();
    let primary = sample(&plan, &[0.2; 5], 1.).unwrap();
    let foreign = sample(&other, &[0.2; 5], 1.).unwrap();
    assert!(VegaKtStepSample::antithetic_pair(primary.clone(), foreign.clone()).is_err());
    assert!(
        plan.estimate(
            &executor(1, 2),
            1,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| Ok(foreign.clone())
        )
        .is_err()
    );
    assert!(
        plan.estimate(
            &executor(1, 2),
            1,
            VegaKtStepSamplingUnit::AntitheticPair,
            |_| Ok(primary.clone())
        )
        .is_err()
    );
    let pair =
        VegaKtStepSample::antithetic_pair(primary.clone(), sample(&plan, &[-0.2; 5], 1.).unwrap())
            .unwrap();
    assert!(VegaKtStepSample::antithetic_pair(primary, pair.clone()).is_err());
    assert!(
        plan.estimate(
            &executor(1, 2),
            1,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| Ok(pair.clone())
        )
        .is_err()
    );
}

#[test]
fn mismatched_paths_initial_state_and_unrepresentable_time_endpoints_fail() {
    let plan = plan();
    let grid = plan.variance_grid();
    let wrong_time = LocalVolLogEulerPlan::with_constant_forward(
        LocalVolTimeGrid::compile(vec![0., 0.11, 0.23, 0.4, 0.7, 1.], 1.).unwrap(),
        100.,
    )
    .unwrap();
    let path = wrong_time.evolve_path(grid, 100., &[0.2; 5]).unwrap();
    assert!(plan.sample_from_path(&path, &[0.; 6]).is_err());
    let wrong_spot = plan
        .simulation_plan()
        .evolve_path(grid, 101., &[0.2; 5])
        .unwrap();
    assert!(plan.sample_from_path(&wrong_spot, &[0.; 6]).is_err());
    let slope_grid = LocalVarianceGrid::new(
        vec![0., 0.4, 1.],
        vec![-0.4, 0., 0.35],
        vec![0.03, 0.04, 0.05, 0.03, 0.04, 0.05, 0.03, 0.04, 0.05],
        1e-6,
        1.,
    )
    .unwrap();
    let path = plan
        .simulation_plan()
        .evolve_path(&slope_grid, 100., &[0.2; 5])
        .unwrap();
    assert!(plan.sample_from_path(&path, &[0.; 6]).is_err());
    let path = plan
        .simulation_plan()
        .evolve_path(grid, 100., &[0.2; 5])
        .unwrap();
    assert!(plan.sample_from_path(&path, &[0.; 5]).is_err());
    assert!(plan.sample_from_path(&path, &[f64::NAN; 6]).is_err());
    let end = 1. + f64::EPSILON;
    let tiny = LocalVolLogEulerPlan::with_constant_forward(
        LocalVolTimeGrid::compile(vec![0., f64::EPSILON / 2., end], 2.).unwrap(),
        100.,
    )
    .unwrap();
    let coverage =
        LocalVarianceGrid::new(vec![0., end], vec![-0.4, 0., 0.35], vec![0.04; 6], 1e-6, 1.)
            .unwrap();
    assert!(VegaKtStepRecoveryPlan::new(tiny, coverage, &Flat, 0.05).is_err());
}

#[test]
fn one_interval_keeps_only_the_initial_row_and_finite_samples_cannot_hide_moment_overflow() {
    let simulation = LocalVolLogEulerPlan::with_constant_forward(
        LocalVolTimeGrid::compile(vec![1.], 1.).unwrap(),
        100.,
    )
    .unwrap();
    let grid = LocalVarianceGrid::new(vec![0., 1.], vec![-0.4, 0., 0.35], vec![0.04; 6], 1e-6, 1.)
        .unwrap();
    assert!(VegaKtStepRecoveryPlan::new(simulation.clone(), grid.clone(), &Flat, 0.).is_err());
    let single = VegaKtStepRecoveryPlan::new(simulation, grid, &Flat, 0.05).unwrap();
    assert!(single.positive_density_rows().is_empty());
    let result = single
        .estimate(
            &executor(2, 1),
            2,
            VegaKtStepSamplingUnit::IndependentPath,
            |i| {
                let path = single.simulation_plan().evolve_path(
                    single.variance_grid(),
                    100.,
                    &[if i == 0 { 0.2 } else { -0.2 }],
                )?;
                single.sample_from_path(&path, &[0., 1.])
            },
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert!(result.rows()[0].gamma().is_none());
    let plan = plan();
    assert!(
        plan.estimate(
            &executor(1, 2),
            2,
            VegaKtStepSamplingUnit::IndependentPath,
            |i| sample(&plan, &[0.2; 5], if i == 0 { 1e160 } else { -1e160 })
        )
        .is_err()
    );
}

fn quadratic_price(shocks: &[f64], times: &[f64], bumped_step: usize, bump: f64) -> f64 {
    let mut s = 100.;
    for (i, &g) in shocks.iter().enumerate() {
        let sigma = 0.2 + if i == bumped_step { bump } else { 0. };
        let dt = times[i + 1] - times[i];
        s *= (-0.5 * sigma * sigma * dt + sigma * dt.sqrt() * g).exp();
    }
    0.5 * s * s
}

#[test]
fn compiled_step_aggregation_and_gamma_recovery_match_4096_paired_crn_price_bumps() {
    let plan = plan();
    let executor = executor(4, 256);
    let units = 4096;
    let result = plan
        .estimate(
            &executor,
            units,
            VegaKtStepSamplingUnit::AntitheticPair,
            |unit| {
                let shocks =
                    plan.simulation_plan()
                        .path_shocks(41, unit, RandomDomain::Valuation, false)?;
                let path =
                    plan.simulation_plan()
                        .evolve_path(plan.variance_grid(), 100., &shocks)?;
                let mut seeds = vec![0.; 6];
                seeds[5] = path.states()[5];
                let primary = plan.sample_from_path(&path, &seeds)?;
                let opposite: Vec<_> = shocks.iter().map(|g| -g).collect();
                let mate_path =
                    plan.simulation_plan()
                        .evolve_path(plan.variance_grid(), 100., &opposite)?;
                seeds[5] = mate_path.states()[5];
                Ok(VegaKtStepSample::antithetic_pair(
                    primary,
                    plan.sample_from_path(&mate_path, &seeds)?,
                )?)
            },
        )
        .unwrap();
    let crn = executor
        .try_map_reduce_statistics_vector(units, 5, |unit, out| {
            let shocks =
                plan.simulation_plan()
                    .path_shocks(41, unit, RandomDomain::Valuation, false)?;
            let opposite: Vec<_> = shocks.iter().map(|g| -g).collect();
            let times = plan.simulation_plan().time_grid().nodes();
            for (i, value) in out.iter_mut().enumerate() {
                *value = ((quadratic_price(&shocks, times, i, 1e-5)
                    - quadratic_price(&shocks, times, i, -1e-5))
                    + (quadratic_price(&opposite, times, i, 1e-5)
                        - quadratic_price(&opposite, times, i, -1e-5)))
                    / 4e-5;
            }
            Ok::<_, MonteCarloError>(())
        })
        .unwrap();
    assert_eq!(
        plan.positive_density_rows()
            .iter()
            .map(|r| r.maturity().get())
            .collect::<Vec<_>>(),
        vec![0.1, 0.23, 0.4, 0.7]
    );
    for (i, row) in result.rows().iter().enumerate() {
        let reference = crn[i].sum().total() / units as f64;
        close(row.total().mean(), reference, 2e-7);
        let crn_se = (crn[i].moments().sample_variance().unwrap() / units as f64).sqrt();
        close(row.total().standard_error().unwrap(), crn_se, 2e-6);
        let analytic =
            10000. * 0.2 * (row.recovered_maturity() - row.source_maturity()) * 0.04_f64.exp();
        assert!((row.total().mean() - analytic).abs() < 6. * row.total().standard_error().unwrap());
        if i > 0 {
            assert!(row.gamma().is_some() && row.gamma_standard_errors().is_some());
        }
        println!(
            "step={i}, paired_AD={}, paired_CRN={reference}, SE={}",
            row.total().mean(),
            row.total().standard_error().unwrap()
        );
    }
}
