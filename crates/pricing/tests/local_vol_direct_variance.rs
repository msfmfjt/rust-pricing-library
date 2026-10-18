use pricing::MonteCarloError;
use pricing::market::{
    ImpliedVarianceSurface, LocalVarianceGrid, MarketError, ThetaRegion, TotalVarianceDerivatives,
};
use pricing::mc::{
    BarrierBridgeDirection, BarrierBridgeIntervalInput, BarrierBridgePath, DeterministicExecutor,
    ExecutionPolicy, LocalVolLogEulerPlan, LocalVolPath, LocalVolTimeGrid,
};
use pricing::risk::{
    VegaKtStepMoment, VegaKtStepRecoveryPlan, VegaKtStepSample, VegaKtStepSamplingUnit,
};
use serde_json::Value;

const FIXTURE: &str = include_str!("../../../fixtures/local-vol/direct-variance-v0.1.json");
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
fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn fixture() -> Value {
    serde_json::from_str(FIXTURE).unwrap()
}
fn setup(c: &Value) -> (LocalVolLogEulerPlan, LocalVarianceGrid) {
    (
        LocalVolLogEulerPlan::new(
            LocalVolTimeGrid::compile(numbers(&c["times"]), 1.).unwrap(),
            numbers(&c["forwards"]),
        )
        .unwrap(),
        LocalVarianceGrid::new(
            numbers(&c["grid_times"]),
            numbers(&c["nodes"]),
            numbers(&c["variances"]),
            1e-6,
            1.,
        )
        .unwrap(),
    )
}
fn close(a: f64, b: f64) {
    assert!(
        a.is_finite() && (a - b).abs() <= 1e-9 * b.abs(),
        "actual={a:.17e}, reference={b:.17e}"
    );
}
fn moment(a: VegaKtStepMoment, b: &Value) {
    close(a.mean(), b["mean"].as_f64().unwrap());
    close(
        a.sample_variance().unwrap(),
        b["sample_variance"].as_f64().unwrap(),
    );
    close(
        a.standard_error().unwrap(),
        b["standard_error"].as_f64().unwrap(),
    );
}
fn payoff_seeds(
    c: &Value,
    plan: &LocalVolLogEulerPlan,
    grid: &LocalVarianceGrid,
    path: &LocalVolPath,
) -> (f64, Vec<f64>, Vec<f64>) {
    let times = plan.time_grid().nodes();
    let s = path.states();
    let a: Vec<_> = times
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            grid.interpolate(t, (s[i] / plan.forward_normalizers()[i]).ln())
                .unwrap()
                .value
        })
        .collect();
    let end = c["monitor_end"].as_u64().unwrap() as usize;
    let barrier = c["barrier"].as_f64().unwrap();
    let inputs: Vec<_> = (0..end)
        .map(|i| BarrierBridgeIntervalInput {
            direction: if c["direction"] == "up" {
                BarrierBridgeDirection::Up
            } else {
                BarrierBridgeDirection::Down
            },
            left_state: s[i],
            right_state: s[i + 1],
            left_barrier: barrier,
            right_barrier: barrier,
            left_local_variance: a[i],
            right_local_variance: a[i + 1],
            dt: times[i + 1] - times[i],
        })
        .collect();
    let bridge = BarrierBridgePath::evaluate(&inputs).unwrap();
    let q = c["quadratic"].as_f64().unwrap();
    let rebate = c["rebate"].as_f64().unwrap();
    let discount = c["discount"].as_f64().unwrap();
    let vanilla = q * s[s.len() - 1].powi(2);
    let survival = bridge.survival();
    let knockout = c["style"] == "out";
    let value = if knockout {
        rebate + survival * (vanilla - rebate)
    } else {
        vanilla + survival * (rebate - vanilla)
    };
    let survival_seed = discount
        * if knockout {
            vanilla - rebate
        } else {
            rebate - vanilla
        };
    let mut states = vec![0.; s.len()];
    let mut variances = vec![0.; s.len()];
    states[s.len() - 1] =
        discount * 2. * q * s[s.len() - 1] * if knockout { survival } else { 1. - survival };
    for (i, r) in bridge.reverse(survival_seed * survival).iter().enumerate() {
        states[i] += r.left_state;
        states[i + 1] += r.right_state;
        variances[i] += r.left_local_variance;
        variances[i + 1] += r.right_local_variance;
    }
    (discount * value, states, variances)
}
fn sample(
    c: &Value,
    plan: &VegaKtStepRecoveryPlan,
    shocks: &[f64],
) -> Result<VegaKtStepSample, MonteCarloError> {
    let path = plan
        .simulation_plan()
        .evolve_path(plan.variance_grid(), 100., shocks)?;
    let (_, states, variances) =
        payoff_seeds(c, plan.simulation_plan(), plan.variance_grid(), &path);
    plan.sample_from_path_with_variance_seeds(&path, &states, &variances)
}
fn executor(workers: u32) -> DeterministicExecutor {
    DeterministicExecutor::new(ExecutionPolicy::new(workers, Some(2)).unwrap()).unwrap()
}

#[test]
fn bridge_reverse_matches_215_independent_decimal_price_derivatives() {
    let f = fixture();
    let mut count = 0;
    for c in f["cases"].as_array().unwrap() {
        let (plan, grid) = setup(c);
        let path = plan
            .evolve_path(&grid, 100., &numbers(&c["shocks"]))
            .unwrap();
        let (price, states, variances) = payoff_seeds(c, &plan, &grid, &path);
        close(price, c["expected"]["price"].as_f64().unwrap());
        let r = plan
            .reverse_path_with_variance_seeds(&grid, &path, &states, &variances)
            .unwrap();
        let mut total = 0.;
        for (i, step) in r.transition().step_adjoints().iter().enumerate() {
            let mut row = [0.; 3];
            step.accumulate_local_volatility_nodes(&mut row).unwrap();
            for (j, &v) in row.iter().enumerate() {
                close(v, c["expected"]["transition_hat"][i][j].as_f64().unwrap());
                count += 1;
                total += v;
            }
        }
        assert_eq!(r.direct_variance_adjoints().len(), 6);
        for (i, node) in r.direct_variance_adjoints().iter().enumerate() {
            assert_eq!(node.maturity(), plan.time_grid().nodes()[i]);
            let mut row = [0.; 3];
            node.accumulate_local_volatility_nodes(&mut row).unwrap();
            for (j, &v) in row.iter().enumerate() {
                close(v, c["expected"]["direct_hat"][i][j].as_f64().unwrap());
                count += 1;
                total += v;
            }
        }
        for (i, &v) in r.local_variance_value_adjoints().iter().enumerate() {
            close(v, c["expected"]["grid_variance"][i].as_f64().unwrap());
            count += 1;
        }
        close(total, c["expected"]["parallel_vol"].as_f64().unwrap());
        count += 1;
    }
    assert_eq!(count, 215);
}

#[test]
fn direct_state_feedback_is_applied_once_and_reproduces_previous_grid_arithmetic() {
    let f = fixture();
    let c = &f["cases"][0];
    let (plan, grid) = setup(c);
    let path = plan
        .evolve_path(&grid, 100., &numbers(&c["shocks"]))
        .unwrap();
    let (_, mut states, variances) = payoff_seeds(c, &plan, &grid, &path);
    let r = plan
        .reverse_path_with_variance_seeds(&grid, &path, &states, &variances)
        .unwrap();
    let missing = path.reverse_state_adjoints(&states, 9, 3).unwrap();
    assert!(
        (missing.step_adjoints()[0].local_volatility_adjoint()
            - r.transition().step_adjoints()[0].local_volatility_adjoint())
        .abs()
            > 1e-3
    );
    let mut direct_grid = vec![0.; 9];
    for (i, &seed) in variances.iter().enumerate() {
        let interpolation = grid
            .interpolate(plan.time_grid().nodes()[i], (path.states()[i] / 100.).ln())
            .unwrap();
        states[i] +=
            seed * grid.interpolation_log_moneyness_derivative(interpolation) / path.states()[i];
        interpolation.transpose_accumulate(seed, &mut direct_grid, 3);
    }
    let old = path.reverse_state_adjoints(&states, 9, 3).unwrap();
    for (i, &v) in old.local_variance_value_adjoints().iter().enumerate() {
        assert_eq!(
            (v + direct_grid[i]).to_bits(),
            r.local_variance_value_adjoints()[i].to_bits()
        );
    }
}

#[test]
fn direct_rows_keep_terminal_and_monitoring_boundary_separate() {
    let f = fixture();
    for k in [0, 2] {
        let c = &f["cases"][k];
        let (p, g) = setup(c);
        let plan = VegaKtStepRecoveryPlan::new(p, g, &Flat, 0.05).unwrap();
        let result = plan
            .estimate(
                &executor(2),
                1,
                VegaKtStepSamplingUnit::IndependentPath,
                |_| sample(c, &plan, &numbers(&c["shocks"])),
            )
            .unwrap();
        assert!(result.has_unrecovered_direct_variance());
        assert!(
            result
                .rows()
                .iter()
                .all(|r| r.gamma().is_none() && r.gamma_standard_errors().is_none())
        );
        let end = c["monitor_end"].as_u64().unwrap() as usize;
        assert_eq!(result.direct_variance_rows().len(), 6);
        assert_ne!(result.direct_variance_rows()[0].total().mean(), 0.);
        assert_ne!(result.direct_variance_rows()[end].total().mean(), 0.);
        for row in &result.direct_variance_rows()[end + 1..] {
            assert_eq!(row.total().mean(), 0.);
        }
        assert!(result.combined_total().standard_error().is_none());
    }
}

#[test]
fn direct_and_transition_statistics_keep_covariance_and_antithetic_units() {
    let f = fixture();
    let c = &f["cases"][3];
    let (p, g) = setup(c);
    let plan = VegaKtStepRecoveryPlan::new(p, g, &Flat, 0.05).unwrap();
    let shocks: Vec<_> = f["statistics"]["primary_shocks"]
        .as_array()
        .unwrap()
        .iter()
        .map(numbers)
        .collect();
    for (kind_index, kind) in [
        VegaKtStepSamplingUnit::IndependentPath,
        VegaKtStepSamplingUnit::AntitheticPair,
    ]
    .into_iter()
    .enumerate()
    {
        let evaluate = |i: u64| {
            let first = sample(c, &plan, &shocks[i as usize])?;
            if kind == VegaKtStepSamplingUnit::IndependentPath {
                Ok(first)
            } else {
                Ok(VegaKtStepSample::antithetic_pair(
                    first,
                    sample(
                        c,
                        &plan,
                        &shocks[i as usize].iter().map(|v| -v).collect::<Vec<_>>(),
                    )?,
                )?)
            }
        };
        let result = plan.estimate(&executor(1), 5, kind, evaluate).unwrap();
        let parallel = plan.estimate(&executor(4), 5, kind, evaluate).unwrap();
        assert_eq!(format!("{result:?}"), format!("{parallel:?}")); // All finite floats; debug round trips including signed zero.
        let reference = &f["statistics"]["moments"][kind_index];
        moment(result.combined_total(), &reference[0]);
        moment(result.total(), &reference[1]);
        for (i, row) in result.direct_variance_rows().iter().enumerate() {
            moment(row.total(), &reference[i + 2]);
        }
        let naive = result.total().sample_variance().unwrap()
            + result
                .direct_variance_rows()
                .iter()
                .map(|r| r.total().sample_variance().unwrap())
                .sum::<f64>();
        assert!((naive - result.combined_total().sample_variance().unwrap()).abs() > 0.01 * naive);
    }
}

#[test]
fn cancelling_direct_seeds_cannot_reenable_gamma_recovery() {
    let f = fixture();
    let (p, g) = setup(&f["cases"][3]);
    let plan = VegaKtStepRecoveryPlan::new(p, g, &Flat, 0.05).unwrap();
    let path = plan
        .simulation_plan()
        .evolve_path(plan.variance_grid(), 100., &[0.2; 5])
        .unwrap();
    let plus = plan
        .sample_from_path_with_variance_seeds(&path, &[0.; 6], &[1.; 6])
        .unwrap();
    let minus = plan
        .sample_from_path_with_variance_seeds(&path, &[0.; 6], &[-1.; 6])
        .unwrap();
    let pair = VegaKtStepSample::antithetic_pair(plus, minus).unwrap();
    let r = plan
        .estimate(
            &executor(1),
            2,
            VegaKtStepSamplingUnit::AntitheticPair,
            |_| Ok(pair.clone()),
        )
        .unwrap();
    assert_eq!(r.combined_total().mean(), 0.);
    assert!(r.has_unrecovered_direct_variance());
    assert!(r.rows().iter().all(|r| r.gamma().is_none()));
}

#[test]
fn zero_direct_seeds_preserve_transition_estimates_and_invalid_seeds_fail() {
    let f = fixture();
    let (p, g) = setup(&f["cases"][3]);
    let plan = VegaKtStepRecoveryPlan::new(p, g, &Flat, 0.05).unwrap();
    let path = plan
        .simulation_plan()
        .evolve_path(plan.variance_grid(), 100., &[0.2; 5])
        .unwrap();
    let mut seeds = [0.; 6];
    seeds[5] = 1.;
    let old = plan.sample_from_path(&path, &seeds).unwrap();
    let zero = plan
        .sample_from_path_with_variance_seeds(&path, &seeds, &[0.; 6])
        .unwrap();
    let a = plan
        .estimate(
            &executor(1),
            2,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| Ok(old.clone()),
        )
        .unwrap();
    let b = plan
        .estimate(
            &executor(2),
            2,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| Ok(zero.clone()),
        )
        .unwrap();
    assert_eq!(a, b);
    assert!(!b.has_unrecovered_direct_variance());
    assert!(b.rows()[1].gamma().is_some());
    for bad in [
        vec![0.; 5],
        vec![f64::NAN; 6],
        vec![f64::INFINITY; 6],
        vec![f64::MAX; 6],
    ] {
        assert!(
            plan.sample_from_path_with_variance_seeds(&path, &seeds, &bad)
                .is_err()
        );
    }
    // 2 * seed would overflow, although the final effective-Vol derivative is finite.
    let large = plan
        .simulation_plan()
        .reverse_path_with_variance_seeds(
            plan.variance_grid(),
            &path,
            &[0.; 6],
            &[1e308, 0., 0., 0., 0., 0.],
        )
        .unwrap();
    close(
        large.direct_variance_adjoints()[0].local_volatility_adjoint(),
        4e307,
    );
    let wrong = LocalVolLogEulerPlan::with_constant_forward(
        LocalVolTimeGrid::compile(vec![0., 0.11, 0.23, 0.4, 0.7, 1.], 1.).unwrap(),
        100.,
    )
    .unwrap();
    assert!(
        wrong
            .reverse_path_with_variance_seeds(plan.variance_grid(), &path, &seeds, &[0.; 6])
            .is_err()
    );
}
