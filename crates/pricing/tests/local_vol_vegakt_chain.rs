use pricing::MonteCarloError;
use pricing::market::{
    ImpliedVarianceSurface, LocalVarianceGrid, MarketError, ThetaRegion, TotalVarianceDerivatives,
};
use pricing::mc::{
    DeterministicExecutor, ExecutionPolicy, LocalVolLogEulerPlan, LocalVolTimeGrid, RandomDomain,
};
use pricing::risk::{
    ReportingIvBasis, VegaKtChainEstimate, VegaKtChainPlan, VegaKtChainSample,
    VegaKtInitialGammaPlan, VegaKtStepMoment, VegaKtStepRecoveryPlan, VegaKtStepSamplingUnit,
};
use serde_json::Value;
const FIXTURE: &str = include_str!("../../../fixtures/local-vol/vegakt-chain-v0.1.json");
struct Flat(f64);
impl ImpliedVarianceSurface for Flat {
    fn total_variance_derivatives(
        &self,
        t: f64,
        _: f64,
    ) -> Result<TotalVarianceDerivatives, MarketError> {
        Ok(TotalVarianceDerivatives {
            total_variance: self.0 * t,
            log_moneyness_derivative: 0.,
            log_moneyness_second_derivative: 0.,
            time_derivative: self.0,
            theta: self.0 * t,
            theta_derivative: self.0,
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
fn executor(w: u32) -> DeterministicExecutor {
    DeterministicExecutor::new(ExecutionPolicy::new(w, Some(2)).unwrap()).unwrap()
}
fn parts() -> (
    VegaKtInitialGammaPlan,
    VegaKtStepRecoveryPlan,
    ReportingIvBasis,
) {
    let f: Value = serde_json::from_str(FIXTURE).unwrap();
    let c = &f["case"];
    let ts = numbers(&c["times"]);
    let xs = numbers(&c["nodes"]);
    let r = numbers(&c["report_nodes"]);
    let p = LocalVolLogEulerPlan::with_constant_forward(
        LocalVolTimeGrid::compile(ts.clone(), 1.).unwrap(),
        100.,
    )
    .unwrap();
    let g = LocalVarianceGrid::new(vec![0., 1.], xs.clone(), vec![0.04; 2 * xs.len()], 1e-6, 1.)
        .unwrap();
    let initial =
        VegaKtInitialGammaPlan::new(p.clone(), g.clone(), numbers(&c["initial_strikes"]), 0.04)
            .unwrap();
    let steps = VegaKtStepRecoveryPlan::new(p, g, &Flat(0.04), 0.001).unwrap();
    let basis = ReportingIvBasis::new(ts[1..].to_vec(), r.clone(), vec![0.2; 3 * r.len()]).unwrap();
    (initial, steps, basis)
}
fn plan(q: usize) -> VegaKtChainPlan {
    let (i, s, b) = parts();
    VegaKtChainPlan::new(i, s, b, q).unwrap()
}
fn payoff(s: &[f64]) -> Result<(f64, Vec<f64>), MonteCarloError> {
    let st = s[s.len() - 1];
    let mut seeds = vec![0.; s.len()];
    seeds[s.len() - 1] = st + 6e-10 * st.powi(5);
    Ok((0.5 * st * st + 1e-10 * st.powi(6), seeds))
}
fn channels(r: &VegaKtChainEstimate) -> Vec<VegaKtStepMoment> {
    let mut v = r.position_iv_buckets.clone();
    v.extend(&r.offsetting_hedge_quantities);
    v.extend(&r.initial_bump_bucket_differences);
    v.extend(&r.quadrature_bucket_differences);
    v.extend([
        r.total_position_iv_vega,
        r.total_initial_bump_difference,
        r.total_quadrature_difference,
    ]);
    v.extend(&r.excluded_local_vegas);
    v.extend([
        r.initial_reconstruction_difference,
        r.initial_bump_vega_difference,
    ]);
    v.extend(r.initial_tail_vegas);
    v.extend(&r.propagated_tail_iv_vegas);
    v.extend(r.reporting_edge_iv_vegas.iter().flatten());
    v
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        a.is_finite() && (a - b).abs() <= tol,
        "actual={a:.17e}, expected={b:.17e}, tol={tol:.3e}"
    );
}
#[test]
fn joint_buckets_hedges_differences_and_errors_match_independent_decimal_prices() {
    let f: Value = serde_json::from_str(FIXTURE).unwrap();
    check_reference(plan(2), f, "flat_variance_paper_chain_diagnostic_v1");
}
#[test]
fn weak_gamma_chain_matches_independent_price_and_density_integrals() {
    let f: Value = serde_json::from_str(include_str!(
        "../../../fixtures/local-vol/weak-chain-v0.1.json"
    ))
    .unwrap();
    check_reference(
        plan(2).with_integrated_density_recovery().unwrap(),
        f,
        "flat_variance_weak_gamma_chain_diagnostic_v1",
    );
}
fn check_reference(p: VegaKtChainPlan, f: Value, label: &str) {
    let zs: Vec<_> = f["shocks"]
        .as_array()
        .unwrap()
        .iter()
        .map(numbers)
        .collect();
    for (index, kind) in [
        VegaKtStepSamplingUnit::IndependentPath,
        VegaKtStepSamplingUnit::AntitheticPair,
    ]
    .into_iter()
    .enumerate()
    {
        let eval = |i: u64| {
            let z = &zs[i as usize];
            let a = p.sample_from_shocks(z, payoff)?;
            if kind == VegaKtStepSamplingUnit::IndependentPath {
                Ok(a)
            } else {
                Ok(VegaKtChainSample::antithetic_pair(
                    a,
                    p.sample_from_shocks(&z.iter().map(|v| -v).collect::<Vec<_>>(), payoff)?,
                )?)
            }
        };
        let r = p.estimate(&executor(1), 3, kind, eval).unwrap();
        let parallel = p.estimate(&executor(4), 3, kind, eval).unwrap();
        assert_eq!(format!("{r:?}"), format!("{parallel:?}"));
        let channels = channels(&r);
        let expected = f["estimates"][index].as_array().unwrap();
        assert_eq!(channels.len(), expected.len());
        for (j, (m, e)) in channels.iter().zip(expected).enumerate() {
            let diff = (18..36).contains(&j) || j == 37 || j == 38 || j == 42;
            for (a, key) in [
                (m.mean(), "mean"),
                (m.sample_variance().unwrap().sqrt(), "sample_variance"),
                (m.standard_error().unwrap(), "standard_error"),
            ] {
                let b = e[key].as_f64().unwrap();
                let b = if key == "sample_variance" {
                    b.sqrt()
                } else {
                    b
                };
                let tol = if diff { 1e-8 } else { 1e-8 * b.abs() };
                close(a, b, tol);
            }
        }
        assert_eq!(r.rows.len(), 3);
        assert_eq!(r.rows[0].maturity, 0.3);
        assert_eq!(r.rows[1].recovery_source_maturity, Some(0.3));
        assert_eq!(r.rows[2].next_gamma_bounds, None);
        assert_eq!(r.policy_label, label);
        assert_eq!(
            r.rows[1].recovery_density_maturity,
            Some(if label.contains("weak") { 0.6 } else { 0.3 })
        );
        // Total SE must retain correlations, not sum marginal bucket variances.
        let diagonal: f64 = r
            .position_iv_buckets
            .iter()
            .map(|v| v.sample_variance().unwrap())
            .sum();
        assert!((diagonal - r.total_position_iv_vega.sample_variance().unwrap()).abs() > 100.);
        assert!(
            r.reporting_edge_iv_vegas
                .iter()
                .flatten()
                .any(|m| m.mean() != 0.)
        );
    }
}
#[test]
fn rejects_misaligned_models_expiries_densities_and_unrepresentable_quadrature() {
    let (i, s, b) = parts();
    for q in [0, usize::MAX, 1_000_001] {
        assert!(VegaKtChainPlan::new(i.clone(), s.clone(), b.clone(), q).is_err());
    }
    let wrong_b =
        ReportingIvBasis::new(vec![0.3, 0.7, 1.], vec![-0.2, 0., 0.22], vec![0.2; 9]).unwrap();
    assert!(VegaKtChainPlan::new(i.clone(), s.clone(), wrong_b, 2).is_err());
    let wrong_iv =
        ReportingIvBasis::new(vec![0.3, 0.6, 1.], vec![-0.2, 0., 0.22], vec![0.21; 9]).unwrap();
    assert!(VegaKtChainPlan::new(i.clone(), s.clone(), wrong_iv, 2).is_err());
    let wrong_d = VegaKtStepRecoveryPlan::new(
        s.simulation_plan().clone(),
        s.variance_grid().clone(),
        &Flat(0.05),
        0.001,
    )
    .unwrap();
    assert!(VegaKtChainPlan::new(i.clone(), wrong_d, b.clone(), 2).is_err());
    let single = VegaKtStepRecoveryPlan::new(
        s.simulation_plan().clone(),
        s.variance_grid().clone(),
        &Flat(0.04),
        0.99,
    )
    .unwrap();
    assert!(VegaKtChainPlan::new(i.clone(), single, b.clone(), 2).is_err());
    let changed_grid = LocalVarianceGrid::new(
        vec![0., 1.],
        vec![-0.25, -0.12, 0., 0.14, 0.28],
        vec![0.05; 10],
        1e-6,
        1.,
    )
    .unwrap();
    let changed = VegaKtStepRecoveryPlan::new(
        s.simulation_plan().clone(),
        changed_grid,
        &Flat(0.05),
        0.001,
    )
    .unwrap();
    assert!(VegaKtChainPlan::new(i, changed, b, 2).is_err());
}
#[test]
fn sample_origin_pair_units_and_callback_failures_are_not_silently_accepted() {
    let p = plan(2);
    let other = plan(2);
    let z = [0.2, -0.4, 0.7];
    let sample = || p.sample_from_shocks(&z, payoff).unwrap();
    assert!(
        VegaKtChainSample::antithetic_pair(sample(), other.sample_from_shocks(&z, payoff).unwrap())
            .is_err()
    );
    let pair = VegaKtChainSample::antithetic_pair(sample(), sample()).unwrap();
    assert!(VegaKtChainSample::antithetic_pair(pair.clone(), sample()).is_err());
    assert!(
        p.estimate(
            &executor(1),
            0,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| Ok(sample())
        )
        .is_err()
    );
    assert!(
        p.estimate(
            &executor(1),
            1,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| Ok(pair.clone())
        )
        .is_err()
    );
    assert!(
        p.estimate(
            &executor(1),
            1,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| other.sample_from_shocks(&z, payoff)
        )
        .is_err()
    );
    for shocks in [&z[..2], &[0.2, f64::NAN, 0.7]] {
        assert!(p.sample_from_shocks(shocks, payoff).is_err());
    }
    assert!(
        p.sample_from_shocks(&z, |s| Ok((f64::NAN, vec![0.; s.len()])))
            .is_err()
    );
    assert!(p.sample_from_shocks(&z, |_| Ok((0., vec![0.]))).is_err());
    assert!(
        p.sample_from_shocks(&z, |s| Ok((0., vec![f64::INFINITY; s.len()])))
            .is_err()
    );
    let one = p
        .estimate(
            &executor(1),
            1,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| Ok(sample()),
        )
        .unwrap();
    assert!(channels(&one).iter().all(|v| v.standard_error().is_none()));
}
fn first_fixing(s: &[f64]) -> Result<(f64, Vec<f64>), MonteCarloError> {
    let mut seeds = vec![0.; s.len()];
    seeds[1] = s[1];
    Ok((0.5 * s[1] * s[1], seeds))
}
#[test]
fn first_fixing_payoff_localizes_expiry_and_strike_integral_refines_at_second_order() {
    let (i, s, b) = parts();
    let initial = VegaKtInitialGammaPlan::new(
        i.simulation_plan().clone(),
        i.variance_grid().clone(),
        vec![10., 50., 100., 200., 400.],
        0.04,
    )
    .unwrap();
    let exact = 100_f64.powi(2) * 0.2 * 0.3 * (0.04_f64 * 0.3).exp();
    let mut errors = Vec::new();
    for q in [8, 16, 32] {
        let p = VegaKtChainPlan::new(initial.clone(), s.clone(), b.clone(), q).unwrap();
        let r = p
            .estimate(
                &executor(1),
                1,
                VegaKtStepSamplingUnit::IndependentPath,
                |_| p.sample_from_shocks(&[0.2, -0.4, 0.7], first_fixing),
            )
            .unwrap();
        assert!(r.position_iv_buckets[3..].iter().all(|m| m.mean() == 0.));
        assert!(
            r.initial_bump_bucket_differences[3..]
                .iter()
                .all(|m| m.mean() == 0.)
        );
        errors.push((r.total_position_iv_vega.mean() - exact).abs());
        let negative = p
            .estimate(
                &executor(1),
                1,
                VegaKtStepSamplingUnit::IndependentPath,
                |_| {
                    p.sample_from_shocks(&[0.2, -0.4, 0.7], |s| {
                        let (v, d) = first_fixing(s)?;
                        Ok((-v, d.iter().map(|v| -v).collect()))
                    })
                },
            )
            .unwrap();
        for (a, b) in r
            .position_iv_buckets
            .iter()
            .zip(&negative.position_iv_buckets)
        {
            close(a.mean(), -b.mean(), 0.);
        }
    }
    for v in errors.windows(2) {
        assert!((3.8..4.2).contains(&(v[0] / v[1])), "{errors:?}");
    }
    println!("first-fixing finite-strike quadrature errors: {errors:?}");
}
#[test]
fn native_quadratic_chain_reports_joint_uncertainty_and_initial_ad_difference() {
    let times = vec![0., 0.1, 0.2, 0.3, 0.4];
    let x: Vec<_> = (0..=24).map(|i| -0.5 + i as f64 / 24.).collect();
    let p = LocalVolLogEulerPlan::with_constant_forward(
        LocalVolTimeGrid::compile(times.clone(), 1.).unwrap(),
        100.,
    )
    .unwrap();
    let grid = LocalVarianceGrid::new(vec![0., 0.4], x.clone(), vec![0.04; 2 * x.len()], 1e-6, 1.)
        .unwrap();
    let initial = VegaKtInitialGammaPlan::new(
        p.clone(),
        grid.clone(),
        x.iter().map(|x| 100. * x.exp()).collect(),
        0.04,
    )
    .unwrap();
    let steps = VegaKtStepRecoveryPlan::new(p.clone(), grid, &Flat(0.04), 0.0001).unwrap();
    let basis =
        ReportingIvBasis::new(times[1..].to_vec(), vec![-0.5, 0., 0.5], vec![0.2; 12]).unwrap();
    let chain = VegaKtChainPlan::new(initial, steps, basis, 2).unwrap();
    let evaluate = |i| {
        let z = p.path_shocks(41, i, RandomDomain::Valuation, false)?;
        let make = |z: &[f64]| {
            chain.sample_from_shocks(z, |s| {
                let last = s[s.len() - 1];
                let mut d = vec![0.; s.len()];
                d[s.len() - 1] = last;
                Ok((0.5 * last * last, d))
            })
        };
        Ok(VegaKtChainSample::antithetic_pair(
            make(&z)?,
            make(&z.iter().map(|v| -v).collect::<Vec<_>>())?,
        )?)
    };
    let r = chain
        .estimate(
            &executor(4),
            4096,
            VegaKtStepSamplingUnit::AntitheticPair,
            evaluate,
        )
        .unwrap();
    let target = 100_f64.powi(2) * 0.2 * 0.4 * (0.04_f64 * 0.4).exp();
    let total = r.total_position_iv_vega;
    assert!((total.mean() - target).abs() < 6. * total.standard_error().unwrap());
    let diff = r.initial_reconstruction_difference;
    assert!(diff.mean().abs() < 6. * diff.standard_error().unwrap());
    assert!(r.excluded_local_vegas.iter().any(|v| v.mean() != 0.));
    println!(
        "quadratic chain: mean={}, SE={}, analytic={}, initial difference={} (SE={}), quadrature difference={}",
        total.mean(),
        total.standard_error().unwrap(),
        target,
        diff.mean(),
        diff.standard_error().unwrap(),
        r.total_quadrature_difference.mean()
    );
}

#[test]
fn time_varying_flat_variance_uses_discrete_integrated_variance_for_density_and_iv() {
    struct Discrete;
    impl ImpliedVarianceSurface for Discrete {
        fn total_variance_derivatives(
            &self,
            t: f64,
            _: f64,
        ) -> Result<TotalVarianceDerivatives, MarketError> {
            let v = if t <= 0.3 {
                0.04 * t
            } else if t <= 0.6 {
                0.04 * 0.3 + 0.052 * (t - 0.3)
            } else {
                0.04 * 0.3 + 0.052 * 0.3 + 0.064 * (t - 0.6)
            };
            Ok(TotalVarianceDerivatives {
                total_variance: v,
                log_moneyness_derivative: 0.,
                log_moneyness_second_derivative: 0.,
                time_derivative: 0.04 + 0.04 * t,
                theta: v,
                theta_derivative: 0.04 + 0.04 * t,
                theta_region: ThetaRegion::Interpolated,
            })
        }
    }
    let (original, _, _) = parts();
    let simulation = original.simulation_plan().clone();
    let grid = LocalVarianceGrid::new(
        vec![0., 1.],
        vec![-0.25, -0.12, 0., 0.14, 0.28],
        [vec![0.04; 5], vec![0.08; 5]].concat(),
        1e-6,
        1.,
    )
    .unwrap();
    let i = VegaKtInitialGammaPlan::new(
        simulation.clone(),
        grid.clone(),
        original.strike_nodes().to_vec(),
        0.04,
    )
    .unwrap();
    let s = VegaKtStepRecoveryPlan::new(simulation, grid, &Discrete, 0.001).unwrap();
    let times = vec![0.3, 0.6, 1.];
    let ivs: Vec<_> = times
        .iter()
        .flat_map(|&t| {
            let v = Discrete
                .total_variance_derivatives(t, 0.)
                .unwrap()
                .total_variance;
            vec![(v / t).sqrt(); 3]
        })
        .collect();
    let b = ReportingIvBasis::new(times, vec![-0.2, 0., 0.22], ivs).unwrap();
    let p = VegaKtChainPlan::new(i, s, b, 2).unwrap();
    let r = p
        .estimate(
            &executor(1),
            1,
            VegaKtStepSamplingUnit::IndependentPath,
            |_| p.sample_from_shocks(&[0.3, -0.5, 0.1], payoff),
        )
        .unwrap();
    assert!(r.total_position_iv_vega.mean().is_finite());
}
