//! Arithmetic regression against the pre-cache recurrences from PR131.
//! Deliberately shares the compiled kernel/asset map; not a pricing-model oracle.
use super::*;

impl RoughVolatilityPathPlan {
    fn original_heston_path(
        &self,
        m: &RoughHeston,
        k: &PowerKernel,
        initial: f64,
        normals: &[f64],
    ) -> Result<RoughVolatilityPath, HullWhiteError> {
        let (dw, near) = self.innovations(k, m.correlation, normals, false);
        let mut variance = vec![m.initial_variance];
        let mut raw = vec![m.initial_variance];
        let mut negative = 0;
        for i in 1..self.times.len() {
            let mut sum = pricing_numerics::NeumaierSum::new();
            sum.add(m.initial_variance);
            for j in 0..i {
                sum.add(
                    k.drift_weight(&self.times, i, j)
                        * m.mean_reversion
                        * (m.long_run_variance - variance[j]),
                );
                let innovation = if j + 1 == i {
                    near[j]
                } else {
                    k.weights[i][j] * dw[j]
                };
                sum.add(k.fractional_scale * m.vol_of_vol * variance[j].sqrt() * innovation);
            }
            let value = finite_value(sum.total())?;
            negative += u64::from(value < 0.0);
            raw.push(value);
            variance.push(value.max(0.0));
        }
        self.asset_path(initial, variance, raw, normals, 1.0, negative)
    }
    fn original_quadratic_path(
        &self,
        m: &QuadraticRoughHeston,
        k: &PowerKernel,
        initial: f64,
        normals: &[f64],
    ) -> Result<RoughVolatilityPath, HullWhiteError> {
        let (dw, near) = self.innovations(k, 1.0, normals, true);
        let variance_at = |z: f64| {
            if m.quadratic == 0.0 {
                Ok(m.variance_floor)
            } else {
                finite_nonnegative(m.quadratic * (z - m.shift).powi(2) + m.variance_floor)
            }
        };
        let mut latent = vec![m.initial_state];
        let mut variance = vec![variance_at(m.initial_state)?];
        for i in 1..self.times.len() {
            let mut sum = pricing_numerics::NeumaierSum::new();
            sum.add(m.initial_state);
            for j in 0..i {
                sum.add(-m.mean_reversion * latent[j] * k.drift_weight(&self.times, i, j));
                let innovation = if j + 1 == i {
                    near[j]
                } else {
                    k.weights[i][j] * dw[j]
                };
                sum.add(
                    m.mean_reversion
                        * m.vol_of_vol
                        * variance[j].sqrt()
                        * k.fractional_scale
                        * innovation,
                );
            }
            let z = finite_value(sum.total())?;
            latent.push(z);
            variance.push(variance_at(z)?);
        }
        self.asset_path(initial, variance, latent, normals, 1.0, 0)
    }
}

fn original(
    plan: &RoughVolatilityPathPlan,
    z: &[f64],
) -> Result<RoughVolatilityPath, HullWhiteError> {
    match (&plan.model, &plan.driver) {
        (RoughVolatilityModel::RoughHeston(m), CompiledDriver::Power(k)) => {
            plan.original_heston_path(m, k, 100.0, z)
        }
        (RoughVolatilityModel::QuadraticRoughHeston(m), CompiledDriver::Power(k)) => {
            plan.original_quadratic_path(m, k, 100.0, z)
        }
        _ => unreachable!(),
    }
}
fn equal(actual: &RoughVolatilityPath, expected: &RoughVolatilityPath) {
    for (a, b) in [
        (&actual.forwards, &expected.forwards),
        (&actual.variances, &expected.variances),
        (&actual.latent_states, &expected.latent_states),
    ] {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
    assert_eq!(
        actual.negative_variance_nodes,
        expected.negative_variance_nodes
    );
    assert_eq!(
        actual.absorbed_forward_steps,
        expected.absorbed_forward_steps
    );
}

#[test]
fn diffusion_cache_preserves_every_path_node() {
    let mut negative = 0;
    let mut paths = 0;
    let mut errors = 0;
    for h in [0.01, 0.1, 0.3, 0.5] {
        for noise in [0.0, 0.18, 0.7] {
            let models: [RoughVolatilityModel; 2] = [
                RoughHeston::new(h, 0.04, 0.7, 0.055, noise, -0.65)
                    .unwrap()
                    .into(),
                QuadraticRoughHeston::new(h, 0.15, 1.1, noise, 0.8, 0.25, 0.02)
                    .unwrap()
                    .into(),
            ];
            for m in models {
                for times in [
                    vec![0.0, 0.007, 0.13, 0.37, 1.0],
                    (0..=64).map(|i| i as f64 / 64.0).collect(),
                    (0..=64).map(|i| (i as f64 / 64.0).powi(2)).collect(),
                ] {
                    let plan = RoughVolatilityPathPlan::compile(m.clone(), times).unwrap();
                    for seed in [91, 1973] {
                        for path in 0..8 {
                            let mut z = plan.pseudo_shocks(seed, path, RandomDomain::Valuation);
                            for _ in 0..2 {
                                match (plan.evolve_path(100.0, &z), original(&plan, &z)) {
                                    (Ok(actual), Ok(expected)) => {
                                        equal(&actual, &expected);
                                        negative += actual.negative_variance_nodes;
                                        paths += 1;
                                    }
                                    (Err(actual), Err(expected)) => {
                                        // The reference itself rejects some stress paths. Require
                                        // exactly the same error, not just any rejection.
                                        assert_eq!(actual, expected);
                                        errors += 1;
                                    }
                                    (a, b) => panic!("changed success/failure: {a:?} / {b:?}"),
                                }
                                for x in &mut z {
                                    *x = -*x;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(paths + errors, 2304);
    assert!(paths > 0 && errors > 0);
    println!("DIFFUSION_CACHE paths={paths} exact_errors={errors} negative_nodes={negative}");
    assert!(negative > 0, "full truncation must be exercised");
}

#[test]
fn diffusion_cache_preserves_zero_boundary_and_failure_behavior() {
    let models: [RoughVolatilityModel; 6] = [
        RoughHeston::new(0.1, 0.0, 0.7, 0.055, 0.18, -1.0)
            .unwrap()
            .into(),
        RoughHeston::new(0.5, 0.04, 0.0, 0.055, 0.7, 1.0)
            .unwrap()
            .into(),
        RoughHeston::new(0.1, f64::MAX, 0.0, 0.0, f64::MAX, 0.0)
            .unwrap()
            .into(),
        QuadraticRoughHeston::new(0.1, 0.0, 0.0, 0.7, 0.8, 0.0, 0.0)
            .unwrap()
            .into(),
        QuadraticRoughHeston::new(0.5, 0.15, 1.1, 0.7, 0.0, 0.25, 0.02)
            .unwrap()
            .into(),
        QuadraticRoughHeston::new(0.1, 0.15, 1.1, f64::MAX, 0.8, 0.25, 0.02)
            .unwrap()
            .into(),
    ];
    let mut errors = 0;
    for m in models {
        for times in [vec![0.0, 0.5], vec![0.0, 0.1, 0.5, 1.0]] {
            let plan = RoughVolatilityPathPlan::compile(m.clone(), times).unwrap();
            for path in 0..8 {
                let z = plan.pseudo_shocks(91, path, RandomDomain::Valuation);
                match (plan.evolve_path(100.0, &z), original(&plan, &z)) {
                    (Ok(a), Ok(b)) => equal(&a, &b),
                    (Err(a), Err(b)) => {
                        assert_eq!(a, b);
                        errors += 1;
                    }
                    (a, b) => panic!("changed success/failure: {a:?} / {b:?}"),
                }
            }
        }
    }
    assert!(errors > 0, "overflow rejection must be exercised");
}
