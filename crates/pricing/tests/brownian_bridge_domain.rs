use pricing::mc::{BrownianBridgeError, BrownianBridgePlan};

fn invalid_variance(scale: f64, expected_variance: f64) {
    for factors in [1, 3] {
        let error = BrownianBridgePlan::compile(vec![0.0, scale, 2.0 * scale], factors)
            .expect_err("a degenerate bridge must fail before execution");
        assert_eq!(
            error,
            BrownianBridgeError::InvalidConditionalVariance {
                left_index: 0,
                node_index: 1,
                right_index: 2,
                variance_bits: expected_variance.to_bits(),
            }
        );
        assert!(error.to_string().contains("must be finite and positive"));
    }
}

#[test]
fn underflowed_conditional_variance_is_rejected() {
    for scale in [2.0_f64.powi(-600), f64::from_bits(1)] {
        invalid_variance(scale, 0.0);
    }
}

#[test]
fn overflowed_conditional_variance_is_rejected() {
    invalid_variance(2.0_f64.powi(600), f64::INFINITY);
}

#[test]
fn child_intervals_are_checked_after_valid_root_candidates() {
    let small = 2.0_f64.powi(-600);
    let end = 2.0_f64.powi(-500);
    for (times, indices) in [
        (vec![0.0, small, 2.0 * small, 1.0], (0, 1, 2)),
        (
            vec![
                0.0,
                f64::from_bits(end.to_bits() - 2),
                f64::from_bits(end.to_bits() - 1),
                end,
            ],
            (1, 2, 3),
        ),
    ] {
        assert_eq!(
            BrownianBridgePlan::compile(times, 1).unwrap_err(),
            BrownianBridgeError::InvalidConditionalVariance {
                left_index: indices.0,
                node_index: indices.1,
                right_index: indices.2,
                variance_bits: 0.0_f64.to_bits(),
            }
        );
    }
}

#[test]
fn finite_extreme_terminal_only_bridges_remain_supported() {
    // The guard concerns the actual conditional calculation, not an arbitrary
    // lower or upper bound on times. A terminal-only bridge has no such product.
    for terminal in [
        f64::from_bits(1),
        2.0_f64.powi(-600),
        2.0_f64.powi(600),
        f64::MAX,
    ] {
        let plan = BrownianBridgePlan::compile(vec![0.0, terminal], 2).unwrap();
        let increments = plan.apply(&[1.25, -0.75]).unwrap();
        for (&actual, expected) in increments.iter().zip([1.25, -0.75]) {
            assert!((actual - expected).abs() < 2.0e-15);
        }
        let reverse = plan.reverse_one_factor(&[-0.75]).unwrap();
        assert!((reverse[0] + 0.75).abs() < 2.0e-15);
    }
}

#[test]
fn valid_scaled_bridges_preserve_shocks_and_reverse_bits() {
    let times = [0.0, 0.25, 0.5, 0.75, 1.0];
    let normals = [0.3, -1.1, 0.7, 2.0];
    let adjoints = [-0.2, 0.9, 1.3, -0.4];
    let base = BrownianBridgePlan::compile(times.to_vec(), 1).unwrap();
    let bits = |values: Vec<f64>| values.into_iter().map(f64::to_bits).collect::<Vec<_>>();
    for scale in [2.0_f64.powi(-500), 1.0, 2.0_f64.powi(500)] {
        let plan =
            BrownianBridgePlan::compile(times.iter().map(|t| scale * t).collect(), 1).unwrap();
        assert_eq!(
            plan.instructions()
                .iter()
                .map(|i| i.node())
                .collect::<Vec<_>>(),
            [4, 2, 1, 3]
        );
        assert_eq!(
            bits(plan.apply_one_factor(&normals).unwrap()),
            bits(base.apply_one_factor(&normals).unwrap())
        );
        assert_eq!(
            bits(plan.reverse_one_factor(&adjoints).unwrap()),
            bits(base.reverse_one_factor(&adjoints).unwrap())
        );
        // For independent unit-variance normals, the deterministic transform's
        // covariance must be the identity; no Monte Carlo acceptance noise.
        let columns: Vec<_> = (0..4)
            .map(|j| {
                let mut normal = [0.0; 4];
                normal[j] = 1.0;
                plan.apply_one_factor(&normal).unwrap()
            })
            .collect();
        for i in 0..4 {
            for j in 0..4 {
                let covariance: f64 = columns.iter().map(|c| c[i] * c[j]).sum();
                assert!((covariance - f64::from(i == j)).abs() < 2.0e-15);
            }
        }
    }
}
