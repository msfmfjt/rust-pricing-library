//! Grid capacity failures must be typed and precede node generation/calibration.
use std::error::Error;

use pricing::mc::{LocalVolError, LocalVolTimeGrid};
use pricing::{JsonLimits, parse_pricing_task_json};
use serde_json::{Value, json};

#[test]
fn impossible_grid_buffers_return_allocation_errors_before_generating_nodes() {
    // The requested count fits usize, but its f64 buffer cannot fit isize.
    // try_reserve_exact rejects the layout without asking for a huge allocation.
    let intervals = 1_usize << (usize::BITS - 2);
    let step = 1.0 / intervals as f64;
    for times in [vec![1.0], vec![0.0, 0.5, 1.0]] {
        assert_eq!(
            LocalVolTimeGrid::compile(times, step),
            Err(LocalVolError::GridAllocationFailed {
                nodes: intervals + 1
            })
        );
    }
}

#[test]
fn the_complete_grid_count_is_checked_before_any_interval_is_generated() {
    let step = 1.0 / (1_usize << (usize::BITS - 1)) as f64;
    assert_eq!(
        LocalVolTimeGrid::compile(vec![1.0, 2.0], step),
        Err(LocalVolError::StepCountOverflow)
    );
    assert_eq!(
        LocalVolTimeGrid::compile(vec![1.0], f64::from_bits(1)),
        Err(LocalVolError::StepCountOverflow)
    );
}

#[test]
fn ordinary_and_underflowing_interval_ratios_keep_exact_nodes_and_events() {
    let grid = LocalVolTimeGrid::compile(vec![1.0, 0.25, 0.25, 0.6], 0.2).unwrap();
    assert_eq!(
        grid.nodes().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        [0.0_f64, 0.125, 0.25, 0.425, 0.6, 0.8, 1.0].map(f64::to_bits)
    );
    assert_eq!(grid.event_node_indices(), &[0, 2, 4, 6]);
    assert_eq!(grid.maximum_step().to_bits(), 0.2_f64.to_bits());

    let tiny = f64::from_bits(1);
    let grid = LocalVolTimeGrid::compile(vec![tiny], f64::MAX).unwrap();
    assert_eq!(grid.nodes(), &[0.0, tiny]);
    assert_eq!(grid.event_node_indices(), &[0, 1]);
    assert_eq!(grid.step_count(), 1);
}

#[test]
fn task_adapters_preserve_the_typed_grid_failure_and_healthy_plan_reuse() {
    for name in [
        "multi_asset_bs",
        "stochastic_dividend_bs",
        "stochastic_dividend_hw",
    ] {
        let path = format!(
            "{}/../../fixtures/task-v1/{name}.golden.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let original: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let healthy =
            parse_pricing_task_json(&serde_json::to_vec(&original).unwrap(), JsonLimits::DEFAULT)
                .unwrap()
                .compile()
                .unwrap();
        let baseline = healthy.evaluate().unwrap();
        for operation in [json!({"type": "evaluate"}), original["operation"].clone()] {
            let mut value = original.clone();
            value["operation"] = operation;
            let step = json!(1.0 / (1_usize << (usize::BITS - 2)) as f64);
            let source = &mut value["source"];
            if name == "multi_asset_bs" {
                source["numerics"]["maximum_step"] = step;
            } else {
                source["maximum_step"] = step.clone();
                if source["equity"]["numerics"]["maximum_step"].is_number() {
                    source["equity"]["numerics"]["maximum_step"] = step;
                }
            }
            let task =
                parse_pricing_task_json(&serde_json::to_vec(&value).unwrap(), JsonLimits::DEFAULT)
                    .unwrap();
            let error = task.compile().unwrap_err();
            let mut cause: &(dyn Error + 'static) = &error;
            loop {
                if let Some(error) = cause.downcast_ref::<LocalVolError>() {
                    assert!(matches!(error, LocalVolError::GridAllocationFailed { .. }));
                    break;
                }
                cause = cause.source().expect("typed LocalVolError source");
            }
        }
        assert_eq!(healthy.evaluate().unwrap(), baseline);
    }
}
