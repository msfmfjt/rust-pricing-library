//! Public task errors retain native causes for machine-readable handling.
use std::error::Error;

use pricing::mc::{ExecutionError, RqmcPlanError, TryExecutionError};
use pricing::multi_asset::MultiAssetError;
use pricing::product::GraphError;
use pricing::{JsonLimits, PricingTask, parse_pricing_task_json};
use serde_json::{Value, json};

fn document() -> Value {
    serde_json::from_str(include_str!(
        "../../../fixtures/task-v1/multi_asset_bs.golden.json"
    ))
    .unwrap()
}

fn task(value: &Value) -> PricingTask {
    parse_pricing_task_json(&serde_json::to_vec(value).unwrap(), JsonLimits::DEFAULT).unwrap()
}

fn cause<'a, T: Error + 'static>(mut error: &'a (dyn Error + 'static)) -> Option<&'a T> {
    loop {
        if let Some(value) = error.downcast_ref::<T>() {
            return Some(value);
        }
        error = error.source()?;
    }
}

#[test]
fn oversized_reductions_retain_the_typed_execution_error() {
    let mut value = document();
    value["source"]["engine"]["independent_sampling_units"] = json!(u64::MAX);
    value["source"]["numerics"]["reduction_block_size"] = json!(1);
    for operation in ["evaluate", "aad"] {
        value["operation"] = json!({"type": operation});
        let error = task(&value).compile().unwrap().evaluate().unwrap_err();
        match cause::<ExecutionError>(&error).expect("retained executor cause") {
            ExecutionError::TooManyReductionBlocks { count }
            | ExecutionError::ReductionBlockAllocationFailed { count } => {
                assert_eq!(*count, u64::MAX);
            }
            other => panic!("unexpected execution error: {other}"),
        }
    }
}

#[test]
fn rqmc_compile_failure_retains_the_typed_dimension_error() {
    let mut value = document();
    value["operation"] = json!({"type": "evaluate"});
    value["source"]["engine"] = json!({
        "type": "randomized_quasi_monte_carlo", "points_per_scramble": 2,
        "master_scramble_seed": 7, "scramble_count": 2,
        "variance_reduction": {"antithetic": false, "brownian_bridge": false},
    });
    value["source"]["numerics"]["maximum_step"] = json!(1.0 / 11000.0);
    let error = task(&value).compile().expect_err("dimension limit");
    assert!(matches!(
        cause::<RqmcPlanError>(&error),
        Some(RqmcPlanError::DimensionLimitExceeded { .. })
    ));
}

#[test]
fn runtime_graph_errors_retain_the_failing_sampling_unit_and_graph_cause() {
    let mut value = document();
    // The original basket graph multiplies asset 1 by this finite literal.
    // Compilation succeeds, but every sampled payoff overflows at runtime.
    value["source"]["product"]["nodes"][2]["operation"]["value"] = json!(1e308);
    for workers in [1, 3] {
        value["source"]["numerics"]["worker_threads"] = json!(workers);
        for operation in ["evaluate", "aad"] {
            value["operation"] = json!({"type": operation});
            let error = task(&value).compile().unwrap().evaluate().unwrap_err();
            assert!(matches!(
                cause::<TryExecutionError<MultiAssetError>>(&error),
                Some(TryExecutionError::Evaluation {
                    sampling_unit: 0,
                    ..
                })
            ));
            assert!(matches!(
                cause::<GraphError>(&error),
                Some(GraphError::NonFiniteRuntimeValue { .. })
            ));
        }
    }
}
