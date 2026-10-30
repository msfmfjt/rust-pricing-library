//! Resource failures must return before invoking numerical callbacks.
use std::convert::Infallible;
use std::num::NonZeroU32;

use pricing::mc::{AadConfigError, DeterministicExecutor, ExecutionPolicy, TryExecutionError};

fn executor(block_size: u64) -> DeterministicExecutor {
    DeterministicExecutor::new(ExecutionPolicy::new(2, Some(block_size)).unwrap()).unwrap()
}

#[test]
fn impossible_block_tables_fail_before_scalar_evaluation_and_allow_reuse() {
    for block_size in [1, 2] {
        let executor = executor(block_size);
        assert!(
            executor
                .map_reduce(u64::MAX, |_| panic!("must not evaluate"))
                .is_err()
        );
        assert!(matches!(
            executor.try_map_reduce_statistics(u64::MAX, |_| -> Result<f64, Infallible> {
                panic!("must not evaluate")
            }),
            Err(TryExecutionError::Execution(_))
        ));
        let sum = executor.map_reduce(5, |i| i as f64).unwrap();
        assert_eq!(sum.total(), 10.0);
    }
}

#[test]
fn impossible_block_tables_fail_before_vector_and_aad_workspace_evaluation() {
    let executor = executor(1);
    assert!(matches!(
        executor.try_map_reduce_statistics_vector(u64::MAX, 2, |_, _| -> Result<(), Infallible> {
            panic!("must not evaluate")
        }),
        Err(TryExecutionError::Execution(_))
    ));
    let tile = NonZeroU32::new(4).unwrap();
    assert!(matches!(
        executor.try_map_reduce_statistics_array_tiled(
            u64::MAX,
            tile,
            |_| -> Result<[f64; 2], Infallible> { panic!("must not evaluate") }
        ),
        Err(TryExecutionError::Execution(_))
    ));
    assert!(matches!(
        executor.try_map_reduce_statistics_array_with_aad_workspace(
            u64::MAX,
            tile,
            2,
            |_, _, _| -> Result<[f64; 2], AadConfigError> { panic!("must not evaluate") }
        ),
        Err(TryExecutionError::Execution(_))
    ));
}

#[test]
fn maximum_block_size_keeps_empty_and_short_ranges_exact() {
    // The resource constraint applies to the block table, not to block_size itself.
    let executor = executor(u64::MAX);
    assert_eq!(
        executor
            .map_reduce(0, |_| panic!("empty range"))
            .unwrap()
            .total()
            .to_bits(),
        0.0_f64.to_bits()
    );
    let stats = executor
        .try_map_reduce_statistics(5, |i| Ok::<_, Infallible>(i as f64))
        .unwrap();
    assert_eq!(stats.sum().total(), 10.0);
    assert_eq!(stats.moments().count(), 5);
    assert_eq!(stats.moments().mean(), 2.0);
    assert_eq!(stats.moments().sample_variance(), Some(2.5));
}
