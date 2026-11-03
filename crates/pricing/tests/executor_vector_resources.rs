//! Dynamic result widths must fail through Result before numerical callbacks.
use std::convert::Infallible;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};

use pricing::mc::{
    DeterministicExecutor, DeterministicStatistics, ExecutionError, ExecutionPolicy,
    TryExecutionError,
};

fn executor(workers: u32, block: u64) -> DeterministicExecutor {
    DeterministicExecutor::new(ExecutionPolicy::new(workers, Some(block)).unwrap()).unwrap()
}

fn reject_impossible_width(count: u64) {
    let executor = executor(2, 1);
    let callbacks = AtomicUsize::new(0);
    let result = catch_unwind(AssertUnwindSafe(|| {
        executor.try_map_reduce_statistics_vector(count, usize::MAX, |_, _| {
            callbacks.fetch_add(1, Ordering::SeqCst);
            Ok::<_, Infallible>(())
        })
    }))
    .expect("an impossible output width must return an error, not panic");
    assert!(matches!(
        result,
        Err(TryExecutionError::Execution(
            ExecutionError::VectorStatisticsAllocationFailed {
                components: usize::MAX
            }
        ))
    ));
    assert_eq!(callbacks.load(Ordering::SeqCst), 0);
    assert_eq!(executor.map_reduce(5, |i| i as f64).unwrap().total(), 10.0);
}

#[test]
fn impossible_nonempty_vector_width_is_a_typed_error() {
    reject_impossible_width(1);
}

#[test]
fn impossible_empty_vector_width_is_a_typed_error() {
    reject_impossible_width(0);
}

fn assert_same_bits(actual: DeterministicStatistics, expected: DeterministicStatistics) {
    assert_eq!(actual.sum().sum().to_bits(), expected.sum().sum().to_bits());
    assert_eq!(
        actual.sum().correction().to_bits(),
        expected.sum().correction().to_bits()
    );
    assert_eq!(
        actual.sum().total().to_bits(),
        expected.sum().total().to_bits()
    );
    assert_eq!(actual.moments().count(), expected.moments().count());
    assert_eq!(
        actual.moments().mean().to_bits(),
        expected.moments().mean().to_bits()
    );
    assert_eq!(
        actual.moments().second_moment().to_bits(),
        expected.moments().second_moment().to_bits()
    );
    assert_eq!(
        actual.moments().sample_variance().map(f64::to_bits),
        expected.moments().sample_variance().map(f64::to_bits)
    );
}

#[test]
fn vector_results_match_independent_scalar_reductions_and_reset_partial_writes() {
    let values = |i: u64| {
        [
            match i % 4 {
                0 => 1e16,
                1 => 1.0,
                2 => -1e16,
                _ => 2.0,
            },
            if i.is_multiple_of(3) { 7.0 } else { 0.0 },
            1e12 + (i % 5) as f64,
        ]
    };
    for block in [1, 7, u64::MAX] {
        let scalar = executor(1, block);
        let expected: Vec<_> = (0..3)
            .map(|c| {
                scalar
                    .try_map_reduce_statistics(103, |i| Ok::<_, Infallible>(values(i)[c]))
                    .unwrap()
            })
            .collect();
        for workers in [1, 3] {
            let executor = executor(workers, block);
            let result = executor
                .try_map_reduce_statistics_vector(103, 3, |i, out| {
                    assert!(out.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()));
                    let sample = values(i);
                    out[0] = sample[0];
                    if i.is_multiple_of(3) {
                        out[1] = sample[1];
                    }
                    out[2] = sample[2];
                    Ok::<_, Infallible>(())
                })
                .unwrap();
            assert_eq!(result.len(), expected.len());
            for (&actual, &expected) in result.iter().zip(&expected) {
                assert_same_bits(actual, expected);
            }
        }
    }
}

#[test]
fn empty_vectors_and_empty_sampling_keep_their_callback_contracts() {
    let executor = executor(3, 7);
    let callbacks = AtomicUsize::new(0);
    let result = executor
        .try_map_reduce_statistics_vector(19, 0, |_, out| {
            assert!(out.is_empty());
            callbacks.fetch_add(1, Ordering::SeqCst);
            Ok::<_, Infallible>(())
        })
        .unwrap();
    assert!(result.is_empty());
    assert_eq!(callbacks.load(Ordering::SeqCst), 19);
    let result = executor
        .try_map_reduce_statistics_vector(0, 3, |_, _| -> Result<(), Infallible> {
            panic!("an empty sampling plan must not call evaluate")
        })
        .unwrap();
    assert_eq!(result.len(), 3);
    for actual in result {
        assert_same_bits(actual, DeterministicStatistics::default());
    }
}

#[test]
fn evaluation_errors_keep_sampling_order_and_leave_executor_reusable() {
    for workers in [1, 3] {
        let executor = executor(workers, 7);
        for _ in 0..3 {
            let result = executor.try_map_reduce_statistics_vector(103, 3, |i, _| {
                if i == 2 || i == 11 { Err(i) } else { Ok(()) }
            });
            assert!(matches!(
                result,
                Err(TryExecutionError::Evaluation {
                    sampling_unit: 2,
                    source: 2
                })
            ));
        }
        assert_eq!(executor.map_reduce(5, |i| i as f64).unwrap().total(), 10.0);
    }
}
