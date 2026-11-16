//! Array and AAD reductions retain scalar ordering, error precedence and empty semantics.
use std::convert::Infallible;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicUsize, Ordering};

use pricing::mc::{
    AadConfigError, DeterministicExecutor, DeterministicStatistics, ExecutionPolicy,
    TryExecutionError,
};

fn executor(workers: u32, block: u64) -> DeterministicExecutor {
    DeterministicExecutor::new(ExecutionPolicy::new(workers, Some(block)).unwrap()).unwrap()
}

fn sample(i: u64) -> [f64; 3] {
    [
        [1e16, 1.0, -1e16, 2.0][(i % 4) as usize],
        1e12 + (i % 5) as f64,
        if i.is_multiple_of(3) {
            -(i as f64) * 0.125
        } else {
            i as f64 / 7.0
        },
    ]
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
fn tiled_and_aad_arrays_match_scalar_bits_for_each_block_policy() {
    for block in [1, 7, 16, u64::MAX] {
        let scalar = executor(1, block);
        for count in [0, 1, 19, 103] {
            // Block size may change rounding; preserve each policy's own scalar result.
            let expected: [_; 3] = std::array::from_fn(|component| {
                scalar
                    .try_map_reduce_statistics(count, |i| Ok::<_, Infallible>(sample(i)[component]))
                    .unwrap()
            });
            for workers in [1, 3] {
                let executor = executor(workers, block);
                for capacity in [1, 3, 8, 32] {
                    let tile = NonZeroU32::new(capacity).unwrap();
                    let tiled = executor
                        .try_map_reduce_statistics_array_tiled(count, tile, |i| {
                            if i.is_multiple_of(3) {
                                std::thread::yield_now();
                            }
                            Ok::<_, Infallible>(sample(i))
                        })
                        .unwrap();
                    let aad = executor
                        .try_map_reduce_statistics_array_with_aad_workspace(
                            count,
                            tile,
                            2,
                            |i, lane, workspace| {
                                assert_eq!(lane as u64, (i % block) % u64::from(capacity));
                                if lane == 0 {
                                    for slot in 0..2 {
                                        for j in 0..workspace.logical_capacity() {
                                            assert_eq!(
                                                workspace.primal(slot)?.get(j)?.to_bits(),
                                                0
                                            );
                                            assert_eq!(
                                                workspace.adjoint(slot)?.get(j)?.to_bits(),
                                                0
                                            );
                                        }
                                        assert!(
                                            workspace
                                                .primal(slot)?
                                                .padded_lane_bits()
                                                .all(|bits| bits == 0)
                                        );
                                        assert!(
                                            workspace
                                                .adjoint(slot)?
                                                .padded_lane_bits()
                                                .all(|bits| bits == 0)
                                        );
                                    }
                                }
                                let values = sample(i);
                                workspace.primal_mut(0)?.set(lane, values[0])?;
                                workspace.adjoint_mut(0)?.set(lane, values[1])?;
                                workspace.primal_mut(1)?.set(lane, values[2])?;
                                Ok::<_, AadConfigError>([
                                    workspace.primal(0)?.get(lane)?,
                                    workspace.adjoint(0)?.get(lane)?,
                                    workspace.primal(1)?.get(lane)?,
                                ])
                            },
                        )
                        .unwrap();
                    for component in 0..3 {
                        assert_same_bits(tiled[component], expected[component]);
                        assert_same_bits(aad[component], expected[component]);
                    }
                }
            }
        }
    }
}

#[derive(Debug, PartialEq)]
enum TestError {
    Sample(u64),
    Workspace(AadConfigError),
}

impl From<AadConfigError> for TestError {
    fn from(error: AadConfigError) -> Self {
        Self::Workspace(error)
    }
}

fn assert_first_error<const N: usize>(
    result: Result<[DeterministicStatistics; N], TryExecutionError<TestError>>,
) {
    assert!(matches!(
        result,
        Err(TryExecutionError::Evaluation {
            sampling_unit: 2,
            source: TestError::Sample(2)
        })
    ));
}

fn check_error_order<const N: usize>() {
    for block in [1, 7, 16] {
        for workers in [1, 3] {
            let executor = executor(workers, block);
            for capacity in [1, 3, 8] {
                let tile = NonZeroU32::new(capacity).unwrap();
                for _ in 0..3 {
                    let evaluate = |i| {
                        if i == 2 {
                            std::thread::yield_now();
                        }
                        if matches!(i, 2 | 6 | 11) {
                            Err(TestError::Sample(i))
                        } else {
                            Ok([0.0; N])
                        }
                    };
                    assert_first_error(
                        executor.try_map_reduce_statistics_array_tiled(23, tile, evaluate),
                    );
                    assert_first_error(
                        executor.try_map_reduce_statistics_array_with_aad_workspace(
                            23,
                            tile,
                            1,
                            |i, _, _| evaluate(i),
                        ),
                    );
                }
            }
            // Failed calls leave both array paths usable.
            let tile = NonZeroU32::new(3).unwrap();
            let tiled = executor
                .try_map_reduce_statistics_array_tiled(5, tile, |i| Ok::<_, Infallible>([i as f64]))
                .unwrap();
            let aad = executor
                .try_map_reduce_statistics_array_with_aad_workspace(5, tile, 1, |i, _, _| {
                    Ok::<_, AadConfigError>([i as f64])
                })
                .unwrap();
            assert_eq!(tiled[0].sum().total(), 10.0);
            assert_same_bits(aad[0], tiled[0]);
        }
    }
}

#[test]
fn evaluation_errors_keep_lowest_sampling_unit_even_with_zero_components() {
    check_error_order::<3>();
    check_error_order::<0>();
}

#[test]
fn zero_components_evaluate_every_sample_and_empty_input_skips_callbacks() {
    let executor = executor(3, 7);
    let tile = NonZeroU32::new(3).unwrap();
    for count in [0, 19] {
        let calls = AtomicUsize::new(0);
        let tiled = executor
            .try_map_reduce_statistics_array_tiled(count, tile, |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok::<[f64; 0], Infallible>([])
            })
            .unwrap();
        assert!(tiled.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), count as usize);
        calls.store(0, Ordering::SeqCst);
        let aad = executor
            .try_map_reduce_statistics_array_with_aad_workspace(count, tile, 1, |_, _, _| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok::<[f64; 0], AadConfigError>([])
            })
            .unwrap();
        assert!(aad.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), count as usize);
    }
}

#[test]
fn aad_workspace_failure_keeps_source_and_first_block_location() {
    for workers in [1, 3] {
        let executor = executor(workers, 7);
        let tile = NonZeroU32::new(3).unwrap();
        let result = executor.try_map_reduce_statistics_array_with_aad_workspace(
            19,
            tile,
            usize::MAX,
            |_, _, _| -> Result<[f64; 1], TestError> {
                panic!("workspace failure must precede callbacks")
            },
        );
        assert!(matches!(
            result,
            Err(TryExecutionError::Evaluation {
                sampling_unit: 0,
                source: TestError::Workspace(AadConfigError::WorkspaceSizeOverflow)
            })
        ));
        let empty = executor
            .try_map_reduce_statistics_array_with_aad_workspace(
                0,
                tile,
                usize::MAX,
                |_, _, _| -> Result<[f64; 1], TestError> {
                    panic!("empty input must not evaluate")
                },
            )
            .unwrap();
        assert_same_bits(empty[0], DeterministicStatistics::default());
        let reused = executor
            .try_map_reduce_statistics_array_with_aad_workspace(5, tile, 1, |i, _, _| {
                Ok::<_, AadConfigError>([i as f64])
            })
            .unwrap();
        assert_eq!(reused[0].sum().total(), 10.0);
    }
}
