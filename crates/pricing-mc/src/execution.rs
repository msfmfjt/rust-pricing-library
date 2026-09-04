use std::error::Error;
use std::fmt;
use std::num::{NonZeroU32, NonZeroU64};
use std::ops::Range;

use pricing_numerics::{NeumaierSum, reduce_sums};
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionPolicy {
    worker_threads: NonZeroU32,
    reduction_block_size: NonZeroU64,
}

impl ExecutionPolicy {
    pub const VERSION: u32 = 1;
    pub const DEFAULT_REDUCTION_BLOCK_SIZE: u64 = 4096;

    pub fn new(
        worker_threads: u32,
        reduction_block_size: Option<u64>,
    ) -> Result<Self, ExecutorBuildError> {
        let worker_threads =
            NonZeroU32::new(worker_threads).ok_or(ExecutorBuildError::ZeroWorkerThreads)?;
        let resolved_block_size =
            reduction_block_size.unwrap_or(Self::DEFAULT_REDUCTION_BLOCK_SIZE);
        let reduction_block_size = NonZeroU64::new(resolved_block_size)
            .ok_or(ExecutorBuildError::ZeroReductionBlockSize)?;
        Ok(Self {
            worker_threads,
            reduction_block_size,
        })
    }

    #[must_use]
    pub const fn version(self) -> u32 {
        Self::VERSION
    }

    #[must_use]
    pub const fn worker_threads(self) -> NonZeroU32 {
        self.worker_threads
    }

    #[must_use]
    pub const fn reduction_block_size(self) -> NonZeroU64 {
        self.reduction_block_size
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExecutorBuildError {
    ZeroWorkerThreads,
    ZeroReductionBlockSize,
    ThreadCountUnsupported { value: u32 },
    PoolBuild,
}

impl fmt::Display for ExecutorBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWorkerThreads => write!(formatter, "worker-thread count must be positive"),
            Self::ZeroReductionBlockSize => {
                write!(formatter, "reduction-block size must be positive")
            }
            Self::ThreadCountUnsupported { value } => {
                write!(formatter, "worker-thread count {value} does not fit usize")
            }
            Self::PoolBuild => write!(
                formatter,
                "failed to build the calculation-owned thread pool"
            ),
        }
    }
}

impl Error for ExecutorBuildError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExecutionError {
    TooManyReductionBlocks { count: u64 },
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyReductionBlocks { count } => {
                write!(
                    formatter,
                    "reduction-block count {count} does not fit usize"
                )
            }
        }
    }
}

impl Error for ExecutionError {}

/// Executes fixed logical blocks on a calculation-owned Rayon pool.
pub struct DeterministicExecutor {
    policy: ExecutionPolicy,
    pool: ThreadPool,
}

impl DeterministicExecutor {
    pub fn new(policy: ExecutionPolicy) -> Result<Self, ExecutorBuildError> {
        let thread_count = usize::try_from(policy.worker_threads().get()).map_err(|_| {
            ExecutorBuildError::ThreadCountUnsupported {
                value: policy.worker_threads().get(),
            }
        })?;
        let pool = ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .thread_name(|index| format!("pricing-worker-{index}"))
            .build()
            .map_err(|_| ExecutorBuildError::PoolBuild)?;
        Ok(Self { policy, pool })
    }

    #[must_use]
    pub const fn policy(&self) -> ExecutionPolicy {
        self.policy
    }

    pub fn map_reduce<F>(
        &self,
        sampling_units: u64,
        evaluate: F,
    ) -> Result<NeumaierSum, ExecutionError>
    where
        F: Fn(u64) -> f64 + Sync + Send,
    {
        let blocks = fixed_blocks(sampling_units, self.policy.reduction_block_size().get())?;
        let partials = self.pool.install(|| {
            blocks
                .into_par_iter()
                .map(|block| block.map(&evaluate).collect::<NeumaierSum>())
                .collect::<Vec<_>>()
        });
        Ok(reduce_sums(partials))
    }
}

fn fixed_blocks(sampling_units: u64, block_size: u64) -> Result<Vec<Range<u64>>, ExecutionError> {
    let block_count = sampling_units.div_ceil(block_size);
    let capacity = usize::try_from(block_count)
        .map_err(|_| ExecutionError::TooManyReductionBlocks { count: block_count })?;
    let mut blocks = Vec::with_capacity(capacity);
    let mut begin = 0;
    while begin < sampling_units {
        let end = begin.saturating_add(block_size).min(sampling_units);
        blocks.push(begin..end);
        begin = end;
    }
    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    fn executor(worker_threads: u32, block_size: u64) -> DeterministicExecutor {
        let policy = ExecutionPolicy::new(worker_threads, Some(block_size)).expect("valid policy");
        DeterministicExecutor::new(policy).expect("thread pool")
    }

    #[test]
    fn policy_validates_and_records_resolved_values() {
        let policy = ExecutionPolicy::new(3, None).expect("valid policy");
        assert_eq!(policy.version(), 1);
        assert_eq!(policy.worker_threads().get(), 3);
        assert_eq!(policy.reduction_block_size().get(), 4096);
        assert_eq!(
            ExecutionPolicy::new(0, None),
            Err(ExecutorBuildError::ZeroWorkerThreads)
        );
        assert_eq!(
            ExecutionPolicy::new(1, Some(0)),
            Err(ExecutorBuildError::ZeroReductionBlockSize)
        );
    }

    #[test]
    fn fixed_blocks_are_contiguous_and_independent_of_worker_count() {
        assert_eq!(fixed_blocks(0, 4), Ok(Vec::new()));
        assert_eq!(fixed_blocks(10, 4), Ok(vec![0..4, 4..8, 8..10]));
    }

    #[test]
    fn worker_count_and_completion_order_do_not_change_result_bits() {
        let completion_counter = AtomicU64::new(0);
        let evaluate = |index: u64| {
            if index.is_multiple_of(3) {
                std::thread::yield_now();
            }
            completion_counter.fetch_add(1, Ordering::Relaxed);
            match index % 4 {
                0 => 1.0e16,
                1 => 1.0,
                2 => -1.0e16,
                _ => 2.0,
            }
        };
        let single = executor(1, 7)
            .map_reduce(10_003, &evaluate)
            .expect("execution");
        let parallel = executor(4, 7)
            .map_reduce(10_003, &evaluate)
            .expect("execution");
        assert_eq!(single.sum().to_bits(), parallel.sum().to_bits());
        assert_eq!(
            single.correction().to_bits(),
            parallel.correction().to_bits()
        );
        assert_eq!(single.total().to_bits(), parallel.total().to_bits());
        assert_eq!(completion_counter.load(Ordering::Relaxed), 20_006);
    }

    #[test]
    fn empty_execution_has_positive_zero_bits() {
        let result = executor(2, 8)
            .map_reduce(0, |_| unreachable!("no sampling unit"))
            .expect("empty execution");
        assert_eq!(result.sum().to_bits(), 0.0_f64.to_bits());
        assert_eq!(result.correction().to_bits(), 0.0_f64.to_bits());
        assert_eq!(result.total().to_bits(), 0.0_f64.to_bits());
    }
}
