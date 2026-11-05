//! Shared configuration for the explicit Python LSV plan constructors.
use pricing::mc::{ExecutionPolicy, lsv::LsvParticleConfig};
use pyo3::prelude::*;

/// Validate particles before execution, preserving each caller's error domain.
#[allow(clippy::too_many_arguments)]
pub(super) fn particle_config_and_policy(
    particle_count: usize,
    calibration_seed: u64,
    log_bandwidth: f64,
    minimum_effective_samples: f64,
    retain_reverse_trace: bool,
    worker_threads: u32,
    reduction_block_size: Option<u64>,
    invalid: impl Fn(String) -> PyErr,
) -> PyResult<(LsvParticleConfig, ExecutionPolicy)> {
    let particles = LsvParticleConfig::new(
        particle_count,
        calibration_seed,
        log_bandwidth,
        minimum_effective_samples,
        retain_reverse_trace,
    )
    .map_err(|e| invalid(e.to_string()))?;
    let policy = ExecutionPolicy::new(worker_threads, reduction_block_size)
        .map_err(|e| invalid(e.to_string()))?;
    Ok((particles, policy))
}
