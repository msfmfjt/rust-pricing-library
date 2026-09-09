use std::error::Error;
use std::fmt;

use pricing_market::{LocalVarianceBoundaryStats, LocalVarianceGrid};

use crate::{BrownianBridgeError, BrownianBridgePlan, Philox4x32, RandomCoordinate, RandomDomain};

pub const LOCAL_VOL_LOG_EULER_SCHEME: &str = "local-vol-log-euler-f-v1";

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LocalVolError {
    EmptyEventTimes,
    InvalidEventTime { index: usize, bits: u64 },
    NonPositiveFinalTime,
    NonPositiveMaximumStep { bits: u64 },
    StepCountOverflow,
    InvalidForwardNormalizer { index: usize, bits: u64 },
    InvalidInitialState { bits: u64 },
    ShockCountMismatch { expected: usize, actual: usize },
    NonFiniteState { step: usize, bits: u64 },
    BrownianBridge(BrownianBridgeError),
    Market(pricing_market::MarketError),
}

impl fmt::Display for LocalVolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEventTimes => write!(formatter, "Local Volatility event grid is empty"),
            Self::InvalidEventTime { index, bits } => write!(
                formatter,
                "Local Volatility event time {index} is invalid: 0x{bits:016x}"
            ),
            Self::NonPositiveFinalTime => write!(
                formatter,
                "Local Volatility event grid requires a positive final time"
            ),
            Self::NonPositiveMaximumStep { bits } => write!(
                formatter,
                "Local Volatility maximum step must be finite and positive: 0x{bits:016x}"
            ),
            Self::StepCountOverflow => write!(
                formatter,
                "Local Volatility event grid step count overflowed usize"
            ),
            Self::InvalidForwardNormalizer { index, bits } => write!(
                formatter,
                "Local Volatility forward normalizer {index} is invalid: 0x{bits:016x}"
            ),
            Self::InvalidInitialState { bits } => write!(
                formatter,
                "Local Volatility initial f state must be finite and positive: 0x{bits:016x}"
            ),
            Self::ShockCountMismatch { expected, actual } => write!(
                formatter,
                "Local Volatility Log-Euler expected {expected} shocks; received {actual}"
            ),
            Self::NonFiniteState { step, bits } => write!(
                formatter,
                "Local Volatility Log-Euler produced a non-finite state at step {step}: 0x{bits:016x}"
            ),
            Self::BrownianBridge(error) => error.fmt(formatter),
            Self::Market(error) => error.fmt(formatter),
        }
    }
}

impl Error for LocalVolError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::BrownianBridge(error) => Some(error),
            Self::Market(error) => Some(error),
            _ => None,
        }
    }
}

impl From<pricing_market::MarketError> for LocalVolError {
    fn from(value: pricing_market::MarketError) -> Self {
        Self::Market(value)
    }
}

impl From<BrownianBridgeError> for LocalVolError {
    fn from(value: BrownianBridgeError) -> Self {
        Self::BrownianBridge(value)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalVolTimeGrid {
    nodes: Box<[f64]>,
    event_node_indices: Box<[usize]>,
    maximum_step: f64,
}

impl LocalVolTimeGrid {
    pub fn compile(event_times: Vec<f64>, maximum_step: f64) -> Result<Self, LocalVolError> {
        if !maximum_step.is_finite() || maximum_step <= 0.0 {
            return Err(LocalVolError::NonPositiveMaximumStep {
                bits: maximum_step.to_bits(),
            });
        }
        if event_times.is_empty() {
            return Err(LocalVolError::EmptyEventTimes);
        }
        let mut event_times = event_times;
        for (index, time) in event_times.iter().copied().enumerate() {
            if !time.is_finite()
                || time < 0.0
                || (time == 0.0 && time.to_bits() != 0.0_f64.to_bits())
            {
                return Err(LocalVolError::InvalidEventTime {
                    index,
                    bits: time.to_bits(),
                });
            }
        }
        event_times.sort_by(f64::total_cmp);
        event_times.dedup_by(|left, right| left.to_bits() == right.to_bits());
        if event_times[0].to_bits() != 0.0_f64.to_bits() {
            event_times.insert(0, 0.0);
        }
        if event_times[event_times.len() - 1] <= 0.0 {
            return Err(LocalVolError::NonPositiveFinalTime);
        }

        let mut nodes = vec![0.0];
        let mut event_node_indices = vec![0];
        for pair in event_times.windows(2) {
            let left = pair[0];
            let right = pair[1];
            if right <= left {
                continue;
            }
            let interval = right - left;
            let steps = interval_step_count(interval, maximum_step)?;
            let substep = interval / steps as f64;
            for index in 1..steps {
                nodes.push(left + substep * index as f64);
            }
            nodes.push(right);
            event_node_indices.push(nodes.len() - 1);
        }

        Ok(Self {
            nodes: nodes.into_boxed_slice(),
            event_node_indices: event_node_indices.into_boxed_slice(),
            maximum_step,
        })
    }

    #[must_use]
    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    #[must_use]
    pub fn event_node_indices(&self) -> &[usize] {
        &self.event_node_indices
    }

    #[must_use]
    pub const fn maximum_step(&self) -> f64 {
        self.maximum_step
    }

    #[must_use]
    pub fn step_count(&self) -> usize {
        self.nodes.len() - 1
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalVolLogEulerPlan {
    time_grid: LocalVolTimeGrid,
    forward_normalizers: Box<[f64]>,
}

impl LocalVolLogEulerPlan {
    pub fn new(
        time_grid: LocalVolTimeGrid,
        forward_normalizers: Vec<f64>,
    ) -> Result<Self, LocalVolError> {
        if forward_normalizers.len() != time_grid.nodes().len() {
            return Err(LocalVolError::ShockCountMismatch {
                expected: time_grid.nodes().len(),
                actual: forward_normalizers.len(),
            });
        }
        for (index, value) in forward_normalizers.iter().copied().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(LocalVolError::InvalidForwardNormalizer {
                    index,
                    bits: value.to_bits(),
                });
            }
        }
        Ok(Self {
            time_grid,
            forward_normalizers: forward_normalizers.into_boxed_slice(),
        })
    }

    pub fn with_constant_forward(
        time_grid: LocalVolTimeGrid,
        forward: f64,
    ) -> Result<Self, LocalVolError> {
        Self::new(time_grid.clone(), vec![forward; time_grid.nodes().len()])
    }

    #[must_use]
    pub fn time_grid(&self) -> &LocalVolTimeGrid {
        &self.time_grid
    }

    #[must_use]
    pub fn forward_normalizers(&self) -> &[f64] {
        &self.forward_normalizers
    }

    pub fn evolve_path(
        &self,
        local_variance_grid: &LocalVarianceGrid,
        initial_f: f64,
        shocks: &[f64],
    ) -> Result<LocalVolPath, LocalVolError> {
        if !initial_f.is_finite() || initial_f <= 0.0 {
            return Err(LocalVolError::InvalidInitialState {
                bits: initial_f.to_bits(),
            });
        }
        if shocks.len() != self.time_grid.step_count() {
            return Err(LocalVolError::ShockCountMismatch {
                expected: self.time_grid.step_count(),
                actual: shocks.len(),
            });
        }

        let mut states = Vec::with_capacity(self.time_grid.nodes().len());
        let mut variances = Vec::with_capacity(self.time_grid.step_count());
        let mut boundary_stats = LocalVarianceBoundaryStats::default();
        let mut state = initial_f;
        states.push(state);
        for (step, shock) in shocks.iter().copied().enumerate() {
            let time = self.time_grid.nodes()[step];
            let next_time = self.time_grid.nodes()[step + 1];
            let dt = next_time - time;
            let x = (state / self.forward_normalizers[step]).ln();
            let interpolation =
                local_variance_grid.interpolate_and_record(time, x, &mut boundary_stats)?;
            let local_variance = interpolation.value;
            let local_volatility = local_variance.sqrt();
            state *= (-0.5 * local_variance * dt + local_volatility * dt.sqrt() * shock).exp();
            if !state.is_finite() || state <= 0.0 {
                return Err(LocalVolError::NonFiniteState {
                    step: step + 1,
                    bits: state.to_bits(),
                });
            }
            variances.push(local_variance);
            states.push(state);
        }
        Ok(LocalVolPath {
            states: states.into_boxed_slice(),
            local_variances: variances.into_boxed_slice(),
            boundary_stats,
        })
    }

    pub fn path_shocks(
        &self,
        master_seed: u64,
        path: u64,
        domain: RandomDomain,
        brownian_bridge: bool,
    ) -> Result<Vec<f64>, LocalVolError> {
        if brownian_bridge {
            self.brownian_bridge_path_shocks(master_seed, path, domain)
        } else {
            self.step_order_path_shocks(master_seed, path, domain)
        }
    }

    fn step_order_path_shocks(
        &self,
        master_seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, LocalVolError> {
        let generator = Philox4x32::from_seed(master_seed);
        let mut shocks = Vec::with_capacity(self.time_grid.step_count());
        for step in 0..self.time_grid.step_count() {
            let dimension = u32::try_from(step).map_err(|_| LocalVolError::StepCountOverflow)?;
            shocks.push(generator.standard_normal(RandomCoordinate::new(path, dimension, domain)));
        }
        Ok(shocks)
    }

    fn brownian_bridge_path_shocks(
        &self,
        master_seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, LocalVolError> {
        let bridge = BrownianBridgePlan::compile(self.time_grid.nodes().to_vec(), 1)?;
        let generator = Philox4x32::from_seed(master_seed);
        let mut bridge_normals = Vec::with_capacity(self.time_grid.step_count());
        for rank in 0..self.time_grid.step_count() {
            let rank = u32::try_from(rank).map_err(|_| LocalVolError::StepCountOverflow)?;
            let dimension = bridge
                .dimension(rank, 0)
                .ok_or(LocalVolError::StepCountOverflow)?;
            bridge_normals
                .push(generator.standard_normal(RandomCoordinate::new(path, dimension, domain)));
        }
        bridge.apply_one_factor(&bridge_normals).map_err(Into::into)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalVolPath {
    states: Box<[f64]>,
    local_variances: Box<[f64]>,
    boundary_stats: LocalVarianceBoundaryStats,
}

impl LocalVolPath {
    #[must_use]
    pub fn states(&self) -> &[f64] {
        &self.states
    }

    #[must_use]
    pub fn local_variances(&self) -> &[f64] {
        &self.local_variances
    }

    #[must_use]
    pub const fn boundary_stats(&self) -> LocalVarianceBoundaryStats {
        self.boundary_stats
    }
}

fn interval_step_count(interval: f64, maximum_step: f64) -> Result<usize, LocalVolError> {
    let steps = (interval / maximum_step).ceil();
    if !steps.is_finite() || steps > usize::MAX as f64 {
        return Err(LocalVolError::StepCountOverflow);
    }
    Ok(steps as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_substep_grid_preserves_events_and_maximum_step() {
        let grid = LocalVolTimeGrid::compile(vec![1.0, 0.25, 0.25, 0.6], 0.2).expect("grid");
        assert_eq!(grid.nodes()[0].to_bits(), 0.0_f64.to_bits());
        assert!(grid.nodes().contains(&0.25));
        assert!(grid.nodes().contains(&0.6));
        assert!(grid.nodes().contains(&1.0));
        assert!(grid.nodes().windows(2).all(|pair| pair[1] > pair[0]));
        assert!(
            grid.nodes()
                .windows(2)
                .all(|pair| pair[1] - pair[0] <= 0.2 + 1.0e-15)
        );
        assert_eq!(grid.event_node_indices().len(), 4);
    }

    #[test]
    fn event_substep_grid_rejects_missing_or_invalid_inputs() {
        assert!(LocalVolTimeGrid::compile(vec![], 0.2).is_err());
        assert!(LocalVolTimeGrid::compile(vec![0.0], 0.2).is_err());
        assert!(LocalVolTimeGrid::compile(vec![1.0], 0.0).is_err());
        assert!(LocalVolTimeGrid::compile(vec![f64::NAN], 0.2).is_err());
        assert!(LocalVolTimeGrid::compile(vec![-0.0, 1.0], 0.2).is_err());
    }

    #[test]
    fn log_euler_matches_constant_variance_closed_form_for_frozen_shocks() {
        let time_grid = LocalVolTimeGrid::compile(vec![1.0], 0.25).expect("time grid");
        let plan = LocalVolLogEulerPlan::with_constant_forward(time_grid, 100.0).expect("plan");
        let variance_grid = LocalVarianceGrid::new(
            vec![0.0, 1.0],
            vec![-1.0, 1.0],
            vec![0.04, 0.04, 0.04, 0.04],
            0.0001,
            1.0,
        )
        .expect("variance grid");
        let shocks = vec![0.1, -0.2, 0.3, -0.4];
        let path = plan
            .evolve_path(&variance_grid, 100.0, &shocks)
            .expect("path");
        let expected_log_return = -0.5 * 0.04
            + 0.2
                * shocks
                    .iter()
                    .map(|shock| 0.25_f64.sqrt() * shock)
                    .sum::<f64>();
        let expected = 100.0 * expected_log_return.exp();
        assert!((path.states()[path.states().len() - 1] - expected).abs() < 1.0e-12);
        assert!(path.local_variances().iter().all(|value| *value == 0.04));
        assert_eq!(path.boundary_stats().total_flat_count(), 0);
    }

    #[test]
    fn log_euler_records_boundary_use_from_grid_interpolation() {
        let time_grid = LocalVolTimeGrid::compile(vec![0.5], 0.5).expect("time grid");
        let plan = LocalVolLogEulerPlan::with_constant_forward(time_grid, 100.0).expect("plan");
        let variance_grid = LocalVarianceGrid::new(
            vec![0.0, 0.5],
            vec![-0.01, 0.01],
            vec![0.04, 0.04, 0.04, 0.04],
            0.0001,
            1.0,
        )
        .expect("variance grid");
        let path = plan
            .evolve_path(&variance_grid, 150.0, &[0.0])
            .expect("path");
        assert_eq!(path.boundary_stats().right_flat_count, 1);
        assert!(path.boundary_stats().max_right_excursion > 0.39);
    }

    #[test]
    fn path_shocks_are_stable_by_path_and_dimension() {
        let time_grid = LocalVolTimeGrid::compile(vec![1.0], 0.25).expect("time grid");
        let plan = LocalVolLogEulerPlan::with_constant_forward(time_grid, 100.0).expect("plan");
        let first = plan
            .path_shocks(11, 7, RandomDomain::Valuation, false)
            .expect("first");
        let second = plan
            .path_shocks(11, 7, RandomDomain::Valuation, false)
            .expect("second");
        let other_path = plan
            .path_shocks(11, 8, RandomDomain::Valuation, false)
            .expect("other path");
        assert_eq!(first, second);
        assert_ne!(first, other_path);
    }

    #[test]
    fn brownian_bridge_shocks_reuse_compiled_nonuniform_grid_convention() {
        let time_grid = LocalVolTimeGrid::compile(vec![0.1, 0.4, 1.0], 1.0).expect("time grid");
        let plan = LocalVolLogEulerPlan::with_constant_forward(time_grid, 100.0).expect("plan");
        let shocks = plan
            .path_shocks(11, 7, RandomDomain::Valuation, true)
            .expect("bridge shocks");
        assert_eq!(shocks.len(), plan.time_grid().step_count());
        assert!(shocks.iter().all(|shock| shock.is_finite()));
        assert_ne!(
            shocks,
            plan.path_shocks(11, 7, RandomDomain::Valuation, false)
                .expect("plain shocks")
        );
    }
}
