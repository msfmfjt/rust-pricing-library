use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt;

use crate::RqmcConfig;

pub const BROWNIAN_BRIDGE_ABI: &str = "brownian-bridge-f64-v1";

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BrownianBridgeInstruction {
    Terminal {
        node: u32,
        normal_rank: u32,
        standard_deviation: f64,
    },
    Interior {
        node: u32,
        left: u32,
        right: u32,
        normal_rank: u32,
        left_weight: f64,
        right_weight: f64,
        standard_deviation: f64,
    },
}

impl BrownianBridgeInstruction {
    #[must_use]
    pub const fn node(self) -> u32 {
        match self {
            Self::Terminal { node, .. } | Self::Interior { node, .. } => node,
        }
    }

    #[must_use]
    pub const fn normal_rank(self) -> u32 {
        match self {
            Self::Terminal { normal_rank, .. } | Self::Interior { normal_rank, .. } => normal_rank,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BrownianBridgeError {
    TooFewTimeNodes { count: usize },
    NonFiniteTime { index: usize, bits: u64 },
    NonZeroStart { bits: u64 },
    NonIncreasingTime {
        left_index: usize,
        left_bits: u64,
        right_bits: u64,
    },
    ZeroFactorCount,
    DimensionOverflow,
    SobolDimensionLimitExceeded { requested: u32, maximum: u32 },
    NormalCountMismatch { expected: usize, actual: usize },
    IncrementCountMismatch { expected: usize, actual: usize },
}

impl fmt::Display for BrownianBridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewTimeNodes { count } => write!(
                formatter,
                "Brownian bridge needs at least times 0 and T; received {count} nodes"
            ),
            Self::NonFiniteTime { index, bits } => write!(
                formatter,
                "Brownian bridge time {index} is non-finite: 0x{bits:016x}"
            ),
            Self::NonZeroStart { bits } => write!(
                formatter,
                "Brownian bridge must start at positive zero; received 0x{bits:016x}"
            ),
            Self::NonIncreasingTime {
                left_index,
                left_bits,
                right_bits,
            } => write!(
                formatter,
                "Brownian bridge times must increase strictly at pair {left_index}: 0x{left_bits:016x}, 0x{right_bits:016x}"
            ),
            Self::ZeroFactorCount => write!(formatter, "Brownian factor count must be positive"),
            Self::DimensionOverflow => {
                write!(formatter, "Brownian bridge dimension count overflowed u32")
            }
            Self::SobolDimensionLimitExceeded { requested, maximum } => write!(
                formatter,
                "Brownian bridge needs {requested} dimensions, exceeding the Sobol limit {maximum}"
            ),
            Self::NormalCountMismatch { expected, actual } => write!(
                formatter,
                "Brownian bridge expected {expected} input normals; received {actual}"
            ),
            Self::IncrementCountMismatch { expected, actual } => write!(
                formatter,
                "Brownian bridge reverse expected {expected} increment adjoints; received {actual}"
            ),
        }
    }
}

impl Error for BrownianBridgeError {}

#[derive(Clone, Copy, Debug, PartialEq)]
struct IntervalCandidate {
    variance: f64,
    time: f64,
    node: usize,
    left: usize,
    right: usize,
}

impl Eq for IntervalCandidate {}

impl Ord for IntervalCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.variance
            .total_cmp(&other.variance)
            .then_with(|| other.time.total_cmp(&self.time))
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for IntervalCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Precomputed non-uniform Brownian-bridge transform and factor layout.
#[derive(Clone, Debug, PartialEq)]
pub struct BrownianBridgePlan {
    times: Box<[f64]>,
    inverse_sqrt_dt: Box<[f64]>,
    instructions: Box<[BrownianBridgeInstruction]>,
    factor_count: u32,
    effective_dimension: u32,
}

impl BrownianBridgePlan {
    pub fn compile(times: Vec<f64>, factor_count: u32) -> Result<Self, BrownianBridgeError> {
        validate_times(&times)?;
        if factor_count == 0 {
            return Err(BrownianBridgeError::ZeroFactorCount);
        }
        let bridge_normal_count = times.len() - 1;
        let bridge_normal_count_u32 = u32::try_from(bridge_normal_count)
            .map_err(|_| BrownianBridgeError::DimensionOverflow)?;
        let effective_dimension = bridge_normal_count_u32
            .checked_mul(factor_count)
            .ok_or(BrownianBridgeError::DimensionOverflow)?;
        if effective_dimension > RqmcConfig::MAX_SOBOL_DIMENSION {
            return Err(BrownianBridgeError::SobolDimensionLimitExceeded {
                requested: effective_dimension,
                maximum: RqmcConfig::MAX_SOBOL_DIMENSION,
            });
        }

        let inverse_sqrt_dt = times
            .windows(2)
            .map(|pair| 1.0 / (pair[1] - pair[0]).sqrt())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let instructions = compile_instructions(&times).into_boxed_slice();
        Ok(Self {
            times: times.into_boxed_slice(),
            inverse_sqrt_dt,
            instructions,
            factor_count,
            effective_dimension,
        })
    }

    #[must_use]
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    #[must_use]
    pub fn instructions(&self) -> &[BrownianBridgeInstruction] {
        &self.instructions
    }

    #[must_use]
    pub const fn factor_count(&self) -> u32 {
        self.factor_count
    }

    #[must_use]
    pub const fn bridge_normal_count(&self) -> u32 {
        self.effective_dimension / self.factor_count
    }

    #[must_use]
    pub const fn effective_dimension(&self) -> u32 {
        self.effective_dimension
    }

    /// Bridge-rank-major, factor-minor Sobol dimension.
    pub fn dimension(&self, bridge_rank: u32, factor: u32) -> Option<u32> {
        if bridge_rank >= self.bridge_normal_count() || factor >= self.factor_count {
            return None;
        }
        bridge_rank
            .checked_mul(self.factor_count)
            .and_then(|base| base.checked_add(factor))
    }

    /// Applies the bridge independently to every factor and returns normalized
    /// increments in interval-major, factor-minor order.
    pub fn apply(&self, normals: &[f64]) -> Result<Vec<f64>, BrownianBridgeError> {
        let expected = usize::try_from(self.effective_dimension)
            .expect("validated Sobol dimension fits usize");
        if normals.len() != expected {
            return Err(BrownianBridgeError::NormalCountMismatch {
                expected,
                actual: normals.len(),
            });
        }
        let intervals = self.inverse_sqrt_dt.len();
        let factors = usize::try_from(self.factor_count).expect("u32 fits usize");
        let mut increments = vec![0.0; expected];
        let mut factor_normals = vec![0.0; intervals];
        for factor in 0..factors {
            for (rank, normal) in factor_normals.iter_mut().enumerate() {
                *normal = normals[rank * factors + factor];
            }
            let factor_increments = self.apply_one_factor(&factor_normals)?;
            for (interval, increment) in factor_increments.into_iter().enumerate() {
                increments[interval * factors + factor] = increment;
            }
        }
        Ok(increments)
    }

    pub fn apply_one_factor(&self, normals: &[f64]) -> Result<Vec<f64>, BrownianBridgeError> {
        let expected = self.inverse_sqrt_dt.len();
        if normals.len() != expected {
            return Err(BrownianBridgeError::NormalCountMismatch {
                expected,
                actual: normals.len(),
            });
        }
        let mut brownian_values = vec![0.0; self.times.len()];
        for instruction in &self.instructions {
            match *instruction {
                BrownianBridgeInstruction::Terminal {
                    node,
                    normal_rank,
                    standard_deviation,
                } => {
                    brownian_values[to_usize(node)] =
                        standard_deviation * normals[to_usize(normal_rank)];
                }
                BrownianBridgeInstruction::Interior {
                    node,
                    left,
                    right,
                    normal_rank,
                    left_weight,
                    right_weight,
                    standard_deviation,
                } => {
                    brownian_values[to_usize(node)] = left_weight * brownian_values[to_usize(left)]
                        + right_weight * brownian_values[to_usize(right)]
                        + standard_deviation * normals[to_usize(normal_rank)];
                }
            }
        }
        Ok(brownian_values
            .windows(2)
            .zip(self.inverse_sqrt_dt.iter())
            .map(|(values, inverse_sqrt_dt)| (values[1] - values[0]) * inverse_sqrt_dt)
            .collect())
    }

    /// Applies the transpose of `apply_one_factor` for Simulation reverse.
    pub fn reverse_one_factor(
        &self,
        increment_adjoints: &[f64],
    ) -> Result<Vec<f64>, BrownianBridgeError> {
        let expected = self.inverse_sqrt_dt.len();
        if increment_adjoints.len() != expected {
            return Err(BrownianBridgeError::IncrementCountMismatch {
                expected,
                actual: increment_adjoints.len(),
            });
        }
        let mut value_adjoints = vec![0.0; self.times.len()];
        for (interval, (adjoint, inverse_sqrt_dt)) in increment_adjoints
            .iter()
            .zip(self.inverse_sqrt_dt.iter())
            .enumerate()
        {
            let scaled = adjoint * inverse_sqrt_dt;
            value_adjoints[interval] -= scaled;
            value_adjoints[interval + 1] += scaled;
        }
        let mut normal_adjoints = vec![0.0; expected];
        for instruction in self.instructions.iter().rev() {
            match *instruction {
                BrownianBridgeInstruction::Terminal {
                    node,
                    normal_rank,
                    standard_deviation,
                } => {
                    normal_adjoints[to_usize(normal_rank)] +=
                        value_adjoints[to_usize(node)] * standard_deviation;
                }
                BrownianBridgeInstruction::Interior {
                    node,
                    left,
                    right,
                    normal_rank,
                    left_weight,
                    right_weight,
                    standard_deviation,
                } => {
                    let adjoint = value_adjoints[to_usize(node)];
                    normal_adjoints[to_usize(normal_rank)] += adjoint * standard_deviation;
                    value_adjoints[to_usize(left)] += adjoint * left_weight;
                    value_adjoints[to_usize(right)] += adjoint * right_weight;
                }
            }
        }
        Ok(normal_adjoints)
    }
}

fn validate_times(times: &[f64]) -> Result<(), BrownianBridgeError> {
    if times.len() < 2 {
        return Err(BrownianBridgeError::TooFewTimeNodes { count: times.len() });
    }
    for (index, time) in times.iter().enumerate() {
        if !time.is_finite() {
            return Err(BrownianBridgeError::NonFiniteTime {
                index,
                bits: time.to_bits(),
            });
        }
    }
    if times[0].to_bits() != 0.0_f64.to_bits() {
        return Err(BrownianBridgeError::NonZeroStart {
            bits: times[0].to_bits(),
        });
    }
    for (left_index, pair) in times.windows(2).enumerate() {
        if pair[1] <= pair[0] {
            return Err(BrownianBridgeError::NonIncreasingTime {
                left_index,
                left_bits: pair[0].to_bits(),
                right_bits: pair[1].to_bits(),
            });
        }
    }
    Ok(())
}

fn compile_instructions(times: &[f64]) -> Vec<BrownianBridgeInstruction> {
    let terminal = times.len() - 1;
    let mut instructions = Vec::with_capacity(terminal);
    instructions.push(BrownianBridgeInstruction::Terminal {
        node: to_u32(terminal),
        normal_rank: 0,
        standard_deviation: times[terminal].sqrt(),
    });
    let mut heap = BinaryHeap::new();
    if let Some(candidate) = interval_candidate(times, 0, terminal) {
        heap.push(candidate);
    }
    let mut normal_rank = 1_u32;
    while let Some(candidate) = heap.pop() {
        let denominator = times[candidate.right] - times[candidate.left];
        let left_weight = (times[candidate.right] - times[candidate.node]) / denominator;
        let right_weight = (times[candidate.node] - times[candidate.left]) / denominator;
        instructions.push(BrownianBridgeInstruction::Interior {
            node: to_u32(candidate.node),
            left: to_u32(candidate.left),
            right: to_u32(candidate.right),
            normal_rank,
            left_weight,
            right_weight,
            standard_deviation: candidate.variance.sqrt(),
        });
        normal_rank += 1;
        if let Some(left) = interval_candidate(times, candidate.left, candidate.node) {
            heap.push(left);
        }
        if let Some(right) = interval_candidate(times, candidate.node, candidate.right) {
            heap.push(right);
        }
    }
    instructions
}

fn interval_candidate(times: &[f64], left: usize, right: usize) -> Option<IntervalCandidate> {
    if right <= left + 1 {
        return None;
    }
    let denominator = times[right] - times[left];
    (left + 1..right)
        .map(|node| {
            let variance =
                ((times[node] - times[left]) * (times[right] - times[node])) / denominator;
            IntervalCandidate {
                variance,
                time: times[node],
                node,
                left,
                right,
            }
        })
        .max()
}

fn to_u32(value: usize) -> u32 {
    u32::try_from(value).expect("bridge node count was validated to fit u32")
}

fn to_usize(value: u32) -> usize {
    usize::try_from(value).expect("u32 fits usize on supported targets")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_grid_uses_variance_then_earlier_time_order() {
        let plan = BrownianBridgePlan::compile(vec![0.0, 0.25, 0.5, 0.75, 1.0], 1)
            .expect("valid grid");
        let nodes = plan
            .instructions()
            .iter()
            .map(|instruction| instruction.node())
            .collect::<Vec<_>>();
        assert_eq!(nodes, vec![4, 2, 1, 3]);
        let ranks = plan
            .instructions()
            .iter()
            .map(|instruction| instruction.normal_rank())
            .collect::<Vec<_>>();
        assert_eq!(ranks, vec![0, 1, 2, 3]);
    }

    #[test]
    fn nonuniform_coefficients_are_compiled_once() {
        let plan = BrownianBridgePlan::compile(vec![0.0, 0.1, 0.4, 1.0], 1)
            .expect("valid grid");
        assert_eq!(plan.instructions()[0].node(), 3);
        assert_eq!(plan.instructions()[1].node(), 2);
        let BrownianBridgeInstruction::Interior {
            left_weight,
            right_weight,
            standard_deviation,
            ..
        } = plan.instructions()[1]
        else {
            panic!("second instruction must be interior");
        };
        assert_eq!(left_weight.to_bits(), 0.6_f64.to_bits());
        assert_eq!(right_weight.to_bits(), 0.4_f64.to_bits());
        assert_eq!(standard_deviation.to_bits(), 0.24_f64.sqrt().to_bits());
    }

    #[test]
    fn forward_transform_matches_known_uniform_bridge() {
        let plan = BrownianBridgePlan::compile(vec![0.0, 0.25, 0.5, 0.75, 1.0], 1)
            .expect("valid grid");
        let increments = plan
            .apply_one_factor(&[1.0, 2.0, 3.0, 4.0])
            .expect("normal count");
        let root_eighth = 0.125_f64.sqrt();
        let expected = [
            2.0 * (0.75 + 3.0 * root_eighth),
            2.0 * (0.75 - 3.0 * root_eighth),
            2.0 * (-0.25 + 4.0 * root_eighth),
            2.0 * (-0.25 - 4.0 * root_eighth),
        ];
        for (actual, expected) in increments.iter().zip(expected) {
            assert!((actual - expected).abs() < 2.0e-15);
        }
    }

    #[test]
    fn factors_are_rank_major_on_input_and_interval_major_on_output() {
        let plan = BrownianBridgePlan::compile(vec![0.0, 0.5, 1.0], 3).expect("valid grid");
        assert_eq!(plan.effective_dimension(), 6);
        assert_eq!(plan.dimension(0, 0), Some(0));
        assert_eq!(plan.dimension(0, 2), Some(2));
        assert_eq!(plan.dimension(1, 0), Some(3));
        assert_eq!(plan.dimension(1, 2), Some(5));
        assert_eq!(plan.dimension(2, 0), None);
        assert_eq!(plan.dimension(0, 3), None);
        let combined = plan
            .apply(&[1.0, 10.0, 100.0, 2.0, 20.0, 200.0])
            .expect("six dimensions");
        for factor in 0..3 {
            let separate = plan
                .apply_one_factor(&[
                    [1.0, 10.0, 100.0][factor],
                    [2.0, 20.0, 200.0][factor],
                ])
                .expect("two bridge normals");
            assert_eq!(combined[factor].to_bits(), separate[0].to_bits());
            assert_eq!(combined[3 + factor].to_bits(), separate[1].to_bits());
        }
    }

    #[test]
    fn reverse_is_the_exact_transpose_of_forward_linear_map() {
        let plan = BrownianBridgePlan::compile(vec![0.0, 0.1, 0.4, 0.7, 1.0], 1)
            .expect("valid grid");
        let normals = [0.3, -1.1, 0.7, 2.0];
        let increment_adjoints = [-0.2, 0.9, 1.3, -0.4];
        let increments = plan.apply_one_factor(&normals).expect("normal count");
        let normal_adjoints = plan
            .reverse_one_factor(&increment_adjoints)
            .expect("increment count");
        let forward_dot = increments
            .iter()
            .zip(increment_adjoints)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        let reverse_dot = normals
            .iter()
            .zip(normal_adjoints)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        assert!((forward_dot - reverse_dot).abs() < 2.0e-15);
    }

    #[test]
    fn invalid_grids_dimensions_and_lengths_are_rejected() {
        assert!(matches!(
            BrownianBridgePlan::compile(vec![0.0], 1),
            Err(BrownianBridgeError::TooFewTimeNodes { .. })
        ));
        assert!(matches!(
            BrownianBridgePlan::compile(vec![-0.0, 1.0], 1),
            Err(BrownianBridgeError::NonZeroStart { .. })
        ));
        assert!(matches!(
            BrownianBridgePlan::compile(vec![0.0, 0.5, 0.5], 1),
            Err(BrownianBridgeError::NonIncreasingTime { .. })
        ));
        assert!(matches!(
            BrownianBridgePlan::compile(vec![0.0, f64::NAN], 1),
            Err(BrownianBridgeError::NonFiniteTime { .. })
        ));
        assert_eq!(
            BrownianBridgePlan::compile(vec![0.0, 1.0], 0),
            Err(BrownianBridgeError::ZeroFactorCount)
        );
        let too_many_dimensions = (0..=RqmcConfig::MAX_SOBOL_DIMENSION)
            .map(f64::from)
            .collect::<Vec<_>>();
        assert!(matches!(
            BrownianBridgePlan::compile(too_many_dimensions, 2),
            Err(BrownianBridgeError::SobolDimensionLimitExceeded { .. })
        ));
        let plan = BrownianBridgePlan::compile(vec![0.0, 0.5, 1.0], 1)
            .expect("valid grid");
        assert!(matches!(
            plan.apply_one_factor(&[1.0]),
            Err(BrownianBridgeError::NormalCountMismatch { .. })
        ));
        assert!(matches!(
            plan.reverse_one_factor(&[1.0]),
            Err(BrownianBridgeError::IncrementCountMismatch { .. })
        ));
    }
}
