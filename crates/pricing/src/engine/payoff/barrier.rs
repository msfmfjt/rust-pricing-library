//! payoff / barrier implementation.

use crate::engine::plan::simulation::BARRIER_BRIDGE_HIT_WEIGHT;
use crate::engine::plan::simulation::BARRIER_CERTAIN_SURVIVAL_COUNT;
use crate::engine::plan::simulation::BARRIER_DIVIDEND_JUMP_HIT;
use crate::engine::plan::simulation::BARRIER_ENDPOINT_HIT;
use crate::engine::plan::simulation::BARRIER_FINITE_CORRECTION_COUNT;
use crate::engine::plan::simulation::BARRIER_INTERVAL_COUNT;
use crate::engine::plan::simulation::BARRIER_SURVIVAL_UNDERFLOW_COUNT;
use crate::engine::plan::simulation::BARRIER_ZERO_VARIANCE_COUNT;
use crate::engine::plan::simulation::BarrierPathDiagnosticValues;
use crate::engine::plan::simulation::ContinuousBarrierBridgeEvaluation;
use crate::engine::plan::simulation::ContinuousBarrierPayoffTerms;
use crate::engine::plan::simulation::ContinuousBarrierRuntime;
use crate::engine::plan::simulation::ExactContinuousBarrierBridgeEvaluation;
use crate::engine::plan::simulation::ExactLocalVolContinuousBarrierBridgeEvaluation;
use crate::engine::plan::simulation::LocalVolContinuousBarrierBridgeEvaluation;
use crate::engine::plan::simulation::LocalVolRuntime;
use crate::engine::plan::simulation::PATHWISE_COMPONENTS;
use crate::engine::plan::simulation::SmoothedBarrierHitFactor;
use crate::engine::plan::simulation::SmoothedBarrierHitKind;
use crate::engine::plan::simulation::SmoothedContinuousBarrierPath;
use crate::market::LocalVarianceInterpolation;
use crate::mc::{
    BarrierBridgeDirection, BarrierBridgePath, BarrierBridgeStatus, SmoothedBarrierBridgeEndpoint,
    SmoothedBarrierBridgeInterval, SmoothedBarrierBridgeIntervalAdjoints,
};
use crate::product::{BarrierDirection, BarrierStyle, CompactC2Smoothing, OptionSide};

impl ExactContinuousBarrierBridgeEvaluation {
    pub(in crate::engine) fn survival(&self) -> f64 {
        if self.endpoint_touched || self.dividend_jump_touched {
            0.0
        } else {
            self.path.survival()
        }
    }

    pub(in crate::engine) fn diagnostic_values(&self) -> BarrierPathDiagnosticValues {
        BarrierPathDiagnosticValues::from_bridge(
            &self.path,
            self.endpoint_touched,
            self.dividend_jump_touched,
        )
    }
}

impl SmoothedContinuousBarrierPath {
    pub(in crate::engine) fn evaluate(
        intervals: Vec<SmoothedBarrierBridgeInterval>,
        hit_factors: Vec<SmoothedBarrierHitFactor>,
    ) -> Self {
        let mut log_bridge_survival = 0.0;
        let mut finite_correction_count = 0_u32;
        let mut zero_variance_count = 0_u32;
        let mut survival_underflow_count = 0_u32;
        let mut certain_survival_count = 0_u32;
        for interval in &intervals {
            log_bridge_survival += interval.log_survival();
            match interval.status() {
                BarrierBridgeStatus::FiniteCorrection => {
                    finite_correction_count = finite_correction_count.saturating_add(1);
                }
                BarrierBridgeStatus::TouchedEndpoint => {}
                BarrierBridgeStatus::ZeroVariance => {
                    zero_variance_count = zero_variance_count.saturating_add(1);
                }
                BarrierBridgeStatus::SurvivalUnderflow => {
                    survival_underflow_count = survival_underflow_count.saturating_add(1);
                }
                BarrierBridgeStatus::CertainSurvival => {
                    certain_survival_count = certain_survival_count.saturating_add(1);
                }
            }
        }
        let log_hit_survival = hit_factors
            .iter()
            .fold(0.0, |total, factor| total + factor.safety_weight().ln());
        Self {
            intervals: intervals.into_boxed_slice(),
            hit_factors: hit_factors.into_boxed_slice(),
            log_bridge_survival,
            log_hit_survival,
            finite_correction_count,
            zero_variance_count,
            survival_underflow_count,
            certain_survival_count,
        }
    }

    pub(in crate::engine) fn survival(&self) -> f64 {
        (self.log_bridge_survival + self.log_hit_survival).exp()
    }

    pub(in crate::engine) fn reverse(
        &self,
        survival_adjoint: f64,
    ) -> (
        Vec<SmoothedBarrierBridgeIntervalAdjoints>,
        Vec<(Option<usize>, f64)>,
    ) {
        let log_survival_adjoint = survival_adjoint * self.survival();
        let intervals = self
            .intervals
            .iter()
            .map(|interval| interval.reverse(log_survival_adjoint))
            .collect();
        let hit_factors = self
            .hit_factors
            .iter()
            .map(|factor| {
                let derivative = if factor.safety_weight() == 0.0 {
                    0.0
                } else {
                    -log_survival_adjoint * factor.state_derivative / factor.safety_weight()
                };
                (factor.state_index, derivative)
            })
            .collect();
        (intervals, hit_factors)
    }

    pub(in crate::engine) fn diagnostic_values(&self) -> BarrierPathDiagnosticValues {
        let component_hit = |kind| {
            1.0 - self
                .hit_factors
                .iter()
                .filter(|factor| factor.kind == kind)
                .fold(1.0, |survival, factor| survival * factor.safety_weight())
        };
        BarrierPathDiagnosticValues {
            endpoint_hit: component_hit(SmoothedBarrierHitKind::Endpoint),
            dividend_jump_hit: component_hit(SmoothedBarrierHitKind::DividendJump),
            bridge_hit_weight: 1.0 - self.log_bridge_survival.exp(),
            interval_count: self.intervals.len() as f64,
            finite_correction_count: f64::from(self.finite_correction_count),
            zero_variance_count: f64::from(self.zero_variance_count),
            survival_underflow_count: f64::from(self.survival_underflow_count),
            certain_survival_count: f64::from(self.certain_survival_count),
        }
    }
}

impl SmoothedBarrierHitFactor {
    pub(in crate::engine) fn safety_weight(self) -> f64 {
        1.0 - self.hit_weight
    }
}

impl ContinuousBarrierBridgeEvaluation {
    pub(in crate::engine) fn survival(&self) -> f64 {
        match self {
            Self::Exact(evaluation) => evaluation.survival(),
            Self::Smoothed { path, .. } => path.survival(),
        }
    }

    pub(in crate::engine) fn diagnostic_values(&self) -> BarrierPathDiagnosticValues {
        match self {
            Self::Exact(evaluation) => evaluation.diagnostic_values(),
            Self::Smoothed { path, .. } => path.diagnostic_values(),
        }
    }
}

impl ExactLocalVolContinuousBarrierBridgeEvaluation {
    pub(in crate::engine) fn survival(&self) -> f64 {
        if self.endpoint_touched || self.dividend_jump_touched {
            0.0
        } else {
            self.path.survival()
        }
    }

    pub(in crate::engine) fn diagnostic_values(&self) -> BarrierPathDiagnosticValues {
        BarrierPathDiagnosticValues::from_bridge(
            &self.path,
            self.endpoint_touched,
            self.dividend_jump_touched,
        )
    }
}

impl LocalVolContinuousBarrierBridgeEvaluation {
    pub(in crate::engine) fn survival(&self) -> f64 {
        match self {
            Self::Exact(evaluation) => evaluation.survival(),
            Self::Smoothed { path, .. } => path.survival(),
        }
    }

    pub(in crate::engine) fn diagnostic_values(&self) -> BarrierPathDiagnosticValues {
        match self {
            Self::Exact(evaluation) => evaluation.diagnostic_values(),
            Self::Smoothed { path, .. } => path.diagnostic_values(),
        }
    }
}

impl BarrierPathDiagnosticValues {
    pub(in crate::engine) fn from_bridge(
        path: &BarrierBridgePath,
        endpoint_touched: bool,
        dividend_jump_touched: bool,
    ) -> Self {
        let diagnostics = path.diagnostics();
        Self {
            endpoint_hit: if endpoint_touched { 1.0 } else { 0.0 },
            dividend_jump_hit: if dividend_jump_touched { 1.0 } else { 0.0 },
            bridge_hit_weight: if endpoint_touched || dividend_jump_touched {
                0.0
            } else {
                1.0 - path.survival()
            },
            interval_count: f64::from(diagnostics.interval_count),
            finite_correction_count: f64::from(diagnostics.finite_correction_count),
            zero_variance_count: f64::from(diagnostics.zero_variance_count),
            survival_underflow_count: f64::from(diagnostics.survival_underflow_count),
            certain_survival_count: f64::from(diagnostics.certain_survival_count),
        }
    }

    pub(in crate::engine) fn write_to(self, values: &mut [f64; PATHWISE_COMPONENTS]) {
        values[BARRIER_ENDPOINT_HIT] = self.endpoint_hit;
        values[BARRIER_DIVIDEND_JUMP_HIT] = self.dividend_jump_hit;
        values[BARRIER_BRIDGE_HIT_WEIGHT] = self.bridge_hit_weight;
        values[BARRIER_INTERVAL_COUNT] = self.interval_count;
        values[BARRIER_FINITE_CORRECTION_COUNT] = self.finite_correction_count;
        values[BARRIER_ZERO_VARIANCE_COUNT] = self.zero_variance_count;
        values[BARRIER_SURVIVAL_UNDERFLOW_COUNT] = self.survival_underflow_count;
        values[BARRIER_CERTAIN_SURVIVAL_COUNT] = self.certain_survival_count;
    }
}

pub(in crate::engine) const fn bridge_direction(
    direction: BarrierDirection,
) -> BarrierBridgeDirection {
    match direction {
        BarrierDirection::Up => BarrierBridgeDirection::Up,
        BarrierDirection::Down => BarrierBridgeDirection::Down,
    }
}

pub(in crate::engine) fn barrier_touched(
    direction: BarrierDirection,
    state: f64,
    barrier: f64,
) -> bool {
    match direction {
        BarrierDirection::Up => state >= barrier,
        BarrierDirection::Down => state <= barrier,
    }
}

pub(in crate::engine) fn smoothed_endpoint_hit_factor(
    endpoint: SmoothedBarrierBridgeEndpoint,
    state_index: Option<usize>,
    kind: SmoothedBarrierHitKind,
) -> SmoothedBarrierHitFactor {
    SmoothedBarrierHitFactor {
        kind,
        state_index,
        hit_weight: endpoint.hit_weight(),
        state_derivative: endpoint.reverse(1.0, 0.0).state,
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn smoothed_jump_hit_factor(
    direction: BarrierDirection,
    pre_spot: f64,
    post_spot: f64,
    pre_affine_scale: f64,
    post_affine_scale: f64,
    barrier: f64,
    smoothing: CompactC2Smoothing,
    state_index: Option<usize>,
) -> SmoothedBarrierHitFactor {
    let direction_sign = match direction {
        BarrierDirection::Up => 1.0,
        BarrierDirection::Down => -1.0,
    };
    let pre_distance = direction_sign * (pre_spot - barrier);
    let post_distance = direction_sign * (post_spot - barrier);
    let score = smoothing.maximum(pre_distance, post_distance);
    let hit = smoothing.indicator(score.value);
    let score_state_derivative = direction_sign
        * (score.left_first * pre_affine_scale + score.right_first * post_affine_scale);
    SmoothedBarrierHitFactor {
        kind: SmoothedBarrierHitKind::DividendJump,
        state_index,
        hit_weight: hit.value,
        state_derivative: hit.first * score_state_derivative,
    }
}

pub(in crate::engine) fn accumulate_bridge_local_variance_adjoints(
    local_volatility: &LocalVolRuntime,
    states: &[f64],
    node_interpolations: &[LocalVarianceInterpolation],
    variance_seeds: Vec<f64>,
    state_seeds: &mut [f64],
    bridge_grid_adjoints: &mut [f64],
) {
    let x_count = local_volatility.grid.log_moneyness_nodes().len();
    for (node, (interpolation, variance_seed)) in node_interpolations
        .iter()
        .copied()
        .zip(variance_seeds)
        .enumerate()
    {
        state_seeds[node] += variance_seed
            * local_volatility
                .grid
                .interpolation_log_moneyness_derivative(interpolation)
            / states[node];
        interpolation.transpose_accumulate(variance_seed, bridge_grid_adjoints, x_count);
    }
}

pub(in crate::engine) fn continuous_barrier_payoff_terms(
    barrier: &ContinuousBarrierRuntime,
    terminal: f64,
    survival: f64,
) -> ContinuousBarrierPayoffTerms {
    let (signed_intrinsic, terminal_sign) = match barrier.side {
        OptionSide::Call => (terminal - barrier.strike, 1.0),
        OptionSide::Put => (barrier.strike - terminal, -1.0),
    };
    let vanilla = signed_intrinsic.max(0.0) * barrier.notional;
    let vanilla_derivative = if signed_intrinsic >= 0.0 {
        terminal_sign * barrier.notional
    } else {
        0.0
    };
    match barrier.style {
        BarrierStyle::KnockOut => ContinuousBarrierPayoffTerms {
            value: barrier.rebate + survival * (vanilla - barrier.rebate),
            terminal_derivative: survival * vanilla_derivative,
            survival_derivative: vanilla - barrier.rebate,
        },
        BarrierStyle::KnockIn => ContinuousBarrierPayoffTerms {
            value: vanilla + survival * (barrier.rebate - vanilla),
            terminal_derivative: (1.0 - survival) * vanilla_derivative,
            survival_derivative: barrier.rebate - vanilla,
        },
    }
}
