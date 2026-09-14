use crate::core::{Date, NodeId, UnderlyingId};
use crate::product::CompactC2Smoothing;
use crate::product::graph::GraphError;
use crate::product::graph::GraphFingerprint;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompiledOpcode {
    Literal {
        value: f64,
        output: u32,
    },
    TerminalSpot {
        underlying: UnderlyingId,
        observation_date: Date,
        output: u32,
    },
    PreDividendSpot {
        underlying: UnderlyingId,
        observation_date: Date,
        output: u32,
    },
    Add {
        left: u32,
        right: u32,
        output: u32,
    },
    Subtract {
        left: u32,
        right: u32,
        output: u32,
    },
    Multiply {
        left: u32,
        right: u32,
        output: u32,
    },
    Divide {
        numerator: u32,
        denominator: u32,
        output: u32,
    },
    Minimum {
        left: u32,
        right: u32,
        output: u32,
    },
    Maximum {
        left: u32,
        right: u32,
        output: u32,
    },
    Indicator {
        input: u32,
        output: u32,
    },
    SmoothMinimum {
        left: u32,
        right: u32,
        smoothing: CompactC2Smoothing,
        output: u32,
    },
    SmoothMaximum {
        left: u32,
        right: u32,
        smoothing: CompactC2Smoothing,
        output: u32,
    },
    SmoothIndicator {
        input: u32,
        smoothing: CompactC2Smoothing,
        output: u32,
    },
    BarrierHitState {
        previous: u32,
        hit_weight: u32,
        output: u32,
    },
    Negate {
        input: u32,
        output: u32,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompiledPayoff {
    pub(in crate::engine) opcodes: Box<[CompiledOpcode]>,
    pub(in crate::engine) output_slots: Box<[u32]>,
    pub(in crate::engine) source_to_slot: Box<[(NodeId, u32)]>,
    pub(in crate::engine) removed_source_nodes: Box<[NodeId]>,
    pub(in crate::engine) source_fingerprint: GraphFingerprint,
    pub(in crate::engine) tape_fingerprint: GraphFingerprint,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerminalAdjoint {
    pub underlying: UnderlyingId,
    pub observation_date: Date,
    pub value: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreDividendAdjoint {
    pub underlying: UnderlyingId,
    pub observation_date: Date,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PayoffEvaluation {
    pub value: f64,
    pub terminal_adjoints: Box<[TerminalAdjoint]>,
    pub pre_dividend_adjoints: Box<[PreDividendAdjoint]>,
}

impl CompiledPayoff {
    #[must_use]
    pub fn opcodes(&self) -> &[CompiledOpcode] {
        &self.opcodes
    }

    #[must_use]
    pub fn output_slots(&self) -> &[u32] {
        &self.output_slots
    }

    #[must_use]
    pub fn source_to_slot(&self) -> &[(NodeId, u32)] {
        &self.source_to_slot
    }

    #[must_use]
    pub fn removed_source_nodes(&self) -> &[NodeId] {
        &self.removed_source_nodes
    }

    #[must_use]
    pub const fn source_fingerprint(&self) -> GraphFingerprint {
        self.source_fingerprint
    }

    #[must_use]
    pub const fn tape_fingerprint(&self) -> GraphFingerprint {
        self.tape_fingerprint
    }

    #[must_use]
    pub fn terminal_observations(&self) -> Vec<(UnderlyingId, Date)> {
        let mut observations = BTreeSet::new();
        for opcode in &self.opcodes {
            if let CompiledOpcode::TerminalSpot {
                underlying,
                observation_date,
                ..
            }
            | CompiledOpcode::PreDividendSpot {
                underlying,
                observation_date,
                ..
            } = *opcode
            {
                observations.insert((underlying, observation_date));
            }
        }
        observations.into_iter().collect()
    }

    #[must_use]
    pub fn pre_dividend_observations(&self) -> Vec<(UnderlyingId, Date)> {
        self.opcodes
            .iter()
            .filter_map(|opcode| match *opcode {
                CompiledOpcode::PreDividendSpot {
                    underlying,
                    observation_date,
                    ..
                } => Some((underlying, observation_date)),
                _ => None,
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn evaluate<F>(&self, mut observation: F) -> Result<Vec<f64>, GraphError>
    where
        F: FnMut(UnderlyingId, Date) -> Option<f64>,
    {
        self.evaluate_with_pre_dividend_spots(&mut observation, |_, _| None)
    }

    pub fn evaluate_with_pre_dividend_spots<F, G>(
        &self,
        mut observation: F,
        mut pre_dividend_observation: G,
    ) -> Result<Vec<f64>, GraphError>
    where
        F: FnMut(UnderlyingId, Date) -> Option<f64>,
        G: FnMut(UnderlyingId, Date) -> Option<f64>,
    {
        let mut slots = vec![0.0; self.opcodes.len()];
        for opcode in &self.opcodes {
            let (output, value) = execute_opcode(
                *opcode,
                &slots,
                &mut observation,
                &mut pre_dividend_observation,
            )?;
            let output = checked_index(output, slots.len())?;
            if !value.is_finite() {
                return Err(GraphError::NonFiniteRuntimeValue {
                    opcode: opcode.name(),
                    bits: value.to_bits(),
                });
            }
            slots[output] = value;
        }
        self.output_slots
            .iter()
            .map(|slot| checked_index(*slot, slots.len()).map(|index| slots[index]))
            .collect()
    }

    pub fn evaluate_single_with_terminal_adjoint<F>(
        &self,
        mut observation: F,
    ) -> Result<PayoffEvaluation, GraphError>
    where
        F: FnMut(UnderlyingId, Date) -> Option<f64>,
    {
        self.evaluate_single_with_observation_adjoints(&mut observation, |_, _| None)
    }

    pub fn evaluate_single_with_observation_adjoints<F, G>(
        &self,
        mut observation: F,
        mut pre_dividend_observation: G,
    ) -> Result<PayoffEvaluation, GraphError>
    where
        F: FnMut(UnderlyingId, Date) -> Option<f64>,
        G: FnMut(UnderlyingId, Date) -> Option<f64>,
    {
        if self.output_slots.len() != 1 {
            return Err(GraphError::ReverseRequiresSingleOutput {
                count: self.output_slots.len(),
            });
        }
        self.evaluate_output_with_observation_adjoints(
            0,
            &mut observation,
            &mut pre_dividend_observation,
        )
    }

    pub fn evaluate_output_with_observation_adjoints<F, G>(
        &self,
        output_index: usize,
        mut observation: F,
        mut pre_dividend_observation: G,
    ) -> Result<PayoffEvaluation, GraphError>
    where
        F: FnMut(UnderlyingId, Date) -> Option<f64>,
        G: FnMut(UnderlyingId, Date) -> Option<f64>,
    {
        let output_slot =
            self.output_slots
                .get(output_index)
                .copied()
                .ok_or(GraphError::InvalidOutputIndex {
                    index: output_index,
                    count: self.output_slots.len(),
                })?;
        let mut values = vec![0.0; self.opcodes.len()];
        for opcode in &self.opcodes {
            let (output, value) = execute_opcode(
                *opcode,
                &values,
                &mut observation,
                &mut pre_dividend_observation,
            )?;
            let output = checked_index(output, values.len())?;
            if !value.is_finite() {
                return Err(GraphError::NonFiniteRuntimeValue {
                    opcode: opcode.name(),
                    bits: value.to_bits(),
                });
            }
            values[output] = value;
        }
        let output = checked_index(output_slot, values.len())?;
        let value = values[output];
        let mut adjoints = vec![0.0; self.opcodes.len()];
        adjoints[output] = 1.0;
        let mut terminal_adjoints = Vec::new();
        let mut pre_dividend_adjoints = Vec::new();
        for opcode in self.opcodes.iter().rev().copied() {
            reverse_opcode(
                opcode,
                &values,
                &mut adjoints,
                &mut terminal_adjoints,
                &mut pre_dividend_adjoints,
            )?;
        }
        Ok(PayoffEvaluation {
            value,
            terminal_adjoints: terminal_adjoints.into_boxed_slice(),
            pre_dividend_adjoints: pre_dividend_adjoints.into_boxed_slice(),
        })
    }
}

pub(in crate::engine) fn reverse_opcode(
    opcode: CompiledOpcode,
    values: &[f64],
    adjoints: &mut [f64],
    terminal_adjoints: &mut Vec<TerminalAdjoint>,
    pre_dividend_adjoints: &mut Vec<PreDividendAdjoint>,
) -> Result<(), GraphError> {
    let output = checked_index(opcode.output(), adjoints.len())?;
    let output_adjoint = adjoints[output];
    match opcode {
        CompiledOpcode::Literal { .. } => {}
        CompiledOpcode::TerminalSpot {
            underlying,
            observation_date,
            ..
        } => terminal_adjoints.push(TerminalAdjoint {
            underlying,
            observation_date,
            value: output_adjoint,
        }),
        CompiledOpcode::PreDividendSpot {
            underlying,
            observation_date,
            ..
        } => pre_dividend_adjoints.push(PreDividendAdjoint {
            underlying,
            observation_date,
            value: output_adjoint,
        }),
        CompiledOpcode::Add { left, right, .. } => {
            add_adjoint(adjoints, left, output_adjoint, opcode.name())?;
            add_adjoint(adjoints, right, output_adjoint, opcode.name())?;
        }
        CompiledOpcode::Subtract { left, right, .. } => {
            add_adjoint(adjoints, left, output_adjoint, opcode.name())?;
            add_adjoint(adjoints, right, -output_adjoint, opcode.name())?;
        }
        CompiledOpcode::Multiply { left, right, .. } => {
            let left_value = values[checked_index(left, values.len())?];
            let right_value = values[checked_index(right, values.len())?];
            add_adjoint(adjoints, left, output_adjoint * right_value, opcode.name())?;
            add_adjoint(adjoints, right, output_adjoint * left_value, opcode.name())?;
        }
        CompiledOpcode::Divide {
            numerator,
            denominator,
            ..
        } => {
            let numerator_value = values[checked_index(numerator, values.len())?];
            let denominator_value = values[checked_index(denominator, values.len())?];
            add_adjoint(
                adjoints,
                numerator,
                output_adjoint / denominator_value,
                opcode.name(),
            )?;
            add_adjoint(
                adjoints,
                denominator,
                -output_adjoint * numerator_value / (denominator_value * denominator_value),
                opcode.name(),
            )?;
        }
        CompiledOpcode::Minimum { left, right, .. } => {
            let left_value = values[checked_index(left, values.len())?];
            let right_value = values[checked_index(right, values.len())?];
            if left_value <= right_value {
                add_adjoint(adjoints, left, output_adjoint, opcode.name())?;
            } else {
                add_adjoint(adjoints, right, output_adjoint, opcode.name())?;
            }
        }
        CompiledOpcode::Maximum { left, right, .. } => {
            let left_value = values[checked_index(left, values.len())?];
            let right_value = values[checked_index(right, values.len())?];
            if left_value >= right_value {
                add_adjoint(adjoints, left, output_adjoint, opcode.name())?;
            } else {
                add_adjoint(adjoints, right, output_adjoint, opcode.name())?;
            }
        }
        CompiledOpcode::Indicator { .. } => {}
        CompiledOpcode::SmoothMinimum {
            left,
            right,
            smoothing,
            ..
        } => {
            let derivatives = smoothing.minimum(
                values[checked_index(left, values.len())?],
                values[checked_index(right, values.len())?],
            );
            add_adjoint(
                adjoints,
                left,
                output_adjoint * derivatives.left_first,
                opcode.name(),
            )?;
            add_adjoint(
                adjoints,
                right,
                output_adjoint * derivatives.right_first,
                opcode.name(),
            )?;
        }
        CompiledOpcode::SmoothMaximum {
            left,
            right,
            smoothing,
            ..
        } => {
            let derivatives = smoothing.maximum(
                values[checked_index(left, values.len())?],
                values[checked_index(right, values.len())?],
            );
            add_adjoint(
                adjoints,
                left,
                output_adjoint * derivatives.left_first,
                opcode.name(),
            )?;
            add_adjoint(
                adjoints,
                right,
                output_adjoint * derivatives.right_first,
                opcode.name(),
            )?;
        }
        CompiledOpcode::SmoothIndicator {
            input, smoothing, ..
        } => {
            let derivative = smoothing
                .indicator(values[checked_index(input, values.len())?])
                .first;
            add_adjoint(adjoints, input, output_adjoint * derivative, opcode.name())?;
        }
        CompiledOpcode::BarrierHitState {
            previous,
            hit_weight,
            ..
        } => {
            let previous_value = values[checked_index(previous, values.len())?];
            let hit_weight_value = values[checked_index(hit_weight, values.len())?];
            add_adjoint(
                adjoints,
                previous,
                output_adjoint * (1.0 - hit_weight_value),
                opcode.name(),
            )?;
            add_adjoint(
                adjoints,
                hit_weight,
                output_adjoint * (1.0 - previous_value),
                opcode.name(),
            )?;
        }
        CompiledOpcode::Negate { input, .. } => {
            add_adjoint(adjoints, input, -output_adjoint, opcode.name())?;
        }
    }
    Ok(())
}

pub(in crate::engine) fn add_adjoint(
    adjoints: &mut [f64],
    slot: u32,
    contribution: f64,
    opcode: &'static str,
) -> Result<(), GraphError> {
    let index = checked_index(slot, adjoints.len())?;
    let updated = adjoints[index] + contribution;
    if !updated.is_finite() {
        return Err(GraphError::NonFiniteRuntimeAdjoint {
            opcode,
            bits: updated.to_bits(),
        });
    }
    adjoints[index] = updated;
    Ok(())
}

impl CompiledOpcode {
    pub(in crate::engine) const fn name(self) -> &'static str {
        match self {
            Self::Literal { .. } => "literal",
            Self::TerminalSpot { .. } => "terminal_spot",
            Self::PreDividendSpot { .. } => "pre_dividend_spot",
            Self::Add { .. } => "add",
            Self::Subtract { .. } => "subtract",
            Self::Multiply { .. } => "multiply",
            Self::Divide { .. } => "divide",
            Self::Minimum { .. } => "minimum",
            Self::Maximum { .. } => "maximum",
            Self::Indicator { .. } => "indicator",
            Self::SmoothMinimum { .. } => "smooth_minimum",
            Self::SmoothMaximum { .. } => "smooth_maximum",
            Self::SmoothIndicator { .. } => "smooth_indicator",
            Self::BarrierHitState { .. } => "barrier_hit_state",
            Self::Negate { .. } => "negate",
        }
    }

    pub(in crate::engine) const fn output(self) -> u32 {
        match self {
            Self::Literal { output, .. }
            | Self::TerminalSpot { output, .. }
            | Self::PreDividendSpot { output, .. }
            | Self::Add { output, .. }
            | Self::Subtract { output, .. }
            | Self::Multiply { output, .. }
            | Self::Divide { output, .. }
            | Self::Minimum { output, .. }
            | Self::Maximum { output, .. }
            | Self::Indicator { output, .. }
            | Self::SmoothMinimum { output, .. }
            | Self::SmoothMaximum { output, .. }
            | Self::SmoothIndicator { output, .. }
            | Self::BarrierHitState { output, .. }
            | Self::Negate { output, .. } => output,
        }
    }
}

pub(in crate::engine) fn execute_opcode<F>(
    opcode: CompiledOpcode,
    slots: &[f64],
    observation: &mut F,
    pre_dividend_observation: &mut impl FnMut(UnderlyingId, Date) -> Option<f64>,
) -> Result<(u32, f64), GraphError>
where
    F: FnMut(UnderlyingId, Date) -> Option<f64>,
{
    let binary = |left: u32, right: u32| -> Result<(f64, f64), GraphError> {
        Ok((
            slots[checked_index(left, slots.len())?],
            slots[checked_index(right, slots.len())?],
        ))
    };
    match opcode {
        CompiledOpcode::Literal { value, output } => Ok((output, value)),
        CompiledOpcode::TerminalSpot {
            underlying,
            observation_date,
            output,
        } => observation(underlying, observation_date)
            .map(|value| (output, value))
            .ok_or(GraphError::MissingObservation {
                underlying,
                observation_date,
            }),
        CompiledOpcode::PreDividendSpot {
            underlying,
            observation_date,
            output,
        } => pre_dividend_observation(underlying, observation_date)
            .map(|value| (output, value))
            .ok_or(GraphError::MissingPreDividendObservation {
                underlying,
                observation_date,
            }),
        CompiledOpcode::Add {
            left,
            right,
            output,
        } => binary(left, right).map(|(left, right)| (output, left + right)),
        CompiledOpcode::Subtract {
            left,
            right,
            output,
        } => binary(left, right).map(|(left, right)| (output, left - right)),
        CompiledOpcode::Multiply {
            left,
            right,
            output,
        } => binary(left, right).map(|(left, right)| (output, left * right)),
        CompiledOpcode::Divide {
            numerator,
            denominator,
            output,
        } => binary(numerator, denominator)
            .map(|(numerator, denominator)| (output, numerator / denominator)),
        CompiledOpcode::Minimum {
            left,
            right,
            output,
        } => binary(left, right)
            .map(|(left, right)| (output, if left <= right { left } else { right })),
        CompiledOpcode::Maximum {
            left,
            right,
            output,
        } => binary(left, right)
            .map(|(left, right)| (output, if left >= right { left } else { right })),
        CompiledOpcode::Indicator { input, output } => {
            let value = slots[checked_index(input, slots.len())?];
            Ok((output, if value >= 0.0 { 1.0 } else { 0.0 }))
        }
        CompiledOpcode::SmoothMinimum {
            left,
            right,
            smoothing,
            output,
        } => {
            binary(left, right).map(|(left, right)| (output, smoothing.minimum(left, right).value))
        }
        CompiledOpcode::SmoothMaximum {
            left,
            right,
            smoothing,
            output,
        } => {
            binary(left, right).map(|(left, right)| (output, smoothing.maximum(left, right).value))
        }
        CompiledOpcode::SmoothIndicator {
            input,
            smoothing,
            output,
        } => {
            let value = slots[checked_index(input, slots.len())?];
            Ok((output, smoothing.indicator(value).value))
        }
        CompiledOpcode::BarrierHitState {
            previous,
            hit_weight,
            output,
        } => binary(previous, hit_weight)
            .map(|(previous, hit_weight)| (output, previous + (1.0 - previous) * hit_weight)),
        CompiledOpcode::Negate { input, output } => {
            Ok((output, -slots[checked_index(input, slots.len())?]))
        }
    }
}

pub(in crate::engine) fn checked_index(index: u32, length: usize) -> Result<usize, GraphError> {
    let index = usize::try_from(index).map_err(|_| GraphError::InvalidSlot { index, length })?;
    if index < length {
        Ok(index)
    } else {
        Err(GraphError::InvalidSlot {
            index: u32::try_from(index).unwrap_or(u32::MAX),
            length,
        })
    }
}
