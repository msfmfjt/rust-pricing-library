use crate::core::{Date, NodeId, UnderlyingId};
use crate::product::graph::GraphError;
use crate::product::graph::GraphFingerprint;
use crate::product::{CompactC2Smoothing, GraphComparison};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompiledOpcode {
    Select {
        condition: u32,
        then_value: u32,
        else_value: u32,
        output: u32,
    },
    Log {
        input: u32,
        output: u32,
    },
    Exp {
        input: u32,
        output: u32,
    },
    Compare {
        left: u32,
        right: u32,
        operation: GraphComparison,
        output: u32,
    },
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
    pub(in crate::engine) branching: bool,
    pub(in crate::engine) opcodes: Box<[CompiledOpcode]>,
    pub(in crate::engine) output_slots: Box<[u32]>,
    pub(in crate::engine) source_to_slot: Box<[(NodeId, u32)]>,
    pub(in crate::engine) removed_source_nodes: Box<[NodeId]>,
    pub(in crate::engine) source_fingerprint: GraphFingerprint,
    pub(in crate::engine) tape_fingerprint: GraphFingerprint,
}

/// Reusable path values and traversal state. A workspace may be reused across
/// paths for this payoff; resizing for another payoff is explicit in evaluation.
#[derive(Clone, Debug)]
pub struct PayoffWorkspace {
    values: Vec<f64>,
    visited: Vec<bool>,
    pending: Vec<u32>,
    outputs: Vec<f64>,
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
        if !self.branching {
            // Preserve the legacy allocation count and instruction order.
            let mut values = vec![0.0; self.opcodes.len()];
            for &opcode in &self.opcodes {
                let (output, value) = execute_opcode(
                    opcode,
                    &values,
                    &mut observation,
                    &mut pre_dividend_observation,
                )?;
                check_value(opcode, value)?;
                values[output as usize] = value;
            }
            return Ok(self
                .output_slots
                .iter()
                .map(|slot| values[*slot as usize])
                .collect());
        }
        let mut workspace = self.workspace();
        self.evaluate_reusing(
            &mut workspace,
            &mut observation,
            &mut pre_dividend_observation,
        )
        .map(|values| values.to_vec())
    }

    #[must_use]
    pub fn workspace(&self) -> PayoffWorkspace {
        PayoffWorkspace {
            values: vec![0.0; self.opcodes.len()],
            visited: if self.branching {
                vec![false; self.opcodes.len()]
            } else {
                vec![]
            },
            pending: Vec::new(),
            outputs: vec![0.0; self.output_slots.len()],
        }
    }

    /// Evaluate native compiled operations without allocating path arrays after
    /// workspace initialization (and the first branch traversal capacity growth).
    pub fn evaluate_reusing<'a, F, G>(
        &self,
        workspace: &'a mut PayoffWorkspace,
        mut observation: F,
        mut pre_dividend_observation: G,
    ) -> Result<&'a [f64], GraphError>
    where
        F: FnMut(UnderlyingId, Date) -> Option<f64>,
        G: FnMut(UnderlyingId, Date) -> Option<f64>,
    {
        workspace.values.resize(self.opcodes.len(), 0.0);
        workspace.outputs.resize(self.output_slots.len(), 0.0);
        if self.branching {
            workspace.visited.resize(self.opcodes.len(), false);
            workspace.visited.fill(false);
            workspace.values.fill(0.0);
            workspace.pending.clear();
            for &root in &self.output_slots {
                workspace.pending.push(root);
                while let Some(slot) = workspace.pending.pop() {
                    let i = checked_index(slot, self.opcodes.len())?;
                    if workspace.visited[i] {
                        continue;
                    }
                    let opcode = self.opcodes[i];
                    if let CompiledOpcode::Select {
                        condition,
                        then_value,
                        else_value,
                        ..
                    } = opcode
                    {
                        let ci = checked_index(condition, self.opcodes.len())?;
                        if !workspace.visited[ci] {
                            workspace.pending.push(slot);
                            workspace.pending.push(condition);
                            continue;
                        }
                        let selected = if workspace.values[ci] != 0.0 {
                            then_value
                        } else {
                            else_value
                        };
                        if !workspace.visited[checked_index(selected, self.opcodes.len())?] {
                            workspace.pending.push(slot);
                            workspace.pending.push(selected);
                            continue;
                        }
                    } else {
                        let (operands, count) = opcode.operands();
                        let missing = operands[..count]
                            .iter()
                            .any(|v| !workspace.visited[*v as usize]);
                        if missing {
                            workspace.pending.push(slot);
                            for &v in operands[..count].iter().rev() {
                                if !workspace.visited[v as usize] {
                                    workspace.pending.push(v);
                                }
                            }
                            continue;
                        }
                    }
                    let (_, value) = execute_opcode(
                        opcode,
                        &workspace.values,
                        &mut observation,
                        &mut pre_dividend_observation,
                    )?;
                    check_value(opcode, value)?;
                    workspace.values[i] = value;
                    workspace.visited[i] = true;
                }
            }
        } else {
            for &opcode in &self.opcodes {
                let (output, value) = execute_opcode(
                    opcode,
                    &workspace.values,
                    &mut observation,
                    &mut pre_dividend_observation,
                )?;
                check_value(opcode, value)?;
                workspace.values[output as usize] = value;
            }
        }
        for (value, &slot) in workspace.outputs.iter_mut().zip(&self.output_slots) {
            *value = workspace.values[slot as usize];
        }
        Ok(&workspace.outputs)
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
        observation: F,
        pre_dividend_observation: G,
    ) -> Result<PayoffEvaluation, GraphError>
    where
        F: FnMut(UnderlyingId, Date) -> Option<f64>,
        G: FnMut(UnderlyingId, Date) -> Option<f64>,
    {
        self.evaluate_output_with_observation_adjoints_and_values(
            output_index,
            observation,
            pre_dividend_observation,
        )
        .map(|(evaluation, _)| evaluation)
    }

    pub(in crate::engine) fn evaluate_output_with_observation_adjoints_and_values<F, G>(
        &self,
        output_index: usize,
        mut observation: F,
        mut pre_dividend_observation: G,
    ) -> Result<(PayoffEvaluation, Vec<f64>), GraphError>
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
        let (values, visited) = if self.branching {
            let mut workspace = self.workspace();
            self.evaluate_reusing(
                &mut workspace,
                &mut observation,
                &mut pre_dividend_observation,
            )?;
            (workspace.values, workspace.visited)
        } else {
            let mut values = vec![0.0; self.opcodes.len()];
            for &opcode in &self.opcodes {
                let (output, value) = execute_opcode(
                    opcode,
                    &values,
                    &mut observation,
                    &mut pre_dividend_observation,
                )?;
                check_value(opcode, value)?;
                values[output as usize] = value;
            }
            (values, Vec::new())
        };
        let output = checked_index(output_slot, values.len())?;
        let value = values[output];
        let mut adjoints = vec![0.0; self.opcodes.len()];
        adjoints[output] = 1.0;
        let mut terminal_adjoints = Vec::new();
        let mut pre_dividend_adjoints = Vec::new();
        for opcode in self.opcodes.iter().rev().copied() {
            if self.branching && !visited[opcode.output() as usize] {
                continue;
            }
            reverse_opcode(
                opcode,
                &values,
                &mut adjoints,
                &mut terminal_adjoints,
                &mut pre_dividend_adjoints,
            )?;
        }
        Ok((
            PayoffEvaluation {
                value,
                terminal_adjoints: terminal_adjoints.into_boxed_slice(),
                pre_dividend_adjoints: pre_dividend_adjoints.into_boxed_slice(),
            },
            values,
        ))
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
        CompiledOpcode::Select {
            condition,
            then_value,
            else_value,
            ..
        } => {
            let selected = if values[condition as usize] != 0.0 {
                then_value
            } else {
                else_value
            };
            add_adjoint(adjoints, selected, output_adjoint, opcode.name())?;
        }
        CompiledOpcode::Compare { .. } => {}
        CompiledOpcode::Log { input, .. } => add_adjoint(
            adjoints,
            input,
            output_adjoint / values[input as usize],
            opcode.name(),
        )?,
        CompiledOpcode::Exp { input, .. } => add_adjoint(
            adjoints,
            input,
            output_adjoint * values[output],
            opcode.name(),
        )?,
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
                division_denominator_adjoint(output_adjoint, numerator_value, denominator_value),
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

/// Preserve the ordinary pullback's arithmetic, but do not lose a finite
/// derivative when its numerator product or denominator square leaves the
/// normal range. Forward evaluation has already checked finite inputs and a
/// nonzero denominator; incoming adjoints are also checked by `add_adjoint`.
fn division_denominator_adjoint(seed: f64, numerator: f64, denominator: f64) -> f64 {
    let product = -seed * numerator;
    let square = denominator * denominator;
    if square.is_normal() && product.is_normal() {
        return product / square;
    }
    if seed == 0.0 || numerator == 0.0 {
        return product;
    }
    let (seed_fraction, seed_exponent) = binary_parts(seed);
    let (numerator_fraction, numerator_exponent) = binary_parts(numerator);
    let (denominator_fraction, denominator_exponent) = binary_parts(denominator);
    // Every fraction has magnitude in [1, 2), so neither product can leave
    // the normal range. The final scaling is the only possible underflow or
    // overflow; a genuinely nonfinite adjoint remains an evaluation error.
    let fraction =
        -seed_fraction * numerator_fraction / (denominator_fraction * denominator_fraction);
    let exponent = seed_exponent + numerator_exponent - 2 * denominator_exponent;
    scale_binary(fraction, exponent)
}

/// Exact decomposition of a finite, nonzero binary64, including subnormals.
fn binary_parts(value: f64) -> (f64, i32) {
    let (value, adjustment) = if value.is_subnormal() {
        (value * 18_014_398_509_481_984.0, -54) // 2^54, exact normalization
    } else {
        (value, 0)
    };
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 + adjustment;
    let fraction = f64::from_bits((bits & 0x800f_ffff_ffff_ffff) | (1023_u64 << 52));
    (fraction, exponent)
}

/// `fraction` has magnitude in (1/4, 4). Split only at binary64's exponent
/// boundaries so subnormal results are rounded once, at the final multiply.
fn scale_binary(fraction: f64, exponent: i32) -> f64 {
    let power = |exponent: i32| f64::from_bits(((exponent + 1023) as u64) << 52);
    if exponent > 1025 {
        f64::INFINITY.copysign(fraction)
    } else if exponent < -1076 {
        0.0_f64.copysign(fraction)
    } else if exponent > 1023 {
        (fraction * power(exponent - 1023)) * power(1023)
    } else if exponent < -1022 {
        (fraction * power(exponent + 1022)) * f64::MIN_POSITIVE
    } else {
        fraction * power(exponent)
    }
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

fn check_value(opcode: CompiledOpcode, value: f64) -> Result<(), GraphError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(GraphError::NonFiniteRuntimeValue {
            opcode: opcode.name(),
            bits: value.to_bits(),
        })
    }
}
impl CompiledOpcode {
    fn operands(self) -> ([u32; 3], usize) {
        match self {
            Self::Literal { .. } | Self::TerminalSpot { .. } | Self::PreDividendSpot { .. } => {
                ([0; 3], 0)
            }
            Self::Select {
                condition,
                then_value,
                else_value,
                ..
            } => ([condition, then_value, else_value], 3),
            Self::Log { input, .. }
            | Self::Exp { input, .. }
            | Self::Negate { input, .. }
            | Self::Indicator { input, .. }
            | Self::SmoothIndicator { input, .. } => ([input, 0, 0], 1),
            Self::Add { left, right, .. }
            | Self::Subtract { left, right, .. }
            | Self::Multiply { left, right, .. }
            | Self::Minimum { left, right, .. }
            | Self::Maximum { left, right, .. }
            | Self::SmoothMinimum { left, right, .. }
            | Self::SmoothMaximum { left, right, .. }
            | Self::Compare { left, right, .. } => ([left, right, 0], 2),
            Self::Divide {
                numerator,
                denominator,
                ..
            } => ([numerator, denominator, 0], 2),
            Self::BarrierHitState {
                previous,
                hit_weight,
                ..
            } => ([previous, hit_weight, 0], 2),
        }
    }

    pub(in crate::engine) const fn name(self) -> &'static str {
        match self {
            Self::Select { .. } => "select",
            Self::Log { .. } => "log",
            Self::Exp { .. } => "exp",
            Self::Compare { .. } => "compare",
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
            Self::Select { output, .. }
            | Self::Log { output, .. }
            | Self::Exp { output, .. }
            | Self::Compare { output, .. }
            | Self::Literal { output, .. }
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
        CompiledOpcode::Select {
            condition,
            then_value,
            else_value,
            output,
        } => {
            let selected = if slots[checked_index(condition, slots.len())?] != 0.0 {
                then_value
            } else {
                else_value
            };
            Ok((output, slots[checked_index(selected, slots.len())?]))
        }
        CompiledOpcode::Compare {
            left,
            right,
            operation,
            output,
        } => {
            binary(left, right).map(|(a, b)| (output, if operation.test(a, b) { 1.0 } else { 0.0 }))
        }
        CompiledOpcode::Log { input, output } => {
            Ok((output, slots[checked_index(input, slots.len())?].ln()))
        }
        CompiledOpcode::Exp { input, output } => {
            Ok((output, slots[checked_index(input, slots.len())?].exp()))
        }
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

#[cfg(test)]
mod division_adjoint_tests {
    use super::division_denominator_adjoint;

    fn assert_relative(actual: f64, expected: f64) {
        assert!(
            ((actual - expected) / expected).abs() <= 4.0 * f64::EPSILON,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn normal_range_keeps_the_ordinary_pullback() {
        for (seed, numerator, denominator) in [(1.5, 3.0, -0.75), (-2.0, 1e-3, 7.0)] {
            assert_eq!(
                division_denominator_adjoint(seed, numerator, denominator),
                -seed * numerator / (denominator * denominator)
            );
        }
    }

    #[test]
    fn finite_derivatives_survive_an_underflowing_or_overflowing_square() {
        // d(n/d)/dd = -n/d^2 is finite although d^2 leaves binary64.
        assert_relative(division_denominator_adjoint(1.0, 1e-200, 1e-200), -1e200);
        assert_relative(division_denominator_adjoint(2.0, 1e300, 1e200), -2e-100);
        // A subnormal numerator literal is not 3e-310 to full precision, so
        // the reference rescales exactly by powers of two: n * 2^200 / (d * 2^100)^2.
        let (numerator, denominator) = (3e-310, 1e-160);
        let scale = |exponent: i32| 2.0_f64.powi(exponent);
        let expected = (numerator * scale(200)) / (denominator * scale(100)).powi(2);
        assert_relative(
            division_denominator_adjoint(-1.0, numerator, denominator),
            expected,
        );
    }

    #[test]
    fn genuine_range_limits_round_once() {
        assert_eq!(
            division_denominator_adjoint(1.0, 1e300, 1e-10),
            f64::NEG_INFINITY
        );
        assert_eq!(division_denominator_adjoint(0.0, 1e300, 1e-200), -0.0);
        let subnormal = division_denominator_adjoint(1.0, 1e-300, 1e10);
        assert!(subnormal.is_subnormal() && subnormal < 0.0, "{subnormal}");
        assert!((subnormal + 1e-320).abs() <= 4.0 * f64::from_bits(1));
    }
}
