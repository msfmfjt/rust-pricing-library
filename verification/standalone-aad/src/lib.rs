//! Parity adapter for replaying a compiled payoff on the standalone scalar AAD engine.
//! Supply one finite value per observation and zip the resulting VJP with the
//! same observation list. Production pricing continues to use its native tape.
#![forbid(unsafe_code)]

use pricing::core::{Date, UnderlyingId};
use pricing::product::{CompiledOpcode as Op, CompiledPayoff, GraphComparison};
use rust_aad_core::{AadError, CompiledGraph, GraphBuilder, NodeId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Observation {
    Terminal(UnderlyingId, Date),
    PreDividend(UnderlyingId, Date),
}

pub struct AdaptedPayoff {
    pub observations: Vec<Observation>,
    pub graph: CompiledGraph,
}

#[derive(Debug)]
pub enum CompileError {
    Unsupported(Op),
    Aad(AadError),
}

impl From<AadError> for CompileError {
    fn from(error: AadError) -> Self {
        Self::Aad(error)
    }
}

impl AdaptedPayoff {
    /// Translate the native SSA slots once, sharing one input for all reads of
    /// the same underlying, date and observation timing.
    pub fn compile(payoff: &CompiledPayoff) -> Result<Self, CompileError> {
        let mut b = GraphBuilder::new();
        let mut observations = Vec::new();
        let mut inputs = BTreeMap::new();
        let mut slots = Vec::<NodeId>::with_capacity(payoff.opcodes().len());
        for &op in payoff.opcodes() {
            let slot = |i: u32| slots[i as usize];
            let node = match op {
                Op::TerminalSpot {
                    underlying,
                    observation_date,
                    ..
                }
                | Op::PreDividendSpot {
                    underlying,
                    observation_date,
                    ..
                } => {
                    let key = if matches!(op, Op::TerminalSpot { .. }) {
                        Observation::Terminal(underlying, observation_date)
                    } else {
                        Observation::PreDividend(underlying, observation_date)
                    };
                    *inputs.entry(key).or_insert_with(|| {
                        observations.push(key);
                        b.input()
                    })
                }
                Op::Literal { value, .. } => b.constant(value)?,
                Op::Add { left, right, .. } => b.add(slot(left), slot(right))?,
                Op::Subtract { left, right, .. } => b.sub(slot(left), slot(right))?,
                Op::Multiply { left, right, .. } => b.mul(slot(left), slot(right))?,
                Op::Divide {
                    numerator,
                    denominator,
                    ..
                } => b.div(slot(numerator), slot(denominator))?,
                Op::Negate { input, .. } => b.neg(slot(input))?,
                Op::Log { input, .. } => b.ln(slot(input))?,
                Op::Exp { input, .. } => b.exp(slot(input))?,
                Op::Minimum { left, right, .. } | Op::Maximum { left, right, .. } => {
                    // The native tape returns the left operand on a tie, including signed zero.
                    let (left, right) = (slot(left), slot(right));
                    let use_right = if matches!(op, Op::Minimum { .. }) {
                        b.greater(left, right)?
                    } else {
                        b.greater(right, left)?
                    };
                    b.select(use_right, right, left)?
                }
                Op::Compare {
                    left,
                    right,
                    operation: GraphComparison::Greater,
                    ..
                } => b.greater(slot(left), slot(right))?,
                Op::Compare {
                    left,
                    right,
                    operation: GraphComparison::Less,
                    ..
                } => b.greater(slot(right), slot(left))?,
                Op::Select {
                    condition,
                    then_value,
                    else_value,
                    ..
                } => b.select(slot(condition), slot(then_value), slot(else_value))?,
                other => return Err(CompileError::Unsupported(other)),
            };
            slots.push(node);
        }
        let outputs: Vec<_> = payoff
            .output_slots()
            .iter()
            .map(|&s| slots[s as usize])
            .collect();
        Ok(Self {
            observations,
            graph: b.compile(&outputs)?,
        })
    }
}
