use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::mem;

use pricing_core::{Date, FiniteF64, NodeId, UnderlyingId};

use crate::{
    ArithmeticAsianSpec, AsianObservationValue, BarrierDirection, BarrierSpec, BarrierStyle,
    CompactC2Smoothing, DigitalPayout, DigitalSpec, EuropeanVanillaSpec, FixedLookbackSpec,
    OptionSide, ProductSpec,
};

const SOURCE_GRAPH_VERSION: u32 = 1;
const TAPE_ABI_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphLimitPolicy {
    pub source_nodes: usize,
    pub compiled_opcodes: usize,
    pub dependency_edges: usize,
    pub outputs: usize,
    pub events: usize,
    pub value_slots: usize,
    pub state_slots: usize,
    pub reverse_cache_slots: usize,
    pub estimated_bytes: usize,
}

impl GraphLimitPolicy {
    pub const VERSION: u32 = 1;
    pub const DEFAULT: Self = Self {
        source_nodes: 1_000_000,
        compiled_opcodes: 1_000_000,
        dependency_edges: 4_000_000,
        outputs: 64,
        events: 100_000,
        value_slots: 1_000_000,
        state_slots: 1_000_000,
        reverse_cache_slots: 4_000_000,
        estimated_bytes: 1024 * 1024 * 1024,
    };
}

impl Default for GraphLimitPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GraphFingerprint([u8; 32]);

impl GraphFingerprint {
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for GraphFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "blake3-256:")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceOpcode {
    Literal(FiniteF64),
    TerminalSpot {
        underlying: UnderlyingId,
        observation_date: Date,
    },
    PreDividendSpot {
        underlying: UnderlyingId,
        observation_date: Date,
    },
    Add {
        left: NodeId,
        right: NodeId,
    },
    Subtract {
        left: NodeId,
        right: NodeId,
    },
    Multiply {
        left: NodeId,
        right: NodeId,
    },
    Divide {
        numerator: NodeId,
        denominator: NodeId,
    },
    Minimum {
        left: NodeId,
        right: NodeId,
    },
    Maximum {
        left: NodeId,
        right: NodeId,
    },
    Indicator {
        input: NodeId,
    },
    SmoothMinimum {
        left: NodeId,
        right: NodeId,
        smoothing: CompactC2Smoothing,
    },
    SmoothMaximum {
        left: NodeId,
        right: NodeId,
        smoothing: CompactC2Smoothing,
    },
    SmoothIndicator {
        input: NodeId,
        smoothing: CompactC2Smoothing,
    },
    BarrierHitState {
        previous: NodeId,
        hit_weight: NodeId,
    },
    Negate {
        input: NodeId,
    },
}

impl SourceOpcode {
    fn operands(self) -> ([NodeId; 2], usize) {
        match self {
            Self::Literal(_) | Self::TerminalSpot { .. } | Self::PreDividendSpot { .. } => {
                ([NodeId::new(0); 2], 0)
            }
            Self::Negate { input }
            | Self::Indicator { input }
            | Self::SmoothIndicator { input, .. } => ([input, NodeId::new(0)], 1),
            Self::Add { left, right }
            | Self::Subtract { left, right }
            | Self::Multiply { left, right }
            | Self::Minimum { left, right }
            | Self::Maximum { left, right }
            | Self::SmoothMinimum { left, right, .. }
            | Self::SmoothMaximum { left, right, .. } => ([left, right], 2),
            Self::BarrierHitState {
                previous,
                hit_weight,
            } => ([previous, hit_weight], 2),
            Self::Divide {
                numerator,
                denominator,
            } => ([numerator, denominator], 2),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Literal(_) => "literal",
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceNode {
    id: NodeId,
    opcode: SourceOpcode,
}

impl SourceNode {
    #[must_use]
    pub const fn new(id: NodeId, opcode: SourceOpcode) -> Self {
        Self { id, opcode }
    }

    #[must_use]
    pub const fn id(self) -> NodeId {
        self.id
    }

    #[must_use]
    pub const fn opcode(self) -> SourceOpcode {
        self.opcode
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceGraph {
    nodes: Box<[SourceNode]>,
    outputs: Box<[NodeId]>,
}

impl SourceGraph {
    #[must_use]
    pub fn new(nodes: Vec<SourceNode>, outputs: Vec<NodeId>) -> Self {
        Self {
            nodes: nodes.into_boxed_slice(),
            outputs: outputs.into_boxed_slice(),
        }
    }

    #[must_use]
    pub fn nodes(&self) -> &[SourceNode] {
        &self.nodes
    }

    #[must_use]
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    pub fn compile(&self, limits: GraphLimitPolicy) -> Result<CompiledPayoff, GraphError> {
        compile(self, limits)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceGraphBuilder {
    nodes: Vec<SourceNode>,
    next_id: Option<u32>,
}

impl Default for SourceGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceGraphBuilder {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_id: Some(0),
        }
    }

    #[must_use]
    pub fn from_graph(graph: &SourceGraph) -> Self {
        let greatest = graph.nodes.iter().map(|node| node.id.get()).max();
        let next_id = greatest.map_or(Some(0), |value| value.checked_add(1));
        Self {
            nodes: graph.nodes.to_vec(),
            next_id,
        }
    }

    pub fn push(&mut self, opcode: SourceOpcode) -> Result<NodeId, GraphError> {
        let raw = self.next_id.ok_or(GraphError::NodeIdExhausted)?;
        let id = NodeId::new(raw);
        self.next_id = raw.checked_add(1);
        self.nodes.push(SourceNode::new(id, opcode));
        Ok(id)
    }

    pub fn literal(&mut self, value: f64) -> Result<NodeId, GraphError> {
        let value =
            FiniteF64::new(value, "payoff_literal").map_err(|_| GraphError::NonFiniteLiteral {
                bits: value.to_bits(),
            })?;
        self.push(SourceOpcode::Literal(value))
    }

    pub fn finish(self, outputs: Vec<NodeId>) -> SourceGraph {
        SourceGraph::new(self.nodes, outputs)
    }
}

impl EuropeanVanillaSpec {
    pub fn source_graph(&self) -> Result<SourceGraph, GraphError> {
        let mut builder = SourceGraphBuilder::new();
        let spot = builder.push(SourceOpcode::TerminalSpot {
            underlying: self.underlying(),
            observation_date: self.expiry(),
        })?;
        let strike = builder.literal(self.strike().get())?;
        let signed_intrinsic = match self.side() {
            OptionSide::Call => builder.push(SourceOpcode::Subtract {
                left: spot,
                right: strike,
            })?,
            OptionSide::Put => builder.push(SourceOpcode::Subtract {
                left: strike,
                right: spot,
            })?,
        };
        let zero = builder.literal(0.0)?;
        let positive_part = builder.push(SourceOpcode::Maximum {
            left: signed_intrinsic,
            right: zero,
        })?;
        let notional = builder.literal(self.notional().get())?;
        let payoff = builder.push(SourceOpcode::Multiply {
            left: positive_part,
            right: notional,
        })?;
        Ok(builder.finish(vec![payoff]))
    }
}

impl DigitalSpec {
    pub fn source_graph(&self) -> Result<SourceGraph, GraphError> {
        self.build_source_graph(None)
    }

    pub fn smoothed_source_graph(
        &self,
        smoothing: CompactC2Smoothing,
    ) -> Result<SourceGraph, GraphError> {
        self.build_source_graph(Some(smoothing))
    }

    fn build_source_graph(
        &self,
        smoothing: Option<CompactC2Smoothing>,
    ) -> Result<SourceGraph, GraphError> {
        let mut builder = SourceGraphBuilder::new();
        let spot = builder.push(SourceOpcode::TerminalSpot {
            underlying: self.underlying(),
            observation_date: self.expiry(),
        })?;
        let strike = builder.literal(self.strike().get())?;
        let signed_distance = match self.side() {
            OptionSide::Call => builder.push(SourceOpcode::Subtract {
                left: spot,
                right: strike,
            })?,
            OptionSide::Put => builder.push(SourceOpcode::Subtract {
                left: strike,
                right: spot,
            })?,
        };
        let indicator = if let Some(smoothing) = smoothing {
            builder.push(SourceOpcode::SmoothIndicator {
                input: signed_distance,
                smoothing,
            })?
        } else {
            builder.push(SourceOpcode::Indicator {
                input: signed_distance,
            })?
        };
        let payout = builder.literal(self.payout().get())?;
        let payoff_base = match self.payout_kind() {
            DigitalPayout::Cash => payout,
            DigitalPayout::Asset => builder.push(SourceOpcode::Multiply {
                left: spot,
                right: payout,
            })?,
        };
        let payoff = builder.push(SourceOpcode::Multiply {
            left: indicator,
            right: payoff_base,
        })?;
        Ok(builder.finish(vec![payoff]))
    }
}

impl ArithmeticAsianSpec {
    pub fn source_graph(&self) -> Result<SourceGraph, GraphError> {
        let mut builder = SourceGraphBuilder::new();
        let mut weighted_sum = builder.literal(0.0)?;
        for observation in self.observations() {
            let observed = match observation.value() {
                AsianObservationValue::Known(fixing) => builder.literal(fixing.get())?,
                AsianObservationValue::Unknown => builder.push(SourceOpcode::TerminalSpot {
                    underlying: self.underlying(),
                    observation_date: observation.date(),
                })?,
            };
            let weight = builder.literal(observation.weight().get())?;
            let weighted = builder.push(SourceOpcode::Multiply {
                left: observed,
                right: weight,
            })?;
            weighted_sum = builder.push(SourceOpcode::Add {
                left: weighted_sum,
                right: weighted,
            })?;
        }
        let strike = builder.literal(self.strike().get())?;
        let signed_intrinsic = match self.side() {
            OptionSide::Call => builder.push(SourceOpcode::Subtract {
                left: weighted_sum,
                right: strike,
            })?,
            OptionSide::Put => builder.push(SourceOpcode::Subtract {
                left: strike,
                right: weighted_sum,
            })?,
        };
        let zero = builder.literal(0.0)?;
        let positive_part = builder.push(SourceOpcode::Maximum {
            left: signed_intrinsic,
            right: zero,
        })?;
        let notional = builder.literal(self.notional().get())?;
        let payoff = builder.push(SourceOpcode::Multiply {
            left: positive_part,
            right: notional,
        })?;
        Ok(builder.finish(vec![payoff]))
    }
}

impl BarrierSpec {
    pub fn source_graph(&self) -> Result<SourceGraph, GraphError> {
        self.build_source_graph(None, &BTreeSet::new())
    }

    pub fn source_graph_with_dividend_jumps(
        &self,
        jump_dates: &[Date],
    ) -> Result<SourceGraph, GraphError> {
        self.build_source_graph(None, &jump_dates.iter().copied().collect())
    }

    pub fn smoothed_source_graph(
        &self,
        smoothing: CompactC2Smoothing,
    ) -> Result<SourceGraph, GraphError> {
        self.build_source_graph(Some(smoothing), &BTreeSet::new())
    }

    pub fn smoothed_source_graph_with_dividend_jumps(
        &self,
        smoothing: CompactC2Smoothing,
        jump_dates: &[Date],
    ) -> Result<SourceGraph, GraphError> {
        self.build_source_graph(Some(smoothing), &jump_dates.iter().copied().collect())
    }

    fn build_source_graph(
        &self,
        smoothing: Option<CompactC2Smoothing>,
        jump_dates: &BTreeSet<Date>,
    ) -> Result<SourceGraph, GraphError> {
        let mut builder = SourceGraphBuilder::new();
        let strike = builder.literal(self.strike().get())?;
        let terminal = builder.push(SourceOpcode::TerminalSpot {
            underlying: self.underlying(),
            observation_date: self.expiry(),
        })?;
        let signed_intrinsic = match self.side() {
            OptionSide::Call => builder.push(SourceOpcode::Subtract {
                left: terminal,
                right: strike,
            })?,
            OptionSide::Put => builder.push(SourceOpcode::Subtract {
                left: strike,
                right: terminal,
            })?,
        };
        let zero = builder.literal(0.0)?;
        let positive_part = builder.push(SourceOpcode::Maximum {
            left: signed_intrinsic,
            right: zero,
        })?;
        let notional = builder.literal(self.notional().get())?;
        let vanilla_payoff = builder.push(SourceOpcode::Multiply {
            left: positive_part,
            right: notional,
        })?;

        let barrier = builder.literal(self.barrier().get())?;
        let mut hit = zero;
        for date in self.monitoring_dates() {
            let spot = builder.push(SourceOpcode::TerminalSpot {
                underlying: self.underlying(),
                observation_date: *date,
            })?;
            let signed_distance = match self.direction() {
                BarrierDirection::Up => builder.push(SourceOpcode::Subtract {
                    left: spot,
                    right: barrier,
                })?,
                BarrierDirection::Down => builder.push(SourceOpcode::Subtract {
                    left: barrier,
                    right: spot,
                })?,
            };
            let signed_distance = if jump_dates.contains(date) {
                let pre_jump_spot = builder.push(SourceOpcode::PreDividendSpot {
                    underlying: self.underlying(),
                    observation_date: *date,
                })?;
                let pre_jump_distance = match self.direction() {
                    BarrierDirection::Up => builder.push(SourceOpcode::Subtract {
                        left: pre_jump_spot,
                        right: barrier,
                    })?,
                    BarrierDirection::Down => builder.push(SourceOpcode::Subtract {
                        left: barrier,
                        right: pre_jump_spot,
                    })?,
                };
                if let Some(smoothing) = smoothing {
                    builder.push(SourceOpcode::SmoothMaximum {
                        left: pre_jump_distance,
                        right: signed_distance,
                        smoothing,
                    })?
                } else {
                    builder.push(SourceOpcode::Maximum {
                        left: pre_jump_distance,
                        right: signed_distance,
                    })?
                }
            } else {
                signed_distance
            };
            let date_hit = if let Some(smoothing) = smoothing {
                builder.push(SourceOpcode::SmoothIndicator {
                    input: signed_distance,
                    smoothing,
                })?
            } else {
                builder.push(SourceOpcode::Indicator {
                    input: signed_distance,
                })?
            };
            hit = builder.push(SourceOpcode::BarrierHitState {
                previous: hit,
                hit_weight: date_hit,
            })?;
        }

        let active = match self.style() {
            BarrierStyle::KnockIn => hit,
            BarrierStyle::KnockOut => {
                let one = builder.literal(1.0)?;
                builder.push(SourceOpcode::Subtract {
                    left: one,
                    right: hit,
                })?
            }
        };
        let active_payoff = builder.push(SourceOpcode::Multiply {
            left: vanilla_payoff,
            right: active,
        })?;
        let payoff = if let Some(rebate) = self.rebate() {
            let one = builder.literal(1.0)?;
            let inactive = builder.push(SourceOpcode::Subtract {
                left: one,
                right: active,
            })?;
            let rebate = builder.literal(rebate.get())?;
            let inactive_payoff = builder.push(SourceOpcode::Multiply {
                left: inactive,
                right: rebate,
            })?;
            builder.push(SourceOpcode::Add {
                left: active_payoff,
                right: inactive_payoff,
            })?
        } else {
            active_payoff
        };
        Ok(builder.finish(vec![payoff]))
    }
}

impl FixedLookbackSpec {
    pub fn source_graph(&self, valuation_date: Date) -> Result<SourceGraph, GraphError> {
        let mut builder = SourceGraphBuilder::new();
        let mut dates = self
            .monitoring_dates()
            .iter()
            .copied()
            .filter(|date| *date >= valuation_date);
        let first = if let Some(extremum) = self.historical_extremum() {
            builder.literal(extremum.get())?
        } else {
            let date = dates.next().ok_or(GraphError::NoOutputs)?;
            builder.push(SourceOpcode::TerminalSpot {
                underlying: self.underlying(),
                observation_date: date,
            })?
        };
        let mut extremum = first;
        for date in dates {
            let spot = builder.push(SourceOpcode::TerminalSpot {
                underlying: self.underlying(),
                observation_date: date,
            })?;
            extremum = match self.side() {
                OptionSide::Call => builder.push(SourceOpcode::Maximum {
                    left: extremum,
                    right: spot,
                })?,
                OptionSide::Put => builder.push(SourceOpcode::Minimum {
                    left: extremum,
                    right: spot,
                })?,
            };
        }
        let strike = builder.literal(self.strike().get())?;
        let signed_intrinsic = match self.side() {
            OptionSide::Call => builder.push(SourceOpcode::Subtract {
                left: extremum,
                right: strike,
            })?,
            OptionSide::Put => builder.push(SourceOpcode::Subtract {
                left: strike,
                right: extremum,
            })?,
        };
        let zero = builder.literal(0.0)?;
        let positive_part = builder.push(SourceOpcode::Maximum {
            left: signed_intrinsic,
            right: zero,
        })?;
        let notional = builder.literal(self.notional().get())?;
        let payoff = builder.push(SourceOpcode::Multiply {
            left: positive_part,
            right: notional,
        })?;
        Ok(builder.finish(vec![payoff]))
    }
}

impl ProductSpec {
    pub fn source_graph(&self, valuation_date: Date) -> Result<SourceGraph, GraphError> {
        match self {
            Self::EuropeanVanilla(spec) => spec.source_graph(),
            Self::Digital(spec) => spec.source_graph(),
            Self::Barrier(spec) => spec.source_graph(),
            Self::ArithmeticAsian(spec) => spec.source_graph(),
            Self::FixedLookback(spec) => spec.source_graph(valuation_date),
        }
    }
}

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
    opcodes: Box<[CompiledOpcode]>,
    output_slots: Box<[u32]>,
    source_to_slot: Box<[(NodeId, u32)]>,
    removed_source_nodes: Box<[NodeId]>,
    source_fingerprint: GraphFingerprint,
    tape_fingerprint: GraphFingerprint,
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
        let output = checked_index(self.output_slots[0], values.len())?;
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

fn reverse_opcode(
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

fn add_adjoint(
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
    const fn name(self) -> &'static str {
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

    const fn output(self) -> u32 {
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

fn execute_opcode<F>(
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

fn compile(graph: &SourceGraph, limits: GraphLimitPolicy) -> Result<CompiledPayoff, GraphError> {
    enforce_limit("source_nodes", graph.nodes.len(), limits.source_nodes)?;
    enforce_limit("outputs", graph.outputs.len(), limits.outputs)?;
    if graph.outputs.is_empty() {
        return Err(GraphError::NoOutputs);
    }

    let mut nodes = BTreeMap::new();
    for node in graph.nodes.iter().copied() {
        if nodes.insert(node.id, node.opcode).is_some() {
            return Err(GraphError::DuplicateNodeId { node: node.id });
        }
    }
    let edge_count = nodes.values().map(|opcode| opcode.operands().1).sum();
    enforce_limit("dependency_edges", edge_count, limits.dependency_edges)?;
    let estimated_bytes = estimate_bytes(graph.nodes.len(), edge_count, graph.outputs.len())?;
    enforce_limit("estimated_bytes", estimated_bytes, limits.estimated_bytes)?;

    for output in graph.outputs.iter().copied() {
        if !nodes.contains_key(&output) {
            return Err(GraphError::UnknownOutput { node: output });
        }
    }

    let mut indegree = BTreeMap::new();
    let mut outgoing: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
    for (&id, &opcode) in &nodes {
        let (operands, count) = opcode.operands();
        indegree.insert(id, count);
        for operand in operands.into_iter().take(count) {
            if !nodes.contains_key(&operand) {
                return Err(GraphError::UnknownOperand { node: id, operand });
            }
            outgoing.entry(operand).or_default().push(id);
        }
    }
    for destinations in outgoing.values_mut() {
        destinations.sort_unstable();
    }

    let mut ready: BTreeSet<NodeId> = indegree
        .iter()
        .filter_map(|(&id, &degree)| (degree == 0).then_some(id))
        .collect();
    let mut topological = Vec::with_capacity(nodes.len());
    let mut folded: BTreeMap<NodeId, Option<FiniteF64>> = BTreeMap::new();
    while let Some(id) = ready.pop_first() {
        let opcode = nodes[&id];
        topological.push(id);
        folded.insert(id, fold_node(id, opcode, &folded)?);
        if let Some(destinations) = outgoing.get(&id) {
            for destination in destinations {
                let degree = indegree
                    .get_mut(destination)
                    .ok_or(GraphError::InternalOrdering {
                        operand: *destination,
                    })?;
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(*destination);
                }
            }
        }
    }
    if topological.len() != nodes.len() {
        let node = indegree
            .iter()
            .find_map(|(&id, &degree)| (degree != 0).then_some(id))
            .ok_or(GraphError::InternalOrdering {
                operand: graph
                    .outputs
                    .first()
                    .copied()
                    .ok_or(GraphError::NoOutputs)?,
            })?;
        return Err(GraphError::Cycle { node });
    }

    let reachable = reachable_nodes(&nodes, &graph.outputs);
    let removed_source_nodes: Vec<_> = nodes
        .keys()
        .filter(|id| !reachable.contains(id))
        .copied()
        .collect();
    let compiled_count = reachable.len();
    enforce_limit("compiled_opcodes", compiled_count, limits.compiled_opcodes)?;
    enforce_limit("value_slots", compiled_count, limits.value_slots)?;
    let state_count = reachable
        .iter()
        .filter(|id| matches!(nodes[id], SourceOpcode::BarrierHitState { .. }))
        .count();
    enforce_limit("state_slots", state_count, limits.state_slots)?;
    let event_count = reachable
        .iter()
        .filter(|id| matches!(nodes[id], SourceOpcode::PreDividendSpot { .. }))
        .count();
    enforce_limit("events", event_count, limits.events)?;
    let _ = u32::try_from(compiled_count).map_err(|_| GraphError::HardCapacity {
        field: "value_slots",
        observed: compiled_count,
    })?;

    let mut source_to_slot = BTreeMap::new();
    let mut opcodes = Vec::with_capacity(compiled_count);
    for id in topological.into_iter().filter(|id| reachable.contains(id)) {
        let output = u32::try_from(opcodes.len()).map_err(|_| GraphError::HardCapacity {
            field: "compiled_opcodes",
            observed: opcodes.len(),
        })?;
        source_to_slot.insert(id, output);
        let opcode = if let Some(value) = folded[&id] {
            CompiledOpcode::Literal {
                value: value.get(),
                output,
            }
        } else {
            compile_opcode(nodes[&id], output, &source_to_slot)?
        };
        opcodes.push(opcode);
    }
    let output_slots = graph
        .outputs
        .iter()
        .map(|id| source_to_slot[id])
        .collect::<Vec<_>>();
    let mapping = source_to_slot.into_iter().collect::<Vec<_>>();
    let source_fingerprint = fingerprint_source(&nodes, &graph.outputs);
    let tape_fingerprint = fingerprint_tape(&opcodes, &output_slots, &mapping);
    Ok(CompiledPayoff {
        opcodes: opcodes.into_boxed_slice(),
        output_slots: output_slots.into_boxed_slice(),
        source_to_slot: mapping.into_boxed_slice(),
        removed_source_nodes: removed_source_nodes.into_boxed_slice(),
        source_fingerprint,
        tape_fingerprint,
    })
}

fn reachable_nodes(nodes: &BTreeMap<NodeId, SourceOpcode>, outputs: &[NodeId]) -> BTreeSet<NodeId> {
    let mut reachable = BTreeSet::new();
    let mut pending = outputs.to_vec();
    while let Some(id) = pending.pop() {
        if reachable.insert(id) {
            let (operands, count) = nodes[&id].operands();
            pending.extend(operands.into_iter().take(count));
        }
    }
    reachable
}

fn fold_node(
    id: NodeId,
    opcode: SourceOpcode,
    folded: &BTreeMap<NodeId, Option<FiniteF64>>,
) -> Result<Option<FiniteF64>, GraphError> {
    if let SourceOpcode::Literal(value) = opcode {
        return Ok(Some(value));
    }
    if matches!(
        opcode,
        SourceOpcode::TerminalSpot { .. } | SourceOpcode::PreDividendSpot { .. }
    ) {
        return Ok(None);
    }
    let (operands, count) = opcode.operands();
    let values: Option<Vec<f64>> = operands
        .into_iter()
        .take(count)
        .map(|operand| folded[&operand].map(FiniteF64::get))
        .collect();
    let Some(values) = values else {
        return Ok(None);
    };
    let result = match opcode {
        SourceOpcode::Add { .. } => values[0] + values[1],
        SourceOpcode::Subtract { .. } => values[0] - values[1],
        SourceOpcode::Multiply { .. } => values[0] * values[1],
        SourceOpcode::Divide { .. } => values[0] / values[1],
        SourceOpcode::Minimum { .. } => {
            if values[0] <= values[1] {
                values[0]
            } else {
                values[1]
            }
        }
        SourceOpcode::Maximum { .. } => {
            if values[0] >= values[1] {
                values[0]
            } else {
                values[1]
            }
        }
        SourceOpcode::Indicator { .. } => {
            if values[0] >= 0.0 {
                1.0
            } else {
                0.0
            }
        }
        SourceOpcode::SmoothMinimum { smoothing, .. } => {
            smoothing.minimum(values[0], values[1]).value
        }
        SourceOpcode::SmoothMaximum { smoothing, .. } => {
            smoothing.maximum(values[0], values[1]).value
        }
        SourceOpcode::SmoothIndicator { smoothing, .. } => smoothing.indicator(values[0]).value,
        SourceOpcode::BarrierHitState { .. } => values[0] + (1.0 - values[0]) * values[1],
        SourceOpcode::Negate { .. } => -values[0],
        SourceOpcode::Literal(_)
        | SourceOpcode::TerminalSpot { .. }
        | SourceOpcode::PreDividendSpot { .. } => unreachable!(),
    };
    FiniteF64::new(result, "constant_fold")
        .map(Some)
        .map_err(|_| GraphError::NonFiniteConstantFold {
            node: id,
            opcode: opcode.name(),
            operand_bits: values.iter().map(|value| value.to_bits()).collect(),
            result_bits: result.to_bits(),
        })
}

fn compile_opcode(
    opcode: SourceOpcode,
    output: u32,
    slots: &BTreeMap<NodeId, u32>,
) -> Result<CompiledOpcode, GraphError> {
    let slot = |id| {
        slots
            .get(&id)
            .copied()
            .ok_or(GraphError::InternalOrdering { operand: id })
    };
    match opcode {
        SourceOpcode::Literal(value) => Ok(CompiledOpcode::Literal {
            value: value.get(),
            output,
        }),
        SourceOpcode::TerminalSpot {
            underlying,
            observation_date,
        } => Ok(CompiledOpcode::TerminalSpot {
            underlying,
            observation_date,
            output,
        }),
        SourceOpcode::PreDividendSpot {
            underlying,
            observation_date,
        } => Ok(CompiledOpcode::PreDividendSpot {
            underlying,
            observation_date,
            output,
        }),
        SourceOpcode::Add { left, right } => Ok(CompiledOpcode::Add {
            left: slot(left)?,
            right: slot(right)?,
            output,
        }),
        SourceOpcode::Subtract { left, right } => Ok(CompiledOpcode::Subtract {
            left: slot(left)?,
            right: slot(right)?,
            output,
        }),
        SourceOpcode::Multiply { left, right } => Ok(CompiledOpcode::Multiply {
            left: slot(left)?,
            right: slot(right)?,
            output,
        }),
        SourceOpcode::Divide {
            numerator,
            denominator,
        } => Ok(CompiledOpcode::Divide {
            numerator: slot(numerator)?,
            denominator: slot(denominator)?,
            output,
        }),
        SourceOpcode::Minimum { left, right } => Ok(CompiledOpcode::Minimum {
            left: slot(left)?,
            right: slot(right)?,
            output,
        }),
        SourceOpcode::Maximum { left, right } => Ok(CompiledOpcode::Maximum {
            left: slot(left)?,
            right: slot(right)?,
            output,
        }),
        SourceOpcode::Indicator { input } => Ok(CompiledOpcode::Indicator {
            input: slot(input)?,
            output,
        }),
        SourceOpcode::SmoothMinimum {
            left,
            right,
            smoothing,
        } => Ok(CompiledOpcode::SmoothMinimum {
            left: slot(left)?,
            right: slot(right)?,
            smoothing,
            output,
        }),
        SourceOpcode::SmoothMaximum {
            left,
            right,
            smoothing,
        } => Ok(CompiledOpcode::SmoothMaximum {
            left: slot(left)?,
            right: slot(right)?,
            smoothing,
            output,
        }),
        SourceOpcode::SmoothIndicator { input, smoothing } => Ok(CompiledOpcode::SmoothIndicator {
            input: slot(input)?,
            smoothing,
            output,
        }),
        SourceOpcode::BarrierHitState {
            previous,
            hit_weight,
        } => Ok(CompiledOpcode::BarrierHitState {
            previous: slot(previous)?,
            hit_weight: slot(hit_weight)?,
            output,
        }),
        SourceOpcode::Negate { input } => Ok(CompiledOpcode::Negate {
            input: slot(input)?,
            output,
        }),
    }
}

fn estimate_bytes(nodes: usize, edges: usize, outputs: usize) -> Result<usize, GraphError> {
    nodes
        .checked_mul(mem::size_of::<SourceNode>() + mem::size_of::<CompiledOpcode>() + 16)
        .and_then(|value| value.checked_add(edges.checked_mul(mem::size_of::<NodeId>())?))
        .and_then(|value| value.checked_add(outputs.checked_mul(mem::size_of::<u32>())?))
        .ok_or(GraphError::SizeOverflow)
}

fn enforce_limit(field: &'static str, observed: usize, limit: usize) -> Result<(), GraphError> {
    if observed > limit {
        Err(GraphError::SoftLimitExceeded {
            field,
            observed,
            limit,
        })
    } else {
        Ok(())
    }
}

fn checked_index(index: u32, length: usize) -> Result<usize, GraphError> {
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

fn fingerprint_source(
    nodes: &BTreeMap<NodeId, SourceOpcode>,
    outputs: &[NodeId],
) -> GraphFingerprint {
    let mut bytes = b"pricing/source_graph\0".to_vec();
    put_u32(&mut bytes, SOURCE_GRAPH_VERSION);
    put_u64(&mut bytes, nodes.len() as u64);
    for (&id, &opcode) in nodes {
        put_u32(&mut bytes, id.get());
        encode_source_opcode(&mut bytes, opcode);
    }
    put_u64(&mut bytes, outputs.len() as u64);
    for output in outputs {
        put_u32(&mut bytes, output.get());
    }
    GraphFingerprint(*blake3::hash(&bytes).as_bytes())
}

fn fingerprint_tape(
    opcodes: &[CompiledOpcode],
    outputs: &[u32],
    mapping: &[(NodeId, u32)],
) -> GraphFingerprint {
    let mut bytes = b"pricing/payoff_tape\0".to_vec();
    put_u32(&mut bytes, TAPE_ABI_VERSION);
    put_u64(&mut bytes, opcodes.len() as u64);
    for opcode in opcodes {
        encode_compiled_opcode(&mut bytes, *opcode);
    }
    put_u64(&mut bytes, outputs.len() as u64);
    for output in outputs {
        put_u32(&mut bytes, *output);
    }
    put_u64(&mut bytes, mapping.len() as u64);
    for (source, slot) in mapping {
        put_u32(&mut bytes, source.get());
        put_u32(&mut bytes, *slot);
    }
    GraphFingerprint(*blake3::hash(&bytes).as_bytes())
}

fn encode_source_opcode(bytes: &mut Vec<u8>, opcode: SourceOpcode) {
    bytes.push(opcode_tag(opcode));
    match opcode {
        SourceOpcode::Literal(value) => put_u64(bytes, value.to_bits()),
        SourceOpcode::TerminalSpot {
            underlying,
            observation_date,
        } => {
            put_u32(bytes, underlying.get());
            put_date(bytes, observation_date);
        }
        SourceOpcode::PreDividendSpot {
            underlying,
            observation_date,
        } => {
            put_u32(bytes, underlying.get());
            put_date(bytes, observation_date);
        }
        SourceOpcode::Negate { input } | SourceOpcode::Indicator { input } => {
            put_u32(bytes, input.get());
        }
        SourceOpcode::SmoothIndicator { input, smoothing } => {
            put_u32(bytes, input.get());
            put_u64(bytes, smoothing.half_width().get().to_bits());
        }
        SourceOpcode::BarrierHitState {
            previous,
            hit_weight,
        } => {
            put_u32(bytes, previous.get());
            put_u32(bytes, hit_weight.get());
        }
        SourceOpcode::SmoothMinimum {
            left,
            right,
            smoothing,
        }
        | SourceOpcode::SmoothMaximum {
            left,
            right,
            smoothing,
        } => {
            put_u32(bytes, left.get());
            put_u32(bytes, right.get());
            put_u64(bytes, smoothing.half_width().get().to_bits());
        }
        _ => {
            let (operands, count) = opcode.operands();
            for operand in operands.into_iter().take(count) {
                put_u32(bytes, operand.get());
            }
        }
    }
}

fn encode_compiled_opcode(bytes: &mut Vec<u8>, opcode: CompiledOpcode) {
    bytes.push(compiled_opcode_tag(opcode));
    match opcode {
        CompiledOpcode::Literal { value, output } => {
            put_u64(bytes, value.to_bits());
            put_u32(bytes, output);
        }
        CompiledOpcode::TerminalSpot {
            underlying,
            observation_date,
            output,
        } => {
            put_u32(bytes, underlying.get());
            put_date(bytes, observation_date);
            put_u32(bytes, output);
        }
        CompiledOpcode::PreDividendSpot {
            underlying,
            observation_date,
            output,
        } => {
            put_u32(bytes, underlying.get());
            put_date(bytes, observation_date);
            put_u32(bytes, output);
        }
        CompiledOpcode::Negate { input, output } | CompiledOpcode::Indicator { input, output } => {
            put_u32(bytes, input);
            put_u32(bytes, output);
        }
        CompiledOpcode::SmoothIndicator {
            input,
            smoothing,
            output,
        } => {
            put_u32(bytes, input);
            put_u64(bytes, smoothing.half_width().get().to_bits());
            put_u32(bytes, output);
        }
        CompiledOpcode::SmoothMinimum {
            left,
            right,
            smoothing,
            output,
        }
        | CompiledOpcode::SmoothMaximum {
            left,
            right,
            smoothing,
            output,
        } => {
            put_u32(bytes, left);
            put_u32(bytes, right);
            put_u64(bytes, smoothing.half_width().get().to_bits());
            put_u32(bytes, output);
        }
        CompiledOpcode::BarrierHitState {
            previous,
            hit_weight,
            output,
        } => {
            put_u32(bytes, previous);
            put_u32(bytes, hit_weight);
            put_u32(bytes, output);
        }
        CompiledOpcode::Divide {
            numerator,
            denominator,
            output,
        } => {
            put_u32(bytes, numerator);
            put_u32(bytes, denominator);
            put_u32(bytes, output);
        }
        CompiledOpcode::Add {
            left,
            right,
            output,
        }
        | CompiledOpcode::Subtract {
            left,
            right,
            output,
        }
        | CompiledOpcode::Multiply {
            left,
            right,
            output,
        }
        | CompiledOpcode::Minimum {
            left,
            right,
            output,
        }
        | CompiledOpcode::Maximum {
            left,
            right,
            output,
        } => {
            put_u32(bytes, left);
            put_u32(bytes, right);
            put_u32(bytes, output);
        }
    }
}

const fn opcode_tag(opcode: SourceOpcode) -> u8 {
    match opcode {
        SourceOpcode::Literal(_) => 0,
        SourceOpcode::TerminalSpot { .. } => 1,
        SourceOpcode::Add { .. } => 2,
        SourceOpcode::Subtract { .. } => 3,
        SourceOpcode::Multiply { .. } => 4,
        SourceOpcode::Divide { .. } => 5,
        SourceOpcode::Minimum { .. } => 6,
        SourceOpcode::Maximum { .. } => 7,
        SourceOpcode::Negate { .. } => 8,
        SourceOpcode::Indicator { .. } => 9,
        SourceOpcode::SmoothMinimum { .. } => 10,
        SourceOpcode::SmoothMaximum { .. } => 11,
        SourceOpcode::SmoothIndicator { .. } => 12,
        SourceOpcode::BarrierHitState { .. } => 13,
        SourceOpcode::PreDividendSpot { .. } => 14,
    }
}

const fn compiled_opcode_tag(opcode: CompiledOpcode) -> u8 {
    match opcode {
        CompiledOpcode::Literal { .. } => 0,
        CompiledOpcode::TerminalSpot { .. } => 1,
        CompiledOpcode::Add { .. } => 2,
        CompiledOpcode::Subtract { .. } => 3,
        CompiledOpcode::Multiply { .. } => 4,
        CompiledOpcode::Divide { .. } => 5,
        CompiledOpcode::Minimum { .. } => 6,
        CompiledOpcode::Maximum { .. } => 7,
        CompiledOpcode::Negate { .. } => 8,
        CompiledOpcode::Indicator { .. } => 9,
        CompiledOpcode::SmoothMinimum { .. } => 10,
        CompiledOpcode::SmoothMaximum { .. } => 11,
        CompiledOpcode::SmoothIndicator { .. } => 12,
        CompiledOpcode::BarrierHitState { .. } => 13,
        CompiledOpcode::PreDividendSpot { .. } => 14,
    }
}

fn put_date(bytes: &mut Vec<u8>, date: Date) {
    bytes.extend_from_slice(&date.year().to_be_bytes());
    bytes.push(date.month());
    bytes.push(date.day());
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphError {
    NodeIdExhausted,
    NonFiniteLiteral {
        bits: u64,
    },
    NoOutputs,
    DuplicateNodeId {
        node: NodeId,
    },
    UnknownOperand {
        node: NodeId,
        operand: NodeId,
    },
    UnknownOutput {
        node: NodeId,
    },
    Cycle {
        node: NodeId,
    },
    NonFiniteConstantFold {
        node: NodeId,
        opcode: &'static str,
        operand_bits: Vec<u64>,
        result_bits: u64,
    },
    SoftLimitExceeded {
        field: &'static str,
        observed: usize,
        limit: usize,
    },
    HardCapacity {
        field: &'static str,
        observed: usize,
    },
    SizeOverflow,
    InternalOrdering {
        operand: NodeId,
    },
    InvalidSlot {
        index: u32,
        length: usize,
    },
    MissingObservation {
        underlying: UnderlyingId,
        observation_date: Date,
    },
    MissingPreDividendObservation {
        underlying: UnderlyingId,
        observation_date: Date,
    },
    NonFiniteRuntimeValue {
        opcode: &'static str,
        bits: u64,
    },
    ReverseRequiresSingleOutput {
        count: usize,
    },
    NonFiniteRuntimeAdjoint {
        opcode: &'static str,
        bits: u64,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeIdExhausted => write!(formatter, "Source NodeId capacity is exhausted"),
            Self::NonFiniteLiteral { bits } => {
                write!(formatter, "payoff literal is non-finite: 0x{bits:016x}")
            }
            Self::NoOutputs => write!(formatter, "Source graph requires at least one output"),
            Self::DuplicateNodeId { node } => write!(formatter, "duplicate Source NodeId {node}"),
            Self::UnknownOperand { node, operand } => {
                write!(
                    formatter,
                    "Source node {node} references unknown operand {operand}"
                )
            }
            Self::UnknownOutput { node } => write!(formatter, "unknown graph output {node}"),
            Self::Cycle { node } => write!(formatter, "Source graph cycle includes node {node}"),
            Self::NonFiniteConstantFold {
                node,
                opcode,
                result_bits,
                ..
            } => write!(
                formatter,
                "constant fold at node {node} ({opcode}) is non-finite: 0x{result_bits:016x}"
            ),
            Self::SoftLimitExceeded {
                field,
                observed,
                limit,
            } => write!(
                formatter,
                "graph soft limit {field} exceeded: {observed} > {limit}"
            ),
            Self::HardCapacity { field, observed } => {
                write!(
                    formatter,
                    "graph hard capacity {field} exceeded by {observed}"
                )
            }
            Self::SizeOverflow => write!(formatter, "graph workspace-size estimate overflowed"),
            Self::InternalOrdering { operand } => {
                write!(
                    formatter,
                    "operand {operand} was not assigned before its consumer"
                )
            }
            Self::InvalidSlot { index, length } => {
                write!(
                    formatter,
                    "compiled slot {index} is outside length {length}"
                )
            }
            Self::MissingObservation {
                underlying,
                observation_date,
            } => write!(
                formatter,
                "missing observation for underlying {underlying} on {observation_date}"
            ),
            Self::MissingPreDividendObservation {
                underlying,
                observation_date,
            } => write!(
                formatter,
                "missing pre-dividend observation for underlying {underlying} on {observation_date}"
            ),
            Self::NonFiniteRuntimeValue { opcode, bits } => {
                write!(
                    formatter,
                    "runtime {opcode} produced non-finite value 0x{bits:016x}"
                )
            }
            Self::ReverseRequiresSingleOutput { count } => write!(
                formatter,
                "payoff reverse requires exactly one output; received {count}"
            ),
            Self::NonFiniteRuntimeAdjoint { opcode, bits } => write!(
                formatter,
                "runtime {opcode} produced non-finite adjoint 0x{bits:016x}"
            ),
        }
    }
}

impl Error for GraphError {}

#[cfg(test)]
mod tests {
    use pricing_core::CurrencyId;

    use super::*;
    use crate::BarrierMonitoring;

    fn option(side: OptionSide) -> EuropeanVanillaSpec {
        EuropeanVanillaSpec::new(
            UnderlyingId::new(4),
            CurrencyId::new(1),
            "2027-09-04".parse().expect("date"),
            100.0,
            2.0,
            side,
        )
        .expect("option")
    }

    #[test]
    fn european_builder_executes_exact_call_and_put_payoffs() {
        for (side, terminal, expected) in [
            (OptionSide::Call, 120.0, 40.0),
            (OptionSide::Call, 80.0, 0.0),
            (OptionSide::Put, 80.0, 40.0),
            (OptionSide::Put, 120.0, 0.0),
        ] {
            let compiled = option(side)
                .source_graph()
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            assert_eq!(
                compiled.evaluate(|_, _| Some(terminal)).expect("execute"),
                vec![expected]
            );
        }
    }

    #[test]
    fn european_reverse_returns_exact_terminal_adjoint() {
        for (side, terminal, expected_value, expected_adjoint) in [
            (OptionSide::Call, 120.0, 40.0, 2.0),
            (OptionSide::Call, 80.0, 0.0, 0.0),
            (OptionSide::Put, 80.0, 40.0, -2.0),
            (OptionSide::Put, 120.0, 0.0, 0.0),
            (OptionSide::Call, 100.0, 0.0, 2.0),
            (OptionSide::Put, 100.0, 0.0, -2.0),
        ] {
            let compiled = option(side)
                .source_graph()
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            let result = compiled
                .evaluate_single_with_terminal_adjoint(|_, _| Some(terminal))
                .expect("reverse");
            assert_eq!(result.value, expected_value);
            assert_eq!(result.terminal_adjoints.len(), 1);
            assert_eq!(result.terminal_adjoints[0].value, expected_adjoint);
            assert_eq!(result.terminal_adjoints[0].underlying, UnderlyingId::new(4));
            assert_eq!(
                result.terminal_adjoints[0].observation_date,
                "2027-09-04".parse().expect("date")
            );
        }
    }

    #[test]
    fn digital_builder_executes_exact_cash_and_asset_payoffs() {
        for (side, payout_kind, terminal, expected) in [
            (OptionSide::Call, DigitalPayout::Cash, 120.0, 10.0),
            (OptionSide::Call, DigitalPayout::Cash, 99.0, 0.0),
            (OptionSide::Call, DigitalPayout::Cash, 100.0, 10.0),
            (OptionSide::Put, DigitalPayout::Cash, 80.0, 10.0),
            (OptionSide::Put, DigitalPayout::Cash, 101.0, 0.0),
            (OptionSide::Put, DigitalPayout::Cash, 100.0, 10.0),
            (OptionSide::Call, DigitalPayout::Asset, 120.0, 1200.0),
            (OptionSide::Put, DigitalPayout::Asset, 80.0, 800.0),
        ] {
            let product = DigitalSpec::new(
                UnderlyingId::new(4),
                CurrencyId::new(1),
                "2027-09-04".parse().expect("date"),
                100.0,
                10.0,
                side,
                payout_kind,
            )
            .expect("digital");
            let compiled = product
                .source_graph()
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            assert_eq!(
                compiled.evaluate(|_, _| Some(terminal)).expect("execute"),
                vec![expected]
            );
        }
    }

    #[test]
    fn digital_builder_executes_smoothed_cash_and_asset_payoffs_with_adjoints() {
        let smoothing = CompactC2Smoothing::new(2.0).expect("smoothing");
        for (side, payout_kind, expected_value, expected_adjoint) in [
            (OptionSide::Call, DigitalPayout::Cash, 5.0, 4.6875),
            (OptionSide::Put, DigitalPayout::Cash, 5.0, -4.6875),
            (OptionSide::Call, DigitalPayout::Asset, 500.0, 473.75),
            (OptionSide::Put, DigitalPayout::Asset, 500.0, -463.75),
        ] {
            let product = DigitalSpec::new(
                UnderlyingId::new(4),
                CurrencyId::new(1),
                "2027-09-04".parse().expect("date"),
                100.0,
                10.0,
                side,
                payout_kind,
            )
            .expect("digital");
            let compiled = product
                .smoothed_source_graph(smoothing)
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            let result = compiled
                .evaluate_single_with_terminal_adjoint(|_, _| Some(100.0))
                .expect("evaluate");
            assert_eq!(result.value, expected_value);
            assert_eq!(result.terminal_adjoints.len(), 1);
            assert_eq!(result.terminal_adjoints[0].value, expected_adjoint);
        }
    }

    #[test]
    fn digital_exact_and_smoothed_graphs_remain_distinct() {
        let product = DigitalSpec::new(
            UnderlyingId::new(4),
            CurrencyId::new(1),
            "2027-09-04".parse().expect("date"),
            100.0,
            10.0,
            OptionSide::Call,
            DigitalPayout::Cash,
        )
        .expect("digital");
        let exact = product
            .source_graph()
            .expect("exact graph")
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("exact compile");
        let smoothed = product
            .smoothed_source_graph(CompactC2Smoothing::new(2.0).expect("smoothing"))
            .expect("smoothed graph")
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("smoothed compile");
        assert_ne!(exact.source_fingerprint(), smoothed.source_fingerprint());
        assert_ne!(exact.tape_fingerprint(), smoothed.tape_fingerprint());
        assert_eq!(
            exact.evaluate(|_, _| Some(100.0)).expect("exact"),
            vec![10.0]
        );
        assert_eq!(
            smoothed.evaluate(|_, _| Some(100.0)).expect("smoothed"),
            vec![5.0]
        );
    }

    #[test]
    fn barrier_builder_executes_discrete_knock_out_and_knock_in_payoffs() {
        for (style, march_spot, expiry_spot, expected) in [
            (BarrierStyle::KnockOut, 110.0, 115.0, 30.0),
            (BarrierStyle::KnockOut, 125.0, 115.0, 0.0),
            (BarrierStyle::KnockIn, 110.0, 115.0, 0.0),
            (BarrierStyle::KnockIn, 125.0, 115.0, 30.0),
        ] {
            let product = BarrierSpec::new(
                UnderlyingId::new(4),
                CurrencyId::new(1),
                "2027-09-04".parse().expect("expiry"),
                100.0,
                120.0,
                2.0,
                OptionSide::Call,
                BarrierDirection::Up,
                style,
                BarrierMonitoring::Discrete,
                vec![
                    "2027-03-04".parse().expect("monitoring"),
                    "2027-09-04".parse().expect("expiry"),
                ],
                None,
                "2027-09-04".parse().expect("payment"),
            )
            .expect("barrier");
            let compiled = product
                .source_graph()
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            assert_eq!(
                compiled
                    .evaluate(|_, date| match date.to_string().as_str() {
                        "2027-03-04" => Some(march_spot),
                        "2027-09-04" => Some(expiry_spot),
                        _ => None,
                    })
                    .expect("execute"),
                vec![expected]
            );
        }
    }

    #[test]
    fn barrier_builder_pays_rebate_when_vanilla_branch_is_inactive() {
        for (style, march_spot, expected) in [
            (BarrierStyle::KnockOut, 125.0, 7.0),
            (BarrierStyle::KnockIn, 110.0, 7.0),
        ] {
            let product = BarrierSpec::new(
                UnderlyingId::new(4),
                CurrencyId::new(1),
                "2027-09-04".parse().expect("expiry"),
                100.0,
                120.0,
                2.0,
                OptionSide::Call,
                BarrierDirection::Up,
                style,
                BarrierMonitoring::Discrete,
                vec![
                    "2027-03-04".parse().expect("monitoring"),
                    "2027-09-04".parse().expect("expiry"),
                ],
                Some(7.0),
                "2027-09-04".parse().expect("payment"),
            )
            .expect("barrier");
            let compiled = product
                .source_graph()
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            assert_eq!(
                compiled
                    .evaluate(|_, date| match date.to_string().as_str() {
                        "2027-03-04" => Some(march_spot),
                        "2027-09-04" => Some(115.0),
                        _ => None,
                    })
                    .expect("execute"),
                vec![expected]
            );
        }
    }

    #[test]
    fn barrier_hit_state_is_monotone_inclusive_and_limit_checked() {
        let product = BarrierSpec::new(
            UnderlyingId::new(4),
            CurrencyId::new(1),
            "2027-09-04".parse().expect("expiry"),
            100.0,
            120.0,
            2.0,
            OptionSide::Call,
            BarrierDirection::Up,
            BarrierStyle::KnockIn,
            BarrierMonitoring::Discrete,
            vec![
                "2027-03-04".parse().expect("monitoring"),
                "2027-09-04".parse().expect("expiry"),
            ],
            None,
            "2027-09-04".parse().expect("payment"),
        )
        .expect("barrier");
        let graph = product.source_graph().expect("graph");
        assert_eq!(
            graph
                .nodes()
                .iter()
                .filter(|node| matches!(node.opcode(), SourceOpcode::BarrierHitState { .. }))
                .count(),
            2
        );
        let compiled = graph.compile(GraphLimitPolicy::DEFAULT).expect("compile");
        let monitoring = "2027-03-04".parse().expect("monitoring");
        let expiry = "2027-09-04".parse().expect("expiry");
        assert_eq!(
            compiled
                .evaluate(|_, date| (date == monitoring)
                    .then_some(120.0)
                    .or_else(|| { (date == expiry).then_some(110.0) }))
                .expect("inclusive touch"),
            vec![20.0]
        );
        let mut limits = GraphLimitPolicy::DEFAULT;
        limits.state_slots = 1;
        assert!(matches!(
            graph.compile(limits),
            Err(GraphError::SoftLimitExceeded {
                field: "state_slots",
                observed: 2,
                limit: 1,
            })
        ));
    }

    #[test]
    fn smoothed_barrier_knock_parity_and_reverse_hold_for_up_and_down() {
        let smoothing = CompactC2Smoothing::new(2.0).expect("smoothing");
        let monitoring = "2027-03-04".parse().expect("monitoring");
        let expiry = "2027-09-04".parse().expect("expiry");
        for (direction, first_spot, terminal_spot) in [
            (BarrierDirection::Up, 119.5, 121.0),
            (BarrierDirection::Down, 80.5, 79.0),
        ] {
            let barrier = match direction {
                BarrierDirection::Up => 120.0,
                BarrierDirection::Down => 80.0,
            };
            let compile = |style, rebate| {
                BarrierSpec::new(
                    UnderlyingId::new(4),
                    CurrencyId::new(1),
                    expiry,
                    70.0,
                    barrier,
                    2.0,
                    OptionSide::Call,
                    direction,
                    style,
                    BarrierMonitoring::Discrete,
                    vec![monitoring, expiry],
                    rebate,
                    expiry,
                )
                .expect("barrier")
                .smoothed_source_graph(smoothing)
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile")
            };
            let knock_in = compile(BarrierStyle::KnockIn, Some(7.0));
            let knock_out = compile(BarrierStyle::KnockOut, Some(7.0));
            let observe = |date| {
                if date == monitoring {
                    Some(first_spot)
                } else if date == expiry {
                    Some(terminal_spot)
                } else {
                    None
                }
            };
            let knock_in_value = knock_in
                .evaluate(|_, date| observe(date))
                .expect("knock in")[0];
            let knock_out_value = knock_out
                .evaluate(|_, date| observe(date))
                .expect("knock out")[0];
            let vanilla = (terminal_spot - 70.0) * 2.0;
            assert!((knock_in_value + knock_out_value - (vanilla + 7.0)).abs() < 1.0e-12);

            let evaluated = knock_in
                .evaluate_single_with_terminal_adjoint(|_, date| observe(date))
                .expect("reverse");
            for date in [monitoring, expiry] {
                let epsilon = 1.0e-5;
                let bumped = |shift| {
                    knock_in
                        .evaluate(|_, query| {
                            if query == monitoring {
                                Some(first_spot + if date == monitoring { shift } else { 0.0 })
                            } else if query == expiry {
                                Some(terminal_spot + if date == expiry { shift } else { 0.0 })
                            } else {
                                None
                            }
                        })
                        .expect("bump")[0]
                };
                let finite_difference = (bumped(epsilon) - bumped(-epsilon)) / (2.0 * epsilon);
                let adjoint = evaluated
                    .terminal_adjoints
                    .iter()
                    .filter(|item| item.observation_date == date)
                    .map(|item| item.value)
                    .sum::<f64>();
                assert!((adjoint - finite_difference).abs() < 1.0e-7);
            }
        }
    }

    #[test]
    fn smoothed_barrier_dividend_jump_matches_p0_fixture_and_reverse() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../fixtures/path-dependence/reference-cases-v0.1.json"
        ))
        .expect("fixture");
        let expiry: Date = "2027-09-04".parse().expect("expiry");
        for case in fixture["jump_cases"].as_array().expect("jump cases") {
            let parse = |field: &str| {
                case[field]
                    .as_str()
                    .expect("decimal string")
                    .parse::<f64>()
                    .expect("binary64")
            };
            let expected = |field: &str| {
                case["expected"][field]
                    .as_str()
                    .expect("decimal string")
                    .parse::<f64>()
                    .expect("binary64")
            };
            let pre = parse("pre_jump_spot");
            let post = parse("post_jump_spot");
            let smoothing = CompactC2Smoothing::new(parse("half_width")).expect("smoothing");
            let product = BarrierSpec::new(
                UnderlyingId::new(4),
                CurrencyId::new(1),
                expiry,
                1.0,
                parse("barrier"),
                1.0,
                OptionSide::Call,
                BarrierDirection::Down,
                BarrierStyle::KnockIn,
                BarrierMonitoring::Discrete,
                vec![expiry],
                None,
                expiry,
            )
            .expect("barrier");
            let compiled = product
                .smoothed_source_graph_with_dividend_jumps(smoothing, &[expiry])
                .expect("graph")
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            assert_eq!(
                compiled.pre_dividend_observations(),
                vec![(UnderlyingId::new(4), expiry)]
            );
            let value = compiled
                .evaluate_with_pre_dividend_spots(|_, _| Some(post), |_, _| Some(pre))
                .expect("evaluate")[0];
            assert!((value / (post - 1.0) - expected("hit_weight")).abs() < 2.0e-15);

            let evaluation = compiled
                .evaluate_single_with_observation_adjoints(|_, _| Some(post), |_, _| Some(pre))
                .expect("reverse");
            let epsilon = 1.0e-5;
            let finite_difference = |bump_pre: bool| {
                let bumped = |shift| {
                    compiled
                        .evaluate_with_pre_dividend_spots(
                            |_, _| Some(post + if bump_pre { 0.0 } else { shift }),
                            |_, _| Some(pre + if bump_pre { shift } else { 0.0 }),
                        )
                        .expect("bump")[0]
                };
                (bumped(epsilon) - bumped(-epsilon)) / (2.0 * epsilon)
            };
            let pre_adjoint = evaluation
                .pre_dividend_adjoints
                .iter()
                .map(|adjoint| adjoint.value)
                .sum::<f64>();
            let post_adjoint = evaluation
                .terminal_adjoints
                .iter()
                .map(|adjoint| adjoint.value)
                .sum::<f64>();
            assert!((pre_adjoint - finite_difference(true)).abs() < 1.0e-7);
            assert!((post_adjoint - finite_difference(false)).abs() < 1.0e-7);
        }
    }

    #[test]
    fn arithmetic_asian_builder_executes_weighted_average_payoff() {
        let product = ArithmeticAsianSpec::new(
            UnderlyingId::new(4),
            CurrencyId::new(1),
            100.0,
            2.0,
            OptionSide::Call,
            vec![
                crate::AsianObservation::known("2027-03-04".parse().expect("date"), 0.25, 90.0)
                    .expect("known"),
                crate::AsianObservation::unknown("2027-09-04".parse().expect("date"), 0.75)
                    .expect("unknown"),
            ],
            "2027-09-04".parse().expect("payment"),
        )
        .expect("asian");
        let compiled = product
            .source_graph()
            .expect("graph")
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("compile");
        assert_eq!(
            compiled
                .evaluate(|_, date| (date.to_string() == "2027-09-04").then_some(130.0))
                .expect("execute"),
            vec![40.0]
        );
        assert_eq!(
            compiled.terminal_observations(),
            vec![(UnderlyingId::new(4), "2027-09-04".parse().expect("date"))]
        );
    }

    #[test]
    fn fixed_lookback_builder_executes_running_extremum_payoff() {
        let product = FixedLookbackSpec::new(
            UnderlyingId::new(4),
            CurrencyId::new(1),
            100.0,
            2.0,
            OptionSide::Call,
            vec![
                "2027-03-04".parse().expect("date"),
                "2027-06-04".parse().expect("date"),
                "2027-09-04".parse().expect("date"),
            ],
            Some(112.0),
            "2027-09-04".parse().expect("payment"),
        )
        .expect("lookback");
        let compiled = product
            .source_graph("2027-05-04".parse().expect("valuation"))
            .expect("graph")
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("compile");
        assert_eq!(
            compiled
                .evaluate(|_, date| match date.to_string().as_str() {
                    "2027-06-04" => Some(95.0),
                    "2027-09-04" => Some(118.0),
                    _ => None,
                })
                .expect("execute"),
            vec![36.0]
        );
        assert_eq!(
            compiled.terminal_observations(),
            vec![
                (UnderlyingId::new(4), "2027-06-04".parse().expect("date")),
                (UnderlyingId::new(4), "2027-09-04".parse().expect("date")),
            ]
        );
    }

    #[test]
    fn manual_and_standard_graphs_have_identical_fingerprints() {
        let contract = option(OptionSide::Call);
        let standard = contract.source_graph().expect("standard graph");
        let nodes = vec![
            SourceNode::new(
                NodeId::new(0),
                SourceOpcode::TerminalSpot {
                    underlying: contract.underlying(),
                    observation_date: contract.expiry(),
                },
            ),
            SourceNode::new(
                NodeId::new(1),
                SourceOpcode::Literal(
                    FiniteF64::new(contract.strike().get(), "strike").expect("strike"),
                ),
            ),
            SourceNode::new(
                NodeId::new(2),
                SourceOpcode::Subtract {
                    left: NodeId::new(0),
                    right: NodeId::new(1),
                },
            ),
            SourceNode::new(
                NodeId::new(3),
                SourceOpcode::Literal(FiniteF64::new(0.0, "zero").expect("zero")),
            ),
            SourceNode::new(
                NodeId::new(4),
                SourceOpcode::Maximum {
                    left: NodeId::new(2),
                    right: NodeId::new(3),
                },
            ),
            SourceNode::new(
                NodeId::new(5),
                SourceOpcode::Literal(
                    FiniteF64::new(contract.notional().get(), "notional").expect("notional"),
                ),
            ),
            SourceNode::new(
                NodeId::new(6),
                SourceOpcode::Multiply {
                    left: NodeId::new(4),
                    right: NodeId::new(5),
                },
            ),
        ];
        let manual = SourceGraph::new(nodes, vec![NodeId::new(6)]);
        let standard = standard
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("standard");
        let manual = manual.compile(GraphLimitPolicy::DEFAULT).expect("manual");
        assert_eq!(standard.source_fingerprint(), manual.source_fingerprint());
        assert_eq!(standard.tape_fingerprint(), manual.tape_fingerprint());
    }

    #[test]
    fn input_order_does_not_change_kahn_order_or_fingerprints() {
        let graph = option(OptionSide::Put).source_graph().expect("graph");
        let mut reversed = graph.nodes().to_vec();
        reversed.reverse();
        let reordered = SourceGraph::new(reversed, graph.outputs().to_vec());
        let left = graph.compile(GraphLimitPolicy::DEFAULT).expect("left");
        let right = reordered.compile(GraphLimitPolicy::DEFAULT).expect("right");
        assert_eq!(left.source_fingerprint(), right.source_fingerprint());
        assert_eq!(left.tape_fingerprint(), right.tape_fingerprint());
        assert_eq!(left.opcodes(), right.opcodes());
    }

    #[test]
    fn dead_nodes_are_validated_before_removal() {
        let one = FiniteF64::new(1.0, "one").expect("one");
        let zero = FiniteF64::new(0.0, "zero").expect("zero");
        let invalid_dead = SourceGraph::new(
            vec![
                SourceNode::new(NodeId::new(0), SourceOpcode::Literal(one)),
                SourceNode::new(NodeId::new(1), SourceOpcode::Literal(zero)),
                SourceNode::new(
                    NodeId::new(2),
                    SourceOpcode::Divide {
                        numerator: NodeId::new(0),
                        denominator: NodeId::new(1),
                    },
                ),
            ],
            vec![NodeId::new(0)],
        );
        assert!(matches!(
            invalid_dead.compile(GraphLimitPolicy::DEFAULT),
            Err(GraphError::NonFiniteConstantFold {
                node,
                opcode: "divide",
                ..
            }) if node == NodeId::new(2)
        ));

        let valid_dead = SourceGraph::new(
            vec![
                SourceNode::new(NodeId::new(0), SourceOpcode::Literal(one)),
                SourceNode::new(NodeId::new(9), SourceOpcode::Literal(zero)),
            ],
            vec![NodeId::new(0)],
        )
        .compile(GraphLimitPolicy::DEFAULT)
        .expect("compile");
        assert_eq!(valid_dead.removed_source_nodes(), &[NodeId::new(9)]);
    }

    #[test]
    fn cycles_duplicate_ids_unknown_references_and_limits_are_errors() {
        let cycle = SourceGraph::new(
            vec![
                SourceNode::new(
                    NodeId::new(0),
                    SourceOpcode::Negate {
                        input: NodeId::new(1),
                    },
                ),
                SourceNode::new(
                    NodeId::new(1),
                    SourceOpcode::Negate {
                        input: NodeId::new(0),
                    },
                ),
            ],
            vec![NodeId::new(0)],
        );
        assert!(matches!(
            cycle.compile(GraphLimitPolicy::DEFAULT),
            Err(GraphError::Cycle { .. })
        ));

        let mut limits = GraphLimitPolicy::DEFAULT;
        limits.source_nodes = 1;
        assert!(matches!(
            option(OptionSide::Call)
                .source_graph()
                .expect("graph")
                .compile(limits),
            Err(GraphError::SoftLimitExceeded {
                field: "source_nodes",
                ..
            })
        ));
    }

    #[test]
    fn constant_fold_preserves_node_identity_and_signed_zero() {
        let mut builder = SourceGraphBuilder::new();
        let positive = builder.literal(0.0).expect("positive zero");
        let negative = builder.literal(-0.0).expect("negative zero");
        let sum = builder
            .push(SourceOpcode::Add {
                left: positive,
                right: negative,
            })
            .expect("sum");
        let compiled = builder
            .finish(vec![sum])
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("compile");
        assert_eq!(compiled.opcodes().len(), 3);
        assert_eq!(compiled.source_to_slot().len(), 3);

        let positive_only = SourceGraph::new(
            vec![SourceNode::new(
                NodeId::new(0),
                SourceOpcode::Literal(FiniteF64::new(0.0, "value").expect("value")),
            )],
            vec![NodeId::new(0)],
        );
        let negative_only = SourceGraph::new(
            vec![SourceNode::new(
                NodeId::new(0),
                SourceOpcode::Literal(FiniteF64::new(-0.0, "value").expect("value")),
            )],
            vec![NodeId::new(0)],
        );
        assert_ne!(
            positive_only
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile")
                .source_fingerprint(),
            negative_only
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile")
                .source_fingerprint()
        );
    }

    #[test]
    fn smooth_indicator_executes_and_reverses_the_p0_kernel() {
        let mut builder = SourceGraphBuilder::new();
        let date = "2027-09-04".parse().expect("date");
        let input = builder
            .push(SourceOpcode::TerminalSpot {
                underlying: UnderlyingId::new(4),
                observation_date: date,
            })
            .expect("input");
        let output = builder
            .push(SourceOpcode::SmoothIndicator {
                input,
                smoothing: CompactC2Smoothing::new(2.0).expect("smoothing"),
            })
            .expect("indicator");
        let compiled = builder
            .finish(vec![output])
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("compile");

        let evaluation = compiled
            .evaluate_single_with_terminal_adjoint(|_, _| Some(0.0))
            .expect("evaluate");
        assert_eq!(evaluation.value, 0.5);
        assert_eq!(evaluation.terminal_adjoints.len(), 1);
        assert_eq!(evaluation.terminal_adjoints[0].value, 0.468_75);

        for (input, expected) in [
            (-3.0, 0.0),
            (-1.0, 0.103_515_625),
            (1.0, 0.896_484_375),
            (3.0, 1.0),
        ] {
            assert_eq!(
                compiled.evaluate(|_, _| Some(input)).expect("evaluate"),
                vec![expected]
            );
        }
    }

    #[test]
    fn smooth_extrema_execute_and_reverse_as_a_paired_partition() {
        let first_date = "2027-03-04".parse().expect("first date");
        let second_date = "2027-09-04".parse().expect("second date");
        for maximum in [true, false] {
            let mut builder = SourceGraphBuilder::new();
            let left = builder
                .push(SourceOpcode::TerminalSpot {
                    underlying: UnderlyingId::new(4),
                    observation_date: first_date,
                })
                .expect("left");
            let right = builder
                .push(SourceOpcode::TerminalSpot {
                    underlying: UnderlyingId::new(4),
                    observation_date: second_date,
                })
                .expect("right");
            let smoothing = CompactC2Smoothing::new(4.0).expect("smoothing");
            let output = if maximum {
                builder.push(SourceOpcode::SmoothMaximum {
                    left,
                    right,
                    smoothing,
                })
            } else {
                builder.push(SourceOpcode::SmoothMinimum {
                    left,
                    right,
                    smoothing,
                })
            }
            .expect("extremum");
            let compiled = builder
                .finish(vec![output])
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile");
            let evaluation = compiled
                .evaluate_single_with_terminal_adjoint(|_, date| {
                    Some(if date == first_date { 101.0 } else { 99.0 })
                })
                .expect("evaluate");
            assert_eq!(
                evaluation.value,
                if maximum {
                    101.056_640_625
                } else {
                    98.943_359_375
                }
            );
            let left_adjoint = evaluation
                .terminal_adjoints
                .iter()
                .find(|adjoint| adjoint.observation_date == first_date)
                .expect("left adjoint")
                .value;
            let right_adjoint = evaluation
                .terminal_adjoints
                .iter()
                .find(|adjoint| adjoint.observation_date == second_date)
                .expect("right adjoint")
                .value;
            let expected_left = if maximum {
                0.896_484_375
            } else {
                0.103_515_625
            };
            assert_eq!(left_adjoint, expected_left);
            assert_eq!(right_adjoint, 1.0 - expected_left);
        }
    }

    #[test]
    fn smooth_constant_folding_and_fingerprints_include_half_width() {
        let compile = |half_width| {
            let mut builder = SourceGraphBuilder::new();
            let input = builder.literal(0.0).expect("input");
            let output = builder
                .push(SourceOpcode::SmoothIndicator {
                    input,
                    smoothing: CompactC2Smoothing::new(half_width).expect("smoothing"),
                })
                .expect("indicator");
            builder
                .finish(vec![output])
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile")
        };
        let narrow = compile(1.0);
        let wide = compile(2.0);
        assert!(matches!(
            narrow.opcodes()[1],
            CompiledOpcode::Literal { value: 0.5, .. }
        ));
        assert_ne!(narrow.source_fingerprint(), wide.source_fingerprint());

        let compile_dynamic = |half_width| {
            let mut builder = SourceGraphBuilder::new();
            let input = builder
                .push(SourceOpcode::TerminalSpot {
                    underlying: UnderlyingId::new(4),
                    observation_date: "2027-09-04".parse().expect("date"),
                })
                .expect("input");
            let output = builder
                .push(SourceOpcode::SmoothIndicator {
                    input,
                    smoothing: CompactC2Smoothing::new(half_width).expect("smoothing"),
                })
                .expect("indicator");
            builder
                .finish(vec![output])
                .compile(GraphLimitPolicy::DEFAULT)
                .expect("compile")
        };
        assert_ne!(
            compile_dynamic(1.0).tape_fingerprint(),
            compile_dynamic(2.0).tape_fingerprint()
        );
    }

    #[test]
    fn existing_exact_payoff_fingerprint_is_unchanged() {
        let compiled = option(OptionSide::Call)
            .source_graph()
            .expect("graph")
            .compile(GraphLimitPolicy::DEFAULT)
            .expect("compile");
        assert_eq!(
            compiled.tape_fingerprint().to_string(),
            "blake3-256:32f30625f81f30987a54c18c1b62e20549a86959873c2e33f2b68c84e97aabf3"
        );
    }
}
