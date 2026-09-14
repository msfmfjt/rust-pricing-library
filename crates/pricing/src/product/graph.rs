use crate::core::{Date, FiniteF64, NodeId, UnderlyingId};
use crate::product::CompactC2Smoothing;
use std::error::Error;
use std::fmt;

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
pub struct GraphFingerprint(pub(crate) [u8; 32]);

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
    pub(crate) fn operands(self) -> ([NodeId; 2], usize) {
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

    pub(crate) const fn name(self) -> &'static str {
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
    pub(crate) id: NodeId,
    pub(crate) opcode: SourceOpcode,
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
    pub(crate) nodes: Box<[SourceNode]>,
    pub(crate) outputs: Box<[NodeId]>,
}

impl SourceGraph {
    #[must_use]
    pub fn new(nodes: Vec<SourceNode>, outputs: Vec<NodeId>) -> Self {
        Self {
            nodes: nodes.into_boxed_slice(),
            outputs: outputs.into_boxed_slice(),
        }
    }
}

impl SourceGraph {
    #[must_use]
    pub fn nodes(&self) -> &[SourceNode] {
        &self.nodes
    }
}

impl SourceGraph {
    #[must_use]
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceGraphBuilder {
    pub(crate) nodes: Vec<SourceNode>,
    pub(crate) next_id: Option<u32>,
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
    InvalidOutputIndex {
        index: usize,
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
            Self::InvalidOutputIndex { index, count } => write!(
                formatter,
                "payoff output index {index} is outside output count {count}"
            ),
            Self::NonFiniteRuntimeAdjoint { opcode, bits } => write!(
                formatter,
                "runtime {opcode} produced non-finite adjoint 0x{bits:016x}"
            ),
        }
    }
}

impl Error for GraphError {}
