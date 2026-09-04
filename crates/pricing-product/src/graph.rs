use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::mem;

use pricing_core::{Date, FiniteF64, NodeId, UnderlyingId};

use crate::{EuropeanVanillaSpec, OptionSide};

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
    Negate {
        input: NodeId,
    },
}

impl SourceOpcode {
    fn operands(self) -> ([NodeId; 2], usize) {
        match self {
            Self::Literal(_) | Self::TerminalSpot { .. } => ([NodeId::new(0); 2], 0),
            Self::Negate { input } => ([input, NodeId::new(0)], 1),
            Self::Add { left, right }
            | Self::Subtract { left, right }
            | Self::Multiply { left, right }
            | Self::Minimum { left, right }
            | Self::Maximum { left, right } => ([left, right], 2),
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
            Self::Add { .. } => "add",
            Self::Subtract { .. } => "subtract",
            Self::Multiply { .. } => "multiply",
            Self::Divide { .. } => "divide",
            Self::Minimum { .. } => "minimum",
            Self::Maximum { .. } => "maximum",
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

#[derive(Clone, Debug, PartialEq)]
pub struct PayoffEvaluation {
    pub value: f64,
    pub terminal_adjoints: Box<[TerminalAdjoint]>,
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

    pub fn evaluate<F>(&self, mut observation: F) -> Result<Vec<f64>, GraphError>
    where
        F: FnMut(UnderlyingId, Date) -> Option<f64>,
    {
        let mut slots = vec![0.0; self.opcodes.len()];
        for opcode in &self.opcodes {
            let (output, value) = execute_opcode(*opcode, &slots, &mut observation)?;
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
        if self.output_slots.len() != 1 {
            return Err(GraphError::ReverseRequiresSingleOutput {
                count: self.output_slots.len(),
            });
        }
        let mut values = vec![0.0; self.opcodes.len()];
        for opcode in &self.opcodes {
            let (output, value) = execute_opcode(*opcode, &values, &mut observation)?;
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
        for opcode in self.opcodes.iter().rev().copied() {
            reverse_opcode(opcode, &values, &mut adjoints, &mut terminal_adjoints)?;
        }
        Ok(PayoffEvaluation {
            value,
            terminal_adjoints: terminal_adjoints.into_boxed_slice(),
        })
    }
}

fn reverse_opcode(
    opcode: CompiledOpcode,
    values: &[f64],
    adjoints: &mut [f64],
    terminal_adjoints: &mut Vec<TerminalAdjoint>,
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
            Self::Add { .. } => "add",
            Self::Subtract { .. } => "subtract",
            Self::Multiply { .. } => "multiply",
            Self::Divide { .. } => "divide",
            Self::Minimum { .. } => "minimum",
            Self::Maximum { .. } => "maximum",
            Self::Negate { .. } => "negate",
        }
    }

    const fn output(self) -> u32 {
        match self {
            Self::Literal { output, .. }
            | Self::TerminalSpot { output, .. }
            | Self::Add { output, .. }
            | Self::Subtract { output, .. }
            | Self::Multiply { output, .. }
            | Self::Divide { output, .. }
            | Self::Minimum { output, .. }
            | Self::Maximum { output, .. }
            | Self::Negate { output, .. } => output,
        }
    }
}

fn execute_opcode<F>(
    opcode: CompiledOpcode,
    slots: &[f64],
    observation: &mut F,
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
                let degree = indegree.get_mut(destination).expect("known destination");
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
            .expect("cycle leaves non-zero indegree");
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
    if matches!(opcode, SourceOpcode::TerminalSpot { .. }) {
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
        SourceOpcode::Negate { .. } => -values[0],
        SourceOpcode::Literal(_) | SourceOpcode::TerminalSpot { .. } => unreachable!(),
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
        SourceOpcode::Negate { input } => put_u32(bytes, input.get()),
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
        CompiledOpcode::Negate { input, output } => {
            put_u32(bytes, input);
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
}
