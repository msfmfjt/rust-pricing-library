use crate::core::{Date, FiniteF64, NodeId};
use crate::engine::payoff::tape::CompiledOpcode;
use crate::engine::payoff::tape::CompiledPayoff;
use crate::product::graph::GraphError;
use crate::product::graph::GraphFingerprint;
use crate::product::graph::GraphLimitPolicy;
use crate::product::graph::SourceGraph;
use crate::product::graph::SourceGraphBuilder;
use crate::product::graph::SourceNode;
use crate::product::graph::SourceOpcode;
use crate::product::{
    AmericanVanillaSpec, ArithmeticAsianSpec, AsianObservationValue, BarrierDirection, BarrierSpec,
    BarrierStyle, CompactC2Smoothing, DigitalPayout, DigitalSpec, EuropeanVanillaSpec,
    FixedLookbackSpec, OptionSide, ProductSpec,
};
use std::collections::{BTreeMap, BTreeSet};
use std::mem;

pub(in crate::engine) const SOURCE_GRAPH_VERSION: u32 = 1;

pub(in crate::engine) const TAPE_ABI_VERSION: u32 = 1;

impl SourceGraph {
    pub fn compile(&self, limits: GraphLimitPolicy) -> Result<CompiledPayoff, GraphError> {
        compile(self, limits)
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

impl AmericanVanillaSpec {
    pub fn source_graph(&self) -> Result<SourceGraph, GraphError> {
        let mut builder = SourceGraphBuilder::new();
        let strike = builder.literal(self.strike().get())?;
        let zero = builder.literal(0.0)?;
        let notional = builder.literal(self.notional().get())?;
        let mut outputs = Vec::with_capacity(self.exercise_dates().len());
        for &observation_date in self.exercise_dates() {
            let spot = builder.push(SourceOpcode::TerminalSpot {
                underlying: self.underlying(),
                observation_date,
            })?;
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
            let positive_part = builder.push(SourceOpcode::Maximum {
                left: signed_intrinsic,
                right: zero,
            })?;
            outputs.push(builder.push(SourceOpcode::Multiply {
                left: positive_part,
                right: notional,
            })?);
        }
        Ok(builder.finish(outputs))
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

    pub(in crate::engine) fn build_source_graph(
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

    pub(in crate::engine) fn build_source_graph(
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
            Self::AmericanVanilla(spec) => spec.source_graph(),
            Self::Digital(spec) => spec.source_graph(),
            Self::Barrier(spec) => spec.source_graph(),
            Self::ArithmeticAsian(spec) => spec.source_graph(),
            Self::FixedLookback(spec) => spec.source_graph(valuation_date),
        }
    }
}

pub(in crate::engine) fn compile(
    graph: &SourceGraph,
    limits: GraphLimitPolicy,
) -> Result<CompiledPayoff, GraphError> {
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

pub(in crate::engine) fn reachable_nodes(
    nodes: &BTreeMap<NodeId, SourceOpcode>,
    outputs: &[NodeId],
) -> BTreeSet<NodeId> {
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

pub(in crate::engine) fn fold_node(
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

pub(in crate::engine) fn compile_opcode(
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

pub(in crate::engine) fn estimate_bytes(
    nodes: usize,
    edges: usize,
    outputs: usize,
) -> Result<usize, GraphError> {
    nodes
        .checked_mul(mem::size_of::<SourceNode>() + mem::size_of::<CompiledOpcode>() + 16)
        .and_then(|value| value.checked_add(edges.checked_mul(mem::size_of::<NodeId>())?))
        .and_then(|value| value.checked_add(outputs.checked_mul(mem::size_of::<u32>())?))
        .ok_or(GraphError::SizeOverflow)
}

pub(in crate::engine) fn enforce_limit(
    field: &'static str,
    observed: usize,
    limit: usize,
) -> Result<(), GraphError> {
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

pub(in crate::engine) fn fingerprint_source(
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

pub(in crate::engine) fn fingerprint_tape(
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

pub(in crate::engine) fn encode_source_opcode(bytes: &mut Vec<u8>, opcode: SourceOpcode) {
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

pub(in crate::engine) fn encode_compiled_opcode(bytes: &mut Vec<u8>, opcode: CompiledOpcode) {
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

pub(in crate::engine) const fn opcode_tag(opcode: SourceOpcode) -> u8 {
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

pub(in crate::engine) const fn compiled_opcode_tag(opcode: CompiledOpcode) -> u8 {
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

pub(in crate::engine) fn put_date(bytes: &mut Vec<u8>, date: Date) {
    bytes.extend_from_slice(&date.year().to_be_bytes());
    bytes.push(date.month());
    bytes.push(date.day());
}

pub(in crate::engine) fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

pub(in crate::engine) fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_be_bytes());
}
