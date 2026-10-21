//! Validated, model-independent payoff graphs and contractual cash-flow dates.
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use super::{CompiledOpcode, GraphError, GraphLimitPolicy, SourceGraph, SourceOpcode};
use crate::core::{CurrencyId, Date, NodeId, UnderlyingId};

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphProductError {
    Graph(GraphError),
    Invalid(&'static str),
}
impl fmt::Display for GraphProductError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(e) => e.fmt(f),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}
impl Error for GraphProductError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Graph(e) => Some(e),
            Self::Invalid(_) => None,
        }
    }
}
impl From<GraphError> for GraphProductError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

/// An immutable source graph. Output order is cash-flow order, including when
/// two outputs reference the same node or share a payment date.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphProduct {
    currency: CurrencyId,
    underlyings: Vec<UnderlyingId>,
    graph: SourceGraph,
    payment_dates: Vec<Date>,
    observations: Vec<(UnderlyingId, Date)>,
    expiry: Date,
    last_payment: Date,
    pathwise_risk: bool,
}
impl GraphProduct {
    pub fn new(
        currency: CurrencyId,
        underlyings: Vec<UnderlyingId>,
        graph: SourceGraph,
        payment_dates: Vec<Date>,
    ) -> Result<Self, GraphProductError> {
        let declared: BTreeSet<_> = underlyings.iter().copied().collect();
        if declared.is_empty() || declared.len() != underlyings.len() {
            return Err(GraphProductError::Invalid(
                "graph underlyings must be nonempty and unique",
            ));
        }
        if payment_dates.is_empty() || payment_dates.len() != graph.outputs().len() {
            return Err(GraphProductError::Invalid(
                "one payment date is required per graph output",
            ));
        }
        let compiled = graph.compile(GraphLimitPolicy::DEFAULT)?;
        let nodes: BTreeMap<_, _> = graph.nodes().iter().map(|n| (n.id(), n.opcode())).collect();
        for node in graph.nodes() {
            if let SourceOpcode::TerminalSpot { underlying, .. }
            | SourceOpcode::PreDividendSpot { underlying, .. } = node.opcode()
                && !declared.contains(&underlying)
            {
                return Err(GraphProductError::Invalid(
                    "graph contains an undeclared underlying",
                ));
            }
        }
        // The compiler supplies a validated topological order. One shared data-flow
        // pass checks all outputs without recompiling the full graph per cash flow.
        let mut order = compiled.source_to_slot().to_vec();
        order.sort_unstable_by_key(|(_, slot)| *slot);
        let mut latest: BTreeMap<NodeId, Option<Date>> = BTreeMap::new();
        for (id, _) in order {
            let opcode = nodes[&id];
            let date = match opcode {
                SourceOpcode::TerminalSpot {
                    observation_date, ..
                }
                | SourceOpcode::PreDividendSpot {
                    observation_date, ..
                } => Some(observation_date),
                _ => {
                    let (operands, count) = opcode.operands();
                    operands
                        .into_iter()
                        .take(count)
                        .filter_map(|operand| latest[&operand])
                        .max()
                }
            };
            latest.insert(id, date);
        }
        for (&output, &payment) in graph.outputs().iter().zip(&payment_dates) {
            if latest[&output].is_some_and(|date| date > payment) {
                return Err(GraphProductError::Invalid(
                    "cash flow depends on an observation after its payment",
                ));
            }
        }
        let observations = compiled.terminal_observations();
        let last_payment = *payment_dates.iter().max().expect("nonempty payment dates");
        let expiry = observations
            .iter()
            .map(|(_, date)| *date)
            .max()
            .unwrap_or(last_payment);
        let pathwise_risk = !compiled
            .opcodes()
            .iter()
            .any(|opcode| matches!(opcode, CompiledOpcode::Indicator { .. }));
        Ok(Self {
            currency,
            underlyings,
            graph,
            payment_dates,
            observations,
            expiry,
            last_payment,
            pathwise_risk,
        })
    }
    #[must_use]
    pub const fn currency(&self) -> CurrencyId {
        self.currency
    }
    #[must_use]
    pub fn underlyings(&self) -> &[UnderlyingId] {
        &self.underlyings
    }
    pub(crate) const fn primary_underlying(&self) -> UnderlyingId {
        self.underlyings.as_slice()[0]
    }
    #[must_use]
    pub const fn source_graph(&self) -> &SourceGraph {
        &self.graph
    }
    #[must_use]
    pub fn payment_dates(&self) -> &[Date] {
        &self.payment_dates
    }
    #[must_use]
    pub fn observations(&self) -> &[(UnderlyingId, Date)] {
        &self.observations
    }
    #[must_use]
    pub const fn expiry(&self) -> Date {
        self.expiry
    }
    #[must_use]
    pub const fn last_payment_date(&self) -> Date {
        self.last_payment
    }
    #[must_use]
    pub const fn supports_pathwise_risk(&self) -> bool {
        self.pathwise_risk
    }
    #[must_use]
    pub fn has_multiple_payment_dates(&self) -> bool {
        self.payment_dates
            .iter()
            .any(|date| *date != self.last_payment)
    }
}
