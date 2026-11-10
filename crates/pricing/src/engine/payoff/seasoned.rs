//! Freeze published observations, retain unpaid contractual outputs and prune
//! branches disabled by a known call or barrier before asking for more fixings.
use super::compile::{fold_node, reachable_nodes};
use crate::core::{Date, FiniteF64, UnderlyingId};
use crate::product::{
    GraphError, GraphLimitPolicy, GraphProduct, GraphProductError as Error, SourceGraph,
    SourceGraphBuilder, SourceNode, SourceOpcode as Op,
};
use std::collections::{BTreeMap, BTreeSet};

impl crate::product::ProductSpec {
    /// Export a dated payoff in its contractual amount currency. Bermudan
    /// exercise and continuously monitored barriers require their native engines.
    pub fn cashflow_graph(
        &self,
        as_of: Date,
        currency: Option<crate::core::CurrencyId>,
        payment: Option<Date>,
    ) -> Result<GraphProduct, Error> {
        use crate::product::{BarrierMonitoring, ProductSpec};
        if matches!(self, ProductSpec::AmericanVanilla(_))
            || matches!(self,ProductSpec::Barrier(b) if b.monitoring()==BarrierMonitoring::Continuous)
        {
            return Err(Error::Invalid(
                "scheduled exercise or continuous monitoring requires its native valuation",
            ));
        }
        if !matches!(self, ProductSpec::EuropeanVanilla(_)) && payment.is_some() {
            return Err(Error::Invalid(
                "only European vanilla accepts a separate payment date",
            ));
        }
        let currency = currency.unwrap_or(self.currency());
        if let ProductSpec::Graph(g) = self {
            return GraphProduct::new(
                currency,
                g.underlyings().to_vec(),
                g.source_graph().clone(),
                g.payment_dates().to_vec(),
            );
        }
        GraphProduct::new(
            currency,
            vec![self.underlying()],
            self.source_graph(as_of)?,
            vec![payment.unwrap_or_else(|| self.payment_date())],
        )
    }
}

impl GraphProduct {
    /// Spot fixings use the units of the graph observations (local or composite).
    /// Settlements are zero-based indices of the original outputs of this graph.
    pub fn remaining_cashflows(
        &self,
        as_of: Date,
        fixings: &[(UnderlyingId, Date, f64)],
        pre_dividend_fixings: &[(UnderlyingId, Date, f64)],
        settled_outputs: &[usize],
    ) -> Result<Self, Error> {
        let mut history = BTreeMap::new();
        for (pre, records) in [(false, fixings), (true, pre_dividend_fixings)] {
            for &(id, date, value) in records {
                let value = FiniteF64::new(value, "equity_fixing")
                    .map_err(|_| Error::Invalid("equity fixing must be finite"))?;
                if date > as_of || !self.underlyings().contains(&id) || value.get() < 0. {
                    return Err(Error::Invalid(
                        "require a nonnegative fixing for a declared stock on or before valuation",
                    ));
                }
                if history.insert((id, date, pre), value).is_some() {
                    return Err(Error::Invalid("duplicate historical equity fixing"));
                }
            }
        }
        let mut settled = BTreeSet::new();
        for &i in settled_outputs {
            if self.payment_dates().get(i).is_none_or(|p| *p > as_of) || !settled.insert(i) {
                return Err(Error::Invalid(
                    "settled outputs must be unique, due cash flows of this graph",
                ));
            }
        }
        let source = self.source_graph();
        let active: Vec<_> = source
            .outputs()
            .iter()
            .enumerate()
            .filter(|(i, _)| !settled.contains(i))
            .map(|(_, id)| *id)
            .collect();
        let mut nodes: BTreeMap<_, _> = source
            .nodes()
            .iter()
            .map(|n| (n.id(), n.opcode()))
            .collect();
        let needed = reachable_nodes(&nodes, &active);
        let tape = source.compile(GraphLimitPolicy::DEFAULT)?;
        let mut order = tape.source_to_slot().to_vec();
        order.sort_unstable_by_key(|&(_, slot)| slot);
        let mut constants: BTreeMap<_, Option<FiniteF64>> = BTreeMap::new();
        for (id, _) in order {
            if !needed.contains(&id) {
                continue;
            }
            let op = match nodes[&id] {
                op @ (Op::TerminalSpot {
                    underlying,
                    observation_date,
                }
                | Op::PreDividendSpot {
                    underlying,
                    observation_date,
                }) => history
                    .get(&(
                        underlying,
                        observation_date,
                        matches!(op, Op::PreDividendSpot { .. }),
                    ))
                    .map_or(op, |&v| Op::Literal(v)),
                Op::Select {
                    condition,
                    then_value,
                    else_value,
                } if constants[&condition].is_some() => {
                    let chosen = if constants[&condition].expect("known condition").get() != 0. {
                        then_value
                    } else {
                        else_value
                    };
                    nodes[&chosen]
                }
                Op::Multiply { left, right }
                    if [left, right]
                        .into_iter()
                        .any(|k| constants[&k].is_some_and(|v: FiniteF64| v.get() == 0.)) =>
                {
                    Op::Literal(FiniteF64::new(0., "zero").expect("finite zero"))
                }
                op => op,
            };
            let constant = match fold_node(id, op, &constants) {
                // Once branches have been specialized, normal graph compilation
                // checks errors only in the surviving source.
                Err(GraphError::NonFiniteConstantFold { .. }) => None,
                result => result?,
            };
            nodes.insert(id, constant.map_or(op, Op::Literal));
            constants.insert(id, constant);
        }
        let mut outputs = Vec::new();
        let mut payments = Vec::new();
        for (i, (&id, &pay)) in source
            .outputs()
            .iter()
            .zip(self.payment_dates())
            .enumerate()
        {
            if settled.contains(&i) || constants[&id].is_some_and(|v| v.get() == 0.) {
                continue;
            }
            if pay < as_of {
                return Err(Error::Invalid(
                    "overdue cash flow needs confirmed settlement or closeout",
                ));
            }
            outputs.push(id);
            payments.push(pay);
        }
        let graph = if outputs.is_empty() {
            let mut b = SourceGraphBuilder::new();
            let zero = b.literal(0.)?;
            payments.push(as_of);
            b.finish(vec![zero])
        } else {
            let live = reachable_nodes(&nodes, &outputs);
            let remaining = live
                .into_iter()
                .map(|id| SourceNode::new(id, nodes[&id]))
                .collect();
            SourceGraph::new(remaining, outputs)
        };
        let residual = Self::new(
            self.currency(),
            self.underlyings().to_vec(),
            graph,
            payments,
        )?;
        if residual.observations().iter().any(|&(_, d)| d < as_of) {
            return Err(Error::Invalid(
                "an outstanding cash flow needs a historical stock fixing",
            ));
        }
        Ok(residual)
    }
}
