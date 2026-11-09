//! Contract lowering into the existing graph and native pricing task adapters.
use super::compile::{encode_source_opcode, fold_node};
use crate::contract::{
    Aggregate, BooleanExpr, Cashflow, Comparison, Contract, ContractError, DateExpr, NumericExpr,
    Observation, ObservationKind, Term,
};
use crate::core::{CurrencyId, Date, FiniteF64, NodeId, UnderlyingId};
use crate::product::{
    CompiledPayoff, GraphComparison, GraphLimitPolicy, GraphProduct, ProductSpec,
    SourceGraphBuilder, SourceOpcode as Op,
};
use crate::{
    Fingerprint, MultiAssetSpecification, PricingOperation, PricingSource, PricingSpecification,
    PricingTask,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct CompiledCurrencyContract {
    currency: CurrencyId,
    cashflow_ids: Vec<String>,
    product: GraphProduct,
    payoff: CompiledPayoff,
}
impl CompiledCurrencyContract {
    pub fn currency(&self) -> CurrencyId {
        self.currency
    }
    pub fn cashflow_ids(&self) -> &[String] {
        &self.cashflow_ids
    }
    pub fn product(&self) -> &GraphProduct {
        &self.product
    }
    pub fn payoff(&self) -> &CompiledPayoff {
        &self.payoff
    }
}
#[derive(Clone, Debug)]
pub struct CompiledContract {
    source_fingerprint: Fingerprint,
    as_of: Date,
    groups: Vec<CompiledCurrencyContract>,
}
impl CompiledContract {
    pub fn source_fingerprint(&self) -> Fingerprint {
        self.source_fingerprint
    }
    pub fn as_of(&self) -> Date {
        self.as_of
    }
    pub fn groups(&self) -> &[CompiledCurrencyContract] {
        &self.groups
    }
    fn product_for(
        &self,
        currency: CurrencyId,
        underlyings: Vec<UnderlyingId>,
    ) -> Result<GraphProduct, ContractError> {
        if self.groups.len() > 1 {
            return Err(ContractError::Unsupported(
                "multiple currencies require separate native tasks from groups()".into(),
            ));
        }
        if self.groups.is_empty() {
            let mut b = SourceGraphBuilder::new();
            let zero = b.literal(0.0)?;
            return GraphProduct::new(
                currency,
                underlyings,
                b.finish(vec![zero]),
                vec![self.as_of],
            )
            .map_err(Into::into);
        }
        self.groups
            .iter()
            .find(|g| g.currency == currency)
            .map(|g| g.product.clone())
            .ok_or_else(|| {
                ContractError::Invalid("no contract cashflows in template currency".into())
            })
    }
    /// Replace only the product on an existing complete MC source. Model,
    /// measure, curve, seed, grids, execution and risk configuration are retained.
    /// One call prices one currency; no implicit FX conversion is performed.
    pub fn pricing_task(
        &self,
        template: &PricingSpecification,
        operation: PricingOperation,
    ) -> Result<PricingTask, ContractError> {
        if template.valuation_date() != self.as_of {
            return Err(ContractError::Invalid(
                "contract and template valuation dates differ".into(),
            ));
        }
        let product = self.product_for(
            template.product().currency(),
            vec![template.product().underlying()],
        )?;
        let source = PricingSpecification::new(
            self.as_of,
            ProductSpec::Graph(product),
            template.market().clone(),
            template.model().clone(),
            template.engine(),
            template.risk().clone(),
            template.numerics().clone(),
        )
        .map_err(|e| ContractError::Unsupported(e.to_string()))?;
        PricingTask::new(PricingSource::SingleAsset(source), operation)
            .map_err(|e| ContractError::Unsupported(e.to_string()))
    }
    pub fn stochastic_dividend_task(
        &self,
        template: &crate::StochasticDividendSpecification,
        operation: PricingOperation,
    ) -> Result<PricingTask, ContractError> {
        let task = self.pricing_task(template.equity(), PricingOperation::Evaluate)?;
        let PricingSource::SingleAsset(equity) = task.source() else {
            unreachable!()
        };
        let source = crate::StochasticDividendSpecification::new(
            equity.clone(),
            template.dividends(),
            template.volatility_correlations().to_vec(),
            template.rate_correlation(),
            template.maximum_step(),
        )
        .map_err(|e| ContractError::Unsupported(e.to_string()))?;
        PricingTask::new(PricingSource::StochasticDividend(source), operation)
            .map_err(|e| ContractError::Unsupported(e.to_string()))
    }

    pub fn multi_asset_task(
        &self,
        template: &MultiAssetSpecification,
        operation: PricingOperation,
    ) -> Result<PricingTask, ContractError> {
        if template.valuation_date() != self.as_of {
            return Err(ContractError::Invalid(
                "contract and template valuation dates differ".into(),
            ));
        }
        let product = self.product_for(
            template.product().currency(),
            template.product().underlyings().to_vec(),
        )?;
        let source = MultiAssetSpecification::new(
            self.as_of,
            product,
            template.markets().to_vec(),
            template.model().clone(),
            template.engine(),
            template.execution(),
            template.maximum_step(),
        )
        .map_err(|e| ContractError::Unsupported(e.to_string()))?;
        PricingTask::new(PricingSource::MultiAsset(source), operation)
            .map_err(|e| ContractError::Unsupported(e.to_string()))
    }
}
impl Contract {
    /// Lower a source or lifecycle residual. The universe is the ordered model
    /// underlyings, including an anchor for purely fixed cashflows. Historical
    /// observations must first be reduced with explicit fixing events.
    pub fn compile_mc(
        &self,
        as_of: Date,
        underlyings: Vec<UnderlyingId>,
    ) -> Result<CompiledContract, ContractError> {
        let analysis = self.analyze()?;
        let universe: BTreeSet<_> = underlyings.iter().copied().collect();
        if universe.len() != underlyings.len() {
            return Err(ContractError::Invalid("duplicate model underlying".into()));
        }
        // Validate observation support on live lowered nodes below, so a known
        // dead branch does not create a model requirement.
        let registry = self
            .observations()
            .iter()
            .map(|o| (o.id.clone(), o))
            .collect();
        let mut lower = Lowerer {
            builder: SourceGraphBuilder::new(),
            intern: BTreeMap::new(),
            folded: BTreeMap::new(),
            registry,
            as_of,
            universe,
            observations: BTreeMap::new(),
            flows: BTreeMap::new(),
        };
        let one = lower.literal(1.0)?;
        lower.term(self.root(), one, one)?;
        let mut groups = Vec::new();
        for (currency, flows) in &lower.flows {
            let outputs = flows.iter().map(|(id, _)| *id).collect();
            let graph = lower.builder.clone().finish(outputs);
            let product = GraphProduct::new(
                *currency,
                underlyings.clone(),
                graph,
                flows.iter().map(|(_, c)| c.payment_date).collect(),
            )?;
            let payoff = product.source_graph().compile(GraphLimitPolicy::DEFAULT)?;
            groups.push(CompiledCurrencyContract {
                currency: *currency,
                cashflow_ids: flows.iter().map(|(_, c)| c.id.clone()).collect(),
                product,
                payoff,
            });
        }
        let _ = analysis; // Full source validation precedes lowering.
        Ok(CompiledContract {
            source_fingerprint: self.fingerprint()?,
            as_of,
            groups,
        })
    }
}
struct Lowerer<'a> {
    builder: SourceGraphBuilder,
    intern: BTreeMap<Vec<u8>, NodeId>,
    folded: BTreeMap<NodeId, Option<FiniteF64>>,
    registry: BTreeMap<String, &'a Observation>,
    as_of: Date,
    universe: BTreeSet<UnderlyingId>,
    observations: BTreeMap<(UnderlyingId, Date, ObservationKind), String>,
    flows: BTreeMap<CurrencyId, Vec<(NodeId, Cashflow)>>,
}
impl Lowerer<'_> {
    fn push(&mut self, mut op: Op) -> Result<NodeId, ContractError> {
        let candidate = NodeId::new(
            u32::try_from(self.folded.len())
                .map_err(|_| ContractError::Invalid("contract graph too large".into()))?,
        );
        // Invalid arithmetic in a future unselected branch remains a runtime
        // operation. The native compiler/evaluator enforces its selected domain.
        let constant = match fold_node(candidate, op, &self.folded) {
            Ok(v) => v,
            Err(crate::product::GraphError::NonFiniteConstantFold { .. }) => None,
            Err(e) => return Err(e.into()),
        };
        if let Some(v) = constant {
            op = Op::Literal(v);
        }
        let mut key = Vec::new();
        encode_source_opcode(&mut key, op);
        if let Some(id) = self.intern.get(&key) {
            return Ok(*id);
        }
        let id = self.builder.push(op)?;
        self.intern.insert(key, id);
        self.folded.insert(id, constant);
        Ok(id)
    }
    fn literal(&mut self, v: f64) -> Result<NodeId, ContractError> {
        self.push(Op::Literal(
            FiniteF64::new(v, "contract constant")
                .map_err(|e| ContractError::Invalid(e.to_string()))?,
        ))
    }
    fn value(&self, id: NodeId) -> Option<f64> {
        self.folded[&id].map(FiniteF64::get)
    }
    fn select(&mut self, c: NodeId, a: NodeId, b: NodeId) -> Result<NodeId, ContractError> {
        if let Some(v) = self.value(c) {
            Ok(if v != 0.0 { a } else { b })
        } else {
            self.push(Op::Select {
                condition: c,
                then_value: a,
                else_value: b,
            })
        }
    }
    fn observation(&mut self, id: &str) -> Result<NodeId, ContractError> {
        let o = self
            .registry
            .get(id)
            .ok_or_else(|| ContractError::Invalid(format!("unknown observation {id}")))?;
        if o.date < self.as_of {
            return Err(ContractError::MissingFixing(format!(
                "{} from {} on {}",
                o.id, o.fixing_source, o.date
            )));
        }
        if !self.universe.contains(&o.underlying) {
            return Err(ContractError::Unsupported(format!(
                "model lacks underlying {}",
                o.underlying
            )));
        }
        if o.time.is_some() {
            return Err(ContractError::Unsupported(
                "native MC grid has no intraday observation mapping".into(),
            ));
        }
        if o.kind == ObservationKind::Rate {
            return Err(ContractError::Unsupported(
                "native graph does not expose a rate fixing observation".into(),
            ));
        }
        let key = (o.underlying, o.date, o.kind);
        if let Some(source) = self.observations.insert(key, o.fixing_source.clone())
            && source != o.fixing_source
        {
            return Err(ContractError::Unsupported(
                "distinct future fixing sources map to the same native observation".into(),
            ));
        }
        self.push(match o.kind {
            ObservationKind::Spot => Op::TerminalSpot {
                underlying: o.underlying,
                observation_date: o.date,
            },
            ObservationKind::PreDividendSpot => Op::PreDividendSpot {
                underlying: o.underlying,
                observation_date: o.date,
            },
            ObservationKind::Rate => unreachable!(),
        })
    }
    fn numeric(&mut self, e: &NumericExpr) -> Result<NodeId, ContractError> {
        Ok(match e {
            NumericExpr::Constant { value } => self.literal(*value)?,
            NumericExpr::Observation { id } => self.observation(id)?,
            NumericExpr::Add { left, right }
            | NumericExpr::Subtract { left, right }
            | NumericExpr::Multiply { left, right }
            | NumericExpr::Divide { left, right }
            | NumericExpr::Minimum { left, right }
            | NumericExpr::Maximum { left, right } => {
                let a = self.numeric(left)?;
                let b = self.numeric(right)?;
                self.push(match e {
                    NumericExpr::Add { .. } => Op::Add { left: a, right: b },
                    NumericExpr::Subtract { .. } => Op::Subtract { left: a, right: b },
                    NumericExpr::Multiply { .. } => Op::Multiply { left: a, right: b },
                    NumericExpr::Divide { .. } => Op::Divide {
                        numerator: a,
                        denominator: b,
                    },
                    NumericExpr::Minimum { .. } => Op::Minimum { left: a, right: b },
                    _ => Op::Maximum { left: a, right: b },
                })?
            }
            NumericExpr::Abs { input } => {
                let a = self.numeric(input)?;
                let neg = self.push(Op::Negate { input: a })?;
                self.push(Op::Maximum {
                    left: a,
                    right: neg,
                })?
            }
            NumericExpr::Negate { input } => {
                let input = self.numeric(input)?;
                self.push(Op::Negate { input })?
            }
            NumericExpr::Conditional {
                condition,
                then_value,
                else_value,
            } => {
                let c = self.boolean(condition)?;
                if let Some(v) = self.value(c) {
                    self.numeric(if v != 0.0 { then_value } else { else_value })?
                } else {
                    let a = self.numeric(then_value)?;
                    let b = self.numeric(else_value)?;
                    self.select(c, a, b)?
                }
            }
            NumericExpr::Aggregate { operation, values } => {
                let mut ids = Vec::with_capacity(values.len());
                for v in values {
                    ids.push(self.numeric(v)?);
                }
                let mut result = ids[0];
                if *operation == Aggregate::GeometricAverage {
                    result = self.push(Op::Log { input: result })?;
                }
                if matches!(
                    operation,
                    Aggregate::ArithmeticAverage | Aggregate::GeometricAverage
                ) {
                    let n = self.literal(values.len() as f64)?;
                    result = self.push(Op::Divide {
                        numerator: result,
                        denominator: n,
                    })?;
                    for &id in &ids[1..] {
                        let v = if *operation == Aggregate::GeometricAverage {
                            self.push(Op::Log { input: id })?
                        } else {
                            id
                        };
                        let v = self.push(Op::Divide {
                            numerator: v,
                            denominator: n,
                        })?;
                        result = self.push(Op::Add {
                            left: result,
                            right: v,
                        })?;
                    }
                    if *operation == Aggregate::GeometricAverage {
                        result = self.push(Op::Exp { input: result })?;
                    }
                } else {
                    for &id in &ids[1..] {
                        result = self.push(if *operation == Aggregate::RunningMinimum {
                            Op::Minimum {
                                left: result,
                                right: id,
                            }
                        } else {
                            Op::Maximum {
                                left: result,
                                right: id,
                            }
                        })?;
                    }
                }
                result
            }
        })
    }
    fn boolean(&mut self, e: &BooleanExpr) -> Result<NodeId, ContractError> {
        Ok(match e {
            BooleanExpr::Constant { value } => self.literal(if *value { 1.0 } else { 0.0 })?,
            BooleanExpr::Compare {
                operation,
                left,
                right,
            } => {
                let a = self.numeric(left)?;
                let b = self.numeric(right)?;
                self.push(Op::Compare {
                    left: a,
                    right: b,
                    operation: comparison(*operation),
                })?
            }
            BooleanExpr::DateCompare {
                operation,
                left,
                right,
            } => {
                let a = self.date(left)?;
                let b = self.date(right)?;
                self.push(Op::Compare {
                    left: a,
                    right: b,
                    operation: comparison(*operation),
                })?
            }
            BooleanExpr::Not { input } => {
                let c = self.boolean(input)?;
                let one = self.literal(1.0)?;
                let zero = self.literal(0.0)?;
                self.select(c, zero, one)?
            }
            BooleanExpr::And { left, right } | BooleanExpr::Or { left, right } => {
                let a = self.boolean(left)?;
                let and = matches!(e, BooleanExpr::And { .. });
                if self.value(a) == Some(if and { 0.0 } else { 1.0 }) {
                    a
                } else {
                    let b = self.boolean(right)?;
                    let fallback = self.literal(if and { 0.0 } else { 1.0 })?;
                    if and {
                        self.select(a, b, fallback)?
                    } else {
                        self.select(a, fallback, b)?
                    }
                }
            }
        })
    }
    fn date(&mut self, e: &DateExpr) -> Result<NodeId, ContractError> {
        match e {
            DateExpr::Constant { value } => self.literal(f64::from(Date::MIN.days_until(*value))),
            DateExpr::Conditional {
                condition,
                then_value,
                else_value,
            } => {
                let c = self.boolean(condition)?;
                if let Some(v) = self.value(c) {
                    self.date(if v != 0.0 { then_value } else { else_value })
                } else {
                    let a = self.date(then_value)?;
                    let b = self.date(else_value)?;
                    self.select(c, a, b)
                }
            }
        }
    }
    fn term(&mut self, t: &Term, gate: NodeId, scale: NodeId) -> Result<(), ContractError> {
        if self.value(gate) == Some(0.0) {
            return Ok(());
        }
        match t {
            Term::Empty => {}
            Term::Cashflow { cashflow: c } => {
                if c.payment_date < self.as_of {
                    return Err(ContractError::Unsupported(format!(
                        "overdue payment {} requires settlement/explicit closeout handling",
                        c.id
                    )));
                }
                let mut amount = self.numeric(&c.amount)?;
                if self.value(scale) != Some(1.0) {
                    amount = self.push(Op::Multiply {
                        left: amount,
                        right: scale,
                    })?;
                }
                if c.direction.sign() < 0.0 {
                    amount = self.push(Op::Negate { input: amount })?;
                }
                let zero = self.literal(0.0)?;
                let output = self.select(gate, amount, zero)?;
                self.flows
                    .entry(c.currency)
                    .or_default()
                    .push((output, c.clone()));
            }
            Term::Bundle { contracts } => {
                for t in contracts {
                    self.term(t, gate, scale)?;
                }
            }
            Term::Reverse { contract } => {
                let s = self.push(Op::Negate { input: scale })?;
                self.term(contract, gate, s)?;
            }
            Term::Scale { factor, contract } => {
                let f = self.numeric(factor)?;
                let s = self.push(Op::Multiply {
                    left: scale,
                    right: f,
                })?;
                self.term(contract, gate, s)?;
            }
            Term::Conditional {
                condition,
                then_contract,
                else_contract,
            } => {
                let c = self.boolean(condition)?;
                let zero = self.literal(0.0)?;
                let a = self.select(gate, c, zero)?;
                let one = self.literal(1.0)?;
                let not_c = self.select(c, zero, one)?;
                let b = self.select(gate, not_c, zero)?;
                self.term(then_contract, a, scale)?;
                self.term(else_contract, b, scale)?;
            }
            Term::Barrier {
                observations,
                direction,
                level,
                hit,
                on_hit,
                on_no_hit,
                ..
            } => {
                let c = Term::barrier_condition(observations, *direction, *level, *hit);
                self.term(
                    &Term::conditional(c, *on_hit.clone(), *on_no_hit.clone()),
                    gate,
                    scale,
                )?;
            }
            Term::VanillaOption { .. } => return Err(ContractError::Unsupported("exercise policies and continuous monitoring require a native option task; use Contract::pricing_task".into())),
            Term::Exercise { .. } | Term::ContinuousBarrier { .. } => {
                return Err(ContractError::Unsupported(
                    "incomplete exercise or monitoring terms".into(),
                ));
            }
        }
        Ok(())
    }
}
fn comparison(c: Comparison) -> GraphComparison {
    match c {
        Comparison::Less => GraphComparison::Less,
        Comparison::LessEqual => GraphComparison::LessEqual,
        Comparison::Equal => GraphComparison::Equal,
        Comparison::NotEqual => GraphComparison::NotEqual,
        Comparison::GreaterEqual => GraphComparison::GreaterEqual,
        Comparison::Greater => GraphComparison::Greater,
    }
}
