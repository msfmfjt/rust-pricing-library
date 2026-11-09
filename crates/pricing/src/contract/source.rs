use super::{BooleanExpr, Comparison, NumericExpr};
use crate::core::{CurrencyId, Date, UnderlyingId};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContractError {
    Invalid(String),
    MissingFixing(String),
    Unsupported(String),
    EventConflict(String),
    Graph(String),
    Wire(String),
}
impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid contract: {s}"),
            Self::MissingFixing(s) => write!(f, "missing historical fixing: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported contract feature: {s}"),
            Self::EventConflict(s) => write!(f, "event conflict: {s}"),
            Self::Graph(s) => write!(f, "contract graph: {s}"),
            Self::Wire(s) => write!(f, "contract JSON: {s}"),
        }
    }
}
impl Error for ContractError {}
impl From<crate::product::GraphError> for ContractError {
    fn from(e: crate::product::GraphError) -> Self {
        Self::Graph(e.to_string())
    }
}
impl From<crate::product::GraphProductError> for ContractError {
    fn from(e: crate::product::GraphProductError) -> Self {
        Self::Graph(e.to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationKind {
    Spot,
    PreDividendSpot,
    Rate,
}
/// Published fixing source is part of identity. Optional clock/calendar metadata
/// is preserved even when a particular MC model cannot simulate it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub id: String,
    #[serde(with = "super::wire::underlying")]
    pub underlying: UnderlyingId,
    pub kind: ObservationKind,
    #[serde(with = "super::wire::date")]
    pub date: Date,
    pub fixing_source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calendar: Option<String>,
}
impl Observation {
    pub fn spot(
        id: impl Into<String>,
        underlying: UnderlyingId,
        date: Date,
        fixing_source: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            underlying,
            kind: ObservationKind::Spot,
            date,
            fixing_source: fixing_source.into(),
            time: None,
            timezone: None,
            calendar: None,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaymentDirection {
    Receive,
    Pay,
}
impl PaymentDirection {
    pub fn sign(self) -> f64 {
        if self == Self::Receive { 1.0 } else { -1.0 }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cashflow {
    pub id: String,
    #[serde(with = "super::wire::currency")]
    pub currency: CurrencyId,
    pub amount: NumericExpr,
    pub direction: PaymentDirection,
    #[serde(with = "super::wire::date")]
    pub determination_date: Date,
    #[serde(with = "super::wire::date")]
    pub payment_date: Date,
    #[serde(with = "super::wire::date")]
    pub settlement_date: Date,
}
impl Cashflow {
    pub fn new(
        id: impl Into<String>,
        currency: CurrencyId,
        amount: NumericExpr,
        determination_date: Date,
        payment_date: Date,
    ) -> Self {
        Self {
            id: id.into(),
            currency,
            amount,
            direction: PaymentDirection::Receive,
            determination_date,
            payment_date,
            settlement_date: payment_date,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BarrierDirection {
    Up,
    Down,
}
/// Contractual composition. Conditional is an observable fact, never an exercise
/// optimization. Barrier branches are selected after discrete monitoring.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Term {
    Empty,
    VanillaOption {
        option: super::VanillaOption,
    },
    Cashflow {
        cashflow: Cashflow,
    },
    Bundle {
        contracts: Vec<Self>,
    },
    Conditional {
        condition: BooleanExpr,
        then_contract: Box<Self>,
        else_contract: Box<Self>,
    },
    Reverse {
        contract: Box<Self>,
    },
    Scale {
        factor: NumericExpr,
        contract: Box<Self>,
    },
    Barrier {
        id: String,
        observations: Vec<String>,
        direction: BarrierDirection,
        level: f64,
        hit: bool,
        on_hit: Box<Self>,
        on_no_hit: Box<Self>,
    },
    Exercise {
        id: String,
        #[serde(with = "super::wire::date")]
        date: Date,
        contract: Box<Self>,
    },
    ContinuousBarrier {
        id: String,
        contract: Box<Self>,
    },
}
impl Term {
    pub(super) fn visit_options<'a>(&'a self, output: &mut Vec<&'a super::VanillaOption>) {
        match self {
            Self::VanillaOption { option } => output.push(option),
            Self::Bundle { contracts } => contracts
                .iter()
                .for_each(|child| child.visit_options(output)),
            Self::Conditional {
                then_contract,
                else_contract,
                ..
            } => {
                then_contract.visit_options(output);
                else_contract.visit_options(output);
            }
            Self::Barrier {
                on_hit, on_no_hit, ..
            } => {
                on_hit.visit_options(output);
                on_no_hit.visit_options(output);
            }
            Self::Scale { contract, .. } | Self::Reverse { contract } => {
                contract.visit_options(output)
            }
            _ => {}
        }
    }
}
impl From<Cashflow> for Term {
    fn from(cashflow: Cashflow) -> Self {
        Self::Cashflow { cashflow }
    }
}
impl Term {
    pub fn conditional(condition: BooleanExpr, then_contract: Self, else_contract: Self) -> Self {
        Self::Conditional {
            condition,
            then_contract: Box::new(then_contract),
            else_contract: Box::new(else_contract),
        }
    }
    pub fn bundle(contracts: Vec<Self>) -> Self {
        Self::Bundle { contracts }
    }
    pub fn reverse(self) -> Self {
        Self::Reverse {
            contract: Box::new(self),
        }
    }
    pub fn scale(self, factor: NumericExpr) -> Self {
        Self::Scale {
            factor,
            contract: Box::new(self),
        }
    }
    pub fn barrier_condition(
        observations: &[String],
        direction: BarrierDirection,
        level: f64,
        hit: bool,
    ) -> BooleanExpr {
        if hit {
            return BooleanExpr::from(true);
        }
        let mut level_nodes: Vec<_> = observations
            .iter()
            .map(|id| {
                NumericExpr::observation(id).compare(
                    if direction == BarrierDirection::Up {
                        Comparison::GreaterEqual
                    } else {
                        Comparison::LessEqual
                    },
                    level.into(),
                )
            })
            .collect();
        if level_nodes.is_empty() {
            return BooleanExpr::from(false);
        }
        while level_nodes.len() > 1 {
            let mut next = Vec::with_capacity(level_nodes.len().div_ceil(2));
            let mut iter = level_nodes.into_iter();
            while let Some(left) = iter.next() {
                next.push(if let Some(right) = iter.next() {
                    left.or(right)
                } else {
                    left
                });
            }
            level_nodes = next;
        }
        level_nodes.pop().unwrap()
    }
}
/// The only editable contractual source. All analysis, residuals and MC graphs
/// are derived from this validated value.
#[derive(Clone, Debug, PartialEq)]
pub struct Contract {
    pub(crate) id: String,
    pub(crate) observations: Vec<Observation>,
    pub(crate) root: Term,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ContractAnalysis {
    pub required_observations: Vec<Observation>,
    pub potential_cashflows: Vec<Cashflow>,
    pub determined_cashflows: Vec<Cashflow>,
    pub underlying_dependencies: Vec<UnderlyingId>,
    pub currencies: Vec<CurrencyId>,
    pub barrier_monitoring_dates: Vec<Date>,
    /// Exercise dates or the legal American window, independent of the MC grid.
    pub option_features: Vec<super::OptionSchedule>,
    pub horizon: Option<Date>,
}
impl Contract {
    pub fn new(
        id: impl Into<String>,
        mut observations: Vec<Observation>,
        root: Term,
    ) -> Result<Self, ContractError> {
        observations.sort_by(|a, b| a.id.cmp(&b.id));
        let result = Self {
            id: id.into(),
            observations,
            root,
        };
        result.analyze()?;
        Ok(result)
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }
    pub fn root(&self) -> &Term {
        &self.root
    }
    pub fn validate(&self) -> Result<(), ContractError> {
        self.analyze().map(|_| ())
    }
    pub fn analyze(&self) -> Result<ContractAnalysis, ContractError> {
        identifier(&self.id)?;
        if self.observations.len() > 100_000 {
            return Err(ContractError::Invalid("too many observations".into()));
        }
        let mut registry = BTreeMap::new();
        for o in &self.observations {
            identifier(&o.id)?;
            identifier(&o.fixing_source)?;
            if registry.insert(o.id.clone(), o).is_some() {
                return Err(ContractError::Invalid(format!(
                    "duplicate observation {}",
                    o.id
                )));
            }
            for s in [&o.timezone, &o.calendar].into_iter().flatten() {
                identifier(s)?;
            }
            if o.time.is_some() != o.timezone.is_some() {
                return Err(ContractError::Invalid(
                    "observation time requires timezone and vice versa".into(),
                ));
            }
            if let Some(t) = &o.time {
                let parts: Vec<_> = t.split(':').collect();
                if parts.len() != 3
                    || parts
                        .iter()
                        .any(|s| s.len() != 2 || !s.bytes().all(|c| c.is_ascii_digit()))
                    || parts[0].parse::<u8>().unwrap_or(255) > 23
                    || parts[1].parse::<u8>().unwrap_or(255) > 59
                    || parts[2].parse::<u8>().unwrap_or(255) > 59
                {
                    return Err(ContractError::Invalid(
                        "observation time must be HH:MM:SS".into(),
                    ));
                }
            }
        }
        let mut state = AnalysisState {
            registry,
            used: BTreeSet::new(),
            ids: BTreeSet::new(),
            barrier_ids: BTreeSet::new(),
            barriers: BTreeSet::new(),
            options: Vec::new(),
            flows: Vec::new(),
            count: 0,
            prune_constants: false,
            analysis_nodes: 0,
        };
        state.walk(&self.root, &BTreeSet::new(), 0)?;
        // Validate the complete source, including inactive branches. Extract
        // actual requirements separately after deterministic simplification.
        state.analysis_nodes = state.count;
        state.used.clear();
        state.ids.clear();
        state.barrier_ids.clear();
        state.barriers.clear();
        state.options.clear();
        state.flows.clear();
        state.count = 0;
        state.prune_constants = true;
        state.walk(&self.root, &BTreeSet::new(), 0)?;
        let mut required_observations: Vec<_> = state
            .used
            .iter()
            .map(|id| state.registry[id].clone())
            .collect();
        required_observations.sort_by(|a, b| {
            (a.date, &a.time, a.underlying, &a.id).cmp(&(b.date, &b.time, b.underlying, &b.id))
        });
        let underlying_dependencies = required_observations
            .iter()
            .map(|o| o.underlying)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let currencies = state
            .flows
            .iter()
            .map(|c| c.currency)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let horizon = state
            .flows
            .iter()
            .map(|c| c.settlement_date)
            .chain(required_observations.iter().map(|o| o.date))
            .max();
        // A constant amount behind an unresolved predicate is potential, not determined.
        let determined_cashflows = unconditional_constants(&self.root)?;
        for cashflow in &determined_cashflows {
            if let Some(value) = cashflow.amount.constant() {
                super::expression::finite(value)?;
            }
        }
        Ok(ContractAnalysis {
            required_observations,
            potential_cashflows: state.flows,
            determined_cashflows,
            underlying_dependencies,
            currencies,
            barrier_monitoring_dates: state.barriers.into_iter().collect(),
            option_features: state.options,
            horizon,
        })
    }
}
pub(crate) fn identifier(s: &str) -> Result<(), ContractError> {
    if s.trim().is_empty() || s.len() > 256 || s.chars().any(char::is_control) {
        Err(ContractError::Invalid(
            "identifier must be nonblank, bounded and contain no controls".into(),
        ))
    } else {
        Ok(())
    }
}
struct AnalysisState<'a> {
    registry: BTreeMap<String, &'a Observation>,
    used: BTreeSet<String>,
    ids: BTreeSet<String>,
    barrier_ids: BTreeSet<String>,
    barriers: BTreeSet<Date>,
    options: Vec<super::OptionSchedule>,
    flows: Vec<Cashflow>,
    count: usize,
    prune_constants: bool,
    analysis_nodes: usize,
}
impl AnalysisState<'_> {
    fn check_refs(&mut self, refs: &BTreeSet<String>) -> Result<(), ContractError> {
        for id in refs {
            if !self.registry.contains_key(id) {
                return Err(ContractError::Invalid(format!("unknown observation {id}")));
            }
            self.used.insert(id.clone());
        }
        Ok(())
    }
    fn walk(
        &mut self,
        t: &Term,
        guards: &BTreeSet<String>,
        depth: usize,
    ) -> Result<(), ContractError> {
        super::expression::budget(depth, &mut self.count)?;
        match t {
            Term::Empty => {}
            Term::Cashflow { cashflow: c } => {
                identifier(&c.id)?;
                if !self.ids.insert(c.id.clone()) {
                    return Err(ContractError::Invalid(format!(
                        "duplicate cashflow {}",
                        c.id
                    )));
                }
                if c.determination_date > c.payment_date || c.payment_date > c.settlement_date {
                    return Err(ContractError::Invalid(format!(
                        "cashflow {} has inconsistent dates",
                        c.id
                    )));
                }
                let mut refs = guards.clone();
                if self.prune_constants {
                    c.amount.reduce(&BTreeMap::new())?.refs(
                        &mut refs,
                        depth + 1,
                        &mut self.count,
                    )?;
                } else {
                    c.amount.refs(&mut refs, depth + 1, &mut self.count)?;
                }
                self.check_refs(&refs)?;
                if refs
                    .iter()
                    .any(|id| self.registry[id].date > c.determination_date)
                {
                    return Err(ContractError::Invalid(format!(
                        "cashflow {} determined before its observations",
                        c.id
                    )));
                }
                self.flows.push(c.clone());
            }
            Term::Bundle { contracts } => {
                for t in contracts {
                    self.walk(t, guards, depth + 1)?;
                }
            }
            Term::Conditional {
                condition,
                then_contract,
                else_contract,
            } => {
                let reduced = self
                    .prune_constants
                    .then(|| condition.reduce(&BTreeMap::new()))
                    .transpose()?;
                let condition = reduced.as_ref().unwrap_or(condition);
                if self.prune_constants
                    && let Some(value) = condition.constant()
                {
                    return self.walk(
                        if value { then_contract } else { else_contract },
                        guards,
                        depth + 1,
                    );
                }
                let mut refs = guards.clone();
                condition.refs(&mut refs, depth + 1, &mut self.count)?;
                self.check_refs(&refs)?;
                self.walk(then_contract, &refs, depth + 1)?;
                self.walk(else_contract, &refs, depth + 1)?;
            }
            Term::Reverse { contract } => {
                let start = self.flows.len();
                self.walk(contract, guards, depth + 1)?;
                if self.prune_constants {
                    for c in &mut self.flows[start..] {
                        c.direction = if c.direction == PaymentDirection::Pay {
                            PaymentDirection::Receive
                        } else {
                            PaymentDirection::Pay
                        };
                    }
                }
            }
            Term::Scale { factor, contract } => {
                let start = self.flows.len();
                let mut refs = guards.clone();
                let reduced = self
                    .prune_constants
                    .then(|| factor.reduce(&BTreeMap::new()))
                    .transpose()?;
                let factor = reduced.as_ref().unwrap_or(factor);
                factor.refs(&mut refs, depth + 1, &mut self.count)?;
                self.check_refs(&refs)?;
                self.walk(contract, &refs, depth + 1)?;
                if self.prune_constants {
                    // Bound amplification before cloning a factor into many payments.
                    let mut factor_nodes = 0;
                    factor.refs(&mut BTreeSet::new(), 0, &mut factor_nodes)?;
                    self.analysis_nodes = (self.flows.len() - start)
                        .checked_mul(factor_nodes + 1)
                        .and_then(|extra| self.analysis_nodes.checked_add(extra))
                        .ok_or_else(|| ContractError::Invalid("analysis size overflow".into()))?;
                    super::expression::budget(0, &mut self.analysis_nodes)?;
                    for c in &mut self.flows[start..] {
                        c.amount = c.amount.clone().times(factor.clone());
                    }
                }
            }
            Term::Barrier {
                id,
                observations,
                direction: _,
                level,
                hit,
                on_hit,
                on_no_hit,
            } => {
                if self.prune_constants && *hit {
                    return self.walk(on_hit, guards, depth + 1);
                }
                identifier(id)?;
                if !self.barrier_ids.insert(id.clone()) {
                    return Err(ContractError::Invalid(format!("duplicate barrier {id}")));
                }
                super::expression::finite(*level)?;
                if observations.is_empty() {
                    return Err(ContractError::Invalid(
                        "barrier requires monitoring observations".into(),
                    ));
                }
                let unique: BTreeSet<_> = observations.iter().cloned().collect();
                if unique.len() != observations.len() {
                    return Err(ContractError::Invalid(
                        "duplicate barrier observation".into(),
                    ));
                }
                let mut refs = guards.clone();
                refs.extend(unique);
                self.check_refs(&refs)?;
                let dates: Vec<_> = observations
                    .iter()
                    .map(|id| self.registry[id].date)
                    .collect();
                if dates.windows(2).any(|w| w[0] > w[1]) {
                    return Err(ContractError::Invalid(
                        "barrier schedule must be chronological".into(),
                    ));
                }
                self.barriers.extend(dates);
                self.walk(on_hit, &refs, depth + 1)?;
                self.walk(on_no_hit, &refs, depth + 1)?;
            }
            Term::VanillaOption { option } => {
                let expiry = *self
                    .registry
                    .get(&option.expiry_observation)
                    .ok_or_else(|| {
                        ContractError::Invalid("unknown option expiry observation".into())
                    })?;
                option.validate(expiry)?;
                let (first, payment) = match &option.feature {
                    super::OptionFeature::Bermudan { dates } => {
                        for &date in dates {
                            super::expression::budget(depth + 1, &mut self.count)?;
                            let obs = option.observation_on(expiry, date);
                            if self.registry.get(&obs.id).copied() != Some(&obs) {
                                return Err(ContractError::Invalid(format!(
                                    "missing or inconsistent exercise observation {}",
                                    obs.id
                                )));
                            }
                            self.used.insert(obs.id);
                        }
                        (dates[0], expiry.date)
                    }
                    super::OptionFeature::American { start } => {
                        let prefix = format!("{}/spot/", option.id);
                        identifier(&format!("{prefix}{start}"))?;
                        for (key, &registered) in self
                            .registry
                            .range(prefix.clone()..)
                            .take_while(|(key, _)| key.starts_with(&prefix))
                        {
                            if let Ok(date) = key[prefix.len()..].parse::<Date>()
                                && option.permits_exercise(date, expiry.date)
                                && option.observation_on(expiry, date) != *registered
                            {
                                return Err(ContractError::Invalid(format!(
                                    "exercise fixing identity collision for {key}"
                                )));
                            }
                        }
                        (*start, expiry.date)
                    }
                    super::OptionFeature::ContinuousBarrier(b) => {
                        self.barriers.extend([b.start, b.end]);
                        (expiry.date, b.payment_date)
                    }
                };
                if guards.iter().any(|id| self.registry[id].date > first) {
                    return Err(ContractError::Invalid(
                        "option depends on a guard observed after its first decision".into(),
                    ));
                }
                self.options.push(super::OptionSchedule {
                    option_id: option.id.clone(),
                    expiry_observation: expiry.id.clone(),
                    expiry: expiry.date,
                    feature: option.feature.clone(),
                });
                if self.prune_constants {
                    match &option.feature {
                        super::OptionFeature::Bermudan { dates } => {
                            // All alternatives settle the same right; at most one pays.
                            for &date in dates.iter().take(dates.len() - 1) {
                                self.flows.push(
                                    option.cashflow(&option.observation_on(expiry, date), date),
                                );
                            }
                        }
                        super::OptionFeature::ContinuousBarrier(b) => {
                            self.flows.push(Cashflow::new(
                                &option.id,
                                option.currency,
                                b.rebate.into(),
                                b.end,
                                b.payment_date,
                            ));
                        }
                        super::OptionFeature::American { .. } => {}
                    }
                }
                self.walk(
                    &Term::from(option.cashflow(expiry, payment)),
                    guards,
                    depth + 1,
                )?;
            }
            Term::Exercise { .. } => {
                return Err(ContractError::Unsupported(
                    "untyped exercise placeholder: specify a holder option and its legal schedule"
                        .into(),
                ));
            }
            Term::ContinuousBarrier { .. } => {
                return Err(ContractError::Unsupported(
                    "untyped continuous barrier placeholder: specify a monitoring window, level and knock style".into(),
                ));
            }
        }
        Ok(())
    }
}
fn unconditional_constants(t: &Term) -> Result<Vec<Cashflow>, ContractError> {
    let empty = BTreeMap::new();
    Ok(match t {
        Term::Cashflow { cashflow } => {
            if let Some(value) = cashflow.amount.reduce(&empty)?.constant() {
                let mut c = cashflow.clone();
                c.amount = value.into();
                vec![c]
            } else {
                vec![]
            }
        }
        Term::Bundle { contracts } => {
            let mut flows = Vec::new();
            for term in contracts {
                flows.extend(unconditional_constants(term)?);
            }
            flows
        }
        Term::Conditional {
            condition,
            then_contract,
            else_contract,
        } => {
            if let Some(value) = condition.reduce(&empty)?.constant() {
                unconditional_constants(if value { then_contract } else { else_contract })?
            } else {
                vec![]
            }
        }
        Term::Barrier {
            hit: true, on_hit, ..
        } => unconditional_constants(on_hit)?,
        Term::Reverse { contract } => unconditional_constants(contract)?
            .into_iter()
            .map(|mut c| {
                c.direction = if c.direction == PaymentDirection::Pay {
                    PaymentDirection::Receive
                } else {
                    PaymentDirection::Pay
                };
                c
            })
            .collect(),
        Term::Scale { factor, contract } => {
            if let Some(value) = factor.reduce(&empty)?.constant() {
                unconditional_constants(contract)?
                    .into_iter()
                    .map(|mut c| {
                        c.amount = NumericExpr::from(c.amount.constant().unwrap() * value);
                        c
                    })
                    .collect()
            } else {
                vec![]
            }
        }
        _ => vec![],
    })
}
