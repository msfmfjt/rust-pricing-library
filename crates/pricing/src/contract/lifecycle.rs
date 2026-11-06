//! Deterministic, pure event replay. Checkpoints store source plus events only.
use super::{Cashflow, Contract, ContractError, NumericExpr, Observation, Term};
use crate::JsonLimits;
use crate::core::{CurrencyId, Date};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Fixing {
        observation: Observation,
        value: f64,
    },
    Settlement {
        cashflow_id: String,
    },
    /// Explicit confirmation of a coupon/payment amount derived from fixings.
    Determination {
        cashflow_id: String,
        #[serde(with = "super::wire::currency")]
        currency: CurrencyId,
        amount: f64,
    },
    /// Cancels obligations determined strictly after the termination date.
    /// Already determined obligations still require explicit settlement.
    Termination {
        contract_id: String,
    },
    Exercise {
        exercise_id: String,
    },
}
impl Event {
    fn priority(&self) -> u8 {
        match self {
            Self::Fixing { .. } => 0,
            Self::Determination { .. } => 1,
            Self::Termination { .. } => 2,
            Self::Settlement { .. } => 3,
            Self::Exercise { .. } => 4,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LifecycleEvent {
    pub id: String,
    #[serde(with = "super::wire::date")]
    pub effective_date: Date,
    pub event: Event,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeterminedCashflow {
    pub id: String,
    #[serde(with = "super::wire::currency")]
    pub currency: CurrencyId,
    pub amount: f64,
    #[serde(with = "super::wire::date")]
    pub determination_date: Date,
    #[serde(with = "super::wire::date")]
    pub payment_date: Date,
    #[serde(with = "super::wire::date")]
    pub settlement_date: Date,
    pub settled: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub struct LifecycleResult {
    original: Contract,
    events: Vec<LifecycleEvent>,
    as_of: Date,
    pub residual_contract: Contract,
    pub determined_cashflows: Vec<DeterminedCashflow>,
    pub outstanding_payments: Vec<DeterminedCashflow>,
    pub remaining_observations: Vec<Observation>,
    pub applied_event_ids: Vec<String>,
    pub barrier_states: BTreeMap<String, bool>,
}
impl Contract {
    pub fn advance(
        &self,
        events: &[LifecycleEvent],
        as_of: Date,
    ) -> Result<LifecycleResult, ContractError> {
        advance(self, events, as_of)
    }
}
impl LifecycleResult {
    pub fn as_of(&self) -> Date {
        self.as_of
    }
    pub fn events(&self) -> &[LifecycleEvent] {
        &self.events
    }
    pub fn advance(&self, events: &[LifecycleEvent], as_of: Date) -> Result<Self, ContractError> {
        if as_of < self.as_of {
            return Err(ContractError::EventConflict(
                "incremental lifecycle cannot move backwards".into(),
            ));
        }
        let mut all = self.events.clone();
        all.extend_from_slice(events);
        advance(&self.original, &all, as_of)
    }
    pub fn to_json(&self) -> Result<String, ContractError> {
        let contract = serde_json::from_str(&self.original.to_json()?).map_err(wire_error)?;
        serde_json::to_string(&Checkpoint {
            document_kind: "contract_lifecycle".into(),
            schema_version: 1,
            contract,
            events: self.events.clone(),
            as_of: self.as_of,
        })
        .map_err(wire_error)
    }
    pub fn from_json(input: &[u8], limits: JsonLimits) -> Result<Self, ContractError> {
        let text = crate::wire::validate_and_decode(input, limits)
            .map_err(|e| ContractError::Wire(e.to_string()))?;
        let saved: Checkpoint = serde_json::from_str(text).map_err(wire_error)?;
        if saved.document_kind != "contract_lifecycle" || saved.schema_version != 1 {
            return Err(ContractError::Wire("expected contract_lifecycle v1".into()));
        }
        let bytes = serde_json::to_vec(&saved.contract).map_err(wire_error)?;
        advance(
            &Contract::from_json(&bytes, limits)?,
            &saved.events,
            saved.as_of,
        )
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Checkpoint {
    document_kind: String,
    schema_version: u32,
    contract: serde_json::Value,
    events: Vec<LifecycleEvent>,
    #[serde(with = "super::wire::date")]
    as_of: Date,
}
fn wire_error(e: serde_json::Error) -> ContractError {
    ContractError::Wire(e.to_string())
}

pub fn advance(
    contract: &Contract,
    events: &[LifecycleEvent],
    as_of: Date,
) -> Result<LifecycleResult, ContractError> {
    contract.validate()?;
    if events.len() > 100_000 {
        return Err(ContractError::Invalid("too many lifecycle events".into()));
    }
    let mut unique = BTreeMap::new();
    for e in events {
        super::source::identifier(&e.id)?;
        if let Some(previous) = unique.insert(e.id.clone(), e.clone())
            && previous != *e
        {
            return Err(ContractError::EventConflict(format!(
                "event ID {} reused with different content",
                e.id
            )));
        }
        if let Event::Fixing { observation, value } = &e.event {
            super::expression::finite(*value)?;
            if contract
                .observations
                .iter()
                .find(|o| o.id == observation.id)
                != Some(observation)
            {
                return Err(ContractError::EventConflict(format!(
                    "fixing identity/source does not match {}",
                    observation.id
                )));
            }
            if e.effective_date < observation.date {
                return Err(ContractError::EventConflict(
                    "fixing published before observation".into(),
                ));
            }
        }
        match &e.event {
            Event::Termination { contract_id } if contract_id != contract.id() => {
                return Err(ContractError::EventConflict(
                    "termination refers to another contract".into(),
                ));
            }
            Event::Exercise { .. } => {
                return Err(ContractError::Unsupported(
                    "exercise event requires Phase 5 MC policy".into(),
                ));
            }
            Event::Determination { amount, .. } => {
                super::expression::finite(*amount)?;
            }
            _ => {}
        }
    }
    let mut ordered: Vec<_> = unique.into_values().collect();
    ordered.sort_by(|a, b| {
        (a.effective_date, a.event.priority(), &a.id).cmp(&(
            b.effective_date,
            b.event.priority(),
            &b.id,
        ))
    });
    let mut fixings = BTreeMap::new();
    let mut settled = BTreeSet::new();
    let mut termination: Option<Date> = None;
    let mut applied = Vec::new();
    for e in ordered.iter().filter(|e| e.effective_date <= as_of) {
        match &e.event {
            Event::Fixing { observation, value } => {
                if let Some(old) = fixings.insert(observation.id.clone(), *value)
                    && old != *value
                {
                    return Err(ContractError::EventConflict(format!(
                        "conflicting fixing for {}",
                        observation.id
                    )));
                }
            }
            Event::Termination { .. } => {
                termination =
                    Some(termination.map_or(e.effective_date, |d| d.min(e.effective_date)));
            }
            Event::Settlement { cashflow_id } | Event::Determination { cashflow_id, .. } => {
                let reduced = reduce_term(
                    &contract.root,
                    &fixings,
                    termination,
                    &mut BTreeMap::new(),
                    NumericExpr::from(1.0),
                )?;
                let amounts = collect_determined(&reduced, e.effective_date, &BTreeSet::new());
                let amount = amounts
                    .iter()
                    .find(|c| &c.id == cashflow_id)
                    .ok_or_else(|| {
                        ContractError::EventConflict(format!(
                            "{} is not a determined obligation at event date",
                            cashflow_id
                        ))
                    })?;
                if matches!(e.event, Event::Settlement { .. }) {
                    if e.effective_date < amount.settlement_date {
                        return Err(ContractError::EventConflict(
                            "settlement precedes contractual settlement date".into(),
                        ));
                    }
                    settled.insert(cashflow_id.clone());
                } else if let Event::Determination {
                    currency,
                    amount: confirmed,
                    ..
                } = &e.event
                    && (*currency != amount.currency || *confirmed != amount.amount)
                {
                    return Err(ContractError::EventConflict(
                        "determination differs from contractual fixing result".into(),
                    ));
                }
            }
            Event::Exercise { .. } => unreachable!("rejected above"),
        }
        applied.push(e.id.clone());
    }
    let mut barrier_states = BTreeMap::new();
    let reduced = reduce_term(
        &contract.root,
        &fixings,
        termination,
        &mut barrier_states,
        NumericExpr::from(1.0),
    )?;
    let determined_cashflows = collect_determined(&reduced, as_of, &settled);
    let outstanding_payments = determined_cashflows
        .iter()
        .filter(|c| !c.settled)
        .cloned()
        .collect();
    let root = remove_settled(reduced, &settled);
    let residual_contract = Contract::new(&contract.id, contract.observations.clone(), root)?;
    let remaining_observations = residual_contract.analyze()?.required_observations;
    if let Some(o) = remaining_observations.iter().find(|o| o.date < as_of) {
        return Err(ContractError::MissingFixing(format!(
            "{} from {} on {}",
            o.id, o.fixing_source, o.date
        )));
    }
    Ok(LifecycleResult {
        original: contract.clone(),
        events: ordered,
        as_of,
        residual_contract,
        determined_cashflows,
        outstanding_payments,
        remaining_observations,
        applied_event_ids: applied,
        barrier_states,
    })
}
fn bundle(contracts: Vec<Term>) -> Term {
    let mut live: Vec<_> = contracts
        .into_iter()
        .filter(|t| !matches!(t, Term::Empty))
        .collect();
    match live.len() {
        0 => Term::Empty,
        1 => live.pop().unwrap(),
        _ => Term::bundle(live),
    }
}
fn scaled(
    amount: NumericExpr,
    factor: NumericExpr,
    f: &BTreeMap<String, f64>,
) -> Result<NumericExpr, ContractError> {
    if factor.constant() == Some(1.0) {
        Ok(amount)
    } else {
        amount.times(factor).reduce(f)
    }
}
fn reduce_term(
    t: &Term,
    f: &BTreeMap<String, f64>,
    termination: Option<Date>,
    states: &mut BTreeMap<String, bool>,
    factor: NumericExpr,
) -> Result<Term, ContractError> {
    Ok(match t {
        Term::Empty => Term::Empty,
        Term::Cashflow { cashflow: c } => {
            if termination.is_some_and(|d| c.determination_date > d) {
                Term::Empty
            } else {
                let mut c = c.clone();
                c.amount = scaled(c.amount.reduce(f)?, factor, f)?;
                Term::from(c)
            }
        }
        Term::Bundle { contracts } => bundle(
            contracts
                .iter()
                .map(|t| reduce_term(t, f, termination, states, factor.clone()))
                .collect::<Result<_, _>>()?,
        ),
        Term::Reverse { contract } => reduce_term(
            contract,
            f,
            termination,
            states,
            scaled(factor, (-1.0).into(), f)?,
        )?,
        Term::Scale {
            factor: scale,
            contract,
        } => reduce_term(
            contract,
            f,
            termination,
            states,
            scaled(scale.reduce(f)?, factor, f)?,
        )?,
        Term::Conditional {
            condition,
            then_contract,
            else_contract,
        } => {
            // A wholly cancelled subtree has no remaining observation obligation.
            if termination.is_some_and(|d| all_cancelled(t, d)) {
                Term::Empty
            } else {
                let c = condition.reduce(f)?;
                if let Some(v) = c.constant() {
                    reduce_term(
                        if v { then_contract } else { else_contract },
                        f,
                        termination,
                        states,
                        factor,
                    )?
                } else {
                    let a = reduce_term(then_contract, f, termination, states, factor.clone())?;
                    let b = reduce_term(else_contract, f, termination, states, factor)?;
                    if matches!((&a, &b), (Term::Empty, Term::Empty)) {
                        Term::Empty
                    } else {
                        Term::conditional(c, a, b)
                    }
                }
            }
        }
        Term::Barrier {
            id,
            observations,
            direction,
            level,
            hit,
            on_hit,
            on_no_hit,
        } => {
            if termination.is_some_and(|d| all_cancelled(t, d)) {
                Term::Empty
            } else {
                let condition =
                    Term::barrier_condition(observations, *direction, *level, *hit).reduce(f)?;
                let triggered = condition.constant() == Some(true);
                states.insert(id.clone(), triggered);
                if let Some(v) = condition.constant() {
                    reduce_term(
                        if v { on_hit } else { on_no_hit },
                        f,
                        termination,
                        states,
                        factor,
                    )?
                } else {
                    Term::Barrier {
                        id: id.clone(),
                        observations: observations
                            .iter()
                            .filter(|id| !f.contains_key(*id))
                            .cloned()
                            .collect(),
                        direction: *direction,
                        level: *level,
                        hit: false,
                        on_hit: Box::new(reduce_term(
                            on_hit,
                            f,
                            termination,
                            states,
                            factor.clone(),
                        )?),
                        on_no_hit: Box::new(reduce_term(
                            on_no_hit,
                            f,
                            termination,
                            states,
                            factor,
                        )?),
                    }
                }
            }
        }
        Term::Exercise { .. } | Term::ContinuousBarrier { .. } => {
            return Err(ContractError::Unsupported(
                "exercise/continuous barrier".into(),
            ));
        }
    })
}
fn all_cancelled(t: &Term, date: Date) -> bool {
    match t {
        Term::Empty => true,
        Term::Cashflow { cashflow } => cashflow.determination_date > date,
        Term::Bundle { contracts } => contracts.iter().all(|t| all_cancelled(t, date)),
        Term::Conditional {
            then_contract,
            else_contract,
            ..
        } => all_cancelled(then_contract, date) && all_cancelled(else_contract, date),
        Term::Barrier {
            on_hit, on_no_hit, ..
        } => all_cancelled(on_hit, date) && all_cancelled(on_no_hit, date),
        Term::Reverse { contract } | Term::Scale { contract, .. } => all_cancelled(contract, date),
        _ => false,
    }
}
fn collect_determined(
    t: &Term,
    as_of: Date,
    settled: &BTreeSet<String>,
) -> Vec<DeterminedCashflow> {
    match t {
        Term::Cashflow { cashflow: c } if c.determination_date <= as_of => c
            .amount
            .constant()
            .map(|amount| DeterminedCashflow {
                id: c.id.clone(),
                currency: c.currency,
                amount: amount * c.direction.sign(),
                determination_date: c.determination_date,
                payment_date: c.payment_date,
                settlement_date: c.settlement_date,
                settled: settled.contains(&c.id),
            })
            .into_iter()
            .collect(),
        Term::Bundle { contracts } => contracts
            .iter()
            .flat_map(|t| collect_determined(t, as_of, settled))
            .collect(),
        _ => vec![],
    }
}
fn remove_settled(t: Term, settled: &BTreeSet<String>) -> Term {
    match t {
        Term::Cashflow {
            cashflow: Cashflow { id: ref key, .. },
        } if settled.contains(key) => Term::Empty,
        Term::Bundle { contracts } => bundle(
            contracts
                .into_iter()
                .map(|t| remove_settled(t, settled))
                .collect(),
        ),
        other => other,
    }
}
