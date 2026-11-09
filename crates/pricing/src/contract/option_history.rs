use super::{BarrierDirection, ContractError, Observation, OptionFeature, VanillaOption};
use crate::core::Date;
use crate::product::BarrierStyle;
use std::collections::BTreeMap;

/// Historical observations and holder instructions are contractual data, never
/// sampled from an MC exercise policy or inferred from discrete barrier closes.
pub(super) struct OptionHistory<'a> {
    options: BTreeMap<&'a str, (&'a VanillaOption, &'a Observation)>,
    notices: BTreeMap<String, Date>,
    monitoring: BTreeMap<String, (Date, bool)>,
    pub as_of: Date,
    /// Intermediate settlement checks may coexist with other, unresolved rights.
    pub require_history: bool,
}
impl<'a> OptionHistory<'a> {
    pub fn new(contract: &'a super::Contract, as_of: Date) -> Self {
        let mut terms = Vec::new();
        contract.root().visit_options(&mut terms);
        let observations = contract.observations();
        let options = terms
            .into_iter()
            .map(|o| {
                let index = observations
                    .binary_search_by(|s| s.id.cmp(&o.expiry_observation))
                    .expect("validated observation");
                (o.id.as_str(), (o, &observations[index]))
            })
            .collect();
        Self {
            options,
            notices: Default::default(),
            monitoring: Default::default(),
            as_of,
            require_history: false,
        }
    }
    pub fn exercise_observation(&self, id: &str, date: Date) -> Result<Observation, ContractError> {
        let (o, expiry) = self
            .options
            .get(id)
            .ok_or_else(|| ContractError::EventConflict(format!("unknown option {id}")))?;
        if !o.permits_exercise(date, expiry.date) {
            return Err(ContractError::EventConflict(format!(
                "exercise outside the legal schedule of {id}"
            )));
        }
        Ok(o.observation_on(expiry, date))
    }
    pub fn recognizes_fixing(&self, obs: &Observation) -> bool {
        // Synthetic early observations are possible only for an American notice.
        let Some((id, _)) = obs.id.rsplit_once("/spot/") else {
            return false;
        };
        self.options
            .get(id)
            .is_some_and(|(option, _)| matches!(option.feature, OptionFeature::American { .. }))
            && self.exercise_observation(id, obs.date).as_ref() == Ok(obs)
    }
    pub fn check_barrier_date(&self, id: &str, date: Date) -> Result<(), ContractError> {
        if let Some((option, _)) = self.options.get(id)
            && let OptionFeature::ContinuousBarrier(window) = &option.feature
            && (window.start..=window.end).contains(&date)
        {
            return Ok(());
        }
        Err(ContractError::EventConflict(format!(
            "no continuous monitoring for {id} on {date}"
        )))
    }
    pub fn notice(&mut self, id: &str, date: Date) -> Result<(), ContractError> {
        if self
            .notices
            .insert(id.into(), date)
            .is_some_and(|previous| previous != date)
        {
            return Err(ContractError::EventConflict(format!(
                "{id} was already exercised"
            )));
        }
        Ok(())
    }
    pub fn status(&mut self, id: &str, date: Date, hit: bool) -> Result<(), ContractError> {
        if self
            .monitoring
            .get(id)
            .is_some_and(|&(old_date, old_hit)| old_hit != hit && (old_hit || old_date == date))
        {
            return Err(ContractError::EventConflict(format!(
                "inconsistent cumulative monitoring status for {id}"
            )));
        }
        if !self
            .monitoring
            .get(id)
            .is_some_and(|&(_, previous_hit)| previous_hit)
        {
            self.monitoring.insert(id.into(), (date, hit));
        }
        Ok(())
    }

    pub fn resolve(
        &self,
        option: &VanillaOption,
        fixings: &BTreeMap<String, f64>,
        termination: Option<Date>,
        states: &mut BTreeMap<String, bool>,
    ) -> Result<super::Term, ContractError> {
        use super::Term;
        let (_, expiry) = self.options[option.id.as_str()];
        let mut residual = option.clone();
        match &mut residual.feature {
            OptionFeature::Bermudan { .. } | OptionFeature::American { .. } => {
                let decision = self.notices.get(&option.id).copied().unwrap_or(expiry.date);
                if termination.is_some_and(|t| t < decision) {
                    return Ok(Term::Empty);
                }
                if decision <= self.as_of {
                    let obs = option.observation_on(expiry, decision);
                    let mut cash = option.cashflow(&obs, decision);
                    cash.amount = cash.amount.reduce(fixings)?;
                    if decision != expiry.date && cash.amount.constant().is_none() {
                        if !self.require_history {
                            return Ok(Term::VanillaOption { option: residual });
                        }
                        return Err(ContractError::MissingFixing(format!(
                            "exercise fixing {} on {decision}",
                            obs.id
                        )));
                    }
                    return Ok(cash.into());
                }
                match &mut residual.feature {
                    OptionFeature::Bermudan { dates } => dates.retain(|&date| date >= self.as_of),
                    OptionFeature::American { start } => *start = (*start).max(self.as_of),
                    OptionFeature::ContinuousBarrier(_) => unreachable!(),
                }
            }
            OptionFeature::ContinuousBarrier(b) => {
                let observed = self.monitoring.get(&option.id).copied();
                let endpoint_hit = (b.start..=b.end).contains(&expiry.date)
                    && fixings
                        .get(&expiry.id)
                        .is_some_and(|&spot| match b.direction {
                            BarrierDirection::Up => spot >= b.level,
                            BarrierDirection::Down => spot <= b.level,
                        });
                if endpoint_hit && observed == Some((expiry.date, false)) {
                    return Err(ContractError::EventConflict(
                        "no-hit report contradicts observed terminal touch".into(),
                    ));
                }
                let hit = endpoint_hit || observed.is_some_and(|(_, hit)| hit);
                states.insert(option.id.clone(), hit);
                if hit || observed == Some((b.end, false)) {
                    let mut cash = option.cashflow(expiry, b.payment_date);
                    if hit != (b.style == BarrierStyle::KnockIn) {
                        cash.amount = b.rebate.into();
                        cash.determination_date = observed
                            .filter(|(_, was_hit)| *was_hit)
                            .map_or(if hit { expiry.date } else { b.end }, |(date, _)| date);
                    }
                    return Ok(cash.into());
                }
                if termination.is_some_and(|t| t < expiry.date) {
                    return Ok(Term::Empty);
                }
                if self.as_of > b.start {
                    let through = self.as_of.min(b.end);
                    if observed.is_none_or(|(date, _)| date < through) {
                        if !self.require_history {
                            return Ok(Term::VanillaOption { option: residual });
                        }
                        return Err(ContractError::MissingFixing(format!(
                            "continuous monitoring status for {} through {through}",
                            option.id
                        )));
                    }
                    b.start = through;
                }
            }
        }
        Ok(Term::VanillaOption { option: residual })
    }
}

pub(super) fn exercisable(
    term: &super::Term,
    id: &str,
    fixings: &BTreeMap<String, f64>,
) -> Result<bool, ContractError> {
    use super::Term;
    match term {
        Term::VanillaOption { option } => Ok(option.id == id),
        Term::Scale { contract, .. } | Term::Reverse { contract } => {
            exercisable(contract, id, fixings)
        }
        Term::Bundle { contracts } => {
            for child in contracts {
                if exercisable(child, id, fixings)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Term::Conditional {
            condition,
            then_contract,
            else_contract,
        } => match condition.reduce(fixings)?.constant() {
            Some(true) => exercisable(then_contract, id, fixings),
            Some(false) => exercisable(else_contract, id, fixings),
            None => Ok(false),
        },
        Term::Barrier {
            observations,
            direction,
            level,
            hit,
            on_hit,
            on_no_hit,
            ..
        } => {
            let touched = *hit
                || observations.iter().filter_map(|key| fixings.get(key)).any(
                    |&s| match direction {
                        BarrierDirection::Up => s >= *level,
                        BarrierDirection::Down => s <= *level,
                    },
                );
            if touched {
                exercisable(on_hit, id, fixings)
            } else if observations.iter().all(|key| fixings.contains_key(key)) {
                exercisable(on_no_hit, id, fixings)
            } else {
                Ok(false)
            }
        }
        _ => Ok(false),
    }
}
impl super::Contract {
    pub fn exercise_observation_json(
        &self,
        option_id: &str,
        date: Date,
    ) -> Result<String, ContractError> {
        serde_json::to_string(&self.exercise_observation(option_id, date)?)
            .map_err(|e| ContractError::Wire(e.to_string()))
    }
    pub fn exercise_observation(
        &self,
        option_id: &str,
        date: Date,
    ) -> Result<Observation, ContractError> {
        OptionHistory::new(self, date).exercise_observation(option_id, date)
    }
}
