//! Exercise and monitoring rights. Numerical policies belong to the pricing call.
use super::{BarrierDirection, Cashflow, ContractError, NumericExpr, Observation, ObservationKind};
use crate::core::{CurrencyId, Date};
use crate::product::{BarrierStyle, OptionSide};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum OptionFeature {
    Bermudan {
        #[serde(with = "super::wire::dates")]
        dates: Vec<Date>,
    },
    American {
        #[serde(with = "super::wire::date")]
        start: Date,
    },
    ContinuousBarrier(ContinuousBarrierTerms),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuousBarrierTerms {
    #[serde(with = "super::wire::date")]
    pub start: Date,
    #[serde(with = "super::wire::date")]
    pub end: Date,
    #[serde(with = "super::wire::date")]
    pub payment_date: Date,
    pub level: f64,
    pub direction: BarrierDirection,
    pub style: BarrierStyle,
    pub rebate: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VanillaOption {
    pub id: String,
    #[serde(with = "super::wire::currency")]
    pub currency: CurrencyId,
    pub expiry_observation: String,
    pub side: OptionSide,
    pub strike: f64,
    pub notional: f64,
    pub feature: OptionFeature,
}

/// Contractual opportunities for a single option. The expiry fixing closes the
/// legal American window even when the pricing grid has fewer dates.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OptionSchedule {
    pub option_id: String,
    pub expiry_observation: String,
    #[serde(with = "super::wire::date")]
    pub expiry: Date,
    pub feature: OptionFeature,
}

/// The native LSM solver uses the pricing specification's LSM configuration
/// unless the caller supplies an override. American rights require a valuation
/// grid. A continuous barrier requires explicit consent to the native bridge law.
#[derive(Clone, Debug, Default)]
pub struct AdvancedMcSettings {
    pub lsm: Option<crate::mc::LsmConfig>,
    pub american_grid: Option<Vec<Date>>,
    pub brownian_bridge: bool,
}

impl VanillaOption {
    pub(super) fn validate(&self, expiry: &Observation) -> Result<(), ContractError> {
        super::source::identifier(&self.id)?;
        if expiry.kind != ObservationKind::Spot || expiry.time.is_some() {
            return Err(ContractError::Unsupported(
                "option requires a dated post-dividend spot".into(),
            ));
        }
        if !self.strike.is_finite()
            || self.strike <= 0.0
            || !self.notional.is_finite()
            || self.notional <= 0.0
        {
            return Err(ContractError::Invalid(
                "option strike and notional must be finite and positive".into(),
            ));
        }
        match &self.feature {
            OptionFeature::Bermudan { dates } => {
                super::builders::schedule(dates)?;
                if dates.len() > 100_000 || dates.last() != Some(&expiry.date) {
                    return Err(ContractError::Invalid(
                        "Bermudan dates must end at expiry".into(),
                    ));
                }
            }
            OptionFeature::American { start } if *start > expiry.date => {
                return Err(ContractError::Invalid(
                    "American exercise starts after expiry".into(),
                ));
            }
            OptionFeature::ContinuousBarrier(b)
                if b.start > b.end
                    || b.end > expiry.date
                    || b.payment_date < expiry.date
                    || !b.level.is_finite()
                    || b.level <= 0.0
                    || !b.rebate.is_finite()
                    || b.rebate < 0.0 =>
            {
                return Err(ContractError::Invalid(
                    "invalid continuous monitoring window, level, rebate or payment".into(),
                ));
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn permits_exercise(&self, date: Date, expiry: Date) -> bool {
        match &self.feature {
            OptionFeature::Bermudan { dates } => dates.binary_search(&date).is_ok(),
            OptionFeature::American { start } => (*start..=expiry).contains(&date),
            OptionFeature::ContinuousBarrier(_) => false,
        }
    }

    /// A published exercise fixing has the same underlying, clock and source as
    /// the terminal fixing. Its identity also includes the exercise date.
    pub fn observation_on(&self, expiry: &Observation, date: Date) -> Observation {
        let mut spot = expiry.clone();
        if date != expiry.date {
            spot.id = format!("{}/spot/{date}", self.id);
            spot.date = date;
        }
        spot
    }
    pub(super) fn cashflow(&self, spot: &Observation, pay: Date) -> Cashflow {
        Cashflow::new(
            &self.id,
            self.currency,
            super::builders::vanilla(
                NumericExpr::observation(&spot.id),
                self.side,
                self.strike,
                self.notional,
            ),
            spot.date,
            pay,
        )
    }
}
