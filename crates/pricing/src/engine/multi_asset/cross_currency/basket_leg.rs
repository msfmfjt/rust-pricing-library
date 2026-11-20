//! Contractual conversion of graph outputs on their payment dates.
use crate::core::{Date, PositiveF64};
use crate::multi_asset::MultiAssetError as Error;
use crate::product::GraphProduct;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub(super) enum CashConversion {
    Fixed(f64),
    Floating(usize),
}
impl CashConversion {
    pub(super) fn conversion(self, quote: impl FnOnce(usize) -> f64) -> f64 {
        match self {
            Self::Fixed(f) => f,
            Self::Floating(currency) => quote(currency),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct BasketLeg {
    pub graph: GraphProduct,
    pub composite: bool,
    pub quantity: f64,
    default: CashConversion,
    fixed: Option<f64>,
    // Each entry supplies a date for a simulated rate, or an observed rate.
    payments: BTreeMap<Date, (Date, CashConversion)>,
}

impl BasketLeg {
    pub fn new(graph: GraphProduct, default: CashConversion, composite: bool) -> Self {
        Self {
            graph,
            composite,
            quantity: 1.,
            default,
            fixed: None,
            payments: BTreeMap::new(),
        }
    }

    pub fn is_floating(&self) -> bool {
        matches!(self.default, CashConversion::Floating(_)) && self.fixed.is_none()
    }

    pub fn conversion(&self, payment: Date) -> (Date, CashConversion) {
        self.payments.get(&payment).copied().unwrap_or_else(|| {
            (
                payment,
                self.fixed.map_or(self.default, CashConversion::Fixed),
            )
        })
    }

    pub fn set_terms(
        &mut self,
        valuation: Date,
        fixed_rate: Option<f64>,
        schedule: &[(Date, Date)],
        known: &[(Date, f64)],
    ) -> Result<(), Error> {
        if !matches!(self.default, CashConversion::Floating(_)) {
            return Err(Error::Invalid(
                "leg FX terms require a foreign floating-FX leg",
            ));
        }
        if fixed_rate.is_some() && !(schedule.is_empty() && known.is_empty()) {
            return Err(Error::Invalid(
                "a fixed contractual FX rate cannot have a fixing schedule",
            ));
        }
        let fixed = fixed_rate
            .map(|v| PositiveF64::new(v, "fixed_fx_rate").map(|v| v.get()))
            .transpose()?;
        let mut observed = BTreeMap::new();
        for &(date, value) in known {
            let value = PositiveF64::new(value, "known_fx_fixing")?.get();
            if date > valuation || observed.insert(date, value).is_some() {
                return Err(Error::Invalid(
                    "known FX fixings must be unique and on or before valuation",
                ));
            }
        }
        let mut payments = BTreeMap::new();
        let mut used = BTreeSet::new();
        for &(pay, fixing) in schedule {
            if fixing > pay || !self.graph.payment_dates().contains(&pay) {
                return Err(Error::Invalid(
                    "FX fixing must refer to a leg payment and be on or before payment",
                ));
            }
            let entry = if let Some(&value) = observed.get(&fixing) {
                used.insert(fixing);
                (pay, CashConversion::Fixed(value))
            } else if fixing >= valuation {
                (fixing, self.default)
            } else {
                return Err(Error::Invalid(
                    "historical FX fixing requires its observed value",
                ));
            };
            if payments.insert(pay, entry).is_some() {
                return Err(Error::Invalid("duplicate leg payment in FX schedule"));
            }
        }
        if used.len() != observed.len() {
            return Err(Error::Invalid(
                "known FX fixing is unused by this leg schedule",
            ));
        }
        self.payments = payments;
        self.fixed = fixed;
        Ok(())
    }
}
