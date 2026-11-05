//! Instrument conveniences only generate contracts; no pricing algorithms live here.
use super::{
    Aggregate, BarrierDirection, Cashflow, Comparison, Contract, ContractError, NumericExpr,
    Observation, ObservationKind, PaymentDirection, Term,
};
use crate::core::{CurrencyId, Date, UnderlyingId};
use crate::product::{BarrierStyle, OptionSide};

#[derive(Clone, Debug)]
pub struct InstrumentBuilder {
    id: String,
    currency: CurrencyId,
    underlying: UnderlyingId,
    fixing_source: String,
}
#[derive(Clone, Debug)]
pub struct SwapPeriod {
    pub observation_date: Date,
    pub payment_date: Date,
    pub year_fraction: f64,
}
#[derive(Clone, Debug)]
pub struct AutocallPeriod {
    pub observation_date: Date,
    pub payment_date: Date,
    pub call_level: f64,
    pub coupon_level: f64,
    pub coupon_rate: f64,
}
#[derive(Clone, Debug)]
pub struct DiscreteBarrierTerms {
    pub side: OptionSide,
    pub strike: f64,
    pub notional: f64,
    pub expiry: Date,
    pub payment_date: Date,
    pub monitoring_dates: Vec<Date>,
    pub direction: BarrierDirection,
    pub style: BarrierStyle,
    pub level: f64,
    pub rebate: f64,
}
impl InstrumentBuilder {
    pub fn new(
        id: impl Into<String>,
        currency: CurrencyId,
        underlying: UnderlyingId,
        fixing_source: impl Into<String>,
    ) -> Result<Self, ContractError> {
        let id = id.into();
        let fixing_source = fixing_source.into();
        super::source::identifier(&id)?;
        super::source::identifier(&fixing_source)?;
        if id.len() > 200 {
            return Err(ContractError::Invalid(
                "builder ID exceeds 200 bytes".into(),
            ));
        }
        Ok(Self {
            id,
            currency,
            underlying,
            fixing_source,
        })
    }
    fn obs(&self, index: usize, date: Date) -> Observation {
        Observation::spot(
            format!("{}/observation/{index}", self.id),
            self.underlying,
            date,
            &self.fixing_source,
        )
    }
    fn cash(&self, suffix: &str, amount: NumericExpr, determination: Date, payment: Date) -> Term {
        Cashflow::new(
            format!("{}/{suffix}", self.id),
            self.currency,
            amount,
            determination,
            payment,
        )
        .into()
    }
    fn finish(
        &self,
        observations: Vec<Observation>,
        root: Term,
    ) -> Result<Contract, ContractError> {
        Contract::new(&self.id, observations, root)
    }
    pub fn forward(
        &self,
        strike: f64,
        notional: f64,
        observation_date: Date,
        payment_date: Date,
    ) -> Result<Contract, ContractError> {
        finite(&[strike, notional])?;
        let o = self.obs(0, observation_date);
        let amount = NumericExpr::observation(&o.id)
            .minus(strike.into())
            .times(notional.into());
        self.finish(
            vec![o],
            self.cash("forward", amount, observation_date, payment_date),
        )
    }
    pub fn european(
        &self,
        side: OptionSide,
        strike: f64,
        notional: f64,
        expiry: Date,
        payment_date: Date,
    ) -> Result<Contract, ContractError> {
        finite(&[strike, notional])?;
        let o = self.obs(0, expiry);
        let amount = vanilla(NumericExpr::observation(&o.id), side, strike, notional);
        self.finish(vec![o], self.cash("option", amount, expiry, payment_date))
    }
    pub fn digital(
        &self,
        side: OptionSide,
        strike: f64,
        payout: f64,
        expiry: Date,
        payment_date: Date,
    ) -> Result<Contract, ContractError> {
        finite(&[strike, payout])?;
        let o = self.obs(0, expiry);
        let condition = NumericExpr::observation(&o.id).compare(
            if side == OptionSide::Call {
                Comparison::GreaterEqual
            } else {
                Comparison::LessEqual
            },
            strike.into(),
        );
        self.finish(
            vec![o],
            self.cash(
                "digital",
                NumericExpr::select(condition, payout.into(), 0.0.into()),
                expiry,
                payment_date,
            ),
        )
    }
    pub fn asian(
        &self,
        operation: Aggregate,
        side: OptionSide,
        strike: f64,
        notional: f64,
        dates: Vec<Date>,
        payment_date: Date,
    ) -> Result<Contract, ContractError> {
        finite(&[strike, notional])?;
        schedule(&dates)?;
        let observations: Vec<_> = dates
            .iter()
            .enumerate()
            .map(|(i, &d)| self.obs(i, d))
            .collect();
        let average = NumericExpr::Aggregate {
            operation,
            values: observations
                .iter()
                .map(|o| NumericExpr::observation(&o.id))
                .collect(),
        };
        self.finish(
            observations,
            self.cash(
                "path_option",
                vanilla(average, side, strike, notional),
                *dates.last().unwrap(),
                payment_date,
            ),
        )
    }
    /// Both legs remain individually identifiable. Rates are explicit fixing
    /// observations; unsupported future rate simulation is rejected by compile_mc.
    pub fn swap(
        &self,
        notional: f64,
        fixed_rate: f64,
        periods: Vec<SwapPeriod>,
        pay_fixed: bool,
    ) -> Result<Contract, ContractError> {
        finite(&[notional, fixed_rate])?;
        schedule(
            &periods
                .iter()
                .map(|p| p.observation_date)
                .collect::<Vec<_>>(),
        )?;
        let mut observations = Vec::new();
        let mut flows = Vec::new();
        for (i, p) in periods.into_iter().enumerate() {
            if !p.year_fraction.is_finite() || p.year_fraction <= 0.0 {
                return Err(ContractError::Invalid(
                    "swap accrual fraction must be finite and positive".into(),
                ));
            }
            let mut o = self.obs(i, p.observation_date);
            o.kind = ObservationKind::Rate;
            let multiplier = notional * p.year_fraction;
            finite(&[multiplier, fixed_rate * multiplier])?;
            let mut fixed = Cashflow::new(
                format!("{}/fixed/{i}", self.id),
                self.currency,
                (fixed_rate * multiplier).into(),
                p.observation_date,
                p.payment_date,
            );
            let mut floating = Cashflow::new(
                format!("{}/floating/{i}", self.id),
                self.currency,
                NumericExpr::observation(&o.id).times(multiplier.into()),
                p.observation_date,
                p.payment_date,
            );
            if pay_fixed {
                fixed.direction = PaymentDirection::Pay;
            } else {
                floating.direction = PaymentDirection::Pay;
            }
            observations.push(o);
            flows.extend([Term::from(fixed), Term::from(floating)]);
        }
        self.finish(observations, Term::bundle(flows))
    }
    pub fn discrete_barrier(&self, terms: DiscreteBarrierTerms) -> Result<Contract, ContractError> {
        let t = terms;
        finite(&[t.strike, t.notional, t.level, t.rebate])?;
        schedule(&t.monitoring_dates)?;
        if t.monitoring_dates.iter().any(|d| *d > t.expiry) {
            return Err(ContractError::Invalid(
                "barrier monitoring after expiry".into(),
            ));
        }
        let mut observations: Vec<_> = t
            .monitoring_dates
            .iter()
            .enumerate()
            .map(|(i, &d)| self.obs(i, d))
            .collect();
        let monitors = observations.iter().map(|o| o.id.clone()).collect();
        let terminal = if let Some(o) = observations.iter().find(|o| o.date == t.expiry) {
            o.clone()
        } else {
            let o = self.obs(observations.len(), t.expiry);
            observations.push(o.clone());
            o
        };
        let option = self.cash(
            "option",
            vanilla(
                NumericExpr::observation(&terminal.id),
                t.side,
                t.strike,
                t.notional,
            ),
            t.expiry,
            t.payment_date,
        );
        let rebate = self.cash("rebate", t.rebate.into(), t.expiry, t.payment_date);
        let (on_hit, on_no_hit) = if t.style == BarrierStyle::KnockIn {
            (option, rebate)
        } else {
            (rebate, option)
        };
        self.finish(
            observations,
            Term::Barrier {
                id: format!("{}/barrier", self.id),
                observations: monitors,
                direction: t.direction,
                level: t.level,
                hit: false,
                on_hit: Box::new(on_hit),
                on_no_hit: Box::new(on_no_hit),
            },
        )
    }
    /// Coupon rates apply to notional. Coupons are paid when spot >= coupon
    /// level, including a redemption date. Final downside redemption is
    /// notional * final_spot / initial_spot below the protection level.
    pub fn autocall(
        &self,
        notional: f64,
        initial_spot: f64,
        protection_level: f64,
        periods: Vec<AutocallPeriod>,
    ) -> Result<Contract, ContractError> {
        finite(&[notional, initial_spot, protection_level])?;
        if initial_spot <= 0.0
            || notional <= 0.0
            || protection_level < 0.0
            || protection_level > initial_spot
        {
            return Err(ContractError::Invalid("autocall requires positive initial spot/notional and protection in [0, initial_spot]".into()));
        }
        if periods.len() > 20 {
            return Err(ContractError::Invalid(
                "autocall v1 supports at most 20 periods within graph/depth limits".into(),
            ));
        }
        schedule(
            &periods
                .iter()
                .map(|p| p.observation_date)
                .collect::<Vec<_>>(),
        )?;
        let observations: Vec<_> = periods
            .iter()
            .enumerate()
            .map(|(i, p)| self.obs(i, p.observation_date))
            .collect();
        let last = periods.last().unwrap();
        let terminal = NumericExpr::observation(&observations.last().unwrap().id);
        let protected = terminal
            .clone()
            .compare(Comparison::GreaterEqual, protection_level.into());
        let final_amount = NumericExpr::select(
            protected,
            notional.into(),
            terminal
                .divided_by(initial_spot.into())
                .times(notional.into()),
        );
        let mut tail = self.cash(
            "principal",
            final_amount,
            last.observation_date,
            last.payment_date,
        );
        for (i, p) in periods.iter().enumerate().rev() {
            finite(&[
                p.call_level,
                p.coupon_level,
                p.coupon_rate,
                notional * p.coupon_rate,
            ])?;
            let spot = NumericExpr::observation(&observations[i].id);
            let coupon = Term::conditional(
                spot.clone()
                    .compare(Comparison::GreaterEqual, p.coupon_level.into()),
                self.cash(
                    &format!("coupon/{i}"),
                    (notional * p.coupon_rate).into(),
                    p.observation_date,
                    p.payment_date,
                ),
                Term::Empty,
            );
            let redemption = self.cash(
                &format!("redemption/{i}"),
                notional.into(),
                p.observation_date,
                p.payment_date,
            );
            tail = Term::bundle(vec![
                coupon,
                Term::conditional(
                    spot.compare(Comparison::GreaterEqual, p.call_level.into()),
                    redemption,
                    tail,
                ),
            ]);
        }
        self.finish(observations, tail)
    }
}
fn vanilla(spot: NumericExpr, side: OptionSide, strike: f64, notional: f64) -> NumericExpr {
    let intrinsic = if side == OptionSide::Call {
        spot.minus(strike.into())
    } else {
        NumericExpr::from(strike).minus(spot)
    };
    intrinsic.maximum(0.0.into()).times(notional.into())
}
fn finite(values: &[f64]) -> Result<(), ContractError> {
    for &v in values {
        super::expression::finite(v)?;
    }
    Ok(())
}
fn schedule(dates: &[Date]) -> Result<(), ContractError> {
    if dates.is_empty() || dates.windows(2).any(|w| w[0] >= w[1]) {
        Err(ContractError::Invalid(
            "observation schedule must be nonempty, strictly increasing and unique".into(),
        ))
    } else {
        Ok(())
    }
}
impl Contract {
    pub fn bundle(id: impl Into<String>, contracts: Vec<Contract>) -> Result<Self, ContractError> {
        let mut observations = std::collections::BTreeMap::new();
        let mut terms = Vec::new();
        for c in contracts {
            for o in c.observations {
                if let Some(old) = observations.insert(o.id.clone(), o.clone())
                    && old != o
                {
                    return Err(ContractError::Invalid(
                        "conflicting observation identities in bundle".into(),
                    ));
                }
            }
            terms.push(c.root);
        }
        Self::new(
            id,
            observations.into_values().collect(),
            Term::bundle(terms),
        )
    }
    pub fn reversed(&self, id: impl Into<String>) -> Result<Self, ContractError> {
        Self::new(id, self.observations.clone(), self.root.clone().reverse())
    }
    pub fn scaled(&self, id: impl Into<String>, factor: f64) -> Result<Self, ContractError> {
        Self::new(
            id,
            self.observations.clone(),
            self.root.clone().scale(factor.into()),
        )
    }
}
