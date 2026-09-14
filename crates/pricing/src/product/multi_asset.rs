//! Multi-asset contracts lower into the shared payoff graph.
use super::{CompactC2Smoothing, OptionSide, SourceGraph, SourceGraphBuilder, SourceOpcode as Op};
use crate::core::{CurrencyId, Date, NodeId, UnderlyingId};
use crate::multi_asset::MultiAssetError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasketComponent {
    pub underlying: UnderlyingId,
    pub weight: f64,
    pub scale: f64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorstOfComponent {
    pub underlying: UnderlyingId,
    pub reference_level: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiAssetProduct {
    pub(crate) currency: CurrencyId,
    pub(crate) underlyings: Vec<UnderlyingId>,
    pub(crate) graph: SourceGraph,
    pub(crate) payment_dates: Vec<Date>,
}
impl MultiAssetProduct {
    /// Each graph output is a cash flow paid on its corresponding date.
    /// Compilation checks each output's observation dependencies against its payment.
    pub fn from_graph(
        currency: CurrencyId,
        underlyings: Vec<UnderlyingId>,
        graph: SourceGraph,
        payment_dates: Vec<Date>,
    ) -> Result<Self, MultiAssetError> {
        if underlyings.is_empty()
            || underlyings
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != underlyings.len()
        {
            return Err(MultiAssetError::Invalid(
                "product underlyings must be nonempty and unique",
            ));
        }
        if payment_dates.is_empty() || payment_dates.len() != graph.outputs().len() {
            return Err(MultiAssetError::Invalid(
                "one payment date is required per graph output",
            ));
        }
        Ok(Self {
            currency,
            underlyings,
            graph,
            payment_dates,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn basket(
        currency: CurrencyId,
        components: Vec<BasketComponent>,
        side: OptionSide,
        strike: f64,
        notional: f64,
        expiry: Date,
        payment: Date,
        smoothing: Option<CompactC2Smoothing>,
    ) -> Result<Self, MultiAssetError> {
        check_option(strike, notional, expiry, payment)?;
        if components.is_empty() {
            return Err(MultiAssetError::Invalid("empty basket"));
        }
        let mut b = SourceGraphBuilder::new();
        let mut sum = b.literal(0.0)?;
        let mut underlyings = Vec::new();
        for c in components {
            if !c.weight.is_finite() || !c.scale.is_finite() || c.scale <= 0.0 {
                return Err(MultiAssetError::Invalid(
                    "basket weights must be finite and scales positive",
                ));
            }
            underlyings.push(c.underlying);
            let spot = b.push(Op::TerminalSpot {
                underlying: c.underlying,
                observation_date: expiry,
            })?;
            let weight = b.literal(c.weight)?;
            let scale = b.literal(c.scale)?;
            let weighted = b.push(Op::Multiply {
                left: weight,
                right: spot,
            })?;
            let normalized = b.push(Op::Divide {
                numerator: weighted,
                denominator: scale,
            })?;
            sum = b.push(Op::Add {
                left: sum,
                right: normalized,
            })?;
        }
        let payoff = option(&mut b, sum, side, strike, notional, smoothing)?;
        Self::from_graph(currency, underlyings, b.finish(vec![payoff]), vec![payment])
    }
    #[allow(clippy::too_many_arguments)]
    pub fn worst_of(
        currency: CurrencyId,
        components: Vec<WorstOfComponent>,
        side: OptionSide,
        strike: f64,
        notional: f64,
        expiry: Date,
        payment: Date,
        smoothing: Option<CompactC2Smoothing>,
    ) -> Result<Self, MultiAssetError> {
        check_option(strike, notional, expiry, payment)?;
        let mut b = SourceGraphBuilder::new();
        let w = worst(&mut b, &components, expiry, smoothing)?;
        let payoff = option(&mut b, w, side, strike, notional, smoothing)?;
        Self::from_graph(
            currency,
            components.iter().map(|c| c.underlying).collect(),
            b.finish(vec![payoff]),
            vec![payment],
        )
    }
    pub fn currency(&self) -> CurrencyId {
        self.currency
    }
    pub fn underlyings(&self) -> &[UnderlyingId] {
        &self.underlyings
    }
    pub fn source_graph(&self) -> &SourceGraph {
        &self.graph
    }
    pub fn payment_dates(&self) -> &[Date] {
        &self.payment_dates
    }
}
fn check_option(
    strike: f64,
    notional: f64,
    expiry: Date,
    payment: Date,
) -> Result<(), MultiAssetError> {
    if !strike.is_finite() || !notional.is_finite() || notional <= 0.0 || payment < expiry {
        return Err(MultiAssetError::Invalid(
            "invalid strike, notional or payment before expiry",
        ));
    }
    Ok(())
}
fn option(
    b: &mut SourceGraphBuilder,
    value: NodeId,
    side: OptionSide,
    strike: f64,
    notional: f64,
    smoothing: Option<CompactC2Smoothing>,
) -> Result<NodeId, MultiAssetError> {
    let k = b.literal(strike)?;
    let intrinsic = match side {
        OptionSide::Call => b.push(Op::Subtract {
            left: value,
            right: k,
        })?,
        OptionSide::Put => b.push(Op::Subtract {
            left: k,
            right: value,
        })?,
    };
    let zero = b.literal(0.0)?;
    let positive = b.push(match smoothing {
        Some(s) => Op::SmoothMaximum {
            left: intrinsic,
            right: zero,
            smoothing: s,
        },
        None => Op::Maximum {
            left: intrinsic,
            right: zero,
        },
    })?;
    let amount = b.literal(notional)?;
    Ok(b.push(Op::Multiply {
        left: amount,
        right: positive,
    })?)
}
fn worst(
    b: &mut SourceGraphBuilder,
    components: &[WorstOfComponent],
    date: Date,
    smoothing: Option<CompactC2Smoothing>,
) -> Result<NodeId, MultiAssetError> {
    if components.is_empty() {
        return Err(MultiAssetError::Invalid("empty worst-of"));
    }
    let mut value = None;
    for c in components {
        if !c.reference_level.is_finite() || c.reference_level <= 0.0 {
            return Err(MultiAssetError::Invalid(
                "reference levels must be positive",
            ));
        }
        let spot = b.push(Op::TerminalSpot {
            underlying: c.underlying,
            observation_date: date,
        })?;
        let scale = b.literal(c.reference_level)?;
        let x = b.push(Op::Divide {
            numerator: spot,
            denominator: scale,
        })?;
        value = Some(match value {
            None => x,
            Some(left) => b.push(match smoothing {
                Some(s) => Op::SmoothMinimum {
                    left,
                    right: x,
                    smoothing: s,
                },
                None => Op::Minimum { left, right: x },
            })?,
        });
    }
    Ok(value.expect("nonempty components"))
}
fn indicator(
    b: &mut SourceGraphBuilder,
    w: NodeId,
    level: f64,
    smoothing: Option<CompactC2Smoothing>,
) -> Result<NodeId, MultiAssetError> {
    let level = b.literal(level)?;
    let input = b.push(Op::Subtract {
        left: w,
        right: level,
    })?;
    Ok(b.push(match smoothing {
        Some(s) => Op::SmoothIndicator {
            input,
            smoothing: s,
        },
        None => Op::Indicator { input },
    })?)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryTermination {
    Forfeit,
    Pay,
}
#[derive(Clone, Debug, PartialEq)]
pub struct AutocallObservation {
    pub date: Date,
    pub payment_date: Date,
    /// Nominal cash amount, not a rate.
    pub coupon_amount: f64,
    pub coupon_level: f64,
    pub call_level: Option<f64>,
    pub call_coupon_amount: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct AutocallSpec {
    pub currency: CurrencyId,
    pub components: Vec<WorstOfComponent>,
    pub observations: Vec<AutocallObservation>,
    pub notional: f64,
    pub final_barrier: f64,
    pub maturity: Date,
    pub maturity_payment: Date,
    pub memory: bool,
    pub on_autocall: MemoryTermination,
    pub on_maturity: MemoryTermination,
}
impl AutocallSpec {
    /// Unseasoned contract. Observation-date coupons are paid before the call decision.
    pub fn product(
        &self,
        smoothing: Option<CompactC2Smoothing>,
    ) -> Result<MultiAssetProduct, MultiAssetError> {
        check_option(
            self.final_barrier,
            self.notional,
            self.maturity,
            self.maturity_payment,
        )?;
        if self.final_barrier <= 0.0
            || self.observations.is_empty()
            || self.observations.windows(2).any(|w| w[0].date >= w[1].date)
        {
            return Err(MultiAssetError::Invalid(
                "invalid autocall barrier or observation schedule",
            ));
        }
        let mut b = SourceGraphBuilder::new();
        let zero = b.literal(0.0)?;
        let one = b.literal(1.0)?;
        let notional = b.literal(self.notional)?;
        let mut alive = one;
        let mut unpaid = zero;
        let mut outputs = Vec::new();
        let mut payments = Vec::new();
        for o in &self.observations {
            if o.date > self.maturity
                || o.payment_date < o.date
                || [o.coupon_amount, o.call_coupon_amount]
                    .iter()
                    .any(|x| !x.is_finite() || *x < 0.0)
                || !o.coupon_level.is_finite()
                || o.coupon_level <= 0.0
                || o.call_level.is_some_and(|x| !x.is_finite() || x <= 0.0)
            {
                return Err(MultiAssetError::Invalid("invalid autocall observation"));
            }
            let w = worst(&mut b, &self.components, o.date, smoothing)?;
            let pays = indicator(&mut b, w, o.coupon_level, smoothing)?;
            let coupon = b.literal(o.coupon_amount)?;
            let due = if self.memory {
                b.push(Op::Add {
                    left: unpaid,
                    right: coupon,
                })?
            } else {
                coupon
            };
            let paid = b.push(Op::Multiply {
                left: pays,
                right: due,
            })?;
            let not_paid = b.push(Op::Subtract {
                left: one,
                right: pays,
            })?;
            unpaid = if self.memory {
                b.push(Op::Multiply {
                    left: not_paid,
                    right: due,
                })?
            } else {
                zero
            };
            let call = match o.call_level {
                Some(level) => indicator(&mut b, w, level, smoothing)?,
                None => zero,
            };
            let call_coupon = b.literal(o.call_coupon_amount)?;
            let mut redemption = b.push(Op::Add {
                left: notional,
                right: call_coupon,
            })?;
            if self.on_autocall == MemoryTermination::Pay {
                redemption = b.push(Op::Add {
                    left: redemption,
                    right: unpaid,
                })?;
            }
            let redemption = b.push(Op::Multiply {
                left: call,
                right: redemption,
            })?;
            let cash = b.push(Op::Add {
                left: paid,
                right: redemption,
            })?;
            outputs.push(b.push(Op::Multiply {
                left: alive,
                right: cash,
            })?);
            payments.push(o.payment_date);
            let not_called = b.push(Op::Subtract {
                left: one,
                right: call,
            })?;
            alive = b.push(Op::Multiply {
                left: alive,
                right: not_called,
            })?;
        }
        let w = worst(&mut b, &self.components, self.maturity, smoothing)?;
        let protected = indicator(&mut b, w, self.final_barrier, smoothing)?;
        let downside_weight = b.push(Op::Subtract {
            left: one,
            right: protected,
        })?;
        let downside = b.push(Op::Multiply {
            left: downside_weight,
            right: w,
        })?;
        let redemption = b.push(Op::Add {
            left: protected,
            right: downside,
        })?;
        let mut redemption = b.push(Op::Multiply {
            left: notional,
            right: redemption,
        })?;
        if self.on_maturity == MemoryTermination::Pay {
            redemption = b.push(Op::Add {
                left: redemption,
                right: unpaid,
            })?;
        }
        outputs.push(b.push(Op::Multiply {
            left: alive,
            right: redemption,
        })?);
        payments.push(self.maturity_payment);
        MultiAssetProduct::from_graph(
            self.currency,
            self.components.iter().map(|c| c.underlying).collect(),
            b.finish(outputs),
            payments,
        )
    }
}
