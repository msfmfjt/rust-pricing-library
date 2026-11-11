//! Size hedges in observable option-price coordinates, then hold the trades
//! fixed and independently reprice the combined portfolio on shared JPY paths.
use super::*;
use crate::core::FiniteF64;

#[derive(Clone, Debug)]
pub struct BergomiHedgeTerms {
    /// Signed position in units of the original note.
    pub note_quantity: f64,
    /// Shares per option contract in short/long order. Omitted stocks use one.
    pub contract_sizes: BTreeMap<UnderlyingId, [f64; 2]>,
    pub whole_contracts: bool,
}
impl Default for BergomiHedgeTerms {
    fn default() -> Self {
        Self {
            note_quantity: 1.,
            contract_sizes: BTreeMap::new(),
            whole_contracts: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BergomiOptionTrade {
    /// Settlement-currency vega of one long contract per +0.01 volatility.
    pub unit_vega: f64,
    pub theoretical_quantity: f64,
    pub quantity_standard_error: f64,
    /// Signed number of contracts actually used in the independent check.
    pub quantity: f64,
}
#[derive(Clone, Debug)]
pub struct BergomiQuoteHedge {
    pub unhedged: BergomiQuoteRisk,
    pub trades: BTreeMap<UnderlyingId, [BergomiOptionTrade; 2]>,
    pub hedged: BergomiQuoteRisk,
    pub theoretical_hedged: BergomiQuoteRisk,
}

impl CrossCurrencyEquityBasket {
    /// The calibration contracts are the quote instruments themselves. At
    /// unchanged spot FX, only the bumped premium changes in a market mark.
    /// Price every fixed hedge payoff again under the settlement measure for
    /// the residual, including the unbumped tenor's simulated exposure.
    pub fn bergomi_quote_hedge(
        &self,
        quotes: &BTreeMap<UnderlyingId, [BergomiOptionQuote; 2]>,
        terms: &BergomiHedgeTerms,
        bump: f64,
        engine: EngineConfig,
        replications: u32,
        execution: ExecutionPolicy,
    ) -> Result<BergomiQuoteHedge, Error> {
        FiniteF64::new(terms.note_quantity, "note_quantity")?;
        for (id, sizes) in &terms.contract_sizes {
            if !quotes.contains_key(id) {
                return Err(Error::Invalid(
                    "contract sizes must refer to a stock with calibration quotes",
                ));
            }
            for &size in sizes {
                PositiveF64::new(size, "contract_size")?;
            }
        }
        let pairs = self.calibration_pairs(quotes, bump, replications)?;
        let mut portfolio = self.clone();
        for leg in &mut portfolio.legs {
            leg.quantity = terms.note_quantity;
        }
        let unhedged =
            portfolio.evaluate_quote_risk(&pairs, bump, engine, replications, execution)?;
        let mut theoretical_portfolio = portfolio.clone();
        let mut rounded = false;
        let mut trades = BTreeMap::new();
        let underlying_order = self.legs[0].graph.underlyings();
        for (&id, pair) in &pairs {
            let sizes = terms.contract_sizes.get(&id).copied().unwrap_or([1.; 2]);
            let currency = pair[0].local.settlement_currency;
            let conversion = if currency == self.settlement_currency {
                CashConversion::Fixed(1.)
            } else {
                CashConversion::Floating(
                    self.currencies
                        .iter()
                        .position(|c| c.quote.currency == currency)
                        .ok_or(Error::Invalid("missing spot FX for hedge premium"))?,
                )
            };
            let fx = conversion.conversion(|i| self.currencies[i].quote.spot.get());
            let mut add_trade = |j: usize| -> Result<_, Error> {
                let [lo, hi] = pair[j].bumped;
                let unit_vega = PositiveF64::new(
                    fx * sizes[j] * (hi - lo) * (0.01 / (2. * bump)),
                    "option_quote_vega",
                )?
                .get();
                let exposure = unhedged.stocks[&id].vegas[j];
                let theoretical_quantity = FiniteF64::new(
                    -exposure.value().get() / unit_vega,
                    "theoretical_option_quantity",
                )?
                .get();
                let quantity_standard_error = FiniteF64::new(
                    exposure.standard_error().get() / unit_vega,
                    "option_quantity_standard_error",
                )?
                .get();
                let quantity = if terms.whole_contracts {
                    theoretical_quantity.round()
                } else {
                    theoretical_quantity
                };
                rounded |= quantity != theoretical_quantity;
                if theoretical_quantity != 0. {
                    let local = &pair[j].local.legs[0].graph;
                    let graph = GraphProduct::new(
                        currency,
                        underlying_order.to_vec(),
                        local.source_graph().clone(),
                        local.payment_dates().to_vec(),
                    )
                    .map_err(Error::numerical)?;
                    // A listed option observes the local physical share, even for
                    // a note whose contractual performance is FX-adjusted.
                    let unit_leg = BasketLeg::new(graph, conversion, false);
                    for (book, units) in [
                        (&mut portfolio, quantity),
                        (&mut theoretical_portfolio, theoretical_quantity),
                    ] {
                        if units != 0. {
                            let mut leg = unit_leg.clone();
                            leg.quantity =
                                FiniteF64::new(units * sizes[j], "option_hedge_shares")?.get();
                            book.legs.push(leg);
                        }
                    }
                }
                Ok(BergomiOptionTrade {
                    unit_vega,
                    theoretical_quantity,
                    quantity_standard_error,
                    quantity,
                })
            };
            trades.insert(id, [add_trade(0)?, add_trade(1)?]);
        }
        // Each call consumes 3 disjoint seeds per full refit/valuation replication.
        let holdout = replicate(engine, 3 * replications)?;
        let (hedged, theoretical_hedged) = if rounded {
            let [actual, ideal] = Self::quote_samples(
                [&portfolio, &theoretical_portfolio],
                &pairs,
                bump,
                holdout,
                replications,
                execution,
            )?;
            (
                actual.report(&pairs, holdout)?,
                ideal.report(&pairs, holdout)?,
            )
        } else {
            let same =
                portfolio.evaluate_quote_risk(&pairs, bump, holdout, replications, execution)?;
            (same.clone(), same)
        };
        Ok(BergomiQuoteHedge {
            unhedged,
            trades,
            hedged,
            theoretical_hedged,
        })
    }
}
