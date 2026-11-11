//! Quote-space risk by refitting the two OU coordinates to two local premiums.
use super::*;
use crate::EstimatorKind;
use crate::analytical::{BlackForwardOracleInputs, evaluate_black_forward};
use crate::core::UnderlyingId;
use crate::engine::analytic::heston_fourier::black_iv_and_vega;
use crate::engine::multi_asset::cross_currency::option_risk::replicate;
use crate::engine::risk::report::estimate_from_statistics;
use crate::mc::DeterministicStatistics;
use crate::product::{OptionSide, SourceGraphBuilder, SourceOpcode, multi_asset::option};
#[path = "quote_risk/fit.rs"]
mod fit;

/// One European premium per share, paid at expiry in the stock's currency.
/// The supplied market forward sets the fixed Black quote convention.
#[derive(Clone, Copy, Debug)]
pub struct BergomiOptionQuote {
    pub expiry: Date,
    pub side: OptionSide,
    pub strike: f64,
    pub forward: f64,
    pub premium: f64,
}

#[derive(Clone, Debug)]
pub struct BergomiStockQuoteRisk {
    /// Model order, then short/long maturity order for the three quote arrays.
    pub states: [Estimate; 2],
    pub implied_volatilities: [f64; 2],
    pub price_residuals: [Estimate; 2],
    /// Settlement units per one absolute annualized volatility point (0.01).
    pub vegas: [Estimate; 2],
}

#[derive(Clone, Debug)]
pub struct BergomiQuoteRisk {
    pub present_value: Estimate,
    pub stocks: BTreeMap<UnderlyingId, BergomiStockQuoteRisk>,
}

struct LocalQuote {
    local: CrossCurrencyEquityBasket,
    premium: f64,
    implied: f64,
    vega: f64,
    bumped: [f64; 2],
}

impl CrossCurrencyEquityBasket {
    fn local_quote(&self, i: usize, q: BergomiOptionQuote, bump: f64) -> Result<LocalQuote, Error> {
        if q.expiry <= self.valuation {
            return Err(Error::Invalid(
                "calibration expiries must be strictly future",
            ));
        }
        let stock = &self.stocks[i];
        let market = stock.market.forward();
        let id = market.underlying();
        let time = DayCountConvention::Act365F.year_fraction(self.valuation, q.expiry);
        let discount = market.discount_curve().discount(time)?;
        let (implied, vega) = black_iv_and_vega(
            q.forward,
            q.strike,
            discount,
            time,
            matches!(q.side, OptionSide::Call),
            q.premium,
        )
        .map_err(Error::numerical)?;
        if implied <= bump {
            return Err(Error::Invalid(
                "down-bumped Black quote must have positive volatility",
            ));
        }
        let price = |volatility| {
            evaluate_black_forward(BlackForwardOracleInputs {
                forward: q.forward,
                strike: q.strike,
                side: q.side,
                time,
                discount,
                volatility,
                notional: 1.,
            })
            .map(|r| r.price)
            .map_err(Error::numerical)
        };
        let mut graph = SourceGraphBuilder::new();
        let s = graph.push(SourceOpcode::TerminalSpot {
            underlying: id,
            observation_date: q.expiry,
        })?;
        let payoff = option(&mut graph, s, q.side, q.strike, 1., None)?;
        let ccy = stock.market.currency();
        let product = GraphProduct::new(ccy, vec![id], graph.finish(vec![payoff]), vec![q.expiry])
            .map_err(Error::numerical)?;
        let rate = self.stock_rate[i];
        let (rates, correlations) = if let Some(models) = &self.rates {
            let column = self.stocks.len() + self.currencies.len() + rate;
            let rho = self.drivers.canonical()[i * self.drivers.dimension() + column];
            (
                Some(vec![models[rate].clone()]),
                vec![vec![1., rho], vec![rho, 1.]],
            )
        } else {
            (None, vec![vec![1.]])
        };
        let mut local = Self::from_legs(
            self.valuation,
            vec![product],
            vec![stock.clone()],
            ccy,
            Arc::new(market.discount_curve().clone()),
            false,
            vec![],
            correlations,
            rates,
            self.maximum_step,
        )?;
        local.dividend_states.insert(id, self.dividend_states[&id]);
        let bumped = [price(implied - bump)?, price(implied + bump)?];
        if !(bumped[0] < q.premium && q.premium < bumped[1]) {
            return Err(Error::Invalid(
                "Black volatility bump must resolve two distinct prices around the quoted premium",
            ));
        }
        Ok(LocalQuote {
            local,
            premium: q.premium,
            implied,
            vega,
            bumped,
        })
    }

    /// Fit short/long European quotes per selected stock with all other market
    /// inputs fixed. Repeat the entire calibration and paired note repricing
    /// independently to measure both sources of Monte Carlo uncertainty.
    pub fn bergomi_quote_risk(
        &self,
        quotes: &BTreeMap<UnderlyingId, [BergomiOptionQuote; 2]>,
        bump: f64,
        engine: EngineConfig,
        replications: u32,
        execution: ExecutionPolicy,
    ) -> Result<BergomiQuoteRisk, Error> {
        PositiveF64::new(bump, "quote_volatility_bump")?;
        if replications < 2 || replications.checked_mul(3).is_none() || quotes.is_empty() {
            return Err(Error::Invalid(
                "supply a short/long pair per selected stock and at least two independent replications",
            ));
        }
        let mut pairs = BTreeMap::new();
        for (&id, &[short, long]) in quotes {
            if short.expiry >= long.expiry {
                return Err(Error::Invalid(
                    "quotes must have distinct expiries in short, long order",
                ));
            }
            let i = self
                .stocks
                .iter()
                .position(|s| s.market.forward().underlying() == id)
                .ok_or(Error::Invalid("unknown stock in calibration quotes"))?;
            let EquityModelSpec::StochasticVolatility {
                factor: crate::VolatilityFactorSpec::BergomiTwoFactor(f),
                initial_volatility,
            } = &self.stocks[i].model
            else {
                return Err(Error::Invalid(
                    "quote calibration requires two-factor Bergomi volatility",
                ));
            };
            if *initial_volatility == 0.
                || f.vol_of_vol() == 0.
                || f.mixing_weight() == 0.
                || f.mixing_weight() == 1.
                || f.mean_reversions()[0] == f.mean_reversions()[1]
            {
                return Err(Error::Invalid(
                    "two active Bergomi factors with distinct reversions are required to identify both states",
                ));
            }
            if !self.dividend_states.get(&id).is_some_and(|s| s.age > 0.) {
                return Err(Error::Invalid(
                    "supply positive model age and current residual-share and dividend factors before calibration",
                ));
            }
            pairs.insert(
                id,
                [
                    self.local_quote(i, short, bump)?,
                    self.local_quote(i, long, bump)?,
                ],
            );
        }
        let mut present_values = Vec::new();
        let mut samples: BTreeMap<_, Vec<StockSample>> =
            pairs.keys().map(|&id| (id, Vec::new())).collect();
        for run in 0..replications {
            let calibration = replicate(engine, 3 * run)?;
            let validation = replicate(engine, 3 * run + 1)?;
            let valuation = replicate(engine, 3 * run + 2)?;
            let mut base = self.clone();
            let mut fits = BTreeMap::new();
            for (&id, pair) in &pairs {
                let fit = fit_quotes(pair, id, calibration, validation, execution, bump).map_err(
                    |e| {
                        Error::Numerical(format!(
                            "stock {} calibration, replication {}: {e}",
                            id.get(),
                            run + 1
                        ))
                    },
                )?;
                base.dividend_states.get_mut(&id).unwrap().x = fit.x;
                fits.insert(id, fit);
            }
            present_values.push(base.price_mc(valuation, execution)?.value().get());
            for (id, fit) in fits {
                let mut vegas = [0.; 2];
                for (j, states) in fit.scenarios.into_iter().enumerate() {
                    let [lo, hi] = states.map(|x| {
                        let mut note = base.clone();
                        note.dividend_states.get_mut(&id).unwrap().x = x;
                        note
                    });
                    vegas[j] = lo.scenario_pnl(&hi, valuation, execution)?.value().get() * 0.01
                        / (2. * bump);
                }
                samples.get_mut(&id).unwrap().push(StockSample {
                    x: fit.x,
                    errors: fit.errors,
                    vegas,
                });
            }
        }
        let estimate = |values: &[f64]| {
            let kind = match engine {
                EngineConfig::PseudoMonteCarlo(_) => EstimatorKind::PseudoMonteCarlo,
                EngineConfig::RandomizedQuasiMonteCarlo(_) => {
                    EstimatorKind::RandomizedQuasiMonteCarlo
                }
            };
            estimate_from_statistics(
                DeterministicStatistics::from_ordered_values_two_pass(values),
                u64::from(replications),
                1.,
                kind,
            )
            .map_err(Error::numerical)
        };
        let mut stocks = BTreeMap::new();
        for (id, v) in samples {
            let pair = |field: fn(&StockSample) -> [f64; 2]| -> Result<_, Error> {
                let column =
                    |i: usize| estimate(&v.iter().map(|s| field(s)[i]).collect::<Vec<_>>());
                Ok([column(0)?, column(1)?])
            };
            stocks.insert(
                id,
                BergomiStockQuoteRisk {
                    states: pair(|s| s.x)?,
                    implied_volatilities: pairs[&id].each_ref().map(|q| q.implied),
                    price_residuals: pair(|s| s.errors)?,
                    vegas: pair(|s| s.vegas)?,
                },
            );
        }
        Ok(BergomiQuoteRisk {
            present_value: estimate(&present_values)?,
            stocks,
        })
    }
}

struct StockSample {
    x: [f64; 2],
    errors: [f64; 2],
    vegas: [f64; 2],
}
struct QuoteFit {
    x: [f64; 2],
    errors: [f64; 2],
    scenarios: [[[f64; 2]; 2]; 2],
}

fn fit_quotes(
    pair: &[LocalQuote; 2],
    id: UnderlyingId,
    calibration: EngineConfig,
    validation: EngineConfig,
    execution: ExecutionPolicy,
    bump: f64,
) -> Result<QuoteFit, Error> {
    let price = |q: &LocalQuote, x, engine| -> Result<f64, Error> {
        let mut local = q.local.clone();
        local.dividend_states.get_mut(&id).unwrap().x = x;
        Ok(local.price_mc(engine, execution)?.value().get())
    };
    let prices = |x| -> Result<_, Error> {
        let normalized = |i: usize| {
            price(&pair[i], x, calibration).map(|p| (p - pair[i].premium) / pair[i].vega)
        };
        Ok([normalized(0)?, normalized(1)?])
    };
    let start = fit::Fit::start(pair[0].local.dividend_states[&id].x, &prices)?;
    let tolerance = 1e-5 * bump.min(0.01);
    // Check identification at the fitted state and use its local price Jacobian
    // to initialize both sides of each quote bump.
    let base = start
        .solve([0.; 2], tolerance, &prices)?
        .linearized(&prices)?;
    let mut scenarios = [[[0.; 2]; 2]; 2];
    for (j, q) in pair.iter().enumerate() {
        for (state, premium) in scenarios[j].iter_mut().zip(q.bumped) {
            let mut target = [0.; 2];
            target[j] = (premium - q.premium) / q.vega;
            *state = base.solve(target, tolerance, &prices)?.x;
        }
    }
    let errors = [
        price(&pair[0], base.x, validation)? - pair[0].premium,
        price(&pair[1], base.x, validation)? - pair[1].premium,
    ];
    Ok(QuoteFit {
        x: base.x,
        errors,
        scenarios,
    })
}
