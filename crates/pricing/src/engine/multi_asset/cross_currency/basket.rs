//! Multi-stock payoff graphs settled in one currency against one shared FX rate.
use super::exotic::payment_fx_schedule;
use super::model_path::{StockPath, joint_grid};
use super::rates::{CrossCurrencyHullWhite, RateState};
use super::{EquityFxSettlement, dividend_path, sampling};
use crate::core::{Date, DayCountConvention, PositiveF64};
use crate::market::{
    CorrelationFactor, CorrelationToleranceConfig, DiscountCurve, EquityMarket,
    LogLinearDiscountCurve,
};
use crate::mc::{EngineConfig, ExecutionPolicy, LocalVolTimeGrid};
use crate::models::{BlackScholesSpec, BuehlerDividendModel, HullWhite1Factor, ModelSpec};
use crate::multi_asset::MultiAssetError as Error;
use crate::product::{GraphLimitPolicy, GraphProduct};
use crate::{EquityModelSpec, Estimate, EstimatorKind};
use std::{collections::BTreeMap, ops::Range, sync::Arc};

/// One foreign equity in the graph's declared underlying order.
#[derive(Clone, Debug)]
pub struct ForeignEquity {
    pub market: EquityMarket,
    pub model: EquityModelSpec,
    pub dividend_model: Option<BuehlerDividendModel>,
}

#[derive(Clone, Debug)]
pub struct CrossCurrencyEquityBasket {
    valuation: Date,
    product: GraphProduct,
    stocks: Vec<ForeignEquity>,
    domestic_curve: Arc<LogLinearDiscountCurve>,
    settlement: EquityFxSettlement,
    fx: PositiveF64,
    fx_vol: BlackScholesSpec,
    maximum_step: f64,
    // Brownian driver order is identical to the public correlation matrix.
    drivers: CorrelationFactor,
    equity_fx: Vec<f64>,
    rates: Option<Vec<CrossCurrencyHullWhite>>,
    fixings: BTreeMap<Date, Date>,
}

const TOLERANCE: CorrelationToleranceConfig = CorrelationToleranceConfig {
    symmetry_abs_tol: 1e-12,
    diagonal_abs_tol: 1e-12,
    psd_abs_tol: 1e-12,
    psd_rel_tol: 0.,
    zero_pivot_abs_tol: 1e-12,
    zero_pivot_rel_tol: 0.,
};

impl CrossCurrencyEquityBasket {
    /// Correlation order is the declared equities, quoted FX and, when supplied,
    /// the domestic and foreign short rates. FX is domestic per foreign unit.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        valuation: Date,
        product: GraphProduct,
        stocks: Vec<ForeignEquity>,
        domestic_curve: Arc<LogLinearDiscountCurve>,
        settlement: EquityFxSettlement,
        fx: f64,
        fx_volatility: f64,
        correlations: Vec<Vec<f64>>,
        rate_models: Option<(HullWhite1Factor, HullWhite1Factor)>,
        maximum_step: f64,
    ) -> Result<Self, Error> {
        PositiveF64::new(maximum_step, "maximum_step")?;
        let n = stocks.len();
        if n == 0 || n != product.underlyings().len() {
            return Err(Error::Invalid(
                "one foreign equity is required per graph underlying",
            ));
        }
        let currency = stocks[0].market.currency();
        if currency == product.currency() {
            return Err(Error::Invalid(
                "stock and settlement currencies must differ",
            ));
        }
        let df = stocks[0].market.forward().discount_curve();
        for (stock, &id) in stocks.iter().zip(product.underlyings()) {
            let market = stock.market.forward();
            if market.underlying() != id || stock.market.currency() != currency {
                return Err(Error::Invalid(
                    "stocks must follow graph underlying order and share one foreign currency",
                ));
            }
            if market.discount_curve().times() != df.times()
                || market.discount_curve().discount_factors() != df.discount_factors()
            {
                return Err(Error::Invalid(
                    "stocks must share the foreign discount curve",
                ));
            }
            if stock.dividend_model.is_some() {
                dividend_path::validate(&stock.model, market)?;
                if rate_models
                    .as_ref()
                    .is_some_and(|(_, f)| !f.is_deterministic())
                    && !matches!(
                        stock.model,
                        EquityModelSpec::Standard(
                            ModelSpec::BlackScholes(_) | ModelSpec::Black76(_)
                        )
                    )
                {
                    return Err(Error::Invalid(
                        "stochastic dividends and a stochastic foreign rate require constant equity volatility",
                    ));
                }
            } else if market
                .discrete_dividends()
                .is_some_and(|s| !s.events().is_empty())
            {
                return Err(Error::Invalid(
                    "foreign cash dividends require a dividend model for that stock",
                ));
            }
        }
        let width = n + 1 + 2 * usize::from(rate_models.is_some());
        if correlations.len() != width {
            return Err(Error::Invalid(
                "correlation order must be equities, FX, and both rates when supplied",
            ));
        }
        let factor = CorrelationFactor::compile(correlations, TOLERANCE)?;
        let c = |i: usize, j: usize| factor.canonical()[i * width + j];
        let equity_fx = (0..n).map(|i| c(i, n)).collect();
        let rates = rate_models
            .filter(|(d, f)| !d.is_deterministic() || !f.is_deterministic())
            .map(|(d, f)| {
                (0..n)
                    .map(|i| {
                        let ids = [i, n, n + 1, n + 2];
                        let matrix = ids.map(|r| ids.map(|s| c(r, s)));
                        CrossCurrencyHullWhite::new(d.clone(), f.clone(), matrix)
                    })
                    .collect::<Result<Vec<_>, Error>>()
            })
            .transpose()?;
        // With both rate volatilities zero the smaller factor reproduces the
        // deterministic-rate random stream for the same seed and equity/FX block.
        let drivers = if rates.is_none() && width > n + 1 {
            CorrelationFactor::compile(
                (0..=n)
                    .map(|i| (0..=n).map(|j| c(i, j)).collect())
                    .collect(),
                TOLERANCE,
            )?
        } else {
            factor
        };
        if product.payment_dates().iter().any(|&p| p < valuation) {
            return Err(Error::Invalid(
                "historical payments require realized cash flows",
            ));
        }
        Ok(Self {
            valuation,
            product,
            stocks,
            domestic_curve,
            settlement,
            fx: PositiveF64::new(fx, "fx_spot")?,
            fx_vol: BlackScholesSpec::new(fx_volatility)?,
            maximum_step,
            drivers,
            equity_fx,
            rates,
            fixings: BTreeMap::new(),
        })
    }

    /// Each pair is (payment, FX fixing). Unlisted payments fix on payment.
    pub fn with_payment_fx_fixings(mut self, schedule: &[(Date, Date)]) -> Result<Self, Error> {
        if !matches!(self.settlement, EquityFxSettlement::FloatingFx) {
            return Err(Error::Invalid(
                "payment FX fixings require floating_fx settlement",
            ));
        }
        let fixings = payment_fx_schedule(&self.product, self.valuation, schedule)?;
        if fixings.iter().any(|(p, f)| p != f) {
            // Reuse the graph's per-output dependency check for earlier fixings.
            let dates = self
                .product
                .payment_dates()
                .iter()
                .map(|p| *fixings.get(p).unwrap_or(p))
                .collect();
            GraphProduct::new(
                self.product.currency(),
                self.product.underlyings().to_vec(),
                self.product.source_graph().clone(),
                dates,
            )
            .map_err(|_| {
                Error::Invalid(
                    "FX cannot fix before the last equity observation determining its payment",
                )
            })?;
        }
        self.fixings = fixings;
        Ok(self)
    }

    pub fn price_mc(
        &self,
        engine: EngineConfig,
        execution: ExecutionPolicy,
    ) -> Result<Estimate, Error> {
        let product = &self.product;
        let payoff = product.source_graph().compile(GraphLimitPolicy::DEFAULT)?;
        let asset_index: BTreeMap<_, _> = product
            .underlyings()
            .iter()
            .copied()
            .enumerate()
            .map(|(i, u)| (u, i))
            .collect();
        let mut ends = vec![self.valuation; self.stocks.len()];
        let mut dates: BTreeMap<Date, usize> = product
            .payment_dates()
            .iter()
            .chain(self.fixings.values())
            .map(|&d| (d, 0))
            .collect();
        dates.insert(self.valuation, 0);
        for (id, d) in payoff.terminal_observations() {
            if d < self.valuation {
                return Err(Error::Invalid(
                    "historical stock observations require known fixings",
                ));
            }
            let i = asset_index[&id];
            ends[i] = ends[i].max(d);
            dates.insert(d, 0);
        }
        let composite = matches!(self.settlement, EquityFxSettlement::Composite);
        if product.last_payment_date() == self.valuation {
            let spot = |id, _| {
                asset_index.get(&id).map(|&i| {
                    self.stocks[i].market.forward().spot().get()
                        * if composite { self.fx.get() } else { 1. }
                })
            };
            let total = payoff
                .evaluate_with_pre_dividend_spots(spot, spot)?
                .iter()
                .sum::<f64>()
                * self.conversion(self.fx.get());
            return Estimate::new(total, 0., total, total, EstimatorKind::Analytical, 1)
                .map_err(Error::numerical);
        }
        let time = |d| DayCountConvention::Act365F.year_fraction(self.valuation, d);
        let mut events: Vec<_> = dates.keys().map(|&d| time(d)).collect();
        let rates = self.rates.as_ref().map(|r| &r[0]);
        if let Some(r) = rates {
            events.extend(
                r.domestic
                    .volatility_times()
                    .iter()
                    .chain(r.foreign.volatility_times())
                    .copied()
                    .filter(|&t| t <= time(product.last_payment_date())),
            );
        }
        let grid = joint_grid(
            self.stocks
                .iter()
                .zip(&ends)
                .map(|(s, &d)| (&s.model, s.market.forward(), time(d))),
            events,
            self.maximum_step,
        )?;
        let times = grid.nodes();
        for (&d, index) in &mut dates {
            *index = times
                .binary_search_by(|t| t.total_cmp(&time(d)))
                .map_err(|_| Error::Invalid("cash flow date missing from grid"))?;
        }
        let n = times.len() - 1;
        let sigma = self.fx_vol.volatility().get();
        let rate_steps = rates.map(|r| r.basket_steps(times, sigma)).transpose()?;
        let mut residual_count = 0;
        let mut stocks = Vec::new();
        for (i, (stock, end)) in self.stocks.iter().zip(ends).enumerate() {
            let path = stock.compile(
                &times[..=dates[&end]],
                self.rates.as_ref().map(|r| &r[i]),
                residual_count,
            )?;
            residual_count = path.residuals.end;
            stocks.push(path);
        }
        let domestic = times
            .iter()
            .map(|&t| self.domestic_curve.discount(t))
            .collect::<Result<Vec<_>, _>>()?;
        let foreign = self.stocks[0].market.forward().discount_curve();
        let fx_forwards = times
            .iter()
            .zip(&domestic)
            .map(|(&t, d)| Ok(self.fx.get() * foreign.discount(t)? / d))
            .collect::<Result<Vec<_>, Error>>()?;
        let cashflows: Vec<_> = product
            .payment_dates()
            .iter()
            .map(|p| (dates[p], dates[self.fixings.get(p).unwrap_or(p)]))
            .collect();
        let roots: Vec<_> = times.windows(2).map(|w| (w[1] - w[0]).sqrt()).collect();
        let width = self.drivers.dimension();
        let fx_index = stocks.len();
        let blocks = width + 2 * usize::from(rates.is_some());
        let layout = sampling::Layout {
            equity_steps: 0,
            equity_dimension: residual_count,
            brownian_blocks: 0,
            extra_blocks: blocks,
        };
        sampling::estimate(times, layout, engine, execution, |normals| {
            let shared = &normals[residual_count..];
            let mut innovations: Vec<Vec<f64>> = stocks
                .iter()
                .map(|s| Vec::with_capacity(s.steps + s.residuals.len()))
                .collect();
            let mut fx = vec![self.fx.get()];
            let mut discounts = vec![1.];
            let mut foreign_carry = vec![0.];
            let mut foreign_factors = vec![0.];
            let mut state = RateState::default();
            let mut fx_brownian = 0.;
            for (j, &root) in roots.iter().enumerate() {
                let z = |k: usize| shared[k * n + j];
                let driver = |k: usize| {
                    self.drivers.lower()[k * width..k * width + k + 1]
                        .iter()
                        .enumerate()
                        .map(|(l, c)| c * z(l))
                        .sum::<f64>()
                };
                for (i, stock) in stocks.iter().enumerate() {
                    if j < stock.steps {
                        innovations[i].push(driver(i) - self.equity_fx[i] * sigma * root);
                    }
                }
                fx_brownian += root * driver(fx_index);
                if let Some(r) = &rate_steps {
                    r[j].advance(
                        &mut state,
                        [driver(fx_index + 1), driver(fx_index + 2)],
                        [z(width), z(width + 1)],
                    );
                }
                let [d, f] = state.integrated;
                foreign_carry.push(f);
                foreign_factors.push(state.factors[1]);
                let x = fx_forwards[j + 1]
                    * (d - f + sigma * fx_brownian - 0.5 * sigma * sigma * times[j + 1]).exp();
                let discount = domestic[j + 1] * (-d).exp();
                if !x.is_finite() || x <= 0. || !discount.is_finite() || discount <= 0. {
                    return Err(Error::Invalid(
                        "nonpositive or nonfinite simulated discount or FX",
                    ));
                }
                fx.push(x);
                discounts.push(discount);
            }
            let mut spots = Vec::with_capacity(stocks.len());
            for (stock, mut shocks) in stocks.iter().zip(innovations) {
                shocks.extend_from_slice(&normals[stock.residuals.clone()]);
                spots.push(match &stock.path {
                    Some(path) => path.spots(
                        stock.spot,
                        &shocks,
                        &stock.cash_carry,
                        rates.map(|_| &foreign_carry[..=stock.steps]),
                        &foreign_factors[..=stock.steps],
                    )?,
                    None => vec![(stock.spot, None)],
                });
            }
            let observed = |id, d, pre| {
                let j = *dates.get(&d)?;
                let (post, before) = spots[*asset_index.get(&id)?][j];
                Some(
                    (if pre { before.unwrap_or(post) } else { post })
                        * if composite { fx[j] } else { 1. },
                )
            };
            let flows = payoff.evaluate_with_pre_dividend_spots(
                |u, d| observed(u, d, false),
                |u, d| observed(u, d, true),
            )?;
            Ok(flows
                .iter()
                .zip(&cashflows)
                .map(|(a, &(p, f))| a * discounts[p] * self.conversion(fx[f]))
                .sum())
        })
    }

    fn conversion(&self, fx: f64) -> f64 {
        match self.settlement {
            EquityFxSettlement::Quanto { fixed_fx_rate } => fixed_fx_rate.get(),
            EquityFxSettlement::FloatingFx => fx,
            EquityFxSettlement::Composite => 1.,
        }
    }
}

struct CompiledEquity {
    path: Option<StockPath>,
    spot: f64,
    cash_carry: Vec<f64>,
    steps: usize,
    residuals: Range<usize>,
}

impl ForeignEquity {
    fn compile(
        &self,
        nodes: &[f64],
        rates: Option<&CrossCurrencyHullWhite>,
        residual_begin: usize,
    ) -> Result<CompiledEquity, Error> {
        let market = self.market.forward();
        let steps = nodes.len() - 1;
        let path = if steps > 0 {
            let prefix =
                LocalVolTimeGrid::compile(nodes.to_vec(), f64::MAX).map_err(Error::numerical)?;
            Some(StockPath::compile(
                &self.model,
                &prefix,
                market,
                rates,
                self.dividend_model,
            )?)
        } else {
            None
        };
        let count = path.as_ref().map_or(0, |p| p.layout(steps).0 - steps);
        let cash_carry = nodes
            .iter()
            .map(|&t| {
                Ok(market.dividend_curve().discount(t)? / market.discount_curve().discount(t)?)
            })
            .collect::<Result<_, Error>>()?;
        Ok(CompiledEquity {
            path,
            spot: market.spot().get(),
            cash_carry,
            steps,
            residuals: residual_begin..residual_begin + count,
        })
    }
}
