//! Joint settlement-measure paths shared by dated cash flows and exercise policies.
use super::model_path::{StockPath, simulation_grid};
use super::rates::{RateState, RateStep};
use super::{CrossCurrencyEquityOption, CrossCurrencyHullWhite, sampling};
use crate::EquityModelSpec;
use crate::core::{Date, DayCountConvention};
use crate::market::DiscountCurve;
use crate::mc::LocalVolTimeGrid;
use crate::multi_asset::MultiAssetError as Error;
use std::collections::BTreeMap;

pub(super) struct JointPlan {
    pub times: Vec<f64>,
    pub dates: BTreeMap<Date, usize>,
    pub layout: sampling::Layout,
    spot: f64,
    rho: f64,
    sigma: f64,
    dd: Vec<f64>,
    fx_forward: Vec<f64>,
    stock_carry: Vec<f64>,
    path: Option<StockPath>,
    rate_steps: Vec<RateStep>,
    dividends: bool,
}

pub(super) struct JointPath {
    pub spots: Vec<(f64, Option<f64>)>,
    pub fx: Vec<f64>,
    pub discounts: Vec<f64>,
    pub factors: Vec<[f64; 2]>,
}

impl JointPlan {
    pub fn for_option(
        option: &CrossCurrencyEquityOption,
        dates: impl Iterator<Item = Date>,
        last_equity: Date,
        events: Vec<f64>,
    ) -> Result<Self, Error> {
        let base = EquityModelSpec::Standard(crate::models::ModelSpec::BlackScholes(
            option.dynamics.equity,
        ));
        let model = option.equity_model.as_ref().unwrap_or(&base);
        let deterministic;
        let rates = match &option.rates {
            Some(r) => r,
            None => {
                let zero = crate::models::HullWhite1Factor::new(0., vec![0.], vec![0.])
                    .map_err(Error::numerical)?;
                deterministic = CrossCurrencyHullWhite::independent(
                    zero.clone(),
                    zero,
                    option.dynamics.correlation,
                )?;
                &deterministic
            }
        };
        Self::new(
            option,
            model,
            rates,
            dates.map(|d| (d, 0)).collect(),
            last_equity,
            events,
        )
    }

    pub fn new(
        option: &CrossCurrencyEquityOption,
        model: &EquityModelSpec,
        rates: &CrossCurrencyHullWhite,
        mut dates: BTreeMap<Date, usize>,
        last_equity: Date,
        mut events: Vec<f64>,
    ) -> Result<Self, Error> {
        let time = |d| DayCountConvention::Act365F.year_fraction(option.valuation_date, d);
        if dates.is_empty() || dates.keys().any(|&d| d < option.valuation_date) {
            return Err(Error::Invalid(
                "joint paths require outstanding future or valuation-day dates",
            ));
        }
        let end = time(*dates.keys().next_back().unwrap());
        let equity_end = time(last_equity);
        events.extend(dates.keys().map(|&d| time(d)));
        let foreign = option.foreign_market.forward();
        events.extend(
            rates
                .domestic
                .volatility_times()
                .iter()
                .chain(rates.foreign.volatility_times())
                .copied()
                .filter(|&t| t <= end),
        );
        let grid = simulation_grid(model, foreign, events, equity_end, option.maximum_step)?;
        let times = grid.nodes();
        for (&d, index) in &mut dates {
            *index = times
                .binary_search_by(|t| t.total_cmp(&time(d)))
                .map_err(|_| Error::Invalid("missing rate or fixing date"))?;
        }
        let equity_steps = dates.get(&last_equity).copied().unwrap_or(0);
        let spot = foreign.spot().get();
        let path = if equity_steps == 0 {
            None
        } else {
            let prefix = LocalVolTimeGrid::compile(times[..=equity_steps].to_vec(), f64::MAX)
                .map_err(Error::numerical)?;
            Some(StockPath::compile(
                model,
                &prefix,
                foreign,
                Some(rates),
                option.dividends,
            )?)
        };
        let (dim, blocks) = path.as_ref().map_or((0, 0), |p| p.layout(equity_steps));
        let sigma = option.dynamics.fx.volatility().get();
        let rho = option.dynamics.correlation;
        let rate_steps = rates.steps(times, sigma)?;
        let n = rate_steps.len();
        let mut dd = Vec::new();
        let mut fx_forward = Vec::new();
        let mut stock_carry = Vec::new();
        for &t in times {
            dd.push(option.domestic_curve.discount(t)?);
            let df = foreign.discount_curve().discount(t)?;
            fx_forward.push(option.fx_spot.get() * df / dd.last().unwrap());
            stock_carry.push(foreign.dividend_curve().discount(t)? / df);
        }
        // Five extra normal blocks condition FX and both rates on the equity
        // Brownian. After the last equity observation a sixth supplies its tail.
        let tail = equity_steps < n;
        let layout = sampling::Layout {
            equity_steps,
            equity_dimension: dim,
            brownian_blocks: blocks,
            extra_blocks: 5 + usize::from(tail),
        };

        Ok(Self {
            times: times.to_vec(),
            dates,
            layout,
            spot,
            rho,
            sigma,
            dd,
            fx_forward,
            stock_carry,
            path,
            rate_steps,
            dividends: option.dividends.is_some(),
        })
    }

    pub fn simulate(&self, normals: &[f64], exercise: bool) -> Result<JointPath, Error> {
        let times = &self.times;
        let n = self.rate_steps.len();
        let equity_steps = self.layout.equity_steps;
        let dim = self.layout.equity_dimension;
        let (rho, sigma, spot) = (self.rho, self.sigma, self.spot);
        let retain_factors = self.dividends || exercise;
        let extra = &normals[dim..];
        let mut shocks = normals[..dim].to_vec();
        for (z, interval) in shocks.iter_mut().zip(times[..=equity_steps].windows(2)) {
            *z -= rho * sigma * (interval[1] - interval[0]).sqrt();
        }
        let mut rate_carry = Vec::with_capacity(equity_steps + 1);
        let mut factors = Vec::new();
        let mut foreign_factors = Vec::with_capacity(equity_steps + 1);
        let mut state = RateState::default();
        let mut fx = Vec::with_capacity(n + 1);
        let mut discounts = Vec::with_capacity(n + 1);
        for (i, &time) in times.iter().enumerate() {
            if i > 0 {
                let j = i - 1;
                let stock_normal = if j < equity_steps {
                    normals[j]
                } else {
                    extra[5 * n + j]
                };
                self.rate_steps[j].advance(
                    &mut state,
                    [
                        stock_normal,
                        extra[j],
                        extra[n + j],
                        extra[2 * n + j],
                        extra[3 * n + j],
                        extra[4 * n + j],
                    ],
                );
            }
            let [domestic, foreign] = state.integrated;
            let x = self.fx_forward[i]
                * (domestic - foreign - 0.5 * sigma * sigma * time + sigma * state.fx_brownian)
                    .exp();
            let discount = self.dd[i] * (-domestic).exp();
            if !x.is_finite() || x <= 0. || !discount.is_finite() || discount <= 0. {
                return Err(Error::Invalid(
                    "nonfinite or nonpositive simulated FX or discount",
                ));
            }
            if i <= equity_steps {
                rate_carry.push(foreign);
                if retain_factors {
                    factors.push(state.factors);
                    foreign_factors.push(state.factors[1]);
                }
            }
            fx.push(x);
            discounts.push(discount);
        }
        let spots = match &self.path {
            Some(p) => p.spots(
                spot,
                &shocks,
                &self.stock_carry,
                Some(&rate_carry),
                &foreign_factors,
            )?,
            None => vec![(spot, None)],
        };

        Ok(JointPath {
            spots,
            fx,
            discounts,
            factors,
        })
    }
}
