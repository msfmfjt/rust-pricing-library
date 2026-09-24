//! Positive split evolution and carry-funded stochastic cash reserves.

mod bergomi;
pub(crate) mod hull_white;
pub(in crate::engine) mod reverse;
mod rough;
use bergomi::BergomiDividendKernel;
use rough::RoughDividendKernel;
use std::sync::Arc;

use crate::MonteCarloError;
use crate::market::EquityForward;
use crate::mc::LocalVolTimeGrid;
use crate::models::stochastic_dividends::{invalid, nonnegative, positive};
use crate::models::{Bergomi1Factor, Bergomi2Factor, RoughBergomi};
use crate::models::{BuehlerDividendModel, BuehlerDividendState, StochasticDividendError};

impl BuehlerDividendModel {
    /// E[Y(t+h) | f(t),Y(t)]. Also the normalized forecast for a cash dividend
    /// at t+h. No realized future state is used in the current reserve.
    pub fn expected_factor(
        self,
        state: BuehlerDividendState,
        horizon: f64,
    ) -> Result<f64, StochasticDividendError> {
        nonnegative(horizon, "forecast_horizon")?;
        let (a, b) = decay(self.mean_reversion, horizon);
        let result = blend(a, b, state.dividend, self.target(state.equity));
        positive(result, "dividend_forecast")?;
        Ok(result)
    }

    /// Symmetric drift/diffusion/drift splitting. Normals are independent,
    /// ordered equity then dividend. Both coordinates remain reserved even at
    /// zero volatility or perfect correlation. No clipping/floor is applied.
    pub fn evolve(
        self,
        state: BuehlerDividendState,
        equity_volatility: f64,
        dt: f64,
        normals: [f64; 2],
    ) -> Result<BuehlerDividendState, StochasticDividendError> {
        nonnegative(equity_volatility, "equity_volatility")?;
        nonnegative(dt, "step_length")?;
        if normals.iter().any(|z| !z.is_finite()) {
            return Err(invalid("normal"));
        }
        let (a, b) = decay(self.mean_reversion, 0.5 * dt);
        let half = blend(a, b, state.dividend, self.target(state.equity));
        let rho = self.equity_dividend_correlation;
        let z_dividend = rho * normals[0] + ((1.0 - rho) * (1.0 + rho)).sqrt() * normals[1];
        let s = equity_volatility * dt.sqrt();
        let v = self.dividend_volatility * dt.sqrt();
        let equity = state.equity * (-0.5 * s * s + s * normals[0]).exp();
        let noise = half * (-0.5 * v * v + v * z_dividend).exp();
        // Reject overflow/underflow instead of allowing the second drift half
        // to conceal a nonpositive numerical diffusion result.
        positive(equity, "equity_diffusion")?;
        positive(noise, "dividend_diffusion")?;
        let dividend = blend(a, b, noise, self.target(equity));
        BuehlerDividendState::new(equity, dividend)
    }
    fn target(self, equity: f64) -> f64 {
        self.equity_linkage * equity + (1.0 - self.equity_linkage)
    }
}

fn decay(kappa: f64, dt: f64) -> (f64, f64) {
    let x = -kappa * dt;
    (x.exp(), -x.exp_m1())
}
fn blend(a: f64, b: f64, current: f64, target: f64) -> f64 {
    if current == target {
        current
    } else {
        a * current + b * target
    }
}

/// Post-event physical stock S=a*f+b*Y+c, with positive carry-funded residual.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StochasticDividendNode {
    time: f64,
    equity_coefficient: f64,
    dividend_coefficient: f64,
    constant: f64,
    event_mean_cash: Option<f64>,
}
impl StochasticDividendNode {
    #[must_use]
    pub const fn time(self) -> f64 {
        self.time
    }
    #[must_use]
    pub const fn coefficients(self) -> [f64; 3] {
        [
            self.equity_coefficient,
            self.dividend_coefficient,
            self.constant,
        ]
    }
    /// Realized cash at this node. The initial schedule stores its Q-mean.
    pub fn cash_paid(self, state: BuehlerDividendState) -> Result<f64, StochasticDividendError> {
        let cash = self.event_mean_cash.unwrap_or(0.0) * state.dividend;
        nonnegative(cash, "realized_cash")?;
        Ok(cash)
    }
    /// (post-event Spot, optional pre-event Spot). Ex-date observations use
    /// the first component, including when expiry coincides with the ex-date.
    pub fn spots(
        self,
        state: BuehlerDividendState,
    ) -> Result<(f64, Option<f64>), StochasticDividendError> {
        let post = self.equity_coefficient * state.equity
            + self.dividend_coefficient * state.dividend
            + self.constant;
        positive(post, "post_dividend_spot")?;
        let pre = if self.event_mean_cash.is_some() {
            let pre = post + self.cash_paid(state)?;
            positive(pre, "pre_dividend_spot")?;
            Some(pre)
        } else {
            None
        };
        Ok((post, pre))
    }
}

/// Deterministic-rate BS, Bergomi or rough-Bergomi path plan. All supplied future
/// cash means are funded, including those beyond this plan's final time.
#[derive(Clone, Debug)]
pub struct StochasticDividendPathPlan {
    model: BuehlerDividendModel,
    volatility: f64,
    times: Box<[f64]>,
    nodes: Box<[StochasticDividendNode]>,
    risky_spot: f64,
    dimension: u32,
    bergomi: Option<BergomiDividendKernel>,
    rough: Option<Arc<RoughDividendKernel>>,
}
impl StochasticDividendPathPlan {
    pub fn compile(
        market: &EquityForward,
        model: BuehlerDividendModel,
        volatility: f64,
        grid: &LocalVolTimeGrid,
    ) -> Result<Self, MonteCarloError> {
        nonnegative(volatility, "equity_volatility")?;
        let times = grid.nodes();
        let dimension = (times.len() - 1)
            .checked_mul(2)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(invalid("random_dimension"))?;
        let horizon = times[times.len() - 1];
        let spot = market.spot().get();
        let mut terms = Vec::new();
        let mut initial_reserve = 0.0;
        if let Some(schedule) = market.discrete_dividends() {
            for event in schedule.events() {
                if event.beta() != 0.0 {
                    return Err(StochasticDividendError::Unsupported {
                        feature: "proportional cash mixtures; supply fixed-cash Q-means only",
                    }
                    .into());
                }
                if event.ex_time() <= 0.0 {
                    return Err(invalid("strictly_future_ex_date").into());
                }
                if event.ex_time() <= horizon
                    && times
                        .binary_search_by(|t| t.total_cmp(&event.ex_time()))
                        .is_err()
                {
                    return Err(invalid("missing_dividend_time").into());
                }
                // G(t)=D_repo_spread(t)/D_discount(t). This is equity carry,
                // not the collateral-discounted price of a dividend claim.
                let growth = market.forward(event.ex_time())? / spot;
                positive(growth, "carry_growth")?;
                let amount = event.fixed_cash() / growth;
                nonnegative(amount, "initial_reserve_term")?;
                initial_reserve += amount;
                terms.push((event.ex_time(), event.fixed_cash(), amount));
            }
        }
        let risky_spot = spot - initial_reserve;
        positive(risky_spot, "funded_residual_equity")?;
        let mut nodes = Vec::with_capacity(times.len());
        for &time in times {
            let growth = market.forward(time)? / spot;
            positive(growth, "carry_growth")?;
            let mut a = growth * risky_spot;
            let mut b = 0.0;
            let mut c = 0.0;
            let mut event_mean_cash = None;
            for &(ex_time, cash, amount) in &terms {
                if ex_time == time {
                    event_mean_cash = Some(event_mean_cash.unwrap_or(0.0) + cash);
                } else if ex_time > time {
                    let (w, complement) = decay(model.mean_reversion(), ex_time - time);
                    let reserve = growth * amount;
                    a += reserve * complement * model.equity_linkage();
                    b += reserve * w;
                    c += reserve * complement * (1.0 - model.equity_linkage());
                }
            }
            positive(a, "equity_coefficient")?;
            nonnegative(b, "dividend_coefficient")?;
            nonnegative(c, "constant_coefficient")?;
            if let Some(cash) = event_mean_cash {
                nonnegative(cash, "event_cash_mean")?;
            }
            nodes.push(StochasticDividendNode {
                time,
                equity_coefficient: a,
                dividend_coefficient: b,
                constant: c,
                event_mean_cash,
            });
        }
        Ok(Self {
            model,
            volatility,
            times: times.to_vec().into_boxed_slice(),
            nodes: nodes.into_boxed_slice(),
            risky_spot,
            dimension,
            bergomi: None,
            rough: None,
        })
    }
    /// Flat initial forward variance, with explicit dividend/volatility correlation.
    pub fn compile_bergomi(
        market: &EquityForward,
        model: BuehlerDividendModel,
        initial_volatility: f64,
        factor: Bergomi1Factor,
        dividend_volatility_correlation: f64,
        grid: &LocalVolTimeGrid,
    ) -> Result<Self, MonteCarloError> {
        Self::compile(market, model, initial_volatility, grid)?
            .with_bergomi(factor, dividend_volatility_correlation)
    }

    pub fn compile_bergomi_two_factor(
        market: &EquityForward,
        model: BuehlerDividendModel,
        initial_volatility: f64,
        factor: Bergomi2Factor,
        dividend_volatility_correlations: [f64; 2],
        grid: &LocalVolTimeGrid,
    ) -> Result<Self, MonteCarloError> {
        Self::compile(market, model, initial_volatility, grid)?
            .with_bergomi_two_factor(factor, dividend_volatility_correlations)
    }

    /// Riemann--Liouville hybrid scheme with explicit dividend/vol-driver correlation.
    /// Rough-dividend prices and opt-in basic/rough-parameter AAD and Spot Gamma.
    pub fn compile_rough_bergomi(
        market: &EquityForward,
        model: BuehlerDividendModel,
        initial_volatility: f64,
        factor: RoughBergomi,
        dividend_volatility_correlation: f64,
        grid: &LocalVolTimeGrid,
    ) -> Result<Self, MonteCarloError> {
        Self::compile(market, model, initial_volatility, grid)?
            .with_rough_bergomi(factor, dividend_volatility_correlation)
    }

    pub(in crate::engine) fn with_rough_bergomi(
        mut self,
        factor: RoughBergomi,
        correlation: f64,
    ) -> Result<Self, MonteCarloError> {
        self.rough = Some(Arc::new(RoughDividendKernel::compile(
            self.model,
            factor,
            correlation,
            &self.times,
        )?));
        self.bergomi = None;
        self.set_dimension()?;
        Ok(self)
    }

    pub(in crate::engine) fn with_bergomi(
        mut self,
        factor: Bergomi1Factor,
        correlation: f64,
    ) -> Result<Self, MonteCarloError> {
        self.bergomi = Some(BergomiDividendKernel::one(
            self.model,
            factor,
            correlation,
            &self.times,
        )?);
        self.set_dimension()?;
        Ok(self)
    }

    pub(in crate::engine) fn with_bergomi_two_factor(
        mut self,
        factor: Bergomi2Factor,
        correlations: [f64; 2],
    ) -> Result<Self, MonteCarloError> {
        self.bergomi = Some(BergomiDividendKernel::two(
            self.model,
            factor,
            correlations,
            &self.times,
        )?);
        self.set_dimension()?;
        Ok(self)
    }

    /// Rebuild only spot-dependent escrow coefficients on the identical grid.
    /// Normalized f/Y and OU dynamics have no initial-Spot dependence in these
    /// BS/pure-SV models. Preserve the kernel and every reserved random coordinate.
    pub(in crate::engine) fn with_market_spot(
        &self,
        market: &EquityForward,
    ) -> Result<Self, MonteCarloError> {
        let grid =
            LocalVolTimeGrid::compile(self.times.to_vec(), self.times[self.times.len() - 1])?;
        let mut shifted = Self::compile(market, self.model, self.volatility, &grid)?;
        shifted.bergomi = self.bergomi.clone();
        shifted.rough = self.rough.clone();
        shifted.dimension = self.dimension;
        Ok(shifted)
    }

    fn set_dimension(&mut self) -> Result<(), StochasticDividendError> {
        self.dimension = (self.times.len() - 1)
            .checked_mul(self.random_factor_count())
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(invalid("random_dimension"))?;
        Ok(())
    }

    #[must_use]
    pub const fn random_factor_count(&self) -> usize {
        if self.rough.is_some() {
            return 4;
        }
        match &self.bergomi {
            None => 2,
            Some(k) => k.factor_count(),
        }
    }

    #[must_use]
    pub const fn scheme(&self) -> &'static str {
        if self.rough.is_some() {
            return rough::SCHEME;
        }
        match &self.bergomi {
            None => crate::models::STOCHASTIC_DIVIDEND_SCHEME,
            Some(k) => k.scheme(),
        }
    }

    pub(in crate::engine) fn hash_volatility(&self, hash: &mut blake3::Hasher) {
        if let Some(k) = &self.rough {
            k.hash_parameters(hash);
        }
        if let Some(k) = &self.bergomi {
            k.hash_parameters(hash);
        }
    }

    #[must_use]
    pub fn times(&self) -> &[f64] {
        &self.times
    }
    #[must_use]
    pub fn nodes(&self) -> &[StochasticDividendNode] {
        &self.nodes
    }
    #[must_use]
    pub const fn risky_spot(&self) -> f64 {
        self.risky_spot
    }
    #[must_use]
    pub const fn random_dimension(&self) -> u32 {
        self.dimension
    }
    #[must_use]
    pub const fn model(&self) -> BuehlerDividendModel {
        self.model
    }
    /// Independent standardized normals, step-major with `random_factor_count()`
    /// entries per step. Rough always reserves four, even at zero eta or H=1/2.
    pub fn evolve_path(
        &self,
        normals: &[f64],
    ) -> Result<Vec<BuehlerDividendState>, StochasticDividendError> {
        if normals.len() != self.dimension as usize {
            return Err(invalid("normal_count"));
        }
        if normals.iter().any(|z| !z.is_finite()) {
            return Err(invalid("normal"));
        }
        if let Some(kernel) = &self.rough {
            return kernel.evolve(self.model, self.volatility, &self.times, normals);
        }
        if let Some(kernel) = &self.bergomi {
            return kernel.evolve(self.model, self.volatility, &self.times, normals);
        }
        let mut states = Vec::with_capacity(self.times.len());
        let mut state = BuehlerDividendState::initial();
        states.push(state);
        for (pair, z) in self.times.windows(2).zip(normals.as_chunks::<2>().0) {
            state = self
                .model
                .evolve(state, self.volatility, pair[1] - pair[0], [z[0], z[1]])?;
            states.push(state);
        }
        Ok(states)
    }
}
