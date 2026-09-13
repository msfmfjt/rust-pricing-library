//! Explicit escrowed cash dividends backed by stochastic Hull–White bonds.

use crate::hull_white::{b, hw_valid};
use crate::{HullWhite1Factor, HullWhiteError};
use pricing_market::{DiscountCurve, EquityForward};

pub const HULL_WHITE_CASH_DIVIDEND_MODEL: &str = "escrowed-hw-bonds-v1";

#[derive(Clone, Debug)]
struct BondTerm {
    amount: f64,
    duration: f64,
}

/// Post-event coordinate S = scale * normalized_risky_equity + A(t,x).
/// Target coordinate F = normalized_risky_equity + (A(t,x)-A0(t))/scale.
#[derive(Clone, Debug)]
pub struct HullWhiteDividendNode {
    time: f64,
    scale: f64,
    deterministic_reserve: f64,
    bonds: Box<[BondTerm]>,
    rate_volatility: f64,
    event: Option<(f64, f64)>,
}
impl HullWhiteDividendNode {
    #[must_use]
    pub const fn time(&self) -> f64 {
        self.time
    }
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }
    #[must_use]
    pub const fn deterministic_reserve(&self) -> f64 {
        self.deterministic_reserve
    }
    /// Reserve value and its instantaneous dW_rate coefficient.
    pub fn reserve(&self, rate_factor: f64) -> Result<(f64, f64), HullWhiteError> {
        hw_valid(rate_factor, "dividend_rate_factor", 0, false)?;
        let (mut amount, mut duration_amount) = (0.0, 0.0);
        for term in &self.bonds {
            let v = term.amount * (-term.duration * rate_factor).exp();
            amount += v;
            duration_amount += term.duration * v;
        }
        let loading = -self.rate_volatility * duration_amount;
        hw_valid(amount, "dividend_reserve", 0, true)?;
        hw_valid(loading, "dividend_rate_loading", 0, false)?;
        Ok((amount, loading))
    }
    pub fn target_state(&self, risky: f64, rate_factor: f64) -> Result<(f64, f64), HullWhiteError> {
        let (reserve, loading) = self.reserve(rate_factor)?;
        let state = risky + (reserve - self.deterministic_reserve) / self.scale;
        positive(state, "cash_dividend_target_state")?;
        Ok((state, loading / self.scale))
    }
    pub fn spots(
        &self,
        risky: f64,
        rate_factor: f64,
    ) -> Result<(f64, Option<f64>), HullWhiteError> {
        positive(risky, "normalized_risky_equity")?;
        let post = self.scale * risky + self.reserve(rate_factor)?.0;
        positive(post, "post_dividend_spot")?;
        let pre = self.event.map(|(cash, beta)| (post + cash) / (1.0 - beta));
        if let Some(pre) = pre {
            positive(pre, "pre_dividend_spot")?;
        }
        Ok((post, pre))
    }
}

#[derive(Clone, Debug)]
pub struct HullWhiteDividendPlan {
    rates: HullWhite1Factor,
    initial_spot: f64,
    risky_spot: f64,
    nodes: Box<[HullWhiteDividendNode]>,
}
impl HullWhiteDividendPlan {
    pub fn new(
        rates: &HullWhite1Factor,
        market: &EquityForward,
        times: &[f64],
    ) -> Result<Self, HullWhiteError> {
        if times.len() < 2 || times[0] != 0.0 {
            return Err(invalid("dividend_time_grid"));
        }
        for (i, &t) in times.iter().enumerate() {
            hw_valid(t, "dividend_time", i, true)?;
            if i > 0 && t <= times[i - 1] {
                return Err(invalid("dividend_time_order"));
            }
        }
        let events = market.discrete_dividends().map_or(&[][..], |d| d.events());
        let horizon = times[times.len() - 1];
        for e in events.iter().filter(|e| e.ex_time() <= horizon) {
            if !times.contains(&e.ex_time()) {
                return Err(invalid("missing_dividend_time"));
            }
        }
        // A(0-) includes any event at t=0. Spot is the pre-event input, matching
        // the existing schedule convention; node zero is post-event.
        let (mut beta_product, mut reserve0) = (1.0, 0.0);
        for e in events {
            beta_product *= 1.0 - e.beta();
            let p = market.discount_curve().evaluate(e.ex_time())?.discount;
            let q = market.dividend_curve().evaluate(e.ex_time())?.discount;
            reserve0 += e.fixed_cash() / beta_product * p / q;
        }
        let initial_spot = market.spot().get();
        let risky_spot = initial_spot - reserve0;
        positive(risky_spot, "positive_escrowed_risky_spot")?;
        let mut nodes = Vec::with_capacity(times.len());
        for &t in times {
            let p = market.discount_curve().evaluate(t)?.discount;
            let q = market.dividend_curve().evaluate(t)?.discount;
            let past_beta: f64 = events
                .iter()
                .filter(|e| e.ex_time() <= t)
                .map(|e| 1.0 - e.beta())
                .product();
            let scale = past_beta * risky_spot / initial_spot * q / p;
            positive(scale, "dividend_equity_scale")?;
            let (mut future_beta, mut deterministic_reserve) = (1.0, 0.0);
            let mut bonds = Vec::new();
            for e in events.iter().filter(|e| e.ex_time() > t) {
                future_beta *= 1.0 - e.beta();
                if e.fixed_cash() == 0.0 {
                    continue;
                }
                let pj = market.discount_curve().evaluate(e.ex_time())?.discount;
                let qj = market.dividend_curve().evaluate(e.ex_time())?.discount;
                let det = e.fixed_cash() / future_beta * pj / p * q / qj;
                deterministic_reserve += det;
                bonds.push(BondTerm {
                    amount: det * rates.relative_bond(t, e.ex_time(), 0.0)?,
                    duration: b(rates.mean_reversion(), e.ex_time() - t),
                });
            }
            hw_valid(
                deterministic_reserve,
                "deterministic_dividend_reserve",
                0,
                true,
            )?;
            let sigma_index = rates.volatility_times().partition_point(|v| *v <= t) - 1;
            let event = events
                .iter()
                .find(|e| e.ex_time() == t)
                .map(|e| (e.fixed_cash(), e.beta()));
            nodes.push(HullWhiteDividendNode {
                time: t,
                scale,
                deterministic_reserve,
                bonds: bonds.into(),
                rate_volatility: rates.volatilities()[sigma_index],
                event,
            });
        }
        Ok(Self {
            rates: rates.clone(),
            initial_spot,
            risky_spot,
            nodes: nodes.into(),
        })
    }
    #[must_use]
    pub const fn initial_spot(&self) -> f64 {
        self.initial_spot
    }
    #[must_use]
    pub fn rates(&self) -> &HullWhite1Factor {
        &self.rates
    }
    #[must_use]
    pub const fn risky_spot(&self) -> f64 {
        self.risky_spot
    }
    #[must_use]
    pub fn nodes(&self) -> &[HullWhiteDividendNode] {
        &self.nodes
    }
    #[must_use]
    pub fn fingerprint_bytes(&self) -> Vec<u8> {
        let mut bytes = HULL_WHITE_CASH_DIVIDEND_MODEL.as_bytes().to_vec();
        bytes.extend_from_slice(&self.initial_spot.to_bits().to_be_bytes());
        bytes.extend_from_slice(&self.risky_spot.to_bits().to_be_bytes());
        bytes.extend_from_slice(&(self.nodes.len() as u64).to_be_bytes());
        for n in &self.nodes {
            for v in [n.time, n.scale, n.deterministic_reserve, n.rate_volatility] {
                bytes.extend_from_slice(&v.to_bits().to_be_bytes());
            }
            bytes.extend_from_slice(&(n.bonds.len() as u64).to_be_bytes());
            for term in &n.bonds {
                for v in [term.amount, term.duration] {
                    bytes.extend_from_slice(&v.to_bits().to_be_bytes());
                }
            }
            bytes.push(u8::from(n.event.is_some()));
            if let Some((cash, beta)) = n.event {
                for v in [cash, beta] {
                    bytes.extend_from_slice(&v.to_bits().to_be_bytes());
                }
            }
        }
        bytes
    }
}
fn invalid(field: &'static str) -> HullWhiteError {
    HullWhiteError::InvalidInput { field, index: 0 }
}
fn positive(v: f64, field: &'static str) -> Result<(), HullWhiteError> {
    hw_valid(v, field, 0, true)?;
    if v == 0.0 {
        return Err(invalid(field));
    }
    Ok(())
}
