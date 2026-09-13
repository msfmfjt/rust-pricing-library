//! Explicit escrowed cash dividends backed by stochastic Hull–White bonds.

use crate::hull_white::{b, hw_valid};
use crate::{HullWhite1Factor, HullWhiteError};
use pricing_market::{DiscountCurve, EquityForward, LogLinearDiscountCurve};

pub const HULL_WHITE_CASH_DIVIDEND_MODEL: &str = "escrowed-hw-bonds-v1";

#[derive(Clone, Debug)]
struct BondTerm {
    amount: f64,
    deterministic_amount: f64,
    duration: f64,
    maturity: f64,
}

/// Adjoint seeds for deterministic node coefficients. Dividend quotes and the
/// Hull–White parameters are fixed; bond amounts include the convexity factor.
#[derive(Clone, Debug)]
pub struct HullWhiteDividendNodeAdjoints {
    pub scale: f64,
    pub deterministic_reserve: f64,
    pub bond_amounts: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct HullWhiteDividendMarketAdjoints {
    pub spot: f64,
    /// Derivatives with respect to log discount factors; time-zero anchors are fixed.
    pub discount_log_df: Vec<f64>,
    pub dividend_log_df: Vec<f64>,
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
    pub fn zero_adjoints(&self) -> HullWhiteDividendNodeAdjoints {
        HullWhiteDividendNodeAdjoints {
            scale: 0.0,
            deterministic_reserve: 0.0,
            bond_amounts: vec![0.0; self.bonds.len()],
        }
    }
    fn reverse_reserve(
        &self,
        x: f64,
        amount_bar: f64,
        loading_bar: f64,
        out: &mut HullWhiteDividendNodeAdjoints,
    ) -> Result<(), HullWhiteError> {
        if out.bond_amounts.len() != self.bonds.len() {
            return Err(invalid("dividend_adjoint_shape"));
        }
        for (term, bar) in self.bonds.iter().zip(&mut out.bond_amounts) {
            *bar += (-term.duration * x).exp()
                * (amount_bar - self.rate_volatility * term.duration * loading_bar);
        }
        Ok(())
    }
    /// Reverse F and zeta at fixed centered rate state. Returns the risky-state seed.
    pub fn reverse_target(
        &self,
        risky: f64,
        rate_factor: f64,
        f_bar: f64,
        zeta_bar: f64,
        out: &mut HullWhiteDividendNodeAdjoints,
    ) -> Result<f64, HullWhiteError> {
        let (f, zeta) = self.target_state(risky, rate_factor)?;
        out.scale -= (f_bar * (f - risky) + zeta_bar * zeta) / self.scale;
        out.deterministic_reserve -= f_bar / self.scale;
        self.reverse_reserve(rate_factor, f_bar / self.scale, zeta_bar / self.scale, out)?;
        Ok(f_bar)
    }
    /// Reverse physical observations, holding fixed both cash and proportional payouts.
    pub fn reverse_spots(
        &self,
        risky: f64,
        rate_factor: f64,
        post_bar: f64,
        pre_bar: f64,
        out: &mut HullWhiteDividendNodeAdjoints,
    ) -> Result<f64, HullWhiteError> {
        let total = if let Some((_, beta)) = self.event {
            post_bar + pre_bar / (1.0 - beta)
        } else if pre_bar == 0.0 {
            post_bar
        } else {
            return Err(invalid("unexpected_pre_dividend_adjoint"));
        };
        out.scale += total * risky;
        self.reverse_reserve(rate_factor, total, 0.0, out)?;
        Ok(total * self.scale)
    }
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
    market: EquityForward,
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
                    deterministic_amount: det,
                    duration: b(rates.mean_reversion(), e.ex_time() - t),
                    maturity: e.ex_time(),
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
            market: market.clone(),
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
    pub fn zero_adjoints(&self) -> Vec<HullWhiteDividendNodeAdjoints> {
        self.nodes
            .iter()
            .map(HullWhiteDividendNode::zero_adjoints)
            .collect()
    }
    /// Transpose the deterministic reserve/scale construction into Spot and both
    /// initial curves, including every supplied post-expiry dividend. Centered HW
    /// states, relative bonds and relative discounts do not depend on these curves.
    pub fn reverse_market(
        &self,
        seeds: &[HullWhiteDividendNodeAdjoints],
    ) -> Result<HullWhiteDividendMarketAdjoints, HullWhiteError> {
        if seeds.len() != self.nodes.len() {
            return Err(invalid("dividend_adjoint_shape"));
        }
        let p = self.market.discount_curve();
        let q = self.market.dividend_curve();
        let mut out = HullWhiteDividendMarketAdjoints {
            spot: 0.0,
            discount_log_df: vec![0.0; p.times().len()],
            dividend_log_df: vec![0.0; q.times().len()],
        };
        let mut reserve0_bar = 0.0;
        for (node, seed) in self.nodes.iter().zip(seeds) {
            if seed.bond_amounts.len() != node.bonds.len() {
                return Err(invalid("dividend_adjoint_shape"));
            }
            let c = seed.scale * node.scale;
            out.spot += c * (1.0 / self.risky_spot - 1.0 / self.initial_spot);
            reserve0_bar -= c / self.risky_spot;
            transpose_log_curve(p, node.time, -c, &mut out.discount_log_df)?;
            transpose_log_curve(q, node.time, c, &mut out.dividend_log_df)?;
            for (term, &bar) in node.bonds.iter().zip(&seed.bond_amounts) {
                let weight =
                    bar * term.amount + seed.deterministic_reserve * term.deterministic_amount;
                transpose_log_curve(p, term.maturity, weight, &mut out.discount_log_df)?;
                transpose_log_curve(p, node.time, -weight, &mut out.discount_log_df)?;
                transpose_log_curve(q, node.time, weight, &mut out.dividend_log_df)?;
                transpose_log_curve(q, term.maturity, -weight, &mut out.dividend_log_df)?;
            }
        }
        let mut beta_product = 1.0;
        if let Some(schedule) = self.market.discrete_dividends() {
            for event in schedule.events() {
                beta_product *= 1.0 - event.beta();
                let t = event.ex_time();
                let bar = reserve0_bar * event.fixed_cash() / beta_product * p.discount(t)?
                    / q.discount(t)?;
                transpose_log_curve(p, t, bar, &mut out.discount_log_df)?;
                transpose_log_curve(q, t, -bar, &mut out.dividend_log_df)?;
            }
        }
        for &v in std::iter::once(&out.spot)
            .chain(&out.discount_log_df)
            .chain(&out.dividend_log_df)
        {
            hw_valid(v, "dividend_market_adjoint", 0, false)?;
        }
        Ok(out)
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

/// Transpose the log-linear interpolation and its terminal-segment extrapolation.
/// The valuation anchor is fixed at log(1)=0 and never receives a risk seed.
pub fn transpose_log_curve(
    curve: &LogLinearDiscountCurve,
    t: f64,
    seed: f64,
    out: &mut [f64],
) -> Result<(), HullWhiteError> {
    let times = curve.times();
    hw_valid(t, "curve_adjoint_time", 0, true)?;
    hw_valid(seed, "curve_adjoint_seed", 0, false)?;
    if out.len() != times.len() {
        return Err(invalid("curve_adjoint_shape"));
    }
    match times.binary_search_by(|v| v.total_cmp(&t)) {
        Ok(i) => {
            if i != 0 {
                out[i] += seed;
            }
        }
        Err(i) => {
            let right = i.min(times.len() - 1).max(1);
            let left = right - 1;
            let w = (t - times[left]) / (times[right] - times[left]);
            if left != 0 {
                out[left] += seed * (1.0 - w);
            }
            out[right] += seed * w;
        }
    }
    Ok(())
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
