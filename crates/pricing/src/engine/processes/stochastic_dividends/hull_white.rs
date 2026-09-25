//! Constant-volatility Buehler equity/dividends under the domestic money-market
//! measure, with exact joint Hull--White rate and integrated-rate innovations.
//! Cash means are Q means, NOT forward-measure dividend quotes.

pub(in crate::engine) mod correlation_sensitivity;
pub(in crate::engine) mod rate_sensitivity;
pub(in crate::engine) mod reverse;

use super::{BuehlerDividendModel, BuehlerDividendState};
use crate::MonteCarloError;
use crate::engine::processes::hull_white::covariance_loading;
use crate::market::{DiscountCurve, EquityForward};
use crate::mc::LocalVolTimeGrid;
use crate::models::hull_white::b;
use crate::models::stochastic_dividends::{invalid, nonnegative, positive};
use crate::models::{HullWhite1Factor, HybridCorrelation, StochasticDividendError};

pub const STOCHASTIC_DIVIDEND_HW_SCHEME: &str = "buehler-bs-hw-conditional-cash-v1";
// Compile-time numerical integration is independent of the simulation grid.
const QUADRATURE_ABS: f64 = 1e-12;
const QUADRATURE_REL: f64 = 1e-11;
const QUADRATURE_DEPTH: u32 = 20;
const MAX_RESERVE_TERMS: usize = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StochasticDividendHullWhiteState {
    factors: BuehlerDividendState,
    rate_factor: f64,
    integrated_rate_factor: f64,
}
impl StochasticDividendHullWhiteState {
    pub fn new(
        factors: BuehlerDividendState,
        rate_factor: f64,
        integrated_rate_factor: f64,
    ) -> Result<Self, StochasticDividendError> {
        if !rate_factor.is_finite() || !integrated_rate_factor.is_finite() {
            return Err(invalid("rate_state"));
        }
        Ok(Self {
            factors,
            rate_factor,
            integrated_rate_factor,
        })
    }
    pub const fn initial() -> Self {
        Self {
            factors: BuehlerDividendState::initial(),
            rate_factor: 0.0,
            integrated_rate_factor: 0.0,
        }
    }
    pub const fn factors(self) -> BuehlerDividendState {
        self.factors
    }
    pub const fn rate_factor(self) -> f64 {
        self.rate_factor
    }
    pub const fn integrated_rate_factor(self) -> f64 {
        self.integrated_rate_factor
    }
}

#[derive(Clone, Debug)]
struct Step {
    dt: f64,
    decay: f64,
    integral_loading: f64,
    noise: [[f64; 4]; 4],
}
#[derive(Clone, Debug)]
struct CashClaim {
    index: usize,
    // Collateral price at x=0, with state coefficients still to be applied.
    amount: f64,
    duration: f64,
    coefficients: [f64; 3], // f, Y, 1
    carry_weight: f64,
}
impl CashClaim {
    fn value(
        &self,
        state: StochasticDividendHullWhiteState,
    ) -> Result<f64, StochasticDividendError> {
        let [a, b, c] = self.coefficients;
        let value = self.amount
            * (-self.duration * state.rate_factor).exp()
            * (a * state.factors.equity() + b * state.factors.dividend() + c);
        nonnegative(value, "discounted_dividend_claim")?;
        Ok(value)
    }
}
#[derive(Clone, Debug)]
struct Node {
    residual_growth: f64,
    half_integral_variance: f64,
    claims: Vec<CashClaim>,
    event_cash: Option<f64>,
}

/// All supplied cash means are funded, including events beyond the option.
/// The reserve uses conditional DISCOUNTED cash, not P(t,T)*E^Q_t[cash].
#[derive(Clone, Debug)]
pub struct StochasticDividendHullWhitePathPlan {
    model: BuehlerDividendModel,
    volatility: f64,
    times: Box<[f64]>,
    steps: Box<[Step]>,
    nodes: Box<[Node]>,
    cash_times: Box<[f64]>,
    cash_means: Box<[f64]>,
    initial_claims: Box<[f64]>,
    initial_forwards: Box<[f64]>,
    risky_spot: f64,
    dimension: u32,
}
impl StochasticDividendHullWhitePathPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn compile_bs(
        market: &EquityForward,
        model: BuehlerDividendModel,
        volatility: f64,
        rates: &HullWhite1Factor,
        equity_rate_correlation: f64,
        dividend_rate_correlation: f64,
        grid: &LocalVolTimeGrid,
    ) -> Result<Self, MonteCarloError> {
        nonnegative(volatility, "equity_volatility")?;
        // Here the second driver in the existing Gaussian helper is W_D,
        // not a stochastic-volatility factor: its mean reversion is zero.
        let corr = HybridCorrelation::new(
            model.equity_dividend_correlation(),
            equity_rate_correlation,
            dividend_rate_correlation,
        )?;
        let times = grid.nodes();
        let dimension = (times.len() - 1)
            .checked_mul(4)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(invalid("random_dimension"))?;
        let mut cash = Vec::new();
        if let Some(schedule) = market.discrete_dividends() {
            for event in schedule.events() {
                if event.beta() != 0.0 {
                    return Err(StochasticDividendError::Unsupported {
                        feature: "proportional cash mixtures in stochastic-dividend HW",
                    }
                    .into());
                }
                if event.ex_time() <= 0.0 {
                    return Err(invalid("strictly_future_ex_date").into());
                }
                if event.ex_time() <= times[times.len() - 1]
                    && times
                        .binary_search_by(|t| t.total_cmp(&event.ex_time()))
                        .is_err()
                {
                    return Err(invalid("missing_dividend_time").into());
                }
                cash.push((event.ex_time(), event.fixed_cash()));
            }
        }
        if times
            .len()
            .checked_mul(cash.len())
            .is_none_or(|n| n > MAX_RESERVE_TERMS)
        {
            return Err(invalid("hw_reserve_resource_limit").into());
        }
        let mut steps = Vec::new();
        for pair in times.windows(2) {
            let transition = rates.transition(pair[0], pair[1], 0.0, corr)?;
            steps.push(Step {
                dt: pair[1] - pair[0],
                decay: transition.rate_decay,
                integral_loading: transition.integral_loading,
                noise: covariance_loading(transition.covariance)?,
            });
        }
        let mut nodes = Vec::new();
        let mut initial_claims = vec![0.0; cash.len()];
        let mut initial_forwards = vec![0.0; cash.len()];
        let mut initial_reserve = 0.0;
        for (i, &time) in times.iter().enumerate() {
            let mut claims = Vec::new();
            let mut event_cash = None;
            let repo = market.dividend_curve().discount(time)?;
            let discount = market.discount_curve().discount(time)?;
            for (j, &(maturity, mean)) in cash.iter().enumerate() {
                if maturity == time {
                    event_cash = Some(event_cash.unwrap_or(0.0) + mean);
                }
                if maturity <= time {
                    continue;
                }
                let coefficients = cash_coefficients(
                    model,
                    volatility,
                    rates,
                    equity_rate_correlation,
                    dividend_rate_correlation,
                    time,
                    maturity,
                )?;
                let claim = CashClaim {
                    index: j,
                    coefficients,
                    amount: mean
                        * rates.bond_price(market.discount_curve(), time, maturity, 0.0)?,
                    duration: b(rates.mean_reversion(), maturity - time),
                    carry_weight: repo / market.dividend_curve().discount(maturity)?,
                };
                positive(claim.carry_weight, "repo_carry_weight")?;
                if i == 0 {
                    initial_claims[j] = claim.value(StochasticDividendHullWhiteState::initial())?;
                    initial_forwards[j] =
                        initial_claims[j] / market.discount_curve().discount(maturity)?;
                    initial_reserve += claim.carry_weight * initial_claims[j];
                }
                claims.push(claim);
            }
            let growth = repo / discount;
            positive(growth, "residual_carry_growth")?;
            nodes.push(Node {
                residual_growth: growth,
                half_integral_variance: 0.5 * rates.integrated_variance(time)?,
                claims,
                event_cash,
            });
        }
        let risky_spot = market.spot().get() - initial_reserve;
        positive(risky_spot, "funded_residual_equity")?;
        Ok(Self {
            model,
            volatility,
            times: times.to_vec().into(),
            steps: steps.into(),
            nodes: nodes.into(),
            cash_times: cash.iter().map(|x| x.0).collect(),
            cash_means: cash.iter().map(|x| x.1).collect(),
            initial_claims: initial_claims.into(),
            initial_forwards: initial_forwards.into(),
            risky_spot,
            dimension,
        })
    }
    pub fn times(&self) -> &[f64] {
        &self.times
    }
    pub const fn random_dimension(&self) -> u32 {
        self.dimension
    }
    pub const fn random_factor_count(&self) -> usize {
        4
    }
    pub const fn risky_spot(&self) -> f64 {
        self.risky_spot
    }
    pub fn cash_times(&self) -> &[f64] {
        &self.cash_times
    }
    pub fn initial_dividend_claim_values(&self) -> &[f64] {
        &self.initial_claims
    }
    pub fn initial_dividend_forwards(&self) -> &[f64] {
        &self.initial_forwards
    }

    pub fn evolve_path(
        &self,
        normals: &[f64],
    ) -> Result<Vec<StochasticDividendHullWhiteState>, MonteCarloError> {
        if normals.len() != self.dimension as usize || normals.iter().any(|z| !z.is_finite()) {
            return Err(invalid("normal_vector").into());
        }
        let mut states = Vec::with_capacity(self.times.len());
        let mut state = StochasticDividendHullWhiteState::initial();
        states.push(state);
        for (step, z) in self.steps.iter().zip(normals.as_chunks::<4>().0.iter()) {
            let rate_noise: f64 = step.noise[2].iter().zip(z).map(|(a, b)| a * b).sum();
            let integral_noise: f64 = step.noise[3].iter().zip(z).map(|(a, b)| a * b).sum();
            let factors =
                self.model
                    .evolve(state.factors, self.volatility, step.dt, [z[0], z[1]])?;
            state = StochasticDividendHullWhiteState::new(
                factors,
                step.decay * state.rate_factor + rate_noise,
                state.integrated_rate_factor
                    + step.integral_loading * state.rate_factor
                    + integral_noise,
            )?;
            states.push(state);
        }
        Ok(states)
    }
    /// (post-event physical Spot, optional pre-event Spot).
    pub fn spots(
        &self,
        index: usize,
        state: StochasticDividendHullWhiteState,
    ) -> Result<(f64, Option<f64>), StochasticDividendError> {
        self.spots_with_risky_spot(index, state, self.risky_spot)
    }

    /// Initial funding at another Spot, with every Q cash mean and conditional
    /// claim fixed. Sum in the original compile order, including future cash.
    pub(in crate::engine) fn risky_spot_at(&self, spot: f64) -> Result<f64, StochasticDividendError> {
        positive(spot, "gamma_shifted_spot")?;
        let mut reserve = 0.0;
        for claim in &self.nodes[0].claims {
            reserve += claim.carry_weight * self.initial_claims[claim.index];
        }
        let risky_spot = spot - reserve;
        positive(risky_spot, "funded_residual_equity")?;
        Ok(risky_spot)
    }

    /// Spot-only scenarios reuse the normalized factors and conditional cash
    /// claims. The caller validates the funded risky spot before sampling.
    pub(in crate::engine) fn spots_with_risky_spot(
        &self,
        index: usize,
        state: StochasticDividendHullWhiteState,
        risky_spot: f64,
    ) -> Result<(f64, Option<f64>), StochasticDividendError> {
        let node = self.nodes.get(index).ok_or(invalid("time_node_index"))?;
        let mut post = risky_spot
            * node.residual_growth
            * (state.integrated_rate_factor + node.half_integral_variance).exp()
            * state.factors.equity();
        positive(post, "residual_equity_state")?;
        for claim in &node.claims {
            post += claim.carry_weight * claim.value(state)?;
        }
        positive(post, "post_dividend_spot")?;
        let pre = node
            .event_cash
            .map(|mean| post + mean * state.factors.dividend());
        if let Some(pre) = pre {
            positive(pre, "pre_dividend_spot")?;
        }
        Ok((post, pre))
    }
    /// Collateral value of one cash payment, before settlement on its ex-date.
    /// Once the event is past, the claim value is zero. Indices follow cash_times.
    pub fn dividend_claim_value(
        &self,
        node_index: usize,
        cash_index: usize,
        state: StochasticDividendHullWhiteState,
    ) -> Result<f64, StochasticDividendError> {
        let &maturity = self
            .cash_times
            .get(cash_index)
            .ok_or(invalid("cash_index"))?;
        let &time = self
            .times
            .get(node_index)
            .ok_or(invalid("time_node_index"))?;
        if time > maturity {
            return Ok(0.0);
        }
        if time == maturity {
            // Settlement is the actual mean-scaled factor, not a forward quote.
            let mean = self.cash_mean(cash_index)?;
            let value = mean * state.factors.dividend();
            nonnegative(value, "settled_cash")?;
            return Ok(value);
        }
        self.nodes[node_index]
            .claims
            .iter()
            .find(|c| c.index == cash_index)
            .ok_or(invalid("cash_index"))?
            .value(state)
    }
    fn cash_mean(&self, index: usize) -> Result<f64, StochasticDividendError> {
        self.cash_means
            .get(index)
            .copied()
            .ok_or(invalid("cash_index"))
    }
}

// L(t,T) = integral_t^T sigma_r(v) B_a(T-v) dv. It is deterministic,
// includes all rate-volatility breakpoints, and is NOT multiplied by rho.
fn rate_brownian_integral(
    rates: &HullWhite1Factor,
    t: f64,
    maturity: f64,
) -> Result<f64, MonteCarloError> {
    Ok(rates
        .transition(t, maturity, 0.0, HybridCorrelation::new(0.0, 1.0, 0.0)?)?
        .covariance[3][0])
}

/// E_t[exp(-integral r) Y_T] / P(t,T) = A*f_t + B*Y_t + C.
/// This is a Gaussian exponential tilt of the variation-of-constants solution
/// of the Buehler SDE; it does not assume independent discounting and dividends.
#[allow(clippy::too_many_arguments)]
fn cash_coefficients(
    model: BuehlerDividendModel,
    sigma: f64,
    rates: &HullWhite1Factor,
    rho_f: f64,
    rho_d: f64,
    t: f64,
    maturity: f64,
) -> Result<[f64; 3], MonteCarloError> {
    let tau = maturity - t;
    let k = model.mean_reversion();
    let alpha = model.equity_linkage();
    let cf = sigma * rho_f;
    let cd = model.dividend_volatility() * rho_d;
    if rates.is_deterministic() || (cf == 0.0 && cd == 0.0) || (alpha == 0.0 && cd == 0.0) {
        let decay = (-k * tau).exp();
        let complement = -(-k * tau).exp_m1();
        return Ok([alpha * complement, decay, (1.0 - alpha) * complement]);
    }
    let total = rate_brownian_integral(rates, t, maturity)?;
    let by = (-k * tau - cd * total).exp();
    if k == 0.0 {
        return Ok([0.0, by, 0.0]);
    }
    // v=T-u. Split numerical integration at each deterministic sigma_r knot.
    let f = |v: f64| -> Result<[f64; 2], MonteCarloError> {
        let tail = rate_brownian_integral(rates, maturity - v, maturity)?;
        let common = -k * v - cd * tail;
        let value = [
            if alpha == 0.0 {
                0.0
            } else {
                k * (common - cf * (total - tail)).exp()
            },
            if alpha == 1.0 { 0.0 } else { k * common.exp() },
        ];
        for &x in &value {
            nonnegative(x, "cash_quadrature_integrand")?;
        }
        Ok(value)
    };
    let mut cuts = vec![0.0, tau];
    cuts.extend(
        rates
            .volatility_times()
            .iter()
            .filter(|&&u| u > t && u < maturity)
            .map(|&u| maturity - u),
    );
    cuts.sort_by(f64::total_cmp);
    let mut result = [0.0; 2];
    for pair in cuts.windows(2) {
        let [lo, hi] = [pair[0], pair[1]];
        let mid = (lo + hi) * 0.5;
        let samples = [f(lo)?, f(mid)?, f(hi)?];
        let whole = simpson(lo, hi, samples);
        let part = integrate(
            &f,
            lo,
            hi,
            samples,
            whole,
            QUADRATURE_ABS * (hi - lo) / tau,
            QUADRATURE_DEPTH,
        )?;
        for (value, increment) in result.iter_mut().zip(part) {
            *value += increment;
        }
    }
    let result = [alpha * result[0], by, (1.0 - alpha) * result[1]];
    for &x in &result {
        nonnegative(x, "conditional_cash_coefficient")?;
    }
    Ok(result)
}
fn simpson(lo: f64, hi: f64, y: [[f64; 2]; 3]) -> [f64; 2] {
    std::array::from_fn(|j| (hi - lo) / 6.0 * (y[0][j] + 4.0 * y[1][j] + y[2][j]))
}
#[allow(clippy::too_many_arguments)]
fn integrate(
    f: &impl Fn(f64) -> Result<[f64; 2], MonteCarloError>,
    lo: f64,
    hi: f64,
    y: [[f64; 2]; 3],
    whole: [f64; 2],
    absolute: f64,
    depth: u32,
) -> Result<[f64; 2], MonteCarloError> {
    let mid = (lo + hi) * 0.5;
    let left_y = [y[0], f((lo + mid) * 0.5)?, y[1]];
    let right_y = [y[1], f((mid + hi) * 0.5)?, y[2]];
    let left = simpson(lo, mid, left_y);
    let right = simpson(mid, hi, right_y);
    let fine: [f64; 2] = std::array::from_fn(|j| left[j] + right[j]);
    if (0..2)
        .all(|j| (fine[j] - whole[j]).abs() <= 15.0 * (absolute + QUADRATURE_REL * fine[j].abs()))
    {
        return Ok(std::array::from_fn(|j| {
            fine[j] + (fine[j] - whole[j]) / 15.0
        }));
    }
    if depth == 0 || mid == lo || mid == hi {
        return Err(invalid("cash_quadrature_convergence").into());
    }
    let l = integrate(f, lo, mid, left_y, left, absolute * 0.5, depth - 1)?;
    let r = integrate(f, mid, hi, right_y, right, absolute * 0.5, depth - 1)?;
    Ok(std::array::from_fn(|j| l[j] + r[j]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::CurveId;
    use crate::market::LogLinearDiscountCurve;

    #[test]
    fn gaussian_tilt_coefficients_satisfy_discounted_cash_backward_equation() {
        let curve = LogLinearDiscountCurve::new(
            CurveId::new(10),
            vec![0.0, 2.0],
            vec![1.0, (-0.1_f64).exp()],
        )
        .unwrap();
        let (t, maturity, sigma, nu, rf, rd) = (0.31, 1.43, 0.2, 0.35, 0.25, -0.2);
        let model = BuehlerDividendModel::new(0.7, 0.6, nu, -0.25).unwrap();
        let (f, y, x) = (1.3, 0.9, 0.012);
        for a in [0.0, 0.4] {
            let sr = 0.04;
            let rates = HullWhite1Factor::new(a, vec![0.0], vec![sr]).unwrap();
            let price = |time| {
                let c = cash_coefficients(model, sigma, &rates, rf, rd, time, maturity).unwrap();
                rates.bond_price(&curve, time, maturity, x).unwrap() * (c[0] * f + c[1] * y + c[2])
            };
            let c = cash_coefficients(model, sigma, &rates, rf, rd, t, maturity).unwrap();
            let p = price(t);
            let bond = rates.bond_price(&curve, t, maturity, x).unwrap();
            let df = bond * c[0];
            let dy = bond * c[1];
            let duration = b(a, maturity - t);
            let h = 1e-5;
            let time_derivative = (price(t + h) - price(t - h)) / (2.0 * h);
            let rate = 0.05 + x + rates.rate_shift(t).unwrap();
            // Other second derivatives vanish because the claim is affine in f,Y.
            let residual = time_derivative
                + model.mean_reversion()
                    * (model.equity_linkage() * f + 1.0 - model.equity_linkage() - y)
                    * dy
                + a * x * duration * p
                + 0.5 * sr * sr * duration * duration * p
                - rf * sigma * f * sr * duration * df
                - rd * nu * y * sr * duration * dy
                - rate * p;
            assert!(residual.abs() < 2e-8, "a={a}, PDE residual={residual}");
        }
    }

    #[test]
    fn piecewise_joint_rate_covariance_matches_direct_kernel_quadrature() {
        let rates = HullWhite1Factor::new(0.4, vec![0.0, 0.35], vec![0.03, 0.07]).unwrap();
        let corr = HybridCorrelation::new(-0.25, 0.2, -0.3).unwrap();
        let c = rates.transition(0.1, 0.9, 0.0, corr).unwrap().covariance;
        let l = covariance_loading(c).unwrap();
        let mut reference = [[0.0; 4]; 4];
        for (lo, hi, sr) in [(0.1, 0.35, 0.03), (0.35, 0.9, 0.07)] {
            let n = 2000;
            let h = (hi - lo) / n as f64;
            for index in 0..=n {
                let u = lo + index as f64 * h;
                let z = 0.9 - u;
                let kernel = [
                    1.0,
                    1.0,
                    sr * (-0.4_f64 * z).exp(),
                    sr * (-(-0.4 * z).exp_m1()) / 0.4,
                ];
                let rho = [
                    [1.0, -0.25, 0.2, 0.2],
                    [-0.25, 1.0, -0.3, -0.3],
                    [0.2, -0.3, 1.0, 1.0],
                    [0.2, -0.3, 1.0, 1.0],
                ];
                let w = if index == 0 || index == n {
                    1.0
                } else if index % 2 == 0 {
                    2.0
                } else {
                    4.0
                };
                for (i, row) in reference.iter_mut().enumerate() {
                    for (j, value) in row.iter_mut().enumerate() {
                        *value += h * w / 3.0 * kernel[i] * kernel[j] * rho[i][j];
                    }
                }
            }
        }
        for (i, row) in c.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let reconstructed: f64 = l[i].iter().zip(l[j]).map(|(x, y)| x * y).sum();
                assert!((value - reference[i][j]).abs() < 2e-13);
                assert!((value - reconstructed).abs() < 2e-13);
            }
        }
    }

    #[test]
    fn future_rate_knots_change_conditional_cash_and_zero_rate_limit_is_exact() {
        let model = BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap();
        let zero = HullWhite1Factor::new(0.4, vec![0.0, 0.8], vec![0.0, 0.0]).unwrap();
        let c = cash_coefficients(model, 0.2, &zero, 0.25, -0.2, 0.3, 1.4).unwrap();
        let z = -0.7_f64 * (1.4 - 0.3);
        assert_eq!(c, [0.6 * -z.exp_m1(), z.exp(), 0.4 * -z.exp_m1()]);
        let flat = HullWhite1Factor::new(0.4, vec![0.0], vec![0.03]).unwrap();
        let changed = HullWhite1Factor::new(0.4, vec![0.0, 1.1], vec![0.03, 0.09]).unwrap();
        let a = cash_coefficients(model, 0.2, &flat, 0.25, -0.2, 0.3, 1.4).unwrap();
        let b = cash_coefficients(model, 0.2, &changed, 0.25, -0.2, 0.3, 1.4).unwrap();
        assert!((a[1] - b[1]).abs() > 1e-5);
        // Deterministic cash always has unit discounted-factor forecast.
        let fixed = BuehlerDividendModel::new(0.7, 0.0, 0.0, -0.25).unwrap();
        let c = cash_coefficients(fixed, 0.2, &changed, 0.25, -0.2, 0.3, 1.4).unwrap();
        assert!((c[1] + c[2] - 1.0).abs() < 2e-12);
    }
}
