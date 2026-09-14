//! Paid-cash affine observations for a continuous normalized equity state.
//!
//! U is the existing HW equity state, with initial value S0. With
//! G0(t)=Dq(0,t)/Dr(0,t), reconstruct S(t)=b(t)*G0(t)*U(t)+c(t).
//! Between events dc=(r-q)c dt; at an event c+=(1-beta)c- - D and
//! b+=(1-beta)b-. There is no future-dividend reserve or bond diffusion.

use crate::market::{DiscountCurve, EquityForward};
use crate::mc::hull_white::HybridState;
use crate::models::hull_white_dividends::transpose_log_curve;
use crate::models::{HullWhite1Factor, HullWhiteError};

pub(in crate::engine) const HULL_WHITE_AFFINE_DIVIDEND_MODEL: &str =
    "affine-paid-cash-realized-carry-v1";

#[derive(Clone, Debug)]
pub(in crate::engine) struct AffineDividendPlan {
    market: EquityForward,
    times: Box<[f64]>,
    log_carry: Box<[f64]>,
    rate_shift: Box<[f64]>,
    scales: Box<[f64]>,
    events: Box<[Option<(f64, f64)>]>,
}

#[derive(Debug)]
pub(in crate::engine) struct AffineDividendPath {
    pub(in crate::engine) spots: Vec<(f64, Option<f64>)>,
    growth: Vec<f64>,
    before_offsets: Vec<f64>,
}

pub(in crate::engine) struct AffineDividendAdjoints {
    pub(in crate::engine) equity: Vec<f64>,
    pub(in crate::engine) discount_log_df: Vec<f64>,
    pub(in crate::engine) dividend_log_df: Vec<f64>,
}

fn invalid(field: &'static str, index: usize) -> HullWhiteError {
    HullWhiteError::InvalidInput { field, index }
}

impl AffineDividendPlan {
    pub(in crate::engine) fn new(
        market: &EquityForward,
        rates: &HullWhite1Factor,
        times: &[f64],
    ) -> Result<Self, HullWhiteError> {
        if times.len() < 2 || times[0] != 0.0 {
            return Err(invalid("affine_dividend_time_grid", 0));
        }
        for (i, &t) in times.iter().enumerate() {
            if !t.is_finite() || t < 0.0 || (i > 0 && t <= times[i - 1]) {
                return Err(invalid("affine_dividend_time_order", i));
            }
        }
        let schedule = market.discrete_dividends().map_or(&[][..], |d| d.events());
        let horizon = times[times.len() - 1];
        for (i, event) in schedule.iter().enumerate() {
            if event.ex_time() <= horizon && !times.contains(&event.ex_time()) {
                return Err(invalid("missing_affine_dividend_time", i));
            }
        }
        let mut log_carry = Vec::with_capacity(times.len());
        let mut rate_shift = Vec::with_capacity(times.len());
        let mut scales = Vec::with_capacity(times.len());
        let mut events = Vec::with_capacity(times.len());
        let mut beta_product = 1.0;
        for (i, &t) in times.iter().enumerate() {
            let carry = market.dividend_curve().evaluate(t)?.log_discount
                - market.discount_curve().evaluate(t)?.log_discount;
            let event = schedule
                .iter()
                .find(|e| e.ex_time() == t)
                .map(|e| (e.fixed_cash(), e.beta()));
            if let Some((_, beta)) = event {
                beta_product *= 1.0 - beta;
            }
            let scale = beta_product * carry.exp();
            if !scale.is_finite() || scale <= 0.0 {
                return Err(invalid("affine_dividend_scale", i));
            }
            log_carry.push(carry);
            rate_shift.push(if i == 0 {
                0.0
            } else {
                rates.integrated_shift(times[i - 1], t)?
            });
            scales.push(scale);
            events.push(event);
        }
        Ok(Self {
            market: market.clone(),
            times: times.into(),
            log_carry: log_carry.into(),
            rate_shift: rate_shift.into(),
            scales: scales.into(),
            events: events.into(),
        })
    }

    pub(in crate::engine) fn scale(&self, node: usize, pre: bool) -> f64 {
        self.scales[node]
            / if pre {
                self.events[node].map_or(1.0, |(_, beta)| 1.0 - beta)
            } else {
                1.0
            }
    }
    pub(in crate::engine) fn record(
        &self,
        states: &[HybridState],
    ) -> Result<AffineDividendPath, HullWhiteError> {
        if states.len() != self.times.len() {
            return Err(invalid("affine_dividend_state_shape", 0));
        }
        let mut spots = Vec::with_capacity(states.len());
        let mut growth = Vec::with_capacity(states.len());
        let mut before_offsets = Vec::with_capacity(states.len());
        let mut offset = 0.0;
        for (i, state) in states.iter().enumerate() {
            let g = if i == 0 {
                1.0
            } else {
                (self.log_carry[i] - self.log_carry[i - 1]
                    + self.rate_shift[i]
                    + state.integrated_rate_factor
                    - states[i - 1].integrated_rate_factor)
                    .exp()
            };
            if !g.is_finite() || g <= 0.0 {
                return Err(invalid("affine_dividend_realized_carry", i));
            }
            offset *= g;
            before_offsets.push(offset);
            if let Some((cash, beta)) = self.events[i] {
                offset = (1.0 - beta) * offset - cash;
            }
            let post = self.scales[i] * state.normalized_equity + offset;
            let pre = self.events[i].map(|(cash, beta)| (post + cash) / (1.0 - beta));
            if !post.is_finite() || post <= 0.0 || pre.is_some_and(|v| !v.is_finite() || v <= 0.0) {
                // Never floor, resample, or silently switch dividend models.
                return Err(HullWhiteError::NonPositiveState { step: i });
            }
            growth.push(g);
            spots.push((post, pre));
        }
        Ok(AffineDividendPath {
            spots,
            growth,
            before_offsets,
        })
    }

    /// Reverse the observation map at fixed centered HW path, dividend quotes,
    /// and grid. HW model parameters/correlations are not part of this risk API.
    /// Curve refits leave the centered state unchanged but change both G0 and
    /// the carry of every already-paid cash dividend.
    pub(in crate::engine) fn reverse(
        &self,
        path: &AffineDividendPath,
        states: &[HybridState],
        seeds: &[(f64, f64)],
        multiplier: f64,
    ) -> Result<AffineDividendAdjoints, HullWhiteError> {
        let n = self.times.len();
        if states.len() != n || seeds.len() != n || path.spots.len() != n {
            return Err(invalid("affine_dividend_adjoint_shape", 0));
        }
        if !multiplier.is_finite() {
            return Err(invalid("affine_dividend_adjoint_multiplier", 0));
        }
        let mut equity = vec![0.0; n];
        let mut carry_seeds = vec![0.0; n];
        let mut offset_seed = 0.0;
        for i in (0..n).rev() {
            let (post, pre) = seeds[i];
            let total = if let Some((_, beta)) = self.events[i] {
                post + pre / (1.0 - beta)
            } else if pre == 0.0 {
                post
            } else {
                return Err(invalid("unexpected_pre_dividend_adjoint", i));
            } * multiplier;
            if !total.is_finite() {
                return Err(invalid("affine_dividend_adjoint_value", i));
            }
            equity[i] = total * self.scales[i];
            carry_seeds[i] += equity[i] * states[i].normalized_equity;
            offset_seed += total;
            if let Some((_, beta)) = self.events[i] {
                offset_seed *= 1.0 - beta;
            }
            if i > 0 {
                let weight = offset_seed * path.before_offsets[i];
                carry_seeds[i] += weight;
                carry_seeds[i - 1] -= weight;
                offset_seed *= path.growth[i];
            }
        }
        let mut discount_log_df = vec![0.0; self.market.discount_curve().times().len()];
        let mut dividend_log_df = vec![0.0; self.market.dividend_curve().times().len()];
        for (&t, &seed) in self.times.iter().zip(&carry_seeds) {
            transpose_log_curve(self.market.discount_curve(), t, -seed, &mut discount_log_df)?;
            transpose_log_curve(self.market.dividend_curve(), t, seed, &mut dividend_log_df)?;
        }
        if equity
            .iter()
            .chain(&discount_log_df)
            .chain(&dividend_log_df)
            .any(|v| !v.is_finite())
        {
            return Err(invalid("affine_dividend_adjoint_result", 0));
        }
        Ok(AffineDividendAdjoints {
            equity,
            discount_log_df,
            dividend_log_df,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{CurveId, EventId, PositiveF64, UnderlyingId};
    use crate::market::{DividendEvent, DividendQuote, LogLinearDiscountCurve};
    use std::sync::Arc;

    fn market(cash: &[(f64, f64, f64)], r: f64, q: f64) -> EquityForward {
        let curve = |id, rate: f64| {
            let times = vec![0.0, 0.25, 0.5, 1.0, 2.0];
            let dfs = times.iter().map(|t| (-rate * t).exp()).collect();
            Arc::new(LogLinearDiscountCurve::new(CurveId::new(id), times, dfs).unwrap())
        };
        EquityForward::with_discrete_dividends(
            UnderlyingId::new(1),
            PositiveF64::new(100.0, "spot").unwrap(),
            curve(1, r),
            curve(2, q),
            cash.iter()
                .enumerate()
                .map(|(i, &(t, amount, beta))| {
                    DividendEvent::new(
                        EventId::new(i as u32),
                        t,
                        DividendQuote::FixedCashAndProportional {
                            fixed_cash: amount,
                            beta,
                        },
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap()
    }

    fn rates(sigma: f64) -> HullWhite1Factor {
        HullWhite1Factor::new(0.1, vec![0.0], vec![sigma]).unwrap()
    }

    fn states() -> Vec<HybridState> {
        [0.0, 0.004, -0.002, 0.007]
            .iter()
            .enumerate()
            .map(|(i, &integrated)| HybridState {
                normalized_equity: 100.0 + 2.0 * i as f64,
                volatility_factor: 0.0,
                rate_factor: 0.01,
                integrated_rate_factor: integrated,
            })
            .collect()
    }

    #[test]
    fn deterministic_rate_limit_matches_common_forward_and_event_order() {
        let m = market(
            &[(0.0, 2.0, 0.02), (0.5, 4.0, 0.1), (1.0, 1.0, 0.03)],
            0.05,
            0.02,
        );
        let times = [0.0, 0.25, 0.5, 1.0];
        let plan = AffineDividendPlan::new(&m, &rates(0.0), &times).unwrap();
        let states = vec![HybridState::initial(100.0).unwrap(); times.len()];
        let path = plan.record(&states).unwrap();
        for (i, &t) in times.iter().enumerate() {
            let expected = m.evaluate(t).unwrap().spot_contract_forward;
            assert!((path.spots[i].0 - expected).abs() < 1e-12);
            if let Some((cash, beta)) = plan.events[i] {
                let pre = path.spots[i].1.unwrap();
                assert!((path.spots[i].0 - ((1.0 - beta) * pre - cash)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn zero_equity_vol_discounted_gains_are_pathwise_constant() {
        let m = market(
            &[(0.0, 2.0, 0.02), (0.5, 4.0, 0.1), (1.0, 1.0, 0.03)],
            0.05,
            0.0,
        );
        let times = [0.0, 0.25, 0.5, 1.0];
        let rates = rates(0.03);
        let plan = AffineDividendPlan::new(&m, &rates, &times).unwrap();
        let mut states = states();
        for (state, &t) in states.iter_mut().zip(&times) {
            state.normalized_equity = 100.0
                / rates
                    .relative_discount(t, state.integrated_rate_factor)
                    .unwrap();
        }
        let path = plan.record(&states).unwrap();
        let mut dividends = 0.0;
        for (i, &t) in times.iter().enumerate() {
            let df = m.discount_curve().discount(t).unwrap()
                * rates
                    .relative_discount(t, states[i].integrated_rate_factor)
                    .unwrap();
            if let Some((cash, beta)) = plan.events[i] {
                dividends += df * (cash + beta * path.spots[i].1.unwrap());
            }
            assert!((df * path.spots[i].0 + dividends - 100.0).abs() < 2e-12);
        }
    }

    #[test]
    fn observation_reverse_matches_curve_and_state_finite_differences() {
        let schedule = [(0.0, 2.0, 0.02), (0.5, 4.0, 0.1), (1.0, 1.0, 0.03)];
        let times = [0.0, 0.25, 0.5, 1.0];
        let m = market(&schedule, 0.05, 0.02);
        let rates = rates(0.03);
        let plan = AffineDividendPlan::new(&m, &rates, &times).unwrap();
        let states = states();
        let path = plan.record(&states).unwrap();
        let seeds = [(0.3, 0.2), (0.1, 0.0), (0.4, 0.3), (0.7, 0.6)];
        let objective = |p: &AffineDividendPlan, s: &[HybridState]| {
            p.record(s)
                .unwrap()
                .spots
                .iter()
                .zip(seeds)
                .map(|(&(post, pre), (a, b))| a * post + b * pre.unwrap_or(0.0))
                .sum::<f64>()
        };
        let adj = plan.reverse(&path, &states, &seeds, 1.0).unwrap();
        let h = 1e-5;
        for i in 0..states.len() {
            let mut up = states.clone();
            let mut down = states.clone();
            up[i].normalized_equity += h;
            down[i].normalized_equity -= h;
            let fd = (objective(&plan, &up) - objective(&plan, &down)) / (2.0 * h);
            assert!((fd - adj.equity[i]).abs() < 1e-8);
        }
        for discount in [true, false] {
            let bump = |amount| {
                let m = market(
                    &schedule,
                    0.05 + if discount { amount } else { 0.0 },
                    0.02 + if discount { 0.0 } else { amount },
                );
                let p = AffineDividendPlan::new(&m, &rates, &times).unwrap();
                objective(&p, &states)
            };
            let fd = (bump(h) - bump(-h)) / (2.0 * h);
            let bars = if discount {
                &adj.discount_log_df
            } else {
                &adj.dividend_log_df
            };
            let expected: f64 = bars
                .iter()
                .zip(m.discount_curve().times())
                .map(|(bar, t)| -t * bar)
                .sum();
            assert!((fd - expected).abs() < 1e-7, "{fd} != {expected}");
            assert_eq!(bars[0], 0.0);
        }
    }

    #[test]
    fn future_dividends_do_not_create_a_reserve_or_affect_paths() {
        let times = [0.0, 0.25, 0.5, 1.0];
        let rates = rates(0.03);
        let a = market(&[(0.5, 4.0, 0.1)], 0.05, 0.02);
        let b = market(&[(0.5, 4.0, 0.1), (2.0, 1000.0, 0.2)], 0.05, 0.02);
        let a = AffineDividendPlan::new(&a, &rates, &times).unwrap();
        let b = AffineDividendPlan::new(&b, &rates, &times).unwrap();
        assert_eq!(
            a.record(&states()).unwrap().spots,
            b.record(&states()).unwrap().spots
        );
    }

    #[test]
    fn missing_events_and_nonpositive_spots_are_rejected() {
        let m = market(&[(0.5, 200.0, 0.0)], 0.05, 0.02);
        assert!(AffineDividendPlan::new(&m, &rates(0.02), &[0.0, 1.0]).is_err());
        let plan = AffineDividendPlan::new(&m, &rates(0.02), &[0.0, 0.25, 0.5, 1.0]).unwrap();
        assert!(matches!(
            plan.record(&states()),
            Err(HullWhiteError::NonPositiveState { step: 2 })
        ));
    }
}
