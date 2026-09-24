//! Reverse of the positive Buehler split and full carry-funded cash reserve.
//! Optional prepared OU/model/correlation scopes share the unchanged split.
//! The compiled time grid is fixed. No bumps are used.

use super::bergomi::correlation_reverse::{self, BergomiCorrelationReverse};
use super::bergomi::parameter_reverse::BergomiParameterReverse;
use super::*;
use crate::models::hull_white_dividends::transpose_log_curve;

#[derive(Clone, Debug)]
struct NodeJacobian {
    equity: Vec<f64>,
    dividend: Vec<f64>,
    constant: Vec<f64>,
    event_indices: Vec<usize>,
}

/// Built once per risk evaluation, never once per path. Labels use the market
/// schedule order and both curves' original pillar order (including fixed t=0).
pub(in crate::engine) struct ReverseContext {
    pub labels: Vec<String>,
    pub cash_times: Vec<f64>,
    pub discount_times: Vec<f64>,
    pub repo_spread_times: Vec<f64>,
    nodes: Vec<NodeJacobian>,
    payment_weights: Vec<f64>,
    bergomi: Option<BergomiParameterReverse>,
    correlation: Option<BergomiCorrelationReverse>,
    correlation_start: Option<usize>,
}
impl ReverseContext {
    pub fn new(
        path: &StochasticDividendPathPlan,
        market: &EquityForward,
        payment: f64,
    ) -> Result<Self, MonteCarloError> {
        let events = market.discrete_dividends().map_or(&[][..], |d| d.events());
        let p = market.discount_curve();
        let q = market.dividend_curve();
        let cash_times = events.iter().map(|e| e.ex_time()).collect::<Vec<_>>();
        let discount_times = p.times().to_vec();
        let repo_spread_times = q.times().to_vec();
        let mut labels = vec![
            "spot".into(),
            "initial_volatility".into(),
            "dividend_mean_reversion".into(),
            "equity_linkage".into(),
            "dividend_volatility".into(),
        ];
        labels.extend(
            events
                .iter()
                .map(|e| format!("cash_mean[{}]", e.event().get())),
        );
        labels.extend((0..p.times().len()).map(|i| format!("discount_log_df[{i}]")));
        labels.extend((0..q.times().len()).map(|i| format!("repo_spread_log_df[{i}]")));
        let width = labels.len();
        let po = 5 + events.len();
        let qo = po + p.times().len();
        let add_growth = |t: f64, seed: f64, out: &mut [f64]| -> Result<(), MonteCarloError> {
            transpose_log_curve(p, t, -seed, &mut out[po..qo])?;
            transpose_log_curve(q, t, seed, &mut out[qo..])?;
            Ok(())
        };
        let mut nodes = Vec::with_capacity(path.times.len());
        for node in path.nodes.iter() {
            let t = node.time;
            let g = market.forward(t)? / market.spot().get();
            let mut a = vec![0.0; width];
            let mut b = vec![0.0; width];
            let mut c = vec![0.0; width];
            a[0] = g;
            let mut event_indices = Vec::new();
            for (i, e) in events.iter().enumerate() {
                let gi = market.forward(e.ex_time())? / market.spot().get();
                let ratio = g / gi;
                let cash = e.fixed_cash();
                a[5 + i] = -ratio; // All supplied cash, including beyond expiry, is funded.
                if e.ex_time() == t {
                    event_indices.push(5 + i);
                }
                if e.ex_time() > t {
                    let h = e.ex_time() - t;
                    let (w, complement) = decay(path.model.mean_reversion, h);
                    let alpha = path.model.equity_linkage;
                    a[5 + i] += ratio * complement * alpha;
                    b[5 + i] = ratio * w;
                    c[5 + i] = ratio * complement * (1.0 - alpha);
                    let reserve = ratio * cash;
                    a[2] += reserve * h * w * alpha;
                    b[2] -= reserve * h * w;
                    c[2] += reserve * h * w * (1.0 - alpha);
                    a[3] += reserve * complement;
                    c[3] -= reserve * complement;
                }
                // d/dlog G(t_i) = -cash_i * d/dcash_i, including the initial funding term.
                let seeds = [-cash * a[5 + i], -cash * b[5 + i], -cash * c[5 + i]];
                add_growth(e.ex_time(), seeds[0], &mut a)?;
                add_growth(e.ex_time(), seeds[1], &mut b)?;
                add_growth(e.ex_time(), seeds[2], &mut c)?;
            }
            add_growth(t, node.equity_coefficient, &mut a)?;
            add_growth(t, node.dividend_coefficient, &mut b)?;
            add_growth(t, node.constant, &mut c)?;
            nodes.push(NodeJacobian {
                equity: a,
                dividend: b,
                constant: c,
                event_indices,
            });
        }
        let mut payment_weights = vec![0.0; width];
        transpose_log_curve(p, payment, 1.0, &mut payment_weights[po..qo])?;
        Ok(Self {
            labels,
            cash_times,
            discount_times,
            repo_spread_times,
            nodes,
            payment_weights,
            bergomi: None,
            correlation: None,
            correlation_start: None,
        })
    }

    pub fn enable_bergomi_parameters(
        &mut self,
        path: &StochasticDividendPathPlan,
    ) -> Result<(), StochasticDividendError> {
        let kernel = path
            .bergomi
            .as_ref()
            .ok_or(StochasticDividendError::Unsupported {
                feature: "evaluate_bergomi_aad requires a 1F or 2F Bergomi plan",
            })?;
        let prepared = BergomiParameterReverse::new(
            kernel,
            path.model.equity_dividend_correlation(),
            &path.times,
        )?;
        self.labels
            .extend(prepared.labels().iter().map(|s| (*s).to_owned()));
        self.payment_weights.resize(self.labels.len(), 0.0);
        self.bergomi = Some(prepared);
        Ok(())
    }

    /// Append raw symmetric Brownian-correlation entry partials. Bergomi plans
    /// include the entire existing model-parameter prefix, BS the basic prefix.
    pub fn enable_correlations(
        &mut self,
        path: &StochasticDividendPathPlan,
    ) -> Result<(), StochasticDividendError> {
        let rho = path.model.equity_dividend_correlation();
        if (1.0 - rho) * (1.0 + rho) <= 1e-10 {
            return Err(correlation_reverse::unsupported());
        }
        if let Some(kernel) = &path.bergomi {
            // Check the instantaneous domain before the existing model-risk scope.
            let prepared = BergomiCorrelationReverse::new(kernel, rho, &path.times)?;
            self.enable_bergomi_parameters(path)?;
            self.correlation_start = Some(self.labels.len());
            self.labels
                .extend(prepared.labels().iter().map(|s| (*s).to_owned()));
            self.correlation = Some(prepared);
        } else {
            self.correlation_start = Some(self.labels.len());
            self.labels.push("equity_dividend_correlation".into());
        }
        self.payment_weights.resize(self.labels.len(), 0.0);
        Ok(())
    }

    /// Spot-only slice of the reverse: the normalized f/Y states and OU
    /// innovations do not depend on S0. Match the full reverse's node order and
    /// arithmetic, including both pre- and post-dividend payoff seeds.
    pub fn spot_pullback(
        &self,
        states: &[BuehlerDividendState],
        seeds: &[(f64, f64)],
    ) -> Result<f64, StochasticDividendError> {
        if states.len() != self.nodes.len() || seeds.len() != states.len() {
            return Err(invalid("spot_reverse_shape"));
        }
        let mut out = 0.0;
        for i in (0..states.len()).rev() {
            let state = states[i];
            let jac = &self.nodes[i];
            let (post, pre) = seeds[i];
            out += (post + pre)
                * (state.equity * jac.equity[0]
                    + state.dividend * jac.dividend[0]
                    + jac.constant[0]);
        }
        if !out.is_finite() {
            return Err(invalid("spot_reverse_result"));
        }
        Ok(out)
    }

    pub fn pullback(
        &self,
        plan: &StochasticDividendPathPlan,
        normals: &[f64],
        states: &[BuehlerDividendState],
        seeds: &[(f64, f64)],
        discounted_payoff: f64,
    ) -> Result<Vec<f64>, StochasticDividendError> {
        if normals.len() != plan.dimension as usize
            || states.len() != plan.times.len()
            || seeds.len() != states.len()
        {
            return Err(invalid("reverse_shape"));
        }
        if !discounted_payoff.is_finite()
            || seeds.iter().any(|(a, b)| !a.is_finite() || !b.is_finite())
        {
            return Err(invalid("reverse_seed"));
        }
        let loadings = match &plan.bergomi {
            Some(k) => k.volatility_loadings(normals)?,
            None => vec![1.0; states.len() - 1],
        };
        let mut out = self
            .payment_weights
            .iter()
            .map(|w| discounted_payoff * w)
            .collect::<Vec<_>>();
        let mut loading_bars = self.bergomi.as_ref().map(|_| vec![0.0; states.len() - 1]);
        let mut f_bar = 0.0;
        let mut y_bar = 0.0;
        let model = plan.model;
        let alpha = model.equity_linkage;
        let rho = model.equity_dividend_correlation;
        let count = plan.random_factor_count();
        for i in (0..states.len()).rev() {
            let s = states[i];
            let node = plan.nodes[i];
            let jac = &self.nodes[i];
            let (post, pre) = seeds[i];
            let total = post + pre;
            f_bar += total * node.equity_coefficient;
            y_bar += total * node.dividend_coefficient + pre * node.event_mean_cash.unwrap_or(0.0);
            for (j, bar) in out[..jac.equity.len()].iter_mut().enumerate() {
                *bar += total
                    * (s.equity * jac.equity[j] + s.dividend * jac.dividend[j] + jac.constant[j]);
            }
            for &j in &jac.event_indices {
                out[j] += pre * s.dividend;
            }
            if i == 0 {
                break;
            } // f0=Y0=1 are fixed normalization anchors.
            let old = states[i - 1];
            let dt = plan.times[i] - plan.times[i - 1];
            let root = dt.sqrt();
            let z = &normals[count * (i - 1)..count * i];
            let (a, b) = decay(model.mean_reversion, 0.5 * dt);
            let half = blend(a, b, old.dividend, model.target(old.equity));
            let v = model.dividend_volatility * root;
            let zd = rho * z[0] + ((1.0 - rho) * (1.0 + rho)).sqrt() * z[1];
            let ed = (-0.5 * v * v + v * zd).exp();
            let noise = half * ed;
            let sigma = if plan.volatility == 0.0 {
                0.0
            } else {
                plan.volatility * loadings[i - 1]
            };
            let u = sigma * root;
            let ef = (-0.5 * u * u + u * z[0]).exp();
            // Differentiate the mathematical convex blend at current==target:
            // the equality shortcut is a roundoff-preservation identity, not a
            // branch that removes alpha/kappa/state derivatives.
            let noise_bar = a * y_bar;
            let next_f_bar = f_bar + b * alpha * y_bar;
            out[2] += y_bar * (-0.5 * dt * a) * (noise - model.target(s.equity));
            out[3] += y_bar * b * (s.equity - 1.0);
            let half_bar = noise_bar * ed;
            out[4] += noise_bar * noise * (zd - v) * root;
            if let Some(start) = self.correlation_start {
                // Direct dividend-driver rotation. OU loading effects are added
                // separately below, so no indirect volatility effect is lost.
                let dr = z[0] - rho / ((1.0 - rho) * (1.0 + rho)).sqrt() * z[1];
                out[start] += noise_bar * noise * v * dr;
            }
            out[1] += next_f_bar * s.equity * (z[0] - u) * root * loadings[i - 1];
            if let Some(bars) = &mut loading_bars {
                bars[i - 1] = next_f_bar * s.equity * (z[0] - u) * root * plan.volatility;
            }
            out[2] += half_bar * (-0.5 * dt * a) * (old.dividend - model.target(old.equity));
            out[3] += half_bar * b * (old.equity - 1.0);
            f_bar = next_f_bar * ef + half_bar * b * alpha;
            y_bar = half_bar * a;
        }
        if let (Some(prepared), Some(bars), Some(kernel)) =
            (&self.bergomi, &loading_bars, &plan.bergomi)
        {
            let extra = prepared.pullback(kernel, normals, bars)?;
            let start = self.nodes[0].equity.len();
            out[start..start + extra.len()].copy_from_slice(&extra);
        }
        if let (Some(prepared), Some(start), Some(bars), Some(kernel)) = (
            &self.correlation,
            self.correlation_start,
            &loading_bars,
            &plan.bergomi,
        ) {
            let extra = prepared.pullback(kernel, normals, bars)?;
            for (bar, extra) in out[start..].iter_mut().zip(extra) {
                *bar += extra;
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(invalid("reverse_result"));
        }
        Ok(out)
    }
}
