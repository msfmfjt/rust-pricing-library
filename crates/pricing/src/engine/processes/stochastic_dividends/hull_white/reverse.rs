//! Basic reverse at fixed HW parameters, correlations, grid and cash dates.
//! Rate states and the *relative* payment discount are then independent of all
//! active inputs. Curve risk still includes the fitted deterministic discount,
//! initial funding, every conditional cash claim and payment-date P0(U).
use super::*;
use crate::engine::processes::stochastic_dividends::{blend, decay};
use crate::models::hull_white_dividends::transpose_log_curve;

#[derive(Clone, Debug)]
struct ClaimJacobian {
    // Unit-mean carry-weighted bond at x=0. Never divide by a cash mean.
    unit: f64,
    // Derivatives w.r.t. (sigma_f, kappa, alpha, nu_D); columns (A,B,C).
    coefficients: [[f64; 3]; 4],
}

pub(in crate::engine) struct HullWhiteReverseContext {
    pub labels: Vec<String>,
    pub cash_times: Vec<f64>,
    pub discount_times: Vec<f64>,
    pub repo_spread_times: Vec<f64>,
    market: EquityForward,
    claims: Vec<Vec<ClaimJacobian>>,
    funding: Vec<f64>,
    payment_weights: Vec<f64>,
}
impl HullWhiteReverseContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: &StochasticDividendHullWhitePathPlan,
        market: &EquityForward,
        rates: &HullWhite1Factor,
        rho_f: f64,
        rho_d: f64,
        payment: f64,
    ) -> Result<Self, MonteCarloError> {
        let events = market.discrete_dividends().map_or(&[][..], |s| s.events());
        let p = market.discount_curve();
        let q = market.dividend_curve();
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
        let po = 5 + events.len();
        let qo = po + p.times().len();
        let mut funding = vec![0.0; labels.len()];
        let mut claims = Vec::with_capacity(path.times.len());
        for (i, node) in path.nodes.iter().enumerate() {
            let t = path.times[i];
            let mut row = Vec::with_capacity(node.claims.len());
            for claim in &node.claims {
                let maturity = path.cash_times[claim.index];
                let jac = ClaimJacobian {
                    unit: claim.carry_weight * rates.bond_price(p, t, maturity, 0.0)?,
                    coefficients: cash_coefficient_jacobian(
                        path.model,
                        path.volatility,
                        rates,
                        rho_f,
                        rho_d,
                        t,
                        maturity,
                    )?,
                };
                if i == 0 {
                    let value = claim.carry_weight
                        * claim.value(StochasticDividendHullWhiteState::initial())?;
                    funding[5 + claim.index] = jac.unit * claim.coefficients.iter().sum::<f64>();
                    for (j, derivative) in jac.coefficients.iter().enumerate() {
                        funding[1 + j] += jac.unit
                            * path.cash_means[claim.index]
                            * derivative.iter().sum::<f64>();
                    }
                    transpose_log_curve(p, maturity, value, &mut funding[po..qo])?;
                    transpose_log_curve(q, maturity, -value, &mut funding[qo..])?;
                }
                row.push(jac);
            }
            claims.push(row);
        }
        let mut payment_weights = vec![0.0; labels.len()];
        transpose_log_curve(p, payment, 1.0, &mut payment_weights[po..qo])?;
        if funding
            .iter()
            .chain(&payment_weights)
            .any(|x| !x.is_finite())
        {
            return Err(invalid("hw_reverse_coefficients").into());
        }
        Ok(Self {
            labels,
            cash_times: path.cash_times.to_vec(),
            discount_times: p.times().to_vec(),
            repo_spread_times: q.times().to_vec(),
            market: market.clone(),
            claims,
            funding,
            payment_weights,
        })
    }

    pub fn pullback(
        &self,
        path: &StochasticDividendHullWhitePathPlan,
        normals: &[f64],
        states: &[StochasticDividendHullWhiteState],
        seeds: &[(f64, f64)],
        discounted_payoff: f64,
    ) -> Result<Vec<f64>, MonteCarloError> {
        if normals.len() != path.dimension as usize
            || states.len() != path.times.len()
            || seeds.len() != states.len()
        {
            return Err(invalid("hw_reverse_shape").into());
        }
        if !discounted_payoff.is_finite()
            || normals.iter().any(|x| !x.is_finite())
            || seeds.iter().any(|(a, b)| !a.is_finite() || !b.is_finite())
        {
            return Err(invalid("hw_reverse_seed").into());
        }
        let po = 5 + path.cash_times.len();
        let qo = po + self.discount_times.len();
        let p = self.market.discount_curve();
        let q = self.market.dividend_curve();
        let mut out: Vec<f64> = self
            .payment_weights
            .iter()
            .map(|w| w * discounted_payoff)
            .collect();
        let (mut f_bar, mut y_bar, mut funding_bar) = (0.0, 0.0, 0.0);
        let model = path.model;
        let alpha = model.equity_linkage();
        let rho = model.equity_dividend_correlation();
        for i in (0..states.len()).rev() {
            let state = states[i];
            let s = state.factors;
            let node = &path.nodes[i];
            let (post, pre) = seeds[i];
            let seed = post + pre;
            let growth = node.residual_growth
                * (state.integrated_rate_factor + node.half_integral_variance).exp();
            let funded = path.risky_spot * growth * s.equity;
            out[0] += seed * growth * s.equity;
            funding_bar -= seed * growth * s.equity;
            f_bar += seed * path.risky_spot * growth;
            let mut growth_seed = seed * funded;
            for (claim, jac) in node.claims.iter().zip(&self.claims[i]) {
                let [a, b, c] = claim.coefficients;
                let unit = jac.unit * (-claim.duration * state.rate_factor).exp();
                let amount = unit * path.cash_means[claim.index];
                let value_seed = seed * amount * (a * s.equity + b * s.dividend + c);
                growth_seed += value_seed;
                out[5 + claim.index] += seed * unit * (a * s.equity + b * s.dividend + c);
                for (j, &[da, db, dc]) in jac.coefficients.iter().enumerate() {
                    out[1 + j] += seed * amount * (da * s.equity + db * s.dividend + dc);
                }
                f_bar += seed * amount * a;
                y_bar += seed * amount * b;
                let maturity = path.cash_times[claim.index];
                transpose_log_curve(p, maturity, value_seed, &mut out[po..qo])?;
                transpose_log_curve(q, maturity, -value_seed, &mut out[qo..])?;
            }
            transpose_log_curve(p, path.times[i], -growth_seed, &mut out[po..qo])?;
            transpose_log_curve(q, path.times[i], growth_seed, &mut out[qo..])?;
            y_bar += pre * node.event_cash.unwrap_or(0.0);
            for (j, &t) in path.cash_times.iter().enumerate() {
                if t == path.times[i] {
                    out[5 + j] += pre * s.dividend;
                }
            }
            if i == 0 {
                break;
            } // Fixed f0=Y0=1 and x0=I0=0.
            // Exact reverse of the existing drift/diffusion/drift split. No
            // derivatives are dropped at a blend's roundoff-preserving equality.
            let old = states[i - 1].factors;
            let dt = path.steps[i - 1].dt;
            let root = dt.sqrt();
            let z = &normals[4 * (i - 1)..4 * i];
            let (a, b) = decay(model.mean_reversion(), 0.5 * dt);
            let half = blend(a, b, old.dividend, model.target(old.equity));
            let v = model.dividend_volatility() * root;
            let zd = rho * z[0] + ((1.0 - rho) * (1.0 + rho)).sqrt() * z[1];
            let ed = (-0.5 * v * v + v * zd).exp();
            let noise = half * ed;
            let u = path.volatility * root;
            let ef = (-0.5 * u * u + u * z[0]).exp();
            let noise_bar = a * y_bar;
            let next_f_bar = f_bar + b * alpha * y_bar;
            out[2] += y_bar * (-0.5 * dt * a) * (noise - model.target(s.equity));
            out[3] += y_bar * b * (s.equity - 1.0);
            let half_bar = noise_bar * ed;
            out[4] += noise_bar * noise * (zd - v) * root;
            out[1] += next_f_bar * s.equity * (z[0] - u) * root;
            out[2] += half_bar * (-0.5 * dt * a) * (old.dividend - model.target(old.equity));
            out[3] += half_bar * b * (old.equity - 1.0);
            f_bar = next_f_bar * ef + half_bar * b * alpha;
            y_bar = half_bar * a;
        }
        for (bar, derivative) in out.iter_mut().zip(&self.funding) {
            *bar += funding_bar * derivative;
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(invalid("hw_reverse_result").into());
        }
        Ok(out)
    }
}

/// Differentiate the continuous conditional-claim integrals analytically. Use
/// independent derivative quadrature, not a derivative of adaptive panel choices.
/// The primal shortcuts at zero kappa/alpha/sigma/nu do NOT eliminate limiting
/// sensitivities. All four input partials are well-defined from the valid side.
#[allow(clippy::too_many_arguments)]
fn cash_coefficient_jacobian(
    model: BuehlerDividendModel,
    sigma: f64,
    rates: &HullWhite1Factor,
    rho_f: f64,
    rho_d: f64,
    t: f64,
    maturity: f64,
) -> Result<[[f64; 3]; 4], MonteCarloError> {
    let tau = maturity - t;
    let k = model.mean_reversion();
    let alpha = model.equity_linkage();
    let cf = sigma * rho_f;
    let cd = model.dividend_volatility() * rho_d;
    let total = rate_brownian_integral(rates, t, maturity)?;
    let by = (-k * tau - cd * total).exp();
    if rates.is_deterministic() {
        let (w, comp) = decay(k, tau);
        return Ok([
            [0.0; 3],
            [alpha * tau * w, -tau * w, (1.0 - alpha) * tau * w],
            [comp, 0.0, -comp],
            [0.0; 3],
        ]);
    }
    let mut result = [[0.0; 3]; 4];
    result[1][1] = -tau * by;
    result[3][1] = -rho_d * total * by;
    let mut cuts = vec![0.0, tau];
    cuts.extend(
        rates
            .volatility_times()
            .iter()
            .filter(|&&u| u > t && u < maturity)
            .map(|&u| maturity - u),
    );
    cuts.sort_by(f64::total_cmp);
    // Two components use the existing vector Simpson rule. Integrate each
    // parameter's (A,C) tangents independently to enforce derivative error targets.
    for (j, row) in result.iter_mut().enumerate() {
        let integrand = |v: f64| -> Result<[f64; 2], MonteCarloError> {
            let tail = rate_brownian_integral(rates, maturity - v, maturity)?;
            let e = (-k * v - cd * tail).exp();
            let ef = (-k * v - cd * tail - cf * (total - tail)).exp();
            let values = match j {
                0 => [-alpha * k * rho_f * (total - tail) * ef, 0.0],
                1 => [
                    alpha * (1.0 - k * v) * ef,
                    (1.0 - alpha) * (1.0 - k * v) * e,
                ],
                2 => [k * ef, -k * e],
                _ => [
                    -alpha * k * rho_d * tail * ef,
                    -(1.0 - alpha) * k * rho_d * tail * e,
                ],
            };
            if values.iter().any(|x| !x.is_finite()) {
                return Err(invalid("cash_derivative_integrand").into());
            }
            Ok(values)
        };
        for pair in cuts.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            let samples = [integrand(lo)?, integrand((lo + hi) * 0.5)?, integrand(hi)?];
            let value = integrate(
                &integrand,
                lo,
                hi,
                samples,
                simpson(lo, hi, samples),
                QUADRATURE_ABS * (hi - lo) / tau,
                QUADRATURE_DEPTH,
            )?;
            row[0] += value[0];
            row[2] += value[1];
        }
    }
    if result.iter().flatten().any(|x| !x.is_finite()) {
        return Err(invalid("cash_derivative_coefficients").into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cash_coefficient_partials_include_zero_branches_and_future_rate_knots() {
        for sr in [0.0, 0.04] {
            let rates =
                HullWhite1Factor::new(0.4, vec![0.0, 0.8, 1.2], vec![sr, sr * 1.4, sr * 0.7])
                    .unwrap();
            for k in [0.0, 0.7] {
                for alpha in [0.0, 0.6, 1.0] {
                    for nu in [0.0, 0.35] {
                        for sigma in [0.0, 0.2] {
                            let inputs = [sigma, k, alpha, nu];
                            let coefficient = |x: [f64; 4]| {
                                cash_coefficients(
                                    BuehlerDividendModel::new(x[1], x[2], x[3], -0.25).unwrap(),
                                    x[0],
                                    &rates,
                                    0.25,
                                    -0.2,
                                    0.31,
                                    1.43,
                                )
                                .unwrap()
                            };
                            let jac = cash_coefficient_jacobian(
                                BuehlerDividendModel::new(k, alpha, nu, -0.25).unwrap(),
                                sigma,
                                &rates,
                                0.25,
                                -0.2,
                                0.31,
                                1.43,
                            )
                            .unwrap();
                            for (j, row) in jac.iter().enumerate() {
                                let inward = inputs[j] == 0.0 || (j == 2 && alpha == 1.0);
                                for h in [1e-5, 1e-6] {
                                    let mut up = inputs;
                                    let mut down = inputs;
                                    let signed = if j == 2 && alpha == 1.0 { -h } else { h };
                                    up[j] += signed;
                                    if !inward {
                                        down[j] -= signed;
                                    }
                                    let u = coefficient(up);
                                    let d = coefficient(down);
                                    for (c, &derivative) in row.iter().enumerate() {
                                        let fd = (u[c] - d[c])
                                            / (if inward { signed } else { 2.0 * signed });
                                        let budget = if inward { 2e-5 } else { 3e-7 };
                                        assert!(
                                            (derivative - fd).abs() < budget,
                                            "inputs={inputs:?} sr={sr}, j={j} c={c}, AAD={} FD={fd}",
                                            derivative
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn zero_mean_reversion_tilt_and_linkage_limits_match_independent_integrals() {
        // a=0, constant sr: L(t,T)=sr*(T-t)^2/2. Independent midpoint
        // integration of elementary exponentials checks kappa's *nonzero*
        // derivative even though the primal A,C are zero at kappa=0.
        let rates = HullWhite1Factor::new(0.0, vec![0.0], vec![0.04]).unwrap();
        let model = BuehlerDividendModel::new(0.0, 0.6, 0.35, -0.25).unwrap();
        let (tau, sigma, rf, rd): (f64, f64, f64, f64) = (1.12, 0.2, 0.25, -0.2);
        let jac = cash_coefficient_jacobian(model, sigma, &rates, rf, rd, 0.31, 1.43).unwrap();
        let total = 0.04 * tau * tau / 2.0;
        let by = (-0.35 * rd * total).exp();
        assert!((jac[1][1] + tau * by).abs() < 2e-14);
        assert!((jac[3][1] + rd * total * by).abs() < 2e-14);
        let mut expected = [0.0; 2];
        let n = 20_000;
        for i in 0..n {
            let v = tau * (i as f64 + 0.5) / n as f64;
            let tail = 0.04 * v * v / 2.0;
            expected[0] +=
                0.6 * (-0.35 * rd * tail - sigma * rf * (total - tail)).exp() * tau / n as f64;
            expected[1] += 0.4 * (-0.35 * rd * tail).exp() * tau / n as f64;
        }
        assert!((jac[1][0] - expected[0]).abs() < 2e-11);
        assert!((jac[1][2] - expected[1]).abs() < 2e-11);
        assert_eq!(jac[0], [0.0; 3]);
        assert_eq!(jac[2], [0.0; 3]);
    }
}
