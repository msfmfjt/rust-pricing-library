//! Fixed-grid directional derivatives for the Hull--White parameters used by
//! stochastic-dividend paths. The normal coordinates stay fixed while the
//! covariance Cholesky, rate states, escrowed claims, and payment discount move.
use super::*;
use std::collections::HashMap;

const RATE_QUADRATURE_ABS: f64 = 2e-13;
const RATE_QUADRATURE_REL: f64 = 2e-12;
const RATE_QUADRATURE_DEPTH: u32 = 24;
const MIN_NORMALIZED_PIVOT: f64 = 1e-10;

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::engine) struct HullWhiteRateStateTangent {
    pub rate_factor: f64,
    pub integrated_rate_factor: f64,
}

#[derive(Clone, Debug)]
struct StepTangent {
    decay: f64,
    integral_loading: f64,
    noise: [[f64; 4]; 4],
}

#[derive(Clone, Debug, Default)]
struct ClaimTangent {
    amount: f64,
    duration: f64,
    coefficients: [f64; 3],
}

#[derive(Clone, Debug, Default)]
struct NodeTangent {
    half_integral_variance: Vec<f64>,
    claims: Vec<Vec<ClaimTangent>>,
}

/// Sensitivities appended to the existing spot/volatility/dividend/curve risk.
/// The method is defined only where every simulated hybrid covariance has a
/// well-conditioned full-rank Cholesky factor.
pub(in crate::engine) struct HullWhiteRateSensitivityContext {
    labels: Vec<String>,
    steps: Vec<Vec<StepTangent>>,
    nodes: Vec<NodeTangent>,
    risky_spot: Vec<f64>,
    payment_log_constant: Vec<f64>,
    payment_duration: Vec<f64>,
}

impl HullWhiteRateSensitivityContext {
    pub fn new(
        path: &StochasticDividendHullWhitePathPlan,
        rates: &HullWhite1Factor,
        rho_equity_rate: f64,
        rho_dividend_rate: f64,
        payment_time: f64,
    ) -> Result<Self, MonteCarloError> {
        let parameter_count = 1 + rates.volatilities().len();
        let mut labels = Vec::with_capacity(parameter_count);
        labels.push("rate_mean_reversion".to_owned());
        labels.extend((0..rates.volatilities().len()).map(|i| format!("rate_volatility[{i}]")));

        let correlation = HybridCorrelation::new(
            path.model.equity_dividend_correlation(),
            rho_equity_rate,
            rho_dividend_rate,
        )?;
        let mut steps = Vec::with_capacity(path.steps.len());
        for (pair, step) in path.times.windows(2).zip(&path.steps) {
            let transition = rates.transition(pair[0], pair[1], 0.0, correlation)?;
            for i in 0..4 {
                let scale = transition.covariance[i][i].sqrt();
                let pivot = if scale == 0.0 {
                    0.0
                } else {
                    step.noise[i][i] / scale
                };
                if !pivot.is_finite() || pivot <= MIN_NORMALIZED_PIVOT {
                    return Err(invalid("singular_hw_parameter_adjoint").into());
                }
            }
            let covariance_tangents = covariance_derivatives(rates, pair[0], pair[1], correlation)?;
            let mut row = Vec::with_capacity(parameter_count);
            for (parameter, dcov) in covariance_tangents.iter().enumerate() {
                let dnoise = cholesky_direction(step.noise, *dcov)?;
                let da = if parameter == 0 {
                    -(pair[1] - pair[0]) * step.decay
                } else {
                    0.0
                };
                let db = if parameter == 0 {
                    b_mean_reversion_derivative(rates.mean_reversion(), step.dt)
                } else {
                    0.0
                };
                if !da.is_finite()
                    || !db.is_finite()
                    || dnoise.iter().flatten().any(|x| !x.is_finite())
                {
                    return Err(invalid("hw_step_parameter_derivative").into());
                }
                row.push(StepTangent {
                    decay: da,
                    integral_loading: db,
                    noise: dnoise,
                });
            }
            if row.len() != parameter_count {
                return Err(invalid("hw_step_parameter_shape").into());
            }
            steps.push(row);
        }

        let mut variance_cache = HashMap::<(u64, u64), Vec<f64>>::new();
        let mut nodes = Vec::with_capacity(path.nodes.len());
        for (index, node) in path.nodes.iter().enumerate() {
            let time = path.times[index];
            let half_integral_variance: Vec<f64> =
                cached_variance_derivative(rates, 0.0, time, &mut variance_cache, parameter_count)?
                    .into_iter()
                    .map(|x| 0.5 * x)
                    .collect();
            if half_integral_variance.len() != parameter_count {
                return Err(invalid("hw_node_parameter_shape").into());
            }
            let mut claim_rows = Vec::with_capacity(node.claims.len());
            for claim in &node.claims {
                let maturity = path.cash_times[claim.index];
                let var_0_t = cached_variance_derivative(
                    rates,
                    0.0,
                    time,
                    &mut variance_cache,
                    parameter_count,
                )?;
                let var_0_m = cached_variance_derivative(
                    rates,
                    0.0,
                    maturity,
                    &mut variance_cache,
                    parameter_count,
                )?;
                let var_t_m = cached_variance_derivative(
                    rates,
                    time,
                    maturity,
                    &mut variance_cache,
                    parameter_count,
                )?;
                let coefficient_tangents = cash_coefficient_rate_derivatives(
                    path.model,
                    path.volatility,
                    rates,
                    rho_equity_rate,
                    rho_dividend_rate,
                    time,
                    maturity,
                    parameter_count,
                )?;
                let mut tangents = Vec::with_capacity(parameter_count);
                for p in 0..parameter_count {
                    let log_amount = 0.5 * (-var_0_m[p] + var_0_t[p] + var_t_m[p]);
                    tangents.push(ClaimTangent {
                        amount: claim.amount * log_amount,
                        duration: if p == 0 {
                            b_mean_reversion_derivative(rates.mean_reversion(), maturity - time)
                        } else {
                            0.0
                        },
                        coefficients: coefficient_tangents[p],
                    });
                }
                claim_rows.push(tangents);
            }
            nodes.push(NodeTangent {
                half_integral_variance,
                claims: claim_rows,
            });
        }

        // The initial reserve is part of the funded risky spot. Its sensitivity
        // flows through every later residual-equity observation.
        let mut risky_spot = vec![0.0; parameter_count];
        for (claim, tangent_row) in path.nodes[0].claims.iter().zip(&nodes[0].claims) {
            let [a, b, c] = claim.coefficients;
            let projection = a + b + c;
            for p in 0..parameter_count {
                let derivative = &tangent_row[p];
                let projection_tangent = derivative.coefficients.iter().sum::<f64>();
                risky_spot[p] -= claim.carry_weight
                    * (derivative.amount * projection + claim.amount * projection_tangent);
            }
        }

        let expiry = *path.times.last().ok_or(invalid("empty_hw_path"))?;
        let var_0_payment = cached_variance_derivative(
            rates,
            0.0,
            payment_time,
            &mut variance_cache,
            parameter_count,
        )?;
        let var_expiry_payment = cached_variance_derivative(
            rates,
            expiry,
            payment_time,
            &mut variance_cache,
            parameter_count,
        )?;
        let payment_log_constant = (0..parameter_count)
            .map(|p| 0.5 * (-var_0_payment[p] + var_expiry_payment[p]))
            .collect::<Vec<_>>();
        let mut payment_duration = vec![0.0; parameter_count];
        payment_duration[0] =
            b_mean_reversion_derivative(rates.mean_reversion(), payment_time - expiry);

        if risky_spot
            .iter()
            .chain(&payment_log_constant)
            .chain(&payment_duration)
            .any(|x| !x.is_finite())
        {
            return Err(invalid("hw_parameter_context").into());
        }
        Ok(Self {
            labels,
            steps,
            nodes,
            risky_spot,
            payment_log_constant,
            payment_duration,
        })
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn payment_log_constant_derivatives(&self) -> &[f64] {
        &self.payment_log_constant
    }

    pub fn payment_duration_derivatives(&self) -> &[f64] {
        &self.payment_duration
    }

    pub fn evolve_rate_tangents(
        &self,
        path: &StochasticDividendHullWhitePathPlan,
        normals: &[f64],
        states: &[StochasticDividendHullWhiteState],
    ) -> Result<Vec<Vec<HullWhiteRateStateTangent>>, MonteCarloError> {
        if normals.len() != path.dimension as usize
            || states.len() != path.times.len()
            || self.steps.len() + 1 != states.len()
        {
            return Err(invalid("hw_state_tangent_shape").into());
        }
        let parameter_count = self.labels.len();
        let mut result = Vec::with_capacity(states.len());
        let mut current = vec![HullWhiteRateStateTangent::default(); parameter_count];
        result.push(current.clone());
        for i in 0..self.steps.len() {
            let z = &normals[4 * i..4 * i + 4];
            let old_x = states[i].rate_factor;
            let mut next = vec![HullWhiteRateStateTangent::default(); parameter_count];
            for p in 0..parameter_count {
                let step = &self.steps[i][p];
                let rate_noise = (0..4).map(|j| step.noise[2][j] * z[j]).sum::<f64>();
                let integral_noise = (0..4).map(|j| step.noise[3][j] * z[j]).sum::<f64>();
                next[p].rate_factor =
                    path.steps[i].decay * current[p].rate_factor + step.decay * old_x + rate_noise;
                next[p].integrated_rate_factor = current[p].integrated_rate_factor
                    + path.steps[i].integral_loading * current[p].rate_factor
                    + step.integral_loading * old_x
                    + integral_noise;
            }
            if next
                .iter()
                .any(|x| !x.rate_factor.is_finite() || !x.integrated_rate_factor.is_finite())
            {
                return Err(invalid("hw_state_tangent").into());
            }
            current = next.clone();
            result.push(next);
        }
        Ok(result)
    }

    pub fn spot_derivatives(
        &self,
        path: &StochasticDividendHullWhitePathPlan,
        index: usize,
        state: StochasticDividendHullWhiteState,
        state_tangents: &[HullWhiteRateStateTangent],
    ) -> Result<Vec<f64>, MonteCarloError> {
        if index >= path.nodes.len()
            || state_tangents.len() != self.labels.len()
            || self.nodes[index].claims.len() != path.nodes[index].claims.len()
        {
            return Err(invalid("hw_spot_tangent_shape").into());
        }
        let node = &path.nodes[index];
        let node_tangent = &self.nodes[index];
        let factors = state.factors;
        let growth = node.residual_growth
            * (state.integrated_rate_factor + node.half_integral_variance).exp();
        let risky = path.risky_spot * growth * factors.equity();
        let mut result = vec![0.0; self.labels.len()];
        for p in 0..result.len() {
            result[p] = self.risky_spot[p] * growth * factors.equity()
                + risky
                    * (state_tangents[p].integrated_rate_factor
                        + node_tangent.half_integral_variance[p]);
        }
        for (claim, tangent_row) in node.claims.iter().zip(&node_tangent.claims) {
            let [a, b, c] = claim.coefficients;
            let projection = a * factors.equity() + b * factors.dividend() + c;
            let rate_discount = (-claim.duration * state.rate_factor).exp();
            for p in 0..result.len() {
                let derivative = &tangent_row[p];
                let projection_tangent = derivative.coefficients[0] * factors.equity()
                    + derivative.coefficients[1] * factors.dividend()
                    + derivative.coefficients[2];
                let value_tangent = rate_discount
                    * (derivative.amount * projection + claim.amount * projection_tangent
                        - claim.amount
                            * projection
                            * (derivative.duration * state.rate_factor
                                + claim.duration * state_tangents[p].rate_factor));
                result[p] += claim.carry_weight * value_tangent;
            }
        }
        if result.iter().any(|x| !x.is_finite()) {
            return Err(invalid("hw_spot_tangent").into());
        }
        Ok(result)
    }
}

fn cached_variance_derivative(
    rates: &HullWhite1Factor,
    start: f64,
    end: f64,
    cache: &mut HashMap<(u64, u64), Vec<f64>>,
    parameter_count: usize,
) -> Result<Vec<f64>, MonteCarloError> {
    let key = (start.to_bits(), end.to_bits());
    if let Some(value) = cache.get(&key) {
        return Ok(value.clone());
    }
    let corr = HybridCorrelation::new(0.0, 0.0, 0.0)?;
    let matrices = covariance_derivatives(rates, start, end, corr)?;
    let value = matrices.iter().map(|m| m[3][3]).collect::<Vec<_>>();
    if value.len() != parameter_count {
        return Err(invalid("hw_variance_tangent_shape").into());
    }
    cache.insert(key, value.clone());
    Ok(value)
}

fn covariance_derivatives(
    rates: &HullWhite1Factor,
    start: f64,
    end: f64,
    correlation: HybridCorrelation,
) -> Result<Vec<[[f64; 4]; 4]>, MonteCarloError> {
    let parameter_count = 1 + rates.volatilities().len();
    if end <= start {
        return Ok(vec![[[0.0; 4]; 4]; parameter_count]);
    }
    let rho = [
        [
            1.0,
            correlation.equity_vol,
            correlation.equity_rate,
            correlation.equity_rate,
        ],
        [
            correlation.equity_vol,
            1.0,
            correlation.vol_rate,
            correlation.vol_rate,
        ],
        [correlation.equity_rate, correlation.vol_rate, 1.0, 1.0],
        [correlation.equity_rate, correlation.vol_rate, 1.0, 1.0],
    ];
    let width = parameter_count * 16;
    let mut integrated = vec![0.0; width];
    for (segment, &left) in rates.volatility_times().iter().enumerate() {
        let right = rates
            .volatility_times()
            .get(segment + 1)
            .copied()
            .unwrap_or(end);
        let lo = start.max(left);
        let hi = end.min(right);
        if hi <= lo {
            continue;
        }
        let sigma = rates.volatilities()[segment];
        let integrand = |u: f64| {
            let tau = end - u;
            let e = (-rates.mean_reversion() * tau).exp();
            let loading = b(rates.mean_reversion(), tau);
            let kernel = [1.0, 1.0, sigma * e, sigma * loading];
            let mut dk = vec![[0.0; 4]; parameter_count];
            dk[0][2] = -tau * sigma * e;
            dk[0][3] = sigma * b_mean_reversion_derivative(rates.mean_reversion(), tau);
            dk[segment + 1][2] = e;
            dk[segment + 1][3] = loading;
            let mut values = vec![0.0; width];
            for p in 0..parameter_count {
                for i in 0..4 {
                    for j in 0..4 {
                        values[p * 16 + i * 4 + j] =
                            rho[i][j] * (dk[p][i] * kernel[j] + kernel[i] * dk[p][j]);
                    }
                }
            }
            values
        };
        let left_value = integrand(lo);
        let mid_value = integrand(0.5 * (lo + hi));
        let right_value = integrand(hi);
        let whole = vector_simpson(lo, hi, &left_value, &mid_value, &right_value);
        let part = integrate_vector(
            &integrand,
            lo,
            hi,
            [left_value, mid_value, right_value],
            whole,
            RATE_QUADRATURE_ABS * (hi - lo) / (end - start),
            RATE_QUADRATURE_DEPTH,
        )?;
        for (target, increment) in integrated.iter_mut().zip(part) {
            *target += increment;
        }
    }
    let mut result = vec![[[0.0; 4]; 4]; parameter_count];
    for p in 0..parameter_count {
        for i in 0..4 {
            for j in 0..4 {
                result[p][i][j] = integrated[p * 16 + i * 4 + j];
            }
        }
    }
    if result.iter().flatten().flatten().any(|x| !x.is_finite()) {
        return Err(invalid("hw_covariance_tangent").into());
    }
    Ok(result)
}

fn vector_simpson(lo: f64, hi: f64, left: &[f64], mid: &[f64], right: &[f64]) -> Vec<f64> {
    (0..left.len())
        .map(|j| (hi - lo) / 6.0 * (left[j] + 4.0 * mid[j] + right[j]))
        .collect()
}

fn integrate_vector(
    f: &impl Fn(f64) -> Vec<f64>,
    lo: f64,
    hi: f64,
    y: [Vec<f64>; 3],
    whole: Vec<f64>,
    absolute: f64,
    depth: u32,
) -> Result<Vec<f64>, MonteCarloError> {
    let mid = 0.5 * (lo + hi);
    let left_mid = 0.5 * (lo + mid);
    let right_mid = 0.5 * (mid + hi);
    let left_y = [y[0].clone(), f(left_mid), y[1].clone()];
    let right_y = [y[1].clone(), f(right_mid), y[2].clone()];
    let left = vector_simpson(lo, mid, &left_y[0], &left_y[1], &left_y[2]);
    let right = vector_simpson(mid, hi, &right_y[0], &right_y[1], &right_y[2]);
    let fine = (0..whole.len())
        .map(|j| left[j] + right[j])
        .collect::<Vec<_>>();
    if (0..whole.len()).all(|j| {
        (fine[j] - whole[j]).abs() <= 15.0 * (absolute + RATE_QUADRATURE_REL * fine[j].abs())
    }) {
        return Ok((0..whole.len())
            .map(|j| fine[j] + (fine[j] - whole[j]) / 15.0)
            .collect());
    }
    if depth == 0 || mid == lo || mid == hi {
        return Err(invalid("hw_rate_quadrature_convergence").into());
    }
    let left_value = integrate_vector(f, lo, mid, left_y, left, absolute * 0.5, depth - 1)?;
    let right_value = integrate_vector(f, mid, hi, right_y, right, absolute * 0.5, depth - 1)?;
    Ok(left_value
        .iter()
        .zip(right_value)
        .map(|(a, b)| a + b)
        .collect())
}

pub(super) fn cholesky_direction(
    loading: [[f64; 4]; 4],
    dcov: [[f64; 4]; 4],
) -> Result<[[f64; 4]; 4], MonteCarloError> {
    let mut derivative = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..=i {
            let prior = (0..j)
                .map(|k| derivative[i][k] * loading[j][k] + loading[i][k] * derivative[j][k])
                .sum::<f64>();
            if i == j {
                if loading[i][i] == 0.0 {
                    return Err(invalid("singular_hw_parameter_adjoint").into());
                }
                derivative[i][j] = (dcov[i][j] - prior) / (2.0 * loading[i][i]);
            } else {
                if loading[j][j] == 0.0 {
                    return Err(invalid("singular_hw_parameter_adjoint").into());
                }
                derivative[i][j] =
                    (dcov[i][j] - prior - loading[i][j] * derivative[j][j]) / loading[j][j];
            }
        }
    }
    Ok(derivative)
}

fn rate_brownian_integral_derivatives(
    rates: &HullWhite1Factor,
    start: f64,
    maturity: f64,
) -> Vec<f64> {
    let mut result = vec![0.0; 1 + rates.volatilities().len()];
    if maturity <= start {
        return result;
    }
    for (segment, &left) in rates.volatility_times().iter().enumerate() {
        let right = rates
            .volatility_times()
            .get(segment + 1)
            .copied()
            .unwrap_or(maturity);
        let lo = start.max(left);
        let hi = maturity.min(right);
        if hi <= lo {
            continue;
        }
        let h0 = maturity - hi;
        let h1 = maturity - lo;
        let integral =
            integrated_b(rates.mean_reversion(), h1) - integrated_b(rates.mean_reversion(), h0);
        let integral_da = integrated_b_mean_reversion_derivative(rates.mean_reversion(), h1)
            - integrated_b_mean_reversion_derivative(rates.mean_reversion(), h0);
        result[0] += rates.volatilities()[segment] * integral_da;
        result[segment + 1] += integral;
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn cash_coefficient_rate_derivatives(
    model: BuehlerDividendModel,
    sigma: f64,
    rates: &HullWhite1Factor,
    rho_equity_rate: f64,
    rho_dividend_rate: f64,
    time: f64,
    maturity: f64,
    parameter_count: usize,
) -> Result<Vec<[f64; 3]>, MonteCarloError> {
    let tau = maturity - time;
    let k = model.mean_reversion();
    let alpha = model.equity_linkage();
    let cf = sigma * rho_equity_rate;
    let cd = model.dividend_volatility() * rho_dividend_rate;
    let total = rate_brownian_integral(rates, time, maturity)?;
    let dtotal = rate_brownian_integral_derivatives(rates, time, maturity);
    let by = (-k * tau - cd * total).exp();
    let mut result = vec![[0.0; 3]; parameter_count];
    for p in 0..parameter_count {
        result[p][1] = -cd * dtotal[p] * by;
    }
    if (cf == 0.0 && cd == 0.0) || k == 0.0 || tau == 0.0 {
        return Ok(result);
    }

    let mut cuts = vec![0.0, tau];
    cuts.extend(
        rates
            .volatility_times()
            .iter()
            .filter(|&&u| u > time && u < maturity)
            .map(|&u| maturity - u),
    );
    cuts.sort_by(f64::total_cmp);
    for p in 0..parameter_count {
        let integrand = |v: f64| -> Result<[f64; 2], MonteCarloError> {
            let tail_start = maturity - v;
            let tail = rate_brownian_integral(rates, tail_start, maturity)?;
            let dtail = rate_brownian_integral_derivatives(rates, tail_start, maturity)[p];
            let exponent_a = -k * v - cd * tail - cf * (total - tail);
            let exponent_c = -k * v - cd * tail;
            let da = alpha * k * exponent_a.exp() * (-cd * dtail - cf * (dtotal[p] - dtail));
            let dc = (1.0 - alpha) * k * exponent_c.exp() * (-cd * dtail);
            if !da.is_finite() || !dc.is_finite() {
                return Err(invalid("cash_rate_derivative_integrand").into());
            }
            Ok([da, dc])
        };
        for interval in cuts.windows(2) {
            let (lo, hi) = (interval[0], interval[1]);
            let samples = [integrand(lo)?, integrand(0.5 * (lo + hi))?, integrand(hi)?];
            let value = integrate(
                &integrand,
                lo,
                hi,
                samples,
                simpson(lo, hi, samples),
                QUADRATURE_ABS * (hi - lo) / tau,
                QUADRATURE_DEPTH,
            )?;
            result[p][0] += value[0];
            result[p][2] += value[1];
        }
    }
    if result.iter().flatten().any(|x| !x.is_finite()) {
        return Err(invalid("cash_rate_derivative_coefficients").into());
    }
    Ok(result)
}

/// d/da int_0^h exp(-a u) du, with a stable Ho--Lee limit.
fn b_mean_reversion_derivative(a: f64, h: f64) -> f64 {
    let z = a * h;
    if z.abs() < 0.1 {
        let mut term = -0.5;
        let mut sum = term;
        for n in 2..24 {
            term *= -z * n as f64 / ((n * n - 1) as f64);
            sum += term;
        }
        return h * h * sum;
    }
    ((1.0 + z) * (-z).exp() - 1.0) / (a * a)
}

fn integrated_b(a: f64, h: f64) -> f64 {
    let z = a * h;
    if z.abs() < 0.1 {
        let mut term = 0.5;
        let mut sum = term;
        for n in 1..20 {
            term *= -z / (n + 2) as f64;
            sum += term;
        }
        h * h * sum
    } else {
        (h - b(a, h)) / a
    }
}

fn integrated_b_mean_reversion_derivative(a: f64, h: f64) -> f64 {
    let z = a * h;
    if z.abs() < 0.1 {
        let mut term = -1.0 / 6.0;
        let mut sum = term;
        for n in 2..24 {
            term *= -z * n as f64 / ((n - 1) * (n + 2)) as f64;
            sum += term;
        }
        return h * h * h * sum;
    }
    (b(a, h) - h - a * b_mean_reversion_derivative(a, h)) / (a * a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_integral_derivatives_match_piecewise_parameter_bumps() {
        let a = 0.37;
        let times = vec![0.0, 0.4, 1.05];
        let vols = vec![0.025, 0.061, 0.044];
        let rates = HullWhite1Factor::new(a, times.clone(), vols.clone()).unwrap();
        let start = 0.16;
        let maturity = 1.42;
        let tangent = rate_brownian_integral_derivatives(&rates, start, maturity);
        let value = |aa: f64, vv: &[f64]| {
            rate_brownian_integral(
                &HullWhite1Factor::new(aa, times.clone(), vv.to_vec()).unwrap(),
                start,
                maturity,
            )
            .unwrap()
        };
        let h = 1e-6;
        let fd_a = (value(a + h, &vols) - value(a - h, &vols)) / (2.0 * h);
        assert!((tangent[0] - fd_a).abs() < 2e-10);
        for j in 0..vols.len() {
            let mut up = vols.clone();
            let mut down = vols.clone();
            up[j] += h;
            down[j] -= h;
            let fd = (value(a, &up) - value(a, &down)) / (2.0 * h);
            assert!(
                (tangent[j + 1] - fd).abs() < 2e-10,
                "knot {j}: {} vs {fd}",
                tangent[j + 1]
            );
        }
    }

    #[test]
    fn mean_reversion_loading_derivatives_have_stable_zero_limit() {
        for h in [1e-4, 0.3, 2.0] {
            let a = 0.0;
            let epsilon = 1e-6;
            let b_fd = (b(a + epsilon, h) - b(a, h)) / epsilon;
            let ib_fd = (integrated_b(a + epsilon, h) - integrated_b(a, h)) / epsilon;
            assert!((b_mean_reversion_derivative(a, h) - b_fd).abs() < 2e-6 * h * h);
            assert!(
                (integrated_b_mean_reversion_derivative(a, h) - ib_fd).abs() < 2e-6 * h * h * h
            );
        }
    }

    #[test]
    fn joint_covariance_and_conditional_claim_tangents_match_parameter_bumps() {
        let times = vec![0.0, 0.35, 0.8, 1.2];
        let vols = vec![0.03, 0.055, 0.04, 0.07];
        let rates = HullWhite1Factor::new(0.42, times.clone(), vols.clone()).unwrap();
        let corr = HybridCorrelation::new(-0.25, 0.22, -0.18).unwrap();
        let derivative = covariance_derivatives(&rates, 0.1, 1.0, corr).unwrap();
        let bump_rates = |parameter: usize, bump: f64| {
            let mut a = 0.42;
            let mut bumped_vols = vols.clone();
            if parameter == 0 {
                a += bump;
            } else {
                bumped_vols[parameter - 1] += bump;
            }
            HullWhite1Factor::new(a, times.clone(), bumped_vols).unwrap()
        };
        let h = 1e-6;
        for (p, derivative_row) in derivative.iter().enumerate() {
            let up = bump_rates(p, h)
                .transition(0.1, 1.0, 0.0, corr)
                .unwrap()
                .covariance;
            let down = bump_rates(p, -h)
                .transition(0.1, 1.0, 0.0, corr)
                .unwrap()
                .covariance;
            for i in 0..4 {
                for j in 0..4 {
                    let fd = (up[i][j] - down[i][j]) / (2.0 * h);
                    assert!(
                        (derivative_row[i][j] - fd).abs() < 2e-9,
                        "covariance p={p} ({i},{j}): {} vs {fd}",
                        derivative_row[i][j]
                    );
                }
            }
        }

        let model = BuehlerDividendModel::new(0.73, 0.61, 0.32, -0.25).unwrap();
        let coefficient = cash_coefficient_rate_derivatives(
            model,
            0.2,
            &rates,
            0.22,
            -0.18,
            0.21,
            1.43,
            derivative.len(),
        )
        .unwrap();
        for (p, coefficient_row) in coefficient.iter().enumerate() {
            let value = |r: &HullWhite1Factor| {
                cash_coefficients(model, 0.2, r, 0.22, -0.18, 0.21, 1.43).unwrap()
            };
            let up = value(&bump_rates(p, h));
            let down = value(&bump_rates(p, -h));
            for c in 0..3 {
                let fd = (up[c] - down[c]) / (2.0 * h);
                assert!(
                    (coefficient_row[c] - fd).abs() < 3e-8,
                    "cash coefficient p={p} c={c}: {} vs {fd}",
                    coefficient_row[c]
                );
            }
        }
    }
}
