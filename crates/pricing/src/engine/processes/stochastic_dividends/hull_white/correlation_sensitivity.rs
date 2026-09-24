//! Raw correlation tangents at fixed independent normal coordinates. One
//! direction changes one symmetric Brownian-correlation pair, without bumps.
use super::rate_sensitivity::cholesky_direction;
use super::*;
use crate::models::BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES;
use pricing_numerics::CorrelationFactor;

const MIN_PIVOT: f64 = 1e-10;
const LABELS: [&str; 3] = [
    "equity_dividend_correlation",
    "equity_rate_correlation",
    "dividend_rate_correlation",
];

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::engine) struct HullWhiteCorrelationStateTangent {
    // The equity factor uses z[0] alone, so its correlation tangent is zero.
    dividend_factor: f64,
    pub rate_factor: f64,
    pub integrated_rate_factor: f64,
}

pub(in crate::engine) struct HullWhiteCorrelationSensitivityContext {
    noise: Vec<[[[f64; 4]; 4]; 3]>,
    claims: Vec<Vec<[[f64; 3]; 3]>>,
    risky_spot: [f64; 3],
}

impl HullWhiteCorrelationSensitivityContext {
    pub fn new(
        path: &StochasticDividendHullWhitePathPlan,
        rates: &HullWhite1Factor,
        rho_equity_rate: f64,
        rho_dividend_rate: f64,
    ) -> Result<Self, MonteCarloError> {
        let rho = path.model.equity_dividend_correlation();
        let instantaneous = CorrelationFactor::compile(
            vec![
                vec![1.0, rho, rho_equity_rate],
                vec![rho, 1.0, rho_dividend_rate],
                vec![rho_equity_rate, rho_dividend_rate, 1.0],
            ],
            BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES,
        )
        .map_err(|_| invalid("singular_hw_correlation_adjoint"))?;
        // Finite-step covariance can be full rank even when the instantaneous
        // driver law is on the PSD boundary. Raw two-sided partials need both.
        if instantaneous
            .diagnostics()
            .pivots
            .iter()
            .any(|p| !p.is_finite() || *p <= MIN_PIVOT)
        {
            return Err(invalid("singular_hw_correlation_adjoint").into());
        }
        let correlation = HybridCorrelation::new(rho, rho_equity_rate, rho_dividend_rate)?;
        let mut noise = Vec::with_capacity(path.steps.len());
        for (pair, step) in path.times.windows(2).zip(&path.steps) {
            let covariance = rates
                .transition(pair[0], pair[1], 0.0, correlation)?
                .covariance;
            for (i, row) in covariance.iter().enumerate() {
                let pivot = step.noise[i][i] / row[i].sqrt();
                if !pivot.is_finite() || pivot <= MIN_PIVOT {
                    return Err(invalid("singular_hw_correlation_adjoint").into());
                }
            }
            let directions = covariance_derivatives(rates, pair[0], pair[1])?;
            let mut row = [[[0.0; 4]; 4]; 3];
            for (target, direction) in row.iter_mut().zip(directions) {
                *target = cholesky_direction(step.noise, direction)?;
                if target.iter().flatten().any(|x| !x.is_finite()) {
                    return Err(invalid("hw_correlation_noise_derivative").into());
                }
            }
            noise.push(row);
        }
        let mut claims = Vec::with_capacity(path.nodes.len());
        for (index, node) in path.nodes.iter().enumerate() {
            let mut row = Vec::with_capacity(node.claims.len());
            for claim in &node.claims {
                row.push(cash_coefficient_derivatives(
                    path.model,
                    path.volatility,
                    rates,
                    rho_equity_rate,
                    rho_dividend_rate,
                    path.times[index],
                    path.cash_times[claim.index],
                )?);
            }
            claims.push(row);
        }
        let mut risky_spot = [0.0; 3];
        for (claim, directions) in path.nodes[0].claims.iter().zip(&claims[0]) {
            for (derivative, coefficients) in risky_spot.iter_mut().zip(directions) {
                *derivative -= claim.carry_weight * claim.amount * coefficients.iter().sum::<f64>();
            }
        }
        if risky_spot.iter().any(|x| !x.is_finite()) {
            return Err(invalid("hw_correlation_funding_derivative").into());
        }
        Ok(Self {
            noise,
            claims,
            risky_spot,
        })
    }

    pub fn labels(&self) -> &'static [&'static str] {
        &LABELS
    }

    pub fn evolve_tangents(
        &self,
        path: &StochasticDividendHullWhitePathPlan,
        normals: &[f64],
        states: &[StochasticDividendHullWhiteState],
    ) -> Result<Vec<[HullWhiteCorrelationStateTangent; 3]>, MonteCarloError> {
        if normals.len() != path.dimension as usize
            || states.len() != path.times.len()
            || self.noise.len() + 1 != states.len()
        {
            return Err(invalid("hw_correlation_state_shape").into());
        }
        let rho = path.model.equity_dividend_correlation();
        let root = ((1.0 - rho) * (1.0 + rho)).sqrt();
        let mut current = [HullWhiteCorrelationStateTangent::default(); 3];
        let mut result = Vec::with_capacity(states.len());
        result.push(current);
        for (i, step) in path.steps.iter().enumerate() {
            let z = &normals[4 * i..4 * i + 4];
            let mut next = [HullWhiteCorrelationStateTangent::default(); 3];
            let (a, b) = super::super::decay(path.model.mean_reversion(), 0.5 * step.dt);
            let half = super::super::blend(
                a,
                b,
                states[i].factors.dividend(),
                path.model.target(states[i].factors.equity()),
            );
            let v = path.model.dividend_volatility() * step.dt.sqrt();
            let ed = (-0.5 * v * v + v * (rho * z[0] + root * z[1])).exp();
            next[0].dividend_factor =
                a * ed * (a * current[0].dividend_factor + half * v * (z[0] - rho / root * z[1]));
            for (p, next) in next.iter_mut().enumerate() {
                let dnoise = &self.noise[i][p];
                next.rate_factor = step.decay * current[p].rate_factor
                    + dnoise[2].iter().zip(z).map(|(a, b)| a * b).sum::<f64>();
                next.integrated_rate_factor = current[p].integrated_rate_factor
                    + step.integral_loading * current[p].rate_factor
                    + dnoise[3].iter().zip(z).map(|(a, b)| a * b).sum::<f64>();
            }
            if next.iter().any(|s| {
                !s.dividend_factor.is_finite()
                    || !s.rate_factor.is_finite()
                    || !s.integrated_rate_factor.is_finite()
            }) {
                return Err(invalid("hw_correlation_state_derivative").into());
            }
            current = next;
            result.push(next);
        }
        Ok(result)
    }

    /// Post- and pre-cash derivatives. Only the latter includes the tangent of
    /// the realized cash jump; absent pre-event payoff seeds are zero.
    pub fn spot_derivatives(
        &self,
        path: &StochasticDividendHullWhitePathPlan,
        index: usize,
        state: StochasticDividendHullWhiteState,
        tangents: &[HullWhiteCorrelationStateTangent; 3],
    ) -> Result<[(f64, f64); 3], MonteCarloError> {
        if index >= path.nodes.len() || self.claims[index].len() != path.nodes[index].claims.len() {
            return Err(invalid("hw_correlation_spot_shape").into());
        }
        let node = &path.nodes[index];
        let f = state.factors.equity();
        let y = state.factors.dividend();
        let growth = node.residual_growth
            * (state.integrated_rate_factor + node.half_integral_variance).exp();
        let mut result = [(0.0, 0.0); 3];
        for (p, (post, pre)) in result.iter_mut().enumerate() {
            *post = growth
                * f
                * (self.risky_spot[p] + path.risky_spot * tangents[p].integrated_rate_factor);
            for (claim, directions) in node.claims.iter().zip(&self.claims[index]) {
                let [a, b, c] = claim.coefficients;
                let [da, db, dc] = directions[p];
                let projection = a * f + b * y + c;
                *post += claim.carry_weight
                    * claim.amount
                    * (-claim.duration * state.rate_factor).exp()
                    * (da * f + db * y + dc + b * tangents[p].dividend_factor
                        - claim.duration * tangents[p].rate_factor * projection);
            }
            *pre = *post + node.event_cash.unwrap_or(0.0) * tangents[p].dividend_factor;
            if !post.is_finite() || !pre.is_finite() {
                return Err(invalid("hw_correlation_spot_derivative").into());
            }
        }
        Ok(result)
    }
}

fn covariance_derivatives(
    rates: &HullWhite1Factor,
    start: f64,
    end: f64,
) -> Result<[[[f64; 4]; 4]; 3], MonteCarloError> {
    // Unit-correlation kernels, not C_ij / rho_ij: zero entries are supported.
    let kernels = rates
        .transition(start, end, 0.0, HybridCorrelation::new(0.0, 1.0, 0.0)?)?
        .covariance;
    let mut result = [[[0.0; 4]; 4]; 3];
    result[0][0][1] = end - start;
    result[0][1][0] = end - start;
    for j in 2..4 {
        result[1][0][j] = kernels[0][j];
        result[1][j][0] = kernels[0][j];
        result[2][1][j] = kernels[0][j];
        result[2][j][1] = kernels[0][j];
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn cash_coefficient_derivatives(
    model: BuehlerDividendModel,
    sigma: f64,
    rates: &HullWhite1Factor,
    rho_f: f64,
    rho_d: f64,
    time: f64,
    maturity: f64,
) -> Result<[[f64; 3]; 3], MonteCarloError> {
    let tau = maturity - time;
    let k = model.mean_reversion();
    let alpha = model.equity_linkage();
    let nu = model.dividend_volatility();
    let cf = sigma * rho_f;
    let cd = nu * rho_d;
    let total = rate_brownian_integral(rates, time, maturity)?;
    let mut result = [[0.0; 3]; 3];
    result[2][1] = -nu * total * (-k * tau - cd * total).exp();
    if k == 0.0 || tau == 0.0 {
        return Ok(result);
    }
    // The derivative is not zero when cf=cd=0: changing a zero correlation
    // activates the Gaussian tilt even though primal coefficients simplify.
    let mut cuts = vec![0.0, tau];
    cuts.extend(
        rates
            .volatility_times()
            .iter()
            .filter(|&&u| u > time && u < maturity)
            .map(|&u| maturity - u),
    );
    cuts.sort_by(f64::total_cmp);
    for (p, coefficients) in result.iter_mut().enumerate().skip(1) {
        let integrand = |v: f64| -> Result<[f64; 2], MonteCarloError> {
            let tail = rate_brownian_integral(rates, maturity - v, maturity)?;
            let common = -k * v - cd * tail;
            let (d_a, d_c) = if p == 1 {
                (-sigma * (total - tail), 0.0)
            } else {
                (-nu * tail, -nu * tail)
            };
            let value = [
                alpha * k * (common - cf * (total - tail)).exp() * d_a,
                (1.0 - alpha) * k * common.exp() * d_c,
            ];
            if value.iter().any(|x| !x.is_finite()) {
                return Err(invalid("cash_correlation_derivative_integrand").into());
            }
            Ok(value)
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
            coefficients[0] += value[0];
            coefficients[2] += value[1];
        }
    }
    if result.iter().flatten().any(|x| !x.is_finite()) {
        return Err(invalid("cash_correlation_derivative_coefficients").into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlation_cholesky_and_cash_directions_match_two_width_recompiles() {
        for a in [0.0, 0.4] {
            let rates =
                HullWhite1Factor::new(a, vec![0.0, 0.4, 1.15], vec![0.04, 0.07, 0.09]).unwrap();
            for rho in [[0.0; 3], [-0.25, 0.25, -0.2]] {
                let covariance = |r: [f64; 3]| {
                    rates
                        .transition(
                            0.2,
                            1.0,
                            0.0,
                            HybridCorrelation::new(r[0], r[1], r[2]).unwrap(),
                        )
                        .unwrap()
                        .covariance
                };
                let loading = covariance_loading(covariance(rho)).unwrap();
                let directions = covariance_derivatives(&rates, 0.2, 1.0).unwrap();
                for (p, dcov) in directions.iter().enumerate() {
                    let derivative = cholesky_direction(loading, *dcov).unwrap();
                    for h in [1e-5, 1e-6] {
                        let mut up = rho;
                        let mut down = rho;
                        up[p] += h;
                        down[p] -= h;
                        let upper = covariance_loading(covariance(up)).unwrap();
                        let lower = covariance_loading(covariance(down)).unwrap();
                        for (i, row) in derivative.iter().enumerate() {
                            for (j, &value) in row.iter().enumerate() {
                                let fd = (upper[i][j] - lower[i][j]) / (2.0 * h);
                                assert!(
                                    (value - fd).abs() < 3e-8,
                                    "Cholesky a={a}, rho={rho:?}, p={p}, {i},{j}, {value} != {fd}"
                                );
                            }
                        }
                    }
                }
                for d in [
                    [0.7, 0.6, 0.35],
                    [0.0, 0.6, 0.35],
                    [0.7, 0.0, 0.0],
                    [0.7, 1.0, 0.35],
                ] {
                    let model = BuehlerDividendModel::new(d[0], d[1], d[2], rho[0]).unwrap();
                    let derivatives =
                        cash_coefficient_derivatives(model, 0.2, &rates, rho[1], rho[2], 0.2, 1.4)
                            .unwrap();
                    for (p, derivative) in derivatives.iter().enumerate() {
                        for h in [1e-5, 1e-6] {
                            let mut up = rho;
                            let mut down = rho;
                            up[p] += h;
                            down[p] -= h;
                            let cash = |r: [f64; 3]| {
                                cash_coefficients(
                                    BuehlerDividendModel::new(d[0], d[1], d[2], r[0]).unwrap(),
                                    0.2,
                                    &rates,
                                    r[1],
                                    r[2],
                                    0.2,
                                    1.4,
                                )
                                .unwrap()
                            };
                            let upper = cash(up);
                            let lower = cash(down);
                            for (i, &value) in derivative.iter().enumerate() {
                                let fd = (upper[i] - lower[i]) / (2.0 * h);
                                assert!(
                                    (value - fd).abs() < 3e-8,
                                    "cash a={a}, rho={rho:?}, p={p}, i={i}, {value} != {fd}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn instantaneous_boundary_is_rejected_when_integrated_covariance_is_full_rank() {
        let request = crate::parse_request_json(
            include_bytes!("../../../../../../../fixtures/v1/pricing_request.golden.json"),
            crate::JsonLimits::DEFAULT,
        )
        .unwrap();
        let rates = HullWhite1Factor::new(0.4, vec![0.0, 0.3], vec![0.04, 0.07]).unwrap();
        let grid = LocalVolTimeGrid::compile(vec![0.0, 1.0], 1.0).unwrap();
        let path = StochasticDividendHullWhitePathPlan::compile_bs(
            request.market().equity().forward(),
            BuehlerDividendModel::new(0.7, 0.6, 0.35, 0.0).unwrap(),
            0.2,
            &rates,
            1.0,
            0.0,
            &grid,
        )
        .unwrap();
        let cov = rates
            .transition(
                0.0,
                1.0,
                0.0,
                HybridCorrelation::new(0.0, 1.0, 0.0).unwrap(),
            )
            .unwrap()
            .covariance;
        for (i, row) in cov.iter().enumerate() {
            assert!(path.steps[0].noise[i][i] / row[i].sqrt() > 1e-6);
        }
        assert!(HullWhiteCorrelationSensitivityContext::new(&path, &rates, 1.0, 0.0).is_err());
    }
}
