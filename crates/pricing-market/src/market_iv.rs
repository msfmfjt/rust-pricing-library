//! Quote-node IV surface: natural cubic total variance in log moneyness,
//! linear total variance in time, and its exact quote transpose.
//! No strike extrapolation or arbitrage repair is silently applied.

use crate::{ImpliedVarianceSurface, MarketError, ThetaRegion, TotalVarianceDerivatives};

pub const MARKET_IV_INTERPOLATION: &str = "natural-cubic-w-linear-time-v1";
type TimeWeights = (Vec<(usize, f64, f64)>, ThetaRegion);

#[derive(Clone, Debug, PartialEq)]
pub struct MarketIvSurface {
    times: Box<[f64]>,
    log_nodes: Box<[f64]>,
    volatilities: Box<[f64]>,
    /// Each column maps nodal total variance to natural spline second derivatives.
    second_derivative_map: Box<[f64]>,
}

impl MarketIvSurface {
    pub fn new(
        times: Vec<f64>,
        log_nodes: Vec<f64>,
        volatilities: Vec<f64>,
    ) -> Result<Self, MarketError> {
        validate_axis(&times, true)?;
        validate_axis(&log_nodes, false)?;
        let count = times
            .len()
            .checked_mul(log_nodes.len())
            .ok_or_else(|| invalid("market_iv_shape", f64::INFINITY))?;
        if count != volatilities.len() {
            return Err(invalid("market_iv_shape", volatilities.len() as f64));
        }
        for &v in &volatilities {
            if !v.is_finite() || v <= 0.0 {
                return Err(invalid("market_iv_quote", v));
            }
        }
        for (i, &t) in times.iter().enumerate() {
            for &v in &volatilities[i * log_nodes.len()..(i + 1) * log_nodes.len()] {
                if !(t * v * v).is_finite() || t * v * v <= 0.0 {
                    return Err(invalid("market_iv_total_variance", t * v * v));
                }
            }
        }
        let n = log_nodes.len();
        let mut map = vec![
            0.0;
            n.checked_mul(n)
                .ok_or_else(|| invalid("market_iv_shape", n as f64))?
        ];
        for col in 0..n {
            let mut y = vec![0.0; n];
            y[col] = 1.0;
            let m = natural_seconds(&log_nodes, &y);
            for row in 0..n {
                map[row * n + col] = m[row];
            }
        }
        if map.iter().any(|v| !v.is_finite()) {
            return Err(invalid("market_iv_spline_conditioning", f64::INFINITY));
        }
        let surface = Self {
            times: times.into(),
            log_nodes: log_nodes.into(),
            volatilities: volatilities.into(),
            second_derivative_map: map.into(),
        };
        // Check every quote as well as the caller's eventual target samples.
        // Cubic interpolation is not an arbitrage-constrained optimizer.
        for &t in surface.maturity_nodes() {
            for &x in surface.log_moneyness_nodes() {
                surface.forward_call_evaluation(t, x, 1.0)?;
            }
        }
        Ok(surface)
    }

    #[must_use]
    pub fn maturity_nodes(&self) -> &[f64] {
        &self.times
    }
    #[must_use]
    pub fn log_moneyness_nodes(&self) -> &[f64] {
        &self.log_nodes
    }
    #[must_use]
    pub fn implied_volatilities(&self) -> &[f64] {
        &self.volatilities
    }

    /// Seeds in order (w, w_x, w_xx, w_t); axes and quote interpretation fixed.
    pub fn transpose_accumulate(
        &self,
        time: f64,
        x: f64,
        seeds: [f64; 4],
        out: &mut [f64],
    ) -> Result<(), MarketError> {
        if out.len() != self.volatilities.len()
            || seeds.iter().any(|v| !v.is_finite())
            || out.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("market_iv_adjoint", out.len() as f64));
        }
        let spatial = self.spatial_weights(x)?;
        let (rows, _) = self.time_weights(time)?;
        let n = self.log_nodes.len();
        for (row, weight, slope) in rows {
            for (j, b) in spatial.iter().enumerate() {
                let i = row * n + j;
                let w_bar = weight * (seeds[0] * b[0] + seeds[1] * b[1] + seeds[2] * b[2])
                    + slope * seeds[3] * b[0];
                out[i] += w_bar * 2.0 * self.times[row] * self.volatilities[i];
                if !out[i].is_finite() {
                    return Err(invalid("market_iv_adjoint", out[i]));
                }
            }
        }
        Ok(())
    }

    fn spatial_weights(&self, x: f64) -> Result<Vec<[f64; 3]>, MarketError> {
        let n = self.log_nodes.len();
        if !x.is_finite() || x < self.log_nodes[0] || x > self.log_nodes[n - 1] {
            return Err(invalid("market_iv_log_moneyness_out_of_range", x));
        }
        let i = self
            .log_nodes
            .partition_point(|v| *v <= x)
            .saturating_sub(1)
            .min(n - 2);
        let h = self.log_nodes[i + 1] - self.log_nodes[i];
        let b = (x - self.log_nodes[i]) / h;
        let a = 1.0 - b;
        let mut weights = Vec::with_capacity(n);
        for j in 0..n {
            let left = self.second_derivative_map[i * n + j];
            let right = self.second_derivative_map[(i + 1) * n + j];
            let mut w = [
                ((a * a * a - a) * left + (b * b * b - b) * right) * h * h / 6.0,
                ((1.0 - 3.0 * a * a) * left + (3.0 * b * b - 1.0) * right) * h / 6.0,
                a * left + b * right,
            ];
            if j == i {
                w[0] += a;
                w[1] -= 1.0 / h;
            }
            if j == i + 1 {
                w[0] += b;
                w[1] += 1.0 / h;
            }
            weights.push(w);
        }
        Ok(weights)
    }

    fn time_weights(&self, t: f64) -> Result<TimeWeights, MarketError> {
        if !t.is_finite() || t <= 0.0 {
            return Err(invalid("market_iv_query_time", t));
        }
        let last = self.times.len() - 1;
        if t < self.times[0] {
            return Ok((
                vec![(0, t / self.times[0], 1.0 / self.times[0])],
                ThetaRegion::ShortExtrapolated,
            ));
        }
        if t >= self.times[last] {
            return Ok((
                vec![(last, t / self.times[last], 1.0 / self.times[last])],
                if t == self.times[last] {
                    ThetaRegion::Knot
                } else {
                    ThetaRegion::LongExtrapolated
                },
            ));
        }
        let i = self.times.partition_point(|v| *v <= t) - 1;
        let h = self.times[i + 1] - self.times[i];
        let b = (t - self.times[i]) / h;
        Ok((
            vec![(i, 1.0 - b, -1.0 / h), (i + 1, b, 1.0 / h)],
            if t == self.times[i] {
                ThetaRegion::Knot
            } else {
                ThetaRegion::Interpolated
            },
        ))
    }
}

impl ImpliedVarianceSurface for MarketIvSurface {
    fn total_variance_derivatives(
        &self,
        time: f64,
        x: f64,
    ) -> Result<TotalVarianceDerivatives, MarketError> {
        let spatial = self.spatial_weights(x)?;
        let (rows, region) = self.time_weights(time)?;
        let mut v = [0.0; 4];
        for (row, weight, slope) in rows {
            for (j, b) in spatial.iter().enumerate() {
                let sigma = self.volatilities[row * self.log_nodes.len() + j];
                let w = self.times[row] * sigma * sigma;
                for k in 0..3 {
                    v[k] += weight * b[k] * w;
                }
                v[3] += slope * b[0] * w;
            }
        }
        if v.iter().any(|v| !v.is_finite()) || v[0] <= 0.0 || v[3] <= 0.0 {
            return Err(invalid("market_iv_nonpositive_or_nonfinite_variance", v[3]));
        }
        // This surface has no SSVI theta parameter; theta is the local w value.
        Ok(TotalVarianceDerivatives {
            total_variance: v[0],
            log_moneyness_derivative: v[1],
            log_moneyness_second_derivative: v[2],
            time_derivative: v[3],
            theta: v[0],
            theta_derivative: v[3],
            theta_region: region,
        })
    }
}

fn natural_seconds(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut diagonal = vec![1.0; n];
    let mut upper = vec![0.0; n];
    let mut rhs = vec![0.0; n];
    for i in 1..n - 1 {
        let left = x[i] - x[i - 1];
        let right = x[i + 1] - x[i];
        let multiplier = left / diagonal[i - 1];
        diagonal[i] = 2.0 * (left + right) - multiplier * upper[i - 1];
        upper[i] = right;
        rhs[i] =
            6.0 * ((y[i + 1] - y[i]) / right - (y[i] - y[i - 1]) / left) - multiplier * rhs[i - 1];
    }
    for i in (1..n - 1).rev() {
        rhs[i] = (rhs[i] - upper[i] * rhs[i + 1]) / diagonal[i];
    }
    rhs
}

fn invalid(parameter: &'static str, value: f64) -> MarketError {
    MarketError::InvalidSurfaceParameter {
        parameter,
        bits: value.to_bits(),
    }
}
fn validate_axis(axis: &[f64], time: bool) -> Result<(), MarketError> {
    if axis.len() < 2
        || axis.iter().any(|v| !v.is_finite() || (time && *v <= 0.0))
        || axis
            .windows(2)
            .any(|p| p[0] >= p[1] || !(p[1] - p[0]).is_finite())
    {
        return Err(invalid(
            if time {
                "market_iv_maturities"
            } else {
                "market_iv_log_nodes"
            },
            axis.len() as f64,
        ));
    }
    Ok(())
}
