//! Transpose the actual finite hybrid history, at fixed Brownian correlations.
//! H changes every older-cell weight, the exact near cell and grid centering.
//! Prepared coefficient derivatives are shared by all paths; no bumps are used.
use super::*;

pub(in crate::engine) struct RoughParameterReverse {
    weight_h: Box<[Box<[f64]>]>,
    variance_h: Box<[f64]>,
    near_h: Box<[[f64; 2]]>,
}

// d log(mean power kernel)/dH. x/(exp(x)-1)-1 is evaluated without
// cancellation for narrow lag intervals; log1p keeps adjacent lags distinct.
fn log_weight_h(h: f64, lo: f64, hi: f64) -> f64 {
    let p = h + 0.5;
    if lo == 0.0 {
        return 0.5 / h + hi.ln() - 1.0 / p;
    }
    let x = -p * ((lo - hi) / hi).ln_1p();
    let correction = if x.abs() < 1e-3 {
        let x2 = x * x;
        -0.5 * x + x2 / 12.0 - x2 * x2 / 720.0 + x2 * x2 * x2 / 30240.0
    } else if x.is_infinite() {
        -1.0
    } else {
        // Equivalent to x/expm1(x)-1; stable even at very large x.
        x * (-x).exp() / (-(-x).exp_m1()) - 1.0
    };
    0.5 / h + hi.ln() + correction / p
}

impl RoughParameterReverse {
    pub(in crate::engine) fn new(
        kernel: &RoughDividendKernel,
        times: &[f64],
    ) -> Result<Self, StochasticDividendError> {
        if times.len() != kernel.weights.len() || times.len() != kernel.steps.len() + 1 {
            return Err(invalid("rough_reverse_grid"));
        }
        let h = kernel.factor.hurst();
        let p = h + 0.5;
        let mut weight_h = vec![Vec::new().into_boxed_slice()];
        let mut variance_h = vec![0.0];
        let mut near_h = Vec::with_capacity(kernel.steps.len());
        for (i, &time) in times.iter().enumerate().skip(1) {
            let dt = time - times[i - 1];
            let log_dt = dt.ln();
            let cell = kernel.steps[i - 1];
            let da = cell.average * (0.5 / h + log_dt - 1.0 / p);
            // derivative of dt^H * (1/2-H)/(H+1/2), with no division by
            // (1/2-H). This includes the one-sided derivative at H=1/2.
            let db = cell.residual * log_dt - dt.powf(h) / (p * p);
            near_h.push([da, db]);
            let mut row = Vec::with_capacity(i - 1);
            let mut variance = NeumaierSum::new();
            variance.add(2.0 * log_dt * dt.powf(2.0 * h));
            for (j, &w) in kernel.weights[i].iter().enumerate() {
                let dw = w * log_weight_h(h, time - times[j + 1], time - times[j]);
                row.push(dw);
                variance.add(2.0 * w * dw * (times[j + 1] - times[j]));
            }
            weight_h.push(row.into_boxed_slice());
            variance_h.push(variance.total());
        }
        if weight_h
            .iter()
            .flat_map(|r| r.iter())
            .chain(&variance_h)
            .chain(near_h.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err(invalid("rough_reverse_coefficients"));
        }
        Ok(Self {
            weight_h: weight_h.into_boxed_slice(),
            variance_h: variance_h.into_boxed_slice(),
            near_h: near_h.into_boxed_slice(),
        })
    }

    pub(in crate::engine) fn pullback(
        &self,
        kernel: &RoughDividendKernel,
        normals: &[f64],
        loading_bars: &[f64],
    ) -> Result<[f64; 2], StochasticDividendError> {
        if loading_bars.len() != kernel.steps.len() || loading_bars.iter().any(|v| !v.is_finite()) {
            return Err(invalid("rough_reverse_loading_seeds"));
        }
        let (drivers, increments) = kernel.driver_path(normals)?;
        let eta = kernel.factor.vol_of_vol();
        let mut h_bar = NeumaierSum::new();
        let mut eta_bar = NeumaierSum::new();
        for (i, &bar) in loading_bars.iter().enumerate().rev() {
            if bar == 0.0 {
                continue;
            }
            let x = drivers[i];
            let v = kernel.variances[i];
            let loading = (0.5 * eta * x - 0.25 * eta * eta * v).exp();
            positive(loading, "rough_reverse_loading")?;
            let exponent_bar = bar * loading;
            eta_bar.add(exponent_bar * (0.5 * x - 0.5 * eta * v));
            h_bar.add(-0.25 * exponent_bar * eta * eta * self.variance_h[i]);
            if i == 0 {
                continue;
            }
            // X_i = A_i dW_(i-1) + B_i z_(i-1,3) + sum_(j<i-1) w_ij dW_j.
            // The Brownian increments have no H/eta dependence. Transpose each
            // history row into its H-dependent coefficients, not into future X.
            let driver_bar = 0.5 * eta * exponent_bar;
            let [da, db] = self.near_h[i - 1];
            h_bar.add(driver_bar * da * increments[i - 1]);
            h_bar.add(driver_bar * db * normals[4 * (i - 1) + 3]);
            for (&dw, &increment) in self.weight_h[i].iter().zip(&increments) {
                h_bar.add(driver_bar * dw * increment);
            }
        }
        let result = [h_bar.total(), eta_bar.total()];
        if result.iter().any(|v| !v.is_finite()) {
            return Err(invalid("rough_parameter_reverse"));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn kernel(h: f64, eta: f64, times: &[f64]) -> RoughDividendKernel {
        RoughDividendKernel::compile(
            BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap(),
            RoughBergomi::new(h, eta, -0.4).unwrap(),
            0.15,
            times,
        )
        .unwrap()
    }
    #[test]
    fn kernel_h_derivatives_match_independent_power_integral_quadrature() {
        for h in [0.01, 0.1, 0.49, 0.5] {
            let factor = RoughBergomi::new(h, 0.6, -0.4).unwrap();
            for (lo, hi) in [(0.1, 0.100001), (0.1, 0.3), (1.0, 2.0)] {
                let n = 2048;
                let dx = (hi - lo) / n as f64;
                let mut sum = 0.0;
                for j in 0..=n {
                    let x = lo + dx * j as f64;
                    let c = if j == 0 || j == n {
                        1.0
                    } else if j % 2 == 0 {
                        2.0
                    } else {
                        4.0
                    };
                    sum += c * (2.0 * h).sqrt() * x.powf(h - 0.5) * (0.5 / h + x.ln());
                }
                let expected = sum * dx / (3.0 * (hi - lo));
                let actual = factor.average_kernel(lo, hi).unwrap() * log_weight_h(h, lo, hi);
                assert!(
                    (actual - expected).abs() < 2e-10,
                    "H={h}, lags=({lo},{hi}): {actual} vs {expected}"
                );
            }
        }
    }
    #[test]
    fn history_reverse_matches_recompiled_loadings_including_endpoint_and_eta_zero() {
        let times = [0.0, 0.031, 0.17, 0.36, 0.8, 1.0];
        let z: Vec<f64> = (0..20).map(|i| ((i as f64 + 0.4) * 1.7).sin()).collect();
        let seeds = [0.3, -0.7, 0.2, 0.9, -0.1];
        for h in [0.01, 0.1, 0.49, 0.5] {
            for eta in [0.0, 0.6] {
                let k = kernel(h, eta, &times);
                let ctx = RoughParameterReverse::new(&k, &times).unwrap();
                let actual = ctx.pullback(&k, &z, &seeds).unwrap();
                let value = |h, eta| {
                    kernel(h, eta, &times)
                        .volatility_loadings(&z)
                        .unwrap()
                        .iter()
                        .zip(seeds)
                        .map(|(x, s)| x * s)
                        .sum::<f64>()
                };
                for (p, &aad) in actual.iter().enumerate() {
                    let eps = 1e-7;
                    let expected = if p == 0 {
                        if h == 0.5 {
                            (value(h, eta) - value(h - eps, eta)) / eps
                        } else {
                            (value(h + eps, eta) - value(h - eps, eta)) / (2.0 * eps)
                        }
                    } else if eta == 0.0 {
                        (value(h, eta + eps) - value(h, eta)) / eps
                    } else {
                        (value(h, eta + eps) - value(h, eta - eps)) / (2.0 * eps)
                    };
                    assert!(
                        (aad - expected).abs() < 3e-6,
                        "H={h}, eta={eta}, p={p}: {aad} vs {expected}"
                    );
                }
                // Last-step normals are future information relative to every
                // left-endpoint volatility: neither value nor H/eta risk uses them.
                let mut future = z.clone();
                future[16..].fill(5.0);
                assert_eq!(
                    k.volatility_loadings(&z).unwrap(),
                    k.volatility_loadings(&future).unwrap()
                );
                assert_eq!(actual, ctx.pullback(&k, &future, &seeds).unwrap());
            }
        }
    }
}
