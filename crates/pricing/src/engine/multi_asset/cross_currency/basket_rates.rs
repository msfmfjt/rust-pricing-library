//! Joint Hull–White transitions under the settlement money-market measure.
use super::basket::TOLERANCE;
use super::rates::centered_b_product;
use crate::market::CorrelationFactor;
use crate::models::HullWhite1Factor;
use crate::models::hull_white::{b, integrated_b};
use crate::multi_asset::MultiAssetError as Error;

struct CurrencyStep {
    reversion: f64,
    decay: f64,
    loading: f64,
    integral_shift: f64,
    endpoint_drift: f64,
    endpoint_weight: f64,
    integral_weight: f64,
}

/// Conditional on all market Brownian increments, each rate contributes one
/// bridge innovation shared by its endpoint and integral. Rate bridges across
/// currencies have the covariance of their centered OU kernels.
pub(super) struct BasketRateStep {
    currencies: Vec<CurrencyStep>,
    bridge_loading: Vec<f64>,
}

impl BasketRateStep {
    pub(super) fn compile(
        models: &[HullWhite1Factor],
        drivers: &CorrelationFactor,
        fx_vols: &[f64],
        times: &[f64],
    ) -> Result<Vec<Self>, Error> {
        let count = models.len();
        let width = drivers.dimension();
        let rate_start = width - count;
        let fx_start = rate_start - fx_vols.len();
        let corr = |i: usize, j: usize| drivers.canonical()[i * width + j];
        times
            .windows(2)
            .map(|pair| {
                let (start, end) = (pair[0], pair[1]);
                let dt = end - start;
                let mut kernels = Vec::with_capacity(count);
                let mut scales = Vec::with_capacity(count);
                let mut currencies = Vec::with_capacity(count);
                for (i, m) in models.iter().enumerate() {
                    let a = m.mean_reversion();
                    let vol =
                        m.volatilities()[m.volatility_times().partition_point(|&t| t <= start) - 1];
                    let kernel = centered_b_product(a, a, dt).sqrt();
                    if !(kernel.is_finite() && kernel > 0.) {
                        return Err(Error::Invalid("invalid Hull-White bridge variance"));
                    }
                    kernels.push(kernel);
                    scales.push(vol * kernel);
                    let loading = b(a, dt);
                    let mean_integral = integrated_b(a, dt);
                    let quanto = if i == 0 {
                        0.
                    } else {
                        -fx_vols[i - 1] * corr(fx_start + i - 1, rate_start + i) * vol
                    };
                    currencies.push(CurrencyStep {
                        reversion: a,
                        decay: (-a * dt).exp(),
                        loading,
                        integral_shift: m.integrated_shift(start, end).map_err(Error::numerical)?
                            + quanto * mean_integral,
                        endpoint_drift: quanto * loading,
                        endpoint_weight: vol * loading / dt.sqrt(),
                        integral_weight: vol * mean_integral / dt.sqrt(),
                    });
                }
                let mut correlation = vec![vec![0.; count]; count];
                for i in 0..count {
                    correlation[i][i] = 1.;
                    for j in 0..i {
                        let (a, k) = (models[i].mean_reversion(), models[j].mean_reversion());
                        let kernel_corr = if a == k {
                            1.
                        } else {
                            centered_b_product(a, k, dt) / kernels[i] / kernels[j]
                        };
                        if !kernel_corr.is_finite() || kernel_corr.abs() > 1. + 1e-12 {
                            return Err(Error::Invalid("invalid Hull-White bridge correlation"));
                        }
                        let rho = corr(rate_start + i, rate_start + j) * kernel_corr.clamp(-1., 1.);
                        correlation[i][j] = rho;
                        correlation[j][i] = rho;
                    }
                }
                let factor = CorrelationFactor::compile(correlation, TOLERANCE)?;
                let bridge_loading = factor
                    .lower()
                    .chunks_exact(count)
                    .zip(scales)
                    .flat_map(|(row, scale)| row.iter().map(move |x| x * scale))
                    .collect();
                Ok(Self {
                    currencies,
                    bridge_loading,
                })
            })
            .collect()
    }

    pub(super) fn advance(
        &self,
        factors: &mut [f64],
        integrated: &mut [f64],
        normals: &[f64],
        bridge: &[f64],
    ) {
        let count = self.currencies.len();
        for (i, c) in self.currencies.iter().enumerate() {
            let residual = self.bridge_loading[i * count..i * count + i + 1]
                .iter()
                .zip(bridge)
                .map(|(a, z)| a * z)
                .sum::<f64>();
            integrated[i] += c.integral_shift
                + c.loading * factors[i]
                + c.integral_weight * normals[i]
                + residual;
            factors[i] = c.decay * factors[i] + c.endpoint_weight * normals[i]
                - c.reversion * residual
                + c.endpoint_drift;
        }
    }
}
