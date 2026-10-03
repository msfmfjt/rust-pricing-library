use crate::models::rough_volatility::invalid;
use crate::models::{HullWhiteError, RoughBergomi};
use pricing_numerics::gamma_half_to_two;

/// Shared hybrid kernel. Near-cell integrals have their exact joint law with
/// the corresponding Brownian increment; older cells use L2 mean weights.
#[derive(Clone, Debug)]
pub(super) struct PowerKernel {
    pub weights: Vec<Box<[f64]>>,
    pub variances: Vec<f64>,
    pub near_loading: Vec<f64>,
    pub near_residual: Vec<f64>,
    pub fractional_scale: f64,
}
impl PowerKernel {
    pub fn compile(hurst: f64, times: &[f64]) -> Result<Self, HullWhiteError> {
        let model = RoughBergomi::new(hurst, 0.0, 0.0)?;
        let (weights, variances) = model.volterra_weights(times)?;
        let mut near_loading = Vec::new();
        let mut near_residual = Vec::new();
        // Stable positive equivalent of 1-2H/(H+1/2)^2.
        let residual_fraction = ((0.5 - hurst) / (0.5 + hurst)).powi(2);
        for window in times.windows(2) {
            let dt = window[1] - window[0];
            near_loading.push((2.0 * hurst).sqrt() * dt.powf(hurst - 0.5) / (hurst + 0.5));
            near_residual.push(dt.powf(hurst) * residual_fraction.sqrt());
        }
        let gamma = gamma_half_to_two(hurst + 0.5).ok_or(invalid("rough_gamma"))?;
        Ok(Self {
            weights,
            variances,
            near_loading,
            near_residual,
            fractional_scale: 1.0 / ((2.0 * hurst).sqrt() * gamma),
        })
    }
    pub fn drift_weight(&self, times: &[f64], i: usize, j: usize) -> f64 {
        if j + 1 == i {
            self.fractional_scale * self.near_loading[j] * (times[j + 1] - times[j])
        } else {
            self.fractional_scale * self.weights[i][j] * (times[j + 1] - times[j])
        }
    }
    pub fn gaussian_at(&self, i: usize, increments: &[f64], near: &[f64]) -> f64 {
        let mut total = pricing_numerics::NeumaierSum::new();
        total.add(near[i - 1]);
        for (w, dw) in self.weights[i].iter().zip(increments) {
            total.add(w * dw);
        }
        total.total()
    }
}
