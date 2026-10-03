use crate::models::rough_volatility::invalid;
use crate::models::{HullWhiteError, Rfsv};
use pricing_numerics::{
    CorrelationFactor, CorrelationToleranceConfig, fractional_ou_correlation, gamma_half_to_two,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub(super) struct RfsvGaussian {
    factor: Option<CorrelationFactor>,
    standard_deviation: f64,
}
impl RfsvGaussian {
    pub fn compile(model: &Rfsv, times: &[f64]) -> Result<Self, HullWhiteError> {
        let variance = 0.5
            * model.vol_of_log_vol.powi(2)
            * gamma_half_to_two(2.0 * model.hurst + 1.0).ok_or(invalid("rfsv_gamma"))?
            / model.mean_reversion.powf(2.0 * model.hurst);
        if !variance.is_finite() || variance < 0.0 {
            return Err(invalid("rfsv_stationary_variance"));
        }
        if model.vol_of_log_vol == 0.0 {
            return Ok(Self {
                factor: None,
                standard_deviation: 0.0,
            });
        }
        if variance == 0.0 {
            return Err(invalid("rfsv_variance_underflow"));
        }
        let dimension = times.len();
        let mut matrix = vec![vec![0.0; dimension]; dimension];
        let mut cache = BTreeMap::new();
        for i in 0..dimension {
            matrix[i][i] = 1.0;
            for j in 0..i {
                let lag = model.mean_reversion * (times[i] - times[j]);
                let correlation = *cache
                    .entry(lag.to_bits())
                    .or_insert_with(|| fractional_ou_correlation(model.hurst, lag));
                let correlation = correlation.ok_or(invalid("rfsv_covariance_quadrature"))?;
                matrix[i][j] = correlation;
                matrix[j][i] = correlation;
            }
        }
        let factor = CorrelationFactor::compile(
            matrix,
            CorrelationToleranceConfig {
                symmetry_abs_tol: 0.0,
                diagonal_abs_tol: 0.0,
                psd_abs_tol: 0.0,
                psd_rel_tol: 0.0,
                zero_pivot_abs_tol: 0.0,
                zero_pivot_rel_tol: 0.0,
            },
        )
        .map_err(|_| invalid("rfsv_covariance_not_positive_definite"))?;
        if factor.diagnostics().rank != dimension {
            return Err(invalid("rfsv_covariance_singular"));
        }
        Ok(Self {
            factor: Some(factor),
            standard_deviation: variance.sqrt(),
        })
    }
    pub fn log_volatilities(
        &self,
        model: &Rfsv,
        normals: &[f64],
    ) -> Result<Vec<f64>, HullWhiteError> {
        let Some(factor) = &self.factor else {
            return Ok(vec![model.mean_log_vol; normals.len()]);
        };
        let dimension = factor.dimension();
        if normals.len() != dimension {
            return Err(invalid("rfsv_normal_count"));
        }
        let fixed_initial = model
            .initial_log_vol
            .map(|x| (x - model.mean_log_vol) / self.standard_deviation);
        let mut values = Vec::with_capacity(dimension);
        for i in 0..dimension {
            let mut sum = pricing_numerics::NeumaierSum::new();
            for (j, &z) in normals[..=i].iter().enumerate() {
                let z = if j == 0 {
                    fixed_initial.unwrap_or(z)
                } else {
                    z
                };
                sum.add(factor.lower()[i * dimension + j] * z);
            }
            let value = if i == 0 {
                model
                    .initial_log_vol
                    .unwrap_or(model.mean_log_vol + self.standard_deviation * sum.total())
            } else {
                model.mean_log_vol + self.standard_deviation * sum.total()
            };
            if !value.is_finite() {
                return Err(invalid("rfsv_log_volatility_overflow"));
            }
            values.push(value);
        }
        Ok(values)
    }
}
