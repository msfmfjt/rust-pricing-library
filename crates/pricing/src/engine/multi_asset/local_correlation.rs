//! Particle projection onto one positive basket of normalized equity martingales.
//! R(t,m)=(1-lambda(t,log B)) R0(t)+lambda(t,log B) R1(t) preserves PSD.
mod calibration;
mod integration;
mod joint;
mod joint_calibration;
mod joint_integration;
mod joint_reverse;
mod reverse;
#[cfg(test)]
mod stress_diagnostics;
use super::*;
use crate::market::{CorrelationFactor, LocalVarianceGrid};
use crate::mc::lsv::LsvParticleConfig;
use crate::mc::{Philox4x32, RandomCoordinate, RandomDomain};
use crate::multi_asset::MultiAssetError as E;

/// Joint endpoint and stochastic-rate basket inputs. The full Brownian order is
/// spots, each asset's volatility factors, then the shared rate Brownian.
#[derive(Clone, Debug, Default)]
pub struct LocalCorrelationExtensions {
    pub second_driver_correlations: Option<Vec<Vec<Vec<f64>>>>,
    /// Paired variance / T-forward log-density target for the normalized basket.
    pub hull_white_target: Option<crate::mc::hull_white::HullWhiteLsvTarget>,
}

/// Treatment of basket targets outside the chosen correlation family's range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalCorrelationFeasibility {
    Reject,
    /// Project onto `[0,1]` and expose the unprojected coefficient and variance residual.
    ProjectAndReport,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalCorrelationConfig {
    /// Positive weights summing to one; B=sum_i weight_i*f_i/F_i(t).
    pub basket_weights: Vec<f64>,
    /// Effective relative basket local variance, on (time, log B).
    pub target: LocalVarianceGrid,
    /// Second PSD endpoint; dates, asset order and tolerances must match the first.
    pub second_correlation: CorrelationTermStructure,
    pub particles: LsvParticleConfig,
    pub feasibility: LocalCorrelationFeasibility,
    /// A smaller conditional variance span is treated as unidentifiable.
    pub minimum_variance_span: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalCorrelationNodeDiagnostics {
    pub endpoint_variances: [f64; 2],
    pub target_variance: f64,
    pub attained_variance: f64,
    /// Stochastic-rate correction included in attained_variance.
    pub rate_correction: f64,
    pub raw_mixing: f64,
    pub effective_samples: f64,
    pub source_node: usize,
    pub fallback: bool,
    pub projected: bool,
    pub unidentifiable: bool,
}

/// Joint calibrated sensitivities to basket and constituent volatility inputs.
/// Axes, weights, correlation endpoints and numerical calibration settings are fixed.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalCorrelationRisk {
    pub basket_time_nodes: Vec<f64>,
    pub basket_log_nodes: Vec<f64>,
    pub basket_variance_adjoints: Vec<f64>,
    pub basket_standard_errors: Option<Vec<f64>>,
    /// BS: one derivative per unit sigma. LV: row-major effective variance nodes.
    pub asset_adjoints: Vec<Vec<f64>>,
    pub asset_standard_errors: Option<Vec<Vec<f64>>>,
    pub asset_time_nodes: Vec<Vec<f64>>,
    pub asset_log_nodes: Vec<Vec<f64>>,
    pub basket_hull_white: Option<MultiAssetHullWhiteLsvRisk>,
    pub asset_hull_white: Vec<Option<MultiAssetHullWhiteLsvRisk>>,
    pub method: &'static str,
}

#[derive(Clone, Debug)]
pub struct LocalCorrelationCalibration {
    joint: Option<joint::Joint>,
    config: LocalCorrelationConfig,
    times: Vec<f64>,
    models: Vec<ModelSpec>,
    endpoints: Vec<[CorrelationFactor; 2]>,
    entries: Vec<usize>,
    mixing: Vec<f64>,
    diagnostics: Vec<LocalCorrelationNodeDiagnostics>,
    weight_sums: Vec<f64>,
    particle_means: Vec<Vec<f64>>,
    trace: Option<std::sync::Arc<Vec<Vec<Vec<f64>>>>>,
}

#[derive(Clone, Debug)]
pub(super) struct LocalCorrelationPath {
    /// Time-major normalized equity states.
    states: Vec<Vec<f64>>,
    independent: Vec<Vec<f64>>,
    pub boundary_counts: Vec<f64>,
}

#[derive(Clone, Copy)]
struct Lookup {
    value: f64,
    derivative: f64,
    left: usize,
    weight: f64,
}
impl Lookup {
    fn transpose(self, seed: f64, output: &mut [f64]) {
        output[self.left] += seed * (1.0 - self.weight);
        output[self.left + 1] += seed * self.weight;
    }
}

impl LocalCorrelationCalibration {
    pub fn config(&self) -> &LocalCorrelationConfig {
        &self.config
    }
    pub fn time_nodes(&self) -> &[f64] {
        &self.times
    }
    pub fn log_nodes(&self) -> &[f64] {
        self.config.target.log_moneyness_nodes()
    }
    pub fn mixing_coefficients(&self) -> &[f64] {
        &self.mixing
    }
    pub fn diagnostics(&self) -> &[LocalCorrelationNodeDiagnostics] {
        &self.diagnostics
    }
    pub fn particle_means(&self) -> &[Vec<f64>] {
        &self.particle_means
    }
    pub fn retains_reverse_trace(&self) -> bool {
        self.trace.is_some()
    }

    /// Piecewise constant in time, linear in log B with flat spatial wings.
    pub fn correlation_at(&self, time: f64, log_basket: f64) -> Result<Vec<Vec<f64>>, E> {
        if !time.is_finite()
            || time < 0.0
            || time > *self.times.last().unwrap()
            || !log_basket.is_finite()
        {
            return Err(E::Invalid(
                "local correlation lookup outside finite time coverage",
            ));
        }
        let row = self.times.partition_point(|t| *t <= time).saturating_sub(1);
        let lambda = self.lookup(row, log_basket).value;
        let endpoints = &self.endpoints[self.entries[row]];
        let n = self.models.len();
        Ok((0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        (1.0 - lambda) * endpoints[0].canonical()[i * n + j]
                            + lambda * endpoints[1].canonical()[i * n + j]
                    })
                    .collect()
            })
            .collect())
    }
    fn lookup(&self, row: usize, x: f64) -> Lookup {
        let xs = self.log_nodes();
        let n = xs.len();
        let j = xs.partition_point(|v| *v <= x).saturating_sub(1).min(n - 2);
        let weight = ((x - xs[j]) / (xs[j + 1] - xs[j])).clamp(0.0, 1.0);
        let left = row * n + j;
        Lookup {
            value: (1.0 - weight) * self.mixing[left] + weight * self.mixing[left + 1],
            derivative: if x <= xs[0] || x >= xs[n - 1] {
                0.0
            } else {
                (self.mixing[left + 1] - self.mixing[left]) / (xs[j + 1] - xs[j])
            },
            left,
            weight,
        }
    }
    fn basket(&self, states: &[f64]) -> f64 {
        self.config
            .basket_weights
            .iter()
            .zip(states)
            .map(|(w, m)| w * m)
            .sum()
    }
    fn sigma(&self, asset: usize, row: usize, m: f64) -> Result<f64, E> {
        match &self.models[asset] {
            ModelSpec::BlackScholes(model) => Ok(model.volatility().get()),
            ModelSpec::LocalVolatility(model) => Ok(model
                .local_variance_grid()
                .interpolate(self.times[row], m.ln())?
                .value
                .sqrt()),
            _ => unreachable!("validated local correlation model"),
        }
    }
    fn parameter_counts(&self) -> Vec<usize> {
        if let Some(j) = &self.joint {
            return j.parameter_counts();
        }
        self.models
            .iter()
            .map(|m| match m {
                ModelSpec::LocalVolatility(m) => m.local_variance_grid().values().len(),
                _ => 1,
            })
            .collect()
    }
    fn empty_asset_adjoints(&self) -> Vec<Vec<f64>> {
        self.parameter_counts()
            .into_iter()
            .map(|n| vec![0.0; n])
            .collect()
    }
    fn endpoint_normals(&self, row: usize, independent: &[f64]) -> Vec<[f64; 2]> {
        let n = self.models.len();
        let ends = &self.endpoints[self.entries[row]];
        (0..n)
            .map(|i| {
                std::array::from_fn(|e| {
                    (0..=i)
                        .map(|j| ends[e].lower()[i * n + j] * independent[e * n + j])
                        .sum()
                })
            })
            .collect()
    }
    fn step(&self, row: usize, states: &[f64], independent: &[f64]) -> Result<Vec<f64>, E> {
        let lambda = self.lookup(row, self.basket(states).ln()).value;
        let normals = self.endpoint_normals(row, independent);
        let dt = self.times[row + 1] - self.times[row];
        states
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let sigma = self.sigma(i, row, *m)?;
                let z = (1.0 - lambda).sqrt() * normals[i][0] + lambda.sqrt() * normals[i][1];
                let next = m * (-0.5 * sigma * sigma * dt + sigma * dt.sqrt() * z).exp();
                if !next.is_finite() || next <= 0.0 {
                    return Err(E::Invalid(
                        "nonpositive/nonfinite local correlation particle",
                    ));
                }
                Ok(next)
            })
            .collect()
    }
    fn calibration_normals(&self, particle: usize, row: usize) -> Result<Vec<f64>, E> {
        let rng = Philox4x32::from_seed(self.config.particles.seed());
        (0..self
            .joint
            .as_ref()
            .map_or(2 * self.models.len(), |j| 2 * j.width))
            .map(|j| {
                let dimension = j
                    .checked_mul(self.times.len() - 1)
                    .and_then(|v| v.checked_add(row))
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or(E::Invalid("local correlation dimension overflow"))?;
                Ok(rng.standard_normal(RandomCoordinate::new(
                    particle as u64,
                    dimension,
                    RandomDomain::LsvCalibration,
                )))
            })
            .collect()
    }
    fn endpoint_variances(&self, row: usize, states: &[f64]) -> Result<[f64; 2], E> {
        let b = self.basket(states);
        let a = states
            .iter()
            .enumerate()
            .map(|(i, m)| Ok(self.config.basket_weights[i] * m * self.sigma(i, row, *m)? / b))
            .collect::<Result<Vec<_>, E>>()?;
        let n = a.len();
        let ends = &self.endpoints[self.entries[row]];
        Ok(std::array::from_fn(|e| {
            let mut sum = 0.0;
            for i in 0..n {
                for j in 0..n {
                    sum += a[i] * ends[e].canonical()[i * n + j] * a[j];
                }
            }
            sum
        }))
    }
    fn evolve(&self, independent: &[Vec<f64>]) -> Result<LocalCorrelationPath, E> {
        if self.joint.is_some() {
            return self.evolve_joint(independent);
        }
        let n = self.models.len();
        let steps = self.times.len() - 1;
        if independent.len() != 2 * n
            || independent
                .iter()
                .any(|v| v.len() != steps || v.iter().any(|x| !x.is_finite()))
        {
            return Err(E::Invalid(
                "local correlation independent shock dimensions/values",
            ));
        }
        let mut states: Vec<Vec<f64>> = vec![vec![1.0; n]];
        let mut boundary_counts = vec![0.0; n];
        for row in 0..steps {
            for (i, m) in states[row].iter().enumerate() {
                if let ModelSpec::LocalVolatility(lv) = &self.models[i] {
                    let xs = lv.local_variance_grid().log_moneyness_nodes();
                    if m.ln() < xs[0] || m.ln() > xs[xs.len() - 1] {
                        boundary_counts[i] += 1.0;
                    }
                }
            }
            let z: Vec<_> = independent.iter().map(|v| v[row]).collect();
            states.push(self.step(row, &states[row], &z)?);
        }
        Ok(LocalCorrelationPath {
            states,
            independent: independent.to_vec(),
            boundary_counts,
        })
    }
}
fn quartic(u: f64) -> f64 {
    if u.abs() < 1.0 {
        (1.0 - u * u).powi(2)
    } else {
        0.0
    }
}
fn quartic_derivative(u: f64) -> f64 {
    if u.abs() < 1.0 {
        -4.0 * u * (1.0 - u * u)
    } else {
        0.0
    }
}
