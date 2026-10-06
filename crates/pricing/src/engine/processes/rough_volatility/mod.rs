//! Explicit, price-only rough model path schemes. New random layouts are
//! isolated from the existing rough Bergomi/HW and stochastic-dividend engines.

mod heston_parameter;
pub use heston_parameter::{HESTON_MC_PARAMETER_NAMES, HestonMcAdjoints, HestonMcRecordedPath};
mod kernel;
pub(in crate::engine) mod lsv;
mod reverse;
mod rfsv;

pub use lsv::{RoughFamilyLsvAdjoints, RoughFamilyLsvPath, RoughFamilyLsvPlan};
pub use reverse::RoughVolatilityRecordedPath;

#[cfg(test)]
mod cache_tests;

use crate::Fingerprint;
use crate::mc::{Philox4x32, RandomCoordinate, RandomDomain};
use crate::models::rough_volatility::invalid;
use crate::models::{
    HullWhiteError, LiftedHeston, MixedRoughBergomi, QuadraticRoughHeston, RoughHeston, RoughSabr,
    RoughVolatilityModel,
};
use kernel::PowerKernel;
use rfsv::RfsvGaussian;

/// Full finite-grid path. Variances are the nonnegative diffusion coefficients.
/// Latent states retain raw Heston variance, quadratic Z, the centered Bergomi
/// Gaussian driver, or RFSV log volatility. No statistical error estimate here
/// includes time discretization, covariance quadrature or truncation bias.
#[derive(Clone, Debug, PartialEq)]
pub struct RoughVolatilityPath {
    pub forwards: Vec<f64>,
    pub variances: Vec<f64>,
    pub latent_states: Vec<f64>,
    pub negative_variance_nodes: u64,
    pub absorbed_forward_steps: u64,
}

/// The variance history has no asset-level or leverage dependence at fixed shocks.
/// Keep it separate from asset evolution so calibration never simulates and
/// discards an unrelated unlevered asset path (which could under/overflow).
#[derive(Clone, Debug)]
pub(in crate::engine) struct VarianceHistory {
    pub(in crate::engine) variances: Vec<f64>,
    latent_states: Vec<f64>,
    negative_variance_nodes: u64,
}

#[derive(Clone, Debug)]
enum CompiledDriver {
    Power(PowerKernel),
    Lift,
    Rfsv(RfsvGaussian),
}

#[derive(Clone, Debug)]
pub struct RoughVolatilityPathPlan {
    model: RoughVolatilityModel,
    times: Box<[f64]>,
    driver: CompiledDriver,
    fingerprint: Fingerprint,
}
impl RoughVolatilityPathPlan {
    /// Compiles an increasing grid starting at zero. Dense fractional-OU
    /// covariance is limited to 256 steps, Volterra convolution to 2048, and
    /// the linear-memory Markovian lift to 65536. Limits precede allocation.
    pub fn compile(model: RoughVolatilityModel, times: Vec<f64>) -> Result<Self, HullWhiteError> {
        if times.len() < 2
            || times[0].to_bits() != 0.0_f64.to_bits()
            || times.len() - 1 > Self::maximum_time_steps(&model)
        {
            return Err(invalid("rough_time_grid_size_or_origin"));
        }
        for i in 1..times.len() {
            if !times[i].is_finite() || times[i] <= times[i - 1] {
                return Err(invalid("rough_time_grid_order"));
            }
        }
        let driver = match &model {
            RoughVolatilityModel::RoughHeston(m) => {
                CompiledDriver::Power(PowerKernel::compile(m.hurst, &times)?)
            }
            RoughVolatilityModel::QuadraticRoughHeston(m) => {
                CompiledDriver::Power(PowerKernel::compile(m.hurst, &times)?)
            }
            RoughVolatilityModel::MixedRoughBergomi(m) => {
                CompiledDriver::Power(PowerKernel::compile(m.hurst, &times)?)
            }
            RoughVolatilityModel::RoughSabr(m) => {
                CompiledDriver::Power(PowerKernel::compile(m.hurst, &times)?)
            }
            RoughVolatilityModel::LiftedHeston(_) => CompiledDriver::Lift,
            RoughVolatilityModel::Rfsv(m) => {
                CompiledDriver::Rfsv(RfsvGaussian::compile(m, &times)?)
            }
        };
        match &model {
            RoughVolatilityModel::MixedRoughBergomi(m) => {
                for &t in &times {
                    m.forward_variance.value(t)?;
                }
            }
            RoughVolatilityModel::RoughSabr(m) => {
                for &t in &times {
                    m.forward_variance.value(t)?;
                }
            }
            _ => {}
        }
        let mut hash = blake3::Hasher::new();
        hash.update(b"rough-volatility-path/v1\0");
        model.fingerprint_into(&mut hash);
        hash.update(&(times.len() as u64).to_le_bytes());
        for &t in &times {
            hash.update(&t.to_le_bytes());
        }
        let fingerprint = Fingerprint::from_bytes(*hash.finalize().as_bytes());
        Ok(Self {
            model,
            times: times.into_boxed_slice(),
            driver,
            fingerprint,
        })
    }
    #[must_use]
    pub const fn maximum_time_steps(model: &RoughVolatilityModel) -> usize {
        match model {
            RoughVolatilityModel::Rfsv(_) => 256,
            RoughVolatilityModel::LiftedHeston(_) => 65_536,
            _ => 2_048,
        }
    }
    #[must_use]
    pub fn model(&self) -> &RoughVolatilityModel {
        &self.model
    }
    #[must_use]
    pub fn time_nodes(&self) -> &[f64] {
        &self.times
    }
    #[must_use]
    pub fn plan_fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }
    /// Factor-major layout: spot Brownian, independent variance Brownian (if
    /// present), then near-cell residuals. RFSV instead reserves n+1 Gaussian
    /// log-volatility coordinates after its n spot-Brownian coordinates.
    #[must_use]
    pub fn random_dimension(&self) -> u32 {
        let n = self.times.len() - 1;
        match self.model {
            RoughVolatilityModel::LiftedHeston(_)
            | RoughVolatilityModel::QuadraticRoughHeston(_) => (2 * n) as u32,
            RoughVolatilityModel::Rfsv(_) => (2 * n + 1) as u32,
            _ => (3 * n) as u32,
        }
    }
    #[must_use]
    pub(crate) fn brownian_block_count(&self) -> usize {
        match self.model {
            RoughVolatilityModel::QuadraticRoughHeston(_) | RoughVolatilityModel::Rfsv(_) => 1,
            _ => 2,
        }
    }
    #[must_use]
    pub const fn scheme(&self) -> &'static str {
        match self.model {
            RoughVolatilityModel::RoughHeston(_) => "rough-heston-hybrid-full-truncation-v1",
            RoughVolatilityModel::LiftedHeston(_) => {
                "lifted-heston-semi-implicit-full-truncation-v1"
            }
            RoughVolatilityModel::QuadraticRoughHeston(_) => "quadratic-rough-heston-hybrid-v1",
            RoughVolatilityModel::MixedRoughBergomi(_) => {
                "mixed-rough-bergomi-hybrid-discrete-centering-v1"
            }
            RoughVolatilityModel::RoughSabr(_) => {
                "rough-sabr-hybrid-lognormal-normal-absorbing-euler-v1"
            }
            RoughVolatilityModel::Rfsv(_) => "rfsv-stationary-fou-covariance-log-euler-v1",
        }
    }
    #[must_use]
    pub fn pseudo_shocks(&self, seed: u64, path: u64, domain: RandomDomain) -> Vec<f64> {
        let rng = Philox4x32::from_seed(seed);
        (0..self.random_dimension())
            .map(|d| rng.standard_normal(RandomCoordinate::new(path, d, domain)))
            .collect()
    }
    pub fn evolve_path(
        &self,
        initial_forward: f64,
        normals: &[f64],
    ) -> Result<RoughVolatilityPath, HullWhiteError> {
        if normals.len() != self.random_dimension() as usize
            || normals.iter().any(|z| !z.is_finite())
        {
            return Err(invalid("rough_normal_shape_or_value"));
        }
        if !initial_forward.is_finite() {
            return Err(invalid("rough_initial_forward"));
        }
        let signed = matches!(&self.model, RoughVolatilityModel::RoughSabr(m) if m.beta == 0.0);
        let absorbing = matches!(&self.model, RoughVolatilityModel::RoughSabr(m) if m.beta > 0.0 && m.beta < 1.0);
        if !signed && (initial_forward < 0.0 || (initial_forward == 0.0 && !absorbing)) {
            return Err(invalid("rough_initial_forward"));
        }
        let history = self.variance_history(normals)?;
        let beta = match &self.model {
            RoughVolatilityModel::RoughSabr(m) => m.beta,
            _ => 1.0,
        };
        self.asset_path(
            initial_forward,
            history.variances,
            history.latent_states,
            normals,
            beta,
            history.negative_variance_nodes,
        )
    }

    pub(in crate::engine) fn variance_history(
        &self,
        normals: &[f64],
    ) -> Result<VarianceHistory, HullWhiteError> {
        if normals.len() != self.random_dimension() as usize
            || normals.iter().any(|z| !z.is_finite())
        {
            return Err(invalid("rough_normal_shape_or_value"));
        }
        match (&self.model, &self.driver) {
            (RoughVolatilityModel::RoughHeston(m), CompiledDriver::Power(k)) => {
                self.heston_variance(m, k, normals)
            }
            (RoughVolatilityModel::LiftedHeston(m), CompiledDriver::Lift) => {
                self.lifted_variance(m, normals)
            }
            (RoughVolatilityModel::QuadraticRoughHeston(m), CompiledDriver::Power(k)) => {
                self.quadratic_variance(m, k, normals)
            }
            (RoughVolatilityModel::MixedRoughBergomi(m), CompiledDriver::Power(k)) => {
                self.mixed_variance(m, k, normals)
            }
            (RoughVolatilityModel::RoughSabr(m), CompiledDriver::Power(k)) => {
                self.sabr_variance(m, k, normals)
            }
            (RoughVolatilityModel::Rfsv(m), CompiledDriver::Rfsv(g)) => {
                let n = self.times.len() - 1;
                let latent_states = g.log_volatilities(m, &normals[n..])?;
                let variances = latent_states
                    .iter()
                    .map(|x| positive_exponential(2.0 * x))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(VarianceHistory {
                    variances,
                    latent_states,
                    negative_variance_nodes: 0,
                })
            }
            _ => Err(invalid("rough_compiled_driver_mismatch")),
        }
    }
    fn innovations(
        &self,
        kernel: &PowerKernel,
        rho: f64,
        normals: &[f64],
        feedback: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.times.len() - 1;
        let independent = (1.0 - rho * rho).max(0.0).sqrt();
        let residual_offset = if feedback { n } else { 2 * n };
        let mut increments = Vec::with_capacity(n);
        let mut near = Vec::with_capacity(n);
        for j in 0..n {
            let dt = self.times[j + 1] - self.times[j];
            let z = if feedback {
                normals[j]
            } else {
                rho * normals[j] + independent * normals[n + j]
            };
            let dw = dt.sqrt() * z;
            increments.push(dw);
            near.push(
                kernel.near_loading[j] * dw
                    + kernel.near_residual[j] * normals[residual_offset + j],
            );
        }
        (increments, near)
    }
    fn heston_variance(
        &self,
        m: &RoughHeston,
        k: &PowerKernel,
        normals: &[f64],
    ) -> Result<VarianceHistory, HullWhiteError> {
        let (dw, near) = self.innovations(k, m.correlation, normals, false);
        let mut variance = vec![m.initial_variance];
        let mut raw = vec![m.initial_variance];
        // Each left-node coefficient is reused by every later history sum.
        // Retain the original multiplication order, including the sqrt rounding.
        // The final node is never a diffusion input, so do not compute its coefficient.
        let mut diffusion = Vec::with_capacity(self.times.len() - 1);
        diffusion.push(k.fractional_scale * m.vol_of_vol * m.initial_variance.sqrt());
        let mut negative = 0;
        for i in 1..self.times.len() {
            let mut sum = pricing_numerics::NeumaierSum::new();
            sum.add(m.initial_variance);
            for j in 0..i {
                sum.add(
                    k.drift_weight(&self.times, i, j)
                        * m.mean_reversion
                        * (m.long_run_variance - variance[j]),
                );
                let innovation = if j + 1 == i {
                    near[j]
                } else {
                    k.weights[i][j] * dw[j]
                };
                sum.add(diffusion[j] * innovation);
            }
            let value = finite_value(sum.total())?;
            negative += u64::from(value < 0.0);
            raw.push(value);
            variance.push(value.max(0.0));
            if i + 1 < self.times.len() {
                diffusion.push(k.fractional_scale * m.vol_of_vol * variance[i].sqrt());
            }
        }
        Ok(VarianceHistory {
            variances: variance,
            latent_states: raw,
            negative_variance_nodes: negative,
        })
    }
    fn lifted_variance(
        &self,
        m: &LiftedHeston,
        normals: &[f64],
    ) -> Result<VarianceHistory, HullWhiteError> {
        let h = &m.heston;
        let n = self.times.len() - 1;
        let mut factors = vec![0.0; m.weights.len()];
        let mut variance = vec![h.initial_variance];
        let mut raw = vec![h.initial_variance];
        let mut negative = 0;
        for j in 0..n {
            let dt = self.times[j + 1] - self.times[j];
            let dw = dt.sqrt()
                * (h.correlation * normals[j]
                    + (1.0 - h.correlation.powi(2)).max(0.0).sqrt() * normals[n + j]);
            let common = h.mean_reversion * (h.long_run_variance - variance[j]) * dt
                + h.vol_of_vol * variance[j].sqrt() * dw;
            let mut sum = pricing_numerics::NeumaierSum::new();
            sum.add(h.initial_variance);
            for ((factor, &rate), &weight) in factors.iter_mut().zip(&m.rates).zip(&m.weights) {
                let denominator = 1.0 + rate * dt;
                if !denominator.is_finite() {
                    return Err(invalid("lifted_time_scale_overflow"));
                }
                *factor = finite_value((*factor + common) / denominator)?;
                sum.add(weight * *factor);
            }
            let value = finite_value(sum.total())?;
            negative += u64::from(value < 0.0);
            raw.push(value);
            variance.push(value.max(0.0));
        }
        Ok(VarianceHistory {
            variances: variance,
            latent_states: raw,
            negative_variance_nodes: negative,
        })
    }
    fn quadratic_variance(
        &self,
        m: &QuadraticRoughHeston,
        k: &PowerKernel,
        normals: &[f64],
    ) -> Result<VarianceHistory, HullWhiteError> {
        let (dw, near) = self.innovations(k, 1.0, normals, true);
        let variance_at = |z: f64| {
            if m.quadratic == 0.0 {
                Ok(m.variance_floor)
            } else {
                finite_nonnegative(m.quadratic * (z - m.shift).powi(2) + m.variance_floor)
            }
        };
        let mut latent = vec![m.initial_state];
        let mut variance = vec![variance_at(m.initial_state)?];
        // Use the same parenthesization as the former inner-loop expression.
        let mut diffusion = Vec::with_capacity(self.times.len() - 1);
        diffusion.push(m.mean_reversion * m.vol_of_vol * variance[0].sqrt() * k.fractional_scale);
        for i in 1..self.times.len() {
            let mut sum = pricing_numerics::NeumaierSum::new();
            sum.add(m.initial_state);
            for j in 0..i {
                sum.add(-m.mean_reversion * latent[j] * k.drift_weight(&self.times, i, j));
                let innovation = if j + 1 == i {
                    near[j]
                } else {
                    k.weights[i][j] * dw[j]
                };
                sum.add(diffusion[j] * innovation);
            }
            let z = finite_value(sum.total())?;
            latent.push(z);
            variance.push(variance_at(z)?);
            if i + 1 < self.times.len() {
                diffusion.push(
                    m.mean_reversion * m.vol_of_vol * variance[i].sqrt() * k.fractional_scale,
                );
            }
        }
        Ok(VarianceHistory {
            variances: variance,
            latent_states: latent,
            negative_variance_nodes: 0,
        })
    }
    fn mixed_variance(
        &self,
        m: &MixedRoughBergomi,
        k: &PowerKernel,
        normals: &[f64],
    ) -> Result<VarianceHistory, HullWhiteError> {
        let (dw, near) = self.innovations(k, m.correlation, normals, false);
        let mut latent = vec![0.0];
        let mut variance = vec![m.forward_variance.value(0.0)?];
        for i in 1..self.times.len() {
            let x = finite_value(k.gaussian_at(i, &dw, &near))?;
            let xi = m.forward_variance.value(self.times[i])?;
            let mut sum = pricing_numerics::NeumaierSum::new();
            if xi != 0.0 {
                for (&weight, &eta) in m.weights.iter().zip(&m.vol_of_vols) {
                    if weight != 0.0 {
                        let log_component = finite_value(
                            xi.ln() + weight.ln() + eta * x - 0.5 * eta.powi(2) * k.variances[i],
                        )?;
                        // An individual mixture component may round to zero;
                        // losing every positive component is an explicit error.
                        sum.add(log_component.exp());
                    }
                }
            }
            latent.push(x);
            let value = finite_nonnegative(sum.total())?;
            if xi > 0.0 && value == 0.0 {
                return Err(invalid("rough_variance_underflow"));
            }
            variance.push(value);
        }
        Ok(VarianceHistory {
            variances: variance,
            latent_states: latent,
            negative_variance_nodes: 0,
        })
    }
    fn sabr_variance(
        &self,
        m: &RoughSabr,
        k: &PowerKernel,
        normals: &[f64],
    ) -> Result<VarianceHistory, HullWhiteError> {
        let (dw, near) = self.innovations(k, m.correlation, normals, false);
        let mut latent = vec![0.0];
        let mut variance = vec![m.forward_variance.value(0.0)?];
        for i in 1..self.times.len() {
            let x = finite_value(k.gaussian_at(i, &dw, &near))?;
            let xi = m.forward_variance.value(self.times[i])?;
            let value = if xi == 0.0 {
                0.0
            } else {
                positive_exponential(
                    xi.ln() + m.vol_of_vol * x - 0.5 * m.vol_of_vol.powi(2) * k.variances[i],
                )?
            };
            latent.push(x);
            variance.push(finite_nonnegative(value)?);
        }
        Ok(VarianceHistory {
            variances: variance,
            latent_states: latent,
            negative_variance_nodes: 0,
        })
    }
    fn asset_path(
        &self,
        initial: f64,
        variance: Vec<f64>,
        latent: Vec<f64>,
        normals: &[f64],
        beta: f64,
        negative: u64,
    ) -> Result<RoughVolatilityPath, HullWhiteError> {
        let mut forwards = vec![initial];
        let mut absorbed = 0;
        for j in 0..self.times.len() - 1 {
            let dt = self.times[j + 1] - self.times[j];
            let previous = forwards[j];
            let v = finite_nonnegative(variance[j])?;
            let noise = v.sqrt() * dt.sqrt() * normals[j];
            let next = if beta == 1.0 {
                let next = previous * (-0.5 * v * dt + noise).exp();
                if next <= 0.0 {
                    return Err(invalid("rough_asset_underflow"));
                }
                next
            } else if beta == 0.0 {
                previous + noise
            } else if previous == 0.0 {
                0.0
            } else {
                let proposal = finite_value(previous + previous.powf(beta) * noise)?;
                if proposal <= 0.0 {
                    absorbed += 1;
                }
                proposal.max(0.0)
            };
            forwards.push(finite_value(next)?);
        }
        Ok(RoughVolatilityPath {
            forwards,
            variances: variance,
            latent_states: latent,
            negative_variance_nodes: negative,
            absorbed_forward_steps: absorbed,
        })
    }
}
fn finite_value(value: f64) -> Result<f64, HullWhiteError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("rough_path_overflow"))
    }
}
fn finite_nonnegative(value: f64) -> Result<f64, HullWhiteError> {
    finite_value(value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(invalid("rough_negative_diffusion_variance"))
    }
}

fn positive_exponential(log_value: f64) -> Result<f64, HullWhiteError> {
    let value = finite_value(finite_value(log_value)?.exp())?;
    if value == 0.0 {
        Err(invalid("rough_variance_underflow"))
    } else {
        Ok(value)
    }
}
