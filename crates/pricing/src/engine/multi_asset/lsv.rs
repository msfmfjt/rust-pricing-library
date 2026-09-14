//! Marginal particle calibration and the exact joint spot/OU Gaussian law.
use super::*;
use crate::market::{CorrelationFactor, LocalVarianceGrid};
use crate::mc::LocalVolTimeGrid;
use crate::mc::lsv::{
    BergomiLsvPlan, CalibratedBergomiLsv, LSV_CALIBRATION_REVERSE, LsvParticleConfig,
    calibrate_bergomi_lsv,
};
use crate::models::Bergomi1Factor;
use crate::multi_asset::MultiAssetError as E;

/// Apply a marginal Bergomi particle calibration to this asset's LV target.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiAssetLsvConfig {
    pub factor: Bergomi1Factor,
    pub particles: LsvParticleConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiAssetLsvRisk {
    pub time_nodes: Vec<f64>,
    pub log_moneyness_nodes: Vec<f64>,
    /// dPV/d(original effective target variance), including particle feedback.
    pub node_adjoints: Vec<f64>,
    /// Across independent RQMC scrambles, conditional on calibration. None for MC.
    pub standard_errors: Option<Vec<f64>>,
    pub method: &'static str,
}

#[derive(Clone, Debug)]
pub(super) struct LsvAsset {
    pub calibration: CalibratedBergomiLsv,
    pub process: BergomiLsvPlan,
    pub target: LocalVarianceGrid,
}
impl LsvAsset {
    pub fn compile(
        target: &LocalVarianceGrid,
        grid: &LocalVolTimeGrid,
        config: MultiAssetLsvConfig,
    ) -> Result<Self, E> {
        let xs = target.log_moneyness_nodes();
        let mut values = Vec::with_capacity(grid.nodes().len() * xs.len());
        for &t in grid.nodes() {
            for &x in xs {
                values.push(target.interpolate(t, x)?.value);
            }
        }
        let refined = LocalVarianceGrid::new(
            grid.nodes().to_vec(),
            xs.to_vec(),
            values,
            target.floor(),
            target.cap(),
        )?;
        // m=f/F starts at one and is a martingale. This fixes log(f/F) under
        // Spot/curve carry changes without introducing drift into calibration.
        let calibration = calibrate_bergomi_lsv(&refined, config.factor, 1.0, config.particles)
            .map_err(E::numerical)?;
        let process = calibration.pricing_plan(grid).map_err(E::numerical)?;
        Ok(Self {
            calibration,
            process,
            target: target.clone(),
        })
    }
    pub fn target_reverse(&self, leverage: &[f64]) -> Result<Vec<f64>, E> {
        let refined = self
            .calibration
            .reverse_leverage(leverage)
            .map_err(E::numerical)?;
        let xs = self.target.log_moneyness_nodes();
        let mut original = vec![0.0; self.target.values().len()];
        for (r, &t) in self.calibration.target().time_nodes().iter().enumerate() {
            for (j, &x) in xs.iter().enumerate() {
                self.target.interpolate(t, x)?.transpose_accumulate(
                    refined[r * xs.len() + j],
                    &mut original,
                    xs.len(),
                );
            }
        }
        Ok(original)
    }
    pub fn risk(
        &self,
        node_adjoints: Vec<f64>,
        standard_errors: Option<Vec<f64>>,
    ) -> MultiAssetLsvRisk {
        MultiAssetLsvRisk {
            time_nodes: self.target.time_nodes().to_vec(),
            log_moneyness_nodes: self.target.log_moneyness_nodes().to_vec(),
            node_adjoints,
            standard_errors,
            method: LSV_CALIBRATION_REVERSE,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct LsvDrivers {
    pub entries: Vec<CorrelationFactor>,
    pub intervals: Vec<CorrelationFactor>,
    /// Standard deviations of OU innovations. Spot entries are one because the
    /// stock log-Euler kernel consumes normalized Brownian increments.
    pub scales: Vec<Vec<f64>>,
    pub asset_indices: Vec<usize>,
}
impl LsvDrivers {
    pub fn compile(
        correlation: &CorrelationTermStructure,
        times: &[f64],
        interval_indices: &[usize],
        configs: &[Option<MultiAssetLsvConfig>],
        supplied: Option<Vec<Vec<Vec<f64>>>>,
    ) -> Result<Option<Self>, E> {
        let n = configs.len();
        let assets: Vec<_> = configs
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_ref().map(|c| (i, c.factor)))
            .collect();
        if assets.is_empty() {
            if supplied.is_some() {
                return Err(E::Invalid(
                    "driver correlations require at least one LSV asset",
                ));
            }
            return Ok(None);
        }
        if supplied
            .as_ref()
            .is_some_and(|s| s.len() != correlation.entries().len())
        {
            return Err(E::Invalid(
                "driver correlations must have one matrix per spot correlation date",
            ));
        }
        let dimension = n
            .checked_add(assets.len())
            .ok_or(E::Invalid("LSV driver dimension overflow"))?;
        let mut entries = Vec::new();
        for (e, (_, spots)) in correlation.entries().iter().enumerate() {
            let r = spots.canonical();
            let matrix = if let Some(supplied) = &supplied {
                supplied[e].clone()
            } else {
                // dV_i = rho_i dW_i + sqrt(1-rho_i^2) dZ_i, with Z independent
                // across assets and of every W. Integrate the OU kernels jointly.
                let mut matrix = vec![vec![0.0; dimension]; dimension];
                for i in 0..n {
                    matrix[i][..n].copy_from_slice(&r[i * n..(i + 1) * n]);
                }
                for (a, &(i, factor)) in assets.iter().enumerate() {
                    for j in 0..n {
                        matrix[j][n + a] = r[j * n + i] * factor.correlation();
                        matrix[n + a][j] = matrix[j][n + a];
                    }
                    for (b, &(j, other)) in assets.iter().enumerate() {
                        matrix[n + a][n + b] = if a == b {
                            1.0
                        } else {
                            factor.correlation() * other.correlation() * r[i * n + j]
                        };
                    }
                }
                matrix
            };
            if matrix.len() != dimension {
                return Err(E::Invalid("wrong spot/volatility driver matrix dimension"));
            }
            let full = CorrelationFactor::compile(matrix, correlation.tolerances())?;
            for i in 0..n {
                for j in 0..n {
                    if full.canonical()[i * dimension + j] != r[i * n + j] {
                        return Err(E::Invalid(
                            "driver spot block must equal canonical spot correlations",
                        ));
                    }
                }
            }
            for (a, &(i, factor)) in assets.iter().enumerate() {
                if full.canonical()[i * dimension + n + a] != factor.correlation() {
                    return Err(E::Invalid(
                        "own spot/volatility driver correlation must equal the calibration factor",
                    ));
                }
            }
            entries.push(full);
        }
        let mut intervals = Vec::new();
        let mut scales = Vec::new();
        for (step, w) in times.windows(2).enumerate() {
            let dt = w[1] - w[0];
            let brownian = entries[interval_indices[step]].canonical();
            let mut joint = vec![vec![0.0; dimension]; dimension];
            let mut scale = vec![1.0; dimension];
            for i in 0..n {
                joint[i][..n].copy_from_slice(&brownian[i * dimension..i * dimension + n]);
            }
            for (a, &(_, factor)) in assets.iter().enumerate() {
                let transition = factor.transition(dt)?;
                scale[n + a] = transition.variance.sqrt();
                let ka = factor.mean_reversion() * dt;
                for j in 0..n {
                    joint[j][n + a] =
                        brownian[j * dimension + n + a] * kernel_correlation(0.0, ka)?;
                    joint[n + a][j] = joint[j][n + a];
                }
                for (b, &(_, other)) in assets.iter().enumerate() {
                    joint[n + a][n + b] = if a == b {
                        1.0
                    } else {
                        brownian[(n + a) * dimension + n + b]
                            * kernel_correlation(ka, other.mean_reversion() * dt)?
                    };
                }
            }
            intervals.push(CorrelationFactor::compile(joint, correlation.tolerances())?);
            scales.push(scale);
        }
        Ok(Some(Self {
            entries,
            intervals,
            scales,
            asset_indices: assets.iter().map(|x| x.0).collect(),
        }))
    }
}

// Normalized integral of two exponential kernels on the unit interval.
fn kernel_correlation(a: f64, b: f64) -> Result<f64, E> {
    if !a.is_finite()
        || !b.is_finite()
        || !(a + b).is_finite()
        || !(2.0 * a).is_finite()
        || !(2.0 * b).is_finite()
    {
        return Err(E::Invalid("unrepresentable OU kernel interval"));
    }
    if a == b {
        return Ok(1.0);
    }
    let phi = |x: f64| if x == 0.0 { 1.0 } else { -(-x).exp_m1() / x };
    let value = phi(a + b) / (phi(2.0 * a).sqrt() * phi(2.0 * b).sqrt());
    if !value.is_finite() || value <= 0.0 || value > 1.0 + 32.0 * f64::EPSILON {
        return Err(E::Invalid("invalid normalized OU kernel covariance"));
    }
    // Only roundoff in the analytic Cauchy--Schwarz bound, never user correlations.
    Ok(value.min(1.0))
}

impl MultiAssetPricingPlan {
    pub fn random_factor_count(&self) -> usize {
        self.assets.len()
            + self
                .lsv_drivers
                .as_ref()
                .map_or(0, |d| d.asset_indices.len())
    }
    pub fn lsv_calibrations(&self) -> Vec<Option<&CalibratedBergomiLsv>> {
        self.assets
            .iter()
            .map(|a| a.lsv.as_ref().map(|l| &l.calibration))
            .collect()
    }
    /// Rows: all spot Brownian drivers, then LSV volatility drivers in asset order.
    /// Entries have the dates of `correlation()`; includes out-of-horizon entries.
    pub fn lsv_driver_correlations(&self) -> &[CorrelationFactor] {
        self.lsv_drivers.as_ref().map_or(&[], |d| &d.entries)
    }
    /// Per-interval covariance of (dW_spots, OU innovations), in driver order.
    pub fn lsv_transition_covariances(&self) -> Vec<Vec<Vec<f64>>> {
        let Some(d) = &self.lsv_drivers else {
            return Vec::new();
        };
        let n = self.assets.len();
        let width = self.random_factor_count();
        d.intervals
            .iter()
            .enumerate()
            .map(|(s, c)| {
                let mut scale = d.scales[s].clone();
                scale[..n].fill((self.times[s + 1] - self.times[s]).sqrt());
                (0..width)
                    .map(|i| {
                        (0..width)
                            .map(|j| c.canonical()[i * width + j] * scale[i] * scale[j])
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
}
