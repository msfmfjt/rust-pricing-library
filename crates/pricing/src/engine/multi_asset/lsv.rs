//! Marginal particle calibration and joint spot/OU/rough Gaussian innovations.
use super::lsv_kernels::{LsvCalibration, LsvProcess};
use super::*;
use crate::engine::calibration::capabilities::CalibrationReverse;
use crate::market::{CorrelationFactor, LocalVarianceGrid};
use crate::mc::LocalVolTimeGrid;
use crate::mc::lsv::{CalibratedBergomiLsv, LSV_CALIBRATION_REVERSE, LsvParticleConfig};
use crate::models::{Bergomi1Factor, Bergomi2Factor, RoughBergomi, ou_kernel_correlation};
use crate::multi_asset::MultiAssetError as E;

/// Apply a marginal Bergomi particle calibration to this asset's LV target.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiAssetLsvConfig {
    pub factor: Bergomi1Factor,
    pub particles: LsvParticleConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiAssetLsv2FactorConfig {
    pub factor: Bergomi2Factor,
    pub particles: LsvParticleConfig,
}

/// Rough-LSV through the shared HW adapter (zero rate volatility is allowed).
/// H and eta are fixed during target/Spot/curve risk; eta is log-variance vol-of-vol.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiAssetRoughLsvConfig {
    pub factor: RoughBergomi,
    pub particles: LsvParticleConfig,
}

/// Per-asset choice of one-/two-factor or rough Bergomi LSV.
#[derive(Clone, Debug, PartialEq)]
pub enum MultiAssetBergomiLsvConfig {
    OneFactor(MultiAssetLsvConfig),
    TwoFactor(MultiAssetLsv2FactorConfig),
    Rough(MultiAssetRoughLsvConfig),
}
impl From<MultiAssetLsvConfig> for MultiAssetBergomiLsvConfig {
    fn from(c: MultiAssetLsvConfig) -> Self {
        Self::OneFactor(c)
    }
}
impl From<MultiAssetLsv2FactorConfig> for MultiAssetBergomiLsvConfig {
    fn from(c: MultiAssetLsv2FactorConfig) -> Self {
        Self::TwoFactor(c)
    }
}
impl From<MultiAssetRoughLsvConfig> for MultiAssetBergomiLsvConfig {
    fn from(c: MultiAssetRoughLsvConfig) -> Self {
        Self::Rough(c)
    }
}
impl MultiAssetBergomiLsvConfig {
    /// Number of volatility Brownian drivers; a rough driver also needs one
    /// near-cell Gaussian innovation and its complete history convolution.
    pub fn factor_count(&self) -> usize {
        match self {
            Self::OneFactor(_) | Self::Rough(_) => 1,
            Self::TwoFactor(_) => 2,
        }
    }
    pub(super) fn components(&self) -> Vec<Bergomi1Factor> {
        match self {
            Self::OneFactor(c) => vec![c.factor],
            Self::TwoFactor(c) => c.factor.components().to_vec(),
            // Only the Brownian history increment uses this k=0 carrier.
            // The separate near-cell integral and Volterra convolution supply
            // the rough state; it is never approximated by this OU component.
            Self::Rough(c) => vec![
                Bergomi1Factor::new(0.0, 0.0, c.factor.correlation())
                    .expect("validated rough correlation"),
            ],
        }
    }
    fn factor_correlation(&self) -> f64 {
        match self {
            Self::OneFactor(_) | Self::Rough(_) => 1.0,
            Self::TwoFactor(c) => c.factor.factor_correlation(),
        }
    }
    pub(super) fn rough(&self) -> Option<RoughBergomi> {
        match self {
            Self::Rough(c) => Some(c.factor),
            _ => None,
        }
    }
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
    pub calibration: LsvCalibration,
    pub process: LsvProcess,
    pub target: LocalVarianceGrid,
}
impl LsvAsset {
    pub fn compile(
        target: &LocalVarianceGrid,
        grid: &LocalVolTimeGrid,
        config: MultiAssetBergomiLsvConfig,
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
        let calibration = LsvCalibration::compile(&refined, config).map_err(E::numerical)?;
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
            .calibration_pullback(leverage)
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
    /// HW-only factorization order -> external driver order. The spot prefix is
    /// fixed; the OU/rate suffix is pivoted without dropping Gaussian columns.
    pub permutations: Option<Vec<Vec<usize>>>,
    /// Standard deviations of OU/rate innovations. Spot entries are one for
    /// deterministic-rate kernels, sqrt(dt) for HW kernels consuming dW.
    pub scales: Vec<Vec<f64>>,
    pub asset_indices: Vec<usize>,
    /// Assets whose near-cell integrals follow the two HW coordinates.
    pub rough_asset_indices: Vec<usize>,
}
impl LsvDrivers {
    pub fn compile(
        correlation: &CorrelationTermStructure,
        times: &[f64],
        interval_indices: &[usize],
        configs: &[Option<MultiAssetBergomiLsvConfig>],
        supplied: Option<Vec<Vec<Vec<f64>>>>,
    ) -> Result<Option<Self>, E> {
        let n = configs.len();
        let assets: Vec<_> = configs
            .iter()
            .enumerate()
            .flat_map(|(i, c)| {
                c.as_ref()
                    .map(|c| c.components())
                    .unwrap_or_default()
                    .into_iter()
                    .map(move |f| (i, f))
            })
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
                // dV_i = rho_i dW_i + H_i dZ_i, with Z independent across assets
                // and of every W. H_i H_i^T = C_vol - rho_i rho_i^T retains each
                // asset's specified marginal. Integrate the OU kernels jointly.
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
                        } else if i == j {
                            configs[i].as_ref().expect("LSV").factor_correlation()
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
                for (b, &(j, _)) in assets.iter().enumerate() {
                    if a != b
                        && i == j
                        && full.canonical()[(n + a) * dimension + n + b]
                            != configs[i].as_ref().expect("LSV").factor_correlation()
                    {
                        return Err(E::Invalid(
                            "own volatility/volatility correlation must equal the calibration factor",
                        ));
                    }
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
                    joint[j][n + a] = brownian[j * dimension + n + a]
                        * ou_kernel_correlation(0.0, ka).map_err(E::numerical)?;
                    joint[n + a][j] = joint[j][n + a];
                }
                for (b, &(_, other)) in assets.iter().enumerate() {
                    joint[n + a][n + b] = if a == b {
                        1.0
                    } else {
                        brownian[(n + a) * dimension + n + b]
                            * ou_kernel_correlation(ka, other.mean_reversion() * dt)
                                .map_err(E::numerical)?
                    };
                }
            }
            intervals.push(CorrelationFactor::compile(joint, correlation.tolerances())?);
            scales.push(scale);
        }
        Ok(Some(Self {
            entries,
            intervals,
            permutations: None,
            scales,
            asset_indices: assets.iter().map(|x| x.0).collect(),
            rough_asset_indices: Vec::new(),
        }))
    }
}

impl MultiAssetPricingPlan {
    pub fn random_factor_count(&self) -> usize {
        self.driver_layout.base_factor_count() * (1 + usize::from(self.local_correlation.is_some()))
    }
    /// One-factor calibrations in asset order; None for BS/LV/two-factor assets.
    /// Use `lsv_two_factor_calibrations` for the complementary two-factor objects.
    pub fn lsv_calibrations(&self) -> Vec<Option<&CalibratedBergomiLsv>> {
        self.assets
            .iter()
            .map(|a| {
                a.lsv.as_ref().and_then(|l| match &l.calibration {
                    LsvCalibration::One(c) => Some(c.as_ref()),
                    LsvCalibration::Two(_) => None,
                })
            })
            .collect()
    }
    /// Two-factor calibrations in asset order. The existing `lsv_calibrations`
    /// accessor continues to expose only one-factor calibrations.
    pub fn lsv_two_factor_calibrations(
        &self,
    ) -> Vec<Option<&CalibratedBergomiLsv<Bergomi2Factor>>> {
        self.assets
            .iter()
            .map(|a| {
                a.lsv.as_ref().and_then(|l| match &l.calibration {
                    LsvCalibration::One(_) => None,
                    LsvCalibration::Two(c) => Some(c.as_ref()),
                })
            })
            .collect()
    }
    pub fn lsv_volatility_factor_counts(&self) -> Vec<usize> {
        self.assets
            .iter()
            .map(|a| {
                a.hw.as_ref().map_or_else(
                    || a.lsv.as_ref().map_or(0, |l| l.calibration.factor_count()),
                    |h| h.factor_count(),
                )
            })
            .collect()
    }
    /// Rows: all spot Brownian drivers, then LSV volatility drivers in asset order,
    /// with the shared rate Brownian appended in HW mode.
    /// Entries have the dates of `correlation()`; includes out-of-horizon entries.
    pub fn lsv_driver_correlations(&self) -> &[CorrelationFactor] {
        self.lsv_drivers.as_ref().map_or(&[], |d| &d.entries)
    }
    /// Per-interval covariance of (dW_spots, OU innovations), in driver order.
    /// HW mode uses dV for rough history carriers, then appends the rate OU
    /// state, integrated rate, and near-cell rough integrals in asset order.
    pub fn lsv_transition_covariances(&self) -> Vec<Vec<Vec<f64>>> {
        let Some(d) = &self.lsv_drivers else {
            return Vec::new();
        };
        let n = self.assets.len();
        let width = d.scales[0].len();
        d.intervals
            .iter()
            .enumerate()
            .map(|(s, c)| {
                let mut scale = d.scales[s].clone();
                scale[..n].fill((self.times[s + 1] - self.times[s]).sqrt());
                let order = d.permutations.as_ref().map(|p| &p[s]);
                let mut covariance = vec![vec![0.0; width]; width];
                for i in 0..width {
                    let row = order.map_or(i, |p| p[i]);
                    for j in 0..width {
                        let col = order.map_or(j, |p| p[j]);
                        covariance[row][col] =
                            c.canonical()[i * width + j] * scale[row] * scale[col];
                    }
                }
                covariance
            })
            .collect()
    }
}
