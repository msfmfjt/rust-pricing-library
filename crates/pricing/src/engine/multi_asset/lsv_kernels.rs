//! Type dispatch at the asset boundary; numerical kernels remain shared.
use super::lsv::MultiAssetBergomiLsvConfig;
use crate::engine::calibration::capabilities::CalibrationReverse;
use crate::engine::processes::capabilities::PathReverse;
use crate::market::LocalVarianceGrid;
use crate::mc::LocalVolTimeGrid;
use crate::mc::lsv::{
    BergomiLsvPath, BergomiLsvPlan, CalibratedBergomiLsv, LsvError, LsvLeverageSurface,
    LsvParticleConfig, calibrate_bergomi_lsv,
};
use crate::models::{Bergomi2Factor, BergomiDynamics};

#[derive(Clone, Debug)]
pub(super) enum LsvCalibration {
    One(Box<CalibratedBergomiLsv>),
    Two(Box<CalibratedBergomiLsv<Bergomi2Factor>>),
}
#[derive(Clone, Debug)]
pub(super) enum LsvProcess {
    One(BergomiLsvPlan),
    Two(BergomiLsvPlan<Bergomi2Factor>),
}
pub(super) enum LsvPath {
    One(BergomiLsvPath),
    Two(BergomiLsvPath<Bergomi2Factor>),
}
impl LsvCalibration {
    pub fn compile(
        target: &LocalVarianceGrid,
        config: MultiAssetBergomiLsvConfig,
    ) -> Result<Self, LsvError> {
        Ok(match config {
            MultiAssetBergomiLsvConfig::OneFactor(c) => Self::One(Box::new(calibrate_bergomi_lsv(
                target,
                c.factor,
                1.0,
                c.particles,
            )?)),
            MultiAssetBergomiLsvConfig::TwoFactor(c) => Self::Two(Box::new(calibrate_bergomi_lsv(
                target,
                c.factor,
                1.0,
                c.particles,
            )?)),
            MultiAssetBergomiLsvConfig::Rough(_) => {
                return Err(LsvError::InvalidInput {
                    field: "rough_lsv_requires_paired_hw_target",
                    index: 0,
                });
            }
        })
    }
    pub fn surface(&self) -> &LsvLeverageSurface {
        match self {
            Self::One(c) => c.surface(),
            Self::Two(c) => c.surface(),
        }
    }
    pub fn config(&self) -> &LsvParticleConfig {
        match self {
            Self::One(c) => c.config(),
            Self::Two(c) => c.config(),
        }
    }
    pub fn target(&self) -> &LocalVarianceGrid {
        match self {
            Self::One(c) => c.target(),
            Self::Two(c) => c.target(),
        }
    }
    pub fn parameters(&self) -> Vec<f64> {
        match self {
            Self::One(c) => c.factor().parameters(),
            Self::Two(c) => c.factor().parameters(),
        }
    }
    pub fn factor_count(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Two(_) => 2,
        }
    }
    pub fn pricing_plan(&self, grid: &LocalVolTimeGrid) -> Result<LsvProcess, LsvError> {
        match self {
            Self::One(c) => Ok(LsvProcess::One(c.pricing_plan(grid)?)),
            Self::Two(c) => Ok(LsvProcess::Two(c.pricing_plan(grid)?)),
        }
    }
}
impl CalibrationReverse for LsvCalibration {
    type Adjoints = Vec<f64>;
    type Error = LsvError;
    fn validate_calibration_reverse(&self) -> Result<(), LsvError> {
        match self {
            Self::One(c) => c.validate_calibration_reverse(),
            Self::Two(c) => c.validate_calibration_reverse(),
        }
    }
    fn calibration_pullback(&self, seeds: &[f64]) -> Result<Vec<f64>, LsvError> {
        match self {
            Self::One(c) => c.calibration_pullback(seeds),
            Self::Two(c) => c.calibration_pullback(seeds),
        }
    }
}
impl LsvProcess {
    pub fn evolve(&self, spot: &[f64], innovations: &[Vec<f64>]) -> Result<LsvPath, LsvError> {
        match self {
            Self::One(p) => Ok(LsvPath::One(p.evolve_with_ou_innovations(
                1.0,
                spot,
                &innovations[0],
            )?)),
            Self::Two(p) => {
                let v: Vec<_> = innovations.iter().flatten().copied().collect();
                Ok(LsvPath::Two(p.evolve_with_ou_innovations(1.0, spot, &v)?))
            }
        }
    }
}
impl LsvPath {
    pub fn states(&self) -> &[f64] {
        match self {
            Self::One(p) => p.states(),
            Self::Two(p) => p.states(),
        }
    }
    pub fn reverse_leverage(&self, seeds: &[f64]) -> Result<Box<[f64]>, LsvError> {
        match self {
            Self::One(p) => Ok(p.path_pullback(seeds)?.squared_leverage),
            Self::Two(p) => Ok(p.path_pullback(seeds)?.squared_leverage),
        }
    }
}
