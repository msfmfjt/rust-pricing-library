//! Family adapters to the existing payoff / MC / particle-reverse execution core.
use super::*;
use crate::engine::calibration::lsv::{
    CalibratedRoughFamilyLsv, calibrate_rough_family_lsv_parallel,
};
use crate::engine::processes::rough_volatility::{
    RoughFamilyLsvAdjoints, RoughFamilyLsvPath, RoughFamilyLsvPlan,
};
use crate::models::RoughVolatilityModel;

#[derive(Clone, Debug)]
pub struct RoughFamilyLsvPricingPlan {
    core: LsvPricingCore<CalibratedRoughFamilyLsv>,
}
impl RoughFamilyLsvPricingPlan {
    pub fn compile(
        target_request: &PricingRequest,
        model: RoughVolatilityModel,
        particles: LsvParticleConfig,
        policy: ExecutionPolicy,
    ) -> Result<Self, MonteCarloError> {
        let mut hash = blake3::Hasher::new();
        model.fingerprint_into(&mut hash);
        let mut tag = b"pricing/rough-family-lsv-plan/v1\0".to_vec();
        tag.extend_from_slice(hash.finalize().as_bytes());
        LsvPricingCore::compile(
            target_request,
            particles,
            policy,
            &tag,
            &[],
            |target, initial, particles, executor| {
                calibrate_rough_family_lsv_parallel(target, model, initial, particles, executor)
            },
        )
        .map(|core| Self { core })
    }
    #[must_use]
    pub fn calibration(&self) -> &CalibratedRoughFamilyLsv {
        &self.core.calibration
    }
    #[must_use]
    pub fn plan_fingerprint(&self) -> Fingerprint {
        self.core.fingerprint
    }
    #[must_use]
    pub fn execution_policy(&self) -> ExecutionPolicy {
        self.core.policy
    }
    pub fn evaluate(&self) -> Result<LsvPrice, MonteCarloError> {
        self.core.evaluate()
    }
    pub fn evaluate_local_variance_risk(&self) -> Result<LsvLocalVarianceRisk, MonteCarloError> {
        self.core.evaluate_local_variance_risk()
    }
}
impl CalibratedModel for CalibratedRoughFamilyLsv {
    type Plan = RoughFamilyLsvPlan;
    const SCHEME: &'static str = "rough-family-fixed-driver-lsv-log-euler-v1";
    const RANDOM_BLOCKS: usize = 3;
    fn random_dimension(&self, n: usize) -> usize {
        match self.model() {
            RoughVolatilityModel::Rfsv(_) => 2 * n + 1,
            RoughVolatilityModel::LiftedHeston(_)
            | RoughVolatilityModel::QuadraticRoughHeston(_) => 2 * n,
            _ => 3 * n,
        }
    }
    fn brownian_block_count(&self) -> usize {
        match self.model() {
            RoughVolatilityModel::Rfsv(_) | RoughVolatilityModel::QuadraticRoughHeston(_) => 1,
            _ => 2,
        }
    }
    fn surface(&self) -> &LsvLeverageSurface {
        self.surface()
    }
    fn config(&self) -> &LsvParticleConfig {
        self.config()
    }
    fn target(&self) -> &LocalVarianceGrid {
        self.target()
    }
    fn pricing_plan(&self, grid: &LocalVolTimeGrid) -> Result<Self::Plan, LsvError> {
        self.pricing_plan(grid)
    }
}
impl PathModel for RoughFamilyLsvPlan {
    fn times(&self) -> &[f64] {
        self.times()
    }
    fn pseudo_shocks(
        &self,
        seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, LsvError> {
        Ok(self.pseudo_shocks(seed, path, domain))
    }
    fn evolve_states(
        &self,
        initial: f64,
        shocks: &[f64],
        out: &mut Vec<f64>,
    ) -> Result<(), LsvError> {
        self.evolve_states(initial, shocks, out)
    }
}
impl LeveragePathModel for RoughFamilyLsvPlan {
    type Path = RoughFamilyLsvPath;
    fn evolve_path(&self, initial: f64, shocks: &[f64]) -> Result<Self::Path, LsvError> {
        self.evolve_path(initial, shocks)
    }
    fn path_states(path: &Self::Path) -> &[f64] {
        path.states()
    }
    fn leverage_adjoints(a: RoughFamilyLsvAdjoints) -> Box<[f64]> {
        a.squared_leverage
    }
}

mod spot_delta;
pub use spot_delta::{RoughFamilyLsvDelta, RoughFamilyLsvDeltaConvention};

mod market_iv;
pub use market_iv::{RoughFamilyLsvMarketIvRisk, RoughFamilyLsvMarketIvRiskPlan};

mod gamma;
pub use gamma::RoughFamilyLsvGamma;

mod heston_parameter;
pub use heston_parameter::HestonLsvParameterRisk;

#[cfg(test)]
mod heston_parameter_tests;

mod quadratic_parameter;
pub use quadratic_parameter::QuadraticHestonLsvParameterRisk;
mod mixed_parameter;
pub use mixed_parameter::MixedBergomiLsvParameterRisk;

#[cfg(test)]
mod mixed_parameter_tests;

#[cfg(test)]
mod quadratic_parameter_tests;
