//! Static price, recording and model adapters. Reverse capabilities are optional.
use super::*;
use crate::mc::lsv::LsvPathAdjoints;

/// A calibrated LSV model the shared pricing core can evaluate.
pub(super) trait CalibratedModel: Clone + std::fmt::Debug + Send + Sync {
    type Plan: PathModel;
    const SCHEME: &'static str;
    /// Independent Gaussian blocks per time step in the pricing layout.
    const RANDOM_BLOCKS: usize;
    fn surface(&self) -> &LsvLeverageSurface;
    fn config(&self) -> &LsvParticleConfig;
    fn target(&self) -> &LocalVarianceGrid;
    fn pricing_plan(&self, grid: &LocalVolTimeGrid) -> Result<Self::Plan, LsvError>;
}

/// The path plan of a calibrated LSV model.
pub(super) trait PathModel: Clone + std::fmt::Debug + Send + Sync {
    fn times(&self) -> &[f64];
    fn pseudo_shocks(
        &self,
        seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, LsvError>;
    fn evolve_states(
        &self,
        initial_f: f64,
        shocks: &[f64],
        states: &mut Vec<f64>,
    ) -> Result<(), LsvError>;
}

/// Recording and path reverse are optional for a price-only path model.
pub(super) trait LeveragePathModel: PathModel {
    type Path: PathReverse<Error = LsvError>;
    fn evolve_path(&self, initial_f: f64, shocks: &[f64]) -> Result<Self::Path, LsvError>;
    fn path_states(path: &Self::Path) -> &[f64];
    fn leverage_adjoints(adjoints: <Self::Path as PathReverse>::Adjoints) -> Box<[f64]>;
}

impl<F: BergomiDynamics> CalibratedModel for CalibratedBergomiLsv<F> {
    type Plan = BergomiLsvPlan<F>;
    const SCHEME: &'static str = if F::FACTOR_COUNT == 1 {
        BERGOMI_LSV_SCHEME
    } else {
        BERGOMI_TWO_FACTOR_LSV_SCHEME
    };
    const RANDOM_BLOCKS: usize = 1 + F::FACTOR_COUNT;
    fn surface(&self) -> &LsvLeverageSurface {
        self.surface()
    }
    fn config(&self) -> &LsvParticleConfig {
        self.config()
    }
    fn target(&self) -> &LocalVarianceGrid {
        self.target()
    }
    fn pricing_plan(&self, grid: &LocalVolTimeGrid) -> Result<BergomiLsvPlan<F>, LsvError> {
        self.pricing_plan(grid)
    }
}

impl<F: BergomiDynamics> PathModel for BergomiLsvPlan<F> {
    fn times(&self) -> &[f64] {
        self.times()
    }
    fn pseudo_shocks(
        &self,
        seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, LsvError> {
        self.pseudo_shocks(seed, path, domain)
    }
    fn evolve_states(
        &self,
        initial_f: f64,
        shocks: &[f64],
        states: &mut Vec<f64>,
    ) -> Result<(), LsvError> {
        self.evolve_states(initial_f, shocks, states)
    }
}

impl<F: BergomiDynamics> LeveragePathModel for BergomiLsvPlan<F> {
    type Path = BergomiLsvPath<F>;
    fn evolve_path(&self, initial_f: f64, shocks: &[f64]) -> Result<BergomiLsvPath<F>, LsvError> {
        self.evolve_path(initial_f, shocks)
    }
    fn path_states(path: &BergomiLsvPath<F>) -> &[f64] {
        path.states()
    }
    fn leverage_adjoints(adjoints: LsvPathAdjoints<F::State>) -> Box<[f64]> {
        adjoints.squared_leverage
    }
}

impl CalibratedModel for CalibratedRoughBergomiLsv {
    type Plan = RoughBergomiLsvPlan;
    const SCHEME: &'static str = ROUGH_BERGOMI_LSV_SCHEME;
    const RANDOM_BLOCKS: usize = ROUGH_RANDOM_BLOCKS;
    fn surface(&self) -> &LsvLeverageSurface {
        self.surface()
    }
    fn config(&self) -> &LsvParticleConfig {
        self.config()
    }
    fn target(&self) -> &LocalVarianceGrid {
        self.target()
    }
    fn pricing_plan(&self, grid: &LocalVolTimeGrid) -> Result<RoughBergomiLsvPlan, LsvError> {
        self.pricing_plan(grid)
    }
}

impl PathModel for RoughBergomiLsvPlan {
    fn times(&self) -> &[f64] {
        self.times()
    }
    fn pseudo_shocks(
        &self,
        seed: u64,
        path: u64,
        domain: RandomDomain,
    ) -> Result<Vec<f64>, LsvError> {
        self.pseudo_shocks(seed, path, domain)
    }
    fn evolve_states(
        &self,
        initial_f: f64,
        shocks: &[f64],
        states: &mut Vec<f64>,
    ) -> Result<(), LsvError> {
        self.evolve_states(initial_f, shocks, states)
    }
}

impl LeveragePathModel for RoughBergomiLsvPlan {
    type Path = RoughBergomiLsvPath;
    fn evolve_path(&self, initial_f: f64, shocks: &[f64]) -> Result<RoughBergomiLsvPath, LsvError> {
        self.evolve_path(initial_f, shocks)
    }
    fn path_states(path: &RoughBergomiLsvPath) -> &[f64] {
        path.states()
    }
    fn leverage_adjoints(adjoints: LsvPathAdjoints<()>) -> Box<[f64]> {
        adjoints.squared_leverage
    }
}
