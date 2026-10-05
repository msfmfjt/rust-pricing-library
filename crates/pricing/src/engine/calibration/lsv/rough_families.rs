//! Connect the six fixed rough variance histories to the existing quartic
//! particle estimator and its discrete target-variance reverse.
use super::*;
use crate::engine::processes::rough_volatility::lsv::validate_model;
use crate::engine::processes::rough_volatility::{RoughFamilyLsvPlan, RoughVolatilityPathPlan};
use crate::models::RoughVolatilityModel;

#[derive(Clone, Debug)]
pub struct CalibratedRoughFamilyLsv {
    model: RoughVolatilityModel,
    core: Calibration,
}
struct FamilyTable {
    values: Vec<f64>,
    rows: usize,
    initial: [f64; 3],
    blocks: usize,
}
impl ParticleDriver for FamilyTable {
    type State = ();
    type Step = ();
    fn random_blocks(&self) -> usize {
        self.blocks
    }
    fn trivial(&self) -> bool {
        false
    }
    fn initial_moments(&self) -> [f64; 3] {
        self.initial
    }
    fn allows_zero_multiplier(&self) -> bool {
        true
    }
    fn multiplier(&self, _: (), particle: usize, row: usize) -> f64 {
        self.values[particle * self.rows + row]
    }
    fn step(&self, _: f64) -> Result<(), LsvError> {
        Ok(())
    }
    fn advance(&self, _: (), _: (), _: usize, _: usize, _: f64) -> Result<(), LsvError> {
        Ok(())
    }
}

pub fn calibrate_rough_family_lsv(
    target: &LocalVarianceGrid,
    model: RoughVolatilityModel,
    initial_forward: f64,
    config: LsvParticleConfig,
) -> Result<CalibratedRoughFamilyLsv, LsvError> {
    calibrate_family(target, model, initial_forward, config, None)
}
pub fn calibrate_rough_family_lsv_parallel(
    target: &LocalVarianceGrid,
    model: RoughVolatilityModel,
    initial_forward: f64,
    config: LsvParticleConfig,
    executor: &DeterministicExecutor,
) -> Result<CalibratedRoughFamilyLsv, LsvError> {
    calibrate_family(target, model, initial_forward, config, Some(executor))
}
fn resource_error() -> LsvError {
    LsvError::InvalidInput {
        field: "rough_lsv_calibration_resource_limit",
        index: 0,
    }
}
fn calibrate_family(
    target: &LocalVarianceGrid,
    model: RoughVolatilityModel,
    initial_forward: f64,
    config: LsvParticleConfig,
    executor: Option<&DeterministicExecutor>,
) -> Result<CalibratedRoughFamilyLsv, LsvError> {
    validate_model(&model)?;
    valid(initial_forward, "initial_forward", 0, true)?;
    let nt = target.time_nodes().len();
    let np = config.particle_count();
    // Bound the table, retained state trace, target rows and history work BEFORE allocation.
    let cells = np.checked_mul(nt).ok_or_else(resource_error)?;
    let nodes = nt
        .checked_mul(target.log_moneyness_nodes().len())
        .ok_or_else(resource_error)?;
    let per_path = match &model {
        RoughVolatilityModel::LiftedHeston(m) => nt.checked_mul(m.weights.len()),
        _ => nt.checked_mul(nt),
    }
    .ok_or_else(resource_error)?;
    let work = np.checked_mul(per_path).ok_or_else(resource_error)?;
    let regression_work = cells
        .checked_mul(target.log_moneyness_nodes().len())
        .ok_or_else(resource_error)?;
    if cells > 8_000_000
        || nodes > 1_000_000
        || work > 2_000_000_000
        || regression_work > 2_000_000_000
    {
        return Err(resource_error());
    }
    let plan = RoughVolatilityPathPlan::compile(model.clone(), target.time_nodes().to_vec())?;
    let mut values = vec![0.0; cells];
    // One bounded chunk per worker task; deterministic error order is retained.
    let fill = |(chunk, rows): (usize, &mut [f64])| {
        for (local, out) in rows.chunks_mut(nt).enumerate() {
            let particle = chunk * PARTICLE_CHUNK + local;
            let shocks =
                plan.pseudo_shocks(config.seed(), particle as u64, RandomDomain::LsvCalibration);
            let history = plan.variance_history(&shocks)?;
            for (a, v) in out.iter_mut().zip(history.variances) {
                *a = v.sqrt();
            }
        }
        Ok::<(), LsvError>(())
    };
    let results = match executor {
        None => values
            .chunks_mut(PARTICLE_CHUNK * nt)
            .enumerate()
            .map(fill)
            .collect::<Vec<_>>(),
        Some(e) => e.install(|| {
            values
                .par_chunks_mut(PARTICLE_CHUNK * nt)
                .enumerate()
                .map(fill)
                .collect::<Vec<_>>()
        }),
    };
    for result in results {
        result?;
    }
    let mut sums = [NeumaierSum::new(); 3];
    for row in values.chunks_exact(nt) {
        for (s, v) in sums
            .iter_mut()
            .zip([row[0].powi(2), row[0].powi(3), row[0].powi(4)])
        {
            s.add(v);
        }
    }
    let initial = sums.map(|s| s.total() / np as f64);
    for (i, &v) in initial.iter().enumerate() {
        valid(v, "rough_lsv_initial_moment", i, true)?;
    }
    let driver = FamilyTable {
        values,
        rows: nt,
        initial,
        blocks: plan.brownian_block_count(),
    };
    let core = calibrate(target, &driver, initial_forward, config, executor)?;
    Ok(CalibratedRoughFamilyLsv { model, core })
}
impl CalibratedRoughFamilyLsv {
    #[must_use]
    pub fn model(&self) -> &RoughVolatilityModel {
        &self.model
    }
    #[must_use]
    pub fn surface(&self) -> &LsvLeverageSurface {
        &self.core.surface
    }
    #[must_use]
    pub fn target(&self) -> &LocalVarianceGrid {
        &self.core.target
    }
    #[must_use]
    pub fn config(&self) -> &LsvParticleConfig {
        &self.core.config
    }
    #[must_use]
    pub fn conditional_moments(&self) -> &[LsvConditionalMoments] {
        &self.core.moments
    }
    #[must_use]
    pub fn diagnostics(&self) -> &[LsvCalibrationRowDiagnostics] {
        &self.core.diagnostics
    }
    pub fn pricing_plan(&self, grid: &LocalVolTimeGrid) -> Result<RoughFamilyLsvPlan, LsvError> {
        RoughFamilyLsvPlan::new(self.model.clone(), self.core.surface.clone(), grid)
    }
    /// Full discrete particle VJP at fixed model/shocks/bandwidth/donor topology.
    /// Includes the effect of previous leverage rows on the conditional estimator.
    /// Does not differentiate the stochastic-volatility model parameters.
    pub fn reverse_leverage(&self, seeds: &[f64]) -> Result<Vec<f64>, LsvError> {
        self.core.reverse(seeds, false)
    }
}
impl CalibrationReverse for CalibratedRoughFamilyLsv {
    type Adjoints = Vec<f64>;
    type Error = LsvError;
    fn validate_calibration_reverse(&self) -> Result<(), LsvError> {
        self.core.validate_reverse()
    }
    fn calibration_pullback(&self, seeds: &[f64]) -> Result<Vec<f64>, LsvError> {
        self.reverse_leverage(seeds)
    }
}

mod heston_parameter;

mod mixed_parameter;
