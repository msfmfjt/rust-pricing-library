//! Fixed-target particle pullback into Quadratic rough Heston parameters.
use super::*;
use crate::engine::processes::rough_volatility::QuadraticHestonMcRiskPlan;

impl CalibratedRoughFamilyLsv {
    /// Full Leverage derivative at a fixed relative target, optional left-H derivative.
    pub fn reverse_quadratic_heston_parameters(
        &self,
        seeds: &[f64],
        include_hurst: bool,
    ) -> Result<Vec<f64>, LsvError> {
        let path = RoughVolatilityPathPlan::compile(
            self.model.clone(),
            self.core.surface.times().to_vec(),
        )?;
        let plan = QuadraticHestonMcRiskPlan::compile(&path, include_hurst)?;
        self.quadratic_heston_parameter_pullback(seeds, &plan)
    }
    pub(in crate::engine) fn quadratic_heston_parameter_pullback(
        &self,
        seeds: &[f64],
        plan: &QuadraticHestonMcRiskPlan,
    ) -> Result<Vec<f64>, LsvError> {
        // Compilation is shared by all scramble pullbacks in the pricing layer.
        if plan.path_plan().model() != &self.model
            || plan.path_plan().time_nodes() != self.core.surface.times()
        {
            return Err(LsvError::InvalidInput {
                field: "quadratic_heston_lsv_reverse_plan",
                index: 0,
            });
        }
        let variance = self.core.reverse_particle_variances(seeds)?;
        let nt = self.core.surface.times().len();
        let mut sums = vec![NeumaierSum::new(); plan.width()];
        for (i, row) in variance.chunks_exact(nt).enumerate() {
            let z = plan.path_plan().pseudo_shocks(
                self.core.config.seed(),
                i as u64,
                RandomDomain::LsvCalibration,
            );
            // Only variance is regenerated. No unrelated unlevered stock path.
            let adjoints = plan.reverse(&z, row)?;
            for (s, a) in sums.iter_mut().zip(adjoints) {
                s.add(a);
            }
        }
        let result = sums.into_iter().map(NeumaierSum::total).collect::<Vec<_>>();
        for (i, &a) in result.iter().enumerate() {
            valid(a, "quadratic_heston_lsv_parameter_adjoint", i, false)?;
        }
        Ok(result)
    }
}
