//! Fixed-target particle pullback into Mixed rough Bergomi eta/rho.
use super::*;
use crate::engine::processes::rough_volatility::MixedBergomiVarianceRiskPlan;

impl CalibratedRoughFamilyLsv {
    /// Component eta/rho risk of Leverage at fixed relative Local Variance target.
    /// Hurst, weights, xi, calibration seed and support/donor topology are fixed.
    /// Requires a retained particle reverse trace.
    pub fn reverse_mixed_bergomi_parameters(&self, seeds: &[f64]) -> Result<Vec<f64>, LsvError> {
        let plan = MixedBergomiVarianceRiskPlan::compile(
            self.model.clone(),
            self.core.surface.times().to_vec(),
        )?;
        self.mixed_bergomi_parameter_pullback(seeds, &plan)
    }
    /// Extend component eta/rho Leverage adjoints by Hurst, including the
    /// complete kernel and finite-grid centering derivatives. Weights/xi fixed.
    pub fn reverse_mixed_bergomi_parameters_with_hurst(
        &self,
        seeds: &[f64],
    ) -> Result<Vec<f64>, LsvError> {
        let plan = MixedBergomiVarianceRiskPlan::compile_with_hurst(
            self.model.clone(),
            self.core.surface.times().to_vec(),
        )?;
        self.mixed_bergomi_parameter_pullback(seeds, &plan)
    }
    pub(in crate::engine) fn mixed_bergomi_parameter_pullback(
        &self,
        seeds: &[f64],
        plan: &MixedBergomiVarianceRiskPlan,
    ) -> Result<Vec<f64>, LsvError> {
        // Compilation is shared by all scramble pullbacks in the pricing layer.
        if plan.path_plan().model() != &self.model
            || plan.path_plan().time_nodes() != self.core.surface.times()
        {
            return Err(LsvError::InvalidInput {
                field: "mixed_bergomi_lsv_reverse_plan",
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
            valid(a, "mixed_bergomi_lsv_parameter_adjoint", i, false)?;
        }
        Ok(result)
    }
}
