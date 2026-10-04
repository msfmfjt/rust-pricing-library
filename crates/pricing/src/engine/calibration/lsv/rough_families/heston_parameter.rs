//! Fixed-target particle pullback into Heston variance histories and parameters.
use super::*;
use crate::engine::processes::rough_volatility::HestonVarianceRiskPlan;

impl CalibratedRoughFamilyLsv {
    /// Reverse squared-leverage seeds to v0, kappa, theta, nu, rho and,
    /// optionally, power-kernel Rough Heston H. The target Local Variance
    /// nodes/axes, initial forward, seeds, bandwidth and donor topology are fixed.
    /// Requires a retained calibration trace. Not a market-IV recalibration.
    pub fn reverse_heston_parameters(
        &self,
        seeds: &[f64],
        include_hurst: bool,
    ) -> Result<Vec<f64>, LsvError> {
        let plan = HestonVarianceRiskPlan::compile(
            self.model.clone(),
            self.core.surface.times().to_vec(),
            include_hurst,
        )?;
        self.heston_parameter_pullback(seeds, &plan)
    }
    pub(in crate::engine) fn heston_parameter_pullback(
        &self,
        seeds: &[f64],
        plan: &HestonVarianceRiskPlan,
    ) -> Result<Vec<f64>, LsvError> {
        // Compilation is shared by all scramble pullbacks in the pricing layer.
        if plan.path_plan().model() != &self.model
            || plan.path_plan().time_nodes() != self.core.surface.times()
        {
            return Err(LsvError::InvalidInput {
                field: "heston_lsv_reverse_plan",
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
            valid(a, "heston_lsv_parameter_adjoint", i, false)?;
        }
        Ok(result)
    }
}

impl Calibration {
    /// Cotangents on a_i,r^2 = diffusion variance. Both the kernel regression
    /// denominator and all earlier particle positions depend on model parameters.
    /// This separate path leaves the existing fixed-model target VJP unchanged.
    fn reverse_particle_variances(&self, seeds: &[f64]) -> Result<Vec<f64>, LsvError> {
        length("leverage_adjoints", self.surface.values.len(), seeds.len())?;
        for (i, &a) in seeds.iter().enumerate() {
            valid(a, "leverage_adjoint", i, false)?;
        }
        let trace = self
            .trace
            .as_ref()
            .ok_or(LsvError::ReverseTraceNotRetained)?;
        let nt = self.surface.times.len();
        let m = self.surface.log_nodes.len();
        let np = self.config.particle_count;
        let cells = np.checked_mul(nt).ok_or_else(resource_error)?;
        if cells > 8_000_000 {
            return Err(resource_error());
        }
        let mut vb = vec![0.0; cells];
        let mut lbar = seeds.to_vec();
        let mut state_bar = vec![0.0; np];
        let rng = Philox4x32::from_seed(self.config.seed);
        for r in (0..nt).rev() {
            if r + 1 < nt {
                let dt = self.surface.times[r + 1] - self.surface.times[r];
                for (i, bar) in state_bar.iter_mut().enumerate() {
                    let f = trace[r].states[i];
                    let next = trace[r + 1].states[i];
                    let lookup = self
                        .surface
                        .lookup_row(r, (f / self.surface.initial_f).ln());
                    let a2 = trace[r].multipliers[i].powi(2);
                    let q = lookup.value * a2;
                    let z = rng.standard_normal(RandomCoordinate::new(
                        i as u64,
                        r as u32,
                        RandomDomain::LsvCalibration,
                    ));
                    let qb = if a2 == 0.0 {
                        0.0
                    } else {
                        *bar * next * (-0.5 * dt + dt.sqrt() * z / (2.0 * q.sqrt()))
                    };
                    let lb = qb * a2;
                    vb[i * nt + r] += qb * lookup.value;
                    lookup.transpose(lb, &mut lbar);
                    *bar = *bar * next / f + lb * lookup.derivative_log_f / f;
                }
            }
            let mut moment_bar = vec![0.0; m];
            for j in 0..m {
                let moment = self.moments[r * m + j];
                let a = self
                    .target
                    .interpolate(self.surface.times[r], self.surface.log_nodes[j])?
                    .value;
                moment_bar[moment.source_node] -= lbar[r * m + j] * a / moment.second.powi(2);
            }
            if r == 0 {
                // The primal first row uses empirical initial moments, not a
                // conditional regression and not an assumed unit multiplier.
                let b = moment_bar.into_iter().collect::<NeumaierSum>().total() / np as f64;
                for row in vb.chunks_exact_mut(nt) {
                    row[0] += b;
                }
                continue;
            }
            let h = self.config.log_bandwidth;
            for (j, mb) in moment_bar
                .into_iter()
                .enumerate()
                .filter(|(_, b)| *b != 0.0)
            {
                let sum_w = trace[r].weight_sums[j];
                valid(sum_w, "heston_lsv_weight_sum", r * m + j, true)?;
                let m2 = self.moments[r * m + j].second;
                for (i, bar) in state_bar.iter_mut().enumerate() {
                    let f = trace[r].states[i];
                    let u = ((f / self.surface.initial_f).ln() - self.surface.log_nodes[j]) / h;
                    let w = quartic(u);
                    let dw_df = quartic_derivative(u) / (h * f);
                    *bar += mb * dw_df * (trace[r].multipliers[i].powi(2) - m2) / sum_w;
                    vb[i * nt + r] += mb * w / sum_w;
                }
            }
        }
        for (i, &a) in vb.iter().chain(&state_bar).chain(&lbar).enumerate() {
            valid(a, "heston_lsv_particle_variance_adjoint", i, false)?;
        }
        Ok(vb)
    }
}
