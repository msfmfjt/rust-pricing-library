//! Same-path joint readouts of deterministic initial Gamma and recovered second Gamma.
use crate::MonteCarloError;
use crate::core::FiniteF64;
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::{DeterministicExecutor, EngineConfig, RandomDomain};
use crate::risk::{
    RiskConfigError, VegaKtConvolutionPlan, VegaKtGammaRecoveryPolicy, VegaKtJointGammaAudit,
    VegaKtJointGammaDirections, VegaKtJointGammaStatistics, VegaKtStepMoment,
};
use pricing_numerics::{
    CenteredCovariance, CenteredMoment, NeumaierSum, reduce_covariances, reduce_moments,
    reduce_sums,
};
use rayon::prelude::*;
fn invalid(s: &'static str) -> MonteCarloError {
    RiskConfigError::VegaKtChainInput { quantity: s }.into()
}
fn finite(v: f64) -> Result<f64, MonteCarloError> {
    Ok(FiniteF64::new(v, "joint_gamma_statistic")
        .map_err(RiskConfigError::from)?
        .get())
}
fn sum(v: impl Iterator<Item = f64>) -> f64 {
    v.collect::<NeumaierSum>().total()
}
impl SimulationPlan {
    pub(in crate::engine) fn evaluate_vega_kt_joint_gamma_audit(
        &self,
        strikes: Vec<f64>,
        initial_bins: Vec<f64>,
        policy: &VegaKtGammaRecoveryPolicy,
        directions: VegaKtJointGammaDirections,
        subdivisions: usize,
        relative_tolerance: f64,
    ) -> Result<VegaKtJointGammaAudit, MonteCarloError> {
        let lv = self.state_only_path_gamma_runtime()?;
        let width = lv.grid.log_moneyness_nodes().len();
        let d = directions.risk_directions.len();
        if !(1..=16).contains(&d)
            || directions.risk_directions.iter().any(|r| {
                r.len() != width + 1
                    || r.iter().any(|v| !v.is_finite())
                    || r.iter().all(|v| *v == 0.)
            })
        {
            return Err(invalid(
                "1..16 finite nonzero joint risk directions must contain first variance plus all second variance nodes",
            ));
        }
        let EngineConfig::PseudoMonteCarlo(engine) = self.engine else {
            return Err(invalid("pseudo-MC required for joint Gamma"));
        };
        let count = engine.independent_sampling_units().get();
        let block_size = self.execution_policy.reduction_block_size().get();
        let blocks = count.div_ceil(block_size);
        let components = 3 * d;
        let covariance_components = components * (components + 1) / 2;
        if !(2..=10_000_000).contains(&count)
            || blocks as u128 * (components + covariance_components) as u128 > 4_000_000
        {
            return Err(invalid(
                "joint Gamma sample or covariance storage budget exceeded",
            ));
        }
        let initial = self.evaluate_vega_kt_initial_smile_gamma_audit(
            initial_bins,
            subdivisions,
            relative_tolerance,
        )?;
        let second = self.evaluate_vega_kt_two_step_smile_gamma_audit(
            strikes.clone(),
            policy,
            directions.conservation_basis.clone(),
        )?;
        let mut result = VegaKtJointGammaAudit {
            directions,
            initial,
            second,
            joint_validation: None,
        };
        let Some(initial_value) = result.initial.estimate else {
            return Ok(result);
        };
        if result.second.estimate.is_none() {
            return Ok(result);
        }
        let (op, _) = VegaKtConvolutionPlan::from_gaussian_variance_source(
            self.spot,
            result.second.source_variance,
            result.second.terminal_maturity - result.second.source_maturity,
            &result.second.source_log_nodes,
            &result.second.second_step_variances,
            strikes,
            result
                .second
                .selected_regularization
                .expect("successful second Gamma has a selected inverse"),
        )?;
        let op = if result.directions.conservation_basis.is_empty() {
            op
        } else {
            op.conserve_local_vega_directions(result.directions.conservation_basis.clone(), false)?
        };
        let sample = |z: &[f64]| -> Result<Vec<f64>, MonteCarloError> {
            let path = lv.plan.evolve_path(&lv.grid, self.spot, z)?;
            let (_, seeds) = self.chain_discounted_payoff_and_seeds(lv, path.states())?;
            let ad = path.reverse_state_adjoints(&seeds, lv.grid.values().len(), width)?;
            let initial_ad = sum(ad.local_variance_value_adjoints()[..width].iter().copied());
            let b = &ad.local_variance_value_adjoints()[width..2 * width];
            let reconstructed = op.reconstruct(&op.recover(b)?)?;
            let mut values = vec![0.; components];
            for (j, w) in result.directions.risk_directions.iter().enumerate() {
                values[j] =
                    finite(w[0] * initial_ad + sum(w[1..].iter().zip(b).map(|(a, b)| a * b)))?;
                values[d + j] = finite(
                    w[0] * initial_value
                        + sum(w[1..].iter().zip(&reconstructed).map(|(a, b)| a * b)),
                )?;
                values[2 * d + j] = finite(values[d + j] - values[j])?;
            }
            Ok(values)
        };
        let antithetic = engine.variance_reduction().antithetic();
        let pairs: Vec<_> = (0..components)
            .flat_map(|i| (i..components).map(move |j| (i, j)))
            .collect();
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let partials = executor
            .install(|| {
                (0..blocks)
                    .into_par_iter()
                    .map(|block| {
                        let mut sums = vec![NeumaierSum::new(); components];
                        let mut moments = vec![CenteredMoment::new(); components];
                        let mut cov = vec![CenteredCovariance::new(); pairs.len()];
                        for index in block * block_size..count.min(block * block_size + block_size)
                        {
                            let z = lv.plan.path_shocks(
                                engine.master_seed(),
                                index,
                                RandomDomain::Valuation,
                                engine.variance_reduction().brownian_bridge(),
                            )?;
                            let mut values = sample(&z)?;
                            if antithetic {
                                let mate = sample(&z.iter().map(|z| -z).collect::<Vec<_>>())?;
                                for (a, b) in values.iter_mut().zip(mate) {
                                    *a = 0.5 * *a + 0.5 * b;
                                }
                            }
                            for ((s, m), &v) in sums.iter_mut().zip(&mut moments).zip(&values) {
                                s.add(v);
                                m.add(v);
                            }
                            for (c, &(i, j)) in cov.iter_mut().zip(&pairs) {
                                c.add(values[i], values[j]);
                            }
                        }
                        Ok::<_, MonteCarloError>((sums, moments, cov))
                    })
                    .collect::<Vec<_>>()
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let moments = (0..components)
            .map(|i| {
                let mean = finite(
                    reduce_sums(partials.iter().map(|p| p.0[i]).collect()).total() / count as f64,
                )?;
                let variance = reduce_moments(partials.iter().map(|p| p.1[i]).collect())
                    .sample_variance()
                    .map(finite)
                    .transpose()?;
                let se = variance
                    .map(|v| finite(v.sqrt() / (count as f64).sqrt()))
                    .transpose()?;
                Ok(VegaKtStepMoment {
                    mean,
                    sample_variance: variance,
                    standard_error: se,
                })
            })
            .collect::<Result<Vec<_>, MonteCarloError>>()?;
        let mut cov = vec![0.; components * components];
        for (index, &(i, j)) in pairs.iter().enumerate() {
            let value = finite(
                reduce_covariances(partials.iter().map(|p| p.2[index]).collect())
                    .sample_covariance()
                    .expect("at least two samples"),
            )?;
            cov[i * components + j] = value;
            cov[j * components + i] = value;
        }
        let allowances = result
            .directions
            .risk_directions
            .iter()
            .map(|w| finite(w[0].abs() * result.initial.functional_tolerance))
            .collect::<Result<Vec<_>, _>>()?;
        let roundoff_allowances = (0..d)
            .map(|i| {
                finite(
                    256. * f64::EPSILON
                        * (moments[i].mean().abs()
                            + moments[d + i].mean().abs()
                            + moments[i].sample_variance().unwrap().sqrt()
                            + moments[d + i].sample_variance().unwrap().sqrt()),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let passed = moments[2 * d..]
            .iter()
            .zip(&allowances)
            .zip(&roundoff_allowances)
            .all(|((m, a), roundoff)| {
                m.standard_error()
                    .is_some_and(|se| m.mean().abs() <= 6. * se + a + roundoff)
            });
        result.joint_validation = Some(VegaKtJointGammaStatistics {
            independent_sampling_units: count,
            antithetic,
            native_directions: moments[..d].to_vec(),
            recovered_directions: moments[d..2 * d].to_vec(),
            differences: moments[2 * d..].to_vec(),
            joint_sample_covariance: cov,
            initial_error_allowances: allowances,
            roundoff_allowances,
            passed,
        });
        Ok(result)
    }
}
