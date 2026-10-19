//! Nonflat second transition. No flat initial-Gamma or Gaussian terminal shortcut.
use crate::MonteCarloError;
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::{DeterministicExecutor, EngineConfig, RandomDomain};
use crate::risk::{
    RiskConfigError, VegaKtConvolutionPlan, VegaKtGammaRecoveryPolicy, VegaKtGammaRecoveryStatus,
    VegaKtStepMoment, VegaKtTwoStepSmileGammaAudit, VegaKtVarianceGammaCandidate,
    VegaKtVarianceGammaQuality,
};
use pricing_numerics::NeumaierSum;

fn invalid(s: &'static str) -> MonteCarloError {
    RiskConfigError::VegaKtChainInput { quantity: s }.into()
}
fn sum(v: impl Iterator<Item = f64>) -> f64 {
    v.collect::<NeumaierSum>().total()
}
fn norm(v: impl Iterator<Item = f64>) -> f64 {
    v.fold(0., f64::hypot)
}
fn quality(
    b: &[VegaKtStepMoment],
    r: &[VegaKtStepMoment],
    q: &[VegaKtStepMoment],
    p: &VegaKtGammaRecoveryPolicy,
) -> Result<VegaKtVarianceGammaQuality, MonteCarloError> {
    let metrics = |v: &[VegaKtStepMoment]| -> Result<(f64, f64), MonteCarloError> {
        let errors = v
            .iter()
            .map(|m| {
                m.standard_error()
                    .ok_or_else(|| invalid("at least two variance-Gamma samples required"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((norm(v.iter().map(|m| m.mean())), norm(errors.into_iter())))
    };
    let (b, bse) = metrics(b)?;
    let (r, rse) = metrics(r)?;
    let (q, qse) = metrics(q)?;
    if [b, bse, r, rse, q, qse].iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite variance-Gamma quality"));
    }
    let ratio = |a: f64, b: f64| (b > 0.).then(|| a / b).filter(|v| v.is_finite());
    let reconstruction_margin_ratio = ratio(
        r + p.mc_margin_multiplier() * rse,
        b - p.mc_margin_multiplier() * bse,
    );
    let gamma_relative_standard_error = ratio(qse, q);
    Ok(VegaKtVarianceGammaQuality {
        sensitivity_norm: b,
        sensitivity_mc_norm: bse,
        reconstruction_norm: r,
        reconstruction_mc_norm: rse,
        gamma_norm: q,
        gamma_mc_norm: qse,
        reconstruction_margin_ratio,
        gamma_relative_standard_error,
        passed: reconstruction_margin_ratio.is_some_and(|v| v <= p.reconstruction_relative_limit())
            && gamma_relative_standard_error
                .is_some_and(|v| v <= p.gamma_relative_standard_error_limit()),
    })
}

impl SimulationPlan {
    pub(super) fn variance_gamma_candidate(
        &self,
        op: &VegaKtConvolutionPlan,
        directions: &[Vec<f64>],
        range: (usize, u64, u64),
        policy: &VegaKtGammaRecoveryPolicy,
    ) -> Result<VegaKtVarianceGammaCandidate, MonteCarloError> {
        let (row, start, samples) = range;
        let lv = self.state_only_path_gamma_runtime()?;
        let EngineConfig::PseudoMonteCarlo(engine) = self.engine else {
            return Err(invalid("pseudo-MC required for variance-Gamma audit"));
        };
        let antithetic = engine.variance_reduction().antithetic();
        let m = lv.grid.log_moneyness_nodes().len();
        let n = op.gamma_strikes().len();
        let d = directions.len();
        let components = 2 + n + 3 * m + 3 + 3 * d;
        let sample = |z: &[f64]| -> Result<Vec<f64>, MonteCarloError> {
            let path = lv.plan.evolve_path(&lv.grid, self.spot, z)?;
            let (price, seeds) = self.chain_discounted_payoff_and_seeds(lv, path.states())?;
            let ad = path.reverse_state_adjoints(&seeds, lv.grid.values().len(), m)?;
            // Exact node-variance adjoints, including interpolation. Dividing
            // effective-volatility hats by nodal sigma is NOT valid here.
            let b = &ad.local_variance_value_adjoints()[row * m..(row + 1) * m];
            let q = op.recover(b)?;
            let reconstructed = op.reconstruct(&q)?;
            let residual: Vec<_> = reconstructed.iter().zip(b).map(|(a, b)| a - b).collect();
            let mut out = Vec::with_capacity(components);
            out.extend([
                price,
                sum(ad.local_variance_value_adjoints()[..m].iter().copied()),
            ]);
            out.extend(&q);
            out.extend(b);
            out.extend(&residual);
            out.extend(std::iter::repeat_n(0., m));
            out.extend([
                sum(b.iter().copied()),
                sum(reconstructed.iter().copied()),
                sum(residual.iter().copied()),
            ]);
            for values in [b, reconstructed.as_slice(), residual.as_slice()] {
                for direction in directions {
                    out.push(sum(values.iter().zip(direction).map(|(a, b)| a * b)));
                }
            }
            Ok(out)
        };
        let statistics = DeterministicExecutor::new(self.execution_policy)?
            .try_map_reduce_statistics_vector(
                samples,
                components,
                |i, out| -> Result<(), MonteCarloError> {
                    let z = lv.plan.path_shocks(
                        engine.master_seed(),
                        start + i,
                        RandomDomain::Valuation,
                        engine.variance_reduction().brownian_bridge(),
                    )?;
                    let primary = sample(&z)?;
                    if antithetic {
                        let mate = sample(&z.iter().map(|z| -z).collect::<Vec<_>>())?;
                        for ((o, a), b) in out.iter_mut().zip(primary).zip(mate) {
                            *o = 0.5 * a + 0.5 * b;
                        }
                    } else {
                        out.copy_from_slice(&primary);
                    }
                    for j in 0..m {
                        out[2 + n + 2 * m + j] = f64::from(out[2 + n + j] != 0.);
                    }
                    Ok(())
                },
            )?;
        let v = statistics
            .into_iter()
            .map(|s| super::vegakt_steps::moment(s, samples))
            .collect::<Result<Vec<_>, _>>()?;
        let b = &v[2 + n..2 + n + m];
        let r = &v[2 + n + m..2 + n + 2 * m];
        let q = &v[2..2 + n];
        let offset = 2 + n + 3 * m;
        Ok(VegaKtVarianceGammaCandidate {
            relative_regularization: op.relative_regularization(),
            sampling_index_start: start,
            independent_sampling_units: samples,
            inverse_noise_gain_bound: op.inverse_noise_gain_bound(),
            resolution_trace: op.resolution_trace(),
            native_price: v[0],
            unrecovered_initial_variance_sensitivity: v[1],
            gamma_masses: q.to_vec(),
            variance_sensitivities: b.to_vec(),
            reconstruction_differences: r.to_vec(),
            nonzero_sampling_units: v[2 + n + 2 * m..offset]
                .iter()
                .map(|s| (s.mean() * samples as f64).round() as u64)
                .collect(),
            variance_sensitivity_total: v[offset],
            reconstruction_total: v[offset + 1],
            total_difference: v[offset + 2],
            variance_direction_risks: v[offset + 3..offset + 3 + d].to_vec(),
            recovered_direction_risks: v[offset + 3 + d..offset + 3 + 2 * d].to_vec(),
            direction_differences: v[offset + 3 + 2 * d..].to_vec(),
            quality: quality(b, r, q, policy)?,
        })
    }

    pub(in crate::engine) fn evaluate_vega_kt_two_step_smile_gamma_audit(
        &self,
        strikes: Vec<f64>,
        policy: &VegaKtGammaRecoveryPolicy,
        directions: Vec<Vec<f64>>,
    ) -> Result<VegaKtTwoStepSmileGammaAudit, MonteCarloError> {
        let lv = self.state_only_path_gamma_runtime()?;
        let times = lv.plan.time_grid().nodes();
        let nodes = lv.grid.log_moneyness_nodes();
        let m = nodes.len();
        let n = strikes.len();
        if times.len() != 3
            || lv.grid.time_nodes() != times
            || lv.grid.values()[..m]
                .iter()
                .any(|&v| v != lv.grid.values()[0])
        {
            return Err(invalid(
                "exactly two simulation/grid intervals and a flat first variance row required",
            ));
        }
        if policy.preserves_local_vega_total()
            || !policy.local_vega_conservation_directions().is_empty()
        {
            return Err(invalid(
                "volatility conservation options cannot be reused for variance-node recovery; supply variance_conservation_directions",
            ));
        }
        if !directions.is_empty() {
            policy
                .clone()
                .with_local_vega_direction_conservation(vec![directions.clone()])?;
            if directions[0].len() != m {
                return Err(invalid("variance direction width must match source grid"));
            }
        }
        let EngineConfig::PseudoMonteCarlo(engine) = self.engine else {
            return Err(invalid("pseudo-MC required for variance-Gamma audit"));
        };
        let count = engine.independent_sampling_units().get();
        if !(2..=10_000_000).contains(&count) {
            return Err(invalid(
                "2..10000000 variance-Gamma validation samples required",
            ));
        }
        let end = count
            .checked_add(policy.pilot_sampling_units())
            .ok_or_else(|| invalid("variance-Gamma pilot counter overflow"))?;
        let d = directions.len();
        let components = 2 + n + 3 * m + 3 + 3 * d;
        let blocks = count
            .max(policy.pilot_sampling_units())
            .div_ceil(self.execution_policy.reduction_block_size().get());
        if blocks as u128 * components as u128 > 4_000_000 {
            return Err(invalid("variance-Gamma reduction storage budget exceeded"));
        }
        let source_variance = lv.grid.values()[0] * times[1];
        let second = lv.grid.values()[m..2 * m].to_vec();
        // Compile every inverse (and validate rank/quadrature) before any samples.
        let mut operators = Vec::new();
        let mut accuracies = Vec::new();
        for &lambda in policy.candidate_regularizations() {
            let (op, accuracy) = VegaKtConvolutionPlan::from_gaussian_variance_source(
                self.spot,
                source_variance,
                times[2] - times[1],
                nodes,
                &second,
                strikes.clone(),
                lambda,
            )?;
            operators.push(if directions.is_empty() {
                op
            } else {
                op.conserve_local_vega_directions(directions.clone(), false)?
            });
            accuracies.push(accuracy);
        }
        let antithetic = engine.variance_reduction().antithetic();
        let evaluate = |op: &VegaKtConvolutionPlan, start, samples| {
            self.variance_gamma_candidate(op, &directions, (1, start, samples), policy)
        };
        let pilot_candidates = operators
            .iter()
            .map(|op| evaluate(op, count, policy.pilot_sampling_units()))
            .collect::<Result<Vec<_>, _>>()?;
        let choice = pilot_candidates.iter().position(|c| c.quality.passed);
        let validation = choice
            .map(|i| evaluate(&operators[i], 0, count))
            .transpose()?;
        let status = match &validation {
            None => VegaKtGammaRecoveryStatus::NoPilotCandidate,
            Some(v) if v.quality.passed => VegaKtGammaRecoveryStatus::PositiveTimePolicyPassed,
            Some(_) => VegaKtGammaRecoveryStatus::ValidationFailed,
        };
        let estimate = validation
            .as_ref()
            .filter(|v| v.quality.passed)
            .map(|v| v.recovered_direction_risks.clone());
        Ok(VegaKtTwoStepSmileGammaAudit {
            request_fingerprint: self.request_fingerprint(),
            plan_fingerprint: self.plan_fingerprint(),
            policy: policy.clone(),
            variance_conservation_directions: directions,
            status,
            validation_sampling_units: count,
            antithetic,
            pilot_index_start: count,
            pilot_index_end_exclusive: end,
            source_maturity: times[1],
            terminal_maturity: times[2],
            source_variance,
            source_log_nodes: nodes.to_vec(),
            second_step_variances: second,
            gamma_strikes: strikes,
            operator_accuracy: accuracies.remove(0),
            forward_matrix: operators[0].forward_matrix().to_vec(),
            pilot_candidates,
            selected_regularization: choice.map(|i| operators[i].relative_regularization()),
            validation,
            estimate,
        })
    }
}
