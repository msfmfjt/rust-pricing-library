//! Conditional terminal-payoff variance risk without linear-payoff MC cancellation.
use crate::MonteCarloError;
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::{DeterministicExecutor, EngineConfig, RandomDomain};
use crate::risk::{
    RiskConfigError, VegaKtConvolutionPlan, VegaKtGammaRecoveryPolicy, VegaKtStepMoment,
    VegaKtVarianceGammaCandidate, VegaKtVarianceGammaQuality,
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
    pub(super) fn multi_step_variance_gamma_candidate(
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
            let terminal = row + 1;
            let observations = self.local_vol_state_observations(
                lv,
                path.states(),
                *path.states().last().unwrap(),
            )?;
            let payoff = super::vegakt_terminal::try_affine_observation_payoff_gamma_measure(
                &self.payoff,
                |u, date| {
                    if u != self.underlying {
                        return None;
                    }
                    let position = self
                        .observation_dates
                        .iter()
                        .position(|d| *d == Some(date))?;
                    let node = observations[position].node_index;
                    if node == terminal {
                        Some((0., 1.))
                    } else if node < terminal {
                        Some((path.states()[node], 0.))
                    } else {
                        None
                    }
                },
            )?
            .ok_or_else(|| invalid("piecewise affine terminal payoff required"))?;
            let source = path.states()[row];
            let x = (source / self.spot).ln();
            let nodes = lv.grid.log_moneyness_nodes();
            let dt = lv.plan.time_grid().nodes()[terminal] - lv.plan.time_grid().nodes()[row];
            let sd = (lv
                .grid
                .interpolate(lv.plan.time_grid().nodes()[row], x)?
                .value
                * dt)
                .sqrt();
            let risk = sum(payoff.gamma().atoms().iter().map(|a| {
                a.mass()
                    * source
                    * pricing_numerics::standard_normal_pdf(
                        (source.ln() - a.strike().ln()) / sd + sd / 2.,
                    )
                    * dt
                    / (2. * sd)
            }));
            let index = nodes
                .partition_point(|z| *z <= x)
                .saturating_sub(1)
                .min(m - 2);
            let h = ((x - nodes[index]) / (nodes[index + 1] - nodes[index])).clamp(0., 1.);
            let mut conditional = vec![0.; m];
            conditional[index] = (1. - h) * risk;
            conditional[index + 1] = h * risk;
            let b = &conditional[..];
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
}
