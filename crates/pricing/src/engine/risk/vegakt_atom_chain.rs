//! Signed atom Gamma chain with analytic full-domain propagation and joint MC.
use super::vegakt_measure::invalid;
use crate::MonteCarloError;
use crate::core::FiniteF64;
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::{DeterministicExecutor, EngineConfig, RandomDomain};
use crate::risk::{
    RiskConfigError, VegaKtAtomChainDiagnostic, VegaKtAtomChainEstimate, VegaKtConvolutionPlan,
    VegaKtGammaAtoms, VegaKtGammaRecoveryPolicy, VegaKtGammaRecoveryStatus, VegaKtStepMoment,
};
use pricing_numerics::{
    CenteredCovariance, CenteredMoment, NeumaierSum, reduce_covariances, reduce_moments,
    reduce_sums,
};
use rayon::prelude::*;

fn finite(v: f64) -> Result<f64, MonteCarloError> {
    Ok(FiniteF64::new(v, "atom_chain_statistic")
        .map_err(RiskConfigError::from)?
        .get())
}
fn sum(v: impl IntoIterator<Item = f64>) -> f64 {
    v.into_iter().collect::<NeumaierSum>().total()
}

impl SimulationPlan {
    pub(in crate::engine) fn evaluate_vega_kt_atom_chain_diagnostic(
        &self,
        strikes: Vec<f64>,
        policy: &VegaKtGammaRecoveryPolicy,
    ) -> Result<VegaKtAtomChainDiagnostic, MonteCarloError> {
        let lv = self.flat_path_gamma_runtime()?;
        let report = lv
            .vega_kt
            .as_ref()
            .ok_or_else(|| invalid("reporting IV basis required"))?;
        let basis = &report.basis;
        let times = lv.plan.time_grid().nodes();
        if basis.maturity_nodes() != &times[1..] {
            return Err(invalid("reporting expiries must equal positive simulation nodes").into());
        }
        let EngineConfig::PseudoMonteCarlo(engine) = self.engine else {
            return Err(invalid("pseudo-MC required for joint atom chain").into());
        };
        let count = engine.independent_sampling_units().get();
        let buckets = basis.bucket_count();
        if !(2..=10_000_000).contains(&count)
            || buckets > 256
            || (report.full_bucket_covariance && buckets > 64)
        {
            return Err(invalid(
                "atom chain requires 2..10000000 samples, <=256 buckets, <=64 with full covariance",
            )
            .into());
        }
        // Reject storage-overbudget requests before the pilot allocates its
        // own block statistics or evaluates any paths.
        let components = 2 * buckets + 4;
        let covariance_components = buckets
            + if report.full_bucket_covariance {
                buckets * (buckets + 1) / 2
            } else {
                0
            };
        let block_size = self.execution_policy.reduction_block_size().get();
        let block_count = count.div_ceil(block_size);
        let storage_blocks = block_count.max(policy.pilot_sampling_units().div_ceil(block_size));
        let recovery_components = (times.len().saturating_sub(2) as u128)
            * (strikes.len() as u128 + 3 * lv.grid.log_moneyness_nodes().len() as u128 + 3)
            + 3 * policy
                .local_vega_conservation_directions()
                .iter()
                .map(|r| r.len() as u128)
                .sum::<u128>()
            + 1;
        let storage_components =
            ((components + covariance_components) as u128).max(recovery_components);
        if storage_blocks as u128 * storage_components > 4_000_000 {
            return Err(invalid("atom chain reduction storage budget exceeded").into());
        }
        let width = lv.grid.log_moneyness_nodes().len();
        let bw = basis.log_moneyness_nodes().len();
        let mut variances = Vec::new();
        let mut cumulative = Vec::new();
        let mut total = 0.;
        let mut sigmas = Vec::new();
        for (i, pair) in times.windows(2).enumerate() {
            let variance = lv.grid.interpolate(pair[0], 0.)?.value;
            let dt = pair[1] - pair[0];
            variances.push(variance * dt);
            sigmas.push(variance.sqrt());
            total += variance * dt;
            cumulative.push(total);
            let sigma = (total / pair[1]).sqrt();
            if basis.implied_volatilities()[i * bw..(i + 1) * bw]
                .iter()
                .any(|&v| (v - sigma).abs() > 1e-10 * v.abs().max(sigma.abs()))
            {
                return Err(
                    invalid("reporting IV differs from discrete flat lognormal model").into(),
                );
            }
        }
        let recovery = self.evaluate_vega_kt_gamma_recovery_audit(strikes.clone(), policy)?;
        let mut result = VegaKtAtomChainDiagnostic {
            policy_label: if !policy.local_vega_conservation_directions().is_empty() {
                "flat_variance_direction_constrained_signed_atom_iv_chain_v1"
            } else if policy.preserves_local_vega_total() {
                "flat_variance_conservative_signed_atom_iv_chain_v1"
            } else {
                "flat_variance_signed_atom_iv_chain_v1"
            },
            reporting_basis: basis.clone(),
            interval_variances: variances.clone(),
            recovery,
            estimate: None,
        };
        if result.recovery.status != VegaKtGammaRecoveryStatus::PositiveTimePolicyPassed {
            return Ok(result);
        }
        let lambda = result
            .recovery
            .selected_regularization
            .expect("passed recovery has selection");
        let mut operators = Vec::new();
        // Each column deposits Gamma_i at i, and subtracts its backward
        // propagation at i-1. No histogram relocation or integration cutoff.
        let mut columns = Vec::new();
        for i in 1..variances.len() {
            let op = VegaKtConvolutionPlan::new(
                self.spot,
                cumulative[i - 1],
                variances[i],
                sigmas[i] * (times[i + 1] - times[i]),
                lv.grid.log_moneyness_nodes(),
                strikes.clone(),
                lambda,
            )?;
            operators.push(if !policy.local_vega_conservation_directions().is_empty() {
                op.conserve_local_vega_directions(
                    policy.local_vega_conservation_directions()[i - 1].clone(),
                    policy.preserves_local_vega_total(),
                )?
            } else if policy.preserves_local_vega_total() {
                op.conserve_local_vega_total()?
            } else {
                op
            });
            let mut row = Vec::new();
            for &k in &strikes {
                let atoms = VegaKtGammaAtoms::new(vec![(k, 1.)])?;
                let direct = atoms.project_terminal(basis, times[i + 1], self.spot)?;
                let previous = atoms.project_propagated(
                    basis,
                    times[i],
                    self.spot,
                    cumulative[i - 1],
                    variances[i],
                )?;
                row.push(
                    direct
                        .position_risk()
                        .raw_buckets()
                        .iter()
                        .zip(previous)
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>(),
                );
            }
            columns.push(row);
        }
        let mut nodal_vegas = Vec::new();
        for (i, &t) in basis.maturity_nodes().iter().enumerate() {
            for (j, &x) in basis.log_moneyness_nodes().iter().enumerate() {
                let sd = basis.implied_volatilities()[i * bw + j] * t.sqrt();
                nodal_vegas.push(
                    self.spot
                        * pricing_numerics::standard_normal_pdf(-x / sd + 0.5 * sd)
                        * t.sqrt(),
                );
            }
        }
        if nodal_vegas.iter().any(|&v| !v.is_finite() || v <= 0.) {
            return Err(invalid("positive representable nodal IV Vegas required").into());
        }
        // Layout: buckets, hedge quantities, total IV, total local, difference, price.
        let sample = |z: &[f64]| -> Result<Vec<f64>, MonteCarloError> {
            let payoff = self.conditional_initial_payoff(lv, z)?;
            let initial = payoff
                .gamma()
                .project_terminal(basis, times[1], self.spot)?;
            let path = lv.plan.evolve_path(&lv.grid, self.spot, z)?;
            let (price, seeds) = self.chain_discounted_payoff_and_seeds(lv, path.states())?;
            let ad = path.reverse_state_adjoints(&seeds, lv.grid.values().len(), width)?;
            let mut sums = initial
                .position_risk()
                .raw_buckets()
                .iter()
                .map(|&v| {
                    let mut s = NeumaierSum::new();
                    s.add(v);
                    s
                })
                .collect::<Vec<_>>();
            let mut local = NeumaierSum::new();
            let mut row = vec![0.; width];
            for (i, step) in ad.step_adjoints().iter().enumerate() {
                row.fill(0.);
                step.accumulate_local_volatility_nodes(&mut row)?;
                local.add(sum(row.iter().copied()));
                if i > 0 {
                    let gamma = operators[i - 1].recover(&row)?;
                    for (q, column) in gamma.iter().zip(&columns[i - 1]) {
                        for (s, &v) in sums.iter_mut().zip(column) {
                            s.add(q * v);
                        }
                    }
                }
            }
            let mut out = sums.into_iter().map(NeumaierSum::total).collect::<Vec<_>>();
            let total = sum(out.iter().copied());
            out.extend(out.clone().iter().zip(&nodal_vegas).map(|(v, b)| -v / b));
            out.extend([total, local.total(), total - local.total(), price]);
            out.into_iter().map(finite).collect()
        };
        let antithetic = engine.variance_reduction().antithetic();
        let mut pairs = (0..buckets)
            .map(|i| (i, components - 1))
            .collect::<Vec<_>>();
        if report.full_bucket_covariance {
            for i in 0..buckets {
                for j in i..buckets {
                    pairs.push((i, j));
                }
            }
        }
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        // Centered covariance, fixed block order; antithetic averaging precedes
        // every accumulation. No uncentered products or synthetic independence.
        let partials = executor
            .install(|| {
                (0..block_count)
                    .into_par_iter()
                    .map(|b| {
                        let mut sums = vec![NeumaierSum::new(); components];
                        let mut moments = vec![CenteredMoment::new(); components];
                        let mut cov = vec![CenteredCovariance::new(); pairs.len()];
                        for index in b * block_size..count.min(b * block_size + block_size) {
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
        let m = (0..components)
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
        let cov = (0..pairs.len())
            .map(|i| {
                finite(
                    reduce_covariances(partials.iter().map(|p| p.2[i]).collect())
                        .sample_covariance()
                        .expect("at least two samples"),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let full = report.full_bucket_covariance.then(|| {
            let mut matrix = vec![0.; buckets * buckets];
            for (&(i, j), &v) in pairs[buckets..].iter().zip(&cov[buckets..]) {
                matrix[i * buckets + j] = v;
                matrix[j * buckets + i] = v;
            }
            matrix
        });
        result.estimate = Some(VegaKtAtomChainEstimate {
            independent_sampling_units: count,
            antithetic,
            raw_buckets: m[..buckets].to_vec(),
            hedge_quantities: m[buckets..2 * buckets].to_vec(),
            total_iv_vega: m[2 * buckets],
            native_parallel_local_vega: m[2 * buckets + 1],
            iv_minus_local_parallel: m[2 * buckets + 2],
            constant_volatility_term_structure: sigmas.iter().all(|&s| s == sigmas[0]),
            native_price: m[2 * buckets + 3],
            bucket_price_sample_covariance: cov[..buckets].to_vec(),
            full_bucket_sample_covariance: full,
        });
        Ok(result)
    }
}
