//! Conditional payoff Gamma, without initial density division or price bumps.
use super::vegakt_measure::{invalid, log_ratio};
use super::vegakt_terminal::try_affine_observation_payoff_gamma_measure;
use crate::MonteCarloError;
use crate::core::{FiniteF64, PositiveF64};
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::{DeterministicExecutor, EngineConfig, RandomDomain};
use crate::risk::{RiskConfigError, VegaKtInitialMeasureDiagnostic};
use pricing_numerics::{NeumaierSum, standard_normal_cdf as cdf, standard_normal_pdf as pdf};

impl SimulationPlan {
    /// Scope is checked by the caller. Future shocks determine an exact affine
    /// map from the first state to every later observation; node zero stays fixed.
    pub(super) fn conditional_initial_payoff(
        &self,
        lv: &crate::engine::plan::simulation::LocalVolRuntime,
        z: &[f64],
    ) -> Result<crate::risk::VegaKtTerminalPayoff, MonteCarloError> {
        // The flat-space evolution is affine in S1. Keep valuation-date
        // observations fixed, and preserve every later observation date.
        let future =
            lv.plan
                .continue_states_from_first_node(&lv.grid, self.spot, self.spot, &z[1..])?;
        let observations = self.local_vol_state_observations(lv, &future, self.spot)?;
        let payoff = try_affine_observation_payoff_gamma_measure(&self.payoff, |u, date| {
            if u != self.underlying {
                return None;
            }
            let i = self
                .observation_dates
                .iter()
                .position(|d| *d == Some(date))?;
            let obs = observations[i];
            Some(if obs.node_index == 0 {
                (obs.post_spot, 0.)
            } else {
                (0., obs.post_spot / self.spot)
            })
        })?
        .ok_or_else(|| invalid("conditional piecewise-affine payoff required"))?;
        Ok(payoff)
    }

    pub(in crate::engine) fn evaluate_vega_kt_initial_gamma_diagnostic(
        &self,
        boundaries: Vec<f64>,
    ) -> Result<VegaKtInitialMeasureDiagnostic, MonteCarloError> {
        let lv = self.flat_path_gamma_runtime()?;
        if boundaries.is_empty()
            || boundaries.len() > 129
            || boundaries.iter().any(|b| !b.is_finite() || *b <= 0.)
            || boundaries.windows(2).any(|b| b[0] >= b[1])
        {
            return Err(invalid("1..129 positive ordered Gamma bin boundaries required").into());
        }
        let EngineConfig::PseudoMonteCarlo(engine) = self.engine else {
            return Err(invalid("pseudo-MC required for conditional initial Gamma").into());
        };
        let times = lv.plan.time_grid().nodes();
        if times.len() < 2 {
            return Err(invalid("positive first simulation interval required").into());
        }
        let variance = lv.grid.interpolate(0., 0.)?.value;
        let first_variance = PositiveF64::new(variance * times[1], "initial_measure_variance")
            .map_err(RiskConfigError::from)?
            .get();
        let sd = first_variance.sqrt();
        let sigma_dt = variance.sqrt() * times[1];
        let bins = boundaries.len() + 1;
        let offset = 3 * bins;
        let components = offset + 10;
        let width = lv.grid.log_moneyness_nodes().len();
        let sample = |z: &[f64]| -> Result<Vec<f64>, MonteCarloError> {
            let payoff = self.conditional_initial_payoff(lv, z)?;
            let mut out = vec![0.; components];
            let mut masses = vec![NeumaierSum::new(); bins];
            let mut vegas = vec![NeumaierSum::new(); bins];
            let mut price = NeumaierSum::new();
            let mut total_mass = NeumaierSum::new();
            let mut total_vega = NeumaierSum::new();
            price.add(payoff.cash());
            price.add(payoff.slope() * self.spot);
            for a in payoff.gamma().atoms() {
                let d1 = -log_ratio(a.strike(), self.spot) / sd + 0.5 * sd;
                let d2 = d1 - sd;
                // Use put-call parity on the in-the-money side to avoid
                // subtracting two large near-certain exercise probabilities.
                let call = if self.spot >= a.strike() {
                    self.spot - a.strike() + a.strike() * cdf(-d2) - self.spot * cdf(-d1)
                } else {
                    self.spot * cdf(d1) - a.strike() * cdf(d2)
                };
                let vega = a.mass() * self.spot * pdf(d1) * sigma_dt / sd;
                let bin = boundaries.partition_point(|&b| b <= a.strike());
                masses[bin].add(a.mass());
                vegas[bin].add(vega);
                total_mass.add(a.mass());
                total_vega.add(vega);
                price.add(a.mass() * call);
            }
            for i in 0..bins {
                out[i] = masses[i].total();
                out[bins + i] = vegas[i].total();
            }
            let path = lv.plan.evolve_path(&lv.grid, self.spot, z)?;
            let (native_price, seeds) =
                self.chain_discounted_payoff_and_seeds(lv, path.states())?;
            let ad = path.reverse_state_adjoints(&seeds, lv.grid.values().len(), width)?;
            let mut row = vec![0.; width];
            ad.step_adjoints()[0].accumulate_local_volatility_nodes(&mut row)?;
            let native_vega = row.into_iter().collect::<NeumaierSum>().total();
            out[offset..].copy_from_slice(&[
                payoff.cash(),
                payoff.slope(),
                total_mass.total(),
                total_vega.total(),
                native_vega,
                total_vega.total() - native_vega,
                price.total(),
                native_price,
                price.total() - native_price,
                payoff.value(path.states()[1])? - native_price,
            ]);
            for &v in &out {
                FiniteF64::new(v, "initial_measure_sample").map_err(RiskConfigError::from)?;
            }
            Ok(out)
        };
        let count = engine.independent_sampling_units().get();
        let antithetic = engine.variance_reduction().antithetic();
        let stats = DeterministicExecutor::new(self.execution_policy)?
            .try_map_reduce_statistics_vector(
                count,
                components,
                |i, out| -> Result<(), MonteCarloError> {
                    let z = lv.plan.path_shocks(
                        engine.master_seed(),
                        i,
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
                    for j in 0..bins {
                        out[2 * bins + j] = f64::from(out[j] != 0.);
                    }
                    Ok(())
                },
            )?;
        let m = stats
            .into_iter()
            .map(|s| super::vegakt_steps::moment(s, count))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(VegaKtInitialMeasureDiagnostic {
            policy_label: "flat_variance_conditional_initial_gamma_measure_v1",
            request_fingerprint: self.request_fingerprint(),
            plan_fingerprint: self.plan_fingerprint(),
            maturity: times[1],
            first_step_variance: first_variance,
            gamma_bin_boundaries: boundaries,
            gamma_bin_masses: m[..bins].to_vec(),
            gamma_bin_first_step_vegas: m[bins..2 * bins].to_vec(),
            gamma_bin_nonzero_sampling_units: m[2 * bins..offset]
                .iter()
                .map(|v| (v.mean() * count as f64).round() as u64)
                .collect(),
            independent_sampling_units: count,
            antithetic,
            cash: m[offset],
            slope: m[offset + 1],
            total_gamma_mass: m[offset + 2],
            recovered_initial_local_vega: m[offset + 3],
            native_initial_local_vega: m[offset + 4],
            initial_local_vega_difference: m[offset + 5],
            conditional_price: m[offset + 6],
            native_price: m[offset + 7],
            price_difference: m[offset + 8],
            payoff_reconstruction_difference: m[offset + 9],
        })
    }
}
