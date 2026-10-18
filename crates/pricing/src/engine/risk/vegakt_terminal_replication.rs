//! Exact paper equation-(9) risk of a statically replicable terminal payoff.
//! This is market-IV risk, not the derivative of an uncalibrated simulation grid.
use crate::engine::plan::simulation::SimulationPlan;
use crate::risk::{VegaKtBucketEstimate, vega_kt_projection_from_parts, vega_kt_report};
use crate::{MonteCarloError, VegaKtResult};

pub(in crate::engine) const TERMINAL_REPLICATION_LABEL: &str = "terminal_payoff_replication_iv_v1";

impl SimulationPlan {
    pub(in crate::engine) fn compile_terminal_replication_vega_kt(
        &self,
    ) -> Result<Option<VegaKtResult>, MonteCarloError> {
        let Some(lv) = self.local_volatility.as_ref() else {
            return Ok(None);
        };
        let Some(kt) = lv.vega_kt.as_ref() else {
            return Ok(None);
        };
        if self.terminal_vega_kt.is_some()
            || self.bass.is_some()
            || self.early_exercise.is_some()
            || self
                .continuous_barrier
                .as_ref()
                .is_some_and(|barrier| barrier.resolved_survival().is_none())
            || self.payoff_smoothing.is_some()
            || lv.dividends.is_some()
            || lv
                .node_affine_coordinates
                .iter()
                .any(|c| c.a() != 0. || c.b() != 1.)
            || !kt.basis.maturity_nodes().contains(&self.time)
        {
            return Ok(None);
        }
        // Do not use the exact market-risk path to conceal an inconsistent flat
        // model rejected by the existing flat path's cumulative-IV check.
        let width = lv.grid.log_moneyness_nodes().len();
        if lv
            .grid
            .values()
            .chunks_exact(width)
            .all(|r| r.iter().all(|&v| v == r[0]))
        {
            let times = lv.plan.time_grid().nodes();
            let bw = kt.basis.log_moneyness_nodes().len();
            for (i, &t) in kt.basis.maturity_nodes().iter().enumerate() {
                if t > self.time {
                    break;
                }
                let mut total = pricing_numerics::NeumaierSum::new();
                for pair in times.windows(2) {
                    let dt = (t.min(pair[1]) - pair[0]).max(0.);
                    total.add(lv.grid.interpolate(pair[0], 0.)?.value * dt);
                }
                let sigma = (total.total() / t).sqrt();
                if kt.basis.implied_volatilities()[i * bw..(i + 1) * bw]
                    .iter()
                    .any(|&v| (v - sigma).abs() > 1e-10 * v.abs().max(sigma.abs()))
                {
                    return Ok(None);
                }
            }
        }
        let Some(payoff) = super::vegakt_terminal::try_terminal_payoff_gamma_measure(
            &self.payoff,
            self.underlying,
            self.expiry,
        )?
        else {
            return Ok(None);
        };
        // cash + slope*S_T + sum q*(S_T-K)+ holds path by path for every LV.
        // Only the signed call masses have market-IV risk. Deterministic carry
        // changes forward and discount; no local-Gamma inversion is required.
        let forward = *lv
            .plan
            .forward_normalizers()
            .last()
            .expect("terminal forward");
        let measure = payoff
            .gamma()
            .project_terminal(&kt.basis, self.time, forward)?;
        let p = measure.position_risk();
        let raw: Vec<_> = p.raw_buckets().iter().map(|v| self.discount * v).collect();
        let mut stats = p.reporting_stats();
        stats.left_edge_sensitivity *= self.discount;
        stats.right_edge_sensitivity *= self.discount;
        let projection = vega_kt_projection_from_parts(
            raw.clone(),
            0.,
            self.discount * p.pre_projection(),
            stats,
        )?;
        // No Monte Carlo samples are manufactured for exact replication.
        // Null sample moments mean not applicable, not zero estimated variance.
        let estimates = raw
            .iter()
            .map(|&v| VegaKtBucketEstimate {
                raw_mean: v,
                market_scaled_mean: 0.01 * v,
                sample_variance: None,
                price_covariance: None,
            })
            .collect();
        let covariance = if kt.full_bucket_covariance {
            Some(vec![
                None;
                super::vegakt::full_bucket_covariance_len(raw.len())?
            ])
        } else {
            None
        };
        let density = kt
            .density_rows
            .iter()
            .find(|r| r.maturity().get() == self.time)
            .expect("reporting terminal maturity");
        let mut report = vega_kt_report(&kt.basis, density, projection, estimates, covariance)?;
        report.policy_label = TERMINAL_REPLICATION_LABEL;
        report.truncation_order = "not_applicable";
        report.residual_diagnostics.active_domain.start_index = 0;
        report.residual_diagnostics.active_domain.end_index =
            kt.basis.log_moneyness_nodes().len() - 1;
        report.residual_diagnostics.excluded_probability_mass = 0.;
        Ok(Some(VegaKtResult::try_from(&report)?))
    }
}
