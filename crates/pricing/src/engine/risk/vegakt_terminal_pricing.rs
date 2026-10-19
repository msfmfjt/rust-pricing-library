//! Deterministic terminal-measure boundary for a restricted native pricing plan.
use super::vegakt_measure::invalid;
use crate::MonteCarloError;
use crate::core::FiniteF64;
use crate::engine::plan::simulation::SimulationPlan;
use crate::risk::{
    RiskConfigError, VEGA_KT_TERMINAL_MEASURE_LABEL, VegaKtTerminalDiagnostic,
    VegaKtTerminalLocalRow, VegaKtTerminalPayoff,
};
use pricing_numerics::NeumaierSum;

pub(in crate::engine) struct TerminalMeasureModel {
    pub payoff: VegaKtTerminalPayoff,
    pub variances: Vec<f64>,
    pub cumulative: Vec<f64>,
}

impl SimulationPlan {
    /// Scope selection is separate from numerical evaluation: unsupported inputs
    /// return None, while a numerical failure in an eligible model is an error.
    pub(in crate::engine) fn terminal_measure_model(
        &self,
    ) -> Result<Option<TerminalMeasureModel>, MonteCarloError> {
        let Some(lv) = self.local_volatility.as_ref() else {
            return Ok(None);
        };
        if self.bass.is_some()
            || self.early_exercise.is_some()
            || self
                .continuous_barrier
                .as_ref()
                .is_some_and(|barrier| barrier.resolved_survival().is_none())
            || self.payoff_smoothing.is_some()
        {
            return Ok(None);
        }
        if self.discount != 1.
            || lv.dividends.is_some()
            || lv
                .node_affine_coordinates
                .iter()
                .any(|c| c.a() != 0. || c.b() != 1.)
            || lv
                .plan
                .forward_normalizers()
                .iter()
                .any(|&f| f != self.spot)
        {
            return Ok(None);
        }
        let width = lv.grid.log_moneyness_nodes().len();
        if lv
            .grid
            .values()
            .chunks_exact(width)
            .any(|row| row.iter().any(|&x| x != row[0]))
        {
            return Ok(None);
        }
        let Some(report) = lv.vega_kt.as_ref() else {
            return Ok(None);
        };
        let times = lv.plan.time_grid().nodes();
        // Reporting dates need not be simulation dates. Retain an exact terminal
        // quote: interpolating an off-grid expiry would introduce a different
        // implied-volatility term-structure assumption.
        if report.basis.maturity_nodes().last().copied() != Some(self.time)
            || times.last().copied() != Some(self.time)
        {
            return Ok(None);
        }
        let mut cumulative = vec![0.];
        let mut variances = Vec::new();
        for (i, pair) in times.windows(2).enumerate() {
            let variance = lv.grid.interpolate(pair[0], 0.)?.value;
            variances.push(variance);
            let total = cumulative[i] + variance * (pair[1] - pair[0]);
            cumulative.push(total);
        }
        let bw = report.basis.log_moneyness_nodes().len();
        for (i, &maturity) in report.basis.maturity_nodes().iter().enumerate() {
            // Integrate the actual left-endpoint Log-Euler variance, including
            // partial intervals for reporting dates between simulation nodes.
            // At an exact node use the stored sum to preserve existing bits.
            let total = match times.binary_search_by(|t| t.total_cmp(&maturity)) {
                Ok(j) => cumulative[j],
                Err(j) => cumulative[j - 1] + variances[j - 1] * (maturity - times[j - 1]),
            };
            let sigma = (total / maturity).sqrt();
            if report.basis.implied_volatilities()[i * bw..(i + 1) * bw]
                .iter()
                .any(|&v| (v - sigma).abs() > 1e-10 * v.abs().max(sigma.abs()))
            {
                return Ok(None);
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
        Ok(Some(TerminalMeasureModel {
            payoff,
            variances,
            cumulative,
        }))
    }

    pub(in crate::engine) fn evaluate_vega_kt_terminal_diagnostic(
        &self,
    ) -> Result<VegaKtTerminalDiagnostic, MonteCarloError> {
        let TerminalMeasureModel {
            payoff,
            variances,
            cumulative,
        } = self
            .terminal_measure_model()?
            .ok_or_else(|| invalid("outside restricted terminal-measure model/payoff scope"))?;
        let lv = self
            .local_volatility
            .as_ref()
            .expect("validated terminal model");
        let report = lv
            .vega_kt
            .as_ref()
            .expect("validated terminal reporting basis");
        let times = lv.plan.time_grid().nodes();
        let total_variance = cumulative[cumulative.len() - 1];
        let projection = payoff
            .gamma()
            .project_terminal(&report.basis, self.time, self.spot)?;
        // Flat-variance Markov propagation of this same terminal measure obeys
        // the semigroup law. Thus H_i Gamma_(i+1) - Gamma_i is identically zero
        // at every intermediate expiry; only the terminal hedge measure remains.
        // This statement does not apply to state/path-dependent observation seeds.
        let mut remaining = vec![0.; times.len()];
        for i in (0..variances.len()).rev() {
            remaining[i] = remaining[i + 1] + variances[i] * (times[i + 1] - times[i]);
        }
        let mut local_rows = Vec::new();
        let mut local_total = NeumaierSum::new();
        for i in 0..variances.len() {
            let sigma = variances[i].sqrt();
            let expected = payoff.gamma().expected_source_hat_vegas(
                self.spot,
                cumulative[i],
                remaining[i],
                sigma * (times[i + 1] - times[i]),
                lv.grid.log_moneyness_nodes(),
            )?;
            for &v in &expected {
                local_total.add(v);
            }
            local_rows.push(VegaKtTerminalLocalRow {
                source_maturity: times[i],
                destination_maturity: times[i + 1],
                source_total_variance: cumulative[i],
                interval_volatility: sigma,
                expected_source_hat_vegas: expected,
            });
        }
        Ok(VegaKtTerminalDiagnostic {
            policy_label: VEGA_KT_TERMINAL_MEASURE_LABEL,
            payoff,
            reporting_basis: report.basis.clone(),
            projection,
            source_log_moneyness_nodes: lv.grid.log_moneyness_nodes().to_vec(),
            local_rows,
            total_variance,
            expected_local_parallel_vega: FiniteF64::new(
                local_total.total(),
                "terminal_expected_local_vega",
            )
            .map_err(RiskConfigError::from)?
            .get(),
            request_fingerprint: self.request_fingerprint(),
            plan_fingerprint: self.plan_fingerprint(),
        })
    }
}
