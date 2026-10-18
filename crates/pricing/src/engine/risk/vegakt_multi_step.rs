//! Last-row Gamma with recursively integrated actual source marginals.
use crate::MonteCarloError;
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::EngineConfig;
use crate::risk::{
    RiskConfigError, VegaKtConvolutionPlan, VegaKtGammaRecoveryPolicy, VegaKtGammaRecoveryStatus,
    VegaKtMultiStepTerminalGammaAudit,
};
fn invalid(s: &'static str) -> MonteCarloError {
    RiskConfigError::VegaKtChainInput { quantity: s }.into()
}
impl SimulationPlan {
    pub(in crate::engine) fn evaluate_vega_kt_multi_step_terminal_gamma_audit(
        &self,
        strikes: Vec<f64>,
        policy: &VegaKtGammaRecoveryPolicy,
        directions: Vec<Vec<f64>>,
    ) -> Result<VegaKtMultiStepTerminalGammaAudit, MonteCarloError> {
        let lv = self.state_only_path_gamma_runtime()?;
        let times = lv.plan.time_grid().nodes();
        let nodes = lv.grid.log_moneyness_nodes();
        let m = nodes.len();
        let n = strikes.len();
        let count_steps = times.len() - 1;
        if !(2..=16).contains(&count_steps) || lv.grid.time_nodes() != times {
            return Err(invalid(
                "2..16 matching simulation and variance-grid intervals required",
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
        let rows: Vec<Vec<f64>> = lv
            .grid
            .values()
            .chunks_exact(m)
            .take(count_steps)
            .map(|r| r.to_vec())
            .collect();
        let (base, accuracy) = VegaKtConvolutionPlan::from_multi_transition_variance_source(
            self.spot,
            times,
            nodes,
            &rows,
            strikes.clone(),
            policy.candidate_regularizations()[0],
        )?;
        let operators = policy
            .candidate_regularizations()
            .iter()
            .map(|&lambda| {
                let op = base.with_unconstrained_regularization(lambda)?;
                if directions.is_empty() {
                    Ok(op)
                } else {
                    op.conserve_local_vega_directions(directions.clone(), false)
                }
            })
            .collect::<Result<Vec<_>, RiskConfigError>>()?;
        let antithetic = engine.variance_reduction().antithetic();
        let evaluate = |op: &VegaKtConvolutionPlan, start, samples| {
            self.multi_step_variance_gamma_candidate(
                op,
                &directions,
                (count_steps - 1, start, samples),
                policy,
            )
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
        Ok(VegaKtMultiStepTerminalGammaAudit {
            request_fingerprint: self.request_fingerprint(),
            plan_fingerprint: self.plan_fingerprint(),
            policy: policy.clone(),
            variance_conservation_directions: directions,
            status,
            validation_sampling_units: count,
            antithetic,
            pilot_index_start: count,
            pilot_index_end_exclusive: end,
            source_maturity: times[count_steps - 1],
            terminal_maturity: times[count_steps],
            time_nodes: times.to_vec(),
            variance_rows: rows,
            source_log_nodes: nodes.to_vec(),
            gamma_strikes: strikes,
            operator_accuracy: accuracy,
            forward_matrix: operators[0].forward_matrix().to_vec(),
            pilot_candidates,
            selected_regularization: choice.map(|i| operators[i].relative_regularization()),
            validation,
            estimate,
        })
    }
}
