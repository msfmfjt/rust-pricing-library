//! Third transition Gamma using the actual non-Gaussian second marginal.
use crate::MonteCarloError;
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::EngineConfig;
use crate::risk::{
    RiskConfigError, VegaKtConvolutionPlan, VegaKtGammaRecoveryPolicy, VegaKtGammaRecoveryStatus,
    VegaKtThreeStepTerminalGammaAudit,
};
fn invalid(s: &'static str) -> MonteCarloError {
    RiskConfigError::VegaKtChainInput { quantity: s }.into()
}
impl SimulationPlan {
    pub(in crate::engine) fn evaluate_vega_kt_three_step_terminal_gamma_audit(
        &self,
        strikes: Vec<f64>,
        policy: &VegaKtGammaRecoveryPolicy,
        directions: Vec<Vec<f64>>,
    ) -> Result<VegaKtThreeStepTerminalGammaAudit, MonteCarloError> {
        self.three_step_gamma_audit(strikes, policy, directions, None)
    }

    pub(in crate::engine) fn evaluate_vega_kt_three_step_terminal_readout_audit(
        &self,
        strikes: Vec<f64>,
        policy: &VegaKtGammaRecoveryPolicy,
        directions: Vec<Vec<f64>>,
        readouts: Vec<Vec<f64>>,
    ) -> Result<VegaKtThreeStepTerminalGammaAudit, MonteCarloError> {
        self.three_step_gamma_audit(strikes, policy, directions, Some(readouts))
    }

    fn three_step_gamma_audit(
        &self,
        strikes: Vec<f64>,
        policy: &VegaKtGammaRecoveryPolicy,
        directions: Vec<Vec<f64>>,
        readout_directions: Option<Vec<Vec<f64>>>,
    ) -> Result<VegaKtThreeStepTerminalGammaAudit, MonteCarloError> {
        let lv = self.state_only_path_gamma_runtime()?;
        let times = lv.plan.time_grid().nodes();
        let nodes = lv.grid.log_moneyness_nodes();
        let m = nodes.len();
        let n = strikes.len();
        if times.len() != 4
            || lv.grid.time_nodes() != times
            || lv.grid.values()[..m]
                .iter()
                .any(|&v| v != lv.grid.values()[0])
        {
            return Err(invalid(
                "exactly three simulation/grid intervals and a flat first variance row required",
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
        let readouts = readout_directions.as_ref().unwrap_or(&directions);
        if readout_directions.is_some()
            && (!(1..=17).contains(&readouts.len())
                || readouts.iter().any(|r| {
                    r.len() != m
                        || r.iter().any(|v| !v.is_finite())
                        || !r.iter().map(|v| v.abs()).sum::<f64>().is_finite()
                }))
        {
            return Err(invalid(
                "1..17 finite variance readouts with source node width required",
            ));
        }
        let d = readouts.len();
        let components = 2 + n + 3 * m + 3 + 3 * d;
        let blocks = count
            .max(policy.pilot_sampling_units())
            .div_ceil(self.execution_policy.reduction_block_size().get());
        if blocks as u128 * components as u128 > 4_000_000 {
            return Err(invalid("variance-Gamma reduction storage budget exceeded"));
        }
        let first_step_variance = lv.grid.values()[0] * times[1];
        let second = lv.grid.values()[m..2 * m].to_vec();
        let third = lv.grid.values()[2 * m..3 * m].to_vec();
        let (base, accuracy) = VegaKtConvolutionPlan::from_two_transition_variance_source(
            self.spot,
            first_step_variance,
            [times[2] - times[1], times[3] - times[2]],
            nodes,
            [&second, &third],
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
        if readout_directions.is_some() {
            for op in &operators {
                for j in 0..m {
                    let mut unit = vec![0.; m];
                    unit[j] = 1.;
                    let rebuilt = op.reconstruct(&op.recover(&unit)?)?;
                    for w in readouts {
                        let norm = w.iter().map(|x| x.abs()).sum::<f64>();
                        let value = w.iter().zip(&rebuilt).map(|(a, b)| a * b).sum::<f64>();
                        if !value.is_finite() || (value - w[j]).abs() > 1e-9 * norm {
                            return Err(invalid(
                                "variance readout is not conserved by every candidate inverse",
                            ));
                        }
                    }
                }
            }
        }
        let antithetic = engine.variance_reduction().antithetic();
        let evaluate = |op: &VegaKtConvolutionPlan, start, samples| {
            self.variance_gamma_candidate(op, readouts, (2, start, samples), policy)
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
        Ok(VegaKtThreeStepTerminalGammaAudit {
            request_fingerprint: self.request_fingerprint(),
            plan_fingerprint: self.plan_fingerprint(),
            policy: policy.clone(),
            variance_conservation_directions: directions,
            variance_risk_directions: readout_directions,
            status,
            validation_sampling_units: count,
            antithetic,
            pilot_index_start: count,
            pilot_index_end_exclusive: end,
            source_maturity: times[2],
            terminal_maturity: times[3],
            first_step_variance,
            source_log_nodes: nodes.to_vec(),
            second_step_variances: second,
            third_step_variances: third,
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
