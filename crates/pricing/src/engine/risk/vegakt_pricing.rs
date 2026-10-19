//! Restricted paper-chain evaluation using the compiled pricing payoff and RNG.
use crate::MonteCarloError;
use crate::engine::plan::simulation::{LocalVolRuntime, ReportingIvSurface, SimulationPlan};
use crate::mc::{DeterministicExecutor, EngineConfig, RandomDomain};
use crate::models::LocalVolatilityReportingBasis;
use crate::product::CompiledOpcode;
use crate::risk::{
    RiskConfigError, VegaKtChainDiagnostic, VegaKtChainPlan, VegaKtChainSample,
    VegaKtInitialGammaPlan, VegaKtStepRecoveryPlan, VegaKtStepSamplingUnit,
};

fn unsupported(quantity: &'static str) -> MonteCarloError {
    RiskConfigError::VegaKtChainInput { quantity }.into()
}

impl SimulationPlan {
    pub(in crate::engine) fn evaluate_vega_kt_chain_diagnostic(
        &self,
        initial_strikes: Vec<f64>,
        relative_bump: f64,
        subdivisions: usize,
    ) -> Result<VegaKtChainDiagnostic, MonteCarloError> {
        let lv = self
            .local_volatility
            .as_ref()
            .ok_or_else(|| unsupported("local-volatility pricing model required"))?;
        if self.early_exercise.is_some()
            || self.continuous_barrier.is_some()
            || self.payoff_smoothing.is_some()
        {
            return Err(unsupported(
                "early exercise, continuous barrier and smoothing are unsupported",
            ));
        }
        // Pathwise state seeds do not recover distributional derivatives of an
        // indicator. Keep the supported opcode set explicit for future products.
        if self.payoff.opcodes().iter().any(|op| {
            !matches!(
                op,
                CompiledOpcode::Literal { .. }
                    | CompiledOpcode::TerminalSpot { .. }
                    | CompiledOpcode::Add { .. }
                    | CompiledOpcode::Subtract { .. }
                    | CompiledOpcode::Multiply { .. }
                    | CompiledOpcode::Divide { .. }
                    | CompiledOpcode::Minimum { .. }
                    | CompiledOpcode::Maximum { .. }
            )
        }) {
            return Err(unsupported(
                "continuous state-only payoff graph required; indicators and pre-dividend observations are unsupported",
            ));
        }
        if self.discount != 1.
            || lv.dividends.is_some()
            || lv
                .node_affine_coordinates
                .iter()
                .any(|c| c.a() != 0. || c.b() != 1.)
        {
            return Err(unsupported(
                "unit discount and identity dividend coordinates required",
            ));
        }
        let EngineConfig::PseudoMonteCarlo(engine) = self.engine else {
            return Err(unsupported(
                "independent pseudo-MC units required; RQMC is unsupported",
            ));
        };
        let report = lv
            .vega_kt
            .as_ref()
            .ok_or_else(|| unsupported("VegaKT reporting request required"))?;
        if report.full_bucket_covariance {
            return Err(unsupported(
                "full bucket covariance is not provided by the chain diagnostic",
            ));
        }
        let basis = LocalVolatilityReportingBasis::new(
            report.basis.maturity_nodes().to_vec(),
            report.basis.log_moneyness_nodes().to_vec(),
            report.basis.implied_volatilities().to_vec(),
        )
        .map_err(RiskConfigError::from)?;
        // Validate the original model before constructing an equivalent flat
        // auxiliary grid. This grid changes only recovery resolution, never the
        // report coordinates, valuation RNG, payoffs or acceptance tolerances.
        VegaKtInitialGammaPlan::new(
            lv.plan.clone(),
            lv.grid.clone(),
            initial_strikes.clone(),
            relative_bump,
        )?;
        let first_variance = lv.grid.interpolate(0., 0.)?.value * lv.plan.time_grid().nodes()[1];
        let maximum_spacing = (0.5 * first_variance.sqrt()).min(0.5 * relative_bump);
        let recovery_nodes =
            crate::risk::refine_vega_kt_log_nodes(lv.grid.log_moneyness_nodes(), maximum_spacing)?;
        let width = lv.grid.log_moneyness_nodes().len();
        let recovery_values = lv
            .grid
            .values()
            .chunks_exact(width)
            .flat_map(|row| std::iter::repeat_n(row[0], recovery_nodes.len()))
            .collect();
        let recovery_grid = crate::market::LocalVarianceGrid::new(
            lv.grid.time_nodes().to_vec(),
            recovery_nodes.clone(),
            recovery_values,
            lv.grid.floor(),
            lv.grid.cap(),
        )?;
        let refined_initial = crate::risk::refine_vega_kt_log_nodes(
            &initial_strikes.iter().map(|s| s.ln()).collect::<Vec<_>>(),
            maximum_spacing,
        )?
        .into_iter()
        .map(f64::exp)
        .collect::<Vec<_>>();
        let initial = VegaKtInitialGammaPlan::new(
            lv.plan.clone(),
            recovery_grid.clone(),
            refined_initial.clone(),
            relative_bump,
        )?;
        let threshold = report.density_rows[0]
            .active_domain()
            .relative_threshold()
            .get();
        let step = VegaKtStepRecoveryPlan::new(
            lv.plan.clone(),
            recovery_grid,
            &ReportingIvSurface::new(&basis),
            threshold,
        )?;
        let chain = VegaKtChainPlan::new(initial, step, report.basis.clone(), subdivisions)?
            .with_integrated_density_recovery()?;
        let antithetic = engine.variance_reduction().antithetic();
        let brownian_bridge = engine.variance_reduction().brownian_bridge();
        let kind = if antithetic {
            VegaKtStepSamplingUnit::AntitheticPair
        } else {
            VegaKtStepSamplingUnit::IndependentPath
        };
        let executor = DeterministicExecutor::new(self.execution_policy)?;
        let estimate = chain.estimate(
            &executor,
            engine.independent_sampling_units().get(),
            kind,
            |i| {
                let shocks = lv.plan.path_shocks(
                    engine.master_seed(),
                    i,
                    RandomDomain::Valuation,
                    brownian_bridge,
                )?;
                let sample = |z: &[f64]| {
                    chain.sample_from_shocks(z, |states| {
                        self.chain_discounted_payoff_and_seeds(lv, states)
                    })
                };
                let primary = sample(&shocks)?;
                if antithetic {
                    Ok(VegaKtChainSample::antithetic_pair(
                        primary,
                        sample(&shocks.iter().map(|v| -v).collect::<Vec<_>>())?,
                    )?)
                } else {
                    Ok(primary)
                }
            },
        )?;
        Ok(VegaKtChainDiagnostic {
            estimate,
            request_fingerprint: self.request_fingerprint(),
            plan_fingerprint: self.plan_fingerprint(),
            initial_strikes: refined_initial.into_boxed_slice(),
            requested_initial_strikes: initial_strikes.into_boxed_slice(),
            recovery_log_moneyness_nodes: recovery_nodes.into_boxed_slice(),
            maximum_recovery_log_spacing: maximum_spacing,
            relative_bump,
            master_seed: engine.master_seed(),
            brownian_bridge,
            evaluated_paths: engine.evaluated_paths(),
        })
    }

    pub(super) fn chain_discounted_payoff_and_seeds(
        &self,
        lv: &LocalVolRuntime,
        states: &[f64],
    ) -> Result<(f64, Vec<f64>), MonteCarloError> {
        let observations = self.local_vol_state_observations(lv, states, self.spot)?;
        let payoff = self.payoff.evaluate_single_with_observation_adjoints(
            |underlying, date| {
                if underlying != self.underlying {
                    return None;
                }
                self.observation_dates
                    .iter()
                    .position(|d| *d == Some(date))
                    .map(|i| observations[i].post_spot)
            },
            |_, _| None,
        )?;
        let mut seeds = vec![0.; states.len()];
        for a in &payoff.terminal_adjoints {
            let i = self
                .observation_dates
                .iter()
                .position(|d| *d == Some(a.observation_date))
                .ok_or_else(|| {
                    unsupported("payoff observation is not mapped to a simulation node")
                })?;
            if a.underlying != self.underlying {
                return Err(unsupported("payoff underlying mismatch"));
            }
            seeds[observations[i].node_index] +=
                self.discount * a.value * self.observation_affine_coordinates[i].b();
        }
        if !payoff.pre_dividend_adjoints.is_empty() {
            return Err(unsupported("pre-dividend payoff partials are unsupported"));
        }
        Ok((self.discount * payoff.value, seeds))
    }
}
