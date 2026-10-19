//! Compiled path-dependent payoff AD -> full conditional operator inversion.
use crate::MonteCarloError;
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::{DeterministicExecutor, EngineConfig, RandomDomain};
use crate::product::CompiledOpcode;
use crate::risk::{
    RiskConfigError, VegaKtConvolutionDiagnostic, VegaKtConvolutionPlan, VegaKtConvolutionRow,
};
use pricing_numerics::NeumaierSum;

fn invalid(quantity: &'static str) -> MonteCarloError {
    RiskConfigError::VegaKtChainInput { quantity }.into()
}
impl SimulationPlan {
    pub(super) fn flat_path_gamma_runtime(
        &self,
    ) -> Result<&crate::engine::plan::simulation::LocalVolRuntime, MonteCarloError> {
        let lv = self.state_only_path_gamma_runtime()?;
        let width = lv.grid.log_moneyness_nodes().len();
        if lv
            .grid
            .values()
            .chunks_exact(width)
            .any(|r| r.iter().any(|&v| v != r[0]))
        {
            return Err(invalid("spatially flat variance required"));
        }
        Ok(lv)
    }

    pub(super) fn state_only_path_gamma_runtime(
        &self,
    ) -> Result<&crate::engine::plan::simulation::LocalVolRuntime, MonteCarloError> {
        let lv = self
            .local_volatility
            .as_ref()
            .ok_or_else(|| invalid("local volatility required"))?;
        if self.bass.is_some()
            || self.early_exercise.is_some()
            || self.continuous_barrier.is_some()
            || self.payoff_smoothing.is_some()
            || lv.dividends.is_some()
            || self.discount != 1.
            || lv
                .plan
                .forward_normalizers()
                .iter()
                .any(|&f| f != self.spot)
            || lv
                .node_affine_coordinates
                .iter()
                .any(|c| c.a() != 0. || c.b() != 1.)
        {
            return Err(invalid("flat-space zero-carry state-only scope required"));
        }
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
                    | CompiledOpcode::Negate { .. }
            )
        }) {
            return Err(invalid(
                "continuous payoff graph required; indicator distributional derivatives unsupported",
            ));
        }
        Ok(lv)
    }

    pub(in crate::engine) fn evaluate_vega_kt_gamma_recovery_diagnostic(
        &self,
        gamma_strikes: Vec<f64>,
        relative_regularization: f64,
    ) -> Result<VegaKtConvolutionDiagnostic, MonteCarloError> {
        let EngineConfig::PseudoMonteCarlo(engine) = self.engine else {
            return Err(invalid("pseudo-MC required for Gamma recovery"));
        };
        self.convolution_gamma_in_range(
            gamma_strikes,
            relative_regularization,
            0,
            engine.independent_sampling_units().get(),
        )
    }

    /// Internal disjoint-counter pilot; the caller must disclose this range.
    pub(super) fn convolution_gamma_in_range(
        &self,
        gamma_strikes: Vec<f64>,
        relative_regularization: f64,
        sampling_index_start: u64,
        count: u64,
    ) -> Result<VegaKtConvolutionDiagnostic, MonteCarloError> {
        self.convolution_gamma_in_range_with_conservation(
            gamma_strikes,
            relative_regularization,
            sampling_index_start,
            count,
            false,
        )
    }

    pub(super) fn convolution_gamma_in_range_with_conservation(
        &self,
        gamma_strikes: Vec<f64>,
        relative_regularization: f64,
        sampling_index_start: u64,
        count: u64,
        preserve_local_vega_total: bool,
    ) -> Result<VegaKtConvolutionDiagnostic, MonteCarloError> {
        self.convolution_gamma_in_range_with_constraints(
            gamma_strikes,
            relative_regularization,
            sampling_index_start,
            count,
            preserve_local_vega_total,
            &[],
        )
    }

    pub(super) fn convolution_gamma_in_range_with_constraints(
        &self,
        gamma_strikes: Vec<f64>,
        relative_regularization: f64,
        sampling_index_start: u64,
        count: u64,
        preserve_local_vega_total: bool,
        directions: &[Vec<Vec<f64>>],
    ) -> Result<VegaKtConvolutionDiagnostic, MonteCarloError> {
        if count == 0 || sampling_index_start.checked_add(count).is_none() {
            return Err(invalid("invalid Gamma sampling counter range"));
        }
        let lv = self.flat_path_gamma_runtime()?;
        let EngineConfig::PseudoMonteCarlo(engine) = self.engine else {
            return Err(invalid(
                "pseudo-MC required; RQMC points are not independent samples",
            ));
        };
        let nodes = lv.grid.log_moneyness_nodes();
        let width = nodes.len();
        let times = lv.plan.time_grid().nodes();
        if times.len() < 3 {
            return Err(invalid("at least two simulation intervals required"));
        }
        let direction_count = directions.first().map_or(0, Vec::len);
        if !directions.is_empty()
            && (directions.len() != times.len() - 2
                || directions
                    .iter()
                    .any(|r| r.len() != direction_count || r.iter().any(|d| d.len() != width)))
        {
            return Err(invalid(
                "Gamma conservation directions must match recovery times and source nodes",
            ));
        }
        let n = gamma_strikes.len();
        if !(2..=129).contains(&n) || !(2..=129).contains(&width) {
            return Err(invalid("2..129 recovery and source nodes required"));
        }
        let base_row_width = n + 3 * width + 3;
        let row_width = base_row_width + 3 * direction_count;
        let components = (times.len() - 2)
            .checked_mul(row_width)
            .and_then(|n| n.checked_add(1))
            .filter(|&n| n <= 131072)
            .ok_or_else(|| invalid("Gamma diagnostic component budget exceeded"))?;
        (times.len() - 2)
            .checked_mul(width * n)
            .filter(|&n| n <= 1_000_000)
            .ok_or_else(|| invalid("Gamma diagnostic operator budget exceeded"))?;
        let mut cumulative = 0.;
        let mut operators = Vec::new();
        let mut variances = Vec::new();
        for (i, pair) in times.windows(2).enumerate() {
            let variance = lv.grid.interpolate(pair[0], 0.)?.value;
            let dt = pair[1] - pair[0];
            if i > 0 {
                let op = VegaKtConvolutionPlan::new(
                    self.spot,
                    cumulative,
                    variance * dt,
                    variance.sqrt() * dt,
                    nodes,
                    gamma_strikes.clone(),
                    relative_regularization,
                )?;
                operators.push(if !directions.is_empty() {
                    op.conserve_local_vega_directions(
                        directions[i - 1].clone(),
                        preserve_local_vega_total,
                    )?
                } else if preserve_local_vega_total {
                    op.conserve_local_vega_total()?
                } else {
                    op
                });
                variances.push((cumulative, variance * dt));
            }
            cumulative += variance * dt;
        }
        // Every recovered/residual component is a linear transform of one path's
        // full AD vector. Pair first, then compute moments: covariance is retained.
        let sample = |z: &[f64]| -> Result<Vec<f64>, MonteCarloError> {
            let path = lv.plan.evolve_path(&lv.grid, self.spot, z)?;
            let (_, seeds) = self.chain_discounted_payoff_and_seeds(lv, path.states())?;
            let ad = path.reverse_state_adjoints(&seeds, lv.grid.values().len(), width)?;
            let mut out = Vec::with_capacity(components);
            let mut row = vec![0.; width];
            ad.step_adjoints()[0].accumulate_local_volatility_nodes(&mut row)?;
            out.push(row.iter().copied().collect::<NeumaierSum>().total());
            for (i, op) in operators.iter().enumerate() {
                row.fill(0.);
                ad.step_adjoints()[i + 1].accumulate_local_volatility_nodes(&mut row)?;
                let gamma = op.recover(&row)?;
                let reconstructed = op.reconstruct(&gamma)?;
                let difference: Vec<_> =
                    reconstructed.iter().zip(&row).map(|(a, b)| a - b).collect();
                out.extend(gamma);
                out.extend(&row);
                out.extend(&difference);
                out.extend(std::iter::repeat_n(0., width));
                out.extend([
                    row.iter().copied().collect::<NeumaierSum>().total(),
                    reconstructed
                        .iter()
                        .copied()
                        .collect::<NeumaierSum>()
                        .total(),
                    difference.iter().copied().collect::<NeumaierSum>().total(),
                ]);
                for values in [&row, &reconstructed, &difference] {
                    for direction in op.conservation_directions() {
                        out.push(
                            values
                                .iter()
                                .zip(direction)
                                .map(|(a, b)| a * b)
                                .collect::<NeumaierSum>()
                                .total(),
                        );
                    }
                }
            }
            Ok(out)
        };
        let antithetic = engine.variance_reduction().antithetic();
        let stats = DeterministicExecutor::new(self.execution_policy)?
            .try_map_reduce_statistics_vector(
                count,
                components,
                |i, output| -> Result<(), MonteCarloError> {
                    let z = lv.plan.path_shocks(
                        engine.master_seed(),
                        sampling_index_start + i,
                        RandomDomain::Valuation,
                        engine.variance_reduction().brownian_bridge(),
                    )?;
                    let primary = sample(&z)?;
                    if antithetic {
                        let mate = sample(&z.iter().map(|z| -z).collect::<Vec<_>>())?;
                        for ((v, a), b) in output.iter_mut().zip(primary).zip(mate) {
                            *v = 0.5 * a + 0.5 * b;
                        }
                    } else {
                        output.copy_from_slice(&primary);
                    }
                    for row in output[1..].chunks_exact_mut(row_width) {
                        for j in 0..width {
                            row[n + 2 * width + j] = f64::from(row[n + j] != 0.);
                        }
                    }
                    Ok(())
                },
            )?;
        let moments = stats
            .into_iter()
            .map(|s| super::vegakt_steps::moment(s, count))
            .collect::<Result<Vec<_>, _>>()?;
        let rows = operators
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let m = &moments[1 + i * row_width..1 + (i + 1) * row_width];
                VegaKtConvolutionRow {
                    source_maturity: times[i + 1],
                    recovered_maturity: times[i + 2],
                    source_total_variance: variances[i].0,
                    step_variance: variances[i].1,
                    operator_scale: op.operator_scale,
                    inverse_noise_gain_bound: op.inverse_noise_gain_bound,
                    resolution_trace: op.resolution_trace,
                    gamma_masses: m[..n].to_vec(),
                    local_vega: m[n..n + width].to_vec(),
                    local_vega_nonzero_sampling_units: m[n + 2 * width..n + 3 * width]
                        .iter()
                        .map(|v| (v.mean() * count as f64).round() as u64)
                        .collect(),
                    reconstruction_difference: m[n + width..n + 2 * width].to_vec(),
                    local_vega_total: m[base_row_width - 3],
                    reconstructed_total: m[base_row_width - 2],
                    total_difference: m[base_row_width - 1],
                    local_vega_directions: m[base_row_width..base_row_width + direction_count]
                        .to_vec(),
                    reconstructed_directions: m
                        [base_row_width + direction_count..base_row_width + 2 * direction_count]
                        .to_vec(),
                    direction_differences: m[base_row_width + 2 * direction_count..].to_vec(),
                }
            })
            .collect();
        Ok(VegaKtConvolutionDiagnostic {
            policy_label: if !directions.is_empty() {
                "flat_variance_direction_constrained_convolution_gamma_diagnostic_v1"
            } else if preserve_local_vega_total {
                "flat_variance_conservative_convolution_gamma_diagnostic_v1"
            } else {
                "flat_variance_convolution_gamma_diagnostic_v1"
            },
            request_fingerprint: self.request_fingerprint(),
            plan_fingerprint: self.plan_fingerprint(),
            gamma_strikes,
            source_log_nodes: nodes.to_vec(),
            relative_regularization,
            preserves_local_vega_total: preserve_local_vega_total,
            conservation_directions: directions.to_vec(),
            independent_sampling_units: count,
            antithetic,
            unrecovered_initial_local_vega: moments[0],
            rows,
        })
    }
}
