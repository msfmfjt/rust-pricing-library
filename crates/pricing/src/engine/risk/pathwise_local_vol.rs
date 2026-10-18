//! processes / local vol valuation implementation.

use crate::core::{PathIndex, PositiveF64};
use crate::engine::compile::local_vol::compile_local_vol_runtime;
use crate::engine::payoff::barrier::continuous_barrier_payoff_terms;
use crate::engine::plan::simulation::BUMP_DELTA;
use crate::engine::plan::simulation::BUMP_GAMMA;
use crate::engine::plan::simulation::DELTA;
use crate::engine::plan::simulation::ExerciseCashflowSelection;
use crate::engine::plan::simulation::GAMMA;
use crate::engine::plan::simulation::LocalVolBumpRuntimes;
use crate::engine::plan::simulation::LocalVolContinuousBarrierBridgeEvaluation;
use crate::engine::plan::simulation::LocalVolPathwise;
use crate::engine::plan::simulation::LocalVolRuntime;
use crate::engine::plan::simulation::LocalVolRuntimeInputs;
use crate::engine::plan::simulation::PATHWISE_COMPONENTS;
use crate::engine::plan::simulation::PRICE;
use crate::engine::plan::simulation::SimulationPlan;
use crate::engine::plan::simulation::VEGA;
use crate::engine::risk::local_vol_allocation::allocate_local_vol_nodes;
use crate::{MonteCarloError, ResultBuildError};

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_bump_runtimes(
        &self,
    ) -> Result<Option<LocalVolBumpRuntimes>, MonteCarloError> {
        if !self.request_delta && self.request_gamma.is_none() {
            return Ok(None);
        }
        let local_volatility =
            self.local_volatility
                .as_ref()
                .ok_or(MonteCarloError::UnsupportedModel {
                    model: "local_volatility",
                })?;
        let spot_bump = self.validation_spot_bump;
        let down_spot = self.spot - spot_bump;
        let up_spot = self.spot + spot_bump;
        let down = self.local_vol_runtime_for_spot(local_volatility, down_spot)?;
        let up = self.local_vol_runtime_for_spot(local_volatility, up_spot)?;
        Ok(Some(LocalVolBumpRuntimes {
            down_spot,
            down,
            up_spot,
            up,
            spot_bump,
        }))
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_runtime_for_spot(
        &self,
        local_volatility: &LocalVolRuntime,
        spot: f64,
    ) -> Result<LocalVolRuntime, MonteCarloError> {
        let spot = PositiveF64::new(spot, "spot").map_err(ResultBuildError::from)?;
        let market_forward = self.market_forward.with_spot(spot)?;
        compile_local_vol_runtime(LocalVolRuntimeInputs {
            grid: local_volatility.grid.clone(),
            reporting_iv_basis: None,
            vega_kt: None,
            market_forward: &market_forward,
            valuation_date: self.valuation_date,
            expiry_time: self.time,
            event_times: &self.observation_times,
        })
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_pathwise_values(
        &self,
        local_volatility: &LocalVolRuntime,
        bump_runtimes: Option<&LocalVolBumpRuntimes>,
        shocks: &[f64],
        path: PathIndex,
    ) -> Result<[f64; PATHWISE_COMPONENTS], MonteCarloError> {
        let pathwise = self.local_vol_pathwise_values_and_buckets(
            local_volatility,
            bump_runtimes,
            shocks,
            path,
        )?;
        Ok(pathwise.values)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_pathwise_values_and_buckets(
        &self,
        local_volatility: &LocalVolRuntime,
        bump_runtimes: Option<&LocalVolBumpRuntimes>,
        shocks: &[f64],
        path: PathIndex,
    ) -> Result<LocalVolPathwise, MonteCarloError> {
        self.local_vol_pathwise_values_and_buckets_inner(
            local_volatility,
            bump_runtimes,
            shocks,
            path,
            None,
        )
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_fixed_policy_pathwise_values(
        &self,
        local_volatility: &LocalVolRuntime,
        bump_runtimes: Option<&LocalVolBumpRuntimes>,
        shocks: &[f64],
        path: PathIndex,
        selection: ExerciseCashflowSelection,
    ) -> Result<LocalVolPathwise, MonteCarloError> {
        self.local_vol_pathwise_values_and_buckets_inner(
            local_volatility,
            bump_runtimes,
            shocks,
            path,
            Some(selection),
        )
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_pathwise_values_and_buckets_inner(
        &self,
        local_volatility: &LocalVolRuntime,
        bump_runtimes: Option<&LocalVolBumpRuntimes>,
        shocks: &[f64],
        path: PathIndex,
        selection: Option<ExerciseCashflowSelection>,
    ) -> Result<LocalVolPathwise, MonteCarloError> {
        let path_state = if let (Some(dividends), Some(schedule)) = (
            local_volatility.dividends.as_ref(),
            local_volatility.dividend_schedule.as_ref(),
        ) {
            local_volatility.plan.evolve_path_with_dividend_checks(
                &local_volatility.grid,
                self.spot,
                shocks,
                dividends,
                schedule,
                path,
            )?
        } else {
            local_volatility
                .plan
                .evolve_path(&local_volatility.grid, self.spot, shocks)?
        };
        let observations =
            self.local_vol_path_observations(local_volatility, &path_state, self.spot)?;
        let continuous = if let Some(barrier) = &self.continuous_barrier {
            let bridge = self.local_vol_continuous_barrier_bridge(
                local_volatility,
                barrier,
                &path_state,
                self.spot,
            )?;
            let terminal = observations[barrier.expiry_observation_index].post_spot;
            let payoff = continuous_barrier_payoff_terms(barrier, terminal, bridge.survival());
            Some((bridge, payoff))
        } else {
            None
        };
        let graph_payoff = if continuous.is_none() {
            let observation = |underlying, date| {
                if underlying != self.underlying {
                    return None;
                }
                self.observation_dates
                    .iter()
                    .position(|observation_date| *observation_date == Some(date))
                    .map(|index| observations[index].post_spot)
            };
            let pre_dividend_observation = |underlying, date| {
                if underlying != self.underlying {
                    return None;
                }
                self.observation_dates
                    .iter()
                    .position(|observation_date| *observation_date == Some(date))
                    .and_then(|index| observations[index].pre_dividend_spot)
            };
            Some(if let Some(selection) = selection {
                self.payoff.evaluate_output_with_observation_adjoints(
                    selection.output_index,
                    observation,
                    pre_dividend_observation,
                )?
            } else {
                self.payoff.evaluate_single_with_observation_adjoints(
                    observation,
                    pre_dividend_observation,
                )?
            })
        } else {
            None
        };
        let discount = selection.map_or(self.discount, |selected| selected.discount);
        let price = discount
            * continuous.as_ref().map_or_else(
                || graph_payoff.as_ref().expect("graph payoff").value,
                |(_, payoff)| payoff.value,
            );
        let mut values = [0.0; PATHWISE_COMPONENTS];
        let mut raw_buckets = None;
        let mut allocation_audit = None;
        values[PRICE] = price;
        if let Some((bridge, _)) = &continuous {
            bridge.diagnostic_values().write_to(&mut values);
        }
        if self.request_vega || local_volatility.vega_kt.is_some() {
            let mut state_seeds = vec![0.0; path_state.states().len()];
            let mut direct_variance_seeds = vec![0.0; path_state.states().len()];
            if let Some((bridge, payoff)) = &continuous {
                let barrier = self
                    .continuous_barrier
                    .as_ref()
                    .expect("continuous Barrier");
                let terminal_observation = observations[barrier.expiry_observation_index];
                let terminal_coordinate =
                    self.observation_affine_coordinates[barrier.expiry_observation_index];
                state_seeds[terminal_observation.node_index] +=
                    discount * payoff.terminal_derivative * terminal_coordinate.b();
                if let LocalVolContinuousBarrierBridgeEvaluation::Exact(bridge) = bridge
                    && !bridge.endpoint_touched
                    && !bridge.dividend_jump_touched
                {
                    let interval_adjoints = bridge
                        .path
                        .reverse(discount * payoff.survival_derivative * bridge.survival());

                    for ((left_node, right_node), adjoints) in bridge
                        .interval_node_indices
                        .iter()
                        .copied()
                        .zip(interval_adjoints)
                    {
                        state_seeds[left_node] += adjoints.left_state;
                        state_seeds[right_node] += adjoints.right_state;
                        direct_variance_seeds[left_node] += adjoints.left_local_variance;
                        direct_variance_seeds[right_node] += adjoints.right_local_variance;
                    }
                }
                if let LocalVolContinuousBarrierBridgeEvaluation::Smoothed {
                    path,
                    interval_node_indices,
                    ..
                } = bridge
                {
                    let (interval_adjoints, hit_factor_adjoints) =
                        path.reverse(discount * payoff.survival_derivative);

                    for ((left_node, right_node), adjoints) in
                        interval_node_indices.iter().copied().zip(interval_adjoints)
                    {
                        state_seeds[left_node] += adjoints.left_endpoint.state;
                        state_seeds[right_node] += adjoints.right_endpoint.state;
                        direct_variance_seeds[left_node] += adjoints.left_local_variance;
                        direct_variance_seeds[right_node] += adjoints.right_local_variance;
                    }
                    for (state_index, adjoint) in hit_factor_adjoints {
                        state_seeds
                            [state_index.expect("Local Vol hit factors use node indices")] +=
                            adjoint;
                    }
                }
            } else {
                let payoff = graph_payoff.as_ref().expect("graph payoff");
                for adjoint in &payoff.terminal_adjoints {
                    if adjoint.underlying != self.underlying {
                        continue;
                    }
                    if let Some(index) = self
                        .observation_dates
                        .iter()
                        .position(|date| *date == Some(adjoint.observation_date))
                    {
                        let observation = observations[index];
                        let coordinate = self.observation_affine_coordinates[index];
                        state_seeds[observation.node_index] +=
                            discount * adjoint.value * coordinate.b();
                    }
                }
                for adjoint in &payoff.pre_dividend_adjoints {
                    if adjoint.underlying != self.underlying {
                        continue;
                    }
                    if let Some(index) = self
                        .observation_dates
                        .iter()
                        .position(|date| *date == Some(adjoint.observation_date))
                    {
                        let observation = observations[index];
                        let coordinate = self.observation_pre_dividend_coordinates[index]
                            .expect("pre-dividend adjoints have a matching coordinate");
                        state_seeds[observation.node_index] +=
                            discount * adjoint.value * coordinate.b();
                    }
                }
            }
            let reverse = local_volatility.plan.reverse_path_with_variance_seeds(
                &local_volatility.grid,
                &path_state,
                &state_seeds,
                &direct_variance_seeds,
            )?;
            let local_vol_node_adjoints = reverse
                .local_variance_value_adjoints()
                .iter()
                .zip(local_volatility.grid.values())
                .map(|(adjoint, variance)| adjoint * 2.0 * variance.sqrt())
                .collect::<Vec<_>>();
            let vega = local_vol_node_adjoints
                .iter()
                .copied()
                .collect::<pricing_numerics::NeumaierSum>()
                .total();
            values[VEGA] = vega;
            if local_volatility.vega_kt.is_some() && self.terminal_replication_vega_kt.is_none() {
                let (buckets, audit) = if let Some(terminal) = &self.terminal_vega_kt {
                    terminal.sample(*path_state.states().last().expect("terminal state"), shocks)?
                } else {
                    allocate_local_vol_nodes(local_volatility, &local_vol_node_adjoints)?
                };
                raw_buckets = Some(buckets);
                allocation_audit = Some(audit);
            }
        }
        if let Some(bumps) = bump_runtimes {
            let down = self.local_vol_discounted_payoff_at_spot_inner(
                &bumps.down,
                bumps.down_spot,
                shocks,
                path,
                selection,
            )?;
            let up = self.local_vol_discounted_payoff_at_spot_inner(
                &bumps.up,
                bumps.up_spot,
                shocks,
                path,
                selection,
            )?;
            let delta = (up - down) / (2.0 * bumps.spot_bump);
            values[DELTA] = delta;
            values[BUMP_DELTA] = delta;
            if self.request_gamma.is_some() {
                let gamma = (up - 2.0 * price + down) / bumps.spot_bump.powi(2);
                values[GAMMA] = gamma;
                values[BUMP_GAMMA] = gamma;
            }
        }
        Ok(LocalVolPathwise {
            values,
            raw_buckets,
            allocation_audit,
        })
    }
}
