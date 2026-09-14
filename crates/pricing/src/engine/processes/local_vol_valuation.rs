//! processes / local vol valuation implementation.

use crate::core::PathIndex;

use crate::market::{
    ImpliedVarianceSurface, LocalVarianceInterpolation, MarketError, ThetaRegion,
    TotalVarianceDerivatives,
};

use crate::mc::{
    BarrierBridgeIntervalInput, BarrierBridgePath, LocalVolPath, SmoothedBarrierBridgeEndpoint,
    SmoothedBarrierBridgeEndpointInput, SmoothedBarrierBridgeInterval,
    SmoothedBarrierBridgeIntervalInput, transformed_barrier,
};

use crate::models::LocalVolatilityReportingBasis;

use crate::product::CompactC2Smoothing;

use crate::risk::PayoffSmoothing;

use crate::MonteCarloError;

use crate::engine::plan::simulation::SimulationPlan;

use crate::engine::plan::simulation::LocalVolRuntime;

use crate::engine::plan::simulation::ContinuousBarrierRuntime;

use crate::engine::plan::simulation::SmoothedBarrierHitKind;

use crate::engine::plan::simulation::SmoothedContinuousBarrierPath;

use crate::engine::plan::simulation::ExactLocalVolContinuousBarrierBridgeEvaluation;

use crate::engine::plan::simulation::LocalVolContinuousBarrierBridgeEvaluation;

use crate::engine::plan::simulation::LocalVolPathwise;

use crate::engine::plan::simulation::ExerciseCashflowSelection;

use crate::engine::plan::simulation::LocalVolObservation;

use crate::engine::plan::simulation::ReportingIvSurface;

use crate::engine::payoff::barrier::bridge_direction;

use crate::engine::payoff::barrier::barrier_touched;

use crate::engine::payoff::barrier::smoothed_endpoint_hit_factor;

use crate::engine::payoff::barrier::smoothed_jump_hit_factor;

use crate::engine::payoff::barrier::continuous_barrier_payoff_terms;

impl<'a> ReportingIvSurface<'a> {
    pub(in crate::engine) const fn new(basis: &'a LocalVolatilityReportingBasis) -> Self {
        Self { basis }
    }

    pub(in crate::engine) fn total_variance_value(&self, time_index: usize, x_index: usize) -> f64 {
        let x_count = self.basis.log_forward_moneyness_nodes().len();
        let maturity = self.basis.maturity_nodes()[time_index];
        let volatility = self.basis.implied_volatilities()[time_index * x_count + x_index];
        maturity * volatility * volatility
    }
}

impl ImpliedVarianceSurface for ReportingIvSurface<'_> {
    fn total_variance_derivatives(
        &self,
        time: f64,
        log_moneyness: f64,
    ) -> Result<TotalVarianceDerivatives, MarketError> {
        if !time.is_finite()
            || time < self.basis.maturity_nodes()[0]
            || time > self.basis.maturity_nodes()[self.basis.maturity_nodes().len() - 1]
        {
            return Err(MarketError::InvalidSurfaceQuery {
                coordinate: "time",
                bits: time.to_bits(),
            });
        }
        if !log_moneyness.is_finite() {
            return Err(MarketError::InvalidSurfaceQuery {
                coordinate: "log_moneyness",
                bits: log_moneyness.to_bits(),
            });
        }
        let time_index = lower_cell(self.basis.maturity_nodes(), time);
        let x = log_moneyness
            .max(self.basis.log_forward_moneyness_nodes()[0])
            .min(
                self.basis.log_forward_moneyness_nodes()
                    [self.basis.log_forward_moneyness_nodes().len() - 1],
            );
        let x_index = lower_cell(self.basis.log_forward_moneyness_nodes(), x);
        let time_left = self.basis.maturity_nodes()[time_index];
        let time_right = self.basis.maturity_nodes()[time_index + 1];
        let x_left = self.basis.log_forward_moneyness_nodes()[x_index];
        let x_right = self.basis.log_forward_moneyness_nodes()[x_index + 1];
        let time_weight = interpolation_weight(time_left, time_right, time);
        let x_weight = interpolation_weight(x_left, x_right, x);
        let w00 = self.total_variance_value(time_index, x_index);
        let w01 = self.total_variance_value(time_index, x_index + 1);
        let w10 = self.total_variance_value(time_index + 1, x_index);
        let w11 = self.total_variance_value(time_index + 1, x_index + 1);
        let lower = w00 * (1.0 - x_weight) + w01 * x_weight;
        let upper = w10 * (1.0 - x_weight) + w11 * x_weight;
        let total_variance = lower * (1.0 - time_weight) + upper * time_weight;
        let time_derivative = (upper - lower) / (time_right - time_left);
        let log_moneyness_derivative =
            ((w01 - w00) * (1.0 - time_weight) + (w11 - w10) * time_weight) / (x_right - x_left);
        let log_moneyness_second_derivative = 0.0;
        Ok(TotalVarianceDerivatives {
            total_variance,
            log_moneyness_derivative,
            log_moneyness_second_derivative,
            time_derivative,
            theta: total_variance,
            theta_derivative: time_derivative,
            theta_region: ThetaRegion::Interpolated,
        })
    }
}

pub(in crate::engine) fn lower_cell(nodes: &[f64], value: f64) -> usize {
    match nodes.binary_search_by(|node| node.total_cmp(&value)) {
        Ok(index) => index.min(nodes.len() - 2),
        Err(index) => index.saturating_sub(1).min(nodes.len() - 2),
    }
}

pub(in crate::engine) fn interpolation_weight(left: f64, right: f64, value: f64) -> f64 {
    if value.to_bits() == right.to_bits() {
        1.0
    } else {
        (value - left) / (right - left)
    }
}

pub(in crate::engine) fn average_local_vol_pathwise(
    primary: LocalVolPathwise,
    mate: LocalVolPathwise,
) -> Result<LocalVolPathwise, MonteCarloError> {
    let values =
        std::array::from_fn(|component| (primary.values[component] + mate.values[component]) * 0.5);
    let raw_buckets = match (primary.raw_buckets, mate.raw_buckets) {
        (Some(primary), Some(mate)) => {
            if primary.len() != mate.len() {
                return Err(MonteCarloError::MismatchedLocalVolatilityReportingBasis);
            }
            Some(
                primary
                    .into_iter()
                    .zip(mate)
                    .map(|(left, right)| (left + right) * 0.5)
                    .collect(),
            )
        }
        (None, None) => None,
        _ => return Err(MonteCarloError::MismatchedLocalVolatilityReportingBasis),
    };
    Ok(LocalVolPathwise {
        values,
        raw_buckets,
    })
}

impl LocalVolRuntime {
    pub(in crate::engine) fn maximum_total_variance(&self) -> f64 {
        let horizon = self.plan.time_grid().nodes().last().copied().unwrap_or(0.0);
        horizon * self.grid.cap()
    }
}

impl SimulationPlan {
    #[cfg(test)]
    pub(in crate::engine) fn local_vol_discounted_payoff_at_spot(
        &self,
        local_volatility: &LocalVolRuntime,
        spot: f64,
        shocks: &[f64],
        path: PathIndex,
    ) -> Result<f64, MonteCarloError> {
        self.local_vol_discounted_payoff_at_spot_inner(local_volatility, spot, shocks, path, None)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_discounted_payoff_at_spot_inner(
        &self,
        local_volatility: &LocalVolRuntime,
        spot: f64,
        shocks: &[f64],
        path: PathIndex,
        selection: Option<ExerciseCashflowSelection>,
    ) -> Result<f64, MonteCarloError> {
        let path = if let (Some(dividends), Some(schedule)) = (
            local_volatility.dividends.as_ref(),
            local_volatility.dividend_schedule.as_ref(),
        ) {
            local_volatility.plan.evolve_path_with_dividend_checks(
                &local_volatility.grid,
                spot,
                shocks,
                dividends,
                schedule,
                path,
            )?
        } else {
            local_volatility
                .plan
                .evolve_path(&local_volatility.grid, spot, shocks)?
        };
        let observations = self.local_vol_path_observations(local_volatility, &path, spot)?;
        if let Some(barrier) = &self.continuous_barrier {
            let bridge =
                self.local_vol_continuous_barrier_bridge(local_volatility, barrier, &path, spot)?;
            let terminal = observations[barrier.expiry_observation_index].post_spot;
            let payoff = continuous_barrier_payoff_terms(barrier, terminal, bridge.survival());
            return Ok(self.discount * payoff.value);
        }
        let outputs = self.payoff_outputs_from_local_vol_observations(&observations)?;
        let selection = selection.unwrap_or(ExerciseCashflowSelection {
            output_index: 0,
            discount: self.discount,
        });
        let value = outputs.get(selection.output_index).copied().ok_or(
            crate::product::GraphError::InvalidOutputIndex {
                index: selection.output_index,
                count: outputs.len(),
            },
        )?;
        Ok(selection.discount * value)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_path_observations(
        &self,
        local_volatility: &LocalVolRuntime,
        path: &LocalVolPath,
        spot: f64,
    ) -> Result<Vec<LocalVolObservation>, MonteCarloError> {
        self.local_vol_state_observations(local_volatility, path.states(), spot)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_state_observations(
        &self,
        local_volatility: &LocalVolRuntime,
        states: &[f64],
        spot: f64,
    ) -> Result<Vec<LocalVolObservation>, MonteCarloError> {
        let node_indices = self.observation_local_vol_node_indices.as_ref().ok_or(
            MonteCarloError::UnsupportedModel {
                model: "local_volatility",
            },
        )?;
        Ok(node_indices
            .iter()
            .map(|&node_index| {
                let canonical_f = states[node_index];
                let post_coordinate = local_volatility.node_affine_coordinates[node_index];
                let post_spot = post_coordinate.a() * spot + post_coordinate.b() * canonical_f;
                let pre_dividend_spot = local_volatility.node_pre_dividend_coordinates[node_index]
                    .map(|coordinate| coordinate.a() * spot + coordinate.b() * canonical_f);
                LocalVolObservation {
                    post_spot,
                    pre_dividend_spot,
                    node_index,
                }
            })
            .collect())
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn local_vol_continuous_barrier_bridge(
        &self,
        local_volatility: &LocalVolRuntime,
        barrier: &ContinuousBarrierRuntime,
        path: &LocalVolPath,
        spot: f64,
    ) -> Result<LocalVolContinuousBarrierBridgeEvaluation, MonteCarloError> {
        let nodes = local_volatility.plan.time_grid().nodes();
        let end_node = nodes
            .iter()
            .rposition(|time| *time <= barrier.monitoring_end_time)
            .expect("Local Volatility time grids start at valuation");
        let mut node_interpolations = Vec::with_capacity(end_node + 1);
        for (node, (&time, &state)) in nodes
            .iter()
            .zip(path.states())
            .take(end_node + 1)
            .enumerate()
        {
            let forward = local_volatility.plan.forward_normalizers()[node];
            node_interpolations.push(
                local_volatility
                    .grid
                    .interpolate(time, (state / forward).ln())?,
            );
        }

        if let Some(PayoffSmoothing::CompactC2 { half_width }) = self.payoff_smoothing {
            return self.smoothed_local_vol_continuous_barrier_bridge(
                local_volatility,
                barrier,
                path,
                spot,
                end_node,
                node_interpolations,
                CompactC2Smoothing::from_positive(half_width),
            );
        }

        let mut intervals = Vec::with_capacity(end_node);
        let mut interval_node_indices = Vec::with_capacity(end_node);
        let initial_touched = barrier_touched(barrier.direction, spot, barrier.barrier);
        let mut dividend_jump_touched = false;
        for right_node in 1..=end_node {
            if initial_touched {
                break;
            }
            let left_node = right_node - 1;
            let left_coordinate = local_volatility.node_affine_coordinates[left_node];
            let right_post_coordinate = local_volatility.node_affine_coordinates[right_node];
            let right_pre_coordinate = local_volatility.node_pre_dividend_coordinates[right_node]
                .unwrap_or(right_post_coordinate);
            let left_barrier = transformed_barrier(barrier.barrier, spot, left_coordinate)?;
            let right_barrier = transformed_barrier(barrier.barrier, spot, right_pre_coordinate)?;
            intervals.push(BarrierBridgeIntervalInput {
                direction: bridge_direction(barrier.direction),
                left_state: path.states()[left_node],
                right_state: path.states()[right_node],
                left_barrier,
                right_barrier,
                left_local_variance: node_interpolations[left_node].value,
                right_local_variance: node_interpolations[right_node].value,
                dt: nodes[right_node] - nodes[left_node],
            });
            interval_node_indices.push((left_node, right_node));

            if local_volatility.node_pre_dividend_coordinates[right_node].is_some() {
                let state = path.states()[right_node];
                let pre_spot = right_pre_coordinate.a() * spot + right_pre_coordinate.b() * state;
                let post_spot =
                    right_post_coordinate.a() * spot + right_post_coordinate.b() * state;
                let pre_touched = barrier_touched(barrier.direction, pre_spot, barrier.barrier);
                let post_touched = barrier_touched(barrier.direction, post_spot, barrier.barrier);
                if pre_touched || post_touched {
                    dividend_jump_touched = !pre_touched && post_touched;
                    break;
                }
            }
        }
        let path = BarrierBridgePath::evaluate(&intervals)?;
        let endpoint_touched = initial_touched || path.diagnostics().touched_endpoint_count != 0;
        Ok(LocalVolContinuousBarrierBridgeEvaluation::Exact(
            ExactLocalVolContinuousBarrierBridgeEvaluation {
                path,
                interval_node_indices: interval_node_indices.into_boxed_slice(),
                node_interpolations: node_interpolations.into_boxed_slice(),
                endpoint_touched,
                dividend_jump_touched,
            },
        ))
    }
}

impl SimulationPlan {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::engine) fn smoothed_local_vol_continuous_barrier_bridge(
        &self,
        local_volatility: &LocalVolRuntime,
        barrier: &ContinuousBarrierRuntime,
        path: &LocalVolPath,
        spot: f64,
        end_node: usize,
        node_interpolations: Vec<LocalVarianceInterpolation>,
        smoothing: CompactC2Smoothing,
    ) -> Result<LocalVolContinuousBarrierBridgeEvaluation, MonteCarloError> {
        let direction = bridge_direction(barrier.direction);
        let nodes = local_volatility.plan.time_grid().nodes();
        let initial_coordinate = local_volatility.node_affine_coordinates[0];
        let initial_barrier = transformed_barrier(barrier.barrier, spot, initial_coordinate)?;
        let initial_endpoint =
            SmoothedBarrierBridgeEndpoint::evaluate(SmoothedBarrierBridgeEndpointInput {
                direction,
                state: path.states()[0],
                transformed_barrier: initial_barrier,
                affine_scale: initial_coordinate.b(),
                smoothing,
            })?;
        let mut hit_factors = vec![smoothed_endpoint_hit_factor(
            initial_endpoint,
            Some(0),
            SmoothedBarrierHitKind::Endpoint,
        )];
        let mut intervals = Vec::with_capacity(end_node);
        let mut interval_node_indices = Vec::with_capacity(end_node);
        for right_node in 1..=end_node {
            let left_node = right_node - 1;
            let left_coordinate = local_volatility.node_affine_coordinates[left_node];
            let right_post_coordinate = local_volatility.node_affine_coordinates[right_node];
            let right_pre_coordinate = local_volatility.node_pre_dividend_coordinates[right_node]
                .unwrap_or(right_post_coordinate);
            let left_barrier = transformed_barrier(barrier.barrier, spot, left_coordinate)?;
            let right_barrier = transformed_barrier(barrier.barrier, spot, right_pre_coordinate)?;
            let interval =
                SmoothedBarrierBridgeInterval::evaluate(SmoothedBarrierBridgeIntervalInput {
                    left_endpoint: SmoothedBarrierBridgeEndpointInput {
                        direction,
                        state: path.states()[left_node],
                        transformed_barrier: left_barrier,
                        affine_scale: left_coordinate.b(),
                        smoothing,
                    },
                    right_endpoint: SmoothedBarrierBridgeEndpointInput {
                        direction,
                        state: path.states()[right_node],
                        transformed_barrier: right_barrier,
                        affine_scale: right_pre_coordinate.b(),
                        smoothing,
                    },
                    left_local_variance: node_interpolations[left_node].value,
                    right_local_variance: node_interpolations[right_node].value,
                    dt: nodes[right_node] - nodes[left_node],
                })?;
            if local_volatility.node_pre_dividend_coordinates[right_node].is_some() {
                let state = path.states()[right_node];
                let pre_spot = right_pre_coordinate.a() * spot + right_pre_coordinate.b() * state;
                let post_spot =
                    right_post_coordinate.a() * spot + right_post_coordinate.b() * state;
                hit_factors.push(smoothed_jump_hit_factor(
                    barrier.direction,
                    pre_spot,
                    post_spot,
                    right_pre_coordinate.b(),
                    right_post_coordinate.b(),
                    barrier.barrier,
                    smoothing,
                    Some(right_node),
                ));
            } else {
                hit_factors.push(smoothed_endpoint_hit_factor(
                    interval.right_endpoint(),
                    Some(right_node),
                    SmoothedBarrierHitKind::Endpoint,
                ));
            }
            intervals.push(interval);
            interval_node_indices.push((left_node, right_node));
        }
        Ok(LocalVolContinuousBarrierBridgeEvaluation::Smoothed {
            path: SmoothedContinuousBarrierPath::evaluate(intervals, hit_factors),
            interval_node_indices: interval_node_indices.into_boxed_slice(),
            node_interpolations: node_interpolations.into_boxed_slice(),
        })
    }
}
