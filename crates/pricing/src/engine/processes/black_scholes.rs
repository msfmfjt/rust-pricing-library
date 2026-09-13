//! processes / black scholes implementation.

use crate::engine::aad::SoaWorkspace;

use crate::MonteCarloError;
use crate::engine::payoff::barrier::barrier_touched;
use crate::engine::payoff::barrier::bridge_direction;
use crate::engine::payoff::barrier::continuous_barrier_payoff_terms;
use crate::engine::payoff::barrier::smoothed_endpoint_hit_factor;
use crate::engine::payoff::barrier::smoothed_jump_hit_factor;
use crate::engine::plan::simulation::BUMP_DELTA;
use crate::engine::plan::simulation::BUMP_GAMMA;
use crate::engine::plan::simulation::BUMP_VEGA;
use crate::engine::plan::simulation::ContinuousBarrierBridgeEvaluation;
use crate::engine::plan::simulation::ContinuousBarrierRuntime;
use crate::engine::plan::simulation::DELTA;
use crate::engine::plan::simulation::DELTA_DIFFERENCE;
use crate::engine::plan::simulation::EarlyExerciseRuntime;
use crate::engine::plan::simulation::ExactContinuousBarrierBridgeEvaluation;
use crate::engine::plan::simulation::GAMMA;
use crate::engine::plan::simulation::GAMMA_DIFFERENCE;
use crate::engine::plan::simulation::PATHWISE_COMPONENTS;
use crate::engine::plan::simulation::PRICE;
use crate::engine::plan::simulation::PathObservation;
use crate::engine::plan::simulation::PathwiseAad;
use crate::engine::plan::simulation::SimulationPlan;
use crate::engine::plan::simulation::SmoothedBarrierHitKind;
use crate::engine::plan::simulation::SmoothedContinuousBarrierPath;
use crate::engine::plan::simulation::VEGA;
use crate::engine::plan::simulation::VEGA_DIFFERENCE;
use crate::engine::plan::simulation::resolve_spot_bump;
use crate::mc::{
    BarrierBridgeIntervalInput, BarrierBridgePath, SmoothedBarrierBridgeEndpoint,
    SmoothedBarrierBridgeEndpointInput, SmoothedBarrierBridgeInterval,
    SmoothedBarrierBridgeIntervalInput, transformed_barrier,
};
use crate::product::CompactC2Smoothing;
use crate::risk::PayoffSmoothing;

impl SimulationPlan {
    pub(in crate::engine) fn discounted_payoff_from_normals(
        &self,
        normals: &[f64],
    ) -> Result<f64, MonteCarloError> {
        let observations = self.path_observations_from_normals(normals, self.spot, self.volatility);
        if let Some(barrier) = &self.continuous_barrier {
            return self.continuous_barrier_discounted_payoff(barrier, &observations);
        }
        let outputs = self.payoff_outputs_from_observations(&observations)?;
        Ok(self.discount
            * outputs
                .first()
                .copied()
                .ok_or(crate::product::GraphError::NoOutputs)?)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn continuous_barrier_path_values_from_normals(
        &self,
        normals: &[f64],
    ) -> Result<[f64; PATHWISE_COMPONENTS], MonteCarloError> {
        let barrier = self
            .continuous_barrier
            .as_ref()
            .expect("continuous Barrier path values require a compiled Barrier");
        let observations = self.path_observations_from_normals(normals, self.spot, self.volatility);
        let bridge =
            self.continuous_barrier_bridge(barrier, &observations, self.spot, self.volatility)?;
        let terminal = observations[barrier.expiry_observation_index].post_spot;
        let payoff = continuous_barrier_payoff_terms(barrier, terminal, bridge.survival());
        let mut values = [0.0; PATHWISE_COMPONENTS];
        values[PRICE] = self.discount * payoff.value;
        bridge.diagnostic_values().write_to(&mut values);
        Ok(values)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn continuous_barrier_discounted_payoff(
        &self,
        barrier: &ContinuousBarrierRuntime,
        observations: &[PathObservation],
    ) -> Result<f64, MonteCarloError> {
        let bridge =
            self.continuous_barrier_bridge(barrier, observations, self.spot, self.volatility)?;
        let terminal = observations[barrier.expiry_observation_index].post_spot;
        let payoff = continuous_barrier_payoff_terms(barrier, terminal, bridge.survival());
        Ok(self.discount * payoff.value)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn continuous_barrier_bridge(
        &self,
        barrier: &ContinuousBarrierRuntime,
        observations: &[PathObservation],
        spot: f64,
        volatility: f64,
    ) -> Result<ContinuousBarrierBridgeEvaluation, MonteCarloError> {
        if let Some(PayoffSmoothing::CompactC2 { half_width }) = self.payoff_smoothing {
            return self.smoothed_continuous_barrier_bridge(
                barrier,
                observations,
                spot,
                volatility,
                CompactC2Smoothing::from_positive(half_width),
            );
        }
        let mut previous_state = spot;
        let mut previous_time = 0.0;
        let mut previous_index = None;
        let mut intervals = Vec::with_capacity(barrier.bridge_observation_indices.len());
        let mut interval_observation_indices =
            Vec::with_capacity(barrier.bridge_observation_indices.len());
        let initial_touched = barrier_touched(barrier.direction, spot, barrier.barrier);
        let mut dividend_jump_touched = false;
        let mut previous_barrier = barrier.barrier;
        let variance = volatility * volatility;
        for &index in &barrier.bridge_observation_indices {
            if initial_touched {
                break;
            }
            let time = self.observation_times[index];
            let state = observations[index].canonical_f;
            let post_coordinate = self.observation_affine_coordinates[index];
            let pre_coordinate =
                self.observation_pre_dividend_coordinates[index].unwrap_or(post_coordinate);
            let pre_barrier = transformed_barrier(barrier.barrier, self.spot, pre_coordinate)?;
            let post_barrier = transformed_barrier(barrier.barrier, self.spot, post_coordinate)?;
            let dt = time - previous_time;
            if dt > 0.0 {
                intervals.push(BarrierBridgeIntervalInput {
                    direction: bridge_direction(barrier.direction),
                    left_state: previous_state,
                    right_state: state,
                    left_barrier: previous_barrier,
                    right_barrier: pre_barrier,
                    left_local_variance: variance,
                    right_local_variance: variance,
                    dt,
                });
                interval_observation_indices.push((previous_index, index));
            }
            if self.observation_pre_dividend_coordinates[index].is_some() {
                let pre_spot = pre_coordinate.a() * self.spot + pre_coordinate.b() * state;
                let post_spot = post_coordinate.a() * self.spot + post_coordinate.b() * state;
                let pre_touched = barrier_touched(barrier.direction, pre_spot, barrier.barrier);
                let post_touched = barrier_touched(barrier.direction, post_spot, barrier.barrier);
                if pre_touched || post_touched {
                    dividend_jump_touched = !pre_touched && post_touched;
                    break;
                }
            }
            previous_state = state;
            previous_barrier = post_barrier;
            previous_time = time;
            previous_index = Some(index);
        }
        let path = BarrierBridgePath::evaluate(&intervals)?;
        let endpoint_touched = initial_touched || path.diagnostics().touched_endpoint_count != 0;
        Ok(ContinuousBarrierBridgeEvaluation::Exact(
            ExactContinuousBarrierBridgeEvaluation {
                path,
                interval_observation_indices: interval_observation_indices.into_boxed_slice(),
                endpoint_touched,
                dividend_jump_touched,
            },
        ))
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn smoothed_continuous_barrier_bridge(
        &self,
        barrier: &ContinuousBarrierRuntime,
        observations: &[PathObservation],
        spot: f64,
        volatility: f64,
        smoothing: CompactC2Smoothing,
    ) -> Result<ContinuousBarrierBridgeEvaluation, MonteCarloError> {
        let direction = bridge_direction(barrier.direction);
        let initial_endpoint =
            SmoothedBarrierBridgeEndpoint::evaluate(SmoothedBarrierBridgeEndpointInput {
                direction,
                state: spot,
                transformed_barrier: barrier.barrier,
                affine_scale: 1.0,
                smoothing,
            })?;
        let mut hit_factors = vec![smoothed_endpoint_hit_factor(
            initial_endpoint,
            None,
            SmoothedBarrierHitKind::Endpoint,
        )];
        let mut intervals = Vec::with_capacity(barrier.bridge_observation_indices.len());
        let mut interval_observation_indices =
            Vec::with_capacity(barrier.bridge_observation_indices.len());
        let mut previous_state = spot;
        let mut previous_time = 0.0;
        let mut previous_index = None;
        let mut previous_barrier = barrier.barrier;
        let mut previous_scale = 1.0;
        let variance = volatility * volatility;
        for &index in &barrier.bridge_observation_indices {
            let time = self.observation_times[index];
            let state = observations[index].canonical_f;
            let post_coordinate = self.observation_affine_coordinates[index];
            let pre_coordinate =
                self.observation_pre_dividend_coordinates[index].unwrap_or(post_coordinate);
            let pre_barrier = transformed_barrier(barrier.barrier, self.spot, pre_coordinate)?;
            let post_barrier = transformed_barrier(barrier.barrier, self.spot, post_coordinate)?;
            let dt = time - previous_time;
            if dt > 0.0 {
                let interval =
                    SmoothedBarrierBridgeInterval::evaluate(SmoothedBarrierBridgeIntervalInput {
                        left_endpoint: SmoothedBarrierBridgeEndpointInput {
                            direction,
                            state: previous_state,
                            transformed_barrier: previous_barrier,
                            affine_scale: previous_scale,
                            smoothing,
                        },
                        right_endpoint: SmoothedBarrierBridgeEndpointInput {
                            direction,
                            state,
                            transformed_barrier: pre_barrier,
                            affine_scale: pre_coordinate.b(),
                            smoothing,
                        },
                        left_local_variance: variance,
                        right_local_variance: variance,
                        dt,
                    })?;
                if self.observation_pre_dividend_coordinates[index].is_some() {
                    let pre_spot = pre_coordinate.a() * self.spot + pre_coordinate.b() * state;
                    let post_spot = post_coordinate.a() * self.spot + post_coordinate.b() * state;
                    hit_factors.push(smoothed_jump_hit_factor(
                        barrier.direction,
                        pre_spot,
                        post_spot,
                        pre_coordinate.b(),
                        post_coordinate.b(),
                        barrier.barrier,
                        smoothing,
                        Some(index),
                    ));
                } else {
                    hit_factors.push(smoothed_endpoint_hit_factor(
                        interval.right_endpoint(),
                        Some(index),
                        SmoothedBarrierHitKind::Endpoint,
                    ));
                }
                intervals.push(interval);
                interval_observation_indices.push((previous_index, index));
            }
            previous_state = state;
            previous_barrier = post_barrier;
            previous_scale = post_coordinate.b();
            previous_time = time;
            previous_index = Some(index);
        }
        Ok(ContinuousBarrierBridgeEvaluation::Smoothed {
            path: SmoothedContinuousBarrierPath::evaluate(intervals, hit_factors),
            interval_observation_indices: interval_observation_indices.into_boxed_slice(),
        })
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn path_observations_from_normals(
        &self,
        normals: &[f64],
        spot: f64,
        volatility: f64,
    ) -> Vec<PathObservation> {
        let spot_scale = spot / self.spot;
        if self.observation_times.len() == 1 {
            let time = self.observation_times[0];
            let normal = normals[0];
            let total_variance = volatility * volatility * time;
            let standard_deviation = total_variance.sqrt();
            let brownian = time.sqrt() * normal;
            let log_return = -0.5 * total_variance + standard_deviation * normal;
            let canonical_f = self.observation_forwards[0] * spot_scale * log_return.exp();
            return vec![self.path_observation(0, canonical_f, brownian)];
        }
        let mut previous_time = 0.0;
        let mut brownian = 0.0;
        self.observation_times
            .iter()
            .zip(self.observation_forwards.iter())
            .zip(normals.iter())
            .enumerate()
            .map(|(index, ((&time, &forward), &normal))| {
                let step = (time - previous_time).max(0.0);
                brownian += step.sqrt() * normal;
                previous_time = time;
                let total_variance = volatility * volatility * time;
                let canonical_f =
                    forward * spot_scale * (-0.5 * total_variance + volatility * brownian).exp();
                self.path_observation(index, canonical_f, brownian)
            })
            .collect()
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn path_observation(
        &self,
        index: usize,
        canonical_f: f64,
        brownian: f64,
    ) -> PathObservation {
        let post_coordinate = self.observation_affine_coordinates[index];
        let post_spot = post_coordinate.a() * self.spot + post_coordinate.b() * canonical_f;
        let pre_dividend_spot = self.observation_pre_dividend_coordinates[index]
            .map(|coordinate| coordinate.a() * self.spot + coordinate.b() * canonical_f);
        PathObservation {
            post_spot,
            pre_dividend_spot,
            canonical_f,
            brownian,
        }
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn fixed_policy_pathwise_values(
        &self,
        normals: &[f64],
        lane: usize,
        workspace: &mut SoaWorkspace,
        stopping_index: usize,
        early_exercise: &EarlyExerciseRuntime,
    ) -> Result<[f64; PATHWISE_COMPONENTS], MonteCarloError> {
        let discount = *early_exercise.discount_factors.get(stopping_index).ok_or(
            crate::product::GraphError::InvalidOutputIndex {
                index: stopping_index,
                count: early_exercise.exercise_dates.len(),
            },
        )?;
        let evaluate = |spot, volatility| {
            self.pathwise_aad_for_output(normals, spot, volatility, stopping_index, discount)
        };
        let base = evaluate(self.spot, self.volatility)?;
        let gamma = if let Some(gamma) = self.request_gamma {
            let bump = resolve_spot_bump(gamma, self.spot);
            let delta_down = evaluate(self.spot - bump, self.volatility)?.delta;
            let delta_up = evaluate(self.spot + bump, self.volatility)?.delta;
            (delta_up - delta_down) / (2.0 * bump)
        } else {
            0.0
        };
        let spot_bump = self.validation_spot_bump;
        let down_spot = evaluate(self.spot - spot_bump, self.volatility)?;
        let up_spot = evaluate(self.spot + spot_bump, self.volatility)?;
        let bump_delta = (up_spot.price - down_spot.price) / (2.0 * spot_bump);
        let bump_gamma = (up_spot.price - 2.0 * base.price + down_spot.price) / spot_bump.powi(2);
        let bump_vega = if self.validation_volatility_bump == 0.0 {
            0.0
        } else {
            let bump = self.validation_volatility_bump;
            let down = evaluate(self.spot, self.volatility - bump)?.price;
            let up = evaluate(self.spot, self.volatility + bump)?.price;
            (up - down) / (2.0 * bump)
        };
        workspace.primal_mut(0)?.set(lane, base.price)?;
        workspace.primal_mut(1)?.set(lane, base.delta)?;
        workspace.primal_mut(2)?.set(lane, base.vega)?;
        workspace.primal_mut(3)?.set(lane, gamma)?;
        let mut values = [0.0; PATHWISE_COMPONENTS];
        values[PRICE] = workspace.primal(0)?.get(lane)?;
        values[DELTA] = workspace.primal(1)?.get(lane)?;
        values[VEGA] = workspace.primal(2)?.get(lane)?;
        values[GAMMA] = workspace.primal(3)?.get(lane)?;
        values[BUMP_DELTA] = bump_delta;
        values[DELTA_DIFFERENCE] = bump_delta - base.delta;
        values[BUMP_VEGA] = bump_vega;
        values[VEGA_DIFFERENCE] = bump_vega - base.vega;
        values[BUMP_GAMMA] = bump_gamma;
        values[GAMMA_DIFFERENCE] = bump_gamma - gamma;
        Ok(values)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn pathwise_values(
        &self,
        normals: &[f64],
        lane: usize,
        workspace: &mut SoaWorkspace,
    ) -> Result<[f64; PATHWISE_COMPONENTS], MonteCarloError> {
        let base = self.pathwise_aad(normals, self.spot, self.volatility)?;
        let gamma = if let Some(gamma) = self.request_gamma {
            let bump = resolve_spot_bump(gamma, self.spot);
            let delta_down = self
                .pathwise_aad(normals, self.spot - bump, self.volatility)?
                .delta;
            let delta_up = self
                .pathwise_aad(normals, self.spot + bump, self.volatility)?
                .delta;
            (delta_up - delta_down) / (2.0 * bump)
        } else {
            0.0
        };
        let spot_bump = self.validation_spot_bump;
        let down_spot = self.pathwise_aad(normals, self.spot - spot_bump, self.volatility)?;
        let up_spot = self.pathwise_aad(normals, self.spot + spot_bump, self.volatility)?;
        let bump_delta = (up_spot.price - down_spot.price) / (2.0 * spot_bump);
        let bump_gamma = (up_spot.price - 2.0 * base.price + down_spot.price) / spot_bump.powi(2);
        let bump_vega = if self.validation_volatility_bump == 0.0 {
            0.0
        } else {
            let bump = self.validation_volatility_bump;
            let down = self
                .pathwise_aad(normals, self.spot, self.volatility - bump)?
                .price;
            let up = self
                .pathwise_aad(normals, self.spot, self.volatility + bump)?
                .price;
            (up - down) / (2.0 * bump)
        };
        workspace.primal_mut(0)?.set(lane, base.price)?;
        workspace.primal_mut(1)?.set(lane, base.delta)?;
        workspace.primal_mut(2)?.set(lane, base.vega)?;
        workspace.primal_mut(3)?.set(lane, gamma)?;
        let mut values = [0.0; PATHWISE_COMPONENTS];
        values[PRICE] = workspace.primal(0)?.get(lane)?;
        values[DELTA] = workspace.primal(1)?.get(lane)?;
        values[VEGA] = workspace.primal(2)?.get(lane)?;
        values[GAMMA] = workspace.primal(3)?.get(lane)?;
        values[BUMP_DELTA] = bump_delta;
        values[DELTA_DIFFERENCE] = bump_delta - base.delta;
        values[BUMP_VEGA] = bump_vega;
        values[VEGA_DIFFERENCE] = bump_vega - base.vega;
        values[BUMP_GAMMA] = bump_gamma;
        values[GAMMA_DIFFERENCE] = bump_gamma - gamma;
        if let Some(diagnostics) = base.barrier_diagnostics {
            diagnostics.write_to(&mut values);
        }
        Ok(values)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn pathwise_aad(
        &self,
        normals: &[f64],
        spot: f64,
        volatility: f64,
    ) -> Result<PathwiseAad, MonteCarloError> {
        if let Some(barrier) = &self.continuous_barrier {
            return self.continuous_barrier_pathwise_aad(barrier, normals, spot, volatility);
        }
        if self.observation_dates.len() == 1
            && self.observation_dates[0] == Some(self.expiry)
            && self
                .observation_pre_dividend_coordinates
                .iter()
                .all(Option::is_none)
        {
            return self.pathwise_aad_single_terminal(normals[0], spot, volatility);
        }
        let observations = self.path_observations_from_normals(normals, spot, volatility);
        let payoff = self.payoff.evaluate_single_with_observation_adjoints(
            |underlying, date| {
                if underlying != self.underlying {
                    return None;
                }
                self.observation_dates
                    .iter()
                    .position(|observation_date| *observation_date == Some(date))
                    .map(|index| observations[index].post_spot)
            },
            |underlying, date| {
                if underlying != self.underlying {
                    return None;
                }
                self.observation_dates
                    .iter()
                    .position(|observation_date| *observation_date == Some(date))
                    .and_then(|index| observations[index].pre_dividend_spot)
            },
        )?;
        let price = self.discount * payoff.value;
        let mut delta = 0.0;
        let mut vega = 0.0;
        for adjoint in &payoff.terminal_adjoints {
            if adjoint.underlying != self.underlying {
                continue;
            }
            if let Some(index) = self
                .observation_dates
                .iter()
                .position(|observation_date| *observation_date == Some(adjoint.observation_date))
            {
                let observation = observations[index];
                let coordinate = self.observation_affine_coordinates[index];
                delta += adjoint.value * coordinate.b() * observation.canonical_f / spot;
                vega += adjoint.value
                    * coordinate.b()
                    * observation.canonical_f
                    * (-volatility * self.observation_times[index] + observation.brownian);
            }
        }
        for adjoint in &payoff.pre_dividend_adjoints {
            if adjoint.underlying != self.underlying {
                continue;
            }
            if let Some(index) = self
                .observation_dates
                .iter()
                .position(|observation_date| *observation_date == Some(adjoint.observation_date))
            {
                let observation = observations[index];
                let coordinate = self.observation_pre_dividend_coordinates[index]
                    .expect("pre-dividend adjoints have a matching coordinate");
                delta += adjoint.value * coordinate.b() * observation.canonical_f / spot;
                vega += adjoint.value
                    * coordinate.b()
                    * observation.canonical_f
                    * (-volatility * self.observation_times[index] + observation.brownian);
            }
        }
        Ok(PathwiseAad {
            price,
            delta: self.discount * delta,
            vega: self.discount * vega,
            barrier_diagnostics: None,
        })
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn pathwise_aad_for_output(
        &self,
        normals: &[f64],
        spot: f64,
        volatility: f64,
        output_index: usize,
        discount: f64,
    ) -> Result<PathwiseAad, MonteCarloError> {
        let observations = self.path_observations_from_normals(normals, spot, volatility);
        let payoff = self.payoff.evaluate_output_with_observation_adjoints(
            output_index,
            |underlying, date| {
                if underlying != self.underlying {
                    return None;
                }
                self.observation_dates
                    .iter()
                    .position(|observation_date| *observation_date == Some(date))
                    .map(|index| observations[index].post_spot)
            },
            |underlying, date| {
                if underlying != self.underlying {
                    return None;
                }
                self.observation_dates
                    .iter()
                    .position(|observation_date| *observation_date == Some(date))
                    .and_then(|index| observations[index].pre_dividend_spot)
            },
        )?;
        let mut delta = 0.0;
        let mut vega = 0.0;
        for adjoint in &payoff.terminal_adjoints {
            if adjoint.underlying != self.underlying {
                continue;
            }
            if let Some(index) = self
                .observation_dates
                .iter()
                .position(|observation_date| *observation_date == Some(adjoint.observation_date))
            {
                let observation = observations[index];
                let coordinate = self.observation_affine_coordinates[index];
                delta += adjoint.value * coordinate.b() * observation.canonical_f / spot;
                vega += adjoint.value
                    * coordinate.b()
                    * observation.canonical_f
                    * (-volatility * self.observation_times[index] + observation.brownian);
            }
        }
        for adjoint in &payoff.pre_dividend_adjoints {
            if adjoint.underlying != self.underlying {
                continue;
            }
            if let Some(index) = self
                .observation_dates
                .iter()
                .position(|observation_date| *observation_date == Some(adjoint.observation_date))
            {
                let observation = observations[index];
                let coordinate = self.observation_pre_dividend_coordinates[index]
                    .expect("pre-dividend adjoints have a matching coordinate");
                delta += adjoint.value * coordinate.b() * observation.canonical_f / spot;
                vega += adjoint.value
                    * coordinate.b()
                    * observation.canonical_f
                    * (-volatility * self.observation_times[index] + observation.brownian);
            }
        }
        Ok(PathwiseAad {
            price: discount * payoff.value,
            delta: discount * delta,
            vega: discount * vega,
            barrier_diagnostics: None,
        })
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn continuous_barrier_pathwise_aad(
        &self,
        barrier: &ContinuousBarrierRuntime,
        normals: &[f64],
        spot: f64,
        volatility: f64,
    ) -> Result<PathwiseAad, MonteCarloError> {
        let observations = self.path_observations_from_normals(normals, spot, volatility);
        let bridge = self.continuous_barrier_bridge(barrier, &observations, spot, volatility)?;
        let survival = bridge.survival();
        let terminal = observations[barrier.expiry_observation_index].post_spot;
        let payoff = continuous_barrier_payoff_terms(barrier, terminal, survival);
        let mut state_adjoints = vec![0.0; observations.len()];
        let terminal_coordinate =
            self.observation_affine_coordinates[barrier.expiry_observation_index];
        state_adjoints[barrier.expiry_observation_index] +=
            payoff.terminal_derivative * terminal_coordinate.b();

        let mut initial_state_adjoint = 0.0;
        let mut variance_adjoint = 0.0;
        match &bridge {
            ContinuousBarrierBridgeEvaluation::Exact(bridge)
                if !bridge.endpoint_touched && !bridge.dividend_jump_touched =>
            {
                let interval_adjoints = bridge.path.reverse(payoff.survival_derivative * survival);
                for ((left_index, right_index), adjoints) in bridge
                    .interval_observation_indices
                    .iter()
                    .copied()
                    .zip(interval_adjoints)
                {
                    if let Some(left_index) = left_index {
                        state_adjoints[left_index] += adjoints.left_state;
                    } else {
                        initial_state_adjoint += adjoints.left_state;
                    }
                    state_adjoints[right_index] += adjoints.right_state;
                    variance_adjoint +=
                        adjoints.left_local_variance + adjoints.right_local_variance;
                }
            }
            ContinuousBarrierBridgeEvaluation::Smoothed {
                path,
                interval_observation_indices,
            } => {
                let (interval_adjoints, hit_factor_adjoints) =
                    path.reverse(payoff.survival_derivative);
                for ((left_index, right_index), adjoints) in interval_observation_indices
                    .iter()
                    .copied()
                    .zip(interval_adjoints)
                {
                    if let Some(left_index) = left_index {
                        state_adjoints[left_index] += adjoints.left_endpoint.state;
                    } else {
                        initial_state_adjoint += adjoints.left_endpoint.state;
                    }
                    state_adjoints[right_index] += adjoints.right_endpoint.state;
                    variance_adjoint +=
                        adjoints.left_local_variance + adjoints.right_local_variance;
                }
                for (state_index, adjoint) in hit_factor_adjoints {
                    if let Some(index) = state_index {
                        state_adjoints[index] += adjoint;
                    } else {
                        initial_state_adjoint += adjoint;
                    }
                }
            }
            ContinuousBarrierBridgeEvaluation::Exact(_) => {}
        }

        let mut delta = initial_state_adjoint;
        let mut vega = 2.0 * volatility * variance_adjoint;
        for (index, (state_adjoint, observation)) in state_adjoints
            .iter()
            .copied()
            .zip(observations.iter().copied())
            .enumerate()
        {
            delta += state_adjoint * observation.canonical_f / spot;
            vega += state_adjoint
                * observation.canonical_f
                * (-volatility * self.observation_times[index] + observation.brownian);
        }
        Ok(PathwiseAad {
            price: self.discount * payoff.value,
            delta: self.discount * delta,
            vega: self.discount * vega,
            barrier_diagnostics: Some(bridge.diagnostic_values()),
        })
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn pathwise_aad_single_terminal(
        &self,
        normal: f64,
        spot: f64,
        volatility: f64,
    ) -> Result<PathwiseAad, MonteCarloError> {
        let total_variance = volatility * volatility * self.time;
        let standard_deviation = total_variance.sqrt();
        let log_return = -0.5 * total_variance + standard_deviation * normal;
        let bumped_forward = self.forward * (spot / self.spot);
        let canonical_f = bumped_forward * log_return.exp();
        let coordinate = self.observation_affine_coordinates[0];
        let terminal = coordinate.a() * self.spot + coordinate.b() * canonical_f;
        let payoff = self
            .payoff
            .evaluate_single_with_terminal_adjoint(|underlying, date| {
                (underlying == self.underlying && date == self.expiry).then_some(terminal)
            })?;
        let terminal_adjoint = payoff
            .terminal_adjoints
            .iter()
            .filter(|adjoint| {
                adjoint.underlying == self.underlying && adjoint.observation_date == self.expiry
            })
            .fold(0.0, |total, adjoint| total + adjoint.value);
        let price = self.discount * payoff.value;
        let delta = self.discount * terminal_adjoint * coordinate.b() * canonical_f / spot;
        let terminal_vega =
            coordinate.b() * canonical_f * (-volatility * self.time + self.time.sqrt() * normal);
        let vega = self.discount * terminal_adjoint * terminal_vega;
        Ok(PathwiseAad {
            price,
            delta,
            vega,
            barrier_diagnostics: None,
        })
    }
}
