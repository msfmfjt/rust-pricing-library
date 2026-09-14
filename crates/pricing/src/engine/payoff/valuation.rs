//! payoff / valuation implementation.

use crate::core::PathIndex;

use crate::MonteCarloError;
use crate::engine::plan::simulation::EarlyExerciseRuntime;
use crate::engine::plan::simulation::ExerciseCashflowSelection;
use crate::engine::plan::simulation::LocalVolObservation;
use crate::engine::plan::simulation::PathObservation;
use crate::engine::plan::simulation::SimulationPlan;
use crate::mc::LocalVolTimeGrid;

impl SimulationPlan {
    pub(crate) fn lsv_time_grid(&self) -> Result<&LocalVolTimeGrid, MonteCarloError> {
        self.local_volatility
            .as_ref()
            .map(|lv| lv.plan.time_grid())
            .ok_or(MonteCarloError::UnsupportedModel {
                model: "LSV calibration requires a LocalVolatility target",
            })
    }
}

impl SimulationPlan {
    pub(crate) fn hybrid_observation_times(&self) -> &[f64] {
        &self.observation_times
    }
}

impl SimulationPlan {
    /// Exact primal observations of the default proportional-dividend map.
    pub(crate) fn hybrid_proportional_spots(
        &self,
        times: &[f64],
        equity: &[f64],
    ) -> Result<Vec<(f64, Option<f64>)>, MonteCarloError> {
        if times.len() != equity.len() {
            return Err(crate::models::HullWhiteError::InvalidInput {
                field: "hybrid_spot_count",
                index: equity.len(),
            }
            .into());
        }
        let mut spots = vec![(0.0, None); times.len()];
        for (i, t) in self.observation_times.iter().enumerate() {
            let index = times.binary_search_by(|v| v.total_cmp(t)).map_err(|_| {
                crate::models::HullWhiteError::InvalidInput {
                    field: "missing_observation_time",
                    index: i,
                }
            })?;
            let c = self.observation_affine_coordinates[i];
            let f = equity[index] * self.observation_forwards[i] / self.spot;
            spots[index] = (
                c.a() * self.spot + c.b() * f,
                self.observation_pre_dividend_coordinates[i].map(|c| c.a() * self.spot + c.b() * f),
            );
        }
        Ok(spots)
    }
}

impl SimulationPlan {
    /// Evaluate on reconstructed physical Spot without another affine mapping.
    pub(crate) fn hybrid_spot_payoff_adjoints(
        &self,
        times: &[f64],
        spots: &[(f64, Option<f64>)],
    ) -> Result<(f64, Vec<(f64, f64)>), MonteCarloError> {
        if times.len() != spots.len() {
            return Err(crate::models::HullWhiteError::InvalidInput {
                field: "hybrid_spot_count",
                index: spots.len(),
            }
            .into());
        }
        let indices = self
            .observation_times
            .iter()
            .map(|t| {
                times.binary_search_by(|x| x.total_cmp(t)).map_err(|_| {
                    crate::models::HullWhiteError::InvalidInput {
                        field: "missing_observation_time",
                        index: 0,
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let at = |u, d, pre| {
            if u != self.underlying {
                return None;
            }
            self.observation_dates
                .iter()
                .position(|date| *date == Some(d))
                .and_then(|i| {
                    if pre {
                        spots[indices[i]].1
                    } else {
                        Some(spots[indices[i]].0)
                    }
                })
        };
        let payoff = self.payoff.evaluate_single_with_observation_adjoints(
            |u, d| at(u, d, false),
            |u, d| at(u, d, true),
        )?;
        let mut seeds = vec![(0.0, 0.0); times.len()];
        for (pre, underlying, date, value) in payoff
            .terminal_adjoints
            .iter()
            .map(|a| (false, a.underlying, a.observation_date, a.value))
            .chain(
                payoff
                    .pre_dividend_adjoints
                    .iter()
                    .map(|a| (true, a.underlying, a.observation_date, a.value)),
            )
        {
            if underlying == self.underlying
                && let Some(i) = self.observation_dates.iter().position(|d| *d == Some(date))
            {
                if pre {
                    seeds[indices[i]].1 += self.discount * value;
                } else {
                    seeds[indices[i]].0 += self.discount * value;
                }
            }
        }
        Ok((self.discount * payoff.value, seeds))
    }
}

impl SimulationPlan {
    /// Evaluate on reconstructed physical Spot without another affine mapping.
    pub(crate) fn hybrid_spot_payoff(
        &self,
        times: &[f64],
        spots: &[(f64, Option<f64>)],
    ) -> Result<f64, MonteCarloError> {
        if times.len() != spots.len() {
            return Err(crate::models::HullWhiteError::InvalidInput {
                field: "hybrid_spot_count",
                index: spots.len(),
            }
            .into());
        }
        let indices = self
            .observation_times
            .iter()
            .map(|t| {
                times.binary_search_by(|x| x.total_cmp(t)).map_err(|_| {
                    crate::models::HullWhiteError::InvalidInput {
                        field: "missing_observation_time",
                        index: 0,
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let at = |u, d, pre| {
            if u != self.underlying {
                return None;
            }
            self.observation_dates
                .iter()
                .position(|date| *date == Some(d))
                .and_then(|i| {
                    if pre {
                        spots[indices[i]].1
                    } else {
                        Some(spots[indices[i]].0)
                    }
                })
        };
        Ok(self.discount
            * self
                .payoff
                .evaluate_with_pre_dividend_spots(|u, d| at(u, d, false), |u, d| at(u, d, true))?
                [0])
    }
}

impl SimulationPlan {
    /// Map normalized hybrid equity states back to contractual Spot. The HW
    /// default compiler rejects cash dividends; proportional pre/post observations use
    /// the same compiled payoff graph and event convention as the stable API.
    pub(crate) fn hybrid_payoff(
        &self,
        times: &[f64],
        normalized_states: &[f64],
    ) -> Result<f64, MonteCarloError> {
        if times.len() != normalized_states.len() {
            return Err(crate::models::HullWhiteError::InvalidInput {
                field: "hybrid_state_count",
                index: normalized_states.len(),
            }
            .into());
        }
        let indices = self
            .observation_times
            .iter()
            .map(|t| {
                times.binary_search_by(|x| x.total_cmp(t)).map_err(|_| {
                    crate::models::HullWhiteError::InvalidInput {
                        field: "missing_observation_time",
                        index: 0,
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let value_at = |underlying, date, pre: bool| {
            if underlying != self.underlying {
                return None;
            }
            self.observation_dates
                .iter()
                .position(|d| *d == Some(date))
                .and_then(|i| {
                    let coordinate = if pre {
                        self.observation_pre_dividend_coordinates[i]
                    } else {
                        Some(self.observation_affine_coordinates[i])
                    }?;
                    let f =
                        normalized_states[indices[i]] * self.observation_forwards[i] / self.spot;
                    Some(coordinate.a() * self.spot + coordinate.b() * f)
                })
        };
        Ok(self.discount
            * self.payoff.evaluate_with_pre_dividend_spots(
                |u, d| value_at(u, d, false),
                |u, d| value_at(u, d, true),
            )?[0])
    }
}

impl SimulationPlan {
    /// Reuse the contractual graph, dividend ordering and observation mapping.
    /// This returns f-state seeds, not a Spot Delta or a market-IV Vega.
    pub(crate) fn lsv_payoff(
        &self,
        states: &[f64],
        path: PathIndex,
        with_reverse: bool,
    ) -> Result<(f64, Option<Vec<f64>>), MonteCarloError> {
        let runtime = self
            .local_volatility
            .as_ref()
            .ok_or(MonteCarloError::UnsupportedModel {
                model: "LSV target",
            })?;
        if states.len() != runtime.plan.time_grid().nodes().len() {
            return Err(crate::mc::lsv::LsvError::LengthMismatch {
                field: "payoff_states",
                expected: runtime.plan.time_grid().nodes().len(),
                actual: states.len(),
            }
            .into());
        }
        if let (Some(dividends), Some(schedule)) = (&runtime.dividends, &runtime.dividend_schedule)
        {
            for (i, checkpoint) in schedule.checkpoints().iter().enumerate() {
                dividends.validate_post_event_f_state(i, path, states[checkpoint.node_index()])?;
            }
        }
        let observations = self.local_vol_state_observations(runtime, states, self.spot)?;
        let post = |underlying, date| {
            if underlying != self.underlying {
                return None;
            }
            self.observation_dates
                .iter()
                .position(|d| *d == Some(date))
                .map(|i| observations[i].post_spot)
        };
        let pre = |underlying, date| {
            if underlying != self.underlying {
                return None;
            }
            self.observation_dates
                .iter()
                .position(|d| *d == Some(date))
                .and_then(|i| observations[i].pre_dividend_spot)
        };
        if !with_reverse {
            return Ok((
                self.discount * self.payoff.evaluate_with_pre_dividend_spots(post, pre)?[0],
                None,
            ));
        }
        let payoff = self
            .payoff
            .evaluate_single_with_observation_adjoints(post, pre)?;
        let mut seeds = vec![0.0; states.len()];
        for a in &payoff.terminal_adjoints {
            if a.underlying == self.underlying
                && let Some(i) = self
                    .observation_dates
                    .iter()
                    .position(|d| *d == Some(a.observation_date))
            {
                seeds[observations[i].node_index] +=
                    self.discount * a.value * self.observation_affine_coordinates[i].b();
            }
        }
        for a in &payoff.pre_dividend_adjoints {
            if a.underlying == self.underlying
                && let Some(i) = self
                    .observation_dates
                    .iter()
                    .position(|d| *d == Some(a.observation_date))
            {
                seeds[observations[i].node_index] += self.discount
                    * a.value
                    * self.observation_pre_dividend_coordinates[i]
                        .expect("validated pre-dividend coordinate")
                        .b();
            }
        }
        Ok((self.discount * payoff.value, Some(seeds)))
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn payoff_outputs_from_observations(
        &self,
        observations: &[PathObservation],
    ) -> Result<Vec<f64>, MonteCarloError> {
        Ok(self.payoff.evaluate_with_pre_dividend_spots(
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
        )?)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn payoff_outputs_from_local_vol_observations(
        &self,
        observations: &[LocalVolObservation],
    ) -> Result<Vec<f64>, MonteCarloError> {
        Ok(self.payoff.evaluate_with_pre_dividend_spots(
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
        )?)
    }
}

impl SimulationPlan {
    pub(in crate::engine) fn exercise_cashflow_selection(
        &self,
        early_exercise: &EarlyExerciseRuntime,
        stopping_index: usize,
    ) -> Result<ExerciseCashflowSelection, MonteCarloError> {
        let discount = *early_exercise.discount_factors.get(stopping_index).ok_or(
            crate::product::GraphError::InvalidOutputIndex {
                index: stopping_index,
                count: early_exercise.exercise_dates.len(),
            },
        )?;
        Ok(ExerciseCashflowSelection {
            output_index: stopping_index,
            discount,
        })
    }
}
