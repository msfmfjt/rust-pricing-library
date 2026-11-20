//! Out-of-sample Longstaff–Schwartz valuation with pathwise currency and discount factors.
use super::joint_path::{JointPlan, PathUse};
use super::{CrossCurrencyEquityOption, EquityFxSettlement, sampling};
use crate::mc::{
    EngineConfig, ExecutionPolicy, ExerciseDecisionModel, LsmConfig, LsmStateVariable,
    fit_exercise_decision,
};
use crate::multi_asset::MultiAssetError as Error;
use crate::product::{AmericanVanillaSpec, OptionSide};
use crate::{Estimate, EstimatorKind};

#[derive(Clone, Copy)]
struct ExerciseNode {
    spot: f64,
    fx: f64,
    factors: [f64; 2],
    immediate: f64,
    weight: f64,
}

impl ExerciseNode {
    fn features(&self, states: &[LsmStateVariable]) -> impl Iterator<Item = f64> {
        states.iter().map(|s| match s {
            LsmStateVariable::Spot => self.spot,
            LsmStateVariable::FxSpot => self.fx,
            LsmStateVariable::DomesticRateFactor => self.factors[0],
            LsmStateVariable::ForeignRateFactor => self.factors[1],
        })
    }

    fn pv(&self) -> f64 {
        self.immediate * self.weight
    }

    fn exercise(&self, features: &[f64], policy: &ExerciseDecisionModel) -> Result<bool, Error> {
        Ok(self.immediate > 0.
            && policy
                .should_exercise(self.immediate, features)
                .map_err(Error::numerical)?)
    }
}

pub(super) fn intrinsic(side: OptionSide, spot: f64, strike: f64) -> f64 {
    match side {
        OptionSide::Call => (spot - strike).max(0.),
        OptionSide::Put => (strike - spot).max(0.),
    }
}

impl CrossCurrencyEquityOption {
    fn exercise_node(
        &self,
        p: &AmericanVanillaSpec,
        stock: f64,
        fx: f64,
        discount: f64,
        factors: [f64; 2],
    ) -> ExerciseNode {
        let (spot, conversion) = match self.settlement {
            EquityFxSettlement::Quanto { fixed_fx_rate } => (stock, fixed_fx_rate.get()),
            EquityFxSettlement::FloatingFx => (stock, fx),
            EquityFxSettlement::Composite => (stock * fx, 1.),
        };
        ExerciseNode {
            spot,
            fx,
            factors,
            immediate: p.notional().get() * intrinsic(p.side(), spot, p.strike().get()),
            weight: discount * conversion,
        }
    }

    pub(super) fn price_joint_exercise(
        &self,
        p: &AmericanVanillaSpec,
        engine: EngineConfig,
        execution: ExecutionPolicy,
        config: &LsmConfig,
    ) -> Result<Estimate, Error> {
        config
            .validate_independent_from(engine)
            .map_err(Error::numerical)?;
        let dates = p.exercise_dates();
        if dates[0] < self.valuation_date {
            return Err(Error::Invalid(
                "past exercise decisions must be resolved before pricing",
            ));
        }
        if p.expiry() == self.valuation_date {
            let v = self
                .exercise_node(
                    p,
                    self.foreign_market.forward().spot().get(),
                    self.fx_spot.get(),
                    1.,
                    [0.; 2],
                )
                .pv();
            return Estimate::new(v, 0., v, v, EstimatorKind::Analytical, 1)
                .map_err(Error::numerical);
        }
        let plan = JointPlan::for_option(self, dates.iter().copied(), p.expiry(), vec![])?;
        let indices: Vec<_> = dates.iter().map(|d| plan.dates[d]).collect();
        let observations = |z: &[f64]| -> Result<Vec<ExerciseNode>, Error> {
            let path = plan.simulate(z, PathUse::Exercise)?;
            Ok(indices
                .iter()
                .map(|&i| {
                    self.exercise_node(
                        p,
                        path.spots[i].0,
                        path.fx[i],
                        path.discounts[i],
                        path.factors[i],
                    )
                })
                .collect())
        };
        let states = config.state_variables();
        let terminal = dates.len() - 1;
        let mut policy = Vec::new();
        if terminal > 0 {
            let mut training = Vec::new();
            let count = usize::try_from(config.training_trajectory_count())
                .map_err(|_| Error::Invalid("exercise training size exceeds addressable memory"))?;
            training
                .try_reserve_exact(count)
                .map_err(Error::numerical)?;
            sampling::training_paths(&plan.times, plan.layout, config.training_engine(), |z| {
                training.push(observations(z)?);
                Ok(())
            })?;
            let mut future_pv: Vec<_> = training.iter().map(|p| p[terminal].pv()).collect();
            for t in (0..terminal).rev() {
                let nodes = || training.iter().map(|p| &p[t]);
                let features: Vec<_> = nodes().flat_map(|n| n.features(states)).collect();
                let cash: Vec<_> = nodes().map(|n| n.immediate).collect();
                // Convert the future settlement PV back into exercise-date payoff units.
                // For floating exercise the divisor includes that path's contemporaneous FX.
                let targets: Vec<_> = nodes().zip(&future_pv).map(|(n, v)| v / n.weight).collect();
                let decision = fit_exercise_decision(
                    config.basis().clone(),
                    &features,
                    count,
                    &cash,
                    &targets,
                    config.itm_abs_tolerance(),
                    config.cpqr_config(),
                    config.max_matrix_elements(),
                )
                .map_err(Error::numerical)?
                .into_decision();
                for ((n, f), v) in nodes()
                    .zip(features.chunks_exact(states.len()))
                    .zip(&mut future_pv)
                {
                    if n.exercise(f, &decision)? {
                        *v = n.pv();
                    }
                }
                policy.push(decision);
            }
            policy.reverse();
        }
        sampling::estimate(&plan.times, plan.layout, engine, execution, |z| {
            let nodes = observations(z)?;
            for (node, decision) in nodes.iter().zip(&policy) {
                if node.exercise(&node.features(states).collect::<Vec<_>>(), decision)? {
                    return Ok(node.pv());
                }
            }
            Ok(nodes[terminal].pv())
        })
    }
}
