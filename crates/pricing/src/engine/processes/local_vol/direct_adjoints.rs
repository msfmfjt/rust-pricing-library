use super::{
    LocalVolError, LocalVolLogEulerPlan, LocalVolPath, LocalVolReverseAdjoints,
    LocalVolStepAdjoint, validate_adjoint,
};
use crate::market::LocalVarianceGrid;

/// Explicit payoff sensitivity at an actual simulation node, including the
/// terminal node. This is not a transition and has no assigned interval length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalVolDirectVarianceAdjoint {
    maturity: f64,
    adjoint: LocalVolStepAdjoint,
}
impl LocalVolDirectVarianceAdjoint {
    #[must_use]
    pub const fn maturity(self) -> f64 {
        self.maturity
    }
    #[must_use]
    pub const fn local_variance_adjoint(self) -> f64 {
        self.adjoint.local_variance_adjoint()
    }
    #[must_use]
    pub const fn local_volatility_adjoint(self) -> f64 {
        self.adjoint.local_volatility_adjoint()
    }
    pub fn accumulate_local_volatility_nodes(self, row: &mut [f64]) -> Result<(), LocalVolError> {
        self.adjoint.accumulate_local_volatility_nodes(row)
    }
}

/// Complete reverse for a payoff depending on states and effective local
/// variances. Direct variance dependence is retained separately from transitions.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalVolPayoffReverseAdjoints {
    transition: LocalVolReverseAdjoints,
    direct: Box<[LocalVolDirectVarianceAdjoint]>,
    grid_adjoints: Box<[f64]>,
}
impl LocalVolPayoffReverseAdjoints {
    /// Includes state feedback from direct variance seeds, but not their direct
    /// grid deposition. Do not add that state feedback a second time.
    #[must_use]
    pub const fn transition(&self) -> &LocalVolReverseAdjoints {
        &self.transition
    }
    #[must_use]
    pub fn direct_variance_adjoints(&self) -> &[LocalVolDirectVarianceAdjoint] {
        &self.direct
    }
    /// Combined transition and direct derivatives with respect to grid variance.
    #[must_use]
    pub fn local_variance_value_adjoints(&self) -> &[f64] {
        &self.grid_adjoints
    }
}

impl LocalVolLogEulerPlan {
    pub(crate) fn validate_cached_path(
        &self,
        grid: &LocalVarianceGrid,
        path: &LocalVolPath,
    ) -> Result<(), LocalVolError> {
        let times = self.time_grid().nodes();
        if path.states().len() != times.len() {
            return Err(LocalVolError::AdjointCountMismatch {
                expected: times.len(),
                actual: path.states().len(),
            });
        }
        for (i, cache) in path.step_cache().iter().enumerate() {
            let dt = times[i + 1] - times[i];
            let x = (cache.state_before / self.forward_normalizers()[i]).ln();
            let expected = grid.interpolate(times[i], x)?;
            let derivative =
                grid.interpolation_log_moneyness_derivative(expected) / cache.state_before;
            let exponential = self.forward_normalizers()[i + 1] / self.forward_normalizers()[i]
                * (-0.5 * expected.value * dt + expected.value.sqrt() * dt.sqrt() * cache.shock)
                    .exp();
            if cache.dt.to_bits() != dt.to_bits()
                || cache.interpolation != expected
                || cache.local_variance != expected.value
                || cache.local_variance_state_derivative != derivative
                || cache.exponential != exponential
            {
                return Err(LocalVolError::CachedPathMismatch { step: i });
            }
        }
        Ok(())
    }

    /// `state_seeds` are partial payoff derivatives holding node variances fixed.
    /// `variance_seeds` are partial derivatives with respect to effective variance
    /// at every simulation node (N+1), including time zero and the terminal node.
    /// Both must include payoff discounting. State feedback through variance is
    /// added here exactly once; callers must not pre-apply it.
    pub fn reverse_path_with_variance_seeds(
        &self,
        grid: &LocalVarianceGrid,
        path: &LocalVolPath,
        state_seeds: &[f64],
        variance_seeds: &[f64],
    ) -> Result<LocalVolPayoffReverseAdjoints, LocalVolError> {
        self.validate_cached_path(grid, path)?;
        for seeds in [state_seeds, variance_seeds] {
            if seeds.len() != path.states().len() {
                return Err(LocalVolError::AdjointCountMismatch {
                    expected: path.states().len(),
                    actual: seeds.len(),
                });
            }
            for (i, &v) in seeds.iter().enumerate() {
                validate_adjoint(v, i)?;
            }
        }
        let width = grid.log_moneyness_nodes().len();
        let mut states = state_seeds.to_vec();
        let mut direct_grid = vec![0.0; grid.values().len()];
        let mut direct = Vec::with_capacity(states.len());
        for (i, &seed) in variance_seeds.iter().enumerate() {
            let time = self.time_grid().nodes()[i];
            let interpolation = grid.interpolate(
                time,
                (path.states()[i] / self.forward_normalizers()[i]).ln(),
            )?;
            // Preserve the original bridge chain-rule arithmetic order.
            states[i] += seed * grid.interpolation_log_moneyness_derivative(interpolation)
                / path.states()[i];
            validate_adjoint(states[i], i)?;
            let volatility_adjoint = seed * (2.0 * interpolation.value.sqrt());
            validate_adjoint(volatility_adjoint, i)?;
            interpolation.transpose_accumulate(seed, &mut direct_grid, width);
            direct.push(LocalVolDirectVarianceAdjoint {
                maturity: time,
                adjoint: LocalVolStepAdjoint::new(
                    seed,
                    volatility_adjoint,
                    interpolation.lower_log_moneyness_index,
                    interpolation.log_moneyness_weight,
                    width,
                ),
            });
        }
        let transition = path.reverse_state_adjoints(&states, grid.values().len(), width)?;
        for (i, (direct, &value)) in direct_grid
            .iter_mut()
            .zip(transition.local_variance_value_adjoints())
            .enumerate()
        {
            *direct += value;
            validate_adjoint(*direct, i)?;
        }
        Ok(LocalVolPayoffReverseAdjoints {
            transition,
            direct: direct.into_boxed_slice(),
            grid_adjoints: direct_grid.into_boxed_slice(),
        })
    }
}
