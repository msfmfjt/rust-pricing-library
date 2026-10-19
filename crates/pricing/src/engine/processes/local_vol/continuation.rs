use super::{LocalVolError, LocalVolLogEulerPlan};
use crate::market::LocalVarianceGrid;

impl LocalVolLogEulerPlan {
    /// Fix the first positive-time state exactly and evolve only later intervals.
    /// Returns original-index states [S0, fixed S1, S2, ...]. There is no first
    /// shock and no synthetic first transition. This is the canonical coordinate;
    /// affine dividend reconstruction/checks are not performed by this method.
    pub fn continue_states_from_first_node(
        &self,
        grid: &LocalVarianceGrid,
        initial_state: f64,
        first_state: f64,
        remaining_shocks: &[f64],
    ) -> Result<Vec<f64>, LocalVolError> {
        for state in [initial_state, first_state] {
            if !state.is_finite() || state <= 0.0 {
                return Err(LocalVolError::InvalidInitialState {
                    bits: state.to_bits(),
                });
            }
        }
        let expected = self.time_grid().step_count() - 1;
        if remaining_shocks.len() != expected {
            return Err(LocalVolError::ShockCountMismatch {
                expected,
                actual: remaining_shocks.len(),
            });
        }
        let mut states = Vec::with_capacity(self.time_grid().nodes().len());
        states.extend([initial_state, first_state]);
        let mut state = first_state;
        for (offset, &shock) in remaining_shocks.iter().enumerate() {
            let step = offset + 1;
            let time = self.time_grid().nodes()[step];
            let dt = self.time_grid().nodes()[step + 1] - time;
            let variance = grid
                .interpolate(time, (state / self.forward_normalizers()[step]).ln())?
                .value;
            let exponential = self.forward_normalizers()[step + 1]
                / self.forward_normalizers()[step]
                * (-0.5 * variance * dt + variance.sqrt() * dt.sqrt() * shock).exp();
            state *= exponential;
            if !state.is_finite() || state <= 0.0 {
                return Err(LocalVolError::NonFiniteState {
                    step: step + 1,
                    bits: state.to_bits(),
                });
            }
            states.push(state);
        }
        Ok(states)
    }
}
