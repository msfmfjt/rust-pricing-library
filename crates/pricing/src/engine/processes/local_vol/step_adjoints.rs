use super::{LocalVolError, validate_adjoint};

/// A transition's local sensitivities before projection onto the variance grid.
/// The recovery basis uses the same spatial hats as that grid but keeps each
/// simulation interval separate: there is no interpolation between time rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalVolStepAdjoint {
    local_variance_adjoint: f64,
    local_volatility_adjoint: f64,
    lower_log_moneyness_index: usize,
    log_moneyness_weight: f64,
    spatial_node_count: usize,
}

impl LocalVolStepAdjoint {
    pub(super) fn new(
        local_variance_adjoint: f64,
        local_volatility_adjoint: f64,
        lower_log_moneyness_index: usize,
        log_moneyness_weight: f64,
        spatial_node_count: usize,
    ) -> Self {
        Self {
            local_variance_adjoint,
            local_volatility_adjoint,
            lower_log_moneyness_index,
            log_moneyness_weight,
            spatial_node_count,
        }
    }

    /// Derivative with respect to the effective variance used by this step.
    #[must_use]
    pub const fn local_variance_adjoint(self) -> f64 {
        self.local_variance_adjoint
    }

    /// Paper equation (10), per unit absolute local volatility at this step.
    /// This is not a derivative with respect to a node of the original grid:
    /// nodal-volatility conversion also needs interpolation weights and the
    /// ratio of nodal volatility to the effective volatility of this step.
    #[must_use]
    pub const fn local_volatility_adjoint(self) -> f64 {
        self.local_volatility_adjoint
    }

    /// Deposit integrated local Vega onto this interval's spatial recovery row.
    /// Do not divide by hat areas here: equation-(11) recovery does that later.
    /// Flat spatial extrapolation deposits into the nearest endpoint. A length
    /// error or nonfinite sum leaves the row unchanged.
    pub fn accumulate_local_volatility_nodes(self, row: &mut [f64]) -> Result<(), LocalVolError> {
        if row.len() != self.spatial_node_count {
            return Err(LocalVolError::AdjointCountMismatch {
                expected: self.spatial_node_count,
                actual: row.len(),
            });
        }
        let left = self.lower_log_moneyness_index;
        let right = left + 1;
        let left_value =
            row[left] + self.local_volatility_adjoint * (1.0 - self.log_moneyness_weight);
        let right_value = row[right] + self.local_volatility_adjoint * self.log_moneyness_weight;
        validate_adjoint(left_value, left)?;
        validate_adjoint(right_value, right)?;
        row[left] = left_value;
        row[right] = right_value;
        Ok(())
    }
}
