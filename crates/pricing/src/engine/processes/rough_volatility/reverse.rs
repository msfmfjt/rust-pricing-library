//! Reverse of the initial forward, with all model parameters and shocks fixed.
//! Volatility histories are independent of this initial state. This does not
//! return partial shock adjoints disguised as full stochastic-driver risk.
use super::*;

#[derive(Clone, Debug)]
pub struct RoughVolatilityRecordedPath {
    path: RoughVolatilityPath,
    state_derivatives: Box<[f64]>,
}
impl RoughVolatilityPathPlan {
    pub fn evolve_recorded_path(
        &self,
        initial_forward: f64,
        normals: &[f64],
    ) -> Result<RoughVolatilityRecordedPath, HullWhiteError> {
        let path = self.evolve_path(initial_forward, normals)?;
        let beta = match &self.model {
            RoughVolatilityModel::RoughSabr(m) => m.beta,
            _ => 1.0,
        };
        if initial_forward == 0.0 && beta > 0.0 && beta < 1.0 {
            return Err(invalid("rough_absorbing_initial_derivative"));
        }
        let mut state_derivatives = Vec::with_capacity(self.times.len() - 1);
        for (j, &normal) in normals.iter().enumerate().take(self.times.len() - 1) {
            let previous = path.forwards[j];
            let next = path.forwards[j + 1];
            let d = if beta == 1.0 {
                next / previous
            } else if beta == 0.0 {
                1.0
            } else if previous == 0.0 {
                0.0
            } else {
                let noise =
                    path.variances[j].sqrt() * (self.times[j + 1] - self.times[j]).sqrt() * normal;
                let proposal = previous + previous.powf(beta) * noise;
                if proposal == 0.0 {
                    return Err(invalid("rough_absorption_derivative_kink"));
                }
                if proposal < 0.0 {
                    0.0
                } else {
                    1.0 + beta * previous.powf(beta - 1.0) * noise
                }
            };
            state_derivatives.push(finite_value(d)?);
        }
        Ok(RoughVolatilityRecordedPath {
            path,
            state_derivatives: state_derivatives.into_boxed_slice(),
        })
    }
}
impl RoughVolatilityRecordedPath {
    #[must_use]
    pub fn path(&self) -> &RoughVolatilityPath {
        &self.path
    }
    /// VJP of all forward observations with respect to their initial forward.
    /// Kinks in the payoff are a separate responsibility of its compiled tape.
    pub fn reverse_initial_forward(&self, seeds: &[f64]) -> Result<f64, HullWhiteError> {
        if seeds.len() != self.path.forwards.len() || seeds.iter().any(|s| !s.is_finite()) {
            return Err(invalid("rough_forward_seed_shape_or_value"));
        }
        let mut bar = seeds[seeds.len() - 1];
        for j in (0..self.state_derivatives.len()).rev() {
            bar = finite_value(seeds[j] + bar * self.state_derivatives[j])?;
        }
        Ok(bar)
    }
}
