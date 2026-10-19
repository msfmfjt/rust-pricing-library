//! Finite-grid reverse for the existing fixed-driver quadratic model.
use super::hurst::PowerKernelHurst;
use super::*;
use pricing_numerics::NeumaierSum;

const NAMES: [&str; 6] = [
    "initial_state",
    "mean_reversion",
    "vol_of_vol",
    "quadratic",
    "shift",
    "variance_floor",
];

/// Cached reverse plan. Leverage is NOT inserted into the feedback-state SDE.
/// Hurst is fixed unless selected; H=1/2 then uses its left derivative.
#[derive(Clone, Debug)]
pub struct QuadraticHestonMcRiskPlan {
    base: RoughVolatilityPathPlan,
    hurst: Option<PowerKernelHurst>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct QuadraticHestonMcAdjoints {
    pub initial_forward: f64,
    /// Natural scalar coordinates, optionally followed by Hurst.
    pub parameters: Box<[f64]>,
}
#[derive(Debug)]
pub struct QuadraticHestonMcRecordedPath<'a> {
    plan: &'a QuadraticHestonMcRiskPlan,
    normals: &'a [f64],
    path: RoughVolatilityPath,
}
impl QuadraticHestonMcRiskPlan {
    pub fn compile(
        base: &RoughVolatilityPathPlan,
        include_hurst: bool,
    ) -> Result<Self, HullWhiteError> {
        let RoughVolatilityModel::QuadraticRoughHeston(m) = &base.model else {
            return Err(HullWhiteError::Unsupported {
                feature: "parameter reverse requires Quadratic rough Heston",
            });
        };
        let CompiledDriver::Power(k) = &base.driver else {
            return Err(invalid("rough_compiled_driver_mismatch"));
        };
        Ok(Self {
            base: base.clone(),
            hurst: if include_hurst {
                Some(PowerKernelHurst::compile(m.hurst, &base.times, k)?)
            } else {
                None
            },
        })
    }
    #[must_use]
    pub fn path_plan(&self) -> &RoughVolatilityPathPlan {
        &self.base
    }
    #[must_use]
    pub fn parameter_names(&self) -> Vec<String> {
        let mut names = NAMES.iter().map(|x| (*x).to_string()).collect::<Vec<_>>();
        if self.hurst.is_some() {
            names.push("hurst".into());
        }
        names
    }
    pub(in crate::engine) fn width(&self) -> usize {
        6 + usize::from(self.hurst.is_some())
    }
    pub fn evolve_path<'a>(
        &'a self,
        initial_forward: f64,
        normals: &'a [f64],
    ) -> Result<QuadraticHestonMcRecordedPath<'a>, HullWhiteError> {
        let path = self.base.evolve_path(initial_forward, normals)?;
        self.check_variances(&path.variances)?;
        Ok(QuadraticHestonMcRecordedPath {
            plan: self,
            normals,
            path,
        })
    }
    fn check_variances(&self, variance: &[f64]) -> Result<(), HullWhiteError> {
        if variance.len() != self.base.times.len()
            || variance.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(invalid("quadratic_risk_variance_shape_or_value"));
        }
        // No artificial floor. The full parameter scope differentiates the
        // square root with respect to variance; it is singular at exact zero.
        if variance[..variance.len() - 1].contains(&0.0) {
            return Err(invalid("quadratic_risk_zero_variance"));
        }
        Ok(())
    }
    pub(in crate::engine) fn reverse(
        &self,
        normals: &[f64],
        seeds: &[f64],
    ) -> Result<Vec<f64>, HullWhiteError> {
        if normals.len() != self.base.random_dimension() as usize
            || normals.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("rough_normal_shape_or_value"));
        }
        let history = self.base.variance_history(normals)?;
        self.variance_reverse(normals, &history.variances, &history.latent_states, seeds)
    }
    fn variance_reverse(
        &self,
        normals: &[f64],
        variance: &[f64],
        latent: &[f64],
        seeds: &[f64],
    ) -> Result<Vec<f64>, HullWhiteError> {
        let RoughVolatilityModel::QuadraticRoughHeston(m) = &self.base.model else {
            return Err(invalid("quadratic_risk_model"));
        };
        let CompiledDriver::Power(k) = &self.base.driver else {
            return Err(invalid("rough_compiled_driver_mismatch"));
        };
        let times = &self.base.times;
        let n = times.len() - 1;
        if seeds.len() != n + 1
            || latent.len() != n + 1
            || seeds.iter().chain(latent).any(|v| !v.is_finite())
        {
            return Err(invalid("quadratic_risk_seed_shape_or_value"));
        }
        self.check_variances(variance)?;
        let (dw, near) = self.base.innovations(k, 1.0, normals, true);
        let mut bv = seeds.to_vec();
        let mut bz = vec![0.0; n + 1];
        let mut g = vec![NeumaierSum::new(); self.width()];
        for i in (0..=n).rev() {
            let x = latent[i] - m.shift;
            // Differentiate the defining polynomial even at quadratic=0;
            // the primal shortcut there does not imply zero quadratic risk.
            if bv[i] != 0.0 {
                g[3].add(bv[i] * x * x);
                g[4].add(-bv[i] * 2.0 * m.quadratic * x);
                g[5].add(bv[i]);
                bz[i] += bv[i] * 2.0 * m.quadratic * x;
            }
            let b = bz[i];
            if b == 0.0 {
                continue;
            }
            g[0].add(b); // Every Volterra row contains the original z0 source.
            for j in 0..i {
                let d = k.drift_weight(times, i, j);
                let innovation = if j + 1 == i {
                    near[j]
                } else {
                    k.weights[i][j] * dw[j]
                };
                let load = k.fractional_scale * innovation;
                let root = variance[j].sqrt();
                g[1].add(b * (-latent[j] * d + m.vol_of_vol * root * load));
                g[2].add(b * m.mean_reversion * root * load);
                bz[j] -= b * m.mean_reversion * d;
                bv[j] += b * m.mean_reversion * m.vol_of_vol * load / (2.0 * root);
                if let Some(dk) = &self.hurst {
                    let dl = dk.loading(i, j);
                    let di = if j + 1 == i {
                        dl * dw[j] + dk.residual[j] * normals[n + j]
                    } else {
                        dl * dw[j]
                    };
                    g[6].add(
                        b * (-m.mean_reversion * latent[j] * dl * (times[j + 1] - times[j])
                            + m.mean_reversion * m.vol_of_vol * root * di),
                    );
                }
            }
        }
        let out = g.into_iter().map(NeumaierSum::total).collect::<Vec<_>>();
        if out.iter().chain(&bv).chain(&bz).any(|v| !v.is_finite()) {
            return Err(invalid("quadratic_risk_nonfinite_adjoint"));
        }
        Ok(out)
    }
}
impl QuadraticHestonMcRecordedPath<'_> {
    #[must_use]
    pub fn path(&self) -> &RoughVolatilityPath {
        &self.path
    }
    /// Fixed cotangents on forward and diffusion-variance observations, not Z.
    pub fn reverse(
        &self,
        forward_seeds: &[f64],
        variance_seeds: &[f64],
    ) -> Result<QuadraticHestonMcAdjoints, HullWhiteError> {
        let n = self.plan.base.times.len() - 1;
        if forward_seeds.len() != n + 1
            || variance_seeds.len() != n + 1
            || forward_seeds
                .iter()
                .chain(variance_seeds)
                .any(|v| !v.is_finite())
        {
            return Err(invalid("quadratic_risk_seed_shape_or_value"));
        }
        let mut bv = variance_seeds.to_vec();
        let mut bf = forward_seeds[n];
        for j in (0..n).rev() {
            let next = bf * self.path.forwards[j + 1];
            let dt = self.plan.base.times[j + 1] - self.plan.base.times[j];
            if next != 0.0 {
                bv[j] += next
                    * (-0.5 * dt
                        + 0.5 * dt.sqrt() * self.normals[j] / self.path.variances[j].sqrt());
            }
            bf = finite_value(forward_seeds[j] + next / self.path.forwards[j])?;
        }
        let parameters = self.plan.variance_reverse(
            self.normals,
            &self.path.variances,
            &self.path.latent_states,
            &bv,
        )?;
        Ok(QuadraticHestonMcAdjoints {
            initial_forward: bf,
            parameters: parameters.into(),
        })
    }
}
