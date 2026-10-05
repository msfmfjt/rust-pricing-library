//! Fixed-kernel, fixed-weight parameter reverse for the shared-driver mixture.
use super::mixed_hurst::MixedKernelHurst;
use super::*;
use pricing_numerics::NeumaierSum;

#[derive(Clone, Debug, PartialEq)]
pub struct MixedBergomiMcAdjoints {
    pub initial_forward: f64,
    /// Selected eta/rho(/Hurst) or simplex-weight/xi coordinates, per absolute unit.
    pub parameters: Box<[f64]>,
}

#[derive(Debug)]
pub struct MixedBergomiMcRecordedPath<'a> {
    pub(super) plan: &'a RoughVolatilityPathPlan,
    pub(super) normals: &'a [f64],
    pub(super) path: RoughVolatilityPath,
    pub(super) shape: bool,
    pub(super) hurst: Option<&'a MixedKernelHurst>,
}

impl RoughVolatilityPathPlan {
    pub(in crate::engine) fn mixed_parameter_domain(
        &self,
    ) -> Result<&MixedRoughBergomi, HullWhiteError> {
        let RoughVolatilityModel::MixedRoughBergomi(m) = &self.model else {
            return Err(HullWhiteError::Unsupported {
                feature: "parameter reverse requires Mixed rough Bergomi",
            });
        };
        if m.correlation.abs() >= 1.0 {
            return Err(invalid("mixed_bergomi_risk_interior_rho"));
        }
        Ok(m)
    }

    /// Component log-variance volatilities followed by the shared correlation.
    /// Hurst, mixture weights, forward-variance curve and Gaussian inputs are fixed.
    pub fn mixed_bergomi_parameter_names(&self) -> Result<Vec<String>, HullWhiteError> {
        let m = self.mixed_parameter_domain()?;
        let mut names = (0..m.weights.len())
            .map(|i| format!("vol_of_vol[{i}]"))
            .collect::<Vec<_>>();
        names.push("correlation".into());
        Ok(names)
    }

    pub fn evolve_mixed_bergomi_parameter_path<'a>(
        &'a self,
        initial_forward: f64,
        normals: &'a [f64],
    ) -> Result<MixedBergomiMcRecordedPath<'a>, HullWhiteError> {
        self.mixed_parameter_domain()?;
        let path = self.evolve_path(initial_forward, normals)?;
        Ok(MixedBergomiMcRecordedPath {
            plan: self,
            normals,
            path,
            hurst: None,
            shape: false,
        })
    }

    fn mixed_variance_reverse(
        &self,
        normals: &[f64],
        latent: &[f64],
        seeds: &[f64],
        hurst: Option<&MixedKernelHurst>,
    ) -> Result<Vec<f64>, HullWhiteError> {
        let m = self.mixed_parameter_domain()?;
        let CompiledDriver::Power(k) = &self.driver else {
            return Err(invalid("rough_compiled_driver_mismatch"));
        };
        let n = self.times.len() - 1;
        if normals.len() != self.random_dimension() as usize
            || latent.len() != n + 1
            || seeds.len() != n + 1
            || normals
                .iter()
                .chain(seeds)
                .chain(latent)
                .any(|x| !x.is_finite())
        {
            return Err(invalid("mixed_bergomi_risk_seed_shape_or_value"));
        }
        let q = (1.0 - m.correlation * m.correlation).sqrt();
        let dw = (0..n)
            .map(|j| {
                (self.times[j + 1] - self.times[j]).sqrt()
                    * (normals[j] - m.correlation / q * normals[n + j])
            })
            .collect::<Vec<_>>();
        // The newest-cell independent residual has fixed loading and fixed draw:
        // only its correlated Brownian contribution depends on rho.
        let near = dw
            .iter()
            .zip(&k.near_loading)
            .map(|(d, c)| d * c)
            .collect::<Vec<_>>();
        let dx_h = hurst
            .map(|d| d.gaussian_derivatives(self, normals))
            .transpose()?;
        let count = m.weights.len();
        let mut g = vec![NeumaierSum::new(); count + 1 + usize::from(hurst.is_some())];
        // V(0)=xi(0), independently of eta and rho.
        for i in 1..=n {
            if seeds[i] == 0.0 {
                continue;
            }
            let xi = m.forward_variance.value(self.times[i])?;
            if xi == 0.0 {
                continue;
            }
            let dx = finite_value(k.gaussian_at(i, &dw, &near))?;
            for (j, (&w, &eta)) in m.weights.iter().zip(&m.vol_of_vols).enumerate() {
                if w == 0.0 {
                    continue;
                }
                // Reproduce exactly the primal component, including discrete centering.
                let component = finite_value(
                    xi.ln() + w.ln() + eta * latent[i] - 0.5 * eta.powi(2) * k.variances[i],
                )?
                .exp();
                let b = seeds[i] * component;
                g[j].add(b * (latent[i] - eta * k.variances[i]));
                g[count].add(b * eta * dx);
                if let (Some(d), Some(dx)) = (hurst, &dx_h) {
                    g[count + 1].add(b * (eta * dx[i] - 0.5 * eta.powi(2) * d.variances[i]));
                }
            }
        }
        let result = g.into_iter().map(NeumaierSum::total).collect::<Vec<_>>();
        if result.iter().any(|x| !x.is_finite()) {
            return Err(invalid("mixed_bergomi_risk_nonfinite_adjoint"));
        }
        Ok(result)
    }
}

impl MixedBergomiMcRecordedPath<'_> {
    #[must_use]
    pub fn path(&self) -> &RoughVolatilityPath {
        &self.path
    }

    /// Reverse fixed cotangents on forward and diffusion-variance observations.
    /// Eta/rho/H scopes fix zero xi nodes. The separate shape scope requires
    /// positive curve inputs and differentiates their original coordinates.
    pub fn reverse(
        &self,
        forward_seeds: &[f64],
        variance_seeds: &[f64],
    ) -> Result<MixedBergomiMcAdjoints, HullWhiteError> {
        let n = self.plan.times.len() - 1;
        if forward_seeds.len() != n + 1
            || variance_seeds.len() != n + 1
            || forward_seeds
                .iter()
                .chain(variance_seeds)
                .any(|x| !x.is_finite())
        {
            return Err(invalid("mixed_bergomi_risk_seed_shape_or_value"));
        }
        let mut bv = variance_seeds.to_vec();
        let mut bf = forward_seeds[n];
        for j in (0..n).rev() {
            let next_bar = bf * self.path.forwards[j + 1];
            let v = self.path.variances[j];
            if v > 0.0 && next_bar != 0.0 {
                let dt = self.plan.times[j + 1] - self.plan.times[j];
                bv[j] += next_bar * (-0.5 * dt + 0.5 * dt.sqrt() * self.normals[j] / v.sqrt());
            }
            bf = finite_value(forward_seeds[j] + next_bar / self.path.forwards[j])?;
        }
        let parameters = if self.shape {
            self.plan
                .mixed_shape_reverse(&self.path.latent_states, &bv)?
        } else {
            self.plan.mixed_variance_reverse(
                self.normals,
                &self.path.latent_states,
                &bv,
                self.hurst,
            )?
        };
        Ok(MixedBergomiMcAdjoints {
            initial_forward: bf,
            parameters: parameters.into(),
        })
    }
}

/// Variance-only route for LSV: never generate an unused unlevered asset path.
#[derive(Clone, Debug)]
pub(in crate::engine) struct MixedBergomiVarianceRiskPlan {
    path: RoughVolatilityPathPlan,
    hurst: Option<MixedKernelHurst>,
    shape: bool,
}
impl MixedBergomiVarianceRiskPlan {
    pub(in crate::engine) fn compile(
        model: RoughVolatilityModel,
        times: Vec<f64>,
    ) -> Result<Self, HullWhiteError> {
        let path = RoughVolatilityPathPlan::compile(model, times)?;
        path.mixed_parameter_domain()?;
        Ok(Self {
            path,
            hurst: None,
            shape: false,
        })
    }
    pub(in crate::engine) fn compile_with_hurst(
        model: RoughVolatilityModel,
        times: Vec<f64>,
    ) -> Result<Self, HullWhiteError> {
        let mut plan = Self::compile(model, times)?;
        plan.hurst = Some(MixedKernelHurst::compile(&plan.path)?);
        Ok(plan)
    }
    pub(in crate::engine) fn compile_shape(
        model: RoughVolatilityModel,
        times: Vec<f64>,
    ) -> Result<Self, HullWhiteError> {
        let path = RoughVolatilityPathPlan::compile(model, times)?;
        path.mixed_shape_domain()?;
        Ok(Self {
            path,
            hurst: None,
            shape: true,
        })
    }
    pub(in crate::engine) fn path_plan(&self) -> &RoughVolatilityPathPlan {
        &self.path
    }
    pub(in crate::engine) fn width(&self) -> usize {
        if self.shape {
            return self.path.mixed_shape_width();
        }
        match &self.path.model {
            RoughVolatilityModel::MixedRoughBergomi(m) => {
                m.weights.len() + 1 + usize::from(self.hurst.is_some())
            }
            _ => unreachable!(),
        }
    }
    pub(in crate::engine) fn reverse(
        &self,
        normals: &[f64],
        seeds: &[f64],
    ) -> Result<Vec<f64>, HullWhiteError> {
        let history = self.path.variance_history(normals)?;
        if self.shape {
            return self.path.mixed_shape_reverse(&history.latent_states, seeds);
        }
        self.path.mixed_variance_reverse(
            normals,
            &history.latent_states,
            seeds,
            self.hurst.as_ref(),
        )
    }
}
