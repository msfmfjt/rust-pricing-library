//! Fixed-H/eta/rho derivatives of simplex mixture weights and original xi inputs.
//! The last normalized weight is the balancing component, not an independent input.
use super::*;
use crate::models::rough_volatility::ForwardVarianceKind;
use pricing_numerics::NeumaierSum;

impl RoughVolatilityPathPlan {
    pub(super) fn mixed_shape_domain(&self) -> Result<&MixedRoughBergomi, HullWhiteError> {
        let RoughVolatilityModel::MixedRoughBergomi(m) = &self.model else {
            return Err(HullWhiteError::Unsupported {
                feature: "shape reverse requires Mixed rough Bergomi",
            });
        };
        if m.weights.iter().any(|&w| w <= 0.0) {
            return Err(invalid("mixed_shape_strictly_positive_weights"));
        }
        let (count, positive) = match &m.forward_variance.kind {
            ForwardVarianceKind::Knots { values, .. } => {
                (values.len(), values.iter().all(|&x| x > 0.0))
            }
            ForwardVarianceKind::Exponential { initial, .. } => (2, *initial > 0.0),
        };
        if !positive || count + m.weights.len() - 1 > 4096 {
            return Err(invalid("mixed_shape_curve_domain_or_size"));
        }
        Ok(m)
    }

    /// Coordinates are transfers from the last normalized weight to each earlier
    /// weight, followed by original xi knot values (times fixed), or exponential
    /// initial variance and growth. The simplex boundary and zero xi are rejected.
    /// Rho is fixed, so its endpoints remain allowed for this separate risk scope.
    pub fn mixed_bergomi_shape_names(&self) -> Result<Vec<String>, HullWhiteError> {
        let m = self.mixed_shape_domain()?;
        let last = m.weights.len() - 1;
        let mut names = (0..last)
            .map(|i| format!("weight_transfer[{i},{last}]"))
            .collect::<Vec<_>>();
        match &m.forward_variance.kind {
            ForwardVarianceKind::Knots { values, .. } => {
                names.extend((0..values.len()).map(|i| format!("forward_variance[{i}]")));
            }
            ForwardVarianceKind::Exponential { .. } => {
                names.push("forward_variance_initial".into());
                names.push("forward_variance_growth".into());
            }
        }
        Ok(names)
    }

    pub(super) fn mixed_shape_width(&self) -> usize {
        let RoughVolatilityModel::MixedRoughBergomi(m) = &self.model else {
            unreachable!()
        };
        m.weights.len() - 1
            + match &m.forward_variance.kind {
                ForwardVarianceKind::Knots { values, .. } => values.len(),
                ForwardVarianceKind::Exponential { .. } => 2,
            }
    }

    /// Record the unchanged primal path for the weight/xi coordinate reverse.
    pub fn evolve_mixed_bergomi_shape_path<'a>(
        &'a self,
        initial_forward: f64,
        normals: &'a [f64],
    ) -> Result<MixedBergomiMcRecordedPath<'a>, HullWhiteError> {
        self.mixed_shape_domain()?;
        Ok(MixedBergomiMcRecordedPath {
            plan: self,
            normals,
            path: self.evolve_path(initial_forward, normals)?,
            hurst: None,
            shape: true,
        })
    }

    pub(super) fn mixed_shape_reverse(
        &self,
        latent: &[f64],
        seeds: &[f64],
    ) -> Result<Vec<f64>, HullWhiteError> {
        let m = self.mixed_shape_domain()?;
        let CompiledDriver::Power(k) = &self.driver else {
            return Err(invalid("rough_compiled_driver_mismatch"));
        };
        if seeds.len() != self.times.len()
            || latent.len() != seeds.len()
            || seeds.iter().chain(latent).any(|x| !x.is_finite())
        {
            return Err(invalid("mixed_shape_seed_shape_or_value"));
        }
        let nw = m.weights.len() - 1;
        let mut result = vec![NeumaierSum::new(); self.mixed_shape_width()];
        for (i, (&t, &seed)) in self.times.iter().zip(seeds).enumerate() {
            if seed == 0.0 {
                continue;
            }
            let xi = m.forward_variance.value(t)?;
            if xi <= 0.0 {
                return Err(invalid("mixed_shape_nonpositive_variance"));
            }
            // V(0)=xi(0) exactly; its weight derivative is zero on the simplex.
            let bxi = if i == 0 {
                seed
            } else {
                let mut components = Vec::with_capacity(m.weights.len());
                let mut sum = NeumaierSum::new();
                for (&w, &eta) in m.weights.iter().zip(&m.vol_of_vols) {
                    let component = finite_value(
                        xi.ln() + w.ln() + eta * latent[i] - 0.5 * eta.powi(2) * k.variances[i],
                    )?
                    .exp();
                    sum.add(component);
                    components.push(component);
                }
                let reference = components[nw] / m.weights[nw];
                for j in 0..nw {
                    result[j].add(seed * (components[j] / m.weights[j] - reference));
                }
                seed * sum.total() / xi
            };
            // Exact transpose of the existing linear/flat or exponential curve,
            // into its ORIGINAL inputs, not interpolated simulation-time values.
            match &m.forward_variance.kind {
                ForwardVarianceKind::Knots { times, .. } => {
                    let upper = times.partition_point(|&x| x <= t);
                    if upper == times.len() {
                        result[nw + upper - 1].add(bxi);
                    } else {
                        let lower = upper - 1;
                        let w = (t - times[lower]) / (times[upper] - times[lower]);
                        result[nw + lower].add(bxi * (1.0 - w));
                        result[nw + upper].add(bxi * w);
                    }
                }
                ForwardVarianceKind::Exponential { initial, .. } => {
                    result[nw].add(bxi * xi / initial);
                    result[nw + 1].add(bxi * xi * t);
                }
            }
        }
        result
            .into_iter()
            .map(|s| finite_value(s.total()))
            .collect()
    }
}
