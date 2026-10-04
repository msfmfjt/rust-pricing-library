//! Fixed-kernel reverse of the existing full-truncation Heston path schemes.
//! This differentiates the finite path, not the continuous-time model or a fit.
use super::*;
use pricing_numerics::NeumaierSum;

/// Natural parameter order, shared by path adjoints and MC estimates.
pub const HESTON_MC_PARAMETER_NAMES: [&str; 5] = [
    "initial_variance",
    "mean_reversion",
    "long_run_variance",
    "vol_of_vol",
    "correlation",
];

#[derive(Clone, Debug, PartialEq)]
pub struct HestonMcAdjoints {
    pub initial_forward: f64,
    /// Per one absolute parameter unit, in `HESTON_MC_PARAMETER_NAMES` order.
    pub parameters: [f64; 5],
}

/// Borrows an immutable plan and its exact post-bridge Gaussian inputs.
/// Owns the unchanged primal path; no dense derivative tape or parameter bumps.
#[derive(Debug)]
pub struct HestonMcRecordedPath<'a> {
    plan: &'a RoughVolatilityPathPlan,
    normals: &'a [f64],
    path: RoughVolatilityPath,
}

impl RoughVolatilityPathPlan {
    pub(in crate::engine) fn heston_parameter_domain(
        &self,
    ) -> Result<&RoughHeston, HullWhiteError> {
        let h = match &self.model {
            RoughVolatilityModel::RoughHeston(h) => h,
            RoughVolatilityModel::LiftedHeston(m) => &m.heston,
            _ => {
                return Err(HullWhiteError::Unsupported {
                    feature: "MC parameter reverse requires rough or lifted Heston",
                });
            }
        };
        if h.initial_variance <= 0.0 || h.correlation.abs() >= 1.0 {
            return Err(invalid("heston_mc_risk_positive_v0_interior_rho"));
        }
        Ok(h)
    }

    /// Reverse-compatible path at fixed Hurst/kernel, time grid and Gaussian
    /// samples. Requires v0>0 and |rho|<1. Strictly negative raw-variance nodes
    /// have zero truncation derivative. Exact zero preterminal raw nodes are
    /// rejected: the composed square-root/truncation map has no finite ordinary
    /// derivative there. This does not impose a positive variance floor.
    pub fn evolve_heston_parameter_path<'a>(
        &'a self,
        initial_forward: f64,
        normals: &'a [f64],
    ) -> Result<HestonMcRecordedPath<'a>, HullWhiteError> {
        self.heston_parameter_domain()?;
        let path = self.evolve_path(initial_forward, normals)?;
        if path.latent_states[..self.times.len() - 1].contains(&0.0) {
            return Err(invalid("heston_mc_risk_zero_raw_variance_kink"));
        }
        Ok(HestonMcRecordedPath {
            plan: self,
            normals,
            path,
        })
    }
}

impl HestonMcRecordedPath<'_> {
    #[must_use]
    pub fn path(&self) -> &RoughVolatilityPath {
        &self.path
    }

    /// VJP of forward AND diffusion-variance observations. Seeds are fixed
    /// cotangents, one of each per time node. Terminal raw zero is rejected only
    /// when seeded. Latent raw-variance observations are not this contract.
    pub fn reverse(
        &self,
        forward_seeds: &[f64],
        variance_seeds: &[f64],
    ) -> Result<HestonMcAdjoints, HullWhiteError> {
        self.reverse_impl(forward_seeds, variance_seeds, None)
            .map(|r| r.0)
    }

    pub(super) fn reverse_impl(
        &self,
        forward_seeds: &[f64],
        variance_seeds: &[f64],
        hurst_kernel: Option<&super::hurst::PowerKernelHurst>,
    ) -> Result<(HestonMcAdjoints, f64), HullWhiteError> {
        let times = &self.plan.times;
        let n = times.len() - 1;
        if forward_seeds.len() != n + 1
            || variance_seeds.len() != n + 1
            || forward_seeds
                .iter()
                .chain(variance_seeds)
                .any(|x| !x.is_finite())
        {
            return Err(invalid("heston_mc_risk_seed_shape_or_value"));
        }
        let h = self.plan.heston_parameter_domain()?;
        let mut bv = variance_seeds.to_vec();
        let mut bf = forward_seeds[n];
        // Log-Euler asset reverse. The spot normal is fixed; rho only changes
        // variance innovations under the existing factor-major construction.
        for j in (0..n).rev() {
            let next_bar = bf * self.path.forwards[j + 1];
            let v = self.path.variances[j];
            if v > 0.0 && next_bar != 0.0 {
                let dt = times[j + 1] - times[j];
                bv[j] += next_bar * (-0.5 * dt + 0.5 * dt.sqrt() * self.normals[j] / v.sqrt());
            }
            bf = finite_value(forward_seeds[j] + next_bar / self.path.forwards[j])?;
        }
        let q = (1.0 - h.correlation * h.correlation).sqrt();
        let dw_rho: Vec<_> = (0..n)
            .map(|j| {
                (times[j + 1] - times[j]).sqrt()
                    * (self.normals[j] - h.correlation / q * self.normals[n + j])
            })
            .collect();
        let mut g: [NeumaierSum; 5] = std::array::from_fn(|_| NeumaierSum::new());
        let mut gh = NeumaierSum::new();
        match (&self.plan.model, &self.plan.driver) {
            (RoughVolatilityModel::RoughHeston(_), CompiledDriver::Power(k)) => {
                let (dw, near) = self.plan.innovations(k, h.correlation, self.normals, false);
                for i in (1..=n).rev() {
                    let bar = self.raw_bar(i, bv[i])?;
                    if bar == 0.0 {
                        continue;
                    }
                    g[0].add(bar);
                    for j in 0..i {
                        let v = self.path.variances[j];
                        let root = v.sqrt();
                        let d = k.drift_weight(times, i, j);
                        let (innovation, rho_loading) = if j + 1 == i {
                            (near[j], k.near_loading[j])
                        } else {
                            (k.weights[i][j] * dw[j], k.weights[i][j])
                        };
                        if let Some(dk) = hurst_kernel {
                            let dc = dk.loading(i, j);
                            let di = if j + 1 == i {
                                dc * dw[j] + dk.residual[j] * self.normals[2 * n + j]
                            } else {
                                dc * dw[j]
                            };
                            gh.add(
                                bar * (dc
                                    * (times[j + 1] - times[j])
                                    * h.mean_reversion
                                    * (h.long_run_variance - v)
                                    + h.vol_of_vol * root * di),
                            );
                        }
                        g[1].add(bar * d * (h.long_run_variance - v));
                        g[2].add(bar * d * h.mean_reversion);
                        g[3].add(bar * k.fractional_scale * root * innovation);
                        g[4].add(
                            bar * k.fractional_scale
                                * h.vol_of_vol
                                * root
                                * rho_loading
                                * dw_rho[j],
                        );
                        if v > 0.0 {
                            bv[j] += bar
                                * (-d * h.mean_reversion
                                    + k.fractional_scale * h.vol_of_vol * innovation
                                        / (2.0 * root));
                        }
                    }
                }
            }
            (RoughVolatilityModel::LiftedHeston(m), CompiledDriver::Lift) => {
                // Factor derivatives need only a backward vector: fixed rates
                // and weights make their state Jacobian independent of U.
                let mut bu = vec![0.0; m.weights.len()];
                for j in (0..n).rev() {
                    let bar = self.raw_bar(j + 1, bv[j + 1])?;
                    g[0].add(bar);
                    let dt = times[j + 1] - times[j];
                    let mut common_bar = NeumaierSum::new();
                    for ((b, &w), &rate) in bu.iter_mut().zip(&m.weights).zip(&m.rates) {
                        *b = (*b + w * bar) / (1.0 + rate * dt);
                        common_bar.add(*b);
                    }
                    let b = common_bar.total();
                    let v = self.path.variances[j];
                    let root = v.sqrt();
                    let dw =
                        dt.sqrt() * (h.correlation * self.normals[j] + q * self.normals[n + j]);
                    g[1].add(b * (h.long_run_variance - v) * dt);
                    g[2].add(b * h.mean_reversion * dt);
                    g[3].add(b * root * dw);
                    g[4].add(b * h.vol_of_vol * root * dw_rho[j]);
                    if v > 0.0 {
                        bv[j] += b * (-h.mean_reversion * dt + h.vol_of_vol * dw / (2.0 * root));
                    }
                }
            }
            _ => return Err(invalid("rough_compiled_driver_mismatch")),
        }
        g[0].add(bv[0]);
        let parameters = g.map(|s| s.total());
        if parameters.iter().any(|p| !p.is_finite()) || bv.iter().any(|v| !v.is_finite()) {
            return Err(invalid("heston_mc_risk_nonfinite_adjoint"));
        }
        let hurst = finite_value(gh.total())?;
        Ok((
            HestonMcAdjoints {
                initial_forward: bf,
                parameters,
            },
            hurst,
        ))
    }

    fn raw_bar(&self, i: usize, seed: f64) -> Result<f64, HullWhiteError> {
        if !seed.is_finite() {
            return Err(invalid("heston_mc_risk_nonfinite_adjoint"));
        }
        let raw = self.path.latent_states[i];
        if raw == 0.0 && seed != 0.0 {
            return Err(invalid("heston_mc_risk_zero_raw_variance_kink"));
        }
        Ok(if raw > 0.0 { seed } else { 0.0 })
    }
}
