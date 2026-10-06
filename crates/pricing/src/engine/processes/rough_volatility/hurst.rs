//! Hurst reverse of the existing hybrid full-truncation Rough Heston path.
//! Gaussian inputs and time nodes are fixed. At H=1/2 the derivative is from below.
use super::*;
use pricing_numerics::digamma_half_to_two;

/// Derivatives of the combined, unnormalized kernel loadings. Combining the
/// sqrt(2H) normalization before differentiation avoids artificial cancellation.
#[derive(Clone, Debug)]
pub(super) struct PowerKernelHurst {
    older: Vec<Box<[f64]>>,
    near: Vec<f64>,
    pub(super) residual: Vec<f64>,
}
impl PowerKernelHurst {
    fn compile(h: f64, times: &[f64], k: &PowerKernel) -> Result<Self, HullWhiteError> {
        let a = h + 0.5;
        let psi = digamma_half_to_two(a).ok_or(invalid("rough_hurst_digamma"))?;
        let mut older = Vec::with_capacity(times.len());
        older.push(Vec::new().into_boxed_slice());
        let mut near = Vec::with_capacity(times.len() - 1);
        let mut residual = Vec::with_capacity(times.len() - 1);
        for i in 1..times.len() {
            let mut row = Vec::with_capacity(i - 1);
            for j in 0..i - 1 {
                let lo = times[i] - times[j + 1];
                let hi = times[i] - times[j];
                // d log[(hi^a-lo^a)/a] / da, stable for almost equal lags.
                let log_ratio = if (hi - lo) / hi < 0.5 {
                    (-(hi - lo) / hi).ln_1p()
                } else {
                    (lo / hi).ln()
                };
                let x = -a * log_ratio;
                let correction = if x < 1e-3 {
                    let x2 = x * x;
                    (-0.5 * x + x2 * (1.0 / 12.0 + x2 * (-1.0 / 720.0 + x2 / 30240.0))) / a
                } else {
                    (x / x.exp_m1() - 1.0) / a
                };
                row.push(finite_value(
                    k.fractional_scale * k.weights[i][j] * (hi.ln() - psi + correction),
                )?);
            }
            older.push(row.into_boxed_slice());
            let dt = times[i] - times[i - 1];
            near.push(finite_value(
                k.fractional_scale * k.near_loading[i - 1] * (dt.ln() - 1.0 / a - psi),
            )?);
            // Differentiate dt^H (1/2-H)/[(H+1/2) sqrt(2H) Gamma(H+1/2)].
            // Do NOT divide by (1/2-H): at H=1/2 the residual is zero but
            // its LEFT derivative is -sqrt(dt), not zero.
            let ratio = (0.5 - h) / a;
            residual.push(finite_value(
                k.fractional_scale
                    * dt.powf(h)
                    * (ratio * (dt.ln() - 0.5 / h - psi) - 1.0 / (a * a)),
            )?);
        }
        Ok(Self {
            older,
            near,
            residual,
        })
    }
    pub(super) fn loading(&self, i: usize, j: usize) -> f64 {
        if j + 1 == i {
            self.near[j]
        } else {
            self.older[i][j]
        }
    }
}

/// Cached kernel derivatives for Pure-SV Rough Heston only. A finite lift's
/// coefficients do not define a unique H derivative and are explicitly rejected.
#[derive(Clone, Debug)]
pub struct RoughHestonMcHurstPlan {
    base: RoughVolatilityPathPlan,
    kernel: PowerKernelHurst,
}
impl RoughHestonMcHurstPlan {
    pub(super) fn kernel_derivative(&self) -> &PowerKernelHurst {
        &self.kernel
    }
    pub fn compile(base: &RoughVolatilityPathPlan) -> Result<Self, HullWhiteError> {
        let h = match &base.model {
            RoughVolatilityModel::RoughHeston(h) => h,
            _ => {
                return Err(HullWhiteError::Unsupported {
                    feature: "MC Hurst reverse requires power-kernel Rough Heston",
                });
            }
        };
        base.heston_parameter_domain()?;
        let CompiledDriver::Power(k) = &base.driver else {
            return Err(invalid("rough_compiled_driver_mismatch"));
        };
        Ok(Self {
            kernel: PowerKernelHurst::compile(h.hurst, &base.times, k)?,
            base: base.clone(),
        })
    }
    #[must_use]
    pub fn path_plan(&self) -> &RoughVolatilityPathPlan {
        &self.base
    }
    pub fn evolve_path<'a>(
        &'a self,
        initial_forward: f64,
        normals: &'a [f64],
    ) -> Result<RoughHestonMcHurstRecordedPath<'a>, HullWhiteError> {
        Ok(RoughHestonMcHurstRecordedPath {
            record: self
                .base
                .evolve_heston_parameter_path(initial_forward, normals)?,
            kernel: &self.kernel,
        })
    }
}

/// One unchanged primal path plus a reference to the cached kernel derivative.
#[derive(Debug)]
pub struct RoughHestonMcHurstRecordedPath<'a> {
    record: HestonMcRecordedPath<'a>,
    kernel: &'a PowerKernelHurst,
}
/// Per one absolute H unit, holding the five scalar parameters fixed.
#[derive(Clone, Debug, PartialEq)]
pub struct RoughHestonMcHurstAdjoints {
    pub initial_forward: f64,
    pub hurst: f64,
}
impl RoughHestonMcHurstRecordedPath<'_> {
    #[must_use]
    pub fn path(&self) -> &RoughVolatilityPath {
        self.record.path()
    }
    /// Fixed cotangents on forward and diffusion-variance observations. The
    /// parent exact-zero/full-truncation and seed-domain checks remain active.
    pub fn reverse(
        &self,
        forward_seeds: &[f64],
        variance_seeds: &[f64],
    ) -> Result<RoughHestonMcHurstAdjoints, HullWhiteError> {
        let (r, hurst) =
            self.record
                .reverse_impl(forward_seeds, variance_seeds, Some(self.kernel))?;
        Ok(RoughHestonMcHurstAdjoints {
            initial_forward: r.initial_forward,
            hurst,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kernel_derivatives_match_independent_high_precision_references() {
        let data: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/rough-volatility/mc-hurst.json"
        )))
        .unwrap();
        for row in data["kernel"].as_array().unwrap() {
            let h = row["h"].as_f64().unwrap();
            let lo = row["lo"].as_f64().unwrap();
            let hi = row["hi"].as_f64().unwrap();
            let ts = if lo == 0.0 {
                vec![0.0, hi]
            } else {
                vec![0.0, hi - lo, hi]
            };
            let k = PowerKernel::compile(h, &ts).unwrap();
            let d = PowerKernelHurst::compile(h, &ts, &k).unwrap();
            let got = d.loading(ts.len() - 1, 0);
            let want = row["derivative"].as_f64().unwrap();
            assert!(
                (got - want).abs() < 2e-12 * (1.0 + want.abs()),
                "{row}: {got} vs {want}"
            );
        }
    }
}
