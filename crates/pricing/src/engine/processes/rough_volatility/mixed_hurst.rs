//! Hurst derivative of the normalized Gaussian hybrid kernel, including the
//! finite-grid variance used to center every mixture component. No price bumps.
use super::*;
use pricing_numerics::NeumaierSum;

#[derive(Clone, Debug)]
pub(super) struct MixedKernelHurst {
    older: Vec<Box<[f64]>>,
    near: Vec<f64>,
    residual: Vec<f64>,
    pub(super) variances: Vec<f64>,
}
impl MixedKernelHurst {
    pub(super) fn compile(base: &RoughVolatilityPathPlan) -> Result<Self, HullWhiteError> {
        let m = base.mixed_parameter_domain()?;
        let CompiledDriver::Power(k) = &base.driver else {
            return Err(invalid("rough_compiled_driver_mismatch"));
        };
        let h = m.hurst;
        let a = h + 0.5;
        let mut older = vec![Vec::new().into_boxed_slice()];
        let mut near = Vec::with_capacity(base.times.len() - 1);
        let mut residual = Vec::with_capacity(base.times.len() - 1);
        let mut variances = vec![0.0];
        for i in 1..base.times.len() {
            let dt = base.times[i] - base.times[i - 1];
            near.push(finite_value(
                k.near_loading[i - 1] * (0.5 / h + dt.ln() - 1.0 / a),
            )?);
            // r=dt^H (1/2-H)/(H+1/2). At H=1/2 r=0, but r'_-= -sqrt(dt).
            residual.push(finite_value(
                dt.powf(h) * ((0.5 - h) / a * dt.ln() - 1.0 / (a * a)),
            )?);
            let mut row = Vec::with_capacity(i - 1);
            let mut dq = NeumaierSum::new();
            dq.add(2.0 * dt.ln() * dt.powf(2.0 * h));
            for j in 0..i - 1 {
                let lo = base.times[i] - base.times[j + 1];
                let hi = base.times[i] - base.times[j];
                let log_ratio = if (hi - lo) / hi < 0.5 {
                    (-(hi - lo) / hi).ln_1p()
                } else {
                    (lo / hi).ln()
                };
                let x = -a * log_ratio;
                // Stable d log[(hi^a-lo^a)/a]/da, no difference of large powers.
                let correction = if x < 1e-3 {
                    let x2 = x * x;
                    (-0.5 * x + x2 * (1.0 / 12.0 + x2 * (-1.0 / 720.0 + x2 / 30240.0))) / a
                } else {
                    (x / x.exp_m1() - 1.0) / a
                };
                let dw = finite_value(k.weights[i][j] * (0.5 / h + hi.ln() + correction))?;
                dq.add(2.0 * k.weights[i][j] * dw * (base.times[j + 1] - base.times[j]));
                row.push(dw);
            }
            older.push(row.into_boxed_slice());
            variances.push(finite_value(dq.total())?);
        }
        Ok(Self {
            older,
            near,
            residual,
            variances,
        })
    }
    pub(super) fn gaussian_derivatives(
        &self,
        base: &RoughVolatilityPathPlan,
        normals: &[f64],
    ) -> Result<Vec<f64>, HullWhiteError> {
        let m = base.mixed_parameter_domain()?;
        let n = base.times.len() - 1;
        let q = (1.0 - m.correlation * m.correlation).sqrt();
        let dw = (0..n)
            .map(|j| {
                (base.times[j + 1] - base.times[j]).sqrt()
                    * (m.correlation * normals[j] + q * normals[n + j])
            })
            .collect::<Vec<_>>();
        let mut result = vec![0.0];
        for i in 1..=n {
            let mut sum = NeumaierSum::new();
            sum.add(self.near[i - 1] * dw[i - 1] + self.residual[i - 1] * normals[2 * n + i - 1]);
            for (&w, &z) in self.older[i].iter().zip(&dw) {
                sum.add(w * z);
            }
            result.push(finite_value(sum.total())?);
        }
        Ok(result)
    }
}

/// Component eta/shared rho reverse extended by one Hurst coordinate. Mixture
/// weights, xi curve, grid and Gaussian draws are fixed. H=1/2 uses the left
/// derivative. Compilation caches kernel and discrete-centering derivatives.
#[derive(Clone, Debug)]
pub struct MixedBergomiMcHurstPlan {
    base: RoughVolatilityPathPlan,
    pub(super) kernel: MixedKernelHurst,
}
impl MixedBergomiMcHurstPlan {
    pub fn compile(base: &RoughVolatilityPathPlan) -> Result<Self, HullWhiteError> {
        Ok(Self {
            kernel: MixedKernelHurst::compile(base)?,
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
    ) -> Result<MixedBergomiMcRecordedPath<'a>, HullWhiteError> {
        let mut record = self
            .base
            .evolve_mixed_bergomi_parameter_path(initial_forward, normals)?;
        record.hurst = Some(&self.kernel);
        Ok(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mixed_hurst_normalized_kernel_matches_high_precision() {
        let data: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/rough-volatility/mixed-hurst.json"
        )))
        .unwrap();
        for row in data["kernel"].as_array().unwrap() {
            let h = row["hurst"].as_f64().unwrap();
            let t = row["times"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let m = MixedRoughBergomi::new(
                h,
                -0.6,
                vec![1.],
                vec![0.7],
                crate::models::ForwardVarianceCurve::constant(0.04).unwrap(),
            )
            .unwrap();
            let p = RoughVolatilityPathPlan::compile(m.into(), t).unwrap();
            let k = MixedKernelHurst::compile(&p).unwrap();
            let got = k
                .near
                .iter()
                .chain(&k.residual)
                .chain(&k.variances)
                .chain(k.older.iter().flat_map(|x| x.iter()));
            let want = row["derivatives"].as_array().unwrap();
            assert_eq!(got.clone().count(), want.len());
            for (a, b) in got.zip(want) {
                let b = b.as_f64().unwrap();
                assert!(
                    (a - b).abs() <= 2e-12 * (1.0 + b.abs()),
                    "H={h} got={a} reference={b}"
                );
            }
        }
    }
}
