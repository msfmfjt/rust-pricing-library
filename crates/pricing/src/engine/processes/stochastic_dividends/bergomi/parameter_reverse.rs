//! Analytic model-parameter reverse at fixed independent normals and correlations.
//! Coefficient Jacobians are prepared once; OU states are reversed once per path.
//! See design/adr/0016-stochastic-dividend-bergomi-risk.md for the SPD domain.

use super::*;

const MIN_PIVOT: f64 = 1e-10;

pub(in crate::engine::processes::stochastic_dividends) struct BergomiParameterReverse {
    inner: PreparedModel,
}
enum PreparedModel {
    One(Prepared<1, 3, 2>),
    Two(Prepared<2, 4, 4>),
}
struct Prepared<const N: usize, const D: usize, const P: usize> {
    steps: Box<[StepDerivative<N, D>]>,
    centering: Box<[[f64; P]]>,
    weight_derivative: [f64; N],
}
struct StepDerivative<const N: usize, const D: usize> {
    // First index selects the mean-reversion parameter, second the OU state.
    lower: [[[f64; D]; N]; N],
    decay: [f64; N],
}

impl BergomiParameterReverse {
    pub(in crate::engine::processes::stochastic_dividends) fn new(
        kernel: &BergomiDividendKernel,
        sd: f64,
        times: &[f64],
    ) -> Result<Self, StochasticDividendError> {
        let inner = match kernel {
            BergomiDividendKernel::One(k) => {
                let id = &k.identity;
                let matrix = [[1.0, sd, id[2]], [sd, 1.0, id[3]], [id[2], id[3], 1.0]];
                PreparedModel::One(Prepared::new(k, [id[0]], matrix, times, [0.0])?)
            }
            BergomiDividendKernel::Two(k) => {
                let id = &k.identity;
                let matrix = [
                    [1.0, sd, id[4], id[5]],
                    [sd, 1.0, id[7], id[8]],
                    [id[4], id[7], 1.0, id[6]],
                    [id[5], id[8], id[6], 1.0],
                ];
                let theta = id[3];
                let norm2 =
                    (1.0 - 2.0 * theta).powi(2) + 2.0 * theta * (1.0 - theta) * (1.0 + id[6]);
                if norm2 <= MIN_PIVOT {
                    return Err(unsupported());
                }
                let log_norm_derivative = (1.0 - id[6]) * (2.0 * theta - 1.0) / norm2;
                let dw = [
                    -1.0 / norm2.sqrt() - k.weights[0] * log_norm_derivative,
                    1.0 / norm2.sqrt() - k.weights[1] * log_norm_derivative,
                ];
                PreparedModel::Two(Prepared::new(k, [id[0], id[1]], matrix, times, dw)?)
            }
        };
        Ok(Self { inner })
    }
    pub(in crate::engine::processes::stochastic_dividends) fn labels(
        &self,
    ) -> &'static [&'static str] {
        match &self.inner {
            PreparedModel::One(_) => &["bergomi_mean_reversion[0]", "bergomi_vol_of_vol"],
            PreparedModel::Two(_) => &[
                "bergomi_mean_reversion[0]",
                "bergomi_mean_reversion[1]",
                "bergomi_vol_of_vol",
                "bergomi_mixing_weight",
            ],
        }
    }
    pub(in crate::engine::processes::stochastic_dividends) fn pullback(
        &self,
        kernel: &BergomiDividendKernel,
        normals: &[f64],
        loading_bars: &[f64],
    ) -> Result<Vec<f64>, StochasticDividendError> {
        match (&self.inner, kernel) {
            (PreparedModel::One(p), BergomiDividendKernel::One(k)) => {
                p.pullback(k, normals, loading_bars)
            }
            (PreparedModel::Two(p), BergomiDividendKernel::Two(k)) => {
                p.pullback(k, normals, loading_bars)
            }
            _ => Err(invalid("bergomi_reverse_family")),
        }
    }
}
fn unsupported() -> StochasticDividendError {
    StochasticDividendError::Unsupported {
        feature: "Bergomi parameter AAD requires integrated correlation pivots and weight variance > 1e-10; use evaluate_aad for the basic risk scope",
    }
}

// d log(phi(x))/dx, phi(x)=integral_0^1 exp(-xu)du. The Taylor branch
// removes cancellation at zero; exp(-x) avoids overflow for large x.
fn log_phi_derivative(x: f64) -> f64 {
    if x < 1e-3 {
        let x2 = x * x;
        -0.5 + x * (1.0 / 12.0 + x2 * (-1.0 / 720.0 + x2 / 30240.0))
    } else {
        (-x).exp() / -(-x).exp_m1() - 1.0 / x
    }
}
impl<const N: usize, const D: usize> StepDerivative<N, D> {
    fn new(k: [f64; N], matrix: [[f64; D]; D], dt: f64) -> Result<Self, StochasticDividendError> {
        let a: [f64; D] = std::array::from_fn(|i| if i < 2 { 0.0 } else { k[i - 2] * dt });
        let mut c = vec![vec![0.0; D]; D];
        for i in 0..D {
            for j in 0..D {
                c[i][j] = matrix[i][j]
                    * ou_kernel_correlation(a[i], a[j])
                        .map_err(|_| invalid("bergomi_ou_covariance"))?;
            }
        }
        let factor =
            CorrelationFactor::compile(c.clone(), BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES)
                .map_err(|_| invalid("bergomi_reverse_covariance"))?;
        if factor.diagnostics().pivots.iter().any(|p| *p <= MIN_PIVOT) {
            return Err(unsupported());
        }
        let l = factor.lower();
        let mut lower = [[[0.0; D]; N]; N];
        for (p, rows) in lower.iter_mut().enumerate() {
            // Differentiate the same unpivoted Cholesky factor, not an unrelated
            // square root. Correlations are fixed but integrated C depends on k.
            let mut dl = [[0.0; D]; D];
            for j in 0..D {
                let sum: f64 = (0..j).map(|h| 2.0 * l[j * D + h] * dl[j][h]).sum();
                dl[j][j] = -sum / (2.0 * l[j * D + j]);
                for i in j + 1..D {
                    let di = if i == p + 2 { dt } else { 0.0 };
                    let dj = if j == p + 2 { dt } else { 0.0 };
                    let dc = c[i][j]
                        * ((di + dj) * log_phi_derivative(a[i] + a[j])
                            - di * log_phi_derivative(2.0 * a[i])
                            - dj * log_phi_derivative(2.0 * a[j]));
                    let sum: f64 = (0..j)
                        .map(|h| dl[i][h] * l[j * D + h] + l[i * D + h] * dl[j][h])
                        .sum();
                    dl[i][j] = (dc - sum - l[i * D + j] * dl[j][j]) / l[j * D + j];
                }
            }
            for (i, row) in rows.iter_mut().enumerate() {
                let marginal = Bergomi1Factor::new(k[i], 0.0, 0.0)
                    .and_then(|m| m.transition(dt))
                    .map_err(|_| invalid("bergomi_reverse_variance"))?;
                let scale = marginal.variance.sqrt();
                let ds = if i == p {
                    scale * dt * log_phi_derivative(2.0 * a[i + 2])
                } else {
                    0.0
                };
                for (j, value) in row.iter_mut().enumerate() {
                    *value = ds * l[(i + 2) * D + j] + scale * dl[i + 2][j];
                }
            }
        }
        let decay = k.map(|k| -dt * (-k * dt).exp());
        if lower
            .iter()
            .flatten()
            .flatten()
            .chain(&decay)
            .any(|x| !x.is_finite())
        {
            return Err(invalid("bergomi_coefficient_derivative"));
        }
        Ok(Self { lower, decay })
    }
}
impl<const N: usize, const D: usize, const P: usize> Prepared<N, D, P> {
    fn new(
        kernel: &Kernel<N, D>,
        k: [f64; N],
        matrix: [[f64; D]; D],
        times: &[f64],
        weight_derivative: [f64; N],
    ) -> Result<Self, StochasticDividendError> {
        let steps = times
            .windows(2)
            .map(|t| StepDerivative::new(k, matrix, t[1] - t[0]))
            .collect::<Result<Vec<_>, _>>()?;
        let mut centering = Vec::with_capacity(kernel.steps.len());
        for &t in &times[..times.len() - 1] {
            let mut dc = [0.0; P];
            if t > 0.0 {
                let s = Step::<N, D>::compile(k, matrix, t)?;
                let ds = StepDerivative::<N, D>::new(k, matrix, t)?;
                for j in 0..D {
                    let exposure: f64 = (0..N).map(|i| kernel.weights[i] * s.lower[i][j]).sum();
                    for (p, value) in dc.iter_mut().take(N).enumerate() {
                        let de: f64 = (0..N).map(|i| kernel.weights[i] * ds.lower[p][i][j]).sum();
                        *value += 2.0 * kernel.vol_of_vol * exposure * de;
                    }
                    if N == 2 {
                        let de: f64 = (0..N).map(|i| weight_derivative[i] * s.lower[i][j]).sum();
                        dc[N + 1] += 2.0 * kernel.vol_of_vol * exposure * de;
                    }
                }
            }
            if dc.iter().any(|x| !x.is_finite()) {
                return Err(invalid("bergomi_centering_derivative"));
            }
            centering.push(dc);
        }
        Ok(Self {
            steps: steps.into_boxed_slice(),
            centering: centering.into_boxed_slice(),
            weight_derivative,
        })
    }
    fn pullback(
        &self,
        kernel: &Kernel<N, D>,
        normals: &[f64],
        bars: &[f64],
    ) -> Result<Vec<f64>, StochasticDividendError> {
        if normals.len() != D * kernel.steps.len() || bars.len() != kernel.steps.len() {
            return Err(invalid("bergomi_parameter_reverse_shape"));
        }
        let mut x = [0.0; N];
        let mut trace = Vec::with_capacity(kernel.steps.len());
        for (step, z) in kernel.steps.iter().zip(normals.as_chunks::<D>().0) {
            trace.push(x);
            for (j, value) in x.iter_mut().enumerate() {
                *value = step.decay[j] * *value
                    + step.lower[j].iter().zip(z).map(|(l, z)| l * z).sum::<f64>();
            }
        }
        let mut out = [0.0; P];
        let mut x_bar = [0.0; N];
        for i in (0..kernel.steps.len()).rev() {
            let step = &kernel.steps[i];
            let ds = &self.steps[i];
            let z = &normals[D * i..D * (i + 1)];
            let x = trace[i];
            for (p, value) in out.iter_mut().take(N).enumerate() {
                *value += x_bar[p] * ds.decay[p] * x[p];
                for (j, bar) in x_bar.iter().enumerate() {
                    *value += bar
                        * ds.lower[p][j]
                            .iter()
                            .zip(z)
                            .map(|(l, z)| l * z)
                            .sum::<f64>();
                }
            }
            for (j, bar) in x_bar.iter_mut().enumerate() {
                *bar *= step.decay[j];
            }
            let factor: f64 = kernel.weights.iter().zip(x).map(|(w, x)| w * x).sum();
            let loading = (kernel.vol_of_vol * (factor - kernel.centering[i])).exp();
            let q_bar = bars[i] * loading;
            // Center = nu*Var[Z]. Do not divide by nu: nu=0 has a valid inward derivative.
            out[N] += q_bar * (factor - 2.0 * kernel.centering[i]);
            for (p, value) in out.iter_mut().take(N).enumerate() {
                *value -= q_bar * kernel.vol_of_vol * self.centering[i][p];
            }
            if N == 2 {
                let df: f64 = self
                    .weight_derivative
                    .iter()
                    .zip(x)
                    .map(|(w, x)| w * x)
                    .sum();
                out[N + 1] += q_bar * kernel.vol_of_vol * (df - self.centering[i][N + 1]);
            }
            for (j, bar) in x_bar.iter_mut().enumerate() {
                *bar += q_bar * kernel.vol_of_vol * kernel.weights[j];
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(invalid("bergomi_parameter_reverse_result"));
        }
        Ok(out.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficient_derivatives_match_recompiled_ou_factors_including_zero_decay_rate() {
        let matrix = [
            [1.0, -0.25, -0.4, -0.2],
            [-0.25, 1.0, 0.15, -0.1],
            [-0.4, 0.15, 1.0, 0.3],
            [-0.2, -0.1, 0.3, 1.0],
        ];
        for k in [[0.0, 0.0], [1e-7, 1e-7], [0.8, 2.1], [30.0, 0.2]] {
            for dt in [0.003, 0.125, 1.3] {
                let ds = StepDerivative::<2, 4>::new(k, matrix, dt).unwrap();
                for p in 0..2 {
                    let h = 1e-6;
                    let mut plus = k;
                    plus[p] += h;
                    let mut minus = k;
                    minus[p] = (minus[p] - h).max(0.0);
                    let a = Step::<2, 4>::compile(plus, matrix, dt).unwrap();
                    let b = Step::<2, 4>::compile(minus, matrix, dt).unwrap();
                    for i in 0..2 {
                        for j in 0..4 {
                            let fd = (a.lower[i][j] - b.lower[i][j]) / (plus[p] - minus[p]);
                            assert!(
                                (ds.lower[p][i][j] - fd).abs() < 3e-6,
                                "k={k:?}, dt={dt}, p={p}, i={i}, j={j}, analytic={}, fd={fd}",
                                ds.lower[p][i][j]
                            );
                        }
                    }
                    let fd = (a.decay[p] - b.decay[p]) / (plus[p] - minus[p]);
                    assert!((ds.decay[p] - fd).abs() < 3e-6);
                }
            }
        }
    }

    #[test]
    fn log_phi_derivative_has_stable_zero_and_large_argument_limits() {
        assert_eq!(log_phi_derivative(0.0), -0.5);
        assert!((log_phi_derivative(1e-12) + 0.5 - 1e-12 / 12.0).abs() < 1e-16);
        assert!((log_phi_derivative(1000.0) + 0.001).abs() < 1e-16);
        // Independent Simpson integration of -int u exp(-xu) / int exp(-xu).
        for x in [0.0, 0.000999, 0.001001, 0.5, 10.0] {
            let mut den = 0.0;
            let mut num = 0.0;
            for i in 0..=4096 {
                let t = f64::from(i) / 4096.0;
                let w = if i == 0 || i == 4096 {
                    1.0
                } else if i % 2 == 0 {
                    2.0
                } else {
                    4.0
                };
                den += w * (-x * t).exp();
                num -= w * t * (-x * t).exp();
            }
            assert!((log_phi_derivative(x) - num / den).abs() < 2e-12, "x={x}");
        }
    }
}
