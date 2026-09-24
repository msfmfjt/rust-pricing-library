//! Raw Brownian-correlation derivatives at fixed independent normals.
//! The open instantaneous-SPD domain is required as well as integrated SPD.
//! Coefficient Jacobians are prepared once; no covariance bumps are used.

use super::*;

const MIN_PIVOT: f64 = 1e-10;

pub(in crate::engine::processes::stochastic_dividends) struct BergomiCorrelationReverse {
    inner: PreparedModel,
}
enum PreparedModel {
    One(Prepared<1, 3, 3>),
    Two(Prepared<2, 4, 6>),
}
struct Prepared<const N: usize, const D: usize, const C: usize> {
    lower: Box<[[[[f64; D]; N]; C]]>,
    centering: Box<[[f64; C]]>,
    weights: [[f64; N]; C],
}

pub(in crate::engine::processes::stochastic_dividends) fn unsupported() -> StochasticDividendError {
    StochasticDividendError::Unsupported {
        feature: "correlation AAD requires instantaneous and integrated correlation pivots and weight variance > 1e-10; existing price/basic AAD methods remain available",
    }
}

impl BergomiCorrelationReverse {
    pub(in crate::engine::processes::stochastic_dividends) fn new(
        kernel: &BergomiDividendKernel,
        sd: f64,
        times: &[f64],
    ) -> Result<Self, StochasticDividendError> {
        let inner = match kernel {
            BergomiDividendKernel::One(k) => {
                let id = &k.identity;
                let matrix = [[1.0, sd, id[2]], [sd, 1.0, id[3]], [id[2], id[3], 1.0]];
                PreparedModel::One(Prepared::new(
                    k,
                    [id[0]],
                    matrix,
                    [(0, 1), (0, 2), (1, 2)],
                    times,
                    [[0.0]; 3],
                )?)
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
                if !norm2.is_finite() || norm2 <= MIN_PIVOT {
                    return Err(unsupported());
                }
                let mut dw = [[0.0; 2]; 6];
                // Changing rho_V1,V2 changes both normalized weights, even at fixed theta.
                dw[5] = k.weights.map(|w| -w * theta * (1.0 - theta) / norm2);
                PreparedModel::Two(Prepared::new(
                    k,
                    [id[0], id[1]],
                    matrix,
                    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
                    times,
                    dw,
                )?)
            }
        };
        Ok(Self { inner })
    }
    pub(in crate::engine::processes::stochastic_dividends) fn labels(
        &self,
    ) -> &'static [&'static str] {
        match &self.inner {
            PreparedModel::One(_) => &[
                "equity_dividend_correlation",
                "spot_volatility_correlation[0]",
                "dividend_volatility_correlation[0]",
            ],
            PreparedModel::Two(_) => &[
                "equity_dividend_correlation",
                "spot_volatility_correlation[0]",
                "spot_volatility_correlation[1]",
                "dividend_volatility_correlation[0]",
                "dividend_volatility_correlation[1]",
                "volatility_factor_correlation",
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
            _ => Err(invalid("correlation_reverse_family")),
        }
    }
}

fn factor_spd<const D: usize>(
    matrix: [[f64; D]; D],
) -> Result<CorrelationFactor, StochasticDividendError> {
    let factor = CorrelationFactor::compile(
        matrix.iter().map(|r| r.to_vec()).collect(),
        BERGOMI_TWO_FACTOR_CORRELATION_TOLERANCES,
    )
    .map_err(|_| unsupported())?;
    if factor
        .diagnostics()
        .pivots
        .iter()
        .any(|p| !p.is_finite() || *p <= MIN_PIVOT)
    {
        return Err(unsupported());
    }
    Ok(factor)
}

fn lower_derivatives<const N: usize, const D: usize, const C: usize>(
    k: [f64; N],
    matrix: [[f64; D]; D],
    edges: [(usize, usize); C],
    dt: f64,
) -> Result<[[[f64; D]; N]; C], StochasticDividendError> {
    let a: [f64; D] = std::array::from_fn(|i| if i < 2 { 0.0 } else { k[i - 2] * dt });
    let mut c = [[0.0; D]; D];
    let mut kernel = [[0.0; D]; D];
    for i in 0..D {
        for j in 0..D {
            kernel[i][j] =
                ou_kernel_correlation(a[i], a[j]).map_err(|_| invalid("correlation_ou_kernel"))?;
            c[i][j] = matrix[i][j] * kernel[i][j];
        }
    }
    let factor = factor_spd(c)?;
    let l = factor.lower();
    let mut result = [[[0.0; D]; N]; C];
    for (p, rows) in result.iter_mut().enumerate() {
        let (e0, e1) = edges[p];
        let mut dl = [[0.0; D]; D];
        // Tangent of the SAME unpivoted factor used in pricing. Each parameter
        // changes one symmetric off-diagonal pair, not two independent entries.
        for j in 0..D {
            let sum: f64 = (0..j).map(|h| 2.0 * l[j * D + h] * dl[j][h]).sum();
            dl[j][j] = -sum / (2.0 * l[j * D + j]);
            for i in j + 1..D {
                // Do not divide C_ij by rho_ij: zero correlation is supported.
                let dc = if (j, i) == (e0, e1) {
                    kernel[i][j]
                } else {
                    0.0
                };
                let sum: f64 = (0..j)
                    .map(|h| dl[i][h] * l[j * D + h] + l[i * D + h] * dl[j][h])
                    .sum();
                dl[i][j] = (dc - sum - l[i * D + j] * dl[j][j]) / l[j * D + j];
            }
        }
        for (i, row) in rows.iter_mut().enumerate() {
            let scale = Bergomi1Factor::new(k[i], 0.0, 0.0)
                .and_then(|m| m.transition(dt))
                .map_err(|_| invalid("correlation_ou_variance"))?
                .variance
                .sqrt();
            for (j, value) in row.iter_mut().enumerate() {
                *value = scale * dl[i + 2][j];
            }
        }
    }
    if result.iter().flatten().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("correlation_coefficient_derivative"));
    }
    Ok(result)
}

impl<const N: usize, const D: usize, const C: usize> Prepared<N, D, C> {
    fn new(
        kernel: &Kernel<N, D>,
        k: [f64; N],
        matrix: [[f64; D]; D],
        edges: [(usize, usize); C],
        times: &[f64],
        weights: [[f64; N]; C],
    ) -> Result<Self, StochasticDividendError> {
        // Integrated SPD need not imply instantaneous SPD for unequal OU kernels.
        // Independent raw entry partials require an open instantaneous-SPD domain.
        factor_spd(matrix)?;
        let lower = times
            .windows(2)
            .map(|t| lower_derivatives(k, matrix, edges, t[1] - t[0]))
            .collect::<Result<Vec<_>, _>>()?;
        let mut centering = Vec::with_capacity(kernel.steps.len());
        for &t in &times[..times.len() - 1] {
            let mut dc = [0.0; C];
            if t > 0.0 {
                let s = Step::<N, D>::compile(k, matrix, t)?;
                let ds = lower_derivatives(k, matrix, edges, t)?;
                for (j, _) in s.lower[0].iter().enumerate() {
                    let exposure: f64 = (0..N).map(|i| kernel.weights[i] * s.lower[i][j]).sum();
                    for (p, value) in dc.iter_mut().enumerate() {
                        let de: f64 = (0..N)
                            .map(|i| {
                                weights[p][i] * s.lower[i][j] + kernel.weights[i] * ds[p][i][j]
                            })
                            .sum();
                        *value += 2.0 * kernel.vol_of_vol * exposure * de;
                    }
                }
            }
            if dc.iter().any(|v| !v.is_finite()) {
                return Err(invalid("correlation_centering_derivative"));
            }
            centering.push(dc);
        }
        Ok(Self {
            lower: lower.into_boxed_slice(),
            centering: centering.into_boxed_slice(),
            weights,
        })
    }
    fn pullback(
        &self,
        kernel: &Kernel<N, D>,
        normals: &[f64],
        bars: &[f64],
    ) -> Result<Vec<f64>, StochasticDividendError> {
        if normals.len() != D * kernel.steps.len() || bars.len() != kernel.steps.len() {
            return Err(invalid("correlation_reverse_shape"));
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
        let mut out = [0.0; C];
        let mut x_bar = [0.0; N];
        for i in (0..kernel.steps.len()).rev() {
            let step = &kernel.steps[i];
            let z = &normals[D * i..D * (i + 1)];
            let x = trace[i];
            for (p, value) in out.iter_mut().enumerate() {
                for (j, bar) in x_bar.iter().enumerate() {
                    *value += bar
                        * self.lower[i][p][j]
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
            for (p, value) in out.iter_mut().enumerate() {
                let df: f64 = self.weights[p].iter().zip(x).map(|(w, x)| w * x).sum();
                *value += q_bar * kernel.vol_of_vol * (df - self.centering[i][p]);
            }
            for (j, bar) in x_bar.iter_mut().enumerate() {
                *bar += q_bar * kernel.vol_of_vol * kernel.weights[j];
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(invalid("correlation_reverse_result"));
        }
        Ok(out.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    #[test]
    fn correlation_coefficients_match_factor_bumps_including_zero_entries() {
        for matrix in [
            [
                [1.0, -0.25, -0.4, -0.2],
                [-0.25, 1.0, 0.15, -0.1],
                [-0.4, 0.15, 1.0, 0.3],
                [-0.2, -0.1, 0.3, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ] {
            for k in [[0.0, 0.0], [1e-7, 1e-7], [0.8, 2.1], [30.0, 0.2]] {
                for dt in [0.003, 0.125, 1.3] {
                    let tangent = lower_derivatives::<2, 4, 6>(k, matrix, EDGES, dt).unwrap();
                    for (p, &(i, j)) in EDGES.iter().enumerate() {
                        let h = 1e-6;
                        let mut a = matrix;
                        let mut b = matrix;
                        a[i][j] += h;
                        a[j][i] += h;
                        b[i][j] -= h;
                        b[j][i] -= h;
                        let a = Step::<2, 4>::compile(k, a, dt).unwrap();
                        let b = Step::<2, 4>::compile(k, b, dt).unwrap();
                        for (m, row) in tangent[p].iter().enumerate() {
                            for (n, &value) in row.iter().enumerate() {
                                let fd = (a.lower[m][n] - b.lower[m][n]) / (2.0 * h);
                                assert!(
                                    (value - fd).abs() < 3e-6,
                                    "k={k:?}, dt={dt}, p={p}, ({m},{n}), aad={value}, fd={fd}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn factor_correlation_centering_matches_independent_ou_moment_formula() {
        let d = BuehlerDividendModel::new(0.7, 0.6, 0.35, -0.25).unwrap();
        let times = [0.0, 0.13, 0.5, 1.0];
        let b = Bergomi2Factor::new([0.8, 2.1], 0.3, 0.35, [-0.4, -0.2], 0.3).unwrap();
        let kernel = BergomiDividendKernel::two(d, b, [0.15, -0.1], &times).unwrap();
        let prepared = BergomiCorrelationReverse::new(&kernel, -0.25, &times).unwrap();
        let PreparedModel::Two(p) = prepared.inner else {
            panic!("wrong family")
        };
        for (i, &t) in times[..times.len() - 1].iter().enumerate() {
            let w = b.normalized_weights();
            let integral = |rate: f64| -(-rate * t).exp_m1() / rate;
            let cov = 0.3 * integral(2.9);
            let variance =
                w[0] * w[0] * integral(1.6) + w[1] * w[1] * integral(4.2) + 2.0 * w[0] * w[1] * cov;
            let norm2 = 0.65_f64.powi(2) + 0.35_f64.powi(2) + 2.0 * 0.65 * 0.35 * 0.3;
            let expected =
                0.3 * (2.0 * w[0] * w[1] * integral(2.9) - 2.0 * 0.65 * 0.35 / norm2 * variance);
            assert!((p.centering[i][5] - expected).abs() < 2e-13);
            assert!(p.centering[i][..5].iter().all(|v| v.abs() < 2e-13));
        }
    }
}
