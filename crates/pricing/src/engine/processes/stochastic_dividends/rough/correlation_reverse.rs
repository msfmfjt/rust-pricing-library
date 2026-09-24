//! Transpose the finite Volterra history into raw Brownian-correlation entries.
//! H, eta, grid and independent normals are fixed for these partials. The fourth
//! (near-cell residual) normal is independent: it has no correlation derivative.
use super::*;

const MIN_PIVOT: f64 = 1e-10;

pub(in crate::engine) struct RoughCorrelationReverse {
    // One row per symmetric correlation entry, one column per independent
    // Brownian normal. These nine coefficients are shared by all paths/steps.
    volatility_loading: [[f64; 3]; 3],
}
fn unsupported() -> StochasticDividendError {
    StochasticDividendError::Unsupported {
        feature: "rough correlation AAD requires instantaneous correlation pivots > 1e-10; fixed-correlation price/AAD/Gamma retain their domains",
    }
}
impl RoughCorrelationReverse {
    pub(in crate::engine) const LABELS: [&'static str; 3] = [
        "equity_dividend_correlation",
        "spot_volatility_correlation[0]",
        "dividend_volatility_correlation[0]",
    ];

    pub(in crate::engine) fn new(
        kernel: &RoughDividendKernel,
        sd: f64,
    ) -> Result<Self, StochasticDividendError> {
        let sv = kernel.factor.correlation();
        let dv = kernel.dividend_volatility_correlation;
        let factor = CorrelationFactor::compile(
            vec![vec![1.0, sd, sv], vec![sd, 1.0, dv], vec![sv, dv, 1.0]],
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
        let l = factor.lower();
        let mut volatility_loading = [[0.0; 3]; 3];
        for (rows, edge) in volatility_loading.iter_mut().zip([(0, 1), (0, 2), (1, 2)]) {
            // Differentiate the actual unpivoted factor. Each input changes both
            // symmetric entries once. Never divide by a correlation: zero is valid.
            let mut dl = [[0.0; 3]; 3];
            for j in 0..3 {
                let sum: f64 = (0..j).map(|k| 2.0 * l[j * 3 + k] * dl[j][k]).sum();
                dl[j][j] = -sum / (2.0 * l[j * 3 + j]);
                for i in j + 1..3 {
                    let dc = if (j, i) == edge { 1.0 } else { 0.0 };
                    let sum: f64 = (0..j)
                        .map(|k| dl[i][k] * l[j * 3 + k] + l[i * 3 + k] * dl[j][k])
                        .sum();
                    dl[i][j] = (dc - sum - l[i * 3 + j] * dl[j][j]) / l[j * 3 + j];
                }
            }
            *rows = dl[2];
        }
        if volatility_loading.iter().flatten().any(|x| !x.is_finite()) {
            return Err(invalid("rough_correlation_coefficients"));
        }
        Ok(Self { volatility_loading })
    }

    pub(in crate::engine) fn pullback(
        &self,
        kernel: &RoughDividendKernel,
        normals: &[f64],
        loading_bars: &[f64],
    ) -> Result<[f64; 3], StochasticDividendError> {
        if loading_bars.len() != kernel.steps.len() || loading_bars.iter().any(|x| !x.is_finite()) {
            return Err(invalid("rough_correlation_loading_seeds"));
        }
        let (drivers, _) = kernel.driver_path(normals)?;
        let eta = kernel.factor.vol_of_vol();
        let mut increment_bars = vec![NeumaierSum::new(); kernel.steps.len()];
        for (i, &bar) in loading_bars.iter().enumerate().rev() {
            if i == 0 || bar == 0.0 || eta == 0.0 {
                continue;
            }
            let loading = (0.5 * eta * drivers[i] - 0.25 * eta * eta * kernel.variances[i]).exp();
            positive(loading, "rough_correlation_loading")?;
            let driver_bar = bar * loading * (0.5 * eta);
            // d Var_grid / d rho = 0: the volatility Brownian marginal has
            // unit variance for every admissible correlation. Do not invent a
            // centering tangent from the equity/dividend cross-covariances.
            increment_bars[i - 1].add(driver_bar * kernel.steps[i - 1].average);
            for (target, &w) in increment_bars.iter_mut().zip(kernel.weights[i].iter()) {
                target.add(driver_bar * w);
            }
        }
        let mut out = [NeumaierSum::new(); 3];
        for ((bar, step), z) in increment_bars
            .iter()
            .zip(kernel.steps.iter())
            .zip(normals.as_chunks::<4>().0)
        {
            let bar = bar.total();
            if bar == 0.0 {
                continue;
            }
            for (target, dl) in out.iter_mut().zip(&self.volatility_loading) {
                let dz: f64 = dl.iter().zip(z).map(|(l, z)| l * z).sum();
                target.add(bar * step.root_dt * dz);
            }
        }
        let out = out.map(|x| x.total());
        if out.iter().any(|x| !x.is_finite()) {
            return Err(invalid("rough_correlation_reverse"));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn kernel(h: f64, eta: f64, rho: [f64; 3], times: &[f64]) -> RoughDividendKernel {
        RoughDividendKernel::compile(
            BuehlerDividendModel::new(0.7, 0.6, 0.35, rho[0]).unwrap(),
            RoughBergomi::new(h, eta, rho[1]).unwrap(),
            rho[2],
            times,
        )
        .unwrap()
    }

    #[test]
    fn coefficient_tangents_preserve_joint_covariance_and_marginal_centering() {
        for h in [0.01, 0.1, 0.49, 0.5] {
            for rho in [[0.0; 3], [-0.25, -0.4, 0.15], [0.8, 0.6, 0.5]] {
                let times = [0.0, 0.031, 0.17, 0.36, 1.0];
                let k = kernel(h, 0.6, rho, &times);
                let ctx = RoughCorrelationReverse::new(&k, rho[0]).unwrap();
                let b = ((1.0 - rho[0]) * (1.0 + rho[0])).sqrt();
                for (p, tangent) in ctx.volatility_loading.iter().enumerate() {
                    let eps = 1e-6;
                    let mut up = rho;
                    let mut down = rho;
                    up[p] += eps;
                    down[p] -= eps;
                    let u = kernel(h, 0.6, up, &times);
                    let d = kernel(h, 0.6, down, &times);
                    for (j, &value) in tangent.iter().enumerate() {
                        assert!(
                            (value - (u.vol_loading[j] - d.vol_loading[j]) / (2.0 * eps)).abs()
                                < 3e-8
                        );
                    }
                    // Independent analytic differential of the complete newest
                    // cell law (not a price finite difference).
                    for cell in k.steps.iter() {
                        let f = [cell.root_dt, 0.0, 0.0, 0.0];
                        let div = [rho[0] * cell.root_dt, b * cell.root_dt, 0.0, 0.0];
                        let mut dv = [0.0; 4];
                        let mut dd = [0.0; 4];
                        let mut vol = [0.0; 4];
                        for (j, &v) in k.vol_loading.iter().enumerate() {
                            vol[j] = cell.root_dt * v;
                            dv[j] = cell.root_dt * tangent[j];
                        }
                        if p == 0 {
                            dd = [cell.root_dt, -rho[0] / b * cell.root_dt, 0.0, 0.0];
                        }
                        let near: [f64; 4] = std::array::from_fn(|j| {
                            cell.average * vol[j] + if j == 3 { cell.residual } else { 0.0 }
                        });
                        let dn = dv.map(|x| cell.average * x);
                        let dot = |a: [f64; 4], b: [f64; 4]| {
                            a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>()
                        };
                        let dt = cell.root_dt * cell.root_dt;
                        for (actual, expected) in [
                            (dot(f, dd), if p == 0 { dt } else { 0.0 }),
                            (dot(f, dv), if p == 1 { dt } else { 0.0 }),
                            (dot(dd, vol) + dot(div, dv), if p == 2 { dt } else { 0.0 }),
                            (dot(f, dn), if p == 1 { dt * cell.average } else { 0.0 }),
                            (
                                dot(dd, near) + dot(div, dn),
                                if p == 2 { dt * cell.average } else { 0.0 },
                            ),
                            (2.0 * dot(vol, dv), 0.0),
                            (2.0 * dot(near, dn), 0.0),
                        ] {
                            assert!(
                                (actual - expected).abs() < 2e-12,
                                "H={h} p={p} {actual} vs {expected}"
                            );
                        }
                    }
                    assert_eq!(u.variances, k.variances);
                    assert_eq!(d.variances, k.variances);
                }
            }
        }
    }
    #[test]
    fn history_transpose_matches_recompiled_loadings_and_is_causal() {
        let times = [0.0, 0.031, 0.17, 0.36, 0.8, 1.0];
        let z: Vec<f64> = (0..20).map(|i| ((i as f64 + 0.4) * 1.7).sin()).collect();
        let seeds = [0.3, -0.7, 0.2, 0.9, -0.1];
        for h in [0.01, 0.1, 0.49, 0.5] {
            for eta in [0.0, 0.6] {
                for rho in [[0.0; 3], [-0.25, -0.4, 0.15]] {
                    let k = kernel(h, eta, rho, &times);
                    let ctx = RoughCorrelationReverse::new(&k, rho[0]).unwrap();
                    let actual = ctx.pullback(&k, &z, &seeds).unwrap();
                    let value = |rho| {
                        kernel(h, eta, rho, &times)
                            .volatility_loadings(&z)
                            .unwrap()
                            .iter()
                            .zip(seeds)
                            .map(|(v, s)| v * s)
                            .sum::<f64>()
                    };
                    for (p, &aad) in actual.iter().enumerate() {
                        for eps in [1e-5, 1e-6] {
                            let mut up = rho;
                            let mut down = rho;
                            up[p] += eps;
                            down[p] -= eps;
                            let fd = (value(up) - value(down)) / (2.0 * eps);
                            assert!(
                                (aad - fd).abs() < 3e-8,
                                "H={h} eta={eta} p={p} {aad} vs {fd}"
                            );
                        }
                    }
                    if eta == 0.0 {
                        assert_eq!(actual, [0.0; 3]);
                    }
                    let mut future = z.clone();
                    future[16..].fill(5.0);
                    assert_eq!(actual, ctx.pullback(&k, &future, &seeds).unwrap());
                    let mut bad = seeds;
                    bad[0] = f64::NAN;
                    assert!(ctx.pullback(&k, &z, &bad).is_err());
                    assert!(ctx.pullback(&k, &z[..19], &seeds).is_err());
                    let mut bad = z.clone();
                    bad[0] = f64::NAN;
                    assert!(ctx.pullback(&k, &bad, &seeds).is_err());
                }
            }
        }
    }
}
