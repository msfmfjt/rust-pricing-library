//! Exact joint covariance for spots, all volatility OU states and the common
//! HW state/integral. Rate volatility knots are integrated inside every step.
use super::super::lsv::LsvDrivers;
use super::*;
use crate::market::{CorrelationFactor, CorrelationToleranceConfig};
use crate::models::hull_white::b;

pub(in crate::engine::multi_asset) fn compile_drivers(
    correlation: &CorrelationTermStructure,
    times: &[f64],
    intervals: &[usize],
    configs: &[Option<MultiAssetBergomiLsvConfig>],
    supplied: Option<Vec<Vec<Vec<f64>>>>,
    hw: &MultiAssetHullWhiteConfig,
) -> Result<LsvDrivers, E> {
    let n = configs.len();
    let components: Vec<_> = configs
        .iter()
        .enumerate()
        .flat_map(|(i, c)| {
            c.as_ref()
                .map(|c| c.components())
                .unwrap_or_default()
                .into_iter()
                .map(move |f| (i, f))
        })
        .collect();
    let d = n + components.len();
    if hw.lsv_targets.len() != n
        || hw.rate_correlations.len() != d
        || hw
            .rate_correlations
            .iter()
            .any(|v| !v.is_finite() || v.abs() > 1.0)
    {
        return Err(E::Invalid(
            "HW target/rate-correlation dimensions or values are invalid",
        ));
    }
    if let Some(matrices) = &supplied
        && (matrices.len() != correlation.entries().len()
            || matrices
                .iter()
                .any(|m| m.len() != d + 1 || m.iter().any(|r| r.len() != d + 1)))
    {
        return Err(E::Invalid(
            "HW full matrices must have one entry per correlation date and order [spots, volatility factors, rate]",
        ));
    }
    let core = supplied.as_ref().map(|matrices| {
        matrices
            .iter()
            .map(|m| m[..d].iter().map(|r| r[..d].to_vec()).collect())
            .collect()
    });
    let marginal = if components.is_empty() {
        None
    } else {
        LsvDrivers::compile(correlation, times, intervals, configs, core)?
    };
    let mut entries = Vec::new();
    for (index, (_, spot)) in correlation.entries().iter().enumerate() {
        let base = marginal
            .as_ref()
            .map_or(spot.canonical(), |m| m.entries[index].canonical());
        let full = if let Some(m) = &supplied {
            m[index].clone()
        } else {
            let mut m = vec![vec![0.0; d + 1]; d + 1];
            for i in 0..d {
                m[i][..d].copy_from_slice(&base[i * d..(i + 1) * d]);
                m[i][d] = hw.rate_correlations[i];
                m[d][i] = hw.rate_correlations[i];
            }
            m[d][d] = 1.0;
            m
        };
        let full = CorrelationFactor::compile(full, correlation.tolerances())?;
        for i in 0..d {
            if full.canonical()[i * (d + 1) + d] != hw.rate_correlations[i] {
                return Err(E::Invalid(
                    "HW rate correlations must match the constant calibrated marginals at every date",
                ));
            }
            for j in 0..d {
                if full.canonical()[i * (d + 1) + j] != base[i * d + j] {
                    return Err(E::Invalid(
                        "HW equity/volatility block must match the configured marginals",
                    ));
                }
            }
        }
        entries.push(full);
    }
    let rates = &hw.rate_model;
    let reversions: Vec<_> = std::iter::repeat_n(0.0, n)
        .chain(components.iter().map(|(_, f)| f.mean_reversion()))
        .collect();
    let mut joint = Vec::new();
    let mut scales = Vec::new();
    let mut permutations = Vec::new();
    for (step, pair) in times.windows(2).enumerate() {
        let brownian = entries[intervals[step]].canonical();
        let mut covariance = vec![vec![0.0; d + 2]; d + 2];
        let dt = pair[1] - pair[0];
        let rate = rates
            .transition(
                pair[0],
                pair[1],
                0.0,
                HybridCorrelation::new(0.0, 0.0, 0.0).map_err(E::numerical)?,
            )
            .map_err(E::numerical)?
            .covariance;
        covariance[d][d] = rate[2][2];
        covariance[d + 1][d + 1] = rate[3][3];
        covariance[d][d + 1] = rate[2][3];
        covariance[d + 1][d] = rate[3][2];
        for i in 0..d {
            for j in 0..d {
                covariance[i][j] = brownian[i * (d + 1) + j] * b(reversions[i] + reversions[j], dt);
            }
            let cross = rates
                .transition(
                    pair[0],
                    pair[1],
                    reversions[i],
                    HybridCorrelation::new(0.0, 0.0, 1.0).map_err(E::numerical)?,
                )
                .map_err(E::numerical)?
                .covariance;
            for (col, source) in [(d, 2), (d + 1, 3)] {
                covariance[i][col] = brownian[i * (d + 1) + d] * cross[1][source];
                covariance[col][i] = covariance[i][col];
            }
        }
        let scale: Vec<_> = (0..d + 2).map(|i| covariance[i][i].sqrt()).collect();
        let mut normalized = vec![vec![0.0; d + 2]; d + 2];
        for i in 0..d + 2 {
            for j in 0..d + 2 {
                normalized[i][j] = if i == j {
                    1.0
                } else if scale[i] == 0.0 || scale[j] == 0.0 {
                    0.0
                } else {
                    covariance[i][j] / scale[i] / scale[j]
                };
            }
        }
        let (factor, order) = pivoted_suffix(normalized, n, correlation.tolerances())?;
        joint.push(factor);
        permutations.push(order);
        scales.push(scale);
    }
    Ok(LsvDrivers {
        entries,
        intervals: joint,
        permutations: Some(permutations),
        scales,
        asset_indices: components.iter().map(|(i, _)| *i).collect(),
    })
}

/// Preserve spot draws, then select the largest remaining conditional variance.
/// Nearly equal OU kernels can otherwise put a tiny pivot before a much larger
/// rate-integral residual. Only the order changes: CorrelationFactor still
/// validates the original entries with the caller's unchanged PSD tolerances.
fn pivoted_suffix(
    matrix: Vec<Vec<f64>>,
    prefix: usize,
    tolerances: CorrelationToleranceConfig,
) -> Result<(CorrelationFactor, Vec<usize>), E> {
    let n = matrix.len();
    let scale = matrix
        .iter()
        .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
        .fold(1.0, f64::max);
    let zero = tolerances
        .zero_pivot_abs_tol
        .max(tolerances.zero_pivot_rel_tol * scale);
    let mut order: Vec<_> = (0..n).collect();
    let mut lower = vec![vec![0.0; n]; n];
    for j in 0..n {
        let residual = |i: usize| matrix[i][i] - lower[i][..j].iter().map(|v| v * v).sum::<f64>();
        if j >= prefix {
            let mut best = j;
            for i in j + 1..n {
                if residual(order[i]) > residual(order[best]) {
                    best = i;
                }
            }
            order.swap(j, best);
        }
        let row = order[j];
        let pivot = residual(row);
        if pivot > zero {
            lower[row][j] = pivot.sqrt();
            for &i in &order[j + 1..] {
                let sum = (0..j).map(|k| lower[i][k] * lower[row][k]).sum::<f64>();
                lower[i][j] = (matrix[i][row] - sum) / lower[row][j];
            }
        }
    }
    let permuted = order
        .iter()
        .map(|&i| order.iter().map(|&j| matrix[i][j]).collect())
        .collect();
    Ok((CorrelationFactor::compile(permuted, tolerances)?, order))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_one_brownian_different_kernels_reconstruct_without_jitter() {
        // Independent midpoint Gram matrix of seven exponential kernels and
        // one integrated-rate kernel, all driven by the same Brownian motion.
        let k = [0.0, 0.0, 1.0, 0.15, 1.0, 0.15, 0.1, -1.0];
        let kernel = |i: usize, t: f64| {
            if i == 7 {
                -(-0.1 * t).exp_m1() / 0.1
            } else {
                (-k[i] * t).exp()
            }
        };
        let mut gram = vec![vec![0.0; 8]; 8];
        for (i, row) in gram.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = (0..2048)
                    .map(|s| {
                        let t = (s as f64 + 0.5) * 0.5 / 2048.0;
                        kernel(i, t) * kernel(j, t) * 0.5 / 2048.0
                    })
                    .sum();
            }
        }
        let scales: Vec<_> = (0..8).map(|i| gram[i][i].sqrt()).collect();
        let corr: Vec<Vec<f64>> = (0..8)
            .map(|i| {
                (0..8)
                    .map(|j| {
                        if k[i] == k[j] {
                            1.0
                        } else {
                            gram[i][j] / scales[i] / scales[j]
                        }
                    })
                    .collect()
            })
            .collect();
        let tol = CorrelationToleranceConfig {
            symmetry_abs_tol: 1e-12,
            diagonal_abs_tol: 1e-12,
            psd_abs_tol: 1e-12,
            psd_rel_tol: 1e-12,
            zero_pivot_abs_tol: 1e-12,
            zero_pivot_rel_tol: 1e-12,
        };
        let (factor, order) = pivoted_suffix(corr.clone(), 2, tol).unwrap();
        assert_eq!(&order[..2], &[0, 1]);
        assert!(factor.diagnostics().rank < 8);
        let l = factor.lower();
        for i in 0..8 {
            for j in 0..8 {
                let actual: f64 = (0..8).map(|c| l[i * 8 + c] * l[j * 8 + c]).sum();
                assert!((actual - corr[order[i]][order[j]]).abs() < 8e-12);
            }
        }
    }
}
