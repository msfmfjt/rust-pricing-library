//! Two-step nonflat variance-node operator with an exact Gaussian first state.
use super::{VegaKtConvolutionPlan, invalid};
use crate::risk::RiskConfigError;
use pricing_numerics::{NeumaierSum, standard_normal_cdf, standard_normal_pdf};

#[derive(Clone, Debug, PartialEq)]
pub struct VegaKtVarianceOperatorAccuracy {
    pub quadrature_order: usize,
    pub refinement_maximum_difference: f64,
    /// Analytic bound on each omitted operator entry; the quadrature difference
    /// is a convergence diagnostic, not a rigorous total error bound.
    pub omitted_tail_entry_bound: f64,
    pub entry_tolerance: f64,
}

pub(crate) fn legendre(order: usize) -> Result<Vec<(f64, f64)>, RiskConfigError> {
    let mut rule = vec![(0., 0.); order];
    for i in 0..order / 2 {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (order as f64 + 0.5)).cos();
        let mut converged = false;
        for _ in 0..32 {
            let (mut p0, mut p1) = (1., z);
            for k in 2..=order {
                let p = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p;
            }
            let derivative = order as f64 * (z * p1 - p0) / (z * z - 1.);
            let next = z - p1 / derivative;
            if (next - z).abs() < 2e-15 {
                z = next;
                converged = true;
                break;
            }
            z = next;
        }
        if !converged {
            return Err(invalid("variance operator quadrature rule failed"));
        }
        let (mut p0, mut p1) = (1., z);
        for k in 2..=order {
            let p = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
            p0 = p1;
            p1 = p;
        }
        let derivative = order as f64 * (z * p1 - p0) / (z * z - 1.);
        let weight = 2. / ((1. - z * z) * derivative * derivative);
        rule[i] = (-z, weight);
        rule[order - 1 - i] = (z, weight);
    }
    Ok(rule)
}

impl VegaKtConvolutionPlan {
    /// B[j,k] = E[hat_j(X1) * d Black(S1,K_k,v(X1)*dt)/d v(X1)].
    /// Coordinates are annual VARIANCE nodes, not effective-volatility hats.
    /// Source X1 is exactly N(-V1/2,V1); v is linearly interpolated with flat
    /// boundary extension. This does not assume a Gaussian terminal marginal.
    pub fn from_gaussian_variance_source(
        spot: f64,
        source_variance: f64,
        step_dt: f64,
        source_log_nodes: &[f64],
        local_variances: &[f64],
        gamma_strikes: Vec<f64>,
        relative_regularization: f64,
    ) -> Result<(Self, VegaKtVarianceOperatorAccuracy), RiskConfigError> {
        let (m, n) = (source_log_nodes.len(), gamma_strikes.len());
        if !spot.is_finite()
            || spot <= 0.
            || !source_variance.is_finite()
            || source_variance <= 0.
            || !step_dt.is_finite()
            || step_dt <= 0.
            || !(2..=129).contains(&m)
            || !(2..=129).contains(&n)
            || source_log_nodes.iter().any(|x| !x.is_finite())
            || source_log_nodes.windows(2).any(|x| x[0] >= x[1])
            || local_variances.len() != m
            || local_variances.iter().any(|v| !v.is_finite() || *v <= 0.)
            || gamma_strikes.iter().any(|k| !k.is_finite() || *k <= 0.)
            || gamma_strikes.windows(2).any(|k| k[0] >= k[1])
        {
            return Err(invalid("invalid Gaussian-source variance operator inputs"));
        }
        let sd = source_variance.sqrt();
        let mu = -source_variance / 2.;
        let lo = mu - 12. * sd;
        let hi = mu + 12. * sd;
        let min_variance = local_variances
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let tail = spot
            * (standard_normal_cdf(-12. - sd) + standard_normal_cdf(-12. + sd))
            * standard_normal_pdf(0.)
            * step_dt.sqrt()
            / (2. * min_variance.sqrt());
        if !tail.is_finite() {
            return Err(invalid("variance operator tail bound overflow"));
        }
        let mut cuts = vec![lo];
        cuts.extend(
            source_log_nodes
                .iter()
                .copied()
                .filter(|&x| lo < x && x < hi),
        );
        cuts.push(hi);
        let integrate = |order| -> Result<Vec<Vec<f64>>, RiskConfigError> {
            let mut sums = vec![vec![NeumaierSum::new(); n]; m];
            for (z, w) in legendre(order)? {
                for cell in cuts.windows(2) {
                    let x = (cell[0] + cell[1]) / 2. + (cell[1] - cell[0]) / 2. * z;
                    let left = source_log_nodes
                        .partition_point(|v| *v <= x)
                        .saturating_sub(1)
                        .min(m - 2);
                    let weight = ((x - source_log_nodes[left])
                        / (source_log_nodes[left + 1] - source_log_nodes[left]))
                        .clamp(0., 1.);
                    let variance =
                        (1. - weight) * local_variances[left] + weight * local_variances[left + 1];
                    let step_sd = (variance * step_dt).sqrt();
                    let s = spot * x.exp();
                    let probability =
                        w * (cell[1] - cell[0]) / 2. * standard_normal_pdf((x - mu) / sd) / sd;
                    for (j, &strike) in gamma_strikes.iter().enumerate() {
                        let d1 = (spot.ln() - strike.ln() + x) / step_sd + step_sd / 2.;
                        let value =
                            probability * s * standard_normal_pdf(d1) * step_dt / (2. * step_sd);
                        if !value.is_finite() {
                            return Err(invalid("nonfinite variance operator integrand"));
                        }
                        sums[left][j].add((1. - weight) * value);
                        sums[left + 1][j].add(weight * value);
                    }
                }
            }
            Ok(sums
                .into_iter()
                .map(|r| r.into_iter().map(NeumaierSum::total).collect())
                .collect())
        };
        let coarse = integrate(32)?;
        let fine = integrate(64)?;
        let scale = fine.iter().flatten().copied().fold(0., f64::max);
        let difference = coarse
            .iter()
            .flatten()
            .zip(fine.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        let tolerance = 1e-10 * scale;
        if scale <= 0. || !scale.is_finite() || difference > tolerance || tail > tolerance {
            return Err(invalid(
                "variance operator quadrature refinement or tail budget failed",
            ));
        }
        let accuracy = VegaKtVarianceOperatorAccuracy {
            quadrature_order: 64,
            refinement_maximum_difference: difference,
            omitted_tail_entry_bound: tail,
            entry_tolerance: tolerance,
        };
        Ok((
            Self::from_operator(fine, gamma_strikes, relative_regularization)?,
            accuracy,
        ))
    }
}
