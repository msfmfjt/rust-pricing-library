//! Exact model measure of the spatial hats used by native adjoint deposition.
use super::vegakt::normal_interval_probability;
use crate::core::{FiniteF64, PositiveF64};
use crate::risk::{RiskConfigError, mass_lumped_hat_areas};
use pricing_numerics::{NeumaierSum, standard_normal_pdf};

/// E[S_t^2 h_j(log(S_t/S0))] for zero-carry lognormal S_t. The same
/// piecewise-linear log hats, including clamped endpoint tails, are used by the
/// native reverse. These are integrated masses, not point density times area.
pub fn lognormal_spot_squared_hat_masses(
    spot: f64,
    total_variance: f64,
    nodes: &[f64],
) -> Result<Vec<f64>, RiskConfigError> {
    PositiveF64::new(spot, "density_mass_spot")?;
    PositiveF64::new(total_variance, "density_mass_variance")?;
    mass_lumped_hat_areas(nodes)?;
    let scale = FiniteF64::new(spot * spot * total_variance.exp(), "density_mass_scale")?.get();
    let mean = 1.5 * total_variance;
    gaussian_log_hat_probabilities(mean, total_variance, nodes)?
        .into_iter()
        .map(|v| Ok(FiniteF64::new(scale * v, "density_hat_mass")?.get()))
        .collect()
}

/// Gaussian expectations of clamped piecewise-linear hats. Zero variance is a
/// point evaluation, including the time-zero boundary of a conditional bridge.
pub fn gaussian_log_hat_probabilities(
    mean: f64,
    variance: f64,
    nodes: &[f64],
) -> Result<Vec<f64>, RiskConfigError> {
    FiniteF64::new(mean, "gaussian_hat_mean")?;
    FiniteF64::new(variance, "gaussian_hat_variance")?;
    mass_lumped_hat_areas(nodes)?;
    if variance < 0. {
        return Err(RiskConfigError::VegaKtChainInput {
            quantity: "negative Gaussian hat variance",
        });
    }
    if variance == 0. {
        let mut out = vec![0.; nodes.len()];
        let upper = nodes.partition_point(|&x| x <= mean);
        if upper == 0 {
            out[0] = 1.;
        } else if upper == nodes.len() {
            out[upper - 1] = 1.;
        } else {
            let weight = (mean - nodes[upper - 1]) / (nodes[upper] - nodes[upper - 1]);
            out[upper - 1] = 1. - weight;
            out[upper] = weight;
        }
        return Ok(out);
    }
    let sd = variance.sqrt();
    let z = nodes
        .iter()
        .map(|x| FiniteF64::new((x - mean) / sd, "density_mass_z").map(|v| v.get()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut sums = vec![NeumaierSum::new(); nodes.len()];
    sums[0].add(normal_interval_probability(
        f64::NEG_INFINITY,
        z[0],
        f64::INFINITY,
    ));
    sums[nodes.len() - 1].add(normal_interval_probability(
        z[z.len() - 1],
        f64::INFINITY,
        f64::INFINITY,
    ));
    // Positive Gaussian quadrature avoids subtracting nearly equal density
    // moments at a narrow/tail hat. Beyond |z|=40 all binary64 Gaussian mass
    // underflows in probability units. Extremely small weighted tails can still
    // underflow; callers must retain a density-domain exclusion audit.
    let glx = [
        -0.960_289_856_497_536_3,
        -0.796_666_477_413_626_7,
        -0.525_532_409_916_329,
        -0.183_434_642_495_649_8,
        0.183_434_642_495_649_8,
        0.525_532_409_916_329,
        0.796_666_477_413_626_7,
        0.960_289_856_497_536_3,
    ];
    let glw = [
        0.101_228_536_290_376_3,
        0.222_381_034_453_374_5,
        0.313_706_645_877_887_3,
        0.362_683_783_378_362,
        0.362_683_783_378_362,
        0.313_706_645_877_887_3,
        0.222_381_034_453_374_5,
        0.101_228_536_290_376_3,
    ];
    for (i, pair) in z.windows(2).enumerate() {
        let full_width = FiniteF64::new(
            (nodes[i + 1] - nodes[i]) / sd,
            "density_mass_interval_width",
        )?
        .get();
        let lo_offset = 0.0_f64.max(-40. - pair[0]);
        let hi_offset = full_width.min(40. - pair[0]);
        if lo_offset >= hi_offset {
            continue;
        }
        let count = ((hi_offset - lo_offset) / 0.25).ceil().max(1.) as usize;
        let width = (hi_offset - lo_offset) / count as f64;
        for part in 0..count {
            for (&x, &w) in glx.iter().zip(&glw) {
                let offset = lo_offset + (part as f64 + 0.5 + 0.5 * x) * width;
                let at = pair[0] + offset;
                let hat = offset / full_width;
                let mass = 0.5 * width * w * standard_normal_pdf(at);
                sums[i].add((1. - hat) * mass);
                sums[i + 1].add(hat * mass);
            }
        }
    }
    let normalized: Vec<_> = sums.into_iter().map(|s| s.total()).collect();
    let mass = normalized.iter().copied().collect::<NeumaierSum>().total();
    if !mass.is_finite() || (mass - 1.).abs() > 2e-13 {
        return Err(RiskConfigError::VegaKtChainInput {
            quantity: "integrated density hat mass defect",
        });
    }
    normalized
        .into_iter()
        .map(|v| {
            if v < 0. {
                return Err(RiskConfigError::VegaKtChainInput {
                    quantity: "negative integrated density hat mass",
                });
            }
            Ok(FiniteF64::new(v, "gaussian_hat_probability")?.get())
        })
        .collect()
}

/// Subdivide each supplied interval without moving its endpoints. The bound is
/// independent of observed risk values; refinement never fits an acceptance residual.
pub fn refine_vega_kt_log_nodes(
    nodes: &[f64],
    maximum_spacing: f64,
) -> Result<Vec<f64>, RiskConfigError> {
    mass_lumped_hat_areas(nodes)?;
    PositiveF64::new(maximum_spacing, "recovery_maximum_log_spacing")?;
    let mut out = vec![nodes[0]];
    for pair in nodes.windows(2) {
        let count = ((pair[1] - pair[0]) / maximum_spacing).ceil().max(1.);
        if !count.is_finite() || count > 1024. || out.len() + count as usize > 1025 {
            return Err(RiskConfigError::VegaKtChainInput {
                quantity: "recovery refinement exceeds 1025-node budget",
            });
        }
        for i in 1..count as usize {
            let node = pair[0] + (pair[1] - pair[0]) * (i as f64 / count);
            if node <= *out.last().unwrap() || node >= pair[1] {
                return Err(RiskConfigError::VegaKtChainInput {
                    quantity: "unrepresentable recovery refinement",
                });
            }
            out.push(node);
        }
        out.push(pair[1]);
    }
    Ok(out)
}
