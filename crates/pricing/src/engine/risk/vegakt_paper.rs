//! Paper equations (4), (5), and (9), with explicit hedge/risk sign conventions.
//! These operations do not resolve the initial-time Gamma recovery problem.

use crate::core::{FiniteF64, PositiveF64};
use crate::risk::{
    Equation11GammaRow, ReportingIvBasis, ReportingIvProjectionStats, RiskConfigError,
    VegaKtProjection, VegaKtVanillaProjection, integrate_local_gamma_total_variance_transition,
    vega_kt_projection_from_parts,
};
use pricing_numerics::{NeumaierSum, standard_normal_pdf};

impl Equation11GammaRow {
    /// Equation (4) at this row's recovered maturity: propagated next Gamma
    /// minus current Gamma. Both rows must use the same drift-free spot unit.
    /// Each jet is [v, dv/dS, d2v/dS2] at one current active strike, where
    /// v = local_variance * next.delta_t. Derivatives hold that interval fixed.
    /// This method uses constant extrapolation of the next row's active tails;
    /// callers must audit truncation separately and supply smooth variance jets.
    pub fn vanilla_hedge_density_from_next(
        &self,
        next: &Self,
        step_variance_jets: &[[f64; 3]],
    ) -> Result<Vec<f64>, RiskConfigError> {
        if self.recovered_maturity != next.source_maturity {
            return Err(RiskConfigError::VegaKtGammaRowTimeMismatch {
                expected_bits: self.recovered_maturity.get().to_bits(),
                actual_bits: next.source_maturity.get().to_bits(),
            });
        }
        if step_variance_jets.len() != self.strike_nodes.len() {
            return Err(RiskConfigError::VegaKtRecoveryLengthMismatch {
                quantity: "step variance jet",
                expected: self.strike_nodes.len(),
                actual: step_variance_jets.len(),
            });
        }
        self.strike_nodes
            .iter()
            .zip(&self.gamma_values)
            .zip(step_variance_jets)
            .map(|((&spot, &gamma), &[variance, first, second])| {
                let propagated = integrate_local_gamma_total_variance_transition(
                    spot,
                    variance,
                    first,
                    second,
                    &next.strike_nodes,
                    &next.gamma_values,
                )?;
                Ok(FiniteF64::new(propagated - gamma, "vega_kt_vanilla_hedge_density")?.get())
            })
            .collect()
    }

    /// Terminal equation (4), using Gamma(t_{n+1}, K) = 0.
    /// Values are hedge density per unit strike, not quadrature weights.
    #[must_use]
    pub fn terminal_vanilla_hedge_density(&self) -> Vec<f64> {
        self.gamma_values.iter().map(|value| -value).collect()
    }
}

/// Equation (9) at an exact reporting expiry, in the paper's zero-carry unit.
///
/// Weights already include dK quadrature (or represent Dirac masses). The
/// continuum Vega multiplies each weight BEFORE transposed IV interpolation;
/// nodal Vega divides the result to obtain offsetting vanilla hedge quantities.
/// Position risk is the opposite sign, in currency per unit absolute IV.
///
/// Zero projection residual means only that the supplied finite measure was
/// allocated. It does not certify Gamma recovery, omitted tails, the initial
/// boundary, or time interpolation. Those remain the caller's responsibility.
pub fn project_vanilla_hedge_measure_to_reporting_iv(
    basis: &ReportingIvBasis,
    maturity: f64,
    initial_spot: f64,
    strikes: &[f64],
    integrated_hedge_weights: &[f64],
) -> Result<VegaKtVanillaProjection, RiskConfigError> {
    PositiveF64::new(initial_spot, "vega_kt_initial_spot")?;
    PositiveF64::new(maturity, "vega_kt_vanilla_maturity")?;
    let Some(time_index) = basis.maturity_nodes().iter().position(|t| *t == maturity) else {
        return Err(RiskConfigError::VegaKtProjectionMaturityNotNode {
            bits: maturity.to_bits(),
        });
    };
    if integrated_hedge_weights.len() != strikes.len() {
        return Err(RiskConfigError::VegaKtRecoveryLengthMismatch {
            quantity: "integrated vanilla hedge weight",
            expected: strikes.len(),
            actual: integrated_hedge_weights.len(),
        });
    }
    let mut sums = vec![NeumaierSum::new(); basis.bucket_count()];
    let mut total = NeumaierSum::new();
    let mut stats = ReportingIvProjectionStats::default();
    let mut left = NeumaierSum::new();
    let mut right = NeumaierSum::new();
    let mut point_buckets = vec![0.0; basis.bucket_count()];
    for (&strike, &weight) in strikes.iter().zip(integrated_hedge_weights) {
        PositiveF64::new(strike, "vega_kt_vanilla_strike")?;
        FiniteF64::new(weight, "vega_kt_integrated_hedge_weight")?;
        let relative = (strike - initial_spot) / initial_spot;
        let x = if relative.abs() < 0.5 {
            relative.ln_1p()
        } else {
            strike.ln() - initial_spot.ln()
        };
        let sigma = basis.interpolate(maturity, x)?.value();
        let risk = FiniteF64::new(
            -weight * vanilla_vega(initial_spot, maturity, x, sigma)?,
            "vega_kt_vanilla_position_risk",
        )?
        .get();
        point_buckets.fill(0.0);
        let mut point_stats = ReportingIvProjectionStats::default();
        basis.project_weight(maturity, x, risk, &mut point_buckets, &mut point_stats)?;
        for (sum, bucket) in sums.iter_mut().zip(&point_buckets) {
            sum.add(*bucket);
        }
        total.add(risk);
        stats.left_edge_count += point_stats.left_edge_count;
        stats.right_edge_count += point_stats.right_edge_count;
        left.add(point_stats.left_edge_sensitivity);
        right.add(point_stats.right_edge_sensitivity);
    }
    stats.left_edge_sensitivity = left.total();
    stats.right_edge_sensitivity = right.total();
    let raw: Vec<f64> = sums.into_iter().map(NeumaierSum::total).collect();
    let mut hedge_quantities = vec![0.0; raw.len()];
    let width = basis.log_moneyness_nodes().len();
    for (j, &x) in basis.log_moneyness_nodes().iter().enumerate() {
        let index = time_index * width + j;
        if raw[index] == 0.0 {
            continue;
        }
        let nodal_vega = PositiveF64::new(
            vanilla_vega(
                initial_spot,
                maturity,
                x,
                basis.implied_volatilities()[index],
            )?,
            "vega_kt_reporting_nodal_vega",
        )?
        .get();
        hedge_quantities[index] =
            FiniteF64::new(-raw[index] / nodal_vega, "vega_kt_vanilla_hedge_quantity")?.get();
    }
    Ok(VegaKtVanillaProjection {
        position_risk: vega_kt_projection_from_parts(raw, 0.0, total.total(), stats)?,
        hedge_quantities: hedge_quantities.into_boxed_slice(),
    })
}

fn vanilla_vega(spot: f64, maturity: f64, x: f64, sigma: f64) -> Result<f64, RiskConfigError> {
    let sd = PositiveF64::new(sigma * maturity.sqrt(), "vega_kt_vanilla_total_stddev")?.get();
    let d1 = -x / sd + 0.5 * sd;
    Ok(FiniteF64::new(
        spot * standard_normal_pdf(d1) * maturity.sqrt(),
        "vega_kt_vanilla_vega",
    )?
    .get())
}

impl VegaKtVanillaProjection {
    #[must_use]
    pub const fn position_risk(&self) -> &VegaKtProjection {
        &self.position_risk
    }

    #[must_use]
    pub fn hedge_quantities(&self) -> &[f64] {
        &self.hedge_quantities
    }
}
