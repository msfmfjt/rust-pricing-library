//! Compose the specified hat-density, equation-(11), and Gamma transition steps.
//! This component does not construct vanilla hedge weights or IV buckets.

use crate::core::{FiniteF64, PositiveF64};
use crate::risk::{
    ActiveDensityDomain, AnalyticCallDensityRow, EQUATION_11_FIRST_ORDER_LABEL,
    EQUATION_11_TRUNCATION_ORDER, Equation11GammaRow, RiskConfigError,
    equation_11_local_gamma_from_log_moneyness_vega,
    integrate_piecewise_linear_local_gamma_transition, mass_lumped_hat_areas,
};
use pricing_numerics::NeumaierSum;

/// Recover the active part of one positive-time Gamma row from integrated
/// local-volatility node adjoints (currency per unit absolute local volatility).
///
/// All slices share the density row's full log-moneyness grid. Hat areas are
/// formed before trimming the active domain; excluded adjoints are summed as
/// signed local Vega, never replaced with zero Gamma or assigned to IV buckets.
/// The density row cannot represent the singular initial-time distribution.
pub fn recover_equation11_gamma_row(
    density_row: &AnalyticCallDensityRow,
    integrated_local_vol_adjoints: &[f64],
    effective_local_variances: &[f64],
    delta_t: f64,
) -> Result<Equation11GammaRow, RiskConfigError> {
    let nodes = density_row.log_moneyness_nodes();
    for (quantity, actual) in [
        ("node adjoint", integrated_local_vol_adjoints.len()),
        ("local variance", effective_local_variances.len()),
    ] {
        if actual != nodes.len() {
            return Err(RiskConfigError::VegaKtRecoveryLengthMismatch {
                quantity,
                expected: nodes.len(),
                actual,
            });
        }
    }
    let delta_t = PositiveF64::new(delta_t, "vega_kt_recovery_delta_t")?;
    let next_time = density_row.maturity().get() + delta_t.get();
    if !next_time.is_finite() || next_time <= density_row.maturity().get() {
        return Err(RiskConfigError::InvalidVegaKtRecoveryInterval {
            maturity_bits: density_row.maturity().get().to_bits(),
            delta_t_bits: delta_t.get().to_bits(),
        });
    }
    let recovered_maturity = PositiveF64::new(next_time, "vega_kt_recovered_maturity")?;
    let areas = mass_lumped_hat_areas(nodes)?;
    let active_domain = density_row.active_domain();
    let mut strikes = Vec::with_capacity(active_domain.len());
    let mut gamma_values = Vec::with_capacity(active_domain.len());
    let mut input = NeumaierSum::new();
    let mut retained = NeumaierSum::new();
    let mut excluded = NeumaierSum::new();
    for index in 0..nodes.len() {
        let adjoint = FiniteF64::new(
            integrated_local_vol_adjoints[index],
            "vega_kt_recovery_node_adjoint",
        )?
        .get();
        let variance = PositiveF64::new(
            effective_local_variances[index],
            "vega_kt_recovery_local_variance",
        )?;
        input.add(adjoint);
        if !active_domain.contains_index(index) {
            excluded.add(adjoint);
            continue;
        }
        let strike = density_row.forward().get() * nodes[index].exp();
        let gamma = equation_11_local_gamma_from_log_moneyness_vega(
            adjoint / areas[index],
            strike,
            density_row.call_densities()[index],
            variance.get(),
            delta_t.get(),
        )?;
        strikes.push(strike);
        gamma_values.push(gamma.value());
        retained.add(adjoint);
    }
    Ok(Equation11GammaRow {
        source_maturity: density_row.maturity(),
        recovered_maturity,
        delta_t,
        strike_nodes: strikes.into_boxed_slice(),
        gamma_values: gamma_values.into_boxed_slice(),
        active_domain,
        input_local_vega: FiniteF64::new(input.total(), "vega_kt_recovery_input_sum")?.get(),
        retained_local_vega: FiniteF64::new(retained.total(), "vega_kt_recovery_retained_sum")?
            .get(),
        excluded_local_vega: FiniteF64::new(excluded.total(), "vega_kt_recovery_excluded_sum")?
            .get(),
        excluded_probability_mass: density_row.excluded_probability_mass(),
    })
}

impl Equation11GammaRow {
    #[must_use]
    pub const fn source_maturity(&self) -> PositiveF64 {
        self.source_maturity
    }

    #[must_use]
    pub const fn recovered_maturity(&self) -> PositiveF64 {
        self.recovered_maturity
    }

    #[must_use]
    pub const fn delta_t(&self) -> PositiveF64 {
        self.delta_t
    }

    /// Active nodes only, in the density row's source forward coordinate.
    #[must_use]
    pub fn strike_nodes(&self) -> &[f64] {
        &self.strike_nodes
    }

    #[must_use]
    pub fn gamma_values(&self) -> &[f64] {
        &self.gamma_values
    }

    /// Indices refer to the original, untrimmed log-moneyness grid.
    #[must_use]
    pub const fn active_domain(&self) -> ActiveDensityDomain {
        self.active_domain
    }

    #[must_use]
    pub const fn input_local_vega(&self) -> f64 {
        self.input_local_vega
    }

    #[must_use]
    pub const fn retained_local_vega(&self) -> f64 {
        self.retained_local_vega
    }

    #[must_use]
    pub const fn excluded_local_vega(&self) -> f64 {
        self.excluded_local_vega
    }

    #[must_use]
    pub const fn excluded_probability_mass(&self) -> f64 {
        self.excluded_probability_mass
    }

    #[must_use]
    pub const fn policy_label(&self) -> &'static str {
        EQUATION_11_FIRST_ORDER_LABEL
    }

    #[must_use]
    pub const fn truncation_order(&self) -> &'static str {
        EQUATION_11_TRUNCATION_ORDER
    }

    /// Apply the frozen-local-variance Gamma operator back over this row's own
    /// recovery interval. End Gamma is continued constantly outside the active
    /// strikes, as in the transition primitive. This extrapolation does not
    /// recover the excluded local Vega and does not certify an IV-risk mapping.
    /// A single active node is inspectable but cannot define this operator.
    pub fn propagate_to_source(
        &self,
        source_strike: f64,
        effective_local_variance: f64,
    ) -> Result<f64, RiskConfigError> {
        let variance = PositiveF64::new(
            effective_local_variance,
            "vega_kt_recovery_transition_local_variance",
        )?;
        integrate_piecewise_linear_local_gamma_transition(
            source_strike,
            variance.get() * self.delta_t.get(),
            &self.strike_nodes,
            &self.gamma_values,
        )
    }
}
