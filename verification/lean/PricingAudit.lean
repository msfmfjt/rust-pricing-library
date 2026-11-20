import PricingAudit.Barrier
import PricingAudit.Binary64
import PricingAudit.ControlFlow
import PricingAudit.Discrete
import PricingAudit.Dividends
import PricingAudit.Dynamics
import PricingAudit.Evidence
import PricingAudit.Graph
import PricingAudit.Market
import PricingAudit.NumericalRepair
import PricingAudit.Numerics
import PricingAudit.Projection
import PricingAudit.Repair
import PricingAudit.Rough
import PricingAudit.Sampling
import PricingAudit.Scenarios
import PricingAudit.Schema
import PricingAudit.TailBounds
import PricingAudit.Witness
import Lean.Util.CollectAxioms

open Lean

/- Check every named contract, including the axioms of its transitive dependencies. -/
run_cmd
  let results := #[
    -- Barrier
    ``PricingAudit.quintic_complement,
    ``PricingAudit.quintic_range,
    ``PricingAudit.safety_pos_inside,
    ``PricingAudit.indicator_range,
    ``PricingAudit.indicator_complement,
    ``PricingAudit.hitStep_bounds,
    ``PricingAudit.historical_hit_absorbing,
    ``PricingAudit.hitFold_bounds,
    ``PricingAudit.smoothed_path_bounds,
    ``PricingAudit.survival_product,
    ``PricingAudit.rebate_parity,
    -- Binary64
    ``PricingAudit.Binary64.fields_recover_bits,
    ``PricingAudit.Binary64.zero_significand,
    ``PricingAudit.Binary64.numeric_wrapper_guards,
    -- ControlFlow
    ``PricingAudit.ControlFlow.row_from_local,
    ``PricingAudit.ControlFlow.reachability_iff,
    ``PricingAudit.ControlFlow.zero_iff_no_path,
    -- Discrete
    ``PricingAudit.Discrete.gregorian_period,
    ``PricingAudit.Discrete.gregorian_next_year,
    ``PricingAudit.Discrete.weekday_period,
    ``PricingAudit.Discrete.full_calendar_size,
    ``PricingAudit.Discrete.actual_daycount_additive,
    ``PricingAudit.Discrete.checked_id_success,
    ``PricingAudit.Discrete.checked_id_exhaustion,
    ``PricingAudit.Discrete.midpoint_has_exact_short_significand,
    ``PricingAudit.Discrete.max_normal_is_representable,
    ``PricingAudit.Discrete.complex_numerator_exceeds_binary64,
    -- Dividends
    ``PricingAudit.Dividends.initial_scale_positive,
    ``PricingAudit.Dividends.scale_stays_positive,
    ``PricingAudit.Dividends.jump_commutes,
    ``PricingAudit.Dividends.accumulate_spec,
    ``PricingAudit.Dividends.remaining_nonnegative,
    ``PricingAudit.Dividends.zero_cash_suffix,
    ``PricingAudit.Dividends.funded_step,
    ``PricingAudit.Dividends.funded_post_positive,
    ``PricingAudit.Dividends.funded_sweep_zero,
    ``PricingAudit.Dividends.initial_reserve_settles,
    -- Dynamics
    ``PricingAudit.Dynamics.hull_white_loading_composes,
    ``PricingAudit.Dynamics.hull_white_positive_loading,
    ``PricingAudit.Dynamics.ou_loadings,
    ``PricingAudit.Dynamics.lsv_local_variance,
    ``PricingAudit.Dynamics.bridge_survival_range,
    ``PricingAudit.Dynamics.bridge_hit_increases_with_variance,
    ``PricingAudit.Dynamics.log_survival_product,
    ``PricingAudit.Dynamics.whitened_covariance,
    ``PricingAudit.Dynamics.gamma_quadratic,
    ``PricingAudit.Dynamics.gamma_quartic_bias,
    ``PricingAudit.Dynamics.reverse_product_and_quotient,
    ``PricingAudit.Dynamics.exercise_envelope,
    ``PricingAudit.Dynamics.pnl_reconciliation,
    ``PricingAudit.Dynamics.greek_pnl_exact_quadratic,
    -- Evidence
    ``PricingAudit.Evidence.gate_requires_both_bumps,
    -- Graph
    ``PricingAudit.Graph.tangent_hasDerivAt,
    ``PricingAudit.Graph.pullback_tangent,
    ``PricingAudit.Graph.reverse_is_derivative,
    -- Market
    ``PricingAudit.Market.forward_carry,
    ``PricingAudit.Market.intrinsic_parity,
    ``PricingAudit.Market.black_price_and_delta_parity,
    ``PricingAudit.Market.black_vega_gamma,
    ``PricingAudit.Market.black_atm_time_value_positive,
    ``PricingAudit.Market.pchip_interpolates_knots,
    ``PricingAudit.Market.pchip_reproduces_affine,
    ``PricingAudit.Market.durrleman_flat_slice,
    ``PricingAudit.Market.ssvi_atm_variance,
    ``PricingAudit.Market.bergomi_weight_variance,
    ``PricingAudit.Market.bergomi_normalized,
    ``PricingAudit.Market.bergomi_degeneracy,
    -- NumericalRepair
    ``PricingAudit.NumericalRepair.vega_factorization,
    ``PricingAudit.NumericalRepair.black_vega_derivative,
    ``PricingAudit.NumericalRepair.tiny_tail_below_half_subnormal,
    ``PricingAudit.NumericalRepair.mills_divided_step,
    ``PricingAudit.NumericalRepair.exp_quadratic_remainder,
    ``PricingAudit.NumericalRepair.black_integrand_relative_error,
    ``PricingAudit.NumericalRepair.atm_density_bounds,
    ``PricingAudit.NumericalRepair.atm_witness_rounding_cell,
    ``PricingAudit.NumericalRepair.align_dot,
    ``PricingAudit.NumericalRepair.restore_quotient,
    ``PricingAudit.NumericalRepair.quotient_residual,
    ``PricingAudit.NumericalRepair.repaired_max_division,
    -- Numerics
    ``PricingAudit.Numerics.welford_push,
    ``PricingAudit.Numerics.welford_fold,
    ``PricingAudit.Numerics.chan_conserves_raw_moments,
    ``PricingAudit.Numerics.chan_between_nonnegative,
    ``PricingAudit.Numerics.covariance_merge,
    ``PricingAudit.Numerics.compensated_residual,
    ``PricingAudit.Numerics.simpson_cubic,
    ``PricingAudit.Numerics.richardson_cancels_leading_error,
    ``PricingAudit.Numerics.complex_division_reconstructs,
    ``PricingAudit.Numerics.complex_norm_product,
    ``PricingAudit.Numerics.equal_components_div_one_i,
    ``PricingAudit.Numerics.correlation_pivot,
    ``PricingAudit.Numerics.three_driver_determinant,
    ``PricingAudit.Numerics.least_squares_normal_direction,
    -- Projection
    ``PricingAudit.Projection.returns_field,
    -- Repair
    ``PricingAudit.Repair.quintic_hasDerivAt,
    ``PricingAudit.Repair.zero_vol_up_out_delta,
    -- Rough
    ``PricingAudit.Rough.truncation_nonnegative,
    ``PricingAudit.Rough.lifted_nonexpansive,
    ``PricingAudit.Rough.quadratic_floor,
    ``PricingAudit.Rough.quadratic_derivative,
    ``PricingAudit.Rough.log_component_formula,
    ``PricingAudit.Rough.mixed_nonnegative,
    ``PricingAudit.Rough.mixed_flat_vol_of_vol,
    ``PricingAudit.Rough.scalar_variance_positive,
    -- Sampling
    ``PricingAudit.Sampling.philox_round_inverse,
    ``PricingAudit.Sampling.philox_rounds_injective,
    ``PricingAudit.Sampling.philox_known_vector,
    ``PricingAudit.Sampling.address_injective,
    ``PricingAudit.Sampling.midpoint_in_unit_interval,
    ``PricingAudit.Sampling.bridge_weights,
    ``PricingAudit.Sampling.bridge_reverse_duality,
    ``PricingAudit.Sampling.antithetic_linear_cancellation,
    -- Scenarios
    ``PricingAudit.Scenarios.price_monotone,
    ``PricingAudit.Scenarios.price_linear,
    ``PricingAudit.Scenarios.mean_square_error,
    ``PricingAudit.Scenarios.tail_mass_bound,
    ``PricingAudit.Scenarios.dual_gap_grows,
    ``PricingAudit.Scenarios.dual_upper_above_policy,
    -- Schema
    ``PricingAudit.Schema.tag_determines_branch,
    ``PricingAudit.Schema.lookup_exists_of_check,
    -- TailBounds
    ``PricingAudit.Market.black_atm_lower_bound,
    ``PricingAudit.Market.black_witness_is_normal_scale,
    -- Witness
    ``PricingAudit.Witness.exact_safety,
    ``PricingAudit.Witness.rounded_hit,
    ``PricingAudit.Witness.rounded_safety,
    ``PricingAudit.Witness.complementary_evaluation,
    ``PricingAudit.Witness.repaired_weight_pair,
    ``PricingAudit.Witness.repaired_safety_error,
    ``PricingAudit.Witness.dividend_routes_disagree]
  let allowed := #[``propext, ``Classical.choice, ``Quot.sound]
  for name in results do
    let unexpected := (← collectAxioms name).filter fun ax => !allowed.contains ax
    unless unexpected.isEmpty do
      throwError "{name} uses unexpected axioms {unexpected}"
  logInfo m!"Checked axiom dependencies of {results.size} named theorems."
