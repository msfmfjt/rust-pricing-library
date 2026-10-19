"""Checked European replay boundary for the conditional path-Gamma engine."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .domain import BacktestError, Risk, fingerprint, require
from .multi_step_full_smile import evaluate_multi_step_market_gamma


SCOPE = "multi_step_gamma"
POLICY = "conditional_gamma_triangular_market_iv_v1"
RELATIVE_TOLERANCE = 1e-6
RELATIVE_SE_LIMIT = .25
SUBDIVISIONS = 512


def gamma_boundaries(payload, spot, *, put_knots=False):
    boundaries=[spot*x for x in (.8,.9,1.,1.1,1.2)]
    p=payload['product']
    if put_knots and p.get('type')=='fixed_lookback' and p.get('side')=={'type':'put'}:
        # (K-min(H,S))^+ = (K-H)^+ + (min(K,H)-S)^+.
        # Include the known kink in the conditional recovery mesh. This adds a
        # reported bin too; it does not alter the payoff or any precision gate.
        known=[p[name] for name in ('historical_extremum','valuation_date_fixing') if p.get(name) is not None]
        strike=min(p['strike'],*known) if known else p['strike']
        require(math.isfinite(strike) and strike>0.,'extension_provider_scope','positive finite Put Lookback knot required')
        boundaries.append(strike)
    return sorted(set(boundaries))


def implementation_identity(extension=None, basis_policy=None):
    # A checkpoint binds the Python calibration and policy, as well as the native
    # binary already identified by RustPricingProvider. Upgrades cannot resume it.
    root = Path(__file__).parent
    names = ("multi_step_provider.py", "rust_adapter.py", "multi_step_full_smile.py",
             "multi_step_smile.py", "discrete_smile.py", "path_history.py", "domain.py")
    if extension is not None or basis_policy is not None:
        names += ("multi_step_extension.py",)
    if basis_policy is not None:
        names += ("variance_basis.py", "named_variance_basis.py")
    return fingerprint({name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names})


def validation(scope=SCOPE, extension=None, basis_policy=None, sampling_policy=None):
    if sampling_policy == "rqmc_quote_control_512x24":
        require(scope == "multi_step_extension_gamma" and extension == "all_rows" and basis_policy is None,
                "rqmc_provider_policy", "quote control requires the all_rows Call scope")
        accepted = validation(scope, extension)
        accepted.pop("minimum_sampling_units_by_intervals", None)
        return dict(accepted, status="runtime_checked_market_quote_control", sampling_policy=sampling_policy,
                    native_policy="conditional_gamma_fixed_market_quote_control_v1", intervals=[6],
                    products=["european_call"], points_per_scramble=512, scramble_count=24,
                    sampling_unit="independent_scramble_mean", brownian_bridge=False,
                    control="virtual_terminal_ATM_call_fixed_unit_coefficient", cross_instrument_covariance="paired_scramble_means",
                    independent_acceptance="limited_to_documented_panels_not_all_markets",
                    record="design/validation/vega-kt-quote-control-provider-v0.1.md")
    if sampling_policy is not None:
        require(sampling_policy in ("rqmc_256x16", "rqmc_path_16_scrambles") and
                ((scope in ("multi_step_extension_gamma", "multi_step_extension_put_gamma") and extension == "all_rows") or
                 (scope == "multi_step_custom_basis_gamma" and basis_policy == "coupled_row_hats")),
                "rqmc_provider_policy", "unsupported RQMC calibration policy")
        accepted = validation(scope, extension, basis_policy)
        accepted.pop("minimum_sampling_units_by_intervals", None)
        points = ({"points_per_scramble": 256} if sampling_policy == "rqmc_256x16" else
                  {"points_per_scramble_by_shape": {"default": 256, "six_interval_arithmetic_asian": 128}})
        return dict(accepted, status="runtime_checked_multi_step_rqmc", sampling_policy=sampling_policy,
                    native_policy=accepted["native_policy"]+":"+sampling_policy,
                    sampling_unit="independent_scramble_mean", **points, scramble_count=16,
                    brownian_bridge=False, antithetic_required=True,
                    independent_acceptance="limited_to_documented_panels_not_all_markets",
                    record="design/validation/vega-kt-rqmc-provider-v0.1.md")
    if scope == "multi_step_custom_basis_gamma":
        from .named_variance_basis import POLICIES
        require(basis_policy in POLICIES, "custom_basis_provider_policy", "unknown named basis")
        return dict(validation("multi_step_extension_put_gamma", "all_rows"),
                    status="numerically_validated_multi_step_named_basis_call_put", scope=scope,
                    native_policy="conditional_gamma_named_basis_market_iv_v1:"+basis_policy,
                    calibration_extension="custom", variance_basis_policy=basis_policy,
                    custom_basis="named_construction_only", arbitrary_variance_basis="not_accepted",
                    record=("design/validation/local-vol-coupled-basis-provider-v0.1.md" if basis_policy == "coupled_row_hats"
                            else "design/validation/local-vol-custom-basis-provider-v0.1.md"))
    if scope == "multi_step_extension_put_gamma":
        return dict(validation("multi_step_extension_gamma",extension),
                    status="numerically_validated_multi_step_extension_call_put",scope=scope,
                    products=["european_call","european_put","arithmetic_asian_call","arithmetic_asian_put",
                              "fixed_lookback_call","fixed_lookback_put"],
                    conditional_integration_knots="put_lookback_effective_fixed_strike",
                    record="design/validation/local-vol-extension-put-v0.1.md")
    if scope == "multi_step_extension_gamma":
        require(extension in ("all_rows", "linear_time"), "extension_provider_policy", "unknown extension")
        return dict(validation("multi_step_path_gamma"),
                    status="numerically_validated_multi_step_extension_call", scope=scope,
                    products=["european_call", "arithmetic_asian_call", "fixed_lookback_call"],
                    native_policy="conditional_gamma_all_transition_market_iv_v1:"+extension,
                    calibration_extension=extension, unreported_variance_rows="included_in_explicit_basis",
                    minimum_sampling_units_by_intervals={"4":4096, "6":2048},
                    antithetic_required=True, custom_basis="not_accepted",
                    record="design/validation/local-vol-extension-provider-v0.1.md")
    if scope == "multi_step_path_gamma":
        return dict(validation(), status="numerically_validated_multi_step_path", scope=scope,
                    products=["european_call", "european_put", "arithmetic_asian", "fixed_lookback"],
                    path_dependent_ledger="dated_fixings_and_observation_settlement",
                    lookback_valuation="fixed_observations_on_or_before_valuation_date",
                    record="design/validation/local-vol-lookback-fixing-day-v0.1.md")
    return dict(status="numerically_validated_multi_step_european", diagnostic_only=False,
                scope=SCOPE, native_policy=POLICY, intervals=[4, 6],
                products=["european_call", "european_put"],
                calibration_extension="last_interval_before_each_reporting_maturity",
                unreported_variance_rows="fixed", market_calibration_check="refined_price_error_at_most_1e-10",
                relative_tolerance=RELATIVE_TOLERANCE, gamma_relative_standard_error_limit=RELATIVE_SE_LIMIT,
                path_dependent_ledger="not_accepted", cross_instrument_covariance="not_established",
                full_market_iv_bucket_acceptance="not_established",
                record="design/validation/local-vol-multi-step-provider-v0.1.md")


def check_sampling(engine, intervals, extended, sampling_policy=None, product_type=None):
    """Keep MC eligibility unchanged; RQMC uses a distinct, explicit policy."""
    if sampling_policy is not None:
        reduction = engine.get("variance_reduction")
        controlled = sampling_policy == "rqmc_quote_control_512x24"
        points = 512 if controlled else 128 if sampling_policy == "rqmc_path_16_scrambles" and intervals == 6 and product_type == "arithmetic_asian" else 256
        scrambles = 24 if controlled else 16
        require(not controlled or (intervals == 6 and product_type == "european_vanilla"),
                "quote_control_scope", "quote control is limited to six-interval European calls")
        require(sampling_policy in ("rqmc_256x16", "rqmc_path_16_scrambles", "rqmc_quote_control_512x24") and extended
                and engine.get("type") == "randomized_quasi_monte_carlo"
                and type(engine.get("points_per_scramble")) is int and engine["points_per_scramble"] == points
                and type(engine.get("scramble_count")) is int and engine["scramble_count"] == scrambles
                and isinstance(reduction, dict) and reduction.get("antithetic") is True
                and reduction.get("brownian_bridge") is False,
                "rqmc_provider_scope", f"RQMC requires {points} points, {scrambles} scrambles, antithetic=true and brownian_bridge=false")
        return
    minimum_samples = (4096 if intervals == 4 else 2048) if extended else 4096
    require(engine.get("type") == "pseudo_monte_carlo"
            and type(engine.get("independent_sampling_units")) is int
            and engine["independent_sampling_units"] >= minimum_samples,
            "multi_step_provider_scope", f"at least {minimum_samples} independent pseudo-MC sampling units required")
    if extended:
        reduction = engine.get("variance_reduction")
        require(isinstance(reduction, dict) and reduction.get("antithetic") is True,
                "extension_provider_scope", "antithetic sampling required")


def evaluate(provider, payload, snapshot, contract):
    """Return Risk only after calibration, Gamma and native-AD checks all pass."""
    basis = snapshot.basis
    extension = provider.variance_extension
    basis_policy = provider.variance_basis_policy
    extended = extension is not None or basis_policy is not None
    if provider.numerical_scope == "multi_step_extension_gamma":
        require(contract.side == "call", "extension_provider_scope", "only Call-side products are accepted")
    require(contract.expiry == basis.maturity_dates[-1] and contract.payment_day == contract.expiry,
            "multi_step_provider_scope", "terminal expiry with same-day payment required")
    require(provider.reduction_block_size == 256, "multi_step_provider_scope",
            "the accepted Gamma chain uses reduction blocks of 256")
    try:
        grid = payload["model"]["local_variance_grid"]
        reporting = payload["model"]["reporting_iv_basis"]
        times = grid["time_nodes"]
        intervals = len(times)-1
        require(intervals in (4, 6) and len(grid["log_forward_moneyness_nodes"]) == 33
                and len(basis.log_moneyness) == 5 and len(basis.maturity_dates) == intervals//2,
                "multi_step_provider_scope", "four/six intervals, 33 variance nodes and five quotes per two-step maturity required")
        maturities = [(d-basis.valuation_date).days/365. for d in basis.maturity_dates]
        require(reporting["maturity_nodes"] == maturities == times[2::2]
                and reporting["log_forward_moneyness_nodes"] == list(basis.log_moneyness),
                "basis_mismatch", "model reporting times/nodes must equal the dated snapshot basis")
        engine = payload["engine"]
        check_sampling(engine, intervals, extended, provider.sampling_policy, payload["product"]["type"])
    except (KeyError, TypeError, IndexError) as error:
        raise BacktestError("invalid_pricing_request", "invalid multi-step market or sampling shape") from error

    rp = provider.rp
    try:
        if provider.sampling_policy == "rqmc_quote_control_512x24":
            from .quote_control import evaluate as evaluate_controlled
            return evaluate_controlled(provider, payload, snapshot, contract)
        original = rp.PricingRequest.from_json(json.dumps(payload, allow_nan=False))
        # Value and frozen-grid Delta retain their ordinary estimator. Removing
        # VegaKT avoids running the older diagnostic allocation as a side effect.
        ordinary = deepcopy(payload)
        ordinary["risk"].pop("vega_kt", None)
        ordinary["risk"]["vega"] = False
        result = rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(ordinary, allow_nan=False)),
            worker_threads=provider.worker_threads, reduction_block_size=256).evaluate()
        evaluator = evaluate_multi_step_market_gamma
        evaluator_args = {}
        construction = None
        if extended:
            from .multi_step_extension import evaluate_extended_market_gamma
            evaluator = evaluate_extended_market_gamma
            evaluator_args = {"extension": extension}
        if basis_policy is not None:
            from .named_variance_basis import named_variance_basis
            variance_basis, construction = named_variance_basis(ordinary, basis_policy)
            evaluator_args = {"extension": "custom", "variance_basis": variance_basis}
        market = evaluator(ordinary,
            gamma_boundaries(ordinary,snapshot.spot.mid,put_knots=provider.numerical_scope in
                             ('multi_step_extension_put_gamma','multi_step_custom_basis_gamma')), subdivisions=SUBDIVISIONS,
            relative_tolerance=RELATIVE_TOLERANCE, gamma_relative_standard_error_limit=RELATIVE_SE_LIMIT,
            worker_threads=provider.worker_threads, **evaluator_args)
    except BacktestError:
        raise
    except (ValueError, rp.PricingError) as error:
        raise BacktestError("native_pricing_failure", f"{contract.contract_id}: {error}") from error
    audit = market["gamma_audit"]
    require(market["status"] == "path_gamma_chain_passed" and market["estimate"] is not None
            and audit["recovery_passed"] and audit["validation_passed"],
            "multi_step_gamma_acceptance_failure", f"{contract.contract_id}: {market['status']}; no hedge risk released")
    require(result.delta_raw is not None and math.isclose(result.value, audit["native_price"]["mean"], rel_tol=1e-12, abs_tol=1e-12),
            "multi_step_price_mismatch", "Gamma audit must use the ordinary pricing distribution")
    expected = [dict(maturity=t, log_moneyness=k) for t in maturities for k in basis.log_moneyness]
    require(market["coordinate_order"] == expected and len(market["estimate"]) == basis.size,
            "basis_mismatch", "calibrated Gamma coordinate order differs from the snapshot")
    require(market["market_mean_covariance"] is not None and market["market_sample_covariance"] is not None,
            "missing_risk_covariance", "accepted Gamma must retain its same-product covariance")
    if construction is not None:
        require(market["extension_audit"]["extension"] == "custom"
                and market["extension_audit"]["basis_hash"] == construction["basis_hash"]
                and market["tangent_audit"]["basis_hash"] == construction["basis_hash"],
                "custom_basis_mismatch", "Gamma and calibration must use the declared basis")
    buckets = tuple(x["mean"] for x in market["estimate"])
    errors = tuple(x["standard_error"] for x in market["estimate"])
    provenance = dict(original_request=original.fingerprint, provider=provider.identity,
                      snapshot=fingerprint(snapshot), contract=fingerprint(contract), market_gamma=fingerprint(market))
    accepted = validation(provider.numerical_scope, extension, basis_policy, provider.sampling_policy)
    value = Risk(result.value, result.delta_raw, buckets, basis.basis_id,
                 pre_projection=math.fsum(buckets), price_standard_error=result.standard_error,
                 provenance=fingerprint(provenance), bucket_standard_errors=errors,
                 warning_codes=(provider.numerical_scope, "variance_basis_"+basis_policy if basis_policy is not None else
                                "variance_extension_"+extension if extension is not None else "fixed_unreported_variance_rows", "cross_instrument_covariance_not_established",
                                *(w.code for w in result.warnings)),
                 native_policy=accepted["native_policy"], payment_date=contract.payment_day,
                 gamma=result.gamma_raw, gamma_method=result.diagnostics.gamma_method if result.gamma_raw is not None else None,
                 gamma_spot_bump=result.diagnostics.gamma_spot_bump if result.gamma_raw is not None else None,
                 reporting_iv=tuple(reporting["implied_volatilities"]),
                 reporting_forwards=tuple(snapshot.spot.mid for _ in maturities))
    return value, dict(document_kind="multi_step_provider_audit", schema_version=1, status="passed",
                       provenance=provenance, risk_provenance=value.provenance,
                       validation=accepted, market_gamma=market,
                       covariance_scope="same_contract_same_sampling_unit_only",
                       **({"variance_basis_policy": basis_policy, "basis_construction": construction} if construction is not None else {}))
