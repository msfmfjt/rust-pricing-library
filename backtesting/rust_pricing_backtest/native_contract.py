"""Minimum native API consumed by each backtest provider path.

Check structure and call shapes without compiling or evaluating a trade. The
wheel's full API/ABI tests and per-request numerical gates remain separate.
"""

from importlib import import_module
import inspect

from .domain import BacktestError


_MISSING = object()


def _check_class(rp, name, errors, *, members=(), calls=()):
    cls = getattr(rp, name, None)
    if not isinstance(cls, type):
        errors.append(f"{name}: missing native class")
        return
    for member in members:
        if inspect.getattr_static(cls, member, _MISSING) is _MISSING:
            errors.append(f"{name}.{member}: missing member")
    # Positional counts include the receiver for unbound instance methods.
    # Binding accepts compatible added optional parameters; it executes nothing.
    for method, positional, keywords in calls:
        path = name if method is None else f"{name}.{method}"
        function = cls if method is None else getattr(cls, method, None)
        if not callable(function):
            errors.append(f"{path}: missing callable")
            continue
        try:
            inspect.signature(function).bind(*([None] * positional), **dict.fromkeys(keywords))
        except (TypeError, ValueError) as error:
            errors.append(f"{path}: incompatible call signature ({error})")


def validate_native_api(rp, *, multi_step=False, joint=False, quote_control=False):
    """Reject an incompatible build, even if its package version is unchanged."""
    errors = []
    error_type = getattr(rp, "PricingError", None)
    if not isinstance(error_type, type) or not issubclass(error_type, Exception):
        errors.append("PricingError: missing exception class")
    _check_class(rp, "PricingRequest", errors,
                 members=() if quote_control else ("fingerprint",),
                 calls=(("from_json", 1, ()),))

    if not quote_control:
        _check_class(rp, "PricingSpecification", errors,
                     calls=(("from_request", 1, ("worker_threads", "reduction_block_size")),))
        _check_class(rp, "PricingTask", errors,
                     calls=(("from_specification", 1, ()), ("compile", 1, ()),
                            ("from_json", 1, ()), ("to_json", 1, ())))
        _check_class(rp, "CompiledPricingTask", errors, calls=(("evaluate", 1, ()),))
        _check_class(rp, "PricingTaskResult", errors, members=("native_result", "report"))
        _check_class(rp, "SpecificationResult", errors, members=("native_result",))
        _check_class(rp, "ValuationReport", errors,
                     members=("value", "standard_error", "currency_id", "uncertainty_scope", "risks"),
                     calls=(("to_json", 1, ()),))
        _check_class(rp, "ReportedRisk", errors,
                     members=("name", "underlying_ids", "axes", "values", "unit", "calibration", "method"))
        _check_class(rp, "PricingResult", errors, members=("warnings", "diagnostics"))
        _check_class(rp, "PricingWarning", errors, members=("code",))
        _check_class(rp, "Diagnostics", errors, members=("gamma_spot_bump",))

    if not multi_step:
        _check_class(rp, "DiscountCurve", errors,
                     calls=((None, 0, ("curve_id", "times", "discount_factors")), ("discount", 2, ())))
        _check_class(rp, "ValuationReport", errors, members=("independent_sampling_units", "vega_kt"))
        _check_class(rp, "VegaKtResult", errors,
                     members=("raw_unit", "coordinates", "policy_label", "projection", "estimates", "raw_buckets"))
        _check_class(rp, "VegaKtCoordinate", errors, members=("maturity", "log_moneyness", "implied_volatility"))
        _check_class(rp, "VegaKtBucketEstimate", errors, members=("sample_variance",))
        _check_class(rp, "VegaKtProjection", errors, members=("reporting_stats", "signed_residual", "pre_projection"))
        _check_class(rp, "VegaKtReportingStats", errors, members=("left_edge_sensitivity", "right_edge_sensitivity"))

    if multi_step:
        keywords = ("variance_risk_directions", "subdivisions", "relative_tolerance",
                    "gamma_relative_standard_error_limit")
        calls = (("compile", 1, ("worker_threads", "reduction_block_size")),
                 ("evaluate_vega_kt_multi_step_gamma_chain_audit", 3, keywords))
        if joint or quote_control:
            calls += (("evaluate_vega_kt_multi_step_gamma_chain_audit", 3, (*keywords, "retain_samples")),
                      ("evaluate", 1, ()))
            _check_class(rp, "PricingResult", errors, members=("value", "delta_raw", "gamma_raw", "diagnostics"))
            _check_class(rp, "Diagnostics", errors, members=("gamma_method", "gamma_spot_bump"))
        _check_class(rp, "PricingPlan", errors, calls=calls)

    if errors:
        path = "quote-control" if quote_control else "multi-step" if multi_step else "ordinary"
        version = getattr(rp, "__version__", "unknown")
        raise BacktestError("unsupported_native_version",
                            f"rust_pricing {version} is incompatible with the {path} provider: "
                            + "; ".join(dict.fromkeys(errors))
                            + "; install/rebuild the native wheel from the same source revision as the companion package")


def load_native(**requirements):
    """Keep the optional dependency lazy, with a distinct import/ABI failure."""
    try:
        rp = import_module("rust_pricing")
    except (ImportError, OSError) as error:
        raise BacktestError("native_import_failure",
                            "cannot import rust_pricing; install/rebuild the native wheel for this companion "
                            f"package and Python ABI ({type(error).__name__}: {error})") from error
    validate_native_api(rp, **requirements)
    return rp
