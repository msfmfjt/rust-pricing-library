"""Adapter to immutable public Rust pricing requests; no private engine access."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from .domain import BacktestError, Risk, data, fingerprint, require
from .provider_cache import SnapshotCache
from . import task_adapter

EXTENSION_SCOPES = ("multi_step_extension_gamma", "multi_step_extension_put_gamma")
CUSTOM_BASIS_SCOPE = "multi_step_custom_basis_gamma"
MULTI_STEP_SCOPES = ("multi_step_gamma", "multi_step_path_gamma", *EXTENSION_SCOPES, CUSTOM_BASIS_SCOPE)


@dataclass(frozen=True, slots=True)
class _ProviderConfiguration:
    """One lifetime binding for acceptance, execution and provenance."""

    native: object
    identity: str
    diagnostic_only: bool
    numerical_scope: str | None
    variance_extension: str | None
    variance_basis_policy: str | None
    sampling_policy: str | None
    worker_threads: int
    reduction_block_size: int
    cache_max_entries: int


class RustPricingProvider:
    """One canonical pricing_request JSON per live contract per snapshot.

    The caller supplies calibrated Local Volatility grids and a shared reporting
    basis. The first slice supports non-dividend European options, the existing
    frozen-local-variance-grid Delta, and explicit numerical or diagnostic modes.
    Explicit numerical_scope="p0_terminal_replication" accepts only the tested
    terminal methods and checks calibration against market quotes per contract.
    numerical_scope="multi_step_gamma" uses the accepted conditional Gamma chain
    and a checked triangular market calibration on four/six intervals. The
    contract ledger remains European-only. The explicit path scopes admit dated
    observations; multi_step_extension_gamma requires an all_rows or linear_time
    variance_extension and accepts only Call-side payoffs. The explicit
    multi_step_extension_put_gamma scope also accepts Put-side payoffs and Call hedges.
    multi_step_custom_basis_gamma requires variance_basis_policy="row_hats" or
    "weighted_time_hats" or "coupled_row_hats"; arbitrary raw matrices remain diagnostic-only.
    Other uses require diagnostic_only=True.
    Quantity and contract multiplier are applied by the ledger.
    Settings are read-only for this provider's lifetime. Construct a new provider
    to change acceptance or execution settings; clear_cache only releases work.
    """

    def __init__(self, *, diagnostic_only=False, worker_threads=1, reduction_block_size=256,
                 numerical_scope=None, variance_extension=None, variance_basis_policy=None, sampling_policy=None,
                 cache_max_entries=128):
        if hasattr(self, "_configuration"):
            raise AttributeError("provider configuration is read-only; create a new RustPricingProvider")
        require(type(cache_max_entries) is int and cache_max_entries > 0, "invalid_provider_cache",
                "cache_max_entries must be a positive integer")
        require(numerical_scope in (None, "p0_terminal_replication", *MULTI_STEP_SCOPES), "unsupported_numerical_scope",
                "choose an explicitly accepted numerical scope")
        require(variance_extension in ("all_rows", "linear_time") if numerical_scope in EXTENSION_SCOPES
                else variance_extension is None, "extension_provider_policy",
                "the extension scope requires an explicit all_rows or linear_time policy; other scopes do not accept one")
        from .named_variance_basis import POLICIES
        require(variance_basis_policy in POLICIES if numerical_scope == CUSTOM_BASIS_SCOPE
                else variance_basis_policy is None, "custom_basis_provider_policy",
                "the custom-basis scope requires a named accepted variance basis; other scopes do not accept one")
        require(sampling_policy is None or (sampling_policy == "rqmc_quote_control_512x24" and
                numerical_scope == "multi_step_extension_gamma" and variance_extension == "all_rows") or
                (sampling_policy in ("rqmc_256x16", "rqmc_path_16_scrambles") and
                ((numerical_scope in EXTENSION_SCOPES and variance_extension == "all_rows") or
                 (numerical_scope == CUSTOM_BASIS_SCOPE and variance_basis_policy == "coupled_row_hats"))),
                "rqmc_provider_policy", "RQMC requires an explicit sampling policy with all_rows or coupled_row_hats")
        require(not (diagnostic_only and numerical_scope is not None), "conflicting_numerical_modes",
                "choose either an accepted scope or diagnostic mode")
        require(diagnostic_only is True or numerical_scope is not None, "unvalidated_vega_kt",
                "native LV paper decomposition has not passed independent acceptance; "
                "diagnostic_only=True is required for investigation")
        import rust_pricing as rp
        require(worker_threads >= 1 and reduction_block_size >= 1, "invalid_provider", "positive execution settings required")
        module = getattr(rp, "rust_pricing", rp)
        binary_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        identity = f"rust_pricing_lv_v5:{numerical_scope}:{binary_hash}:{worker_threads}:{reduction_block_size}"
        if numerical_scope in MULTI_STEP_SCOPES:
            from .multi_step_provider import implementation_identity
            identity += ":" + implementation_identity(variance_extension, variance_basis_policy)
        if variance_extension is not None:
            identity += ":extension=" + variance_extension
        if variance_basis_policy is not None:
            identity += ":variance_basis=" + variance_basis_policy
        if sampling_policy is not None:
            identity += ":sampling=" + sampling_policy
        if sampling_policy == "rqmc_quote_control_512x24":
            from .portfolio_market_risk import implementation_identity as portfolio_identity
            identity += ":quote_control=" + portfolio_identity()
        self._configuration = _ProviderConfiguration(
            native=rp, identity=identity, diagnostic_only=diagnostic_only, numerical_scope=numerical_scope,
            variance_extension=variance_extension, variance_basis_policy=variance_basis_policy,
            sampling_policy=sampling_policy, worker_threads=worker_threads,
            reduction_block_size=reduction_block_size, cache_max_entries=cache_max_entries)
        self._cache_lock = RLock()
        self._cache = SnapshotCache(None, cache_max_entries)

    def clear_cache(self):
        """Release retained risks, task reports, audits and samples."""
        with self._cache_lock:
            self._cache = SnapshotCache(None, self.cache_max_entries)

    def _snapshot_cache(self, snapshot_hash):
        with self._cache_lock:
            if self._cache.snapshot_hash != snapshot_hash:
                self._cache = SnapshotCache(snapshot_hash, self.cache_max_entries)
            return self._cache

    @property
    def _quote_control_cache(self):
        # Retained for the existing read-only pilot/continuation audit exporters.
        return self._cache.native

    @property
    def identity(self):
        return self._configuration.identity

    @property
    def rp(self):
        return self._configuration.native

    @property
    def diagnostic_only(self):
        return self._configuration.diagnostic_only

    @property
    def numerical_scope(self):
        return self._configuration.numerical_scope

    @property
    def worker_threads(self):
        return self._configuration.worker_threads

    @property
    def reduction_block_size(self):
        return self._configuration.reduction_block_size

    @property
    def cache_max_entries(self):
        return self._configuration.cache_max_entries

    @property
    def variance_extension(self):
        return self._configuration.variance_extension

    @property
    def variance_basis_policy(self):
        return self._configuration.variance_basis_policy

    @property
    def sampling_policy(self):
        return self._configuration.sampling_policy

    @property
    def validation(self):
        if self.numerical_scope in MULTI_STEP_SCOPES:
            from .multi_step_provider import validation
            return validation(self.numerical_scope, self.variance_extension, self.variance_basis_policy, self.sampling_policy)
        if self.numerical_scope == "p0_terminal_replication":
            return {"status": "numerically_validated_p0", "diagnostic_only": False,
                    "scope": self.numerical_scope, "market_calibration_check": "6_price_se_plus_0.002_per_unit",
                    "path_dependent_gamma_recovery": "not_accepted",
                    "record": "design/validation/vega-kt-smile-acceptance-v0.3.md"}
        return {"status": "unvalidated", "diagnostic_only": True,
                "blocker": "lv_paper_gamma_pipeline_incomplete",
                "record": "design/validation/vega-kt-backtest-numerics-v0.1.md"}

    def gamma_audit(self, snapshot, contract):
        """Return a detached native/calibration/covariance audit for this risk."""
        from copy import deepcopy
        require(self.numerical_scope in MULTI_STEP_SCOPES, "unsupported_numerical_scope",
                "Gamma audits require the multi-step scope")
        return deepcopy(self._evaluate(snapshot, contract).audit)

    def pricing_evidence(self, snapshot, contract) -> task_adapter.PricingEvidence | None:
        """Return immutable task/report evidence after normal risk acceptance.

        Quote-control risk uses its dedicated paired-sample report and returns
        None here. No substitute task is compiled solely to manufacture evidence.
        """
        return self._evaluate(snapshot, contract).pricing_evidence

    def joint_market_risk(self, snapshot, contracts, risks):
        """Joint estimation covariance for the same accepted decision risks."""
        from .portfolio_market_risk import evaluate_portfolio
        from .multi_step_provider import SUBDIVISIONS, RELATIVE_TOLERANCE, RELATIVE_SE_LIMIT
        require(self.numerical_scope in (*EXTENSION_SCOPES, CUSTOM_BASIS_SCOPE),
                "joint_provider_scope", "joint replay requires an explicit extension or named basis scope")
        requests = {cid: json.loads(snapshot.pricing_requests[cid]) for cid in sorted(risks)}
        for cid in requests:
            require(self.evaluate(snapshot, contracts[cid]) == risks[cid],
                    "joint_risk_mismatch", "joint inputs differ from evaluated decision risk")
        extension, variance_basis = self.variance_extension, None
        if self.variance_basis_policy is not None:
            from .named_variance_basis import named_variance_basis
            variance_basis, _ = named_variance_basis(requests[min(requests)], self.variance_basis_policy)
            extension = "custom"
        try:
            if self.sampling_policy == "rqmc_quote_control_512x24":
                from .quote_control import portfolio
                return portfolio(self, snapshot, requests)
            return evaluate_portfolio(requests, {cid: 0. for cid in requests}, extension=extension,
                variance_basis=variance_basis, subdivisions=SUBDIVISIONS, relative_tolerance=RELATIVE_TOLERANCE,
                gamma_relative_standard_error_limit=RELATIVE_SE_LIMIT, worker_threads=self.worker_threads)
        except (ValueError, self.rp.PricingError) as error:
            raise BacktestError("joint_native_failure", str(error)) from error

    def evaluate(self, snapshot, contract):
        return self._evaluate(snapshot, contract).risk

    def _evaluate(self, snapshot, contract):
        basis = snapshot.basis
        require(basis.method == "lv_paper_decomposition" and basis.coordinate == "canonical_f_iv"
                and basis.delta_convention == "frozen_local_variance_grid",
                "unsupported_risk_method", "LV paper basis and frozen-grid Delta required")
        require(snapshot.surface_available_at <= snapshot.at, "future_surface", snapshot.snapshot_id)
        require(contract.contract_id in snapshot.pricing_requests, "missing_pricing_request", contract.contract_id)
        raw = snapshot.pricing_requests[contract.contract_id]
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise BacktestError("invalid_pricing_request", f"{contract.contract_id}: {error}") from error
        require(isinstance(payload, dict), "invalid_pricing_request", "request must be an object")
        p, m, model, risk = [payload.get(key, {}) for key in ("product", "market", "model", "risk")]
        require(all(isinstance(v, dict) for v in (p, m, model, risk)),
                "invalid_pricing_request", "product, market, model and risk must be objects")
        require(payload.get("valuation_date") == snapshot.at.date().isoformat(), "request_mismatch", "valuation date")
        from .domain import PathContract
        if isinstance(contract, PathContract):
            from .path_history import native_product
            require(self.numerical_scope in ("multi_step_path_gamma", *EXTENSION_SCOPES, CUSTOM_BASIS_SCOPE), "unsupported_path_scope",
                    "path contracts require the explicit path Gamma scope")
            require(p == native_product(contract,snapshot), "path_request_mismatch",
                    "native payoff differs from the contract schedule or available fixings")
        else:
            require(p.get("type") == "european_vanilla" and p.get("expiry") == contract.expiry.isoformat()
                    and p.get("strike") == contract.strike and p.get("notional") == 1.0
                    and p.get("side") == {"type": contract.side}
                    and p.get("underlying_id") == contract.underlying_id and p.get("currency_id") == contract.currency_id,
                    "request_mismatch", "unit contract differs from the instrument master")
        require(m.get("spot") == snapshot.spot.mid and m.get("underlying_id") == contract.underlying_id
                and m.get("currency_id") == contract.currency_id, "request_mismatch", "market/spot")
        require(model.get("type") == "local_volatility", "unsupported_model", "Local Volatility required")
        require(not m.get("discrete_dividends") and snapshot.dividend_per_share == 0
                and all(x == 1.0 for x in m.get("dividend_curve", {}).get("discount_factors", [])),
                "unsupported_dividends", "the native adapter slice requires zero dividends")
        require(risk.get("delta") is True and risk.get("smile_dynamics") == {"type": "sticky_log_moneyness"},
                "unsupported_delta", "existing frozen-grid bump with sticky_log_moneyness label required")
        request_basis = risk.get("vega_kt", {})
        require(request_basis.get("maturity_nodes") == [d.isoformat() for d in basis.maturity_dates]
                and request_basis.get("log_forward_moneyness_nodes") == list(basis.log_moneyness),
                "basis_mismatch", "request risk coordinates differ")
        require("reporting_iv_basis" in model, "missing_basis", "model lacks reporting basis")
        reporting = model["reporting_iv_basis"]
        grid = model["local_variance_grid"]
        require(self.numerical_scope is not None or (grid["log_forward_moneyness_nodes"] == reporting["log_forward_moneyness_nodes"]
                and all(t in reporting["maturity_nodes"] for t in grid["time_nodes"] if t > 0)),
                "unsupported_grid", "diagnostic LV allocation requires matching local/reporting grids")
        shared = fingerprint({"market": m, "reporting_iv_basis": model["reporting_iv_basis"],
                              "density_threshold": request_basis.get("relative_density_threshold"),
                              **({"model": model} if self.numerical_scope in MULTI_STEP_SCOPES else {})})
        snapshot_hash = fingerprint(snapshot)
        cache = self._snapshot_cache(snapshot_hash)
        cache.check_surface(shared)
        key = fingerprint({"snapshot": snapshot_hash, "contract": contract, "request": payload})
        cached = cache.results.get(key)
        if cached is not None:
            return cached
        if self.numerical_scope in MULTI_STEP_SCOPES:
            from .multi_step_provider import evaluate
            value, audit, evidence = evaluate(self, payload, snapshot, contract, _cache=cache)
            return cache.store(key, shared, value, audit, evidence)
        payment_scale = 1.0
        payment_provenance = None
        try:
            request = self.rp.PricingRequest.from_json(raw)
            curve = self.rp.DiscountCurve(**m["discount_curve"])
            require(hasattr(curve, "discount"), "unsupported_native_version", "rebuild the native library for reporting forwards")
            # This adapter rejects dividends. F(T)=spot/D(T) uses the same public
            # curve as the native forward normalizer, at absolute basis dates.
            discounts = tuple(curve.discount((d-basis.valuation_date).days/365.) for d in basis.maturity_dates)
            require(all(math.isfinite(x) and x > 0 for x in discounts),
                    "invalid_reporting_surface", "positive finite discounts required for forward reconstruction")
            forwards = tuple(snapshot.spot.mid / x for x in discounts)
            if contract.payment_day > contract.expiry:
                discount = snapshot.payment_discounts.get(contract.payment_day)
                require(discount is not None, "missing_payment_discount", f"{contract.contract_id}: {contract.payment_day}")
                discount.check(snapshot.at, contract.payment_day, math.inf)
                expiry_df = curve.discount((contract.expiry - basis.valuation_date).days / 365.)
                payment_df = curve.discount((contract.payment_day - basis.valuation_date).days / 365.)
                require(math.isclose(payment_df, discount.factor, rel_tol=1e-12, abs_tol=1e-14),
                        "payment_discount_mismatch", f"{contract.contract_id}: input factor differs from native valuation curve")
                payment_scale = payment_df / expiry_df
                payment_provenance = {"request": request.fingerprint, "payment_date": contract.payment_day,
                                      "expiry_discount": expiry_df, "payment_discount": data(discount),
                                      "policy": "deterministic_curve_payment_ratio_v1"}
            valuation = task_adapter.evaluate_request(self.rp, request, worker_threads=self.worker_threads,
                                                       reduction_block_size=self.reduction_block_size)
        except BacktestError:
            raise
        except (ValueError, self.rp.PricingError) as error:
            raise BacktestError("native_pricing_failure", f"{contract.contract_id}: {error}") from error
        report, result = valuation.report, valuation.native
        delta, gamma = task_adapter.ordinary_greeks(report, contract)
        kt = report.vega_kt
        require(kt is not None, "missing_risk", contract.contract_id)
        require(kt.raw_unit == basis.unit, "unit_mismatch", kt.raw_unit)
        expected = [((d - basis.valuation_date).days / 365.0, k) for d in basis.maturity_dates for k in basis.log_moneyness]
        actual = [(x.maturity, x.log_moneyness) for x in kt.coordinates]
        require(len(actual) == len(expected) and all(math.isclose(a, b, rel_tol=1e-13, abs_tol=1e-13)
                for pair1, pair2 in zip(actual, expected) for a, b in zip(pair1, pair2)), "basis_mismatch", "result coordinates differ")
        if self.numerical_scope is not None:
            require(kt.policy_label in ("flat_variance_terminal_measure_pathwise_v1", "terminal_payoff_replication_iv_v1"),
                    "unaccepted_native_method", kt.policy_label)
            require(contract.expiry in basis.maturity_dates, "unaccepted_expiry", "P0 requires an exact reporting expiry")
            # A correct static hedge does not certify calibration of an arbitrary
            # supplied LV grid. Test its MC price independently against the market
            # quote price on every live contract before permitting a decision.
            import numpy as np
            from .numerics import black_scholes_call
            row = basis.maturity_dates.index(contract.expiry)
            width = len(basis.log_moneyness)
            expiry_df = discounts[row]
            t = (contract.expiry-basis.valuation_date).days/365.
            sigma = float(np.interp(math.log(contract.strike/forwards[row]), basis.log_moneyness,
                                   reporting['implied_volatilities'][row*width:(row+1)*width]))
            call = black_scholes_call(snapshot.spot.mid, contract.strike, t, sigma, -math.log(expiry_df)/t)['price']
            reference = call if contract.side == 'call' else call-snapshot.spot.mid+contract.strike*expiry_df
            require(abs(report.value-reference) <= 6*report.standard_error+0.002,
                    "native_market_calibration_failure",
                    f"{contract.contract_id}: model price {report.value} vs market replication {reference}")
        stats = kt.projection.reporting_stats
        variances = [b.sample_variance for b in kt.estimates]
        errors = tuple(math.sqrt(v / report.independent_sampling_units) for v in variances) if all(v is not None for v in variances) else None
        value = Risk(report.value * payment_scale, delta.values[0] * payment_scale,
                     tuple(x * payment_scale for x in kt.raw_buckets), basis.basis_id,
                     kt.projection.signed_residual * payment_scale, kt.projection.pre_projection * payment_scale,
                     report.standard_error * payment_scale,
                     (abs(stats.left_edge_sensitivity) + abs(stats.right_edge_sensitivity)) * payment_scale,
                     fingerprint(payment_provenance) if payment_provenance else request.fingerprint,
                     tuple(x * payment_scale for x in errors) if errors is not None else None,
                     (("p0_terminal_replication" if self.numerical_scope else "unvalidated_vega_kt"),
                      *(w.code for w in result.warnings)),
                     native_policy=kt.policy_label, payment_date=contract.payment_day, payment_scaling_factor=payment_scale,
                     gamma=gamma.values[0] * payment_scale if gamma is not None else None,
                     gamma_method=gamma.method if gamma is not None else None,
                     gamma_spot_bump=result.diagnostics.gamma_spot_bump if gamma is not None else None,
                     reporting_iv=tuple(x.implied_volatility for x in kt.coordinates), reporting_forwards=forwards)
        return cache.store(key, shared, value, pricing_evidence=valuation.evidence(contract, payment_scale=payment_scale))
