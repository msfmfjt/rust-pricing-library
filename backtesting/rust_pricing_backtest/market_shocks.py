"""Fix market quote shocks independently of a nested numerical reporting grid.

H maps market IV shocks to reporting IV shocks. Risk is a covector: H.T r,
not interpolated bucket values. No time interpolation or nonnested loss is
silently accepted. The orthogonal complement is an audit in the Euclidean
reporting-quote metric, not an estimated market covariance or probability.
"""

from dataclasses import dataclass, field

import numpy as np

from .domain import BacktestError, RiskBasis, fingerprint, require
from .optimizer import HedgeSolution, solve_hedge


REPLAY_POLICY = {
    "method": "nested_log_iv_market_shocks_v1",
    "strike_policy": "configured_log_forward_moneyness_fixed_for_run",
    "maturity_policy": "exact_current_snapshot_reporting_maturities",
    "coordinate": "canonical_f_iv",
    "objective_application": "vega_kt_strategy_only",
    "risk_unit": "currency_per_unit_absolute_volatility",
    "audit_timing": "held_after_fills_and_proposed_after_lot_rounding",
    "complement_metric": "euclidean_reporting_quote_iv",
    "hedge_effectiveness": "not_established",
}


def _array(value, shape, name):
    try:
        value = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise BacktestError("market_shock_shape", name) from error
    require(value.shape == shape, "market_shock_shape", name)
    require(np.isfinite(value).all(), "market_shock_nonfinite", name)
    return value


def _norm(value):
    scale = float(np.max(abs(value))) if value.size else 0.
    result = scale*float(np.linalg.norm(value/scale)) if scale else 0.
    require(np.isfinite(result), "market_shock_nonfinite", "risk norm overflowed")
    return result


@dataclass(frozen=True)
class MarketShockMap:
    reporting: RiskBasis
    market: RiskBasis
    _strike_map: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        require(isinstance(self.reporting, RiskBasis) and isinstance(self.market, RiskBasis),
                "market_shock_basis", "RiskBasis inputs required")
        for name in ("valuation_date", "maturity_dates", "method", "coordinate", "delta_convention", "unit"):
            require(getattr(self.reporting, name) == getattr(self.market, name),
                    "market_shock_basis", f"{name} differs; no implicit conversion")
        require(self.market.coordinate == "canonical_f_iv", "market_shock_basis", "canonical IV coordinates required")
        fine, coarse = [np.asarray(b.log_moneyness) for b in (self.reporting, self.market)]
        # Accept only binary64 representation noise in otherwise identical knots.
        close = lambda a, b: abs(a-b) <= 16*np.finfo(float).eps*max(1., abs(a), abs(b))
        require(close(fine[0], coarse[0]) and close(fine[-1], coarse[-1]),
                "market_shock_domain", "reporting and market endpoints must agree")
        matches = [int(np.argmin(abs(fine-x))) for x in coarse]
        require(len(set(matches)) == len(coarse) and all(close(fine[i], x) for i, x in zip(matches, coarse)),
                "market_shock_not_nested", "every market knot must occur in the reporting grid")
        h = np.zeros((len(fine), len(coarse)))
        for j, x in enumerate(fine):
            if j in matches:
                h[j, matches.index(j)] = 1.
            else:
                upper = int(np.searchsorted(coarse, x, side="right"))
                require(0 < upper < len(coarse), "market_shock_domain", "reporting knot outside market range")
                weight = (x-coarse[upper-1])/(coarse[upper]-coarse[upper-1])
                h[j, upper-1], h[j, upper] = 1.-weight, weight
        h.setflags(write=False)
        object.__setattr__(self, "_strike_map", h)

    @property
    def map_id(self):
        return fingerprint({"method": "nested_log_iv_market_shocks_v1",
                            "reporting": self.reporting, "market": self.market})

    @property
    def matrix(self):
        """A copy of d reporting_IV / d market_IV, in maturity-major order."""
        return np.kron(np.eye(len(self.market.maturity_dates)), self._strike_map)

    def project_risk(self, raw_buckets):
        r = _array(raw_buckets, (self.reporting.size,), "raw buckets")
        out = (r.reshape(-1, len(self.reporting.log_moneyness)) @ self._strike_map).ravel()
        return _array(out, (self.market.size,), "projected risk")

    def project_hedges(self, raw_matrix):
        a = np.asarray(raw_matrix, dtype=float)
        require(a.ndim == 2 and a.shape[0] == self.reporting.size,
                "market_shock_shape", "hedge matrix must be reporting buckets by instruments")
        _array(a, a.shape, "hedge matrix")
        out = self.matrix.T @ a
        return _array(out, (self.market.size, a.shape[1]), "projected hedge matrix")

    def lift_shock(self, market_iv_change):
        z = _array(market_iv_change, (self.market.size,), "market IV change")
        out = (z.reshape(-1, len(self.market.log_moneyness)) @ self._strike_map.T).ravel()
        return _array(out, (self.reporting.size,), "reporting IV change")

    def project_sample_covariance(self, full_covariance):
        """Transform full sample covariance. Marginal variances are insufficient."""
        n = self.reporting.size
        c = _array(full_covariance, (n, n), "full sample covariance required")
        scale = max(float(np.max(np.abs(c))), np.finfo(float).tiny)
        # Work at unit scale to avoid overflow during validation/transformation.
        normalized = c/scale
        tol = 64*n*np.finfo(float).eps
        require(np.max(abs(normalized-normalized.T)) <= tol,
                "market_shock_covariance", "sample covariance must be symmetric")
        normalized = (normalized+normalized.T)/2
        require(np.linalg.eigvalsh(normalized)[0] >= -tol,
                "market_shock_covariance", "sample covariance must be positive semidefinite")
        h = self.matrix
        out = (h.T @ normalized @ h)*scale
        return _array(out, (self.market.size, self.market.size), "projected covariance")

    def complement_audit(self, raw_residual):
        """Expose risk orthogonal to all chosen shocks; never discard it as zero."""
        r = _array(raw_residual, (self.reporting.size,), "raw hedge residual")
        q, _ = np.linalg.qr(self._strike_map, mode="complete")
        inside, outside = q[:, :self._strike_map.shape[1]], q[:, self._strike_map.shape[1]:]
        rows = r.reshape(-1, len(self.reporting.log_moneyness))
        covered = ((rows @ inside) @ inside.T).ravel()
        # Use the complement columns directly. Subtracting nearly equal vectors
        # can produce a normalized witness that is not orthogonal to H.
        omitted = ((rows @ outside) @ outside.T).ravel()
        norm = _norm(omitted)
        # The witness has unit L2 size in reporting quotes. Multiplying by .01
        # means an L2-one-volatility-point move, not one point at every quote.
        witness = omitted/norm if norm else np.zeros_like(omitted)
        return {"metric": "euclidean_reporting_quote_iv",
                "reporting_l2": _norm(r),
                "covered_l2": _norm(covered), "complement_l2": norm,
                "worst_complement_one_point_pnl": .01*norm,
                "unit_complement_shock": witness.tolist(),
                "complement_buckets": omitted.tolist(),
                "market_hedge_effectiveness": "not_established"}

    def risk_audit(self, raw_buckets):
        """Keep chosen-shock sensitivity and the excluded directions together."""
        return {"market_buckets": self.project_risk(raw_buckets).tolist(),
                "complement": self.complement_audit(raw_buckets)}

    def identity(self):
        from .domain import data
        return {"map_id": self.map_id, "reporting_basis_id": self.reporting.basis_id,
                "market_basis_id": self.market.basis_id, "market_basis": data(self.market)}


@dataclass(frozen=True)
class MarketHedgeResult:
    solution: HedgeSolution
    map_id: str
    reporting_basis_id: str
    market_basis_id: str
    reporting_residual: tuple[float, ...]
    market_residual: tuple[float, ...]
    complement: dict


def solve_market_hedge(shocks, raw_target, raw_hedges, previous, lower, upper, costs, **solver_options):
    """Optimize fixed shocks and always return the full residual/complement audit."""
    require(isinstance(shocks, MarketShockMap), "market_shock_basis", "MarketShockMap required")
    solution = solve_hedge(shocks.project_risk(raw_target), shocks.project_hedges(raw_hedges),
                            previous, lower, upper, costs, **solver_options)
    residual = np.asarray(raw_target)+np.asarray(raw_hedges)@solution.quantities
    return MarketHedgeResult(solution, shocks.map_id, shocks.reporting.basis_id, shocks.market.basis_id,
        tuple(float(x) for x in residual), tuple(float(x) for x in shocks.project_risk(residual)),
        shocks.complement_audit(residual))
