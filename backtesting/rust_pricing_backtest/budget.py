"""Convex cash/gross constraints with a continuous Delta-neutral stock hedge.

Absolute trades/holdings use linear epigraphs. A feasibility LP precedes the
quadratic solve; an independent KKT check follows it. Prices are decision-time
quotes only. Lot rounding and next-snapshot execution require separate checks.
"""

from dataclasses import dataclass
import math

import numpy as np

from .domain import BacktestError, checked_sum, require


POLICY = "decision_quote_cash_gross_delta_neutral_qp_v1"


def backend():
    try:
        import scipy
        from scipy.optimize import linprog, minimize, nnls
    except ImportError as error:
        raise BacktestError("missing_constraint_dependency", "install rust-pricing-backtest[constraints]") from error
    return scipy, linprog, minimize, nnls


def execution_coefficients(quote, multiplier, fee, slippage_bps):
    """Spend for signed change q is linear*q + spread*abs(q), including fees."""
    buy = multiplier*quote.ask*(1+slippage_bps/10000.) + fee
    sell = multiplier*quote.bid*(1-slippage_bps/10000.) - fee
    linear, spread = buy/2+sell/2, buy/2-sell/2
    require(math.isfinite(linear) and math.isfinite(spread) and spread >= 0,
            "nonfinite_budget", "invalid execution coefficients")
    return linear, spread


def estimate(holdings, previous, linear, spreads, gross_weights, *, cash, fixed_gross):
    require(len(holdings) == len(previous) == len(linear) == len(spreads) == len(gross_weights),
            "solver_shape", "budget estimate dimensions")
    changes = [h-p for h, p in zip(holdings, previous)]
    spent = checked_sum((k*q+c*abs(q) for k, c, q in zip(linear, spreads, changes)), "nonfinite_budget", "estimated trade cost")
    cash_after = checked_sum((cash, -spent), "nonfinite_budget", "estimated cash")
    gross = checked_sum((fixed_gross, *(w*abs(h) for w, h in zip(gross_weights, holdings))), "nonfinite_budget", "gross notional")
    return {"cash_after": cash_after, "gross_notional": gross}


def feasible(metrics, cash_floor, gross_limit):
    # Only a binary64 roundoff allowance; this is not solver-tolerance-based
    # enlargement of a financial limit. Actual execution retains its cash check.
    cash_eps = 16*math.ulp(max(1., abs(metrics["cash_after"]), abs(cash_floor)))
    gross_eps = 16*math.ulp(max(1., abs(metrics["gross_notional"]), abs(gross_limit)))
    return metrics["cash_after"] >= cash_floor-cash_eps and metrics["gross_notional"] <= gross_limit+gross_eps


@dataclass(frozen=True)
class BudgetSolution:
    quantities: tuple[float, ...]
    rank: int
    condition_number: float | None
    singular_values: tuple[float, ...]
    objective_before: float
    objective_after: float
    iterations: int
    projected_gradient: None
    stock: float
    cash_after: float
    gross_notional: float
    method: str
    scipy_version: str
    kkt_stationarity: float
    kkt_complementarity: float
    primal_violation: float
    objective_scale: float
    optimizer_status: int | None
    optimizer_message: str
    optimizer_success: bool


def solve_budget(v, matrix, previous, lower, upper, costs, *, deltas, target_delta,
                 stock_previous, stock_lower, stock_upper, linear, spreads, gross_weights,
                 cash, cash_floor, fixed_gross, gross_limit, ridge=1e-8,
                 turnover_penalty=0., tolerance=1e-9, max_iterations=10000):
    """Minimize the existing Vega objective subject to cash, gross and stock bounds.

Stock is n=-target_delta-deltas@h. Arrays linear/spreads/gross_weights contain
options then stock. The fixed target contributes only to fixed_gross and Delta.
The turnover objective retains the existing option-only penalty convention.
"""
    scipy, linprog, minimize, nnls = backend()
    v, a, prev, lo, hi, cost, delta, k, c, weights = [np.asarray(x, dtype=float) for x in
        (v, matrix, previous, lower, upper, costs, deltas, linear, spreads, gross_weights)]
    n = len(prev)
    require(v.ndim == 1 and a.shape == (len(v), n) and all(x.shape == (n,) for x in (prev, lo, hi, cost, delta))
            and all(x.shape == (n+1,) for x in (k, c, weights)), "solver_shape", "budget dimensions")
    scalars = (target_delta, stock_previous, stock_lower, stock_upper, cash, cash_floor,
               fixed_gross, gross_limit, ridge, turnover_penalty, tolerance)
    require(all(np.isfinite(x).all() for x in (v, a, prev, lo, hi, cost, delta, k, c, weights))
            and all(math.isfinite(x) for x in scalars), "solver_nonfinite", "budget inputs")
    require(np.all(lo <= hi) and stock_lower <= stock_upper and np.all(cost >= 0) and np.all(c >= 0)
            and np.all(weights >= 0) and fixed_gross >= 0 and gross_limit >= 0 and ridge >= 0
            and turnover_penalty >= 0 and tolerance > 0 and type(max_iterations) is int and max_iterations > 0,
            "solver_constraints", "budget bounds/settings")
    scaled_a, scaled_v = .01*a, .01*v
    sv = np.linalg.svd(scaled_a, compute_uv=False) if n else np.array([])
    rank = int(np.sum(sv > max(a.shape)*np.finfo(float).eps*(sv[0] if len(sv) else 0)))
    condition = float(sv[0]/sv[-1]) if n and rank == n else None
    q, b = scaled_a.T@scaled_a+ridge*np.eye(n), scaled_a.T@scaled_v
    penalty = turnover_penalty*cost
    neutral = -target_delta

    def actual(h):
        stock = float(neutral-delta@h)
        metrics = estimate([*h, stock], [*prev, stock_previous], k, c, weights, cash=cash, fixed_gross=fixed_gross)
        return stock, metrics

    def original_objective(h):
        value = float(np.linalg.norm(scaled_v+scaled_a@h)**2+ridge*(h@h)+penalty@np.abs(h-prev))
        require(math.isfinite(value), "solver_nonfinite", "budget objective")
        return value

    # No option variable: Delta-only (or an empty option universe) has one exact
    # continuous stock hedge. Infeasibility is not silently converted to a clip.
    if not n:
        stock, metrics = actual(prev)
        require(stock_lower <= stock <= stock_upper and feasible(metrics, cash_floor, gross_limit),
                "budget_infeasible", "Delta-neutral stock hedge exceeds cash, gross or stock bounds")
        return BudgetSolution((), rank, condition, tuple(sv), original_objective(prev), original_objective(prev), 0, None,
                              stock, **metrics, method=POLICY, scipy_version=scipy.__version__,
                              kkt_stationarity=0., kkt_complementarity=0., primal_violation=0., objective_scale=1.,
                              optimizer_status=None, optimizer_message="direct Delta hedge; no option variables", optimizer_success=True)

    # x = [h, |h-prev|, |stock-prev_stock|, |h|, |stock|].
    size, u, stock_trade, y, stock_abs = 3*n+2, n, 2*n, 2*n+1, 3*n+1
    rows, rhs = [], []

    def constraint(terms, bound):
        row = np.zeros(size)
        for index, value in terms.items():
            row[index] = value
        rows.append(row)
        rhs.append(bound)

    for j in range(n):
        constraint({j: 1., u+j: -1.}, prev[j])
        constraint({j: -1., u+j: -1.}, -prev[j])
        constraint({j: 1., y+j: -1.}, 0.)
        constraint({j: -1., y+j: -1.}, 0.)
    constraint({**{j: -delta[j] for j in range(n)}, stock_trade: -1.}, stock_previous-neutral)
    constraint({**{j: delta[j] for j in range(n)}, stock_trade: -1.}, neutral-stock_previous)
    constraint({**{j: -delta[j] for j in range(n)}, stock_abs: -1.}, -neutral)
    constraint({**{j: delta[j] for j in range(n)}, stock_abs: -1.}, neutral)
    constraint({j: -delta[j] for j in range(n)}, stock_upper-neutral)
    constraint({j: delta[j] for j in range(n)}, neutral-stock_lower)
    constraint({**{j: k[j]-k[-1]*delta[j] for j in range(n)},
                **{u+j: c[j] for j in range(n)}, stock_trade: c[-1]},
               cash-cash_floor+float(k[:-1]@prev)+k[-1]*(stock_previous-neutral))
    constraint({**{y+j: weights[j] for j in range(n)}, stock_abs: weights[-1]}, gross_limit-fixed_gross)
    # Include bounds in the same matrix so certification covers all constraints.
    for j in range(n):
        constraint({j: 1.}, hi[j])
        constraint({j: -1.}, -lo[j])
    for j in range(n, size):
        constraint({j: -1.}, 0.)
    g, limits = np.asarray(rows), np.asarray(rhs)
    require(np.isfinite(g).all() and np.isfinite(limits).all() and np.isfinite(q).all() and np.isfinite(b).all(),
            "solver_nonfinite", "budget constraint arithmetic")
    row_scale = np.maximum(np.linalg.norm(g, ord=np.inf, axis=1), 1.)
    g, limits = g/row_scale[:, None], limits/row_scale
    lp = linprog(np.zeros(size), A_ub=g, b_ub=limits, bounds=[(None, None)]*size, method="highs",
                 options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    require(lp.success, "budget_infeasible" if lp.status == 2 else "budget_feasibility_failed", str(lp.message))
    # Scale the objective, preserving its minimizer and recording residuals in
    # this numerical scale. This avoids premature stopping on tiny risk units.
    objective_scale = max(float(np.linalg.norm(q, ord=np.inf)), float(np.max(np.abs(b))), float(np.max(penalty)), 1e-12)
    require(math.isfinite(objective_scale) and np.isfinite(penalty).all(), "solver_nonfinite", "budget objective scale")

    def objective(x):
        h = x[:n]
        return float((h@q@h+2*b@h+penalty@x[u:u+n])/objective_scale)

    def gradient(x):
        grad = np.zeros(size)
        grad[:n] = 2*(q@x[:n]+b)
        grad[u:u+n] = penalty
        return grad/objective_scale

    result = minimize(objective, lp.x, jac=gradient, method="SLSQP",
        constraints={"type": "ineq", "fun": lambda x: limits-g@x, "jac": lambda x: -g},
        options={"ftol": min(tolerance*tolerance, 1e-12), "maxiter": max_iterations})
    require(np.isfinite(result.x).all(), "solver_nonfinite", "budget solution")
    slack = limits-g@result.x
    grad = gradient(result.x)
    require(np.isfinite(slack).all() and np.isfinite(grad).all(), "solver_nonfinite", "budget KKT arithmetic")
    primal = float(max(0., -np.min(slack)))
    active = slack <= max(10*tolerance, 1e-9)
    try:
        multipliers = nnls(g[active].T, -grad, maxiter=max(1000, 10*int(np.sum(active))))[0] if np.any(active) else np.array([])
    except RuntimeError as error:
        raise BacktestError("solver_not_converged", "budget KKT certificate did not converge") from error
    stationarity = float(np.max(np.abs(grad+g[active].T@multipliers)))
    complementarity = float(np.max(np.abs(slack[active]*multipliers))) if len(multipliers) else 0.
    require(max(primal, stationarity, complementarity) <= tolerance,
            "solver_not_converged", f"budget KKT: primal={primal}, stationarity={stationarity}, complementarity={complementarity}; {result.message}")
    # Recompute economic constraints from actual absolute quantities, not slack
    # variables. A numerical success code alone is never an acceptance gate.
    stock, metrics = actual(result.x[:n])
    require(feasible(metrics, cash_floor, gross_limit), "budget_numerical_infeasible", "continuous economic budget check")
    return BudgetSolution(tuple(float(x) for x in result.x[:n]), rank, condition, tuple(float(x) for x in sv),
                          original_objective(prev), original_objective(result.x[:n]), int(result.nit), None,
                          stock, **metrics, method=POLICY, scipy_version=scipy.__version__,
                          kkt_stationarity=stationarity, kkt_complementarity=complementarity, primal_violation=primal,
                          objective_scale=objective_scale, optimizer_status=int(result.status),
                          optimizer_message=str(result.message), optimizer_success=bool(result.success))
