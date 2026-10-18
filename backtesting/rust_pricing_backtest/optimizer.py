"""Deterministic box-constrained quadratic/L1 hedge solver."""

from dataclasses import dataclass
import math
import numpy as np

from .domain import require


@dataclass(frozen=True)
class HedgeSolution:
    quantities: tuple[float, ...]
    rank: int
    condition_number: float | None
    singular_values: tuple[float, ...]
    objective_before: float
    objective_after: float
    iterations: int
    projected_gradient: float


def solve_hedge(v, matrix, previous, lower, upper, costs, *, ridge=1e-8,
                turnover_penalty=0.0, tolerance=1e-9, max_iterations=10_000) -> HedgeSolution:
    """Minimize ||.01(v+A h)||² + ridge||h||² + penalty sum(cost|h-prev|).

    Coordinate descent uses the exact box-constrained soft-threshold solution.
    Termination checks the projected proximal gradient, not only small steps.
    Rank diagnostics describe the unregularized, market-scaled risk matrix.
    """
    v, a, prev, lo, hi, cost = [np.asarray(x, dtype=float) for x in
                              (v, matrix, previous, lower, upper, costs)]
    n = len(prev)
    require(v.ndim == 1 and a.shape == (len(v), n)
            and all(x.shape == (n,) for x in (prev, lo, hi, cost)),
            "solver_shape", "incompatible vector/matrix shapes")
    require(all(np.isfinite(x).all() for x in (v, a, prev, lo, hi, cost)), "solver_nonfinite", "nonfinite input")
    require(np.all(lo <= hi) and np.all(cost >= 0), "solver_constraints", "invalid bounds or costs")
    require(math.isfinite(ridge) and ridge >= 0 and math.isfinite(turnover_penalty)
            and turnover_penalty >= 0 and math.isfinite(tolerance) and tolerance > 0
            and max_iterations > 0, "solver_config", "invalid solver configuration")
    scaled_a, scaled_v = .01 * a, .01 * v
    sv = np.linalg.svd(scaled_a, compute_uv=False) if n else np.array([])
    cutoff = max(a.shape, default=0) * np.finfo(float).eps * (sv[0] if len(sv) else 0)
    rank = int(np.sum(sv > cutoff))
    condition = float(sv[0] / sv[-1]) if n and rank == n else None
    q = scaled_a.T @ scaled_a + ridge * np.eye(n)
    b = scaled_a.T @ scaled_v
    penalty = turnover_penalty * cost

    def objective(h):
        return float(np.linalg.norm(scaled_v + scaled_a @ h) ** 2
                     + ridge * (h @ h) + penalty @ np.abs(h - prev))

    def prox(z, threshold):
        return np.clip(prev + np.sign(z - prev) * np.maximum(np.abs(z - prev) - threshold, 0), lo, hi)

    h = np.clip(prev, lo, hi)
    lipschitz = max(2 * float(np.linalg.norm(q, ord=np.inf)), 1e-30) if n else 1.0
    pg = 0.0
    for iteration in range(1, max_iterations + 1):
        for j in range(n):
            if lo[j] == hi[j]:
                h[j] = lo[j]
            elif q[j, j] > 0:
                z = -(b[j] + q[j] @ h - q[j, j] * h[j]) / q[j, j]
                diff = z - prev[j]
                soft = prev[j] + np.sign(diff) * max(abs(diff) - penalty[j] / (2 * q[j, j]), 0)
                h[j] = np.clip(soft, lo[j], hi[j])
            else:
                h[j] = np.clip(prev[j], lo[j], hi[j])
        grad = 2 * (q @ h + b)
        pg = float(np.max(np.abs(lipschitz * (h - prox(h - grad / lipschitz, penalty / lipschitz))))) if n else 0.0
        if pg <= tolerance:
            return HedgeSolution(tuple(float(x) for x in h), rank, condition,
                                 tuple(float(x) for x in sv), objective(prev), objective(h), iteration, pg)
    require(False, "solver_not_converged", f"projected gradient {pg} exceeds {tolerance}")


def round_lot(value: float, lot: float, lower: float | None = None, upper: float | None = None) -> float:
    """Round toward zero; never increase exposure merely to satisfy a lot."""
    scaled = value / lot
    nearest = round(scaled)
    # Preserve exact-lot analytical solutions perturbed only by binary64 noise.
    # A genuine fractional lot is still rounded toward zero.
    units = nearest if abs(scaled - nearest) <= 8 * math.ulp(scaled) else math.trunc(scaled)
    if lower is not None and upper is not None:
        lo, hi = math.ceil(lower / lot - 1e-12), math.floor(upper / lot + 1e-12)
        require(lo <= hi, "lot_infeasible", "no admissible lot within bounds")
        units = min(hi, max(lo, units))
    return units * lot
