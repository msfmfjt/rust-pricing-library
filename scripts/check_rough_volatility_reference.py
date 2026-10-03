"""Check independent numerical reference fixtures for the rough-model extension.

Requires NumPy and SciPy only for this developer check, not for Rust pricing.
This script does NOT execute or validate the compiled Rust implementation.
Rust integration tests separately consume the retained reference values.

References use the spectral fractional-OU integral, math.gamma, and a scalar
one-cell conditional Gaussian law, independently of production kernel helpers.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "rough-volatility" / "reference.json"


def spectral_correlation(h: float, lag: float) -> float:
    if lag == 0.0:
        return 1.0
    if h == 0.5:
        return math.exp(-lag)
    integral, error = quad(
        lambda omega: omega ** (1.0 - 2.0 * h) / (1.0 + omega * omega),
        0.0, math.inf, weight="cos", wvar=lag,
        epsabs=2e-11, limlst=500, limit=500,
    )
    if error > 1e-9:
        raise AssertionError(f"spectral quadrature error {error} for {(h, lag)}")
    return 2.0 * math.sin(math.pi * h) / math.pi * integral


def second_difference_correlation(h: float, lag: float) -> float:
    if lag == 0.0:
        return 1.0
    if h == 0.5:
        return math.exp(-lag)
    q = 2.0 * h
    def integrand(u: float) -> float:
        return math.exp(-u) * ((u + lag) ** q + abs(u - lag) ** q - 2.0 * lag ** q)
    value = quad(integrand, 0.0, min(lag, 64.0), epsabs=1e-13, epsrel=1e-12, limit=500)[0]
    if lag < 64.0:
        value += quad(integrand, lag, 64.0, epsabs=1e-13, epsrel=1e-12, limit=500)[0]
    return value / (2.0 * math.gamma(q + 1.0))


def references() -> dict:
    covariances = []
    max_gap = 0.0
    for h in [0.001, 0.01, 0.1, 0.3, 0.49, 0.5]:
        for lag in [0.01, 0.1, 1.0, 3.0, 10.0]:
            spectral = spectral_correlation(h, lag)
            alternate = second_difference_correlation(h, lag)
            max_gap = max(max_gap, abs(spectral - alternate))
            if abs(spectral - alternate) > 2e-10:
                raise AssertionError((h, lag, spectral, alternate))
            covariances.append({"hurst": h, "lag": lag, "correlation": spectral})
    cells = []
    for h in [0.01, 0.1, 0.3, 0.5]:
        dt, v0, kappa, theta, nu, rho = 0.25, 0.04, 1.2, 0.07, 0.3, -0.6
        zs, zv, zn = 0.2, -0.4, 0.7
        alpha = h + 0.5
        dw = math.sqrt(dt) * (rho * zs + math.sqrt(1.0 - rho * rho) * zv)
        integral = dt ** alpha / math.gamma(alpha + 1.0)
        second_moment = dt ** (2.0 * h) / (2.0 * h * math.gamma(alpha) ** 2)
        conditional_mean = integral / dt * dw
        residual_variance = max(0.0, second_moment - integral * integral / dt)
        innovation = conditional_mean + math.sqrt(residual_variance) * zn
        variance = v0 + kappa * (theta - v0) * integral + nu * math.sqrt(v0) * innovation
        forward = 100.0 * math.exp(-0.5 * v0 * dt + math.sqrt(v0 * dt) * zs)
        cells.append(dict(hurst=h, dt=dt, initial_variance=v0, mean_reversion=kappa,
                          long_run_variance=theta, vol_of_vol=nu, correlation=rho,
                          normals=[zs, zv, zn], raw_variance=variance, forward=forward))
    # Independent deterministic-rate, zero-cash Black reference for the common
    # request fixture: spot=100, discount=.95, repo-spread factor=.98, T=1.
    f, strike, sigma, discount = 100.0 * 0.98 / 0.95, 100.0, 0.2, 0.95
    d1 = math.log(f / strike) / sigma + sigma / 2.0
    black_call = discount * (f * ndtr(d1) - strike * ndtr(d1 - sigma))
    print(f"Spectral versus second-difference covariance: 30 cases, max gap {max_gap:.3e}")
    # Independently check SPD on an irregular grid, including negative-tail
    # correlations, without covariance clipping or diagonal jitter.
    for h in [0.01, 0.1, 0.3, 0.49, 0.5]:
        times = [0.0, 0.03, 0.1, 0.4, 1.0, 3.0, 10.0]
        cov = np.array([[spectral_correlation(h, abs(t-s)) for t in times] for s in times])
        chol = np.linalg.cholesky(cov)
        assert np.max(np.abs(chol @ chol.T - cov)) < 2e-14
    print("Fractional-OU positive definiteness: 5 irregular-grid cases passed")
    return {"version": 1, "rfsv_correlations": covariances,
            "heston_one_step": cells, "black_call": float(black_call)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="regenerate the explicit reference fixture")
    args = parser.parse_args()
    expected = references()
    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(expected, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Wrote {FIXTURE.relative_to(ROOT)}")
        return
    actual = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert actual.keys() == expected.keys() and actual["version"] == 1
    for group in ["rfsv_correlations", "heston_one_step"]:
        assert len(actual[group]) == len(expected[group])
        for a, e in zip(actual[group], expected[group]):
            assert a.keys() == e.keys()
            for key in a:
                if isinstance(e[key], list):
                    assert a[key] == e[key]
                else:
                    assert abs(a[key]-e[key]) <= 2e-10, (group, key, a[key], e[key])
    assert abs(actual["black_call"] - expected["black_call"]) < 1e-12
    print("Retained covariance, one-cell Heston and Black references match")


if __name__ == "__main__":
    main()
