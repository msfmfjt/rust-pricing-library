"""Independent *finite two-step* prices, not continuous-time rough-model prices.

Integrate out the asset Brownian increments analytically conditional on the
first volatility innovation. Production Rust paths, RNGs and payoff helpers
are never called. A separate stationary-fOU Gaussian quadrature covers RFSV.
Use --write only for an explicitly reviewed fixture update; CI only checks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, roots_hermitenorm

from check_rough_volatility_reference import (
    second_difference_correlation, spectral_correlation,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/rough-volatility/two-step-prices.json"
SPOT, DT = 100.0, 0.5
STRIKES = [90.0, 100.0, 110.0]


def black(forward, strike, integrated_variance):
    width = np.sqrt(integrated_variance)
    d1 = np.log(forward / strike) / width + 0.5 * width
    return forward * ndtr(d1) - strike * ndtr(d1 - width)


def gaussian_price(v0, innovation_correlation, next_variance, strike, split=None):
    """Exact conditional Gaussian law of the two log-Euler asset steps."""
    c = innovation_correlation
    def integrand(y):
        forward = SPOT * math.exp(math.sqrt(v0 * DT) * c * y - 0.5 * v0 * DT * c * c)
        variance = v0 * DT * (1.0 - c * c) + DT * next_variance(y)
        return float(black(forward, strike, variance)) * math.exp(-0.5 * y*y) / math.sqrt(2*math.pi)
    # The finite 14-sigma cutoff is intentionally much wider than the actual
    # Gaussian exponential tilt (less than 0.2 in these cases). Split the
    # Heston truncation kink explicitly; no price comes from a Monte Carlo fit.
    knots = [-14.0] + ([split] if split is not None and -14 < split < 14 else []) + [14.0]
    values = []
    for tolerance in [2e-10, 2e-12]:
        parts = [quad(integrand, a, b, epsabs=tolerance, epsrel=tolerance,
                      limit=500) for a, b in zip(knots, knots[1:])]
        assert sum(error for _, error in parts) < 2e-8
        values.append(sum(value for value, _ in parts))
    assert abs(values[0] - values[1]) < 2e-9
    return values[1]


def build_reference():
    h, rho, v0, kappa, theta, nu = 0.2, -0.65, 0.04, 0.7, 0.055, 0.18
    alpha = h + 0.5
    integral = DT**alpha / math.gamma(alpha + 1)
    singular_std = DT**h / (math.sqrt(2*h) * math.gamma(alpha))
    near_correlation = math.sqrt(2*h) / alpha
    cases = []
    def add(name, parameters, initial_variance, correlation, next_variance, split=None):
        cases.append({"name": name, "parameters": parameters,
                      "call_prices": [gaussian_price(initial_variance, correlation,
                                                     next_variance, k, split) for k in STRIKES]})
    parameters = dict(hurst=h, initial_variance=v0, mean_reversion=kappa,
                      long_run_variance=theta, vol_of_vol=nu, correlation=rho)
    mean = v0 + kappa * (theta-v0) * integral
    std = nu * math.sqrt(v0) * singular_std
    add("rough_heston", parameters, v0, rho*near_correlation,
        lambda y: max(0.0, mean + std*y), -mean/std)

    weights, rates = [0.25, 0.9, 1.8], [0.1, 1.0, 8.0]
    loading = sum(w / (1+x*DT) for w, x in zip(weights, rates))
    mean_lift = v0 + loading * kappa * (theta-v0)*DT
    std_lift = loading * nu * math.sqrt(v0*DT)
    lift_parameters = {k:v for k,v in parameters.items() if k != "hurst"}
    lift_parameters.update(weights=weights, rates=rates)
    add("lifted_heston", lift_parameters, v0, rho,
        lambda y: max(0.0, mean_lift + std_lift*y), -mean_lift/std_lift)

    q = dict(hurst=h, initial_state=0.15, mean_reversion=1.1, vol_of_vol=0.5,
             quadratic=0.8, shift=0.25, variance_floor=0.02)
    qv0 = q["quadratic"]*(q["initial_state"]-q["shift"])**2 + q["variance_floor"]
    zmean = q["initial_state"]*(1-q["mean_reversion"]*integral)
    zstd = q["mean_reversion"]*q["vol_of_vol"]*math.sqrt(qv0)*singular_std
    add("quadratic_rough_heston", q, qv0, near_correlation,
        lambda y: q["quadratic"]*(zmean+zstd*y-q["shift"])**2+q["variance_floor"])

    m = dict(hurst=h, correlation=rho, weights=[0.35, 0.65], vol_of_vols=[0.45, 1.05],
             initial_forward_variance=v0, forward_variance_growth=0.08)
    add("mixed_rough_bergomi", m, v0, rho*near_correlation,
        lambda y: v0*math.exp(m["forward_variance_growth"]*DT)*sum(
            w*math.exp(eta*DT**h*y-0.5*eta*eta*DT**(2*h))
            for w, eta in zip(m["weights"], m["vol_of_vols"])))

    s = dict(hurst=h, correlation=rho, vol_of_vol=0.75, beta=1.0,
             initial_forward_variance=v0, forward_variance_growth=0.06)
    add("rough_sabr", s, v0, rho*near_correlation,
        lambda y: v0*math.exp(s["forward_variance_growth"]*DT+s["vol_of_vol"]*DT**h*y
                             -0.5*s["vol_of_vol"]**2*DT**(2*h)))

    r = dict(hurst=h, mean_reversion=1.3, vol_of_log_vol=0.2,
             mean_log_vol=math.log(0.2))
    covariance = spectral_correlation(h, r["mean_reversion"]*DT)
    assert abs(covariance-second_difference_correlation(h, r["mean_reversion"]*DT)) < 2e-10
    log_sd = math.sqrt(r["vol_of_log_vol"]**2 * math.gamma(2*h+1)
                       / (2*r["mean_reversion"]**(2*h)))
    refinements = []
    for order in [64, 96]:
        nodes, weights = roots_hermitenorm(order)
        x0 = r["mean_log_vol"] + log_sd*nodes[:, None]
        x1 = r["mean_log_vol"] + log_sd*(covariance*nodes[:, None]
                    + math.sqrt(1-covariance*covariance)*nodes[None, :])
        total_variance = DT*(np.exp(2*x0)+np.exp(2*x1))
        joint_weights = weights[:, None]*weights[None, :]/(2*math.pi)
        refinements.append([float(np.sum(joint_weights*black(SPOT, k, total_variance)))
                            for k in STRIKES])
    assert max(abs(a-b) for a,b in zip(*refinements)) < 2e-10
    cases.append({"name":"rfsv", "parameters":r, "call_prices":refinements[-1]})
    return {"version":1, "scope":"finite-two-step-log-euler", "spot":SPOT,
            "time_nodes":[0.0, DT, 2*DT], "strikes":STRIKES, "cases":cases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    expected = build_reference()
    if args.write:
        FIXTURE.write_text(json.dumps(expected, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    else:
        actual = json.loads(FIXTURE.read_text(encoding="utf-8"))
        assert actual.keys() == expected.keys()
        for field in ["version", "scope", "spot", "time_nodes", "strikes"]:
            assert actual[field] == expected[field]
        assert len(actual["cases"]) == len(expected["cases"])
        for a, e in zip(actual["cases"], expected["cases"]):
            assert a.keys() == e.keys() and a["name"] == e["name"]
            assert a["parameters"] == e["parameters"]
            assert len(a["call_prices"]) == len(e["call_prices"])
            for p, q in zip(a["call_prices"], e["call_prices"]):
                assert abs(p-q) <= 2e-9, (a["name"], p, q)
    for case in expected["cases"]:
        print(f'{case["name"]}: '+", ".join(f"{p:.12f}" for p in case["call_prices"]))
    print("18 nondegenerate finite-grid reference prices checked; not a continuous-time accuracy bound.")


if __name__ == "__main__":
    main()
