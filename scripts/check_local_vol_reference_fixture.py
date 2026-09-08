#!/usr/bin/env python3
"""Independently verify the Gate L0 Local Volatility equation fixture."""

from __future__ import annotations

import json
import math
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any


getcontext().prec = 70
D = Decimal
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "local-vol" / "reference-cases-v0.1.json"
PI = D("3.141592653589793238462643383279502884197169399375105820974944")


class Checks:
    def __init__(self, decimal_abs: D, transition_abs: float) -> None:
        self.decimal_abs = decimal_abs
        self.transition_abs = transition_abs
        self.count = 0

    def decimal(self, label: str, actual: D, expected: str) -> None:
        self.count += 1
        error = abs(actual - D(expected))
        if error > self.decimal_abs:
            raise AssertionError(
                f"{label}: actual={actual} expected={expected} error={error}"
            )

    def binary64(self, label: str, actual: float, expected: float) -> None:
        self.count += 1
        error = abs(actual - expected)
        if error > self.transition_abs:
            raise AssertionError(
                f"{label}: actual={actual!r} expected={expected!r} error={error!r}"
            )


def require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise AssertionError(
            f"{label}: field mismatch missing={sorted(expected-actual)} "
            f"unknown={sorted(actual-expected)}"
        )


def ssvi_values(
    theta: D,
    rho: D,
    phi: D,
    phi_theta: D,
    k: D,
    theta_t: D,
    forward: D,
) -> dict[str, D]:
    y = phi * k
    a = y + rho
    q = (a * a + 1 - rho * rho).sqrt()
    shape = 1 + rho * y + q
    w = theta * shape / 2
    w_k = theta * phi * (rho + a / q) / 2
    w_kk = theta * phi * phi * (1 - rho * rho) / (2 * q**3)
    w_theta = shape / 2 + theta * k * phi_theta * (rho + a / q) / 2
    w_t = w_theta * theta_t
    u = 1 - k * w_k / (2 * w)
    density_factor = (
        u * u - (w_k * w_k / 4) * (1 / w + D("0.25")) + w_kk / 2
    )
    sqrt_w = w.sqrt()
    d2 = -k / sqrt_w - sqrt_w / 2
    strike = forward * k.exp()
    normal_pdf = (-d2 * d2 / 2).exp() / (2 * PI).sqrt()
    call_density = normal_pdf / (strike * sqrt_w) * density_factor
    return {
        "phi": phi,
        "phi_theta": phi_theta,
        "w": w,
        "w_k": w_k,
        "w_kk": w_kk,
        "w_t": w_t,
        "density_factor": density_factor,
        "call_density": call_density,
    }


def check_standard_ssvi(fixture: dict[str, Any], checks: Checks) -> None:
    for case in fixture["standard_ssvi"]:
        inputs = {key: D(value) for key, value in case["inputs"].items()}
        theta = inputs["theta"]
        rho = inputs["rho"]
        if case["id"] == "power_regular":
            eta = inputs["eta"]
            gamma = inputs["gamma"]
            phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))
            phi_theta = -phi * (
                gamma / theta + (1 - gamma) / (1 + theta)
            )
        elif case["id"] in {"heston_like_regular", "heston_like_small_theta"}:
            lambda_ = inputs["lambda"]
            z = lambda_ * theta
            if abs(z) <= D("1e-4"):
                phi = D(1) / 2 + z * (
                    -D(1) / 6
                    + z
                    * (
                        D(1) / 24
                        + z
                        * (
                            -D(1) / 120
                            + z
                            * (
                                D(1) / 720
                                + z * (-D(1) / 5040 + z / D(40320))
                            )
                        )
                    )
                )
                phi_theta = lambda_ * (
                    -D(1) / 6
                    + z
                    * (
                        D(1) / 12
                        + z
                        * (
                            -D(1) / 40
                            + z
                            * (
                                D(1) / 180
                                + z * (-D(1) / 1008 + z / D(6720))
                            )
                        )
                    )
                )
            else:
                exp_minus_z = (-z).exp()
                phi = (z - 1 + exp_minus_z) / z**2
                phi_theta = (
                    lambda_ * (2 - z - (z + 2) * exp_minus_z) / z**3
                )
        else:
            raise AssertionError(f"unknown Standard SSVI fixture {case['id']!r}")

        actual = ssvi_values(
            theta,
            rho,
            phi,
            phi_theta,
            inputs["k"],
            inputs["theta_t"],
            inputs["forward"],
        )
        for field, expected in case["expected"].items():
            checks.decimal(f"standard_ssvi.{case['id']}.{field}", actual[field], expected)


def check_essvi(fixture: dict[str, Any], checks: Checks) -> None:
    case = fixture["essvi_interpolation"]
    x = {key: D(value) for key, value in case["inputs"].items()}
    weight = (x["t"] - x["t0"]) / (x["t1"] - x["t0"])
    theta = x["theta0"] + weight * (x["theta1"] - x["theta0"])
    psi = x["psi0"] + weight * (x["psi1"] - x["psi0"])
    rho_psi = x["rho_psi0"] + weight * (x["rho_psi1"] - x["rho_psi0"])
    theta_t = (x["theta1"] - x["theta0"]) / (x["t1"] - x["t0"])
    psi_t = (x["psi1"] - x["psi0"]) / (x["t1"] - x["t0"])
    rho_psi_t = (x["rho_psi1"] - x["rho_psi0"]) / (x["t1"] - x["t0"])
    rho = rho_psi / psi
    rho_t = (rho_psi_t * psi - rho_psi * psi_t) / psi**2
    phi = psi / theta
    y = phi * x["k"]
    y_t = x["k"] * (psi_t * theta - psi * theta_t) / theta**2
    a = y + rho
    q = (a * a + 1 - rho * rho).sqrt()
    q_t = ((y + rho) * (y_t + rho_t) - rho * rho_t) / q
    shape = 1 + rho * y + q
    actual = {
        "theta": theta,
        "psi": psi,
        "rho_psi": rho_psi,
        "rho": rho,
        "rho_t": rho_t,
        "w": theta * shape / 2,
        "w_k": theta * phi * (rho + a / q) / 2,
        "w_kk": theta * phi * phi * (1 - rho * rho) / (2 * q**3),
        "w_t": theta_t * shape / 2
        + theta * (rho_t * y + rho * y_t + q_t) / 2,
    }
    for field, expected in case["expected"].items():
        checks.decimal(f"essvi.{field}", actual[field], expected)


def check_algebraic_cases(fixture: dict[str, Any], checks: Checks) -> None:
    case = fixture["constant_vol_dupire"]
    x = {key: D(value) for key, value in case["inputs"].items()}
    actual = {
        "w": x["sigma"] ** 2 * x["t"],
        "w_t": x["sigma"] ** 2,
        "w_k": D(0),
        "w_kk": D(0),
        "density_factor": D(1),
        "local_variance": x["sigma"] ** 2,
    }
    for field, expected in case["expected"].items():
        checks.decimal(f"constant_vol_dupire.{field}", actual[field], expected)

    case = fixture["equation_11"]
    x = {key: D(value) for key, value in case["inputs"].items()}
    gamma = x["local_vega_strike_density"] / (
        x["strike"] ** 2
        * x["call_density"]
        * x["local_volatility"]
        * x["delta_t"]
    )
    checks.decimal(
        "equation_11.next_local_gamma",
        gamma,
        case["expected"]["next_local_gamma"],
    )

    case = fixture["hat_projection"]
    nodes = [D(value) for value in case["inputs"]["nodes"]]
    x_value = D(case["inputs"]["x"])
    adjoint = D(case["inputs"]["path_adjoint"])
    right_weight = (x_value - nodes[1]) / (nodes[2] - nodes[1])
    weights = [D(0), 1 - right_weight, right_weight]
    masses = [
        (nodes[1] - nodes[0]) / 2,
        (nodes[2] - nodes[0]) / 2,
        (nodes[2] - nodes[1]) / 2,
    ]
    densities = [adjoint * weight / mass for weight, mass in zip(weights, masses)]
    conserved = sum(mass * density for mass, density in zip(masses, densities))
    for index, expected in enumerate(case["expected"]["weights"]):
        checks.decimal(f"hat.weights[{index}]", weights[index], expected)
    for index, expected in enumerate(case["expected"]["lumped_masses"]):
        checks.decimal(f"hat.lumped_masses[{index}]", masses[index], expected)
    for index, expected in enumerate(case["expected"]["density_nodes"]):
        checks.decimal(f"hat.density_nodes[{index}]", densities[index], expected)
    checks.decimal("hat.conserved", conserved, case["expected"]["conserved_adjoint"])


def normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def check_transition(fixture: dict[str, Any], checks: Checks) -> None:
    case = fixture["gamma_transition"]
    spot = case["inputs"]["spot"]
    variance = case["inputs"]["variance"]
    boundaries = [
        math.inf if value == "infinity" else value
        for value in case["inputs"]["cell_boundaries"]
    ]
    mu = math.log(spot) + 1.5 * variance
    standard_deviation = math.sqrt(variance)
    probabilities: list[float] = []
    first_moments: list[float] = []
    for lower, upper in zip(boundaries, boundaries[1:]):
        lower_z = -math.inf if lower == 0.0 else (math.log(lower) - mu) / standard_deviation
        upper_z = math.inf if math.isinf(upper) else (math.log(upper) - mu) / standard_deviation
        probabilities.append(normal_cdf(upper_z) - normal_cdf(lower_z))

        lower_m = -math.inf if lower == 0.0 else (math.log(lower) - mu - variance) / standard_deviation
        upper_m = math.inf if math.isinf(upper) else (math.log(upper) - mu - variance) / standard_deviation
        first_moments.append(
            math.exp(mu + variance / 2)
            * (normal_cdf(upper_m) - normal_cdf(lower_m))
        )

    expected = case["expected"]
    for index, value in enumerate(probabilities):
        checks.binary64(
            f"transition.probability[{index}]",
            value,
            expected["normalized_cell_probability"][index],
        )
    for index, value in enumerate(first_moments):
        checks.binary64(
            f"transition.first_moment[{index}]",
            value,
            expected["normalized_cell_first_moment"][index],
        )
    checks.binary64(
        "transition.total_probability",
        sum(probabilities),
        expected["normalized_total_probability"],
    )
    checks.binary64(
        "transition.total_first_moment",
        sum(first_moments),
        expected["normalized_total_first_moment"],
    )
    checks.binary64(
        "transition.gamma_kernel_mass",
        math.exp(variance),
        expected["gamma_kernel_mass"],
    )
    checks.binary64(
        "transition.gamma_kernel_first_moment",
        spot * math.exp(3 * variance),
        expected["gamma_kernel_first_moment"],
    )


def check_dividend(fixture: dict[str, Any], checks: Checks) -> None:
    case = fixture["affine_dividend"]
    x = {
        key: [D(item) for item in value] if isinstance(value, list) else D(value)
        for key, value in case["inputs"].items()
    }
    alpha = x["fixed_cash"] / x["spot0"]
    a_after = (1 - x["proportional"]) * x["a_before"] - alpha
    b_after = (1 - x["proportional"]) * x["b_before"]
    post_event_spot = a_after * x["spot0"] + b_after * x["f"]
    mapped_strike = (
        x["matching_strike"] + alpha * x["spot0"]
    ) / (1 - x["proportional"])

    pre_event_call = sum(
        probability * max(state - mapped_strike, D(0))
        for state, probability in zip(
            x["pre_event_states"], x["state_probabilities"]
        )
    )
    scaled_pre_event_call = (1 - x["proportional"]) * pre_event_call
    post_event_call = sum(
        probability
        * max(
            (1 - x["proportional"]) * state
            - alpha * x["spot0"]
            - x["matching_strike"],
            D(0),
        )
        for state, probability in zip(
            x["pre_event_states"], x["state_probabilities"]
        )
    )
    actual = {
        "alpha": alpha,
        "a_after": a_after,
        "b_after": b_after,
        "post_event_spot": post_event_spot,
        "mapped_pre_event_strike": mapped_strike,
        "post_event_call": post_event_call,
        "scaled_pre_event_call": scaled_pre_event_call,
    }
    for field, expected in case["expected"].items():
        checks.decimal(f"affine_dividend.{field}", actual[field], expected)


def main() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    require_keys(
        fixture,
        {
            "fixture_version",
            "policy_version",
            "tolerances",
            "standard_ssvi",
            "essvi_interpolation",
            "constant_vol_dupire",
            "equation_11",
            "hat_projection",
            "gamma_transition",
            "affine_dividend",
        },
        "fixture",
    )
    if fixture["fixture_version"] != 1:
        raise AssertionError("unsupported fixture_version")
    if fixture["policy_version"] != "local_vol_vegakt_v1":
        raise AssertionError("unexpected policy_version")

    checks = Checks(
        D(fixture["tolerances"]["decimal_abs"]),
        float(fixture["tolerances"]["transition_abs"]),
    )
    check_standard_ssvi(fixture, checks)
    check_essvi(fixture, checks)
    check_algebraic_cases(fixture, checks)
    check_transition(fixture, checks)
    check_dividend(fixture, checks)
    print(f"Local Volatility reference fixture: {checks.count} checks passed")


if __name__ == "__main__":
    main()
