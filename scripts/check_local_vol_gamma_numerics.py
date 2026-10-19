#!/usr/bin/env python3
"""Independent Decimal integration references for the LV Gamma kernel.

No native library, scipy or normal-CDF implementation is used. --write creates
reviewable reference data; default checks the frozen artifact without changing it.
"""

import argparse
import json
from decimal import Decimal as D, localcontext
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/local-vol/gamma-numerics-v0.1.json"
PI = D(
    "3.141592653589793238462643383279502884197169399375105820974944"
    "59230781640628620899862803482534211706798214808651328230664709384460955"
)


def cdf(z: D) -> D:
    if z.is_infinite():
        return D(0) if z < 0 else D(1)
    # Integrate exp(-x*x/2) from 0 to z using its convergent power series.
    term, total, n = z, z, 0
    while True:
        n += 1
        term *= -z * z / (2 * n)
        updated = total + term / (2 * n + 1)
        if updated == total:
            break
        total = updated
        if n >= 10000:
            raise ArithmeticError("Gaussian reference series did not converge")
    return D(".5") + total / (2 * PI).sqrt()


def reference(spot: D, variance: D, strikes: list[D], gamma: list[D]) -> dict:
    sd = variance.sqrt()
    mean = spot.ln() + D("1.5") * variance
    bounds = [D(0), *strikes, D("Infinity")]
    probabilities, moments = [], []
    for lo, hi in zip(bounds, bounds[1:]):
        zl = (lo.ln() - mean) / sd if lo else D("-Infinity")
        zh = (hi.ln() - mean) / sd
        probabilities.append(cdf(zh) - cdf(zl))
        moments.append(spot * (2 * variance).exp() * (cdf(zh - sd) - cdf(zl - sd)))
    value = gamma[0] * probabilities[0] + gamma[-1] * probabilities[-1]
    for i in range(len(strikes) - 1):
        slope = (gamma[i + 1] - gamma[i]) / (strikes[i + 1] - strikes[i])
        value += (
            (gamma[i] - slope * strikes[i]) * probabilities[i + 1]
            + slope * moments[i + 1]
        )
    return {
        "probabilities": [float(p) for p in probabilities],
        "first_moments": [float(m) for m in moments],
        "gamma_integral": float(variance.exp() * value),
    }


def artifact() -> dict:
    cases = [
        ("central", 100.0, 0.04, [90.0, 110.0], [1.0, 2.0]),
        ("right_tail", 100.0, 0.04, [640.0, 710.0, 790.0], [0.0, 1.0, 3.0]),
        ("left_tail", 100.0, 0.04, [14.0, 15.5, 17.0], [3.0, 1.0, 0.0]),
        ("short_step", 100.0, 1e-20, [99.99999999, 100.0, 100.00000001], [1.0, 2.0, 4.0]),
        ("narrow_hat", 100.0, 0.04, [100.0, 100.000001, 100.000002], [0.0, 1.0, 0.0]),
        ("narrow_tail_hat", 100.0, 0.04, [640.0, 640.00001, 640.00002], [0.0, 1.0, 0.0]),
        (
            "survival_switch_hat", 100.0, 0.04,
            [193.47149442564688, 193.4869727643262, 193.5024523413222],
            [0.0, 1.0, 0.0],
        ),
        (
            "tail_interval_boundary_hat", 100.0, 0.04,
            [236.306616916855, 236.32552220240973, 236.34442900044778],
            [0.0, 1.0, 0.0],
        ),
    ]
    out = []
    for name, spot, variance, strikes, gamma in cases:
        with localcontext() as ctx:
            ctx.prec = 120
            expected = reference(
                D(spot), D(variance), list(map(D, strikes)), list(map(D, gamma))
            )
        out.append({
            "id": name, "spot": spot, "variance": variance,
            "strikes": strikes, "gamma": gamma, "expected": expected,
        })
    return {
        "fixture_version": 1,
        "reference": "120-digit Decimal power-series integral; exact binary64 inputs",
        "formula_source": "https://dlmf.nist.gov/7.2#E1",
        "relative_tolerance": 1e-10,
        "absolute_tolerance": 0.0,
        "cases": out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = json.dumps(artifact(), indent=2, allow_nan=False) + "\n"
    if args.write:
        FIXTURE.write_text(payload, encoding="utf-8")
    elif FIXTURE.read_text(encoding="utf-8") != payload:
        raise AssertionError("Gamma numerical references changed")
    print("Eight independent Gamma kernel reference cases verified")


if __name__ == "__main__":
    main()
