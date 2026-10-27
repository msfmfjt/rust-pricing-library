"""Independently difference scalar eSSVI variance using Decimal arithmetic.

The scalar formula uses theta, psi and rho_psi directly, without the production
phi/rho quotients or their analytic derivatives. Exact affine parameter inputs
come from the separate Fraction reference panel. Knot time differences follow
its specified local right-sided path, not a path crossing a slope discontinuity.
"""

from decimal import Decimal, localcontext
import json

from generate_essvi_parameter_references import generate as parameter_references


def evaluate(case: dict, log_moneyness: float, digits: int, step: int) -> dict:
    with localcontext() as context:
        context.prec = digits
        theta, psi, rho_psi, theta_t, psi_t, rho_psi_t = map(
            Decimal.from_float, case["expected"]
        )
        k = Decimal.from_float(log_moneyness)

        def scalar(t: Decimal, p: Decimal, r: Decimal, x: Decimal) -> Decimal:
            radicand = t * t + 2 * t * r * x + p * p * x * x
            return (t + r * x + radicand.sqrt()) / 2

        h = Decimal(10) ** step
        value = scalar(theta, psi, rho_psi, k)
        up = scalar(theta, psi, rho_psi, k + h)
        down = scalar(theta, psi, rho_psi, k - h)
        derivative_k = (up - down) / (2 * h)
        derivative_kk = (up + down - 2 * value) / (h * h)
        derivative_t = (
            scalar(theta + h * theta_t, psi + h * psi_t, rho_psi + h * rho_psi_t, k)
            - scalar(theta - h * theta_t, psi - h * psi_t, rho_psi - h * rho_psi_t, k)
        ) / (2 * h)
        return {
            "time": case["time"],
            "log_moneyness": log_moneyness,
            "parameters": case["expected"],
            "region": case["region"],
            "expected": dict(zip(
                ("w", "w_k", "w_kk", "w_t"),
                map(str, (value, derivative_k, derivative_kk, derivative_t)),
            )),
        }


def generate() -> dict:
    parameters = parameter_references()
    queries = [-2.0, 0.5, -0.5, 1.0, -1.5, 0.25, -0.75, 2.0, -0.25]
    coarse = [evaluate(case, k, 100, -20)
              for case, k in zip(parameters["cases"], queries, strict=True)]
    refined = [evaluate(case, k, 140, -30)
               for case, k in zip(parameters["cases"], queries, strict=True)]
    with localcontext() as context:
        context.prec = 140
        maximum = max(
            abs(Decimal(left["expected"][field]) - Decimal(right["expected"][field]))
            for left, right in zip(coarse, refined, strict=True)
            for field in left["expected"]
        )
        assert maximum < Decimal("1e-37"), maximum
    return {
        "method": "Centered differences of scalar variance in theta/psi/rho_psi, independently of analytic quotient derivatives.",
        "precision_digits": [100, 140],
        "difference_steps": ["1e-20", "1e-30"],
        "maximum_absolute_crosscheck_difference": str(maximum),
        "parameter_references": parameters,
        "coarse_cases": coarse,
        "cases": refined,
        "limits": "Finite-panel cross-check, not a formal derivative-error bound or whole-domain accuracy proof.",
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
