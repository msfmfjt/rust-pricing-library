"""Print independently differenced SSVI references using Python's Decimal.

Only the scalar variance formula is evaluated. Derivatives come from centered
differences in high precision, not the production analytic derivative formulas.
The two runs vary both precision and step size; Rust consumes frozen literals.
"""

from decimal import Decimal, localcontext
import json


CASES = [
    # family, eta/lambda, gamma (unused for Heston), rho, time, theta, theta_t, k
    ("power", 0.75, 0.25, -0.5, 0.25, 0.0625, 0.25, -1.0),
    ("power", 0.75, 0.25, -0.5, 1.0, 0.25, 0.25, 0.5),
    ("power", 0.75, 0.25, 0.5, 1.5, 0.375, 0.25, -0.5),
    ("power", 1.0, 0.5, 0.0, 2.0, 0.5, 0.25, 2.0),
    ("heston", 1.0, 0.0, -0.5, 1.75, 0.4375, 0.25, 0.5),
    ("heston", 2.0, 0.0, 0.5, 3.5, 0.6875, 0.125, -2.0),
    ("heston", 4.0, 0.0, 0.0, 1.0, 0.25, 0.25, 1.0),
    ("heston", 0.5, 0.0, 0.5, 0.25, 0.0625, 0.25, -0.5),
]


def integrated_phi(z: Decimal, digits: int) -> Decimal:
    """Integral_0^1 (1-u)*exp(-z*u) du, via integrated exponential series."""
    term = Decimal(1)
    result = Decimal(0)
    for order in range(1000):
        contribution = term / ((order + 1) * (order + 2))
        result += contribution
        if order > z and abs(contribution) < Decimal(10) ** (-digits + 5):
            return result
        term *= -z / (order + 1)
    raise AssertionError("Integrated exponential series did not converge")


def evaluate(case: tuple, digits: int, step_exponent: int) -> dict:
    family, *inputs = case
    with localcontext() as context:
        context.prec = digits
        parameter, gamma, rho, time, theta, theta_t, k = map(Decimal.from_float, inputs)

        def scalar(variance: Decimal, moneyness: Decimal) -> Decimal:
            if family == "power":
                phi = (parameter.ln() - gamma * variance.ln()
                       - (1 - gamma) * (1 + variance).ln()).exp()
            else:
                phi = integrated_phi(parameter * variance, digits)
            y = phi * moneyness
            # Expanded radicand, independently of the production (y+rho)^2 form.
            shape = 1 + rho * y + (1 + 2 * rho * y + y * y).sqrt()
            return variance * shape / 2

        h = Decimal(10) ** step_exponent
        value = scalar(theta, k)
        up, down = scalar(theta, k + h), scalar(theta, k - h)
        derivative_k = (up - down) / (2 * h)
        derivative_kk = (up + down - 2 * value) / (h * h)
        # Local affine theta path uses the specified knot derivative. At the
        # terminal knot this is the interpolator's reported derivative, not a
        # centered difference crossing the change in extrapolation slope.
        derivative_t = (scalar(theta + h * theta_t, k)
                        - scalar(theta - h * theta_t, k)) / (2 * h)
        return {
            "family": family,
            "inputs": dict(zip(("parameter", "gamma", "rho", "time", "theta", "theta_t", "k"), inputs)),
            "expected": dict(zip(("w", "w_k", "w_kk", "w_t"),
                                 map(str, (value, derivative_k, derivative_kk, derivative_t)))),
        }


def generate() -> dict:
    coarse = [evaluate(case, 100, -20) for case in CASES]
    refined = [evaluate(case, 140, -30) for case in CASES]
    with localcontext() as context:
        context.prec = 140
        differences = [abs(Decimal(left["expected"][field]) - Decimal(right["expected"][field]))
                       for left, right in zip(coarse, refined, strict=True)
                       for field in left["expected"]]
        maximum = max(differences)
        assert maximum < Decimal("1e-37"), maximum
    return {
        "method": "Centered finite differences of scalar total variance; independent logarithmic/integrated-series phi.",
        "precision_digits": [100, 140],
        "difference_steps": ["1e-20", "1e-30"],
        "maximum_absolute_crosscheck_difference": str(maximum),
        "limits": "Agreement of these finite regular-case panels is not a formal derivative-error bound or whole-domain accuracy proof.",
        "coarse_cases": coarse,
        "cases": refined,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
