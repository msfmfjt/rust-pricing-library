"""Generate independent SSVI forward-call references with Decimal arithmetic.

The Gaussian CDF is integrated by its Taylor series, with pi from Machin's
arctangent identity. No production Rust normal approximation is reused.
Binary64 inputs are lifted exactly; 100- and 140-digit runs cross-check rounding.
This optional generator needs only Python's standard library.
"""

from decimal import Decimal, localcontext
import json


CASES = [
    (0.25, 0.125, 0.0625, 0.5, 128.0),
    (0.25, 0.125, 0.0625, -0.5, 128.0),
    (1.0, -0.25, 0.5, 0.25, 1.0),
    (4.0, 0.5, -0.25, 1.0, 0.125),
    (0.0625, -0.0625, 0.125, -0.25, 4.0),
    (0.5, 0.0, 0.5, 2.0, 16.0),
    (0.25, 0.0, 0.0, 0.0, 100.0),
    (1.0, 0.0, 0.0, -2.0, 1.0),
]


def arctangent_reciprocal(denominator: int, cutoff: Decimal) -> Decimal:
    x = Decimal(1) / denominator
    power = x
    result = Decimal(0)
    for n in range(1000):
        term = power / (2 * n + 1)
        result += term
        if abs(term) < cutoff:
            return result
        power *= -x * x
    raise ArithmeticError("Arctangent series did not converge")


def normal_cdf(x: Decimal, normalization: Decimal, cutoff: Decimal) -> Decimal:
    # Integral from zero to x of exp(-u*u/2), divided by sqrt(2*pi).
    power = x
    result = Decimal(0)
    for n in range(1000):
        term = power / (2 * n + 1)
        result += term
        if n > x * x and abs(term) < cutoff:
            return Decimal('0.5') + result / normalization
        power *= -x * x / (2 * (n + 1))
    raise ArithmeticError("Gaussian integral series did not converge")


def calculate(precision: int) -> list[dict]:
    with localcontext() as context:
        context.prec = precision
        cutoff = Decimal(10) ** (-precision + 5)
        pi = (16 * arctangent_reciprocal(5, cutoff)
              - 4 * arctangent_reciprocal(239, cutoff))
        normalization = (2 * pi).sqrt()
        rows = []
        for inputs in CASES:
            w, slope, curvature, k, forward = map(Decimal.from_float, inputs)
            root = w.sqrt()
            strike = forward * k.exp()
            d1 = (w / 2 - k) / root
            d2 = -(w / 2 + k) / root
            price = (forward * normal_cdf(d1, normalization, cutoff)
                     - strike * normal_cdf(d2, normalization, cutoff))
            g = (1 - k * slope / w + k * k * slope * slope / (4 * w * w)
                 - slope * slope / (4 * w) - slope * slope / 16 + curvature / 2)
            density = (-d2 * d2 / 2).exp() * g / (normalization * strike * root)
            assert price > 0 and density > 0
            rows.append({
                "inputs": dict(zip(("variance", "slope", "curvature", "log_moneyness", "forward"),
                                   map(str, (w, slope, curvature, k, forward)))),
                "input_hex": [value.hex() for value in inputs],
                "expected": dict(zip(("strike", "price", "density_factor", "call_density", "d1", "d2"),
                                     map(str, (strike, price, g, density, d1, d2)))),
            })
        return rows


def references() -> dict:
    rows = calculate(100)
    refined = calculate(140)
    with localcontext() as context:
        context.prec = 150
        differences = [abs(Decimal(row["expected"][key]) - Decimal(other["expected"][key]))
                       for row, other in zip(rows, refined, strict=True)
                       for key in row["expected"]]
        assert max(differences) < Decimal('1e-90')
    return {
        "method": "Analytic lognormal payoff and density identities; Gaussian CDF from integrated Taylor series; pi from Machin's identity.",
        "precision": 100, "cross_check_precision": 140,
        "cross_check_max_absolute_difference": str(max(differences)),
        "input_conversion": "Decimal.from_float preserves the exact binary64 input.",
        "cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(references(), indent=2))
