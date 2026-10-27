"""Print exact eSSVI parameter references for the unequal-interval test panel.

These closed-form affine polynomials are solved independently for knots at
2, 4 and 8. Fraction arithmetic does not use Rust's interpolation implementation.
The Rust test freezes the resulting binary64-exact literals.
"""

from fractions import Fraction as F
import json


def generate() -> dict:
    cases = []
    for text in ("0.5", "2", "2.5", "3.5", "4", "5", "7", "8", "10"):
        time = F(text)
        if time < 2:
            values = [time / 8, time / 8, -time / 16]
            slopes = [F(1, 8), F(1, 8), -F(1, 16)]
            region = "ShortExtrapolated"
        elif time < 4:
            values = [time / 4 - F(1, 4), time / 8, time / 16 - F(1, 4)]
            slopes = [F(1, 4), F(1, 8), F(1, 16)]
            region = "Knot" if time == 2 else "Interpolated"
        elif time < 8:
            values = [time / 16 + F(1, 2), time / 8, F(1, 4) - time / 16]
            slopes = [F(1, 16), F(1, 8), -F(1, 16)]
            region = "Knot" if time == 4 else "Interpolated"
        else:
            values = [time / 32 + F(3, 4), F(1), -F(1, 4)]
            slopes = [F(1, 32), F(0), F(0)]
            region = "Knot" if time == 8 else "LongExtrapolated"
        expected = values + slopes
        assert F.from_float(float(time)) == time
        assert all(F.from_float(float(value)) == value for value in expected)
        cases.append({
            "time": float(time),
            "exact_fractions": list(map(str, expected)),
            "expected": list(map(float, expected)),
            "region": region,
        })
    return {
        "method": "Exact Fraction evaluation of independently solved affine polynomials.",
        "fields": ["theta", "psi", "rho_psi", "theta_t", "psi_t", "rho_psi_t"],
        "slices": [[2.0, 0.25, 0.25, -0.125], [4.0, 0.75, 0.5, 0.0], [8.0, 1.0, 1.0, -0.25]],
        "terminal_theta_slope": 0.03125,
        "cases": cases,
        "limits": "Finite regular-case references; not an overflow/underflow guarantee for all finite queries.",
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
