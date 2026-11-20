"""Exact rational witnesses for the eSSVI validation boundary tests.

Residuals use psi + abs(rho_psi) and psi*(psi + abs(rho_psi)), avoiding
the production rho division. Adjacent inputs are exact binary64 neighbors.
This finite panel does not model all intermediate floating-point arithmetic.
"""

from fractions import Fraction as F
import json
import math


def effective(left: F, right: F, absolute: F, relative: F) -> F:
    return max(absolute, relative * max(F(1), abs(left), abs(right)))


def generate() -> dict:
    panels = {"wing": [], "curvature": [], "correlation_change": []}
    inputs = {
        "wing": [(0.0, 0.0, 4.0), (0.125, 0.0, 3.875), (0.0, 0.03125, 3.875)],
        "curvature": [(0.0, 0.0, 0.375), (0.125, 0.0, 0.34375), (0.0, 0.0625, 0.3515625)],
        "correlation_change": [(0.0, 0.0, 0.5), (0.0625, 0.0, 0.5625), (0.0, 0.0625, 0.5625)],
    }
    for family, cases in inputs.items():
        for absolute, relative, boundary in cases:
            rows = []
            for query in (math.nextafter(boundary, -math.inf), boundary,
                          math.nextafter(boundary, math.inf)):
                x, a, r = map(F.from_float, (query, absolute, relative))
                if family == "wing":
                    left, right = x, F(4)
                elif family == "curvature":
                    left, right = F(3, 2), 4 * x
                else:
                    left, right = x - F(1, 4), F(1, 4)
                tolerance = effective(left, right, a, r)
                accepted = right - left > tolerance if family == "wing" else left - right <= tolerance
                assert F.from_float(float(left)) == left
                assert F.from_float(float(right)) == right
                rows.append({"query": query, "left": float(left), "right": float(right),
                             "left_fraction": str(left), "right_fraction": str(right),
                             "effective_tolerance_fraction": str(tolerance), "accepted": accepted})
            assert [row["accepted"] for row in rows] == (
                [True, False, False] if family == "wing" else
                [False, True, True] if family == "curvature" else [True, True, False]
            )
            panels[family].append({"absolute": absolute, "relative": relative,
                                   "boundary": boundary, "neighbors": rows})

    residuals = []
    for theta, psi, rho_psi in [(2.0, 3.0, 1.5), (0.25, 1.5, 0.75),
                              (0.0625, 0.5, 0.125), (0.0625, 4.0, 2.0)]:
        t, p, c = map(F.from_float, (theta, psi, rho_psi))
        wing = p + abs(c)
        curvature = p * (p + abs(c))
        limit = 4 * t
        condition = "strict_wing_slope" if wing >= 4 else "curvature_bound"
        left, right = (wing, F(4)) if condition == "strict_wing_slope" else (curvature, limit)
        assert condition == "strict_wing_slope" or curvature > limit
        residuals.append({"theta": theta, "psi": psi, "rho_psi_magnitude": rho_psi,
                          "wing_fraction": str(wing), "curvature_fraction": str(curvature),
                          "limit_fraction": str(limit), "condition": condition,
                          "left": float(left), "right": float(right)})
    return {"method": "Exact Fraction arithmetic for dyadic boundary neighbors and independent residual expressions.",
            "panels": panels, "residuals": residuals,
            "limits": "Finite exact witnesses, not a whole-domain numerical proof; nonfinite cases are separate public API tests."}


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
