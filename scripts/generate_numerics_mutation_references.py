"""Regenerate independent references used by the numerical mutation tests.

Optional development tool: requires mpmath 1.3.0; Rust tests embed the rounded
values and have no Python dependency. This prints references, never test results.
"""

import json

import mpmath as mp


def references():
    with mp.workdps(80):
        normal = []
        for text in ("0.5", "2.5", "7", "7.1", "8", "10", "20", "37"):
            x = mp.mpf(text)
            normal.append({
                "x": text,
                "pdf": mp.nstr(mp.exp(-x * x / 2) / mp.sqrt(2 * mp.pi), 40),
                "cdf_negative_x": mp.nstr(mp.erfc(x / mp.sqrt(2)) / 2, 40),
            })
        fractional_ou = []
        for hurst, lag in (("0.1", "0.01"), ("0.25", "1"), ("0.4", "3"),
                           ("0.1", "10"), ("0.25", "64"), ("0.25", "80")):
            h, z = mp.mpf(hurst), mp.mpf(lag)
            q = 2 * h

            def covariance_integrand(u):
                return mp.exp(-u) * ((z + u) ** q + abs(z - u) ** q - 2 * z ** q)

            # Adaptive tanh-sinh quadrature on the infinite interval, splitting
            # at the cusp; no finite tail cutoff or sin^2 regularization.
            value = mp.quad(covariance_integrand, [0, z, mp.inf], method="tanh-sinh")
            value /= 2 * mp.gamma(q + 1)
            fractional_ou.append({"hurst": hurst, "lag": lag, "correlation": mp.nstr(value, 40)})
        return {"mpmath_version": mp.__version__, "decimal_precision": 80,
                "normal": normal, "fractional_ou": fractional_ou}


if __name__ == "__main__":
    print(json.dumps(references(), indent=2))
