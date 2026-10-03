"""Verify the hard Barrier oracle independently of the compiled Python module."""

import json
import math
from pathlib import Path
import unittest

from rough_dividend_barrier_reference import (
    bivariate_cdf, bivariate_partials, cdf, price_by_terminal_conditioning, price_delta,
)


class HardBarrierReference(unittest.TestCase):
    def test_bivariate_probabilities_limits_and_partials(self):
        for rho in (0.0, 0.3, math.sqrt(0.5)):
            self.assertAlmostEqual(float(bivariate_cdf(0, 0, rho)),
                                   0.25 + math.asin(rho) / (2 * math.pi), delta=2e-14)
            self.assertEqual(float(bivariate_cdf(-math.inf, 0.7, rho)), 0.0)
            self.assertEqual(float(bivariate_cdf(math.inf, 0.7, rho)), float(cdf(0.7)))
            for a, b in ((-2.0, 0.3), (0.5, 1.2), (2.0, -1.0)):
                da, db = bivariate_partials(a, b, rho)
                bump = 1e-5
                fd_a = (bivariate_cdf(a + bump, b, rho) - bivariate_cdf(a - bump, b, rho)) / (2 * bump)
                fd_b = (bivariate_cdf(a, b + bump, rho) - bivariate_cdf(a, b - bump, rho)) / (2 * bump)
                self.assertAlmostEqual(float(da), float(fd_a), delta=2e-10)
                self.assertAlmostEqual(float(db), float(fd_b), delta=2e-10)

    def test_fixture_and_two_independent_equity_integrations(self):
        fixture = json.loads((Path(__file__).resolve().parents[2] / "fixtures/stochastic-dividends/rough-barrier-reference.json").read_text())
        expected = (fixture["price"], fixture["delta"])
        for n, cdf_order in ((32, 32), (64, 64), (96, 64), (128, 96)):
            for actual, reference in zip(price_delta(n, cdf_order=cdf_order), expected):
                self.assertAlmostEqual(actual, reference, delta=fixture["quadrature_agreement_tolerance"])
        for n, inner in ((24, 64), (32, 96)):
            self.assertAlmostEqual(price_by_terminal_conditioning(n, inner), expected[0], delta=2e-8)

    def test_hard_delta_includes_moving_monitoring_boundaries(self):
        for spot in (95.0, 100.0, 105.0):
            delta = price_delta(64, spot=spot)[1]
            for bump in (0.005, 0.0025):
                # Use the separate conditional-terminal integral for the check.
                fd = (price_by_terminal_conditioning(32, 96, spot=spot + bump)
                      - price_by_terminal_conditioning(32, 96, spot=spot - bump)) / (2 * bump)
                self.assertAlmostEqual(delta, fd, delta=3e-8)


if __name__ == "__main__":
    unittest.main()
