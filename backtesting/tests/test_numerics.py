"""Checks for an independent oracle and fail-closed numerical reporting."""

import math
import unittest

from rust_pricing_backtest import BacktestError, RustPricingProvider
from rust_pricing_backtest.numerics import black_scholes_call, threshold_check


class NumericalPanelTests(unittest.TestCase):
    def test_oracle_against_standard_atm_call_reference(self):
        result = black_scholes_call(100., 100., 1., .2, .05)
        self.assertAlmostEqual(result["price"], 10.450583572185565, places=12)
        self.assertAlmostEqual(result["delta"], .6368306511756191, places=14)
        self.assertAlmostEqual(result["vega"], 37.52403469169379, places=12)

    def test_vega_oracle_matches_price_derivative_for_unequal_strikes(self):
        for strike in (80., 100., 120.):
            h = 1e-5
            up = black_scholes_call(100., strike, .4, .3+h, .02)["price"]
            down = black_scholes_call(100., strike, .4, .3-h, .02)["price"]
            self.assertAlmostEqual((up-down)/(2*h), black_scholes_call(100., strike, .4, .3, .02)["vega"], places=7)

    def test_nonfinite_measurements_cannot_pass_acceptance(self):
        for observed in (math.nan, math.inf, -math.inf):
            result = threshold_check("gate", observed, .03)
            self.assertFalse(result["passed"])
            self.assertIsNone(result["observed"])
        self.assertFalse(threshold_check("gate", .031, .03)["passed"])
        self.assertTrue(threshold_check("gate", .03, .03)["passed"])

    def test_native_provider_is_blocked_without_explicit_diagnostic_flag(self):
        # The block is checked before importing the optional native extension.
        for flag in (False, None, "true", 1):
            with self.assertRaisesRegex(BacktestError, "unvalidated_vega_kt"):
                RustPricingProvider(diagnostic_only=flag)


if __name__ == "__main__":
    unittest.main()
