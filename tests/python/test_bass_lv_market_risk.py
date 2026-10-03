"""End-to-end IV quote recalibration, units and paired uncertainty."""
import math
import unittest

import rust_pricing as rp


class BassMarketIvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market = rp.BassMarketIvModel.calibrate(
            100.0, [0.5, 1.0], [-2.0, 0.0, 2.0], [0.2]*6,
            [-2.0+4.0*i/800 for i in range(801)],
            config=rp.BassLvConfig(grid_points=401, cdf_tolerance=1e-7),
        )
        cls.risk = cls.market.compile_vega_kt([0.25, 0.75, 1.0])

    def test_metadata_price_and_unit_contract(self):
        r = self.risk
        a = r.price_asian(100.0, paths=2000, seed=42)
        b = r.simulation.price_asian(100.0, paths=2000, seed=42)
        self.assertEqual(a.estimate.price, b.price)
        self.assertEqual(a.estimate.standard_error, b.standard_error)
        self.assertEqual(a.method, "recalibrated-central-crn-v1")
        self.assertEqual(a.maturity_nodes, [0.5, 1.0])
        self.assertEqual(a.log_moneyness_nodes, [-2.0, 0.0, 2.0])
        self.assertEqual(a.implied_volatilities, [0.2]*6)
        self.assertEqual(a.bump_size, 1e-4)
        self.assertEqual(a.vega_per_vol_point, [v*0.01 for v in a.sensitivities])
        self.assertEqual(a.standard_errors_per_vol_point, [v*0.01 for v in a.standard_errors])
        self.assertAlmostEqual(a.bucket_sum, sum(a.sensitivities), places=10)
        self.assertGreater(a.bucket_sum_standard_error, 0.0)
        self.assertGreater(a.parallel_standard_error, 0.0)
        self.assertEqual(len(r.diagnostics), 14)
        for j, d in enumerate(r.diagnostics):
            self.assertEqual(d.quote_index, j//2 if j < 12 else None)
            self.assertEqual(d.shift, 1e-4 if j % 2 == 0 else -1e-4)
            self.assertEqual(len(d.calibration), 2)
            self.assertEqual(len(d.projection), 2)
            self.assertTrue(all(c.cdf_residual <= 1e-7 for c in d.calibration))
        self.assertEqual(a.sensitivities, r.price_asian(100.0, paths=2000, seed=42).sensitivities)
        with self.assertRaises(AttributeError):
            a.bump_size = 0.01

    def test_manual_node_and_parallel_recalibration(self):
        a = self.risk.price_european(100.0, paths=500, seed=9, discount_factor=0.95)
        for j in [1, 4, None]:
            shifts = [1e-4 if j is None or k == j else 0.0 for k in range(6)]
            up = self.market.bumped(shifts).model.compile_simulation([0.25, 0.75, 1.0])
            dn = self.market.bumped([-v for v in shifts]).model.compile_simulation([0.25, 0.75, 1.0])
            u = up.price_european(100.0, paths=500, seed=9, discount_factor=0.95)
            d = dn.price_european(100.0, paths=500, seed=9, discount_factor=0.95)
            expected = a.parallel_sensitivity if j is None else a.sensitivities[j]
            self.assertAlmostEqual(expected, (u.price-d.price)/2e-4, places=8)
        self.assertEqual(self.market.model.spot, 100.0)
        self.assertEqual(len(self.market.projection_diagnostics), 2)

    def test_invalid_inputs_and_scenario_failure(self):
        for bump in [0.0, math.nan, -1.0, 0.2, 1e-30]:
            with self.assertRaises(rp.ValidationError):
                self.market.compile_vega_kt([1.0], bump_size=bump)
        with self.assertRaisesRegex(rp.PricingError, r"quote\[.*shift"):
            self.market.compile_vega_kt([1.0], bump_size=0.15)
        with self.assertRaises(rp.ValidationError):
            self.market.bumped([0.0])
        with self.assertRaises(rp.ValidationError):
            self.risk.price_european(100.0, paths=1)
        with self.assertRaises(rp.ValidationError):
            rp.BassMarketIvModel.calibrate(100.0, [0.5, 1.0], [-2.0, 0.0, 2.0],
                                          [0.2]*5, [-2.0+4*i/100 for i in range(101)])


if __name__ == "__main__":
    unittest.main()
