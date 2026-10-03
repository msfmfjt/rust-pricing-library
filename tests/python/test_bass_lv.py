import math
import unittest
import rust_pricing as rp


class BassLvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.marginals = [rp.BassMarginal.lognormal(t, 100.0, 0.2) for t in (0.5, 1.0, 2.0)]
        cls.model = rp.BassLvModel.calibrate(100.0, cls.marginals)

    def test_public_api_and_diagnostics(self):
        self.assertEqual(self.model.spot, 100.0)
        self.assertEqual(len(self.model.diagnostics), 3)
        for d in self.model.diagnostics:
            self.assertLessEqual(d.cdf_residual, 1e-6)
            self.assertLess(d.marginal_cdf_error, 3e-5)
        self.assertAlmostEqual(self.model.local_volatility(0.75, 100.0), 0.2, delta=0.001)
        self.assertLess(abs(self.model.initial_spot_error), 0.001)
        self.assertAlmostEqual(self.marginals[0].mean, 100.0)
        for p in (0.01, 0.5, 0.99):
            self.assertAlmostEqual(self.marginals[0].cdf(self.marginals[0].quantile(p)), p)
        with self.assertRaises(AttributeError):
            self.model.spot = 1.0

    def test_path_grid_replay_and_pricing(self):
        p = self.model.compile_simulation([0.0, 0.75, 2.0])
        self.assertEqual(p.time_nodes, [0.0, 0.5, 0.75, 1.0, 2.0])
        self.assertEqual(p.normal_count, 4)
        paths = p.sample_paths(4, seed=9)
        self.assertEqual(paths, p.sample_paths(4, seed=9))
        self.assertEqual(len(paths[0]), 3)
        self.assertEqual(paths[0][0], 100.0)
        self.assertNotEqual(paths, p.sample_paths(4, seed=10))
        self.assertEqual(len(p.path_from_normals([0.0] * 4)), 3)
        call = p.price_european(100.0, paths=30_000, seed=735)
        put = p.price_european(100.0, is_call=False, paths=30_000, seed=735)
        reference = self.marginals[-1].call_price(100.0)
        self.assertLess(abs(call.price-reference), 5*call.standard_error+0.02)
        self.assertLess(abs(call.price-put.price), 5*(call.standard_error+put.standard_error))
        discounted = p.price_european(100.0, paths=30_000, seed=735, discount_factor=0.95)
        self.assertAlmostEqual(discounted.price, 0.95*call.price, places=10)
        one = self.model.compile_simulation([1.0])
        self.assertEqual(one.price_european(100.0, paths=100, seed=5).price,
                         one.price_asian(100.0, paths=100, seed=5).price)

    def test_asian_uses_requested_fixings_only(self):
        p = self.model.compile_simulation([0.75, 2.0])
        paths = p.sample_paths(300, seed=3)
        expected = sum(max(sum(s)/2-100, 0) for s in paths)/len(paths)
        self.assertAlmostEqual(p.price_asian(100.0, paths=300, seed=3).price, expected)

    def test_invalid_inputs_and_calibration_failures(self):
        with self.assertRaises(rp.ValidationError):
            rp.BassLvModel.calibrate(101.0, self.marginals)
        with self.assertRaises(rp.ValidationError):
            rp.BassLvModel.calibrate(100.0, self.marginals, config=rp.BassLvConfig(grid_points=100))
        with self.assertRaises(rp.PricingError):
            rp.BassLvModel.calibrate(100.0, self.marginals,
                                     config=rp.BassLvConfig(max_iterations=1, cdf_tolerance=1e-12))
        with self.assertRaises(rp.ValidationError):
            self.model.compile_simulation([1.0, 0.5])
        with self.assertRaises(rp.ValidationError):
            self.marginals[0].quantile(math.nan)
        with self.assertRaises(rp.ValidationError):
            self.model.compile_simulation([1.0]).price_european(100.0, paths=1)
        with self.assertRaises(rp.PricingError):
            self.model.compile_simulation([1.0]).path_from_normals([100.0, 0.0])


class BassLvExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slices = [rp.EssviSlice(0.5, 0.02, 0.06, -0.025),
                      rp.EssviSlice(1.0, 0.04, 0.09, -0.035),
                      rp.EssviSlice(2.0, 0.08, 0.13, -0.045)]
        cls.nodes = [-4.0 + 8.0*i/1600 for i in range(1601)]
        cls.projections = [rp.BassMarginal.from_essvi(t, 100.0, cls.slices, cls.nodes)
                           for t in [0.5, 1.0, 2.0]]
        cls.model = rp.BassLvModel.calibrate(100.0, [p.marginal for p in cls.projections])
        cls.bumps = [rp.BassMappingBump(i, -1.0, 0.0, 1.0) for i in range(3)]
        cls.risk = cls.model.compile_mapping_risk([0.25, 0.75, 1.5, 2.0], cls.bumps)

    def test_surface_projection_and_validation(self):
        for p in self.projections:
            self.assertAlmostEqual(p.marginal.mean, 100.0, places=10)
            self.assertLess(p.diagnostics.max_call_price_error, 0.002)
            self.assertLess(abs(p.diagnostics.mean_scale-1), 1e-5)
            self.assertGreater(p.diagnostics.retained_nodes, 1000)
        with self.assertRaises(rp.ValidationError):
            rp.BassMarginal.from_essvi(1.0, 100.0, self.slices,
                                      [-0.2+0.4*i/100 for i in range(101)])
        with self.assertRaises(rp.ValidationError):
            rp.BassMarginal.from_essvi(1.0, 100.0, self.slices, self.nodes,
                                      relative_mean_tolerance=1e-12)

    def test_mapping_risk_mc_and_bump_parity(self):
        r = self.risk
        a = r.price_asian(100.0, paths=2000, seed=38)
        self.assertEqual(a.estimate.price,
                         r.simulation.price_asian(100.0, paths=2000, seed=38).price)
        self.assertEqual(len(a.sensitivities), 3)
        self.assertTrue(all(v >= 0 for v in a.standard_errors))
        for j in range(3):
            b = [0.0]*3
            b[j] = 1e-4
            up = r.bumped_simulation(b).price_asian(100.0, paths=2000, seed=38).price
            b[j] = -1e-4
            dn = r.bumped_simulation(b).price_asian(100.0, paths=2000, seed=38).price
            self.assertAlmostEqual(a.sensitivities[j], (up-dn)/2e-4, delta=5e-5)
        with self.assertRaises(rp.ValidationError):
            r.bumped_simulation([1000.0]*3)

    def test_semianalytic_risk_and_path_partials(self):
        r = self.risk
        a = r.vanilla_call(2, 100.0)
        self.assertLess(abs(a.price-self.projections[-1].marginal.call_price(100.0)), 0.01)
        for j in range(3):
            b = [0.0]*3
            b[j] = 1e-5
            up = r.bumped_vanilla_call(b, 2, 100.0)
            b[j] = -1e-5
            dn = r.bumped_vanilla_call(b, 2, 100.0)
            self.assertAlmostEqual(a.sensitivities[j], (up-dn)/2e-5, delta=3e-6)
        self.assertEqual(r.vanilla_call(0, 100.0).sensitivities[1:], [0.0, 0.0])
        g = r.path_sensitivities([0.2]*r.simulation.normal_count, [0.25]*4)
        self.assertEqual(len(g), 3)
        self.assertTrue(all(math.isfinite(x) for x in g))
        with self.assertRaises(AttributeError):
            self.bumps[0].center = 1.0


if __name__ == "__main__":
    unittest.main()
