"""Stopping-time FX conversion and out-of-sample exercise under two short rates."""
import unittest

import rust_pricing as rp
from test_cross_currency_models import models
import test_cross_currency_rates as rates
from test_cross_currency_rates import C, TODAY, MID, END


class JointExerciseTests(unittest.TestCase):
    def setUp(self):
        self.inputs = rates.CrossCurrencyRateTests()
        self.inputs.setUp()
        self.mc = rp.Engine.pseudo_monte_carlo(723, 384, antithetic=True, brownian_bridge=True)
        training = rp.Engine.pseudo_monte_carlo(91, 384, antithetic=True, brownian_bridge=True)
        self.states = ["spot", "domestic_rate_factor", "foreign_rate_factor"]
        self.config = rp.LsmConfig(training, state_variables=self.states, max_degree=2)

    def option(self, product, mode, model=None):
        return self.inputs.option(product, mode, model=model).with_hull_white_rates(
            self.inputs.d, self.inputs.f, correlations=C)

    def test_only_expiry_exercise_replicates_european_on_the_same_paths(self):
        for name, model in models(True):
            for mode in ("quanto", "floating_fx", "composite"):
                with self.subTest(model=name, mode=mode):
                    strike = 15000. if mode == "composite" else 100.
                    bermudan = rp.Product.american_vanilla(10, 2, END, strike, 1., "put", [END])
                    vanilla = rp.Product.european_vanilla(10, 2, END, strike, 1., "put")
                    e = self.option(vanilla, mode, model).price_mc(self.mc)
                    b = self.option(bermudan, mode, model).price_mc(self.mc, lsm=self.config)
                    self.assertAlmostEqual(b.value, e.value, places=10)
                    self.assertAlmostEqual(b.standard_error, e.standard_error, places=10)

    def test_exercise_today_fixes_fx_today_and_consumes_principal(self):
        p = rp.Product.american_vanilla(10, 2, END, 500., 1., "put", [TODAY, MID, END])
        qmc = rp.Engine.randomized_quasi_monte_carlo(128, 704, scramble_count=4, antithetic=True)
        lsm = rp.LsmConfig(qmc, state_variables=["spot", "fx_spot"], max_degree=2)
        for mode, expected in [("floating_fx", 60000.), ("quanto", 58000.)]:
            opt = self.option(p, mode)
            e = opt.price_mc(self.mc, lsm=lsm)
            self.assertAlmostEqual(e.value, expected)
            self.assertEqual(e.standard_error, 0.)
            delta = opt.sensitivity("fx_spot", 1, .5, self.mc, lsm=self.config, replications=3)
            self.assertEqual(delta.value, 400. if mode == "floating_fx" else 0.)
            self.assertEqual(delta.standard_error, 0.)
            with self.assertRaises(rp.PricingError):
                opt.price_mc(qmc, lsm=lsm)
            with self.assertRaises(rp.PricingError):
                opt.price_mc(self.mc)

    def test_deterministic_rate_local_vol_exercise_matches_foreign_value(self):
        p = rp.Product.american_vanilla(10, 2, END, 110., 1., "put", [MID, END])
        flat = dict(models(False))["LV"]
        vanilla_engine = self.inputs.option(p, "floating_fx").price_mc(self.mc, lsm=rp.LsmConfig(self.mc))
        live = self.inputs.option(p, "floating_fx", model=flat).price_mc(self.mc, lsm=self.config)
        self.assertAlmostEqual(live.value, vanilla_engine.value,
                               delta=5.*(live.standard_error+vanilla_engine.standard_error))
