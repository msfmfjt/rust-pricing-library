"""Public pure-Bergomi/cash-dividend combination, price-only scope."""
import math
import unittest

import rust_pricing as rp
from test_stochastic_dividends import make_request
from bergomi_dividend_reference import reference


def compile_plan(two=False, request=None, **changes):
    settings = dict(vol_of_vol=.3, dividend_mean_reversion=.7,
                    equity_linkage=.6, dividend_volatility=.35,
                    equity_dividend_correlation=-.25, maximum_step=.5,
                    worker_threads=1, reduction_block_size=64)
    if two:
        settings.update(mean_reversions=[.8, 2.1], mixing_weight=.35,
                        spot_correlations=[-.4, -.2], factor_correlation=.3,
                        dividend_volatility_correlations=[.15, -.1])
        factory = rp.StochasticDividendPlan.compile_bergomi_two_factor
    else:
        settings.update(mean_reversion=.8, correlation=-.4,
                        dividend_volatility_correlation=.15)
        factory = rp.StochasticDividendPlan.compile_bergomi
    return factory(request or make_request(points=2048), **(settings | changes))


class BergomiDividendTest(unittest.TestCase):
    def test_independent_conditional_black_oracle_and_metadata(self):
        for two in (False, True):
            with self.subTest(two=two):
                expected = reference(16, two)
                self.assertLess(abs(expected-reference(12, two)), 5e-6)
                plan = compile_plan(two)
                result = plan.evaluate()
                self.assertLess(abs(result.value-expected), 6*result.standard_error+.002)
                self.assertEqual(plan.random_factor_count, 4 if two else 3)
                self.assertEqual(result.scheme, plan.scheme)
                self.assertEqual(result.uncertainty_scope, 'pricing_only')
                self.assertEqual(result.independent_sampling_units, 8)
                self.assertEqual(result.evaluated_paths, 32768)
                self.assertEqual(result.plan_fingerprint, plan.plan_fingerprint)

    def test_workers_and_fingerprint(self):
        for two in (False, True):
            r = make_request(points=64)
            a = compile_plan(two, r, maximum_step=.125)
            b = compile_plan(two, r, maximum_step=.125, worker_threads=3)
            x, y = a.evaluate(), b.evaluate()
            self.assertEqual(x.value, y.value)
            self.assertEqual(x.standard_error, y.standard_error)
            changed = compile_plan(two, r, maximum_step=.125, vol_of_vol=.4)
            self.assertNotEqual(a.plan_fingerprint, changed.plan_fingerprint)
            with self.assertRaises(AttributeError):
                a.scheme = 'changed'

    def test_fixed_cash_and_zero_stock_volatility(self):
        for two in (False, True):
            r = make_request(sigma=0, dividends=((1., 10.),), strike=80., points=16)
            result = compile_plan(two, r, equity_linkage=0., dividend_volatility=0.).evaluate()
            self.assertAlmostEqual(result.value, .95*(100*.98/.95-10-80), delta=1e-13)
            self.assertLess(result.standard_error, 1e-13)

    def test_psd_and_risk_rejected_before_sampling(self):
        with self.assertRaises(rp.PricingError):
            compile_plan(equity_dividend_correlation=.9, correlation=.9,
                         dividend_volatility_correlation=-.9, vol_of_vol=0.)
        for two in (False, True):
            with self.assertRaises(rp.PricingError):
                compile_plan(two, make_request(risk=rp.RiskRequest(delta=True)))
            self.assertTrue(hasattr(compile_plan(two, make_request(points=16)), 'evaluate_aad'))
            with self.assertRaises(rp.ValidationError):
                compile_plan(two, dividend_mean_reversion=math.nan)


if __name__ == '__main__':
    unittest.main()
