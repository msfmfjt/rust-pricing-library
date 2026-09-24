"""Public rough/Buehler price-only factory and independent finite-grid oracle."""
import math
import unittest
import rust_pricing as rp
from test_stochastic_dividends import make_request
from rough_dividend_reference import reference


def compile_plan(request=None, **changes):
    settings = dict(hurst=.1, vol_of_vol=.6, correlation=-.4,
                    dividend_mean_reversion=.7, equity_linkage=.6,
                    dividend_volatility=.35, equity_dividend_correlation=-.25,
                    dividend_volatility_correlation=.15, maximum_step=.5,
                    worker_threads=1, reduction_block_size=64)
    return rp.StochasticDividendPlan.compile_rough_bergomi(
        request or make_request(points=2048), **(settings | changes))


class RoughDividendTest(unittest.TestCase):
    def test_independent_two_step_reference_and_metadata(self):
        expected = reference(20)
        self.assertLess(abs(expected-reference(24)), 1e-6)
        p = compile_plan()
        result = p.evaluate()
        self.assertLess(abs(result.value-expected), 6*result.standard_error+.002)
        self.assertEqual(p.random_factor_count, 4)
        self.assertEqual(result.scheme, 'buehler-rough-bergomi-joint-hybrid-positive-split-v1')
        self.assertEqual(result.independent_sampling_units, 8)
        self.assertEqual(result.evaluated_paths, 32768)
        self.assertEqual(result.plan_fingerprint, p.plan_fingerprint)

    def test_workers_fixed_cash_and_identity(self):
        r = make_request(points=64)
        a = compile_plan(r, maximum_step=.125)
        b = compile_plan(r, maximum_step=.125, worker_threads=3)
        x, y = a.evaluate(), b.evaluate()
        self.assertEqual(x.value, y.value)
        self.assertEqual(x.standard_error, y.standard_error)
        for change in ({'hurst': .2}, {'vol_of_vol': .7}, {'correlation': -.3},
                       {'dividend_volatility_correlation': .2}):
            self.assertNotEqual(a.plan_fingerprint, compile_plan(r, maximum_step=.125, **change).plan_fingerprint)
        with self.assertRaises(AttributeError):
            a.scheme = 'changed'
        result = compile_plan(make_request(sigma=0, dividends=((1.,10.),),
            strike=80., points=16), equity_linkage=0., dividend_volatility=0.).evaluate()
        self.assertAlmostEqual(result.value, .95*(100*.98/.95-10-80), delta=1e-13)
        self.assertLess(result.standard_error, 1e-13)

    def test_invalid_inputs_psd_and_risk_rejected(self):
        for h in [0., -.1, .51, math.nan, math.inf]:
            with self.assertRaises(rp.ValidationError):
                compile_plan(hurst=h)
        with self.assertRaises(rp.PricingError):
            compile_plan(equity_dividend_correlation=.9, correlation=.9,
                         dividend_volatility_correlation=-.9, vol_of_vol=0.)
        with self.assertRaises(rp.PricingError):
            compile_plan(make_request(risk=rp.RiskRequest(delta=True)))
        p = compile_plan(make_request(points=16))
        baseline = p.evaluate().value
        p.evaluate_aad()
        p.evaluate_rough_aad()
        for method in (p.evaluate_bergomi_aad, p.evaluate_correlation_aad):
            with self.assertRaises(rp.PricingError):
                method()
        p.evaluate_gamma(gamma_absolute_bump=1.)
        self.assertEqual(p.evaluate().value, baseline)


if __name__ == '__main__':
    unittest.main()
