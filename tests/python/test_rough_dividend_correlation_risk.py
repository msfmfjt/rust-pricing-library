"""Raw Brownian-correlation risk; no correlation projection or recalibration."""
import math
import unittest
import rust_pricing as rp
from test_stochastic_dividends import make_request
from test_rough_dividends import compile_plan


class RoughDividendCorrelationRiskTest(unittest.TestCase):
    def test_three_entry_fds_and_exact_hurst_eta_prefix(self):
        request = make_request(points=64, dividends=((.5, 4.), (1., 3.), (1.4, 8.)))
        for hurst in (.1, .49, .5):
            settings = dict(hurst=hurst, maximum_step=.125)
            plan = compile_plan(request, **settings)
            prefix = plan.evaluate_rough_aad()
            risk = plan.evaluate_correlation_aad()
            n = len(prefix.derivatives)
            self.assertEqual(risk.method, 'buehler-joint-correlation-reverse-v1')
            self.assertEqual(risk.parameter_labels[:n], prefix.parameter_labels)
            self.assertEqual(risk.derivatives[:n], prefix.derivatives)
            self.assertEqual(risk.standard_errors[:n], prefix.standard_errors)
            self.assertEqual(risk.price.value, plan.evaluate().value)
            self.assertEqual(risk.price.standard_error, prefix.price.standard_error)
            self.assertEqual(risk.cash_mean_adjoints, prefix.cash_mean_adjoints)
            self.assertEqual(risk.discount_node_dv01, prefix.discount_node_dv01)
            self.assertEqual(risk.repo_spread_node_dv01, prefix.repo_spread_node_dv01)
            self.assertEqual(risk.parameter_labels[n:], [
                'equity_dividend_correlation', 'spot_volatility_correlation[0]',
                'dividend_volatility_correlation[0]'])
            for j, (key, value) in enumerate((('equity_dividend_correlation', -.25),
                                             ('correlation', -.4),
                                             ('dividend_volatility_correlation', .15))):
                for eps in (1e-5, 1e-6):
                    up = compile_plan(request, **(settings | {key: value+eps})).evaluate().value
                    down = compile_plan(request, **(settings | {key: value-eps})).evaluate().value
                    fd = (up-down)/(2*eps)
                    aad = risk.derivatives[n+j]
                    self.assertLessEqual(abs(fd-aad), 3e-5+2e-5*max(abs(fd), abs(aad)))

    def test_worker_replay_immutable_copies_and_price_preservation(self):
        request = make_request(points=32)
        a = compile_plan(request, maximum_step=.125)
        before = a.evaluate()
        x = a.evaluate_correlation_aad()
        y = compile_plan(request, maximum_step=.125, worker_threads=3).evaluate_correlation_aad()
        self.assertEqual(x.derivatives, y.derivatives)
        self.assertEqual(x.standard_errors, y.standard_errors)
        self.assertEqual(x.price.value, y.price.value)
        self.assertNotEqual(x.price.plan_fingerprint, y.price.plan_fingerprint)
        with self.assertRaises(AttributeError):
            x.method = 'changed'
        copy = x.derivatives
        copy[-1] = math.inf
        self.assertTrue(math.isfinite(x.derivatives[-1]))
        labels = x.parameter_labels
        labels[-1] = 'changed'
        self.assertEqual(x.parameter_labels[-1], 'dividend_volatility_correlation[0]')
        self.assertEqual(before.value, a.evaluate().value)
        self.assertEqual(before.standard_error, a.evaluate().standard_error)

    def test_zero_loadings_and_singular_domain_leave_existing_scopes_available(self):
        request = make_request(points=32, strike=60.)
        for hurst in (.1, .5):
            p = compile_plan(request, hurst=hurst, vol_of_vol=0., maximum_step=.125)
            risk = p.evaluate_correlation_aad()
            self.assertEqual(risk.derivatives[-2:], [0., 0.])
            self.assertEqual(risk.standard_errors[-2:], [0., 0.])
            with self.assertRaises(rp.PricingError):
                p.evaluate_bergomi_aad()
            for sd, sv, dv in ((1., 1., 1.), (.2, 1., .2), (0., 1.-1e-12, 0.)):
                p = compile_plan(request, hurst=hurst, vol_of_vol=0.,
                    equity_dividend_correlation=sd, correlation=sv,
                    dividend_volatility_correlation=dv, maximum_step=.125)
                baseline = p.evaluate_rough_aad()
                with self.assertRaisesRegex(rp.PricingError, 'correlation.*1e-10'):
                    p.evaluate_correlation_aad()
                self.assertEqual(baseline.derivatives, p.evaluate_rough_aad().derivatives)
                self.assertEqual(baseline.price.value, p.evaluate().value)
                p.evaluate_gamma(gamma_absolute_bump=1.)
        p = compile_plan(make_request(sigma=0., points=32), maximum_step=.125)
        risk = p.evaluate_correlation_aad()
        self.assertEqual(risk.derivatives[-2:], [0., 0.])
        self.assertEqual(risk.standard_errors[-2:], [0., 0.])


if __name__ == '__main__':
    unittest.main()
