"""Explicit model-risk scope: original AAD prefix is unchanged."""
import math
import unittest

import rust_pricing as rp
from test_stochastic_dividends import make_request, compile_plan as bs
from test_bergomi_dividends import compile_plan


class BergomiParameterRiskTest(unittest.TestCase):
    def test_all_model_parameters_against_recompiled_prices(self):
        request = make_request(points=64, dividends=((.5, 4.), (1.0, 3.), (1.4, 8.)))
        for two in (False, True):
            plan = compile_plan(two, request, maximum_step=.125)
            basic = plan.evaluate_aad()
            risk = plan.evaluate_bergomi_aad()
            prefix = len(basic.derivatives)
            self.assertEqual(risk.price.value, plan.evaluate().value)
            self.assertEqual(risk.price.standard_error, basic.price.standard_error)
            self.assertEqual(risk.parameter_labels[:prefix], basic.parameter_labels)
            self.assertEqual(risk.derivatives[:prefix], basic.derivatives)
            self.assertEqual(risk.standard_errors[:prefix], basic.standard_errors)
            self.assertEqual(risk.cash_mean_adjoints, basic.cash_mean_adjoints)
            self.assertEqual(risk.discount_node_dv01, basic.discount_node_dv01)
            self.assertEqual(risk.repo_spread_node_dv01, basic.repo_spread_node_dv01)
            labels = ['bergomi_mean_reversion[0]']
            if two:
                labels += ['bergomi_mean_reversion[1]']
            labels += ['bergomi_vol_of_vol']
            if two:
                labels += ['bergomi_mixing_weight']
            self.assertEqual(risk.parameter_labels[prefix:], labels)
            self.assertEqual(risk.method, 'buehler-bergomi-parameter-reverse-fixed-correlation-v1')
            for index, label in enumerate(labels):
                for h in (1e-5, 1e-6):
                    if label.startswith('bergomi_mean_reversion'):
                        p = int(label.split('[')[1][0])
                        if two:
                            plus, minus = [.8, 2.1], [.8, 2.1]
                            plus[p] += h
                            minus[p] -= h
                            a, b = dict(mean_reversions=plus), dict(mean_reversions=minus)
                        else:
                            a, b = dict(mean_reversion=.8+h), dict(mean_reversion=.8-h)
                    else:
                        name, value = ('mixing_weight', .35) if label.endswith('mixing_weight') else ('vol_of_vol', .3)
                        a, b = {name: value+h}, {name: value-h}
                    fd = (compile_plan(two, request, maximum_step=.125, **a).evaluate().value
                          - compile_plan(two, request, maximum_step=.125, **b).evaluate().value) / (2*h)
                    actual = risk.derivatives[prefix+index]
                    self.assertLessEqual(abs(actual-fd), 3e-5+2e-5*max(abs(actual), abs(fd)))

    def test_worker_replay_copy_semantics_and_zero_sigma(self):
        for two in (False, True):
            request = make_request(points=32)
            a = compile_plan(two, request, maximum_step=.125).evaluate_bergomi_aad()
            b = compile_plan(two, request, maximum_step=.125, worker_threads=3).evaluate_bergomi_aad()
            self.assertEqual(a.derivatives, b.derivatives)
            self.assertEqual(a.standard_errors, b.standard_errors)
            with self.assertRaises(AttributeError):
                a.method = 'changed'
            values = a.derivatives
            values[-1] = math.inf
            self.assertTrue(math.isfinite(a.derivatives[-1]))
            plan = compile_plan(two, make_request(sigma=0., points=16))
            n = len(plan.evaluate_aad().derivatives)
            risk = plan.evaluate_bergomi_aad()
            self.assertTrue(all(x == 0. for x in risk.derivatives[n:]))
            self.assertTrue(all(x == 0. for x in risk.standard_errors[n:]))

    def test_unsupported_family_and_singular_boundary_keep_original_methods(self):
        request = make_request(points=16)
        p = bs(request)
        p.evaluate_aad()
        with self.assertRaises(rp.PricingError):
            p.evaluate_bergomi_aad()
        for sd in (1., 1.-1e-12):
            p = compile_plan(False, request, equity_dividend_correlation=sd,
                             correlation=0., dividend_volatility_correlation=0.)
            p.evaluate()
            p.evaluate_aad()
            with self.assertRaises(rp.PricingError):
                p.evaluate_bergomi_aad()


if __name__ == '__main__':
    unittest.main()
