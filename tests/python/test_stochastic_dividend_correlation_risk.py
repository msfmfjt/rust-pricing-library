"""Correlation scope: exact prior-risk prefix and full-recompile entry bumps."""
import math
import unittest

import rust_pricing as rp
from test_stochastic_dividends import make_request, compile_plan as bs_plan
from test_bergomi_dividends import compile_plan as bergomi_plan

RHO = [-.25, -.4, -.2, .15, -.1, .3]
LABELS = ["equity_dividend_correlation", "spot_volatility_correlation[0]",
          "spot_volatility_correlation[1]", "dividend_volatility_correlation[0]",
          "dividend_volatility_correlation[1]", "volatility_factor_correlation"]


def compile_plan(family, request, rho=RHO, **changes):
    common = dict(equity_dividend_correlation=rho[0], maximum_step=.125)
    if family == 0:
        return bs_plan(request, **(common | changes))
    if family == 1:
        common.update(correlation=rho[1], dividend_volatility_correlation=rho[3])
    else:
        common.update(spot_correlations=[rho[1], rho[2]], factor_correlation=rho[5],
                      dividend_volatility_correlations=[rho[3], rho[4]])
    return bergomi_plan(family == 2, request, **(common | changes))


class CorrelationRiskTest(unittest.TestCase):
    def test_all_entry_derivatives_and_exact_prefix(self):
        request = make_request(points=64, dividends=((.5, 4.), (1., 3.), (1.4, 8.)))
        for family, active in [(0, [0]), (1, [0, 1, 3]), (2, list(range(6)))]:
            plan = compile_plan(family, request)
            prior = plan.evaluate_aad() if family == 0 else plan.evaluate_bergomi_aad()
            risk = plan.evaluate_correlation_aad()
            n = len(prior.derivatives)
            self.assertEqual(risk.price.value, plan.evaluate().value)
            self.assertEqual(risk.price.standard_error, prior.price.standard_error)
            self.assertEqual(risk.parameter_labels[:n], prior.parameter_labels)
            self.assertEqual(risk.derivatives[:n], prior.derivatives)
            self.assertEqual(risk.standard_errors[:n], prior.standard_errors)
            self.assertEqual(risk.parameter_labels[n:], [LABELS[i] for i in active])
            self.assertEqual(risk.method, 'buehler-joint-correlation-reverse-v1')
            for j, p in enumerate(active):
                for h in [1e-5, 1e-6]:
                    plus, minus = RHO.copy(), RHO.copy()
                    plus[p] += h
                    minus[p] -= h
                    fd = (compile_plan(family, request, plus).evaluate().value
                          - compile_plan(family, request, minus).evaluate().value) / (2*h)
                    aad = risk.derivatives[n+j]
                    self.assertLessEqual(abs(aad-fd), 3e-5+2e-5*max(abs(aad), abs(fd)))

    def test_workers_immutable_copies_and_zero_entries(self):
        request = make_request(points=32)
        for family in range(3):
            plan = compile_plan(family, request, [0.] * 6)
            risk = plan.evaluate_correlation_aad()
            replay = compile_plan(family, request, [0.] * 6, worker_threads=3).evaluate_correlation_aad()
            self.assertEqual(risk.derivatives, replay.derivatives)
            self.assertEqual(risk.standard_errors, replay.standard_errors)
            with self.assertRaises(AttributeError):
                risk.method = 'changed'
            copy = risk.derivatives
            copy[-1] = math.inf
            self.assertTrue(math.isfinite(risk.derivatives[-1]))

    def test_instantaneous_boundary_keeps_existing_pricing_and_basic_risk(self):
        request = make_request(points=16)
        for family in range(3):
            for sd in [1., -1., 1.-1e-12]:
                p = compile_plan(family, request, [sd, 0., 0., 0., 0., 0.])
                p.evaluate()
                p.evaluate_aad()
                with self.assertRaisesRegex(rp.PricingError, 'correlation AAD requires'):
                    p.evaluate_correlation_aad()
        p = compile_plan(2, request, [-.25, -.4, -.4, .15, .15, 1.])
        p.evaluate_bergomi_aad()  # Integrated SPD does not make the raw entry partial feasible.
        with self.assertRaisesRegex(rp.PricingError, 'instantaneous'):
            p.evaluate_correlation_aad()


if __name__ == '__main__':
    unittest.main()
