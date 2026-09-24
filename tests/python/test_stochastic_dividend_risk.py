"""Explicit AAD API; full recompilation for sigma0/cash/curve checks."""
import math
import unittest
import rust_pricing as rp
from test_stochastic_dividends import make_request, compile_plan as bs
from test_bergomi_dividends import compile_plan as bergomi


def plan(family, sigma=.2, cash=25., workers=1):
    request = make_request(sigma=sigma, dividends=((1.4,cash),), points=128)
    if family == 0:
        return bs(request, mean_reversion=.7, maximum_step=.125, worker_threads=workers)
    return bergomi(family == 2, request, maximum_step=.125, worker_threads=workers)


class StochasticDividendRiskTest(unittest.TestCase):
    def test_rust_result_python_properties_and_sampling_metadata(self):
        for family in range(3):
            p=plan(family)
            r=p.evaluate_aad()
            self.assertIsInstance(r,rp.StochasticDividendAadRisk)
            self.assertEqual(r.price.value,p.evaluate().value)
            self.assertEqual(r.price.standard_error,p.evaluate().standard_error)
            self.assertEqual(r.price.plan_fingerprint,p.plan_fingerprint)
            self.assertEqual(r.price.independent_sampling_units,8)
            self.assertEqual(r.cash_times,[1.4])
            self.assertEqual(len(r.parameter_labels),len(r.derivatives))
            self.assertEqual(len(r.derivatives),len(r.standard_errors))
            self.assertTrue(all(math.isfinite(s) and s>=0 for s in r.standard_errors))
            self.assertEqual(r.delta,r.derivatives[0])
            self.assertEqual(r.initial_volatility_vega,r.derivatives[1])
            self.assertEqual(r.initial_volatility_vega_per_vol_point,.01*r.derivatives[1])
            self.assertEqual(r.dividend_volatility_vega_per_vol_point,.01*r.derivatives[4])
            self.assertEqual(r.cash_mean_adjoints,r.derivatives[5:6])
            self.assertEqual(r.discount_node_dv01,[0.,-1e-4*r.derivatives[7]])
            self.assertEqual(r.repo_spread_node_dv01,[0.,-1e-4*r.derivatives[9]])
            self.assertEqual(r.method,'buehler-split-payoff-reverse-fixed-correlation-v1')
            self.assertEqual(r.uncertainty_scope,'sampling_only_fixed_parameters_and_grid')
            with self.assertRaises(AttributeError): r.delta=0.
            copy=r.derivatives;copy[0]=1e99
            self.assertNotEqual(r.derivatives[0],1e99)

    def test_public_risk_matches_full_recompile(self):
        for family in range(3):
            r=plan(family).evaluate_aad()
            for h in [1e-5,1e-6]:
                vega=(plan(family,sigma=.2+h).evaluate().value-plan(family,sigma=.2-h).evaluate().value)/(2*h)
                cash=(plan(family,cash=25+h).evaluate().value-plan(family,cash=25-h).evaluate().value)/(2*h)
                for actual,expected in [(r.initial_volatility_vega,vega),(r.cash_mean_adjoints[0],cash)]:
                    self.assertLess(abs(actual-expected),3e-5+2e-5*max(abs(actual),abs(expected)))

    def test_replay_with_changed_worker_count(self):
        for family in range(3):
            a,b=plan(family).evaluate_aad(),plan(family,workers=3).evaluate_aad()
            self.assertEqual(a.price.value,b.price.value)
            self.assertEqual(a.derivatives,b.derivatives)
            self.assertEqual(a.standard_errors,b.standard_errors)

    def test_zero_sigma_boundary_remains_finite(self):
        for family in range(3):
            r=plan(family,sigma=0.).evaluate_aad()
            self.assertTrue(all(math.isfinite(x) for x in r.derivatives))
            self.assertTrue(all(math.isfinite(x) for x in r.standard_errors))


if __name__=='__main__': unittest.main()
