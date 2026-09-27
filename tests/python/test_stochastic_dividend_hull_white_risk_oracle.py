"""Independent cash/HW expectation checks for every supported first-order risk."""
import json
import math
import unittest

from hw_dividend_reference import b, j, vi
from hw_dividend_risk_reference import Inputs, PARTIALS, partial, price


class HullWhiteRiskReferenceControlsTest(unittest.TestCase):
    def test_previous_frozen_one_step_price_targets(self):
        for k, expected in [(0.0, 7.029077680799348), (0.7, 7.266840212859318)]:
            p = Inputs(kappa=k, payment=1.0, rate_times=(0.0,), rate_vols=(0.04,))
            for order in [32, 40]:
                self.assertAlmostEqual(price(p, order), expected, delta=2e-9)

    def test_no_cash_delayed_gaussian_black_law(self):
        # A different closed-form calculation, with elementary constant-pre-T
        # rate integrals. Future eta cancels when curves are fixed and cash=0.
        for a in [0.0, 0.4]:
            for future_eta in [0.04, 0.09]:
                p = Inputs(rate_a=a, rate_vols=(0.04, future_eta), cash_times=(), cash_means=())
                T, U, eta, sigma = p.expiry, p.payment, p.rate_vols[0], p.sigma
                variance = vi(a, eta, T) + sigma**2 * T + 2.0 * sigma * p.rho_fr * eta * j(a, T)
                forward = p.spot * math.exp((p.log_q - p.log_p) * T
                    - b(a, U - T) * (0.5 * eta**2 * b(a, T)**2 + sigma * p.rho_fr * eta * b(a, T)))
                root = math.sqrt(variance)
                d1 = math.log(forward / p.strike) / root + root / 2.0
                cdf = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
                expected = math.exp(p.log_p * U) * (forward * cdf(d1) - p.strike * cdf(d1 - root))
                self.assertAlmostEqual(price(p), expected, delta=2e-9)

    def test_reference_quadrature_ode_and_stencil_resolution(self):
        for a in [0.0, 0.4]:
            p = Inputs(rate_a=a)
            target = price(p, 40)
            for order in [32, 40]:
                for steps in [256, 512]:
                    self.assertAlmostEqual(price(p, order, steps), target, delta=2e-9)
            for spec in PARTIALS:
                target = partial(p, spec, 40, 0.5)
                alternatives = [partial(p, spec, order, width)
                                for order in [32, 40] for width in [1.0, 0.5]]
                alternatives.append(partial(p, spec, 40, 0.5, 512))
                worst = max(abs(x - target) for x in alternatives)
                self.assertLessEqual(worst, 3e-6, (a, spec[0], target, alternatives))
                print(json.dumps(dict(check='hw_cash_risk_reference', rate_reversion=a,
                                      parameter=spec[0], reference=target,
                                      maximum_resolution_difference=worst)), flush=True)


class StochasticDividendHullWhiteRiskOracleTest(unittest.TestCase):
    def test_all_cash_rate_and_correlation_partials_against_independent_expectations(self):
        # Import only in the native test: the control class runs without Rust.
        import rust_pricing as rp
        for a in [0.0, 0.4]:
            p = Inputs(rate_a=a)
            req = rp.PricingRequest('2026-09-04',
                rp.Product.arithmetic_asian(1, 2, p.strike, 1.0, 'call',
                    [rp.AsianObservation.unknown('2027-09-04', 1.0)], '2027-12-04'),
                rp.Market.equity(2, 1, p.spot,
                    rp.DiscountCurve(10, [0.0, 1.0], [1.0, math.exp(p.log_p)]),
                    rp.DiscountCurve(11, [0.0, 1.0], [1.0, math.exp(p.log_q)]),
                    discrete_dividends=[rp.DividendEvent.fixed_cash(i + 1, t, mean)
                                       for i, (t, mean) in enumerate(zip(p.cash_times, p.cash_means))]),
                rp.Model.black_scholes(p.sigma),
                rp.Engine.randomized_quasi_monte_carlo(8192, 2909, scramble_count=8,
                    antithetic=True, brownian_bridge=True), rp.RiskRequest())
            plan = rp.StochasticDividendHullWhitePlan.compile_bs(req,
                dividend_mean_reversion=p.kappa, equity_linkage=p.alpha, dividend_volatility=p.nu,
                equity_dividend_correlation=p.rho_fd, rate_mean_reversion=p.rate_a,
                rate_volatility_times=list(p.rate_times), rate_volatilities=list(p.rate_vols),
                equity_rate_correlation=p.rho_fr, dividend_rate_correlation=p.rho_dr,
                maximum_step=1.0, worker_threads=1, reduction_block_size=64)
            self.assertEqual(plan.time_nodes, [0.0, 1.0])
            risk = plan.evaluate_correlation_aad()
            baseline = plan.evaluate()
            self.assertEqual(risk.price.value, baseline.value)
            self.assertEqual(risk.price.standard_error, baseline.standard_error)
            self.assertEqual(risk.price.independent_sampling_units, 8)
            self.assertEqual(risk.price.evaluated_paths, 2 * 8192 * 8)
            expected_labels = [s[0] for s in PARTIALS[:7]] + ['discount_log_df[0]',
                'discount_log_df[1]', 'repo_spread_log_df[0]', 'repo_spread_log_df[1]'] + [
                s[0] for s in PARTIALS[9:]]
            self.assertEqual(risk.parameter_labels, expected_labels)
            expected_price = price(p)
            self.assertLessEqual(abs(risk.price.value - expected_price),
                                 6.0 * risk.price.standard_error + 0.002)
            print(json.dumps(dict(check='hw_cash_price_oracle', rate_reversion=a,
                reference=expected_price, estimate=risk.price.value,
                standard_error=risk.price.standard_error)), flush=True)
            for spec in PARTIALS:
                index = risk.parameter_labels.index(spec[0])
                estimate, se = risk.derivatives[index], risk.standard_errors[index]
                expected = partial(p, spec)
                self.assertTrue(math.isfinite(estimate) and math.isfinite(se) and se >= 0.0)
                difference = estimate - expected
                print(json.dumps(dict(check='hw_cash_risk_oracle', rate_reversion=a,
                    parameter=spec[0], reference=expected, estimate=estimate,
                    standard_error=se, difference=difference)), flush=True)
                self.assertLessEqual(abs(difference), 6.0 * se + 2e-5,
                                     (a, spec[0], estimate, expected, se))
            for label in ['discount_log_df[0]', 'repo_spread_log_df[0]']:
                index = risk.parameter_labels.index(label)
                self.assertEqual(risk.derivatives[index], 0.0)
                self.assertEqual(risk.standard_errors[index], 0.0)


if __name__ == '__main__':
    unittest.main()
