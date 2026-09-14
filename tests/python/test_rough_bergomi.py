import math
import unittest

import numpy as np
import rust_pricing as rp


class RoughBergomiTest(unittest.TestCase):
    def request(self, model, spot=100.0, cash=False, points=64):
        return rp.PricingRequest(
            "2026-09-04",
            rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call"),
            rp.Market.equity(
                2, 1, spot,
                rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95]),
                rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98]),
                discrete_dividends=[rp.DividendEvent.fixed_cash(1, 0.5, 6.0)] if cash else [],
            ),
            model, rp.Engine.randomized_quasi_monte_carlo(
                points, 612, scramble_count=8, antithetic=True, brownian_bridge=True),
            rp.RiskRequest(),
        )

    def pure(self, sigma=0.2, spot=100.0, cash=False, workers=1, eta=0.8):
        return rp.HullWhiteEquityPlan.compile_rough_bergomi(
            self.request(rp.Model.black_scholes(sigma), spot=spot, cash=cash),
            rp.RoughBergomiModel(0.1, eta, equity_vol_correlation=-0.5),
            rp.HullWhiteModel(0.2, [0.0, 0.3], [0.005, 0.01]),
            equity_rate_correlation=0.25, vol_rate_correlation=-0.1,
            maximum_step=0.125, worker_threads=workers, reduction_block_size=32,
            cash_dividend_model="escrowed" if cash else None,
        )

    def lsv(self, quotes, trace=True, workers=1):
        target = rp.HullWhiteLsvTarget.from_market_iv(
            [0.4, 1.0], [-0.8, -0.3, 0.0, 0.4, 0.8], quotes,
            [0.0, 0.25, 0.5, 0.75, 1.0], [-0.8, -0.4, 0.0, 0.4, 0.8],
        )
        return rp.HullWhiteEquityPlan.compile_rough_lsv(
            self.request(target.model, cash=True), target,
            rp.RoughBergomiModel(0.1, 0.8, equity_vol_correlation=-0.5),
            rp.HullWhiteModel(0.2, [0.0], [0.005]),
            equity_rate_correlation=0.25, vol_rate_correlation=-0.1,
            particle_count=512, calibration_seed=712, log_bandwidth=0.35,
            minimum_effective_samples=5.0, worker_threads=workers,
            reduction_block_size=32, retain_reverse_trace=trace,
            cash_dividend_model="escrowed",
        )

    def test_model_validation_ownership_and_plan_errors(self):
        model = rp.RoughBergomiModel(0.1, 0.8, equity_vol_correlation=-0.5)
        self.assertEqual((model.hurst, model.vol_of_vol, model.equity_vol_correlation),
                         (0.1, 0.8, -0.5))
        with self.assertRaises(AttributeError):
            model.hurst = 0.2
        for h, eta, rho in [(0.0, 0.8, -0.5), (-0.1, 0.8, -0.5),
                            (0.51, 0.8, -0.5), (math.nan, 0.8, -0.5),
                            (0.1, -0.1, -0.5), (0.1, math.inf, -0.5),
                            (0.1, 0.8, 1.1), (0.1, 0.8, math.nan)]:
            with self.assertRaises(rp.ValidationError):
                rp.RoughBergomiModel(h, eta, equity_vol_correlation=rho)
        rp.RoughBergomiModel(0.5, 0.0, equity_vol_correlation=1.0)
        args = dict(equity_rate_correlation=0.25, vol_rate_correlation=-0.1,
                    maximum_step=0.25, worker_threads=1)
        request = self.request(rp.Model.black_scholes(0.2), cash=True)
        rates = rp.HullWhiteModel(0.2, [0.0], [0.005])
        affine = rp.HullWhiteEquityPlan.compile_rough_bergomi(request, model, rates, **args)
        self.assertEqual(affine.cash_dividend_model, "affine-paid-cash-realized-carry-v1")
        self.assertEqual(affine.risky_spot, 100.0)
        self.assertEqual(affine.evaluate_aad().price.value, affine.evaluate().value)
        with self.assertRaises(rp.ValidationError):
            rp.HullWhiteEquityPlan.compile_rough_bergomi(
                request, model, rates, **args, cash_dividend_model="unknown")
        args.update(equity_rate_correlation=0.9, vol_rate_correlation=0.9)
        with self.assertRaises(rp.ValidationError):
            rp.HullWhiteEquityPlan.compile_rough_bergomi(
                request, model, rates, **args, cash_dividend_model="escrowed")

    def test_pure_aad_units_replay_and_full_recompile_bumps(self):
        for cash in [False, True]:
            plan = self.pure(cash=cash)
            risk = plan.evaluate_aad()
            self.assertEqual(plan.random_factor_count, 5)
            self.assertEqual(risk.price.value, plan.evaluate().value)
            self.assertEqual(risk.price.scheme, "rough-bergomi-hw-hybrid-kappa1-log-euler-v1")
            self.assertIsNone(risk.price.calibration_method)
            self.assertIsNone(risk.vega_kt_raw)
            self.assertEqual(risk.parameter_labels[:2], ["spot", "initial_volatility"])
            for name, value, bar in [("spot", 100.0, risk.delta), ("sigma", 0.2, risk.vega)]:
                h = 1e-6
                up = self.pure(cash=cash, **{name: value+h}).evaluate().value
                down = self.pure(cash=cash, **{name: value-h}).evaluate().value
                self.assertAlmostEqual(bar, (up-down)/(2*h), delta=2e-5)
            replay = self.pure(cash=cash, workers=3).evaluate_aad()
            self.assertEqual(risk.derivatives, replay.derivatives)
            self.assertEqual(risk.standard_errors, replay.standard_errors)
            self.assertNotEqual(plan.plan_fingerprint, self.pure(cash=cash, eta=0.9).plan_fingerprint)

    def test_cash_lsv_vegakt_recalibration_and_trace_contract(self):
        quotes = [0.23, 0.224, 0.22, 0.216, 0.22, 0.236, 0.23, 0.226, 0.222, 0.226]
        plan = self.lsv(quotes)
        risk = plan.evaluate_aad()
        self.assertEqual(risk.price.value, plan.evaluate().value)
        self.assertEqual(risk.price.calibration_method, "rough-lsv-hw-escrowed-quadratic-v1")
        self.assertEqual(len(risk.vega_kt_raw), 10)
        self.assertEqual(risk.vega_kt_implied_volatilities, quotes)
        self.assertEqual(risk.vega_kt_market_scaled, [v*0.01 for v in risk.vega_kt_raw])
        self.assertAlmostEqual(risk.vega, sum(risk.vega_kt_raw), places=12)
        self.assertEqual(risk.uncertainty_scope, "pricing_conditional_on_calibration")
        for i in [2, 7]:
            up, down = quotes.copy(), quotes.copy()
            up[i] += 1e-7
            down[i] -= 1e-7
            fd = (self.lsv(up).evaluate().value-self.lsv(down).evaluate().value)/2e-7
            self.assertAlmostEqual(risk.vega_kt_raw[i], fd, delta=2e-5)
        replay = self.lsv(quotes, workers=3).evaluate_aad()
        self.assertEqual(risk.derivatives, replay.derivatives)
        self.assertEqual(risk.standard_errors, replay.standard_errors)
        no_trace = self.lsv(quotes, trace=False)
        self.assertEqual(risk.price.value, no_trace.evaluate().value)
        with self.assertRaises(rp.PricingError):
            no_trace.evaluate_aad()

    def test_two_step_price_against_independent_conditional_quadrature(self):
        # At the first midpoint, X_H is exactly Gaussian. Integrate the final
        # equity increment analytically, then use two-dimensional Gauss-Hermite
        # quadrature for the first equity increment and near-cell rough driver.
        # This independently checks the price, eta scaling, centering and rho.
        h, eta, rho, sigma, dt = 0.1, 0.8, -0.7, 0.2, 0.5
        r, q = -math.log(0.95), -math.log(0.98)
        correlation = rho * math.sqrt(2*h) / (h+0.5)
        cdf = lambda z: 0.5 * math.erfc(-z/math.sqrt(2.0))
        def reference(order):
            nodes, weights = np.polynomial.hermite.hermgauss(order)
            total = 0.0
            for z1, w1 in zip(nodes*math.sqrt(2), weights):
                s = 100.0*math.exp((r-q-0.5*sigma*sigma)*dt + sigma*math.sqrt(dt)*z1)
                for z2, w2 in zip(nodes*math.sqrt(2), weights):
                    x = dt**h * (correlation*z1+math.sqrt(1-correlation**2)*z2)
                    v = sigma*sigma*math.exp(eta*x-0.5*eta*eta*dt**(2*h))
                    root = math.sqrt(v*dt)
                    d1 = (math.log(s/100)+(r-q+0.5*v)*dt)/root
                    call = s*math.exp(-q*dt)*cdf(d1)-100*math.exp(-r*dt)*cdf(d1-root)
                    total += w1*w2*call
            return math.exp(-r*dt)*total/math.pi
        expected = reference(96)
        self.assertAlmostEqual(expected, reference(128), delta=2e-6)
        plan = rp.HullWhiteEquityPlan.compile_rough_bergomi(
            self.request(rp.Model.black_scholes(sigma), points=4096),
            rp.RoughBergomiModel(h, eta, equity_vol_correlation=rho),
            rp.HullWhiteModel(0.0, [0.0], [0.0]),
            equity_rate_correlation=0.0, vol_rate_correlation=0.0,
            maximum_step=dt, worker_threads=2,
        )
        price = plan.evaluate()
        self.assertAlmostEqual(price.value, expected, delta=6*price.standard_error+2e-6)


if __name__ == "__main__":
    unittest.main()
