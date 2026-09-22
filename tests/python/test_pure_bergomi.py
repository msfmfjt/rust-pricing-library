"""Pure SV public APIs and an independent finite-step Gaussian price oracle."""
import math
import unittest

import numpy as np
import rust_pricing as rp


class PureBergomiTest(unittest.TestCase):
    def request(self, sigma=0.2, spot=100.0, cash=False, points=128,
                discount=0.95, carry=0.98, strike=100.0):
        return rp.PricingRequest(
            "2026-09-04",
            rp.Product.european_vanilla(1, 2, "2027-09-04", strike, 1.0, "call"),
            rp.Market.equity(
                2, 1, spot,
                rp.DiscountCurve(10, [0.0, 1.0], [1.0, discount]),
                rp.DiscountCurve(11, [0.0, 1.0], [1.0, carry]),
                discrete_dividends=[rp.DividendEvent.fixed_cash(1, 0.35, 6.0),
                                    rp.DividendEvent.fixed_cash(2, 1.4, 3.0)] if cash else [],
            ), rp.Model.black_scholes(sigma),
            rp.Engine.randomized_quasi_monte_carlo(points, 612, scramble_count=8,
                antithetic=True, brownian_bridge=True), rp.RiskRequest(),
        )

    def plan(self, two=False, hw=False, workers=1, step=0.125, **market):
        request = self.request(**market)
        args = dict(vol_of_vol=0.6, maximum_step=step,
                    worker_threads=workers, reduction_block_size=64)
        if two:
            args.update(mean_reversions=[0.7, 2.1], mixing_weight=0.35,
                        spot_correlations=[-0.5, -0.3], factor_correlation=0.25)
        if hw:
            rates = rp.HullWhiteModel(0.2, [0.0, 0.3], [0.005, 0.01])
            args.update(equity_rate_correlation=0.2)
            if two:
                return rp.HullWhiteEquityPlan.compile_bergomi_two_factor(request, rates,
                    vol_rate_correlations=[-0.1, 0.1], **args)
            return rp.HullWhiteEquityPlan.compile_bergomi(request, rates,
                vol_mean_reversion=0.7, equity_vol_correlation=-0.5,
                vol_rate_correlation=-0.1, **args)
        if two:
            return rp.StochasticVolatilityPlan.compile_bergomi_two_factor(request, **args)
        return rp.StochasticVolatilityPlan.compile_bergomi(request,
            mean_reversion=0.7, correlation=-0.5, **args)

    def test_public_aad_and_curve_risk_with_dividends(self):
        for two in [False, True]:
            for hw in [False, True]:
                for cash in [False, True]:
                    with self.subTest(two=two, hw=hw, cash=cash):
                        plan = self.plan(two=two, hw=hw, cash=cash)
                        risk = plan.evaluate_aad()
                        self.assertEqual(risk.price.value, plan.evaluate().value)
                        self.assertEqual(risk.parameter_labels[:2], ["spot", "initial_volatility"])
                        self.assertEqual(risk.price.uncertainty_scope, "pricing_only")
                        self.assertIsNone(risk.price.calibration_method)
                        self.assertIsNone(risk.price.calibration_seed)
                        self.assertIsNone(risk.vega_kt_raw)
                        self.assertEqual(plan.random_factor_count, 5 if two else 4)
                        self.assertEqual(plan.cash_dividend_model, "escrowed-hw-bonds-v1")
                        if cash:
                            self.assertLess(plan.risky_spot, 92.0)
                        for name, value, bar, log_bump in [
                            ("sigma", 0.2, risk.vega, False),
                            ("spot", 100.0, risk.delta, False),
                            ("discount", 0.95, risk.discount_log_df_adjoints[1], True),
                            ("carry", 0.98, risk.dividend_log_df_adjoints[1], True),
                        ]:
                            h = 1e-6
                            up = value*math.exp(h) if log_bump else value+h
                            down = value*math.exp(-h) if log_bump else value-h
                            a = self.plan(two=two, hw=hw, cash=cash, **{name: up}).evaluate().value
                            b = self.plan(two=two, hw=hw, cash=cash, **{name: down}).evaluate().value
                            self.assertAlmostEqual(bar, (a-b)/(2*h), delta=3e-5)
                        replay = self.plan(two=two, hw=hw, cash=cash, workers=3).evaluate_aad()
                        self.assertEqual(risk.derivatives, replay.derivatives)
                        self.assertEqual(risk.standard_errors, replay.standard_errors)

    def test_rough_and_zero_rate_adapters_match_existing_api(self):
        request = self.request(cash=True)
        model = rp.RoughBergomiModel(0.1, 0.8, equity_vol_correlation=-0.5)
        args = dict(maximum_step=0.125, worker_threads=1)
        plain = rp.StochasticVolatilityPlan.compile_rough_bergomi(request, model, **args)
        hybrid = rp.HullWhiteEquityPlan.compile_rough_bergomi(request, model,
            rp.HullWhiteModel(0.0, [0.0], [0.0]), equity_rate_correlation=0.0,
            vol_rate_correlation=0.0, **args)
        self.assertEqual(plain.plan_fingerprint, hybrid.plan_fingerprint)
        self.assertEqual(plain.evaluate_aad().derivatives, hybrid.evaluate_aad().derivatives)
        with self.assertRaises(AttributeError):
            plain.risky_spot = 1.0

    def test_zero_initial_volatility_has_finite_right_vega(self):
        # Deep ITM removes payoff-kink ambiguity in the right derivative at zero.
        for two in [False, True]:
            for hw in [False, True]:
                p = self.plan(two=two, hw=hw, sigma=0.0, strike=60.0)
                risk = p.evaluate_aad()
                h = 1e-7
                up = self.plan(two=two, hw=hw, sigma=h, strike=60.0).evaluate().value
                self.assertTrue(math.isfinite(risk.vega))
                self.assertAlmostEqual(risk.vega, (up-risk.price.value)/h, delta=3e-5)

    def test_pure_price_against_independent_two_step_conditional_quadrature(self):
        # Derive Var[Y(dt)] and Cov[Y(dt), W_S(dt)] directly from OU kernels,
        # then integrate the last equity step analytically. This oracle does
        # not use library transitions, leverage, RNG, or calibration.
        nu, sigma, dt = 0.6, 0.2, 0.5
        r, q = -math.log(0.95), -math.log(0.98)
        integral = lambda k: -math.expm1(-k*dt)/k
        cdf = lambda z: 0.5*math.erfc(-z/math.sqrt(2))
        for two in [False, True]:
            theta, rho12 = 0.35, 0.25
            norm = math.sqrt((1-theta)**2+theta**2+2*theta*(1-theta)*rho12)
            w1, w2 = ((1-theta)/norm, theta/norm) if two else (1.0, 0.0)
            var = w1*w1*integral(1.4)+w2*w2*integral(4.2)+2*w1*w2*rho12*integral(2.8)
            cov = w1*(-0.5)*integral(0.7)+w2*(-0.3)*integral(2.1)
            correlation = cov/math.sqrt(var*dt)
            for strike in [90.0, 100.0, 110.0]:
                def reference(order):
                    nodes, weights = np.polynomial.hermite.hermgauss(order)
                    total = 0.0
                    for z1, p1 in zip(nodes*math.sqrt(2), weights):
                        s = 100*math.exp((r-q-0.5*sigma*sigma)*dt+sigma*math.sqrt(dt)*z1)
                        for z2, p2 in zip(nodes*math.sqrt(2), weights):
                            y = math.sqrt(var)*(correlation*z1+math.sqrt(1-correlation**2)*z2)
                            v = sigma*sigma*math.exp(2*nu*y-2*nu*nu*var)
                            root = math.sqrt(v*dt)
                            d1 = (math.log(s/strike)+(r-q+0.5*v)*dt)/root
                            call = s*math.exp(-q*dt)*cdf(d1)-strike*math.exp(-r*dt)*cdf(d1-root)
                            total += p1*p2*call
                    return math.exp(-r*dt)*total/math.pi
                expected = reference(64)
                self.assertAlmostEqual(expected, reference(96), delta=2e-6)
                result = self.plan(two=two, step=dt, points=4096, strike=strike).evaluate()
                self.assertLess(abs(result.value-expected), 6*result.standard_error+2e-6)

    def test_parameter_validation_and_fingerprint(self):
        request = self.request()
        args = dict(mean_reversion=0.7, vol_of_vol=0.6, correlation=-0.5,
                    maximum_step=0.125, worker_threads=1)
        for field, bad in [("mean_reversion", -1.0), ("vol_of_vol", math.nan),
                           ("correlation", 1.1), ("worker_threads", 0)]:
            with self.subTest(field=field), self.assertRaises(rp.ValidationError):
                rp.StochasticVolatilityPlan.compile_bergomi(request, **(args | {field: bad}))
        for step in [0.0, -1.0, math.nan, math.inf]:
            with self.assertRaises(rp.PricingError):
                rp.StochasticVolatilityPlan.compile_bergomi(request, **(args | {"maximum_step": step}))
        p = rp.StochasticVolatilityPlan.compile_bergomi(request, **args)
        for field, value in [("mean_reversion", 0.8), ("vol_of_vol", 0.7),
                             ("correlation", -0.4), ("maximum_step", 0.1)]:
            changed = rp.StochasticVolatilityPlan.compile_bergomi(request, **(args | {field: value}))
            self.assertNotEqual(p.plan_fingerprint, changed.plan_fingerprint)


if __name__ == "__main__":
    unittest.main()
