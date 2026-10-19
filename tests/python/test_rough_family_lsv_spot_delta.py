"""Two explicit Spot conventions, independent Asian payoff/SE and API contracts."""
import bisect
import datetime as dt
import math
import unittest

import rust_pricing as rp
from test_rough_volatility_families import models

TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]
XS = [-0.6, -0.2, 0.13, 0.45, 0.8]
DATES = ["2026-12-04", "2027-03-04", "2027-09-04"]
SEED = 819


def request(spot=100.0, *, product=None, flat=False, qmc=True, smooth=None, xs=XS):
    curve = rp.DiscountCurve(10, [0.0, 1.0, 2.0],
                             [1.0, 1.0 if flat else 0.95, 1.0 if flat else 0.95**2])
    repo = rp.DiscountCurve(11, [0.0, 1.0, 2.0],
                            [1.0, 1.0 if flat else 0.98, 1.0 if flat else 0.98**2])
    market = rp.Market.equity(2, 1, spot, curve, repo,
        discrete_dividends=[rp.DividendEvent.fixed_cash(1, 0.25, 3.0),
                           rp.DividendEvent.fixed_cash(2, 1.5, 4.0)])
    target = rp.Model.local_volatility_from_grid(TIMES, xs,
        [0.04 + 0.001*i + 0.002*j for i in range(5) for j in range(len(xs))], 1e-8, 4.0)
    engine = (rp.Engine.randomized_quasi_monte_carlo(64, SEED, scramble_count=4,
                antithetic=True, brownian_bridge=True) if qmc else
              rp.Engine.pseudo_monte_carlo(SEED, 128, antithetic=True, brownian_bridge=False))
    return rp.PricingRequest("2026-09-04",
        product or rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call"),
        market, target, engine, rp.RiskRequest(payoff_smoothing_half_width=smooth))


def compile_plan(model, spot=100.0, *, workers=1, trace=False, **kwargs):
    return rp.RoughFamilyLsvPlan.compile(request(spot, **kwargs), model,
        particle_count=128, calibration_seed=429, log_bandwidth=0.5,
        minimum_effective_samples=3.0, retain_reverse_trace=trace,
        worker_threads=workers, reduction_block_size=64)


def asian_product():
    obs = [rp.AsianObservation.unknown(date, 1.0/3.0) for date in DATES]
    return rp.Product.arithmetic_asian(1, 2, 100.0, 1.0, "call", obs, "2027-12-04")


def mean_se(values):
    mean = math.fsum(values)/len(values)
    se = math.sqrt(math.fsum((v-mean)**2 for v in values)/(len(values)*(len(values)-1)))
    return mean, se


def asian_frozen_reference(plan, model, shift):
    """Independent LSV recurrence and physical Asian payoff, public shocks/variance.

    Calibration is frozen. The reference cash schedule is 3 at .25 and 4 at1.5,
    with zero carry/discount. No production payoff or reverse path is invoked.
    """
    times = plan.time_nodes
    xs = plan.log_moneyness_nodes
    leverage = plan.squared_leverage
    driver = rp.RoughVolatilityPathPlan.compile(model, times)
    observation_times = [(dt.date.fromisoformat(d)-dt.date(2026, 9, 4)).days/365 for d in DATES]
    observation_indices = [times.index(t) for t in observation_times]
    n = len(times)-1
    pairs = []
    for path in range(128):
        normals = driver.pseudo_shocks(SEED, path)
        values = []
        for sign in [1.0, -1.0]:
            z = [sign*v for v in normals]
            variance = driver.evolve_path(100.0, z).variances
            f = 100.0 + shift*100.0/93.0
            states = [f]
            for j in range(n):
                x = math.log(f/100.0)
                col = min(max(bisect.bisect_right(xs, x)-1, 0), len(xs)-2)
                w = min(max((x-xs[col])/(xs[col+1]-xs[col]), 0.0), 1.0)
                l2 = (1.0-w)*leverage[j*len(xs)+col] + w*leverage[j*len(xs)+col+1]
                v = l2*variance[j]
                if variance[j] != 0.0:
                    step = times[j+1]-times[j]
                    f *= math.exp(-0.5*v*step + math.sqrt(v*step)*z[j])
                states.append(f)
            observed = [(7.0 if t < 0.25 else 4.0) + 0.93*states[i]
                        for t, i in zip(observation_times, observation_indices)]
            values.append(max(sum(v/3.0 for v in observed)-100.0, 0.0))
        pairs.append(0.5*(values[0]+values[1]))
    return pairs


class RoughFamilyLsvSpotDeltaTest(unittest.TestCase):
    def test_conventions_immutable_results_price_and_workers(self):
        differences = []
        for model in models():
            p = compile_plan(model)
            price = p.evaluate()
            for method, convention in [("evaluate_frozen_leverage_delta", "frozen_leverage_in_reference_f"),
                                       ("evaluate_sticky_moneyness_delta", "sticky_relative_local_variance")]:
                result = getattr(p, method)()
                self.assertIsInstance(result, rp.RoughFamilyLsvDelta)
                self.assertEqual(result.price.value, price.value)
                self.assertEqual(result.price.standard_error, price.standard_error)
                self.assertEqual(result.price.plan_fingerprint, price.plan_fingerprint)
                self.assertEqual(result.price.uncertainty_scope, "pricing_conditional_on_calibration")
                self.assertEqual(result.convention, convention)
                self.assertEqual(result.coordinate, "physical_spot_fixed_curves_and_cash_dividends")
                self.assertGreater(result.delta_standard_error, 0.0)
                other = getattr(compile_plan(model, workers=3, trace=True), method)()
                self.assertEqual((result.delta, result.delta_standard_error),
                                 (other.delta, other.delta_standard_error))
                for name in ["delta", "delta_standard_error", "convention", "coordinate", "method", "price"]:
                    with self.assertRaises(AttributeError):
                        setattr(result, name, 0.0)
            differences.append(abs(p.evaluate_frozen_leverage_delta().delta-p.evaluate_sticky_moneyness_delta().delta))
        self.assertGreater(max(differences), 1e-4)

    def test_sticky_asian_delta_full_recalibration_nonflat_carry_and_delayed_payment(self):
        product = asian_product()
        for model in models():
            p = compile_plan(model, product=product)
            got = p.evaluate_sticky_moneyness_delta()
            self.assertEqual(p.time_nodes[-1], 1.0)
            self.assertEqual(got.price.value, p.evaluate().value)
            for eps in [1e-4, 5e-5]:
                up = compile_plan(model, 100.0+eps, product=product)
                dn = compile_plan(model, 100.0-eps, product=product)
                expected = (up.evaluate().value-dn.evaluate().value)/(2*eps)
                self.assertAlmostEqual(got.delta, expected, delta=3e-8*(1+abs(expected)))
                self.assertLess(max(abs(a-b) for a,b in zip(p.squared_leverage, up.squared_leverage)), 2e-13)

    def test_frozen_asian_delta_and_pair_standard_error_independent_reconstruction(self):
        product = asian_product()
        for model in models():
            p = compile_plan(model, product=product, flat=True, qmc=False)
            got = p.evaluate_frozen_leverage_delta()
            values = asian_frozen_reference(p, model, 0.0)
            mean, se = mean_se(values)
            self.assertAlmostEqual(got.price.value, mean, delta=2e-11)
            self.assertAlmostEqual(got.price.standard_error, se, delta=2e-11)
            for eps in [1e-4, 5e-5]:
                up = asian_frozen_reference(p, model, eps)
                dn = asian_frozen_reference(p, model, -eps)
                delta, delta_se = mean_se([(u-d)/(2*eps) for u,d in zip(up,dn)])
                self.assertAlmostEqual(got.delta, delta, delta=3e-8*(1+abs(delta)))
                self.assertAlmostEqual(got.delta_standard_error, delta_se, delta=3e-8*(1+abs(delta_se)))

    def test_discontinuous_payoff_requires_smoothing_and_smoothed_delta_matches_bumps(self):
        digital = rp.Product.digital(1, 2, "2027-09-04", 100.0, 1.0, "call", "cash")
        for model in models():
            hard = compile_plan(model, product=digital)
            self.assertTrue(math.isfinite(hard.evaluate().value))
            for name in ["evaluate_frozen_leverage_delta", "evaluate_sticky_moneyness_delta"]:
                with self.assertRaises(rp.PricingError): getattr(hard, name)()
            soft = compile_plan(model, product=digital, smooth=2.0)
            self.assertTrue(math.isfinite(soft.evaluate_frozen_leverage_delta().delta))
            result = soft.evaluate_sticky_moneyness_delta()
            for e in [1e-4, 5e-5]:
                up = compile_plan(model, 100.0+e, product=digital, smooth=2.0).evaluate().value
                dn = compile_plan(model, 100.0-e, product=digital, smooth=2.0).evaluate().value
                self.assertAlmostEqual(result.delta, (up-dn)/(2*e), delta=3e-8)

    def test_spatial_kink_is_rejected_only_when_lookup_coordinate_moves(self):
        for model in models(stochastic=False):
            # Initial zero lies at a knot with unequal neighboring slopes.
            p = compile_plan(model, xs=[-0.6, -0.2, 0.0, 0.45, 0.8])
            self.assertTrue(math.isfinite(p.evaluate().value))
            with self.assertRaises(rp.PricingError): p.evaluate_frozen_leverage_delta()
            self.assertTrue(math.isfinite(p.evaluate_sticky_moneyness_delta().delta))


if __name__ == "__main__":
    unittest.main()
