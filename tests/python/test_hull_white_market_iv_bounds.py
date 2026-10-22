"""Quote-risk preflight must reject exact Dupire bounds while preserving prices."""
import json
import math
from pathlib import Path
import unittest

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]
BOUND_ERROR = "market_iv_local_variance_bound_kink"


def inputs(sigma=0.25, floor=1e-8, cap=4.0):
    target = rp.HullWhiteLsvTarget.from_market_iv(
        [0.5, 1.0], [-0.5, 0.5], [sigma] * 4,
        [0.0, 0.5, 1.0], [-0.5, 0.5], floor=floor, cap=cap,
    )
    product = rp.Product.european_vanilla(1, 2, "2027-09-04", 100.0, 1.0, "call")
    market = rp.Market.equity(
        2, 1, 100.0, rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95]),
        rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98]),
    )
    engine = rp.Engine.pseudo_monte_carlo(32, 71)
    rates = rp.HullWhiteModel(0.2, [0.0], [0.0])
    model = rp.PricingModelSpec(
        rp.EquityModelSpec.local_stochastic_volatility(
            target.model, rp.VolatilityFactorSpec.bergomi(1.0, 0.0, -0.5),
            rp.LsvParticleConfig(particle_count=32, calibration_seed=72,
                log_bandwidth=0.5, minimum_effective_samples=2.0, retain_reverse_trace=True),
            forward_density=target,
        ),
        rates=rp.RateModelSpec.hull_white(
            rates, equity_rate_correlation=0.0, vol_rate_correlations=[0.0]),
    )
    source = rp.PricingSpecification(
        "2026-09-04", product, market, model, engine, rp.RiskRequest(),
        rp.PricingNumerics(worker_threads=1, reduction_block_size=16),
    )
    request = rp.PricingRequest(
        "2026-09-04", product, market, target.model, engine, rp.RiskRequest())
    return target, rates, request, source


def basket_task(source, basket_target):
    value = json.loads((ROOT / "fixtures/task-v1/multi_asset_joint.golden.json").read_text())
    value["operation"] = {"type": "evaluate"}
    s = value["source"]
    target = json.loads(source.to_json())["model"]["equity"]["forward_density"]
    s["hull_white"]["volatility_times"] = [0.0]
    s["hull_white"]["volatilities"] = [0.0]
    if basket_target:
        s["local_correlation"]["target"] = target["local_variance_grid"]
        s["local_correlation"]["hull_white_target"] = target
    else:
        del s["local_correlation"]
        s["models"][0] = {"type": "local_volatility",
                          "local_variance_grid": target["local_variance_grid"]}
        s["hull_white"]["lsv_targets"][0] = {"type": "present", "value": target}
    return rp.PricingTask.from_json(json.dumps(value))


class HullWhiteMarketIvBoundsTests(unittest.TestCase):
    def assert_price_task_only(self, task):
        restored = rp.PricingTask.from_json(task.to_json())
        self.assertEqual(task.fingerprint, restored.fingerprint)
        plan = restored.compile()
        before = plan.evaluate().report.to_json()
        capability = plan.capability_for("aad")
        self.assertEqual(capability.status, "unsupported")
        self.assertIn(BOUND_ERROR, capability.detail)
        with self.assertRaisesRegex(rp.PricingError, BOUND_ERROR):
            plan.with_operation("aad")
        with self.assertRaisesRegex(rp.PricingError, BOUND_ERROR):
            restored.with_operation("aad").compile()
        self.assertEqual(before, plan.evaluate().report.to_json())

    def test_native_risk_rejects_both_bounds_and_keeps_price(self):
        for floor, cap in [(1e-8, 0.0625), (0.0625, 4.0)]:
            with self.subTest(floor=floor, cap=cap):
                target, rates, request, _ = inputs(floor=floor, cap=cap)
                plan = rp.HullWhiteEquityPlan.compile_lsv(
                    request, target, rates, vol_mean_reversion=1.0, vol_of_vol=0.0,
                    equity_vol_correlation=-0.5, equity_rate_correlation=0.0,
                    vol_rate_correlation=0.0, particle_count=32, calibration_seed=72,
                    log_bandwidth=0.5, minimum_effective_samples=2.0,
                    retain_reverse_trace=True, worker_threads=1, reduction_block_size=16,
                )
                before = plan.evaluate().value
                with self.assertRaisesRegex(rp.PricingError, BOUND_ERROR):
                    plan.evaluate_aad()
                self.assertEqual(before, plan.evaluate().value)

    def test_single_asset_task_preflight(self):
        for floor, cap in [(1e-8, 0.0625), (0.0625, 4.0)]:
            with self.subTest(floor=floor, cap=cap):
                source = inputs(floor=floor, cap=cap)[3]
                self.assert_price_task_only(rp.PricingTask.from_specification(source))

    def test_basket_marginal_and_local_correlation_preflight(self):
        for basket_target in [False, True]:
            sigma = 0.1875 if basket_target else 0.25
            for floor, cap in [(1e-8, sigma * sigma), (sigma * sigma, 4.0)]:
                with self.subTest(basket_target=basket_target, floor=floor, cap=cap):
                    source = inputs(sigma, floor, cap)[3]
                    self.assert_price_task_only(basket_task(source, basket_target))

    def test_adjacent_interior_bounds_keep_aad_supported(self):
        source = inputs(floor=math.nextafter(0.0625, 0.0),
                        cap=math.nextafter(0.0625, math.inf))[3]
        task = rp.PricingTask.from_specification(source)
        plan = task.compile()
        self.assertEqual(plan.capability_for("aad").status, "supported")
        risk = plan.with_operation("aad").evaluate().native_result
        self.assertTrue(all(math.isfinite(v) for v in risk.vega_kt_raw))
        self.assertEqual(risk.price.value, plan.evaluate().report.value)


if __name__ == "__main__":
    unittest.main()
