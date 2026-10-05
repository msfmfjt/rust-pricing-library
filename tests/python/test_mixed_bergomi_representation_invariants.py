"""Equivalent Mixed Bergomi representations must give equivalent finite-grid risks.

This tests coordinate changes, not a continuous-time pricing oracle.  In
particular, simplex gradient differences cannot be applied to marginal errors:
changing the balancing component changes the covariance needed for that error.
"""
import itertools
import json
import math
import unittest

import rust_pricing as rp
from test_mixed_bergomi_parameter_risk import lsv_build, pure_build


WEIGHTS = (0.2, 0.35, 0.45)
ETAS = (0.3, 0.8, 1.1)
CURVE_TIMES = (0.0, 0.4, 0.9, 1.4)
CURVE_VALUES = (0.04, 0.05, 0.045, 0.055)
# Declared before the first numerical execution. These are discrete identity
# checks; neither this tolerance nor a sampling error bounds continuum bias.
IDENTITY_TOLERANCE = 2e-9
BUMP_TOLERANCE = 3e-5
DIRECTION = (1.0 / 16.0, -1.0 / 32.0, -1.0 / 32.0)


def curve(values=CURVE_VALUES):
    return rp.ForwardVarianceCurve.piecewise_linear(CURVE_TIMES, values)


def model(hurst, weights=WEIGHTS, etas=ETAS, xi=None):
    return rp.RoughVolatilityModel.mixed_rough_bergomi(
        hurst=hurst, correlation=-0.6, weights=weights, vol_of_vols=etas,
        forward_variance=curve() if xi is None else xi,
    )


def simplex_change(values, order):
    """Transform gradients from last-component gauge to permuted last gauge."""
    gauge = list(values[:2]) + [0.0]
    return [gauge[order[i]] - gauge[order[-1]] for i in range(2)] + list(values[2:])


class MixedRepresentationInvariants(unittest.TestCase):
    def setUp(self):
        self.comparisons = 0
        self.gaps = {}

    def close(self, got, expected, group, tolerance=IDENTITY_TOLERANCE):
        self.assertTrue(math.isfinite(got) and math.isfinite(expected), group)
        gap = abs(got - expected)
        self.gaps[group] = max(self.gaps.get(group, 0.0), gap)
        self.comparisons += 1
        self.assertLessEqual(gap, tolerance * (1.0 + abs(expected)),
                             f"{group}: {got:.17g} versus {expected:.17g}")

    def vector_close(self, got, expected, group):
        self.assertEqual(len(got), len(expected), group)
        for a, b in zip(got, expected):
            self.close(a, b, group)

    def report(self, cases):
        print(json.dumps({"test": self._testMethodName, "cases": cases,
                          "scalar_comparisons": self.comparisons,
                          "maximum_absolute_gaps": self.gaps}, sort_keys=True))

    def test_component_permutations_change_simplex_gauge_not_economic_risk(self):
        cases = 0
        for h in (0.1, 0.5):
            for qmc in (False, True):
                for build in (pure_build, lsv_build):
                    base = build(model(h), qmc=qmc)
                    shape = base.evaluate_mixed_bergomi_shape_risk()
                    dynamic = base.evaluate_mixed_bergomi_parameter_risk_with_hurst()
                    fields = ["parameter_adjoints"]
                    if build is lsv_build:
                        fields += ["direct_adjoints", "calibration_adjoints"]
                    for order in itertools.permutations(range(3)):
                        with self.subTest(h=h, qmc=qmc, build=build.__name__, order=order):
                            w = [WEIGHTS[i] for i in order]
                            eta = [ETAS[i] for i in order]
                            other = build(model(h, w, eta), qmc=qmc)
                            sr = other.evaluate_mixed_bergomi_shape_risk()
                            dr = other.evaluate_mixed_bergomi_parameter_risk_with_hurst()
                            self.assertEqual(sr.parameter_names, shape.parameter_names)
                            self.assertEqual(dr.parameter_names, dynamic.parameter_names)
                            self.close(sr.price.value, shape.price.value, "price")
                            self.close(sr.price.standard_error, shape.price.standard_error, "price_se")
                            for field in fields:
                                self.vector_close(getattr(sr, field),
                                                  simplex_change(getattr(shape, field), order),
                                                  "simplex_" + field)
                                old = getattr(dynamic, field)
                                self.vector_close(getattr(dr, field),
                                                  [old[i] for i in order] + list(old[3:]),
                                                  "eta_rho_hurst_" + field)
                            if sr.standard_errors is not None:
                                self.vector_close(sr.standard_errors[2:], shape.standard_errors[2:], "xi_se")
                                old = dynamic.standard_errors
                                self.vector_close(dr.standard_errors,
                                                  [old[i] for i in order] + list(old[3:]),
                                                  "eta_rho_hurst_se")
                                # With the same balancing component, this is a
                                # true permutation of sampling channels.
                                if order[-1] == 2:
                                    self.vector_close(sr.standard_errors[:2],
                                                      [shape.standard_errors[i] for i in order[:2]],
                                                      "unchanged_gauge_se")
                                # Otherwise do NOT subtract marginal SEs.
                            else:
                                self.assertIsNone(shape.standard_errors)
                            if build is lsv_build:
                                self.assertEqual(other.extrapolated_moment_nodes, base.extrapolated_moment_nodes)
                                self.vector_close(other.squared_leverage, base.squared_leverage, "leverage")
                            direction = [DIRECTION[i] for i in order]
                            got = sum(a * d for a, d in zip(sr.parameter_adjoints[:2], direction[:2]))
                            expected = sum(a * d for a, d in zip(shape.parameter_adjoints[:2], DIRECTION[:2]))
                            self.close(got, expected, "physical_transfer_direction")
                            for bump in (1e-6, 5e-7):
                                prices = []
                                for sign in (1.0, -1.0):
                                    changed = [x + sign * bump * d for x, d in zip(w, direction)]
                                    prices.append(build(model(h, changed, eta), qmc=qmc).evaluate().value)
                                self.close(got, (prices[0] - prices[1]) / (2.0 * bump),
                                           "full_recalibration_direction", BUMP_TOLERANCE)
                            cases += 1
        self.report(cases)

    def test_splitting_an_identical_component_preserves_all_risk_scopes(self):
        cases = 0
        for h in (0.1, 0.5):
            for qmc in (False, True):
                for build in (pure_build, lsv_build):
                    with self.subTest(h=h, qmc=qmc, build=build.__name__):
                        base = build(model(h), qmc=qmc)
                        split = build(model(h, (0.2, 0.35, 0.18, 0.27), (0.3, 0.8, 1.1, 1.1)), qmc=qmc)
                        a = base.evaluate_mixed_bergomi_shape_risk()
                        b = split.evaluate_mixed_bergomi_shape_risk()
                        da = base.evaluate_mixed_bergomi_parameter_risk_with_hurst()
                        db = split.evaluate_mixed_bergomi_parameter_risk_with_hurst()
                        self.close(b.price.value, a.price.value, "price")
                        self.close(b.price.standard_error, a.price.standard_error, "price_se")
                        fields = ["parameter_adjoints"]
                        if build is lsv_build:
                            fields += ["direct_adjoints", "calibration_adjoints"]
                        for field in fields:
                            old = getattr(a, field)
                            self.vector_close(getattr(b, field), list(old[:2]) + [0.0] + list(old[2:]),
                                              "shape_" + field)
                            old = getattr(da, field)
                            new = getattr(db, field)
                            self.vector_close([new[0], new[1], new[2] + new[3], *new[4:]], old,
                                              "dynamic_" + field)
                        if a.standard_errors is not None:
                            self.vector_close(b.standard_errors[:2], a.standard_errors[:2], "transfer_se")
                            self.close(b.standard_errors[2], 0.0, "identical_transfer_se")
                            self.vector_close(b.standard_errors[3:], a.standard_errors[2:], "xi_se")
                            self.vector_close(db.standard_errors[4:], da.standard_errors[3:], "rho_hurst_se")
                            # Marginal split-eta SEs are NOT summed as independent.
                        if build is lsv_build:
                            self.assertEqual(split.extrapolated_moment_nodes, base.extrapolated_moment_nodes)
                            self.vector_close(split.squared_leverage, base.squared_leverage, "leverage")
                        cases += 1
        self.report(cases)

    def test_finite_xi_curve_changes_cancel_only_after_leverage_recalibration(self):
        cases = 0
        for h in (0.1, 0.5):
            for qmc in (False, True):
                for altered in (curve((0.02, 0.08, 0.03, 0.11)),
                                rp.ForwardVarianceCurve.exponential(0.08, -0.3)):
                    with self.subTest(h=h, qmc=qmc, curve=repr(altered)):
                        original = curve()
                        base_model = model(h, xi=original)
                        new_model = model(h, xi=altered)
                        base = lsv_build(base_model, qmc=qmc)
                        other = lsv_build(new_model, qmc=qmc)
                        self.assertEqual(base.time_nodes, other.time_nodes)
                        self.assertEqual(base.extrapolated_moment_nodes, other.extrapolated_moment_nodes)
                        before = base.evaluate()
                        after = other.evaluate()
                        self.close(after.value, before.value, "price")
                        self.close(after.standard_error, before.standard_error, "price_se")
                        width = len(base.log_moneyness_nodes)
                        self.assertEqual(len(base.squared_leverage), width * len(base.time_nodes))
                        for row, t in enumerate(base.time_nodes):
                            for j in range(width):
                                index = row * width + j
                                self.close(other.squared_leverage[index] * altered.value(t),
                                           base.squared_leverage[index] * original.value(t),
                                           "squared_leverage_times_xi")
                        # A nonvacuous identity: neither xi nor Leverage was ignored.
                        self.assertGreater(max(abs(a-b) for a,b in zip(base.squared_leverage, other.squared_leverage)), 1e-4)
                        pure_before = pure_build(base_model, qmc=qmc).evaluate().value
                        pure_after = pure_build(new_model, qmc=qmc).evaluate().value
                        self.assertGreater(abs(pure_after - pure_before), 1e-3)
                        risk = other.evaluate_mixed_bergomi_shape_risk()
                        self.assertGreater(max(abs(x) for x in risk.direct_adjoints[2:]), 1.0)
                        for j in range(2, len(risk.parameter_names)):
                            self.close(risk.parameter_adjoints[j], 0.0, "total_xi_risk")
                            self.close(risk.direct_adjoints[j] + risk.calibration_adjoints[j], 0.0, "xi_contribution_sum")
                            if qmc:
                                self.close(risk.standard_errors[j], 0.0, "total_xi_se")
                        cases += 1
        self.report(cases)


if __name__ == "__main__":
    unittest.main()
