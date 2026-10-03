"""Immutable bindings, staged limits and independently known constant-variance IV."""
import unittest
import rust_pricing as rp


def problem(targets=(.22,), **config):
    model = rp.RoughVolatilityModel.rough_heston(hurst=.1, initial_variance=.04, mean_reversion=0., long_run_variance=.055, vol_of_vol=0., correlation=-.65)
    quotes = [rp.HestonIvCalibrationQuote.create(1., 100., 100., .97, target) for target in targets]
    variables = [rp.HestonCalibrationVariable.create('initial_variance', .01, .16, .04)]
    return rp.HestonIvCalibrationProblem.compile(model, quotes, variables,
        time_steps=config.get('time_steps', 16), integration_intervals=config.get('integration_intervals', 128), cutoff=64.)


class IvRefinementTests(unittest.TestCase):
    def test_exact_validation_and_early_stop(self):
        p = problem()
        policy = rp.HestonIvRefinementOptions.create()
        r = p.calibrate_refined(policy, max_iterations=40, max_evaluations=60)
        self.assertTrue(r.accepted)
        self.assertEqual(r.termination, 'validation_accepted')
        self.assertEqual(len(r.stages), 1)
        s = r.stages[0]
        self.assertTrue(s.calibration.fit_achieved)
        self.assertEqual(s.initial_parameters, [.04])
        self.assertAlmostEqual(s.calibration.parameters[0], .22**2, delta=2e-7)
        v = s.validation
        self.assertEqual(v.grid_names, ['base', 'time', 'frequency', 'cutoff', 'joint'])
        self.assertEqual(v.configurations, [(16,128,64.), (32,128,64.), (16,256,64.), (16,256,128.), (32,512,128.)])
        self.assertEqual(v.fit_tolerance, policy.fit_tolerance)
        self.assertTrue(v.fit_within_tolerance and v.grid_stable and v.accepted)
        self.assertEqual(r.optimizer_evaluations, s.calibration.evaluations)
        self.assertEqual(r.validation_plan_compilations, 5)
        self.assertEqual(v.plan_compilations, 5)
        standalone = p.validate_grid(s.calibration.parameters, policy)
        self.assertEqual(v.iv_differences, standalone.iv_differences)
        self.assertEqual(p.initial_parameters, [.04])

    def test_stage_limit_and_immutable_history(self):
        p = problem((.2, .3))
        policy = rp.HestonIvRefinementOptions.create(max_stages=2)
        r = p.calibrate_refined(policy, max_iterations=40, max_evaluations=60)
        self.assertFalse(r.accepted)
        self.assertEqual(r.termination, 'stage_limit')
        self.assertEqual(len(r.stages), 2)
        a, b = r.stages
        self.assertEqual(a.calibration.parameters, b.initial_parameters)
        self.assertEqual(a.validation.configurations[-1], b.validation.configurations[0])
        self.assertFalse(a.calibration.fit_achieved)
        self.assertFalse(b.calibration.fit_achieved)
        self.assertAlmostEqual(b.calibration.parameters[0], .0625, delta=2e-7)
        self.assertEqual(r.optimizer_evaluations, sum(s.calibration.evaluations for s in r.stages))
        self.assertEqual(r.validation_plan_compilations, 10)
        for obj, field in [(policy,'max_stages'), (r,'accepted'), (b,'initial_parameters'), (b.validation,'iv_residuals')]:
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(obj, field, None)
        copied = b.validation.iv_residuals
        copied[0][0] = 123.
        self.assertNotEqual(copied, b.validation.iv_residuals)

    def test_validation_is_absolute_iv_not_scaled_residual(self):
        p = problem((.201,))
        policy = rp.HestonIvRefinementOptions.create(max_stages=1)
        v = p.validate_grid([.04], policy)
        self.assertAlmostEqual(v.max_abs_iv_residual, .001, delta=2e-12)
        self.assertLess(v.max_abs_iv_difference, 2e-12)
        self.assertTrue(v.grid_stable)
        self.assertFalse(v.fit_within_tolerance or v.accepted)
        for j in range(5):
            self.assertAlmostEqual(v.model_implied_volatilities[0][j], .2, delta=2e-12)
            self.assertEqual(v.iv_residuals[0][j], v.model_implied_volatilities[0][j]-.201)

    def test_invalid_policy_and_future_grid_fail_explicitly(self):
        for kw in [dict(max_stages=0),dict(max_stages=4),dict(fit_tolerance=float('nan')),dict(grid_tolerance=0.)]:
            with self.subTest(kw=kw), self.assertRaises(rp.ValidationError):
                rp.HestonIvRefinementOptions.create(**kw)
        p = problem((.2,), time_steps=32, integration_intervals=2048)
        with self.assertRaises(rp.ValidationError):
            p.calibrate_refined(rp.HestonIvRefinementOptions.create())
        with self.assertRaises(rp.ValidationError):
            problem().validate_grid([float('nan')], rp.HestonIvRefinementOptions.create())

if __name__ == '__main__':
    unittest.main()
