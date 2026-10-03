"""Disjoint validation, fixed parameters, immutable reports and no optimizer feedback."""
import unittest
import rust_pricing as rp


def quote(t, k, iv=.2, **kwargs):
    return rp.HestonIvCalibrationQuote.create(t, kwargs.get('forward',100.), k,
        kwargs.get('discount',.97), iv, kwargs.get('iv_scale',1.), is_call=kwargs.get('is_call',k>=100.))


def model(lift=False, random=False):
    values=dict(initial_variance=.04,mean_reversion=.7 if random else 0.,
        long_run_variance=.055,vol_of_vol=.18 if random else 0.,correlation=-.65)
    if lift:
        return rp.RoughVolatilityModel.lifted_heston(**values,weights=[.2,.4,.5],rates=[.1,1.,8.])
    return rp.RoughVolatilityModel.rough_heston(hurst=.1,**values)


def problem(lift=False, random=False, quotes=None):
    return rp.HestonIvCalibrationProblem.compile(model(lift,random),quotes or [quote(1.,100.,.22)],
        [rp.HestonCalibrationVariable.create('initial_variance',.01,.16,.04)],
        time_steps=16,integration_intervals=128,cutoff=64.)


class HoldoutTests(unittest.TestCase):
    def test_absolute_iv_flags_and_immutable_reports(self):
        p=problem();policy=rp.HestonIvRefinementOptions.create()
        qs=[quote(.5,95.,.25),quote(1.,110.,.3),quote(.5,105.,.25)]
        r=p.validate_holdout([.0625],qs,policy)
        self.assertEqual(r.plan_compilations,10)
        self.assertTrue(r.grid_stable);self.assertFalse(r.accepted)
        self.assertAlmostEqual(r.max_abs_iv_residual,.05,delta=2e-12)
        for row in r.model_implied_volatilities:
            for v in row:self.assertAlmostEqual(v,.25,delta=2e-12)
        with self.assertRaises(AttributeError):r.accepted=True
        copy=r.iv_residuals;copy[0][0]=999.
        self.assertNotEqual(copy,r.iv_residuals)
        self.assertEqual(p.initial_parameters,[.04])
        shifted=[quote(.5,95.,.25,iv_scale=1000.),quote(1.,110.,.3,iv_scale=1000.),quote(.5,105.,.25,iv_scale=1000.)]
        self.assertEqual(r.iv_residuals,p.validate_holdout([.0625],shifted,policy).iv_residuals)

    def test_overlap_duplicate_and_invalid_inputs_are_errors(self):
        p=problem();policy=rp.HestonIvRefinementOptions.create()
        for qs in [[],[quote(1.,200.,.3,forward=200.,discount=.8,is_call=False)],
                   [quote(.5,95.),quote(.5,285.,forward=300.)]]:
            with self.subTest(quotes=len(qs)),self.assertRaises(rp.ValidationError):
                p.validate_holdout([.04],qs,policy)
        for params in [[],[float('nan')],[.001],[.04,.04]]:
            with self.subTest(params=params),self.assertRaises(rp.ValidationError):
                p.validate_holdout(params,[quote(.5,95.)],policy)
        with self.assertRaises(rp.ValidationError):
            p.validate_holdout([.04],[quote(.5,95.)]*4097,policy)
        self.assertTrue(p.validate_holdout([.04],[quote(1.+1e-9,100.)],policy).accepted)

    def test_holdout_mismatch_does_not_change_calibration(self):
        p=problem();policy=rp.HestonIvRefinementOptions.create()
        a=p.calibrate(max_iterations=40,max_evaluations=60)
        self.assertTrue(a.fit_achieved)
        r=p.validate_holdout(a.parameters,[quote(.5,100.,.4)],policy)
        self.assertFalse(r.accepted)
        b=p.calibrate(max_iterations=40,max_evaluations=60)
        self.assertEqual(a.parameters,b.parameters)
        self.assertEqual(a.evaluations,b.evaluations)
        self.assertEqual(a.evaluation.model_implied_volatilities,b.evaluation.model_implied_volatilities)
        self.assertTrue(p.validate_grid(a.parameters,policy).accepted)

    def test_public_repricing_and_quote_order(self):
        policy=rp.HestonIvRefinementOptions.create()
        qs=[quote(.5,95.),quote(.75,105.)]
        for lift in [False,True]:
            p=problem(lift,True)
            r=p.validate_holdout([.055],qs,policy)
            rev=p.validate_holdout([.055],list(reversed(qs)),policy)
            self.assertEqual(r.iv_residuals,list(reversed(rev.iv_residuals)))
            for col,(n,m,u) in enumerate(r.configurations):
                other=rp.HestonIvCalibrationProblem.compile(model(lift,True),qs,
                    [rp.HestonCalibrationVariable.create('initial_variance',.01,.16,.04)],
                    time_steps=n,integration_intervals=m,cutoff=u)
                ivs=other.evaluate([.055]).model_implied_volatilities
                self.assertEqual(ivs,[row[col] for row in r.model_implied_volatilities])

if __name__=='__main__':unittest.main()
