"""Public exact-IV fit and SSVI adapter, with independent input references."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import unittest
import rust_pricing as rp

ROOT=Path(__file__).resolve().parents[2]
CFG=dict(time_steps=32,integration_intervals=256,cutoff=64.)
P=dict(hurst=.1,initial_variance=.04,mean_reversion=.7,long_run_variance=.055,vol_of_vol=.18,correlation=-.65)
def model(**kw):return rp.RoughVolatilityModel.rough_heston(**(P|kw))
def quotes():
    return [rp.HestonIvCalibrationQuote.create(t,100.,k,.97,.2,is_call=call)
            for t in [.25,1.] for k in [90.,100.,110.] for call in [False,True]]
def variable(name='initial_variance',lo=.01,hi=.1,scale=.04):
    return rp.HestonCalibrationVariable.create(name,lo,hi,scale)
class HestonIvCalibrationTests(unittest.TestCase):
    def test_constant_variance_fit_and_units(self):
        p=rp.HestonIvCalibrationProblem.compile(model(initial_variance=.06,mean_reversion=0.,vol_of_vol=0.),quotes(),[variable()],**CFG)
        r=p.calibrate(residual_tolerance=1e-9)
        self.assertTrue(r.fit_achieved);self.assertEqual(r.termination,'residual_tolerance')
        self.assertAlmostEqual(r.parameters[0],.04,delta=1e-9)
        self.assertEqual(r.parameter_names,['initial_variance']);self.assertEqual(p.maturity_count,2)
        e=r.evaluation;self.assertEqual(e.model_prices,p.evaluate(r.parameters).model_prices)
        for sigma,row in zip(e.model_implied_volatilities,e.jacobian):
            self.assertAlmostEqual(sigma,.2,delta=1e-9)
            self.assertAlmostEqual(row[0],2.5,delta=1e-8)
        for obj,field in [(r,'fit_achieved'),(e,'model_vegas'),(quotes()[0],'iv_scale'),(p,'maturity_count')]:
            with self.assertRaises(AttributeError):setattr(obj,field,0)
    def test_hurst_and_scalar_iv_jacobian(self):
        vs=[variable('hurst',.02,.49,.2),variable()]
        qs=quotes()[::-1]
        p=rp.HestonIvCalibrationProblem.compile(model(),qs,vs,**CFG)
        e=p.evaluate(p.initial_parameters)
        for j in [0,1]:
            for h in [1e-5,5e-6]:
                a=p.initial_parameters;b=p.initial_parameters;a[j]+=h;b[j]-=h
                u=p.evaluate(a);d=p.evaluate(b)
                for i,row in enumerate(e.jacobian):
                    self.assertAlmostEqual(row[j],(u.scaled_residuals[i]-d.scaled_residuals[i])/(2*h),delta=1e-5)
        qs_scaled=[rp.HestonIvCalibrationQuote.create(q.maturity,q.forward,q.strike,q.discount,q.target_volatility,.01,is_call=q.is_call) for q in qs]
        s=rp.HestonIvCalibrationProblem.compile(model(),qs_scaled,vs,**CFG).evaluate(p.initial_parameters)
        for a,b in zip(e.scaled_residuals,s.scaled_residuals):self.assertAlmostEqual(100*a,b,delta=1e-11)
        for a,b in zip(e.jacobian,s.jacobian):
            for x,y in zip(a,b):self.assertAlmostEqual(100*x,y,delta=1e-9)
    def test_ssvi_both_shapes_interpolation_and_otm_side(self):
        f=json.loads((ROOT/'fixtures/rough-volatility/iv-calibration.json').read_text())
        shapes={
            'power_law':rp.HestonCalibrationSsviSurface.power_law([.25,.75,1.5],[.012,.035,.072],.048,-.5,.35,.5),
            'heston_like':rp.HestonCalibrationSsviSurface.heston_like([.25,.75,1.5],[.012,.035,.072],.048,-.5,1.),
        }
        for row in f['ssvi_quotes']:
            q=shapes[row['kind']].quote(row['maturity'],row['forward'],row['strike'],row['discount'])
            self.assertAlmostEqual(q.target_volatility,row['target_volatility'],delta=2e-12)
            self.assertEqual(q.is_call,row['is_call'])
        with self.assertRaises(rp.ValidationError):rp.HestonCalibrationSsviSurface.power_law([.25,.75],[.03,.01],.01,-.5,.35,.5)
        with self.assertRaises(rp.ValidationError):rp.HestonCalibrationSsviSurface.power_law([.25,.75],[.01,.03],.01,-.5,20.,.5)
    def test_invalid_boundaries_and_nonfits(self):
        for vol in [0.,-1.,math.nan,math.inf,9.]:
            with self.assertRaises(rp.ValidationError):rp.HestonIvCalibrationQuote.create(1.,100.,100.,1.,vol)
        with self.assertRaises(rp.ValidationError):rp.HestonIvCalibrationQuote.create(0.,100.,100.,1.,.2)
        p=rp.HestonIvCalibrationProblem.compile(model(initial_variance=.06,mean_reversion=0.,vol_of_vol=0.),quotes(),[variable(lo=.05)],**CFG)
        r=p.calibrate();self.assertFalse(r.fit_achieved);self.assertTrue(r.active_bounds[0]);self.assertEqual(r.parameters[0],.05)
        r=p.calibrate(max_evaluations=2);self.assertFalse(r.fit_achieved)
        for x in [[],[math.nan],[.01]]:
            with self.assertRaises(rp.ValidationError):p.evaluate(x)
    def test_each_new_stub_member_is_required(self):
        spec=importlib.util.spec_from_file_location('iv_smoke',ROOT/'scripts/smoke_test_wheel.py')
        checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
        source=(ROOT/'rust_pricing.pyi').read_text();checker.verify_stub_static_shape(ast.parse(source))
        for cls in [n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and (n.name.startswith('HestonIv') or n.name=='HestonCalibrationSsviSurface')]:
            for name in [None]+[n.name for n in cls.body if isinstance(n,ast.FunctionDef)]:
                tree=ast.parse(source)
                if name is None:tree.body=[n for n in tree.body if not isinstance(n,ast.ClassDef) or n.name!=cls.name]
                else:
                    target=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls.name)
                    target.body=[n for n in target.body if not isinstance(n,ast.FunctionDef) or n.name!=name]
                with self.subTest(cls=cls.name,member=name),self.assertRaises(RuntimeError):checker.verify_stub_static_shape(tree)
if __name__=='__main__':unittest.main()
