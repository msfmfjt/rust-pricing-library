"""Public calibration API: weighting, Jacobians, fit status and packaging."""
import ast
import importlib.util
import math
from pathlib import Path
import unittest
import rust_pricing as rp

ROOT=Path(__file__).resolve().parents[2]
CFG=dict(time_steps=32,integration_intervals=128,cutoff=48.0)
P=dict(hurst=.1,initial_variance=.04,mean_reversion=.7,long_run_variance=.055,vol_of_vol=.18,correlation=-.65)
def model(**kw):return rp.RoughVolatilityModel.rough_heston(**(P|kw))
def quotes(m):
    out=[]
    for t in [.25,.75,1.5]:
        plan=rp.HestonFourierPlan.compile(m,t,**CFG)
        for k in [85.,100.,115.]:
            p=plan.price(100.,k,.97);call=k>=100
            out.append(rp.HestonCalibrationQuote.create(t,100.,k,.97,p.call if call else p.put,.1,is_call=call))
    return out
class HestonCalibrationTests(unittest.TestCase):
    def test_known_variance_fit_and_frozen_results(self):
        qs=quotes(model(mean_reversion=0.,vol_of_vol=0.))
        var=rp.HestonCalibrationVariable.create('initial_variance',.01,.1,.04)
        p=rp.HestonCalibrationProblem.compile(model(initial_variance=.06,mean_reversion=0.,vol_of_vol=0.),qs,[var],**CFG)
        r=p.calibrate()
        self.assertTrue(r.fit_achieved);self.assertEqual(r.termination,'residual_tolerance')
        self.assertEqual(r.parameter_names,['initial_variance']);self.assertEqual(p.maturity_count,3)
        self.assertAlmostEqual(r.parameters[0],.04,delta=1e-8)
        self.assertEqual(r.evaluation.model_prices,p.evaluate(r.parameters).model_prices)
        self.assertTrue(all(b<a for a,b in zip(r.accepted_objectives,r.accepted_objectives[1:])))
        for obj,field in [(r,'fit_achieved'),(r,'parameters'),(r.evaluation,'jacobian'),(qs[0],'price_scale'),(var,'scale')]:
            with self.assertRaises(AttributeError):setattr(obj,field,0)
    def test_hurst_jacobian_two_bumps_and_quote_order(self):
        qs=quotes(model());qs.reverse()
        variables=[rp.HestonCalibrationVariable.create('hurst',.02,.49,.2),rp.HestonCalibrationVariable.create('initial_variance',.01,.1,.04)]
        p=rp.HestonCalibrationProblem.compile(model(),qs,variables,**CFG)
        self.assertEqual(p.parameter_names,['hurst','initial_variance'])
        e=p.evaluate(p.initial_parameters);self.assertEqual(e.scaled_residuals,[0.]*len(qs))
        for j in [0,1]:
            for d in [1e-5,5e-6]:
                a=p.initial_parameters;b=p.initial_parameters;a[j]+=d;b[j]-=d
                up=p.evaluate(a);down=p.evaluate(b)
                for i,row in enumerate(e.jacobian):self.assertAlmostEqual(row[j],(up.scaled_residuals[i]-down.scaled_residuals[i])/(2*d),delta=3e-4)
    def test_bound_and_budget_stops_are_not_fits(self):
        qs=quotes(model(mean_reversion=0.,vol_of_vol=0.))
        p=rp.HestonCalibrationProblem.compile(model(initial_variance=.06,mean_reversion=0.,vol_of_vol=0.),qs,[rp.HestonCalibrationVariable.create('initial_variance',.05,.1,.04)],**CFG)
        r=p.calibrate();self.assertFalse(r.fit_achieved);self.assertEqual(r.active_bounds,[True]);self.assertEqual(r.parameters,[.05])
        r=p.calibrate(max_evaluations=2);self.assertFalse(r.fit_achieved);self.assertEqual(r.evaluations,2);self.assertEqual(r.termination,'max_evaluations')
    def test_invalid_quotes_bounds_parameters_and_lift_hurst(self):
        qs=quotes(model());v=rp.HestonCalibrationVariable.create('hurst',.02,.49,.2)
        with self.assertRaises(rp.ValidationError):rp.HestonCalibrationVariable.create('H',.02,.49,.2)
        with self.assertRaises(rp.ValidationError):rp.HestonCalibrationProblem.compile(model(),qs,[v,v],**CFG)
        bad=rp.HestonCalibrationQuote.create(.25,100.,100.,.97,-1.)
        with self.assertRaises(rp.ValidationError):rp.HestonCalibrationProblem.compile(model(),[bad],[v],**CFG)
        args={k:value for k,value in P.items() if k!='hurst'}
        lift=rp.RoughVolatilityModel.lifted_heston(**args,weights=[1.],rates=[0.])
        with self.assertRaises(rp.ValidationError):rp.HestonCalibrationProblem.compile(lift,qs,[v],**CFG)
        p=rp.HestonCalibrationProblem.compile(model(),qs,[v],**CFG)
        for point in [[],[float('nan')],[.5]]:
            with self.assertRaises(rp.ValidationError):p.evaluate(point)
        with self.assertRaises(rp.ValidationError):p.calibrate(max_evaluations=1)
    def test_all_new_stub_members_are_mandatory(self):
        spec=importlib.util.spec_from_file_location('smoke',ROOT/'scripts/smoke_test_wheel.py')
        checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
        source=(ROOT/'rust_pricing.pyi').read_text();checker.verify_stub_static_shape(ast.parse(source))
        for cls in [n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name.startswith('HestonCalibration')]:
            for name in [None]+[n.name for n in cls.body if isinstance(n,ast.FunctionDef)]:
                tree=ast.parse(source)
                if name is None:tree.body=[n for n in tree.body if not isinstance(n,ast.ClassDef) or n.name!=cls.name]
                else:
                    target=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls.name)
                    target.body=[n for n in target.body if not isinstance(n,ast.FunctionDef) or n.name!=name]
                with self.subTest(cls=cls.name,member=name),self.assertRaises(RuntimeError):checker.verify_stub_static_shape(tree)
if __name__=='__main__':unittest.main()
