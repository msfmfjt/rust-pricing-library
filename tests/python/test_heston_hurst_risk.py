"""Public Python Hurst sensitivity, boundary and packaging contracts."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import unittest
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2]
P=dict(initial_variance=.04,mean_reversion=.7,long_run_variance=.055,vol_of_vol=.18,correlation=-.65)
def plan(h=.1,t=.25,**kwargs):
    model=rp.RoughVolatilityModel.rough_heston(hurst=h,**(P|kwargs))
    return rp.HestonFourierPlan.compile(model,t,time_steps=256,integration_intervals=512,cutoff=128.)
class HurstRiskTests(unittest.TestCase):
    def test_bumps_cached_prices_and_frozen_fields(self):
        p=plan();r=p.hurst_risk_plan()
        for k in [80.,100.,120.]:
            g=r.price(100.,k,.97)
            self.assertIsInstance(r,rp.HestonFourierHurstRiskPlan)
            self.assertIsInstance(g,rp.HestonFourierHurstRisk)
            self.assertEqual(g.price.call,p.price(100.,k,.97).call)
            self.assertEqual(g.price.put,p.price(100.,k,.97).put)
            for field in ['hurst_sensitivity','quadrature_difference','tail_indicator']:
                self.assertTrue(math.isfinite(getattr(g,field)))
                with self.assertRaises(AttributeError):setattr(g,field,0.)
            for step in [1e-4,5e-5]:
                a=plan(.1+step);b=plan(.1-step)
                for opt in ['call','put']:
                    fd=(getattr(a.price(100.,k,.97),opt)-getattr(b.price(100.,k,.97),opt))/(2*step)
                    self.assertAlmostEqual(g.hurst_sensitivity,fd,delta=5e-6)
        z=r.log_transform_derivative(.5,.7)
        up=plan(.1001).log_transform(.5,.7);down=plan(.0999).log_transform(.5,.7)
        for actual,a,b in zip(z,up,down):self.assertAlmostEqual(actual,(a-b)/.0002,delta=2e-8)
    def test_independent_nonflat_deterministic_variance_hurst_risk(self):
        ref=json.loads((ROOT/'fixtures/rough-volatility/hurst-risk.json').read_text())
        for h in [.1,.3]:
            r=plan(h,vol_of_vol=0.).hurst_risk_plan()
            for row in ref['deterministic']:
                if row['hurst']!=h or row['maturity']!=.25:continue
                g=r.price(100.,row['strike'],.97)
                self.assertAlmostEqual(g.hurst_sensitivity,row['derivative'],delta=1e-4)
    def test_boundaries_and_finite_lift_rejection(self):
        for k in [80.,100.,120.]:
            self.assertEqual(plan(t=0.).hurst_risk_plan().price(100.,k,.97).hurst_sensitivity,0.)
        r=plan(.5).hurst_risk_plan();d=.0001
        a=plan(.5).price(100.,100.,.97).call;b=plan(.5-d).price(100.,100.,.97).call;c=plan(.5-2*d).price(100.,100.,.97).call
        self.assertAlmostEqual(r.price(100.,100.,.97).hurst_sensitivity,(3*a-4*b+c)/(2*d),delta=5e-6)
        m=rp.RoughVolatilityModel.lifted_heston(**P,weights=[1.],rates=[0.])
        with self.assertRaises(rp.ValidationError):rp.HestonFourierPlan.compile(m,.25).hurst_risk_plan()
        for v in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(rp.ValidationError):r.price(v,100.,.97)
        with self.assertRaises(rp.ValidationError):r.log_transform_derivative(1.1,0.)
        for z in [(0.,0.),(1.,0.)]:self.assertEqual(r.log_transform_derivative(*z),(0.,0.))
    def test_new_classes_members_and_signatures_are_required(self):
        spec=importlib.util.spec_from_file_location('smoke',ROOT/'scripts/smoke_test_wheel.py')
        checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
        source=(ROOT/'rust_pricing.pyi').read_text();checker.verify_stub_static_shape(ast.parse(source))
        mutations=[('HestonFourierHurstRisk',None),('HestonFourierHurstRiskPlan',None),('HestonFourierPlan','hurst_risk_plan')]
        mutations += [('HestonFourierHurstRisk',p) for p in ['price','hurst_sensitivity','quadrature_difference','tail_indicator']]
        mutations += [('HestonFourierHurstRiskPlan',p) for p in ['price','log_transform_derivative']]
        for name,member in mutations:
            tree=ast.parse(source)
            if member is None:tree.body=[n for n in tree.body if not isinstance(n,ast.ClassDef) or n.name!=name]
            else:
                cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
                cls.body=[n for n in cls.body if not isinstance(n,ast.FunctionDef) or n.name!=member]
            with self.subTest(name=name,member=member),self.assertRaises(RuntimeError):checker.verify_stub_static_shape(tree)
if __name__=='__main__':unittest.main()
