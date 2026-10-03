"""Parameter-risk bindings, numerical contracts and stub mutations."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import unittest
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2]
NAMES=['initial_variance','mean_reversion','long_run_variance','vol_of_vol','correlation']
P=dict(zip(NAMES,[0.04,0.7,0.055,0.18,-0.65]))
def model(family='rough',**kwargs):
    p=P|kwargs
    if family=='lift':return rp.RoughVolatilityModel.lifted_heston(**p,weights=[0.2,0.4,0.5],rates=[0.1,1.0,8.0])
    return rp.RoughVolatilityModel.rough_heston(hurst=0.5 if family=='heston' else 0.1,**p)
def plan(m,t=.25):
    return rp.HestonFourierPlan.compile(m,t,time_steps=256,integration_intervals=512,cutoff=192.0)
class ParameterRiskTests(unittest.TestCase):
    def test_independent_references_frozen_results_and_cached_prices(self):
        refs=json.loads((ROOT/'fixtures/rough-volatility/parameter-risk.json').read_text())
        for family in ['heston','lift']:
            for t in [.25,1.0]:
                base=plan(model(family),t);p=base.parameter_risk_plan()
                for row in refs['rows']:
                    if row['family']!=family or row['maturity']!=t:continue
                    result=p.price(100,row['strike'],.97)
                    self.assertIsInstance(result,rp.HestonFourierParameterRisk)
                    for field in ['sensitivities','quadrature_differences','tail_indicators']:
                        obj=getattr(result,field);self.assertIsInstance(obj,rp.HestonParameterSensitivities)
                        for name in NAMES:
                            self.assertTrue(math.isfinite(getattr(obj,name)))
                            with self.assertRaises(AttributeError):setattr(obj,name,0.0)
                    for name,expected in zip(NAMES,row['derivatives']):
                        self.assertAlmostEqual(getattr(result.sensitivities,name),expected,delta=refs['protocol']['price_abs_tolerance'])
                    for name in ['call','put','quadrature_difference','tail_indicator']:
                        self.assertEqual(getattr(result.price,name),getattr(base.price(100,row['strike'],.97),name))
    def test_rough_price_bumps_and_transform_direction_order(self):
        base=plan(model());r=base.parameter_risk_plan();g=r.price(100,100,.97)
        z=r.log_transform_derivatives(.5,.7)
        self.assertEqual(len(z),5)
        for j,name in enumerate(NAMES):
            h=1e-5;up=plan(model(**{name:P[name]+h}));down=plan(model(**{name:P[name]-h}))
            for option in ['call','put']:
                fd=(getattr(up.price(100,100,.97),option)-getattr(down.price(100,100,.97),option))/(2*h)
                self.assertAlmostEqual(getattr(g.sensitivities,name),fd,delta=5e-6)
            for k,(u,d) in enumerate(zip(up.log_transform(.5,.7),down.log_transform(.5,.7))):
                self.assertAlmostEqual(z[j][k],(u-d)/(2*h),delta=2e-8)
    def test_boundaries_and_deterministic_black_variance_derivative(self):
        risk=plan(model(),0.0).parameter_risk_plan()
        for k in [80,100,120]:
            result=risk.price(100,k,.97)
            self.assertEqual([getattr(result.sensitivities,n) for n in NAMES],[0.0]*5)
        for v in [0.0,-1.0,float('nan'),float('inf')]:
            with self.assertRaises(rp.ValidationError):risk.price(v,100,.97)
        with self.assertRaises(rp.ValidationError):plan(model(initial_variance=0.0)).parameter_risk_plan()
        p=plan(model(mean_reversion=0.0,vol_of_vol=0.0))
        g=p.parameter_risk_plan().price(100,100,.97).sensitivities
        expected=.97*100*math.exp(-.5*.05**2)/math.sqrt(2*math.pi)*.25/(2*.1)
        self.assertAlmostEqual(g.initial_variance,expected,delta=2e-12)
        self.assertEqual(g.long_run_variance,0.0);self.assertEqual(g.correlation,0.0)
    def test_stub_members_and_signatures_are_mandatory(self):
        spec=importlib.util.spec_from_file_location('smoke',ROOT/'scripts/smoke_test_wheel.py')
        checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
        source=(ROOT/'rust_pricing.pyi').read_text();checker.verify_stub_static_shape(ast.parse(source))
        cases=[('HestonParameterSensitivities',None),('HestonFourierParameterRisk',None),
               ('HestonFourierParameterRiskPlan',None),('HestonFourierPlan','parameter_risk_plan')]
        cases += [('HestonParameterSensitivities',n) for n in NAMES]
        cases += [('HestonFourierParameterRisk',n) for n in ['price','sensitivities','quadrature_differences','tail_indicators']]
        cases += [('HestonFourierParameterRiskPlan',n) for n in ['price','log_transform_derivatives']]
        for name,member in cases:
            tree=ast.parse(source)
            if member is None:tree.body=[n for n in tree.body if not isinstance(n,ast.ClassDef) or n.name!=name]
            else:
                cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
                cls.body=[n for n in cls.body if not isinstance(n,ast.FunctionDef) or n.name!=member]
            with self.subTest(name=name,member=member),self.assertRaises(RuntimeError):checker.verify_stub_static_shape(tree)
if __name__=='__main__':unittest.main()
