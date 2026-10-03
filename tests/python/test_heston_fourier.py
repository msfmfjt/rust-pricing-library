"""Public continuous-time transform/price bindings and packaging contracts."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import unittest
import rust_pricing as rp

ROOT=Path(__file__).resolve().parents[2]
def rough(h=0.1,nu=0.18):
    return rp.RoughVolatilityModel.rough_heston(hurst=h,initial_variance=0.04,
        mean_reversion=0.7,long_run_variance=0.055,vol_of_vol=nu,correlation=-0.65)

class HestonFourierTest(unittest.TestCase):
    def test_prices_oracle_parity_and_reusable_frozen_plan(self):
        refs=json.loads((ROOT/'fixtures/rough-volatility/fourier.json').read_text())
        plan=rp.HestonFourierPlan.compile(rough(0.5),1.0,time_steps=1024,
                                          integration_intervals=1024,cutoff=192.0)
        self.assertEqual((plan.time_steps,plan.integration_intervals,plan.cutoff),(1024,1024,192.0))
        for row in refs['prices']:
            if row['family']!='heston' or row['maturity']!=1.0: continue
            p=plan.price(100,row['strike'],0.97)
            self.assertAlmostEqual(p.call,row['call'],delta=0.001)
            self.assertAlmostEqual(p.call-p.put,0.97*(100-row['strike']),delta=1e-12)
            self.assertLess(p.quadrature_difference,1e-5)
            self.assertLess(p.tail_indicator,1e-5)
            self.assertEqual(p.call,plan.price(100,row['strike'],0.97).call)
            with self.assertRaises(AttributeError): p.call=0.0
        with self.assertRaises(AttributeError): plan.time_steps=4
    def test_rough_transforms_lift_and_domain(self):
        refs=json.loads((ROOT/'fixtures/rough-volatility/fourier.json').read_text())
        p=rp.HestonFourierPlan.compile(rough(),1.0,time_steps=2048,integration_intervals=8,cutoff=4.0)
        for row in refs['rough_transforms']:
            if row['hurst']==0.1 and row['maturity']==1.0:
                z=complex(*p.log_transform(*row['exponent']))
                self.assertLess(abs(z-complex(*row['log_transform'])),5e-5)
        self.assertEqual(p.log_transform(0.0,0.0),(0.0,0.0))
        self.assertEqual(p.log_transform(1.0,0.0),(0.0,0.0))
        self.assertEqual(p.characteristic_function(0.0),(1.0,0.0))
        lift=rp.RoughVolatilityModel.lifted_heston_from_rough(rough())
        q=rp.HestonFourierPlan.compile(lift,1.0)
        self.assertGreater(q.price(100,100,1).call,0.0)
        for args in [(-0.1,0.0),(1.1,0.0),(0.5,math.nan)]:
            with self.assertRaises(rp.ValidationError):p.log_transform(*args)
    def test_reject_invalid_and_nonaffine(self):
        for args in [dict(time_steps=0),dict(integration_intervals=9),dict(cutoff=math.nan)]:
            with self.assertRaises(rp.ValidationError):rp.HestonFourierPlan.compile(rough(),1,**args)
        for t in [-1.0,math.nan]:
            with self.assertRaises(rp.ValidationError):rp.HestonFourierPlan.compile(rough(),t)
        m=rp.RoughVolatilityModel.rough_sabr(hurst=0.1,vol_of_vol=0.2,correlation=-0.6,beta=1.0,forward_variance=rp.ForwardVarianceCurve.constant(0.04))
        with self.assertRaises(rp.PricingError):rp.HestonFourierPlan.compile(m,1.0)
        p=rp.HestonFourierPlan.compile(rough(),0.0)
        for f,k,d in [(0,100,1),(100,0,1),(100,100,0)]:
            with self.assertRaises(rp.ValidationError):p.price(f,k,d)
        self.assertEqual(p.price(100,90,0.9).call,9.0)
    def test_stub_mutations_are_rejected(self):
        spec=importlib.util.spec_from_file_location('fourier_wheel',ROOT/'scripts/smoke_test_wheel.py')
        checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
        stub=(ROOT/'rust_pricing.pyi').read_text()
        checker.verify_stub_static_shape(ast.parse(stub))
        for name in ['HestonFourierPlan','HestonFourierPrice']:
            tree=ast.parse(stub);tree.body=[n for n in tree.body if not isinstance(n,ast.ClassDef) or n.name!=name]
            with self.assertRaises(RuntimeError):checker.verify_stub_static_shape(tree)
        tree=ast.parse(stub)
        plan=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HestonFourierPlan')
        method=next(n for n in plan.body if isinstance(n,ast.FunctionDef) and n.name=='compile')
        method.args.kw_defaults[-1]=ast.Constant(512.0)
        with self.assertRaises(RuntimeError):checker.verify_stub_static_shape(tree)
