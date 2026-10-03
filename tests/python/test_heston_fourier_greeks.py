"""Forward-only analytic Greeks and additive Python/wheel contracts."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import unittest
import rust_pricing as rp

ROOT=Path(__file__).resolve().parents[2]

def model(h=0.1,v0=0.04,kappa=0.7,theta=0.055,nu=0.18):
    return rp.RoughVolatilityModel.rough_heston(hurst=h,initial_variance=v0,
        mean_reversion=kappa,long_run_variance=theta,vol_of_vol=nu,correlation=-0.65)

def plan(m,t=1.0):
    return rp.HestonFourierPlan.compile(m,t,time_steps=256,integration_intervals=512,cutoff=192.0)

class HestonFourierGreeksTest(unittest.TestCase):
    def test_independent_reference_nested_price_and_frozen_results(self):
        refs=json.loads((ROOT/'fixtures/rough-volatility/fourier-greeks.json').read_text())
        for t in [0.25,1.0]:
            p=plan(model(0.5),t)
            for row in refs['rows']:
                if row['family']!='heston' or row['maturity']!=t:continue
                g=p.price_and_greeks(100.0,row['strike'],0.97)
                self.assertIsInstance(g,rp.HestonFourierGreeks)
                self.assertIsInstance(g.price,rp.HestonFourierPrice)
                for key in ['call_forward_delta','put_forward_delta','forward_gamma']:
                    tol=refs['protocol']['gamma_abs_tolerance' if key=='forward_gamma' else 'delta_abs_tolerance']
                    self.assertAlmostEqual(getattr(g,key),row[key],delta=tol)
                    with self.assertRaises(AttributeError):setattr(g,key,0.0)
                for key in ['call','put','quadrature_difference','tail_indicator']:
                    self.assertEqual(getattr(g.price,key),getattr(p.price(100,row['strike'],0.97),key))
                with self.assertRaises(AttributeError):g.price.call=0.0
                with self.assertRaises(AttributeError):g.price=p.price(100,100,0.97)
                for key in ['delta_quadrature_difference','gamma_quadrature_difference','delta_tail_indicator','gamma_tail_indicator']:
                    self.assertTrue(math.isfinite(getattr(g,key)))
                    self.assertGreaterEqual(getattr(g,key),0)
                    with self.assertRaises(AttributeError):setattr(g,key,1.0)

    def test_rough_and_lift_parity_scaling_and_price_bumps(self):
        m=model()
        for m in [m,rp.RoughVolatilityModel.lifted_heston_from_rough(m,factors=20,ratio=2.5)]:
            p=plan(m)
            for k in [80.0,100.0,120.0]:
                g=p.price_and_greeks(100,k,0.97)
                self.assertAlmostEqual(g.call_forward_delta-g.put_forward_delta,0.97,delta=2e-15)
                s=p.price_and_greeks(300,3*k,0.97)
                self.assertAlmostEqual(s.call_forward_delta,g.call_forward_delta,delta=1e-12)
                self.assertAlmostEqual(3*s.forward_gamma,g.forward_gamma,delta=1e-12)
                for h in [0.04,0.02]:
                    up=p.price(100+h,k,0.97);down=p.price(100-h,k,0.97)
                    self.assertAlmostEqual((up.call-down.call)/(2*h),g.call_forward_delta,delta=3e-6)
                    self.assertAlmostEqual((up.call-2*g.price.call+down.call)/(h*h),g.forward_gamma,delta=1e-6)

    def test_zero_variance_kinks_and_zero_control_rejected(self):
        for p in [plan(model(),0.0),plan(model(v0=0.0,kappa=0.0))]:
            with self.assertRaises(rp.ValidationError):p.price_and_greeks(100,100,0.97)
            for f in [90,110]:
                g=p.price_and_greeks(f,100,0.97)
                self.assertEqual(g.call_forward_delta,0.97 if f>100 else 0.0)
                self.assertEqual(g.forward_gamma,0.0)
            for bad in [0.0,-1.0,math.nan,math.inf]:
                for args in [(bad,100,1),(100,bad,1),(100,100,bad)]:
                    with self.assertRaises(rp.ValidationError):p.price_and_greeks(*args)
        p=plan(model(0.5,v0=0.0,nu=0.0))
        self.assertGreater(p.price(100,100,0.97).call,0)
        with self.assertRaises(rp.ValidationError):p.price_and_greeks(100,100,0.97)

    def test_wheel_stub_rejects_missing_method_properties_and_class(self):
        spec=importlib.util.spec_from_file_location('greeks_wheel',ROOT/'scripts/smoke_test_wheel.py')
        checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
        source=(ROOT/'rust_pricing.pyi').read_text()
        checker.verify_stub_static_shape(ast.parse(source))
        for cname,member in [('HestonFourierGreeks',None),('HestonFourierPlan','price_and_greeks')]+[
            ('HestonFourierGreeks',key) for key in ['price','call_forward_delta','put_forward_delta','forward_gamma',
                'delta_quadrature_difference','gamma_quadrature_difference','delta_tail_indicator','gamma_tail_indicator']]:
            tree=ast.parse(source)
            if member is None:
                tree.body=[n for n in tree.body if not isinstance(n,ast.ClassDef) or n.name!=cname]
            else:
                cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cname)
                cls.body=[n for n in cls.body if not isinstance(n,ast.FunctionDef) or n.name!=member]
            with self.subTest(cname=cname,member=member),self.assertRaises(RuntimeError):checker.verify_stub_static_shape(tree)

if __name__=='__main__':unittest.main()
