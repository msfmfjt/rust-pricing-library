"""Runtime, independent-reference and packaging contracts for deterministic Fourier pricing."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import unittest
import rust_pricing as rp

ROOT=Path(__file__).resolve().parents[2]
P=dict(initial_variance=0.04,mean_reversion=0.7,long_run_variance=0.055,vol_of_vol=0.18,correlation=-0.65)

def model(name):
    if name=='lifted':
        return rp.RoughVolatilityModel.lifted_heston(**P,weights=[0.6,1.1,0.8],rates=[0.2,1.5,12.0])
    return rp.RoughVolatilityModel.rough_heston(**P,hurst=0.5 if name=='classical' else float(name.removeprefix('rough_')))

class HestonFourierTest(unittest.TestCase):
    def test_independent_transform_and_price_references(self):
        data=json.loads((ROOT/'fixtures/rough-volatility/fourier.json').read_text())
        plans={}
        for row in data['transforms']:
            key=(row['family'],row['maturity'])
            if key not in plans: plans[key]=rp.HestonFourierPlan.compile(model(key[0]),key[1],time_steps=1024,integration_intervals=1024,cutoff=160.)
            z=complex(*plans[key].transform(row['damping'],row['frequency']))
            self.assertLess(abs(z-complex(row['real'],row['imag'])),data['protocol']['transform_tolerance'])
        for row in data['prices']:
            key=(row['family'],row['maturity'])
            if key not in plans: plans[key]=rp.HestonFourierPlan.compile(model(key[0]),key[1],time_steps=1024,integration_intervals=1024,cutoff=160.)
            p=plans[key];call=p.price(row['forward'],row['strike'],discount=row['discount'])
            self.assertAlmostEqual(call,row['call'],delta=data['protocol']['price_tolerance'])
            put=p.price(row['forward'],row['strike'],discount=row['discount'],is_call=False)
            self.assertAlmostEqual(call-put,row['discount']*(row['forward']-row['strike']),delta=2e-13)

    def test_immutable_plans_and_refinement_coordinates(self):
        p=rp.HestonFourierPlan.compile(model('classical'),1.,time_steps=32,integration_intervals=64,cutoff=64.)
        self.assertEqual((p.time_steps,p.integration_intervals,p.cutoff,p.maturity),(32,64,64.,1.))
        with self.assertRaises(AttributeError):p.cutoff=99.
        r=p.refinement(100.,105.,discount=0.97)
        self.assertEqual(r.base_price,p.price(100.,105.,discount=0.97))
        self.assertEqual(r.time_change,r.time_refined_price-r.base_price)
        self.assertEqual(r.quadrature_change,r.quadrature_refined_price-r.time_refined_price)
        self.assertEqual(r.cutoff_change,r.extended_cutoff_price-r.quadrature_refined_price)
        with self.assertRaises(AttributeError):r.base_price=0.
        same=rp.HestonFourierPlan.compile(model('classical'),1.,time_steps=32,integration_intervals=64,cutoff=64.)
        self.assertEqual(same.plan_fingerprint,p.plan_fingerprint)

    def test_limits_input_rejection_and_no_silent_fallback(self):
        flat=rp.RoughVolatilityModel.rough_heston(**(P|{'hurst':0.1,'long_run_variance':0.04,'vol_of_vol':0.0}))
        p=rp.HestonFourierPlan.compile(flat,1.,time_steps=16,integration_intervals=64,cutoff=64.)
        self.assertAlmostEqual(p.price(100.,100.,discount=.97),7.72660043174363,delta=3e-13)
        for kwargs in [dict(time_steps=0),dict(integration_intervals=3),dict(cutoff=math.nan)]:
            with self.assertRaises(rp.ValidationError):rp.HestonFourierPlan.compile(flat,1.,**kwargs)
        unsupported=rp.RoughVolatilityModel.rfsv(hurst=.1,mean_reversion=1.,vol_of_log_vol=.2,mean_log_vol=-1.6)
        with self.assertRaises(rp.ValidationError):rp.HestonFourierPlan.compile(unsupported,1.)
        with self.assertRaises(rp.PricingError):p.transform(1.01,2.)
        with self.assertRaises(rp.PricingError):p.price(-100.,100.)

    def test_zero_initial_variance_with_nonzero_immigration(self):
        # Deterministic V(t)=theta*(1-exp(-k*t)), not constant zero variance.
        theta=.04;k=.7
        m=rp.RoughVolatilityModel.rough_heston(hurst=.5,initial_variance=0.,mean_reversion=k,long_run_variance=theta,vol_of_vol=0.,correlation=0.)
        p=rp.HestonFourierPlan.compile(m,1.,time_steps=1024,integration_intervals=512,cutoff=160.)
        iv=theta*(1.-(-math.expm1(-k))/k)
        reference=100.*math.erf(math.sqrt(iv)/(2.*math.sqrt(2.)))
        self.assertAlmostEqual(p.price(100.,100.),reference,delta=1e-6)

    def test_wheel_guard_rejects_new_api_drift(self):
        spec=importlib.util.spec_from_file_location('fourier_wheel_check',ROOT/'scripts/smoke_test_wheel.py')
        check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)
        original=ast.parse((ROOT/'rust_pricing.pyi').read_text())
        check.verify_stub_static_shape(original)
        for name in ['FourierRefinement','HestonFourierPlan']:
            tree=copy.deepcopy(original);tree.body=[n for n in tree.body if not(isinstance(n,ast.ClassDef) and n.name==name)]
            with self.assertRaises(RuntimeError):check.verify_stub_static_shape(tree)
        for name,member in [('HestonFourierPlan','compile'),('FourierRefinement','base_price')]:
            tree=copy.deepcopy(original)
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
            method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==member)
            method.decorator_list=[]
            with self.assertRaises(RuntimeError):check.verify_stub_static_shape(tree)
        tree=copy.deepcopy(original)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HestonFourierPlan')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compile')
        method.args.kw_defaults[-1]=ast.Constant(99.)
        with self.assertRaises(RuntimeError):check.verify_stub_static_shape(tree)

if __name__=='__main__':unittest.main()
