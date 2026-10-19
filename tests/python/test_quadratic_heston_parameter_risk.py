"""Finite-grid QRH AAD; full rebuilds include particle recalibration."""
import math
import unittest
import rust_pricing as rp
from test_mixed_bergomi_parameter_risk import pure_build, lsv_build

NAMES=['initial_state','mean_reversion','vol_of_vol','quadratic','shift','variance_floor','hurst']
PARAMS=[.1,.7,.4,.5,.2,.03,.1]

def model(params=PARAMS):
    return rp.RoughVolatilityModel.quadratic_rough_heston(**dict(zip(NAMES,params)))

class QuadraticParameterRisk(unittest.TestCase):
    def test_asian_all_parameters_and_hurst_rebuild_and_recalibrate(self):
        count=0;gap=0.
        for h in [.1,.3,.5]:
            params=PARAMS.copy();params[6]=h
            for qmc in [False,True]:
                for build in [pure_build,lsv_build]:
                    plan=build(model(params),qmc=qmc)
                    r=plan.evaluate_quadratic_heston_parameter_risk(include_hurst=True)
                    self.assertEqual(r.parameter_names,NAMES)
                    self.assertEqual(r.price.value,plan.evaluate().value)
                    self.assertEqual(r.price.standard_error,plan.evaluate().standard_error)
                    if build is lsv_build:
                        self.assertEqual(r.standard_errors is not None,qmc)
                        for a,b,c in zip(r.parameter_adjoints,r.direct_adjoints,r.calibration_adjoints):
                            self.assertAlmostEqual(a,b+c,delta=1e-11)
                    for j,adj in enumerate(r.parameter_adjoints):
                        def price(d):
                            p=params.copy();p[j]+=d
                            return build(model(p),qmc=qmc).evaluate().value
                        for e in [1e-6,5e-7]:
                            fd=((3*price(0)-4*price(-e)+price(-2*e))/(2*e) if j==6 and h==.5
                                else (price(e)-price(-e))/(2*e))
                            self.assertAlmostEqual(adj,fd,delta=3e-5*(1+abs(fd)))
                            gap=max(gap,abs(adj-fd));count+=1
        print(f'quadratic_python comparisons={count} max_gap={gap:.15e}')

    def test_frozen_properties_prefix_and_existing_risk_preservation(self):
        for build,cls in [(pure_build,rp.QuadraticHestonMcParameterRisk),(lsv_build,rp.QuadraticHestonLsvParameterRisk)]:
            p=build(model());before=p.evaluate();base=p.evaluate_quadratic_heston_parameter_risk()
            r=p.evaluate_quadratic_heston_parameter_risk(include_hurst=True)
            self.assertIsInstance(r,cls)
            self.assertEqual(r.parameter_names[:-1],base.parameter_names)
            self.assertEqual(r.parameter_adjoints[:-1],base.parameter_adjoints)
            self.assertEqual(r.standard_errors[:-1],base.standard_errors)
            self.assertNotEqual(r.risk_fingerprint,base.risk_fingerprint)
            other=build(model(),workers=3).evaluate_quadratic_heston_parameter_risk(include_hurst=True)
            self.assertEqual(r.parameter_adjoints,other.parameter_adjoints)
            self.assertEqual(r.standard_errors,other.standard_errors)
            self.assertEqual(before.value,p.evaluate().value)
            keys=['parameter_adjoints','parameter_names','standard_errors']
            if build is lsv_build:
                keys+=['direct_adjoints','calibration_adjoints']
                a=p.evaluate_local_variance_risk()
                p.evaluate_quadratic_heston_parameter_risk()
                self.assertEqual(a.node_adjoints,p.evaluate_local_variance_risk().node_adjoints)
            for key in keys:
                copy=getattr(r,key);copy[0]=None;self.assertIsNotNone(getattr(r,key)[0])
            for key in [*keys,'price','method','coordinate','risk_fingerprint']:
                with self.assertRaises(AttributeError):setattr(r,key,None)
            self.assertTrue(all(math.isfinite(x) and x>=0 for x in r.standard_errors))
            with self.assertRaises(TypeError):p.evaluate_quadratic_heston_parameter_risk(True)

    def test_domains_zero_coefficient_and_deterministic_target(self):
        bad=rp.RoughVolatilityModel.rough_heston(hurst=.1,initial_variance=.04,
            mean_reversion=.7,long_run_variance=.055,vol_of_vol=.2,correlation=-.6)
        for build in [pure_build,lsv_build]:
            with self.assertRaises(rp.PricingError):build(bad).evaluate_quadratic_heston_parameter_risk()
        with self.assertRaises(rp.PricingError):lsv_build(model(),trace=False).evaluate_quadratic_heston_parameter_risk()
        zero=PARAMS.copy();zero[3]=zero[5]=0.
        p=pure_build(model(zero));self.assertTrue(math.isfinite(p.evaluate().value))
        with self.assertRaises(rp.PricingError):p.evaluate_quadratic_heston_parameter_risk()
        constant=PARAMS.copy();constant[3]=0.
        r=pure_build(model(constant)).evaluate_quadratic_heston_parameter_risk(include_hurst=True)
        self.assertGreater(abs(r.parameter_adjoints[3]),.01)
        self.assertEqual(r.parameter_adjoints[-1],0.)
        deterministic=PARAMS.copy();deterministic[2]=0.
        r=lsv_build(model(deterministic)).evaluate_quadratic_heston_parameter_risk(include_hurst=True)
        self.assertGreater(abs(r.direct_adjoints[5]),1.)
        for j in [0,1,3,4,5,6]:self.assertAlmostEqual(r.parameter_adjoints[j],0.,delta=1e-9)

    def test_digital_requires_smoothing_and_smoothed_direction_is_recalibrated(self):
        digital=rp.Product.digital(1,2,'2027-09-04',100.,1.,'call','cash')
        direction=[.2,-.1,.3,.2,-.1,.01,.1]
        for build in [pure_build,lsv_build]:
            with self.assertRaises(rp.PricingError):build(model(),product=digital).evaluate_quadratic_heston_parameter_risk()
            r=build(model(),product=digital,smooth=2.).evaluate_quadratic_heston_parameter_risk(include_hurst=True)
            got=sum(a*d for a,d in zip(r.parameter_adjoints,direction))
            for e in [1e-6,5e-7]:
                def price(sign):
                    params=[p+sign*e*d for p,d in zip(PARAMS,direction)]
                    return build(model(params),product=digital,smooth=2.).evaluate().value
                fd=(price(1)-price(-1))/(2*e)
                self.assertAlmostEqual(got,fd,delta=3e-5*(1+abs(fd)))

if __name__=='__main__':unittest.main()
