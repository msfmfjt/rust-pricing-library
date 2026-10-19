"""Pure-SV Hurst reverse: model bumps, payout mapping and immutable contracts."""
import math
import unittest
import rust_pricing as rp
from test_heston_mc_parameter_risk import build, model as old_model, PARAMS, NAMES

def model(h, params=PARAMS):
    return rp.RoughVolatilityModel.rough_heston(hurst=h, **dict(zip(NAMES,params)))

class HurstRisk(unittest.TestCase):
    def test_asian_bumps_interior_and_left_boundary(self):
        for h in [.1,.3,.5]:
            for qmc in [False,True]:
                plan=build(model(h),qmc); risk=plan.evaluate_hurst_risk()
                self.assertEqual(risk.price.value,plan.evaluate().value)
                self.assertEqual(risk.price.standard_error,plan.evaluate().standard_error)
                for e in [1e-6,5e-7]:
                    value=lambda x:build(model(x),qmc).evaluate().value
                    fd=((3*risk.price.value-4*value(h-e)+value(h-2*e))/(2*e) if h==.5
                        else (value(h+e)-value(h-e))/(2*e))
                    self.assertAlmostEqual(risk.hurst_sensitivity,fd,delta=3e-5*(1+abs(fd)))
                self.assertTrue(math.isfinite(risk.standard_error) and risk.standard_error>0)
    def test_workers_immutable_contract_and_old_risk(self):
        p=build(model(.2));old=p.evaluate_heston_parameter_risk();r=p.evaluate_hurst_risk()
        b=build(model(.2),workers=3).evaluate_hurst_risk()
        self.assertEqual(r.hurst_sensitivity,b.hurst_sensitivity)
        self.assertEqual(r.standard_error,b.standard_error)
        self.assertNotEqual(r.risk_fingerprint,b.risk_fingerprint)
        self.assertEqual(old.parameter_adjoints,p.evaluate_heston_parameter_risk().parameter_adjoints)
        self.assertEqual(r.coordinate,'rough_heston_hurst_fixed_scalar_parameters')
        for name in ['price','hurst_sensitivity','standard_error','coordinate','method','risk_fingerprint']:
            with self.assertRaises(AttributeError):setattr(r,name,None)
    def test_domains_and_constant_variance(self):
        with self.assertRaises(rp.PricingError):build(old_model(True)).evaluate_hurst_risk()
        for v,rho in [(0.,-.6),(.04,1.),(.04,-1.)]:
            pars=PARAMS[:];pars[0]=v;pars[4]=rho
            with self.assertRaises(rp.PricingError):build(model(.2,pars)).evaluate_hurst_risk()
        r=build(model(.2,[.04,0.,.04,0.,-.6])).evaluate_hurst_risk()
        self.assertEqual(r.hurst_sensitivity,0.)
        self.assertEqual(r.standard_error,0.)
    def test_digital_smoothing_contract(self):
        product=rp.Product.digital(1,2,'2027-09-04',100.,1.,'call','cash')
        with self.assertRaises(rp.PricingError):build(model(.2),product=product).evaluate_hurst_risk()
        p=build(model(.2),product=product,smooth=2.);r=p.evaluate_hurst_risk()
        for e in [1e-6,5e-7]:
            fd=(build(model(.2+e),product=product,smooth=2.).evaluate().value
                -build(model(.2-e),product=product,smooth=2.).evaluate().value)/(2*e)
            self.assertAlmostEqual(r.hurst_sensitivity,fd,delta=3e-5*(1+abs(fd)))
if __name__=='__main__':unittest.main()
