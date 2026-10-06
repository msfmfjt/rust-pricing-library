"""Fixed-kernel Pure-SV Heston parameter risk, not leverage/model recalibration."""
import math
import unittest
import rust_pricing as rp
from test_rough_volatility_families import models as other_models

NAMES=["initial_variance","mean_reversion","long_run_variance","vol_of_vol","correlation"]
PARAMS=[.04,.7,.055,.15,-.6]

def model(lift=False, params=PARAMS):
    kw=dict(zip(NAMES,params))
    return (rp.RoughVolatilityModel.lifted_heston(**kw,weights=[.2,.8,1.3],rates=[0.,.7,12.]) if lift
            else rp.RoughVolatilityModel.rough_heston(hurst=.2,**kw))

def build(m,qmc=True,product=None,smooth=None,workers=1):
    product=product or rp.Product.arithmetic_asian(1,2,100.,1.,"call",[
        rp.AsianObservation.unknown(d,1/3) for d in ["2026-12-04","2027-03-04","2027-09-04"]],"2027-12-04")
    market=rp.Market.equity(2,1,100.,rp.DiscountCurve(10,[0.,1.,2.],[1.,.97,.97**2]),
        rp.DiscountCurve(11,[0.,1.,2.],[1.,.99,.99**2]),discrete_dividends=[
            rp.DividendEvent.fixed_cash_and_proportional(1,.25,3.,.02),
            rp.DividendEvent.fixed_cash(2,1.5,4.)])
    engine=(rp.Engine.randomized_quasi_monte_carlo(64,819,scramble_count=4,antithetic=True,brownian_bridge=True)
        if qmc else rp.Engine.pseudo_monte_carlo(819,128,antithetic=True,brownian_bridge=False))
    request=rp.PricingRequest("2026-09-04",product,market,rp.Model.black_scholes(.2),engine,
        rp.RiskRequest(payoff_smoothing_half_width=smooth))
    return rp.RoughVolatilityPlan.compile(request,m,maximum_step=.25,worker_threads=workers,reduction_block_size=32)

class HestonMcParameterTest(unittest.TestCase):
    def test_asian_full_parameter_bumps_and_preserved_price(self):
        for lift in [False,True]:
            for qmc in [False,True]:
                p=build(model(lift),qmc); r=p.evaluate_heston_parameter_risk()
                self.assertEqual(r.parameter_names,NAMES)
                self.assertEqual(r.price.value,p.evaluate().value)
                self.assertEqual(r.price.standard_error,p.evaluate().standard_error)
                for k in range(5):
                    for eps in [1e-6,5e-7]:
                        up=PARAMS[:];dn=PARAMS[:];up[k]+=eps;dn[k]-=eps
                        fd=(build(model(lift,up),qmc).evaluate().value-build(model(lift,dn),qmc).evaluate().value)/(2*eps)
                        self.assertAlmostEqual(r.parameter_adjoints[k],fd,delta=3e-5*(1+abs(fd)))
                self.assertTrue(all(math.isfinite(x) and x>0 for x in r.standard_errors))

    def test_copies_frozen_results_and_worker_reproducibility(self):
        for lift in [False,True]:
            a=build(model(lift),workers=1).evaluate_heston_parameter_risk()
            b=build(model(lift),workers=3).evaluate_heston_parameter_risk()
            self.assertEqual(a.parameter_adjoints,b.parameter_adjoints)
            self.assertEqual(a.standard_errors,b.standard_errors)
            self.assertNotEqual(a.risk_fingerprint,b.risk_fingerprint)
            for name in ["parameter_names","parameter_adjoints","standard_errors"]:
                copy=getattr(a,name);copy[0]=None;self.assertIsNotNone(getattr(a,name)[0])
            for name in ["price","parameter_adjoints","method","coordinate","risk_fingerprint"]:
                with self.assertRaises(AttributeError):setattr(a,name,None)
            self.assertEqual(a.coordinate,"fixed_kernel_heston_scalar_parameters")
            self.assertEqual(a.method,"heston-mc-fixed-kernel-parameter-vjp-v1")

    def test_unsupported_models_and_parameter_boundaries(self):
        for m in other_models()[2:]:
            with self.assertRaises(rp.PricingError):build(m).evaluate_heston_parameter_risk()
        for lift in [False,True]:
            for k,v in [(0,0.),(4,1.),(4,-1.)]:
                p=PARAMS[:];p[k]=v
                plan=build(model(lift,p))
                self.assertTrue(math.isfinite(plan.evaluate().value))
                with self.assertRaises(rp.PricingError):plan.evaluate_heston_parameter_risk()

    def test_hard_digital_rejected_smoothed_direction_checked(self):
        digital=rp.Product.digital(1,2,"2027-09-04",100.,1.,"call","cash")
        for lift in [False,True]:
            with self.assertRaises(rp.PricingError):build(model(lift),product=digital).evaluate_heston_parameter_risk()
            p=build(model(lift),product=digital,smooth=2.)
            risk=p.evaluate_heston_parameter_risk()
            d=[.2,-.1,.3,.2,.4]
            got=sum(x*y for x,y in zip(risk.parameter_adjoints,d))
            for e in [1e-6,5e-7]:
                a=build(model(lift,[x+e*y for x,y in zip(PARAMS,d)]),product=digital,smooth=2.).evaluate().value
                b=build(model(lift,[x-e*y for x,y in zip(PARAMS,d)]),product=digital,smooth=2.).evaluate().value
                self.assertAlmostEqual(got,(a-b)/(2*e),delta=3e-5*(1+abs(got)))

if __name__=="__main__":unittest.main()
