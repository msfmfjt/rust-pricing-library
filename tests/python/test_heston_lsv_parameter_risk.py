"""Fixed-target LSV parameter VJPs, full recalibration and immutable contracts."""
import math
import unittest
import rust_pricing as rp

NAMES = ['initial_variance','mean_reversion','long_run_variance','vol_of_vol','correlation']
PARAMS = [.04,.7,.055,.15,-.6]

def model(lift=False, h=.2, params=PARAMS):
    kw = dict(zip(NAMES, params))
    return (rp.RoughVolatilityModel.lifted_heston(**kw, weights=[.2,.8,1.3], rates=[0.,.7,12.]) if lift
            else rp.RoughVolatilityModel.rough_heston(hurst=h, **kw))

def build(m, qmc=True, product=None, smooth=None, workers=1, trace=True):
    product = product or rp.Product.arithmetic_asian(1,2,100.,1.,'call',[
        rp.AsianObservation.unknown(d,1/3) for d in ['2026-12-04','2027-03-04','2027-09-04']], '2027-12-04')
    market = rp.Market.equity(2,1,100.,rp.DiscountCurve(10,[0.,1.,2.],[1.,.97,.97**2]),
        rp.DiscountCurve(11,[0.,1.,2.],[1.,.99,.99**2]),discrete_dividends=[
            rp.DividendEvent.fixed_cash_and_proportional(1,.25,3.,.02),rp.DividendEvent.fixed_cash(2,1.5,4.)])
    times=[0.,.25,.5,.75,1.];xs=[-.6,-.2,.13,.45,.8]
    target=rp.Model.local_volatility_from_grid(times,xs,[.04+.001*i+.002*j for i in range(5) for j in range(5)],1e-8,4.)
    engine=(rp.Engine.randomized_quasi_monte_carlo(64,819,scramble_count=4,antithetic=True,brownian_bridge=True) if qmc else
            rp.Engine.pseudo_monte_carlo(819,128,antithetic=True,brownian_bridge=False))
    request=rp.PricingRequest('2026-09-04',product,market,target,engine,rp.RiskRequest(payoff_smoothing_half_width=smooth))
    return rp.RoughFamilyLsvPlan.compile(request,m,particle_count=128,calibration_seed=429,log_bandwidth=.5,
        minimum_effective_samples=3.,retain_reverse_trace=trace,worker_threads=workers,reduction_block_size=64)

class HestonLsvRiskTest(unittest.TestCase):
    def test_asian_model_bumps_recalibrate_leverage(self):
        for lift,h in [(False,.1),(False,.3),(False,.5),(True,.2)]:
            for qmc in [False,True]:
                plan=build(model(lift,h),qmc);r=plan.evaluate_heston_parameter_risk(include_hurst=not lift)
                self.assertEqual(r.price.value,plan.evaluate().value)
                self.assertEqual(r.parameter_names,NAMES+([] if lift else ['hurst']))
                self.assertEqual(r.standard_errors is not None,qmc)
                if qmc:self.assertTrue(all(math.isfinite(x) and x>=0 for x in r.standard_errors))
                for j,adj in enumerate(r.parameter_adjoints):
                    self.assertAlmostEqual(adj,r.direct_adjoints[j]+r.calibration_adjoints[j],delta=1e-11)
                    for e in [1e-6,5e-7]:
                        def price(shift):
                            p=PARAMS[:];hh=h
                            if j==5:hh+=shift
                            else:p[j]+=shift
                            return build(model(lift,hh,p),qmc).evaluate().value
                        fd=((3*r.price.value-4*price(-e)+price(-2*e))/(2*e) if j==5 and h==.5 else
                            (price(e)-price(-e))/(2*e))
                        self.assertAlmostEqual(adj,fd,delta=3e-5*(1+abs(fd)))

    def test_frozen_properties_workers_and_old_risks(self):
        p=build(model());old=p.evaluate_local_variance_risk();d=p.evaluate_sticky_moneyness_delta()
        r=p.evaluate_heston_parameter_risk(include_hurst=True)
        self.assertIsInstance(r,rp.HestonLsvParameterRisk)
        self.assertEqual(old.node_adjoints,p.evaluate_local_variance_risk().node_adjoints)
        self.assertEqual(d.delta,p.evaluate_sticky_moneyness_delta().delta)
        b=build(model(),workers=3).evaluate_heston_parameter_risk(include_hurst=True)
        self.assertEqual(r.parameter_adjoints,b.parameter_adjoints);self.assertEqual(r.standard_errors,b.standard_errors)
        self.assertNotEqual(r.risk_fingerprint,b.risk_fingerprint)
        for name in ['parameter_names','parameter_adjoints','direct_adjoints','calibration_adjoints','standard_errors']:
            copy=getattr(r,name);copy[0]=None;self.assertIsNotNone(getattr(r,name)[0])
        for name in ['price','parameter_names','parameter_adjoints','direct_adjoints','calibration_adjoints','standard_errors','coordinate','method','risk_fingerprint']:
            with self.assertRaises(AttributeError):setattr(r,name,None)
        self.assertEqual(r.coordinate,'heston_parameters_fixed_relative_local_variance_target')
        self.assertNotEqual(r.risk_fingerprint,p.evaluate_heston_parameter_risk().risk_fingerprint)

    def test_domains_and_deterministic_cancellation(self):
        with self.assertRaises(rp.PricingError):build(model(),trace=False).evaluate_heston_parameter_risk()
        with self.assertRaises(rp.PricingError):build(model(True)).evaluate_heston_parameter_risk(include_hurst=True)
        other=rp.RoughVolatilityModel.mixed_rough_bergomi(hurst=.2,correlation=-.6,weights=[1.],vol_of_vols=[.3],forward_variance=rp.ForwardVarianceCurve.constant(.04))
        with self.assertRaises(rp.PricingError):build(other).evaluate_heston_parameter_risk()
        for lift in [False,True]:
            r=build(model(lift,params=[.04,.7,.055,0.,-.6])).evaluate_heston_parameter_risk(include_hurst=not lift)
            self.assertGreater(abs(r.direct_adjoints[0]),1.)
            for j,a in enumerate(r.parameter_adjoints):
                if j!=3:self.assertAlmostEqual(a,0.,delta=2e-10)

    def test_smoothed_digital_and_unsmoothed_rejection(self):
        product=rp.Product.digital(1,2,'2027-09-04',100.,1.,'call','cash')
        for lift in [False,True]:
            with self.assertRaises(rp.PricingError):build(model(lift),product=product).evaluate_heston_parameter_risk()
            r=build(model(lift),product=product,smooth=2.).evaluate_heston_parameter_risk(include_hurst=not lift)
            direction=[.2,-.1,.3,.2,.4]+([] if lift else [.1])
            got=sum(a*b for a,b in zip(r.parameter_adjoints,direction))
            for e in [1e-6,5e-7]:
                def price(sign):
                    params=[p+sign*e*d for p,d in zip(PARAMS,direction)]
                    return build(model(lift,.2+sign*e*(0. if lift else .1),params),product=product,smooth=2.).evaluate().value
                fd=(price(1)-price(-1))/(2*e)
                self.assertAlmostEqual(got,fd,delta=3e-5*(1+abs(fd)))

if __name__=='__main__':unittest.main()
