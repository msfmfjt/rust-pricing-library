"""Fixed-weight eta/rho risk with full model rebuilds and LSV recalibration."""
import math
import unittest
import rust_pricing as rp
from test_heston_lsv_parameter_risk import build as lsv_build

PARAMS = [.3, .8, -.6]
NAMES = ['vol_of_vol[0]', 'vol_of_vol[1]', 'correlation']


def model(h=.2, params=PARAMS, weights=(.35, .65)):
    return rp.RoughVolatilityModel.mixed_rough_bergomi(
        hurst=h, correlation=params[-1], weights=list(weights),
        vol_of_vols=params[:-1], forward_variance=rp.ForwardVarianceCurve.piecewise_linear(
            [0., .4, 1.], [.04, .05, .045]))


def pure_build(m, qmc=True, workers=1, product=None, smooth=None):
    product = product or rp.Product.arithmetic_asian(1, 2, 100., 1., 'call', [
        rp.AsianObservation.unknown(d, 1/3) for d in
        ['2026-12-04','2027-03-04','2027-09-04']], '2027-12-04')
    market = rp.Market.equity(2, 1, 100., rp.DiscountCurve(10,[0.,1.,2.],[1.,.97,.97**2]),
        rp.DiscountCurve(11,[0.,1.,2.],[1.,.99,.99**2]), discrete_dividends=[
        rp.DividendEvent.fixed_cash_and_proportional(1,.25,3.,.02),rp.DividendEvent.fixed_cash(2,1.5,4.)])
    engine = (rp.Engine.randomized_quasi_monte_carlo(64,819,scramble_count=4,antithetic=True,brownian_bridge=True)
        if qmc else rp.Engine.pseudo_monte_carlo(819,128,antithetic=True,brownian_bridge=False))
    request = rp.PricingRequest('2026-09-04', product, market, rp.Model.black_scholes(.2), engine,
        rp.RiskRequest(payoff_smoothing_half_width=smooth))
    return rp.RoughVolatilityPlan.compile(request,m,maximum_step=.25,worker_threads=workers,reduction_block_size=64)


class MixedRiskTests(unittest.TestCase):
    def test_pure_and_lsv_all_directions_rebuild_model_and_recalibrate(self):
        for h in [.1, .3, .5]:
            for qmc in [False, True]:
                for build in [pure_build, lsv_build]:
                    plan = build(model(h), qmc); r = plan.evaluate_mixed_bergomi_parameter_risk()
                    self.assertEqual(r.parameter_names, NAMES)
                    self.assertEqual(r.price.value, plan.evaluate().value)
                    for j, adj in enumerate(r.parameter_adjoints):
                        for e in [1e-6,5e-7]:
                            def price(sign):
                                p = PARAMS[:]; p[j] += sign*e
                                return build(model(h,p),qmc).evaluate().value
                            fd = (price(1)-price(-1))/(2*e)
                            self.assertAlmostEqual(adj,fd,delta=3e-5*(1+abs(fd)))
                    if build is lsv_build:
                        self.assertEqual(r.standard_errors is not None,qmc)
                        for a,d,c in zip(r.parameter_adjoints,r.direct_adjoints,r.calibration_adjoints):
                            self.assertAlmostEqual(a,d+c,delta=1e-11)

    def test_immutable_dynamic_vectors_and_worker_identity(self):
        for build in [pure_build, lsv_build]:
            plan = build(model());before = plan.evaluate()
            r = plan.evaluate_mixed_bergomi_parameter_risk()
            other = build(model(),workers=3).evaluate_mixed_bergomi_parameter_risk()
            self.assertEqual(r.parameter_adjoints,other.parameter_adjoints)
            self.assertEqual(r.standard_errors,other.standard_errors)
            self.assertEqual(before.value,plan.evaluate().value)
            for key in ['parameter_adjoints','parameter_names','standard_errors']:
                copy = getattr(r,key);copy[0] = None;self.assertIsNotNone(getattr(r,key)[0])
            for key in ['parameter_adjoints','parameter_names','standard_errors','method','coordinate','risk_fingerprint','price']:
                with self.assertRaises(AttributeError):setattr(r,key,None)
            self.assertTrue(all(math.isfinite(x) and x>=0 for x in r.standard_errors))
        with self.assertRaises(rp.PricingError):lsv_build(model(),trace=False).evaluate_mixed_bergomi_parameter_risk()
        single = model(params=[.3,-.6],weights=[1.])
        self.assertEqual(pure_build(single).evaluate_mixed_bergomi_parameter_risk().parameter_names,
            ['vol_of_vol[0]','correlation'])

    def test_smoothing_rejection_zero_weight_and_eta_boundary(self):
        digital = rp.Product.digital(1,2,'2027-09-04',100.,1.,'call','cash')
        for build in [pure_build, lsv_build]:
            with self.assertRaises(rp.PricingError):build(model(),product=digital).evaluate_mixed_bergomi_parameter_risk()
            p = build(model(),product=digital,smooth=2.)
            r = p.evaluate_mixed_bergomi_parameter_risk()
            for e in [1e-6,5e-7]:
                up=PARAMS[:];dn=PARAMS[:];up[1]+=e;dn[1]-=e
                fd=(build(model(params=up),product=digital,smooth=2.).evaluate().value-
                    build(model(params=dn),product=digital,smooth=2.).evaluate().value)/(2*e)
                self.assertAlmostEqual(r.parameter_adjoints[1],fd,delta=3e-5*(1+abs(fd)))
            r=build(model(weights=[1.,0.])).evaluate_mixed_bergomi_parameter_risk()
            self.assertEqual(r.parameter_adjoints[1],0.)
            # eta=0 has the smooth inward derivative, not an invented zero risk.
            base=[0.,.8,-.6];p=build(model(params=base));r=p.evaluate_mixed_bergomi_parameter_risk()
            for e in [1e-6,5e-7]:
                up=base[:];up[0]=e;up2=base[:];up2[0]=2*e
                fd=(-3*r.price.value+4*build(model(params=up)).evaluate().value-build(model(params=up2)).evaluate().value)/(2*e)
                self.assertAlmostEqual(r.parameter_adjoints[0],fd,delta=3e-5*(1+abs(fd)))

    def test_parameter_family_and_correlation_boundary_rejections(self):
        wrong=rp.RoughVolatilityModel.rough_heston(hurst=.2,initial_variance=.04,mean_reversion=.7,
            long_run_variance=.055,vol_of_vol=.15,correlation=-.6)
        for build in [pure_build,lsv_build]:
            with self.assertRaises(rp.PricingError):build(wrong).evaluate_mixed_bergomi_parameter_risk()
            for rho in [-1.,1.]:
                p=build(model(params=[.3,.8,rho]));self.assertTrue(math.isfinite(p.evaluate().value))
                with self.assertRaises(rp.PricingError):p.evaluate_mixed_bergomi_parameter_risk()

if __name__=='__main__':unittest.main()
