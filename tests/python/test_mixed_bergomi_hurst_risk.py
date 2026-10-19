"""Joint eta/rho/Hurst reverse with fixed weights/xi and recalibrated LSV."""
import math
import struct
import unittest
import rust_pricing as rp
from test_mixed_bergomi_parameter_risk import model, pure_build, lsv_build


def bits(xs):
    return [struct.pack('d', x) for x in xs]


def derivative(h, e, f):
    return ((3*f(h)-4*f(h-e)+f(h-2*e))/(2*e) if h == .5 else
            (f(h+e)-f(h-e))/(2*e))


class MixedHurstTests(unittest.TestCase):
    def test_asian_rebuild_full_recalibration_and_exact_old_prefix(self):
        for h in [.03, .1, .3, .5]:
            for qmc in [False, True]:
                for build in [pure_build, lsv_build]:
                    p=build(model(h), qmc)
                    old=p.evaluate_mixed_bergomi_parameter_risk()
                    r=p.evaluate_mixed_bergomi_parameter_risk_with_hurst()
                    self.assertEqual(r.parameter_names, old.parameter_names+['hurst'])
                    self.assertEqual(r.price.value, p.evaluate().value)
                    self.assertEqual(bits(r.parameter_adjoints[:-1]), bits(old.parameter_adjoints))
                    if r.standard_errors is not None:
                        self.assertEqual(bits(r.standard_errors[:-1]), bits(old.standard_errors))
                        self.assertTrue(all(math.isfinite(x) and x >= 0 for x in r.standard_errors))
                    self.assertNotEqual(r.coordinate, old.coordinate)
                    self.assertNotEqual(r.risk_fingerprint, old.risk_fingerprint)
                    for e in [1e-6, 5e-7]:
                        fd=derivative(h,e,lambda hh: build(model(hh),qmc).evaluate().value)
                        self.assertAlmostEqual(r.parameter_adjoints[-1],fd,delta=3e-5*(1+abs(fd)))
                    if build is lsv_build:
                        self.assertEqual(r.standard_errors is not None, qmc)
                        for name in ['direct_adjoints','calibration_adjoints']:
                            self.assertEqual(bits(getattr(r,name)[:-1]),bits(getattr(old,name)))
                        for a,d,c in zip(r.parameter_adjoints,r.direct_adjoints,r.calibration_adjoints):
                            self.assertAlmostEqual(a,d+c,delta=1e-11)
                    self.assertEqual(p.evaluate_mixed_bergomi_parameter_risk().risk_fingerprint, old.risk_fingerprint)

    def test_workers_copy_semantics_and_zero_hurst_risk(self):
        for build in [pure_build, lsv_build]:
            p=build(model())
            r=p.evaluate_mixed_bergomi_parameter_risk_with_hurst()
            other=build(model(),workers=3).evaluate_mixed_bergomi_parameter_risk_with_hurst()
            self.assertEqual(bits(r.parameter_adjoints),bits(other.parameter_adjoints))
            self.assertEqual(bits(r.standard_errors),bits(other.standard_errors))
            for name in ['parameter_names','parameter_adjoints','standard_errors']:
                copy=getattr(r,name);copy[-1]=None;self.assertIsNotNone(getattr(r,name)[-1])
            for name in ['parameter_names','parameter_adjoints','standard_errors','method','coordinate','price','risk_fingerprint']:
                with self.assertRaises(AttributeError):setattr(r,name,None)
            # eta=0 eliminates H from the variance law, but eta derivatives remain.
            zero=build(model(params=[0.,0.,-.6])).evaluate_mixed_bergomi_parameter_risk_with_hurst()
            self.assertEqual(zero.parameter_adjoints[-1],0.)
        m=rp.RoughVolatilityModel.mixed_rough_bergomi(hurst=.1, correlation=-.6,weights=[1.],vol_of_vols=[.7],
                forward_variance=rp.ForwardVarianceCurve.constant(0.))
        self.assertEqual(pure_build(m).evaluate_mixed_bergomi_parameter_risk_with_hurst().parameter_adjoints,[0.,0.,0.])

    def test_smoothing_domains_and_missing_trace(self):
        digital=rp.Product.digital(1,2,'2027-09-04',100.,1.,'call','cash')
        wrong=rp.RoughVolatilityModel.rough_heston(hurst=.2,initial_variance=.04,mean_reversion=.7,
            long_run_variance=.055,vol_of_vol=.15,correlation=-.6)
        for build in [pure_build,lsv_build]:
            with self.assertRaises(rp.PricingError):build(wrong).evaluate_mixed_bergomi_parameter_risk_with_hurst()
            for rho in [-1.,1.]:
                p=build(model(params=[.3,.8,rho]));self.assertTrue(math.isfinite(p.evaluate().value))
                with self.assertRaises(rp.PricingError):p.evaluate_mixed_bergomi_parameter_risk_with_hurst()
            with self.assertRaises(rp.PricingError):build(model(),product=digital).evaluate_mixed_bergomi_parameter_risk_with_hurst()
            r=build(model(),product=digital,smooth=2.).evaluate_mixed_bergomi_parameter_risk_with_hurst()
            for e in [1e-6,5e-7]:
                fd=derivative(.2,e,lambda h:build(model(h),product=digital,smooth=2.).evaluate().value)
                self.assertAlmostEqual(r.parameter_adjoints[-1],fd,delta=3e-5*(1+abs(fd)))
        with self.assertRaises(rp.PricingError):lsv_build(model(),trace=False).evaluate_mixed_bergomi_parameter_risk_with_hurst()

if __name__=='__main__':unittest.main()
