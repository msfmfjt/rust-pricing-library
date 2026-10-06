"""Original xi inputs and sum-preserving mixture-weight transfers, no price bumps."""
import math
import unittest
import rust_pricing as rp
from test_mixed_bergomi_parameter_risk import pure_build, lsv_build


def params(exponential=False):
    return [.2,.35,.04,.2] if exponential else [.2,.35,.04,.05,.045,.055]


def model(p=None, h=.2, exponential=False):
    p=params(exponential) if p is None else p
    xi=(rp.ForwardVarianceCurve.exponential(p[2],p[3]) if exponential else
        rp.ForwardVarianceCurve.piecewise_linear([0.,.4,.9,1.4],p[2:]))
    return rp.RoughVolatilityModel.mixed_rough_bergomi(
        hurst=h, correlation=-.6, weights=[p[0],p[1],1-p[0]-p[1]],
        vol_of_vols=[.3,.8,1.1],forward_variance=xi)


class MixedShapeTests(unittest.TestCase):
    def test_all_coordinates_asian_two_widths_full_recalibration(self):
        for exponential in [False,True]:
            p=params(exponential)
            for h in [.1,.5]:
                for qmc in [False,True]:
                    for build in [pure_build,lsv_build]:
                        b=build(model(p,h,exponential),qmc)
                        a=b.evaluate_mixed_bergomi_shape_risk()
                        self.assertEqual(a.parameter_names[:2],['weight_transfer[0,2]','weight_transfer[1,2]'])
                        self.assertEqual(len(a.parameter_names),len(p))
                        self.assertEqual(a.price.value,b.evaluate().value)
                        for j,g in enumerate(a.parameter_adjoints):
                            for e in [1e-6,5e-7]:
                                up=p[:];dn=p[:];up[j]+=e;dn[j]-=e
                                fd=(build(model(up,h,exponential),qmc).evaluate().value-
                                    build(model(dn,h,exponential),qmc).evaluate().value)/(2*e)
                                self.assertAlmostEqual(g,fd,delta=3e-5*(1+abs(fd)))
                        if build is lsv_build:
                            self.assertEqual(a.standard_errors is not None,qmc)
                            self.assertGreater(abs(a.direct_adjoints[2]),1.)
                            for j in range(2,len(p)):
                                self.assertAlmostEqual(a.parameter_adjoints[j],0.,delta=1e-8)
                                if qmc:self.assertLess(a.standard_errors[j],1e-8)
                            for g,d,c in zip(a.parameter_adjoints,a.direct_adjoints,a.calibration_adjoints):
                                self.assertAlmostEqual(g,d+c,delta=1e-11)

    def test_copy_frozen_workers_old_api_and_coordinate_identity(self):
        for build in [pure_build,lsv_build]:
            p=build(model());old=p.evaluate_mixed_bergomi_parameter_risk_with_hurst()
            a=p.evaluate_mixed_bergomi_shape_risk()
            b=build(model(),workers=3).evaluate_mixed_bergomi_shape_risk()
            self.assertEqual(a.parameter_adjoints,b.parameter_adjoints)
            self.assertEqual(a.standard_errors,b.standard_errors)
            self.assertNotEqual(a.risk_fingerprint,old.risk_fingerprint)
            self.assertNotEqual(a.coordinate,old.coordinate)
            for name in ['parameter_names','parameter_adjoints','standard_errors']:
                copy=getattr(a,name);copy[0]=None;self.assertIsNotNone(getattr(a,name)[0])
            for name in ['parameter_names','parameter_adjoints','standard_errors','coordinate','method','price','risk_fingerprint']:
                with self.assertRaises(AttributeError):setattr(a,name,None)
            again=p.evaluate_mixed_bergomi_parameter_risk_with_hurst()
            self.assertEqual(again.parameter_adjoints,old.parameter_adjoints)
            self.assertEqual(again.standard_errors,old.standard_errors)
            self.assertTrue(all(math.isfinite(x) and x>=0 for x in a.standard_errors))

    def test_smoothing_explicit_domains_and_preserved_boundary_pricing(self):
        digital=rp.Product.digital(1,2,'2027-09-04',100.,1.,'call','cash')
        for build in [pure_build,lsv_build]:
            with self.assertRaises(rp.PricingError):build(model(),product=digital).evaluate_mixed_bergomi_shape_risk()
            a=build(model(),product=digital,smooth=2.).evaluate_mixed_bergomi_shape_risk()
            for j in [0,2]:
                for e in [1e-6,5e-7]:
                    up=params();dn=params();up[j]+=e;dn[j]-=e
                    fd=(build(model(up),product=digital,smooth=2.).evaluate().value-
                        build(model(dn),product=digital,smooth=2.).evaluate().value)/(2*e)
                    self.assertAlmostEqual(a.parameter_adjoints[j],fd,delta=3e-5*(1+abs(fd)))
            zero=params();zero[0]=0.
            p=build(model(zero));self.assertTrue(math.isfinite(p.evaluate().value))
            with self.assertRaises(rp.PricingError):p.evaluate_mixed_bergomi_shape_risk()
        with self.assertRaises(rp.PricingError):lsv_build(model(),trace=False).evaluate_mixed_bergomi_shape_risk()

    def test_curve_nodes_after_last_observation_and_flat_extrapolation(self):
        # Last original knot can affect earlier interpolation; do not discard it.
        xi=rp.ForwardVarianceCurve.piecewise_linear([0.,.4,1.4],[.04,.05,.06])
        m=rp.RoughVolatilityModel.mixed_rough_bergomi(hurst=.1,correlation=-.6,weights=[.2,.35,.45],vol_of_vols=[.3,.8,1.1],forward_variance=xi)
        a=pure_build(m).evaluate_mixed_bergomi_shape_risk()
        self.assertGreater(abs(a.parameter_adjoints[-1]),1e-4)
        xi=rp.ForwardVarianceCurve.constant(.04)
        m=rp.RoughVolatilityModel.mixed_rough_bergomi(hurst=.1,correlation=1.,weights=[1.],vol_of_vols=[.7],forward_variance=xi)
        a=pure_build(m).evaluate_mixed_bergomi_shape_risk()
        self.assertEqual(a.parameter_names,['forward_variance[0]'])
        self.assertTrue(math.isfinite(a.parameter_adjoints[0]))

if __name__=='__main__':unittest.main()
