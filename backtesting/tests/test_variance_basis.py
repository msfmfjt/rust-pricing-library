from copy import deepcopy
import unittest
import numpy as np
from rust_pricing_backtest.domain import BacktestError,fingerprint
from rust_pricing_backtest.variance_basis import tensor_variance_basis
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator as Calibrator
from test_multi_step_full_smile import model


class VarianceBasisTests(unittest.TestCase):
    def test_tensor_interpolation_order_constant_endpoints_and_immutable_inputs(self):
        t=[0.,1.,2.,3.];x=[-1.,0.,1.];tk=[1.,2.];kk=[-1.,1.];s=[[1.,2.],[3.,4.]]
        before=deepcopy([t,x,tk,kk,s]);b,a=tensor_variance_basis(t,x,time_knots=tk,moneyness_knots=kk,parameter_scales=s)
        parameters=np.array([1.,2.,3.,4.]);rows=(b@parameters).reshape(3,3)
        np.testing.assert_array_equal(rows,[[1.,2.5,4.],[5.,7.5,10.],[9.,12.5,16.]])
        self.assertEqual(a['basis_hash'],fingerprint(b.tolist()));self.assertEqual([t,x,tk,kk,s],before)
        s[0][0]=99.;self.assertEqual(a['parameter_scales'][0][0],1.)
        with self.assertRaises(ValueError):b[0,0]=0.

    def test_presets_can_be_represented_exactly(self):
        for n in [4,6]:
            p=model(n);g=p['model']['local_variance_grid'];t=np.array(g['time_nodes']);x=g['log_forward_moneyness_nodes']
            for kind,tk in [('all_rows',(t[:-1]+t[1:])/2),('linear_time',p['model']['reporting_iv_basis']['maturity_nodes'])]:
                b,_=tensor_variance_basis(t,x,time_knots=tk,moneyness_knots=x)
                np.testing.assert_array_equal(b,Calibrator(p,extension=kind).basis)

    def test_metric_scales_change_the_surface_response_but_not_quote_response(self):
        p=model(4);g=p['model']['local_variance_grid'];t=np.array(g['time_nodes']);x=g['log_forward_moneyness_nodes']
        kw=dict(time_knots=(t[:-1]+t[1:])/2,moneyness_knots=x[::2])
        b,_=tensor_variance_basis(t,x,**kw);scales=np.linspace(.6,1.5,4)[:,None]*np.ones((4,17))
        d,_=tensor_variance_basis(t,x,**kw,parameter_scales=scales)
        a=Calibrator(p,extension='custom',variance_basis=b);c=Calibrator(p,extension='custom',variance_basis=d)
        ta,aa=a.all_maturity_iv_tangent();tc,ac=c.all_maturity_iv_tangent()
        self.assertLess(aa['iv_response_residual'],1e-9);self.assertLess(ac['iv_response_residual'],1e-9)
        self.assertGreater(np.max(abs(ta-tc)),.01)
        # Orthogonal reparameterization preserves B B^T and the declared metric.
        q,_=np.linalg.qr(np.random.default_rng(49).normal(size=(b.shape[1],b.shape[1])))
        rotated=Calibrator(p,extension='custom',variance_basis=b@q)
        tr,_=rotated.all_maturity_iv_tangent();np.testing.assert_allclose(ta,tr,rtol=0,atol=1e-10)
        shift=np.linspace(-.0002,.0003,a.iv.size).reshape(a.iv.shape)
        pa,_=a.calibrate_all_maturities(shift);pr,_=rotated.calibrate_all_maturities(shift)
        np.testing.assert_allclose(pa['model']['local_variance_grid']['values'],pr['model']['local_variance_grid']['values'],rtol=0,atol=1e-12)

    def test_invalid_or_unsupported_parameters_are_rejected(self):
        kw=dict(time_nodes=[0.,1.,2.],variance_nodes=[-1.,0.,1.],time_knots=[.5,1.5],moneyness_knots=[-1.,1.])
        for patch in [dict(time_nodes=[0.,0.,1.]),dict(time_nodes=[1.,2.,3.]),dict(time_knots=[0.,.1]),
                dict(time_knots=[-1.,1.]),dict(moneyness_knots=[-2.,1.]),dict(moneyness_knots=[1.,-1.]),
                dict(parameter_scales=[1.,2.,3.,4.]),dict(parameter_scales=[[0.,1.],[1.,1.]]),
                dict(parameter_scales=[[float('inf'),1.],[1.,1.]]),dict(time_knots=['bad']),dict(time_knots=[])]:
            with self.subTest(patch=patch),self.assertRaises(BacktestError):tensor_variance_basis(**{**kw,**patch})
        b,_=tensor_variance_basis(**{**kw,'time_knots':[1.],'moneyness_knots':[0.]})
        self.assertEqual(b.shape,(6,1))


if __name__=='__main__':unittest.main()
