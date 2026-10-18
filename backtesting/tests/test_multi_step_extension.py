"""All-row Jacobians, explicit extension metrics, rank and nonlinear fit gates."""
from copy import deepcopy
import unittest
import numpy as np
from test_multi_step_full_smile import model
from rust_pricing_backtest.domain import BacktestError
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator as Calibrator


class MultiStepExtensionTests(unittest.TestCase):
    def test_every_transition_score_matches_independent_price_differences(self):
        c=Calibrator(model(6));_,j,_=c._surface_prices(c.variance,64,jacobian=True)
        for row in range(6):
            for node in [14,16,18]:
                p=[];h=1e-6
                for sign in [-1,1]:
                    v=c.variance.copy();v[row,node]+=sign*h
                    p.append(c._surface_prices(v,64)[0])
                np.testing.assert_allclose((p[1]-p[0])/(2*h),j[:,row*33+node],atol=2e-7,rtol=2e-8)
        self.assertGreater(np.linalg.norm(j[:,:33]),1.)

    def test_presets_fit_all_maturities_and_tangent_matches_actual_recalibration(self):
        for n in [4,6]:
            for extension in ['all_rows','linear_time']:
                p=model(n);original=deepcopy(p);c=Calibrator(p,extension=extension)
                tangent,a=c.all_maturity_iv_tangent();self.assertLess(a['iv_response_residual'],1e-9)
                direction=np.linspace(-.2,.3,c.iv.size).reshape(c.iv.shape);h=1e-6;values=[]
                for sign in [-1,1]:
                    q,b=c.calibrate_all_maturities(sign*h*direction)
                    v=np.array(q['model']['local_variance_grid']['values']).reshape(c.variance.shape)
                    np.testing.assert_array_equal(v[-1],c.variance[-1])
                    np.testing.assert_allclose((v[:-1]-c.variance[:-1]).ravel(),c.basis@b['parameter_increments'],atol=1e-17,rtol=0)
                    values.append(v[:-1].ravel())
                np.testing.assert_allclose((values[1]-values[0])/(2*h),tangent@direction.ravel(),atol=2e-6,rtol=0)
                q,b=c.calibrate_all_maturities(.001*direction);verified=Calibrator(q,extension=extension)
                self.assertLess(b['maximum_verified_price_error_bound'],1e-10)
                self.assertTrue(np.all(np.max(abs(verified.variance[:-1]-c.variance[:-1]),axis=1)>1e-7))
                self.assertEqual(p,original);self.assertEqual(c.payload,original)
                same,_=c.calibrate_all_maturities(np.zeros_like(c.iv));self.assertEqual(same,original)

    def test_custom_basis_is_copied_and_has_an_explicit_metric(self):
        p=model(4);identity=np.eye(132);c=Calibrator(p,extension='custom',variance_basis=identity)
        identity[:]=0.;self.assertEqual(c.basis[0,0],1.)
        with self.assertRaises(ValueError):c.basis[0,0]=2.
        t,_=c.all_maturity_iv_tangent();u,_=Calibrator(p).all_maturity_iv_tangent()
        np.testing.assert_array_equal(t,u)
        narrow=np.zeros((132,10));narrow[:33,:]=1.
        for basis in [np.zeros((132,132)),narrow,np.ones((132,1))]:
            with self.assertRaisesRegex(BacktestError,'extension_rank'):
                Calibrator(p,extension='custom',variance_basis=basis).all_maturity_iv_tangent()

    def test_invalid_targets_bounds_and_nonconvergence_never_mutate_inputs(self):
        p=model(4);c=Calibrator(p);before=deepcopy(p)
        for shifts in [[],np.zeros(5),[['bad']*5]*2,np.full_like(c.iv,np.nan),np.full_like(c.iv,-1.)]:
            with self.assertRaises(BacktestError):c.calibrate_all_maturities(shifts)
        extreme=np.zeros_like(c.iv);extreme[0,2]=.2
        with self.assertRaisesRegex(BacktestError,'extension_arbitrage'):c.calibrate_all_maturities(extreme)
        with self.assertRaisesRegex(BacktestError,'extension_nonconvergence'):
            Calibrator(p,maximum_iterations=1).calibrate_all_maturities(np.full_like(c.iv,.003))
        for kwargs in [dict(extension='other'),dict(extension='custom'),dict(variance_basis=np.eye(132)),
                dict(extension='custom',variance_basis=np.ones((133,3))),
                dict(extension='custom',variance_basis=np.full((132,10),np.nan))]:
            with self.assertRaises(BacktestError):Calibrator(p,**kwargs)
        self.assertEqual(p,before);self.assertEqual(c.payload,before)

    def test_same_market_quotes_do_not_identify_a_unique_exotic_extension(self):
        c=Calibrator(model(4));d=Calibrator(model(4),extension='linear_time')
        shifts=np.full_like(c.iv,.001);p,_=c.calibrate_all_maturities(shifts);q,_=d.calibrate_all_maturities(shifts)
        self.assertEqual(p['model']['reporting_iv_basis'],q['model']['reporting_iv_basis'])
        self.assertGreater(np.max(abs(np.array(p['model']['local_variance_grid']['values'])-q['model']['local_variance_grid']['values'])),1e-5)
        t,_=c.all_maturity_iv_tangent();u,_=d.all_maturity_iv_tangent()
        self.assertGreater(np.max(abs(t-u)),.01)


if __name__=='__main__':unittest.main()
