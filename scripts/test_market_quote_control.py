"""Analytical shared-error examples and rejection tests for quote controls."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np

from market_quote_control import apply_control
from multi_step_cases import model
from rust_pricing_backtest.domain import BacktestError,fingerprint
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.portfolio_market_risk import sampling_covariance


class QuoteControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=model(6);c=MultiStepExtensionCalibrator(cls.base,extension='all_rows')
        cls.tangent=c.all_maturity_iv_tangent()[0];cls.q=c.iv.size

    def setUp(self):
        p=patch('rust_pricing_backtest.portfolio_market_risk.implementation_identity',return_value='synthetic-test')
        p.start();self.addCleanup(p.stop)

    def fixture(self):
        ids=['control','lower','upper'];requests={cid:deepcopy(self.base) for cid in ids}
        for cid,strike in zip(ids,(100.,95.,105.)):requests[cid]['product']['strike']=strike
        unit=np.array([-3.,-1.,1.,3.]);load=np.arange(1,self.q+1)*.1
        cube=np.zeros((4,3,self.q+1));cube[:,:,0]=10.+unit[:,None]
        cube[:,:,1:]=unit[:,None,None]*load[None,None,:]+np.arange(3)[None,:,None]
        r=dict(status='passed',implementation_identity='synthetic-test',instrument_order=ids,
            request_hashes={cid:fingerprint(p) for cid,p in requests.items()},extension='all_rows',variance_basis=None,
            independent_sampling_units=4,calibration_tangent=self.tangent.tolist(),sampling_unit_values=cube.tolist(),
            price_and_market_sample_covariance=sampling_covariance(cube.reshape(4,-1)).tolist(),
            audits={cid:dict(status='path_gamma_chain_passed',recovery_passed=True,validation_passed=True) for cid in ids},
            instruments={cid:dict(price=10.,buckets=cube[:,i,1:].mean(axis=0).tolist()) for i,cid in enumerate(ids)},
            quantities={'control':0.,'lower':-1.,'upper':-1.})
        r['report_hash']=fingerprint(r)
        policy=dict(control_instrument='control',coefficient_for_every_instrument=1.,coefficient_fitted_from_samples=False)
        return r,requests,policy,cube

    def test_fixed_shared_control_removes_shared_error_and_preserves_prices(self):
        native,requests,policy,cube=self.fixture();before=deepcopy(native)
        result=apply_control(native,requests,policy);adjusted=np.array(result['sampling_unit_values'])
        np.testing.assert_array_equal(adjusted[:,:,0],cube[:,:,0])
        known=np.array(result['control_variate']['expected_gradient'])
        np.testing.assert_allclose(adjusted[:,:,1:],np.broadcast_to(known+np.arange(3)[:,None],(4,3,self.q)),rtol=0,atol=1e-14)
        self.assertLess(max(result['net_bucket_standard_errors']),1e-13)
        self.assertEqual(native,before)
        self.assertEqual(result['audits'],native['audits'])

    def test_lost_cross_instrument_covariance_is_rejected(self):
        native,requests,policy,_=self.fixture()
        cov=np.array(native['price_and_market_sample_covariance']);width=self.q+1
        for i in range(3):
            for j in range(3):
                if i!=j:cov[i*width:(i+1)*width,j*width:(j+1)*width]=0.
        native['price_and_market_sample_covariance']=cov.tolist();native.pop('report_hash');native['report_hash']=fingerprint(native)
        with self.assertRaisesRegex(BacktestError,'quote_control_covariance'):apply_control(native,requests,policy)

    def test_nonquote_control_and_fitted_coefficient_are_rejected(self):
        native,requests,policy,_=self.fixture();requests['control']['product']['strike']=100.01
        native['request_hashes']['control']=fingerprint(requests['control']);native.pop('report_hash');native['report_hash']=fingerprint(native)
        with self.assertRaisesRegex(BacktestError,'control must exactly equal'):apply_control(native,requests,policy)
        native,requests,policy,_=self.fixture();policy['coefficient_fitted_from_samples']=True
        with self.assertRaisesRegex(BacktestError,'quote_control_policy'):apply_control(native,requests,policy)

    def test_control_does_not_hide_native_failure(self):
        native,requests,policy,_=self.fixture();native['audits']['lower']['recovery_passed']=False
        native.pop('report_hash');native['report_hash']=fingerprint(native)
        with self.assertRaisesRegex(BacktestError,'quote_control_native'):apply_control(native,requests,policy)


if __name__=='__main__':unittest.main()
