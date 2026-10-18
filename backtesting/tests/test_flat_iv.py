from copy import deepcopy
import unittest
import numpy as np
from rust_pricing_backtest.domain import BacktestError
from rust_pricing_backtest.flat_iv import recalibrate_flat_market_iv, maturity_risk_statistics


def payload():
    return {'model':{'type':'local_volatility',
        'reporting_iv_basis':{'maturity_nodes':[.5,1.],'log_forward_moneyness_nodes':[-.1,0.,.1],
                              'implied_volatilities':[.2]*6},
        'local_variance_grid':{'time_nodes':[0.,.5,1.],'log_forward_moneyness_nodes':[-.2,.2],
                               'values':[.04]*6,'floor':1e-8,'cap':4.}},'product':{'history':[98.]}}


class FlatMarketIvTests(unittest.TestCase):
    def test_first_maturity_shock_recalibrates_next_interval_and_preserves_terminal_quote(self):
        p=payload();before=deepcopy(p);out,a=recalibrate_flat_market_iv(p,[.01,0.])
        self.assertEqual(p,before);self.assertEqual(out['product'],p['product'])
        v=out['model']['local_variance_grid']['values']
        self.assertAlmostEqual(v[0],.21**2);self.assertAlmostEqual(.5*v[0]+.5*v[2],.04)
        self.assertLess(v[2],.04);np.testing.assert_allclose(a['implied_volatilities'],[.21,.2],rtol=1e-15)
        self.assertFalse(a['clipping_applied']);self.assertNotEqual(a['source_request_hash'],a['recalibrated_request_hash'])

    def test_last_quote_shift_changes_only_last_interval_and_roundtrip(self):
        p=payload();up,_=recalibrate_flat_market_iv(p,[0.,.01]);back,_=recalibrate_flat_market_iv(up,[0.,-.01])
        self.assertAlmostEqual(up['model']['local_variance_grid']['values'][0],.04)
        np.testing.assert_allclose(back['model']['local_variance_grid']['values'],[.04]*6,rtol=1e-14)

    def test_calendar_floor_cap_smile_and_mismatched_base_are_rejected(self):
        for shifts in [[.3,0.],[0.,-.19999],[0.,3.],[0.],[[0.,0.]],[0.,float('nan')]]:
            with self.assertRaises(BacktestError):recalibrate_flat_market_iv(payload(),shifts)
        for key,value in [('implied_volatilities',[.2,.21,.2,.2,.2,.2]),('maturity_nodes',[.25,1.])]:
            p=payload();p['model']['reporting_iv_basis'][key]=value
            with self.assertRaises(BacktestError):recalibrate_flat_market_iv(p,[0.,0.])
        p=payload();p['model']['local_variance_grid']['values']=[.05]*6
        with self.assertRaisesRegex(BacktestError,'base IV'):recalibrate_flat_market_iv(p,[0.,0.])

    def test_projection_matches_actual_correlated_samples_and_preserves_signed_cross_terms(self):
        samples=np.array([[1.,2.,-3.,1.],[3.,5.,-8.,2.],[-1.,-2.,3.,-1.],[4.,6.,-10.,3.]])
        d={'reporting_maturities':[.5,1.],'reporting_log_nodes':[-.1,.1],
           'estimate':{'independent_sampling_units':4,'raw_buckets':[{'mean':v} for v in samples.mean(axis=0)],
                       'full_bucket_sample_covariance':np.cov(samples,rowvar=False,ddof=1).ravel().tolist()}}
        r=maturity_risk_statistics(d);projected=samples.reshape(4,2,2).sum(axis=2)
        np.testing.assert_allclose(r['means'],projected.mean(axis=0),atol=1e-15)
        np.testing.assert_allclose(r['mean_covariance'],np.cov(projected,rowvar=False,ddof=1)/4,atol=1e-15)
        self.assertLess(r['mean_covariance'][0][1],0.)
        d['estimate']['full_bucket_sample_covariance']=None
        with self.assertRaisesRegex(BacktestError,'full joint'):maturity_risk_statistics(d)
        d['estimate']=None
        with self.assertRaisesRegex(BacktestError,'recovery'):maturity_risk_statistics(d)

if __name__=='__main__':unittest.main()
