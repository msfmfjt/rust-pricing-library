"""Portfolio covariance and fixed-strike attribution boundary contracts."""
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
import unittest
from unittest.mock import patch
import numpy as np
from rust_pricing_backtest.domain import BacktestError, fingerprint
from rust_pricing_backtest.portfolio_market_risk import sampling_covariance,explain_pnl,check_report,evaluate_portfolio,explain_backtest


def fixture():
    p=dict(valuation_date='2026-01-05',market=dict(spot=100.),
        model=dict(reporting_iv_basis=dict(maturity_nodes=[30/365,90/365],
            log_forward_moneyness_nodes=[-.1,0.,.1],implied_volatilities=[.2,.21,.22,.23,.24,.25])))
    requests={'a':deepcopy(p),'b':deepcopy(p)}
    r=dict(implementation_identity='unit',status='passed',instrument_order=['a','b'],quantities={'a':-2.,'b':3.},
        request_hashes={cid:fingerprint(v) for cid,v in requests.items()},
        net_buckets=[1.,2.,3.,4.,5.,6.],
        instruments={'a':dict(frozen_grid_delta=.5),'b':dict(frozen_grid_delta=.25)},
        net_price_and_market_mean_covariance=np.eye(7).tolist())
    r['report_hash']=fingerprint(r)
    return requests,r


class PortfolioMarketRiskTests(unittest.TestCase):
    def setUp(self):
        patcher=patch('rust_pricing_backtest.portfolio_market_risk.implementation_identity',return_value='unit')
        patcher.start();self.addCleanup(patcher.stop)

    def test_cross_covariance_and_exact_offset(self):
        a=np.array([[1.,2.],[3.,-1.],[-4.,6.],[2.,0.]])
        joined=np.c_[a,a,-2*a]
        c=sampling_covariance(joined)
        self.assertGreater(abs(c[0,2]),0)
        np.testing.assert_allclose(c[:2,4:],-2*c[:2,:2])
        w=np.kron([[1.,-1.,0.]],np.eye(2))
        np.testing.assert_array_equal(w@c@w.T,np.zeros((2,2)))
        np.testing.assert_allclose(c,np.cov(joined,rowvar=False,ddof=1))
        self.assertGreaterEqual(np.linalg.eigvalsh(c).min(),-1e-12)

    def test_pair_units_are_not_counted_twice(self):
        mates=np.array([[[1.,4.],[9.,2.]],[[2.,1.],[-6.,7.]],[[8.,-1.],[4.,3.]]])
        units=mates.mean(axis=1)
        c=sampling_covariance(units)
        self.assertEqual(c[0,0],19.)
        self.assertNotAlmostEqual(c[0,0]/3,np.cov(mates.reshape(-1,2),rowvar=False)[0,0]/6)

    def test_bad_samples_fail_closed(self):
        for a in ([[1.,2.]],[[1.,float('nan')],[2.,3.]],[[1e308],[-1e308]],[[],[]]):
            with self.assertRaises(BacktestError):sampling_covariance(a)

    def explain(self,**kwargs):
        requests,report=fixture();base=dict(current_spot=101.,current_iv=np.array([[.201,.211,.221],[.231,.241,.251]]),actual_asset_pnl=2.)
        with patch('rust_pricing_backtest.portfolio_market_risk.spot_iv_response',return_value=(np.full(6,.002),{})):
            return explain_pnl(report,requests,**(base|kwargs))

    def test_fixed_strike_delta_correction_and_uncertainty(self):
        row=self.explain()
        self.assertAlmostEqual(row['frozen_grid_delta'],-.25)
        self.assertAlmostEqual(row['delta_correction'],.042)
        self.assertAlmostEqual(row['delta_pnl'],-.292)
        self.assertAlmostEqual(row['vega_pnl'],.021)
        self.assertAlmostEqual(row['available_greek_pnl']+row['unexplained_pnl'],2.)
        self.assertAlmostEqual(row['vega_and_delta_correction_standard_error'],np.sqrt(6)*.001)
        self.assertIn('Theta',row['residual_components'])
        self.assertIsNone(row['gamma_pnl'])

    def test_static_strike_surface_and_absolute_date_roll(self):
        axis=np.array([-.3,-.1,0.,.1,.3]);current=110.
        # A static fixed-strike smile: its k samples move with the forward.
        vols=np.array([.21+.1*(axis+np.log(current/100)),.24+.1*(axis+np.log(current/100))])
        maturities=['2026-02-04','2026-04-05']
        row=self.explain(current_spot=current,current_iv=vols,current_date='2026-01-06',
            current_maturity_dates=maturities,current_log_moneyness=axis,current_forward=[current,current])
        np.testing.assert_allclose(row['iv_change'],0.,atol=1e-16)
        self.assertAlmostEqual(row['vega_pnl'],0.)
        for bad in (['2026-02-05','2026-04-06'],['2026-02-04','2026-02-04']):
            with self.assertRaises(BacktestError):
                self.explain(current_iv=vols,current_date='2026-01-06',current_maturity_dates=bad,
                    current_log_moneyness=axis,current_forward=[current,current])

    def test_incomplete_surface_expiry_and_nonfinite_pnl_rejected(self):
        cases=[dict(current_iv=[[.2]]),dict(current_date='2026-02-04'),dict(current_date='2026-01-04'),
               dict(actual_asset_pnl=float('nan')),dict(current_spot=0),dict(current_iv=[[-.1]*3]*2),
               dict(current_iv=[[.2]*3]*2,current_maturity_dates=['2026-02-04','2026-04-05'],
                    current_log_moneyness=[-.1,0.,.1],current_forward=[200.,200.])]
        for kw in cases:
            with self.subTest(kw=kw),self.assertRaises(BacktestError):self.explain(**kw)

    def test_report_tampering_and_wrong_opening_requests_rejected(self):
        requests,report=fixture();check_report(report,requests)
        report['net_buckets'][0]+=1
        with self.assertRaisesRegex(BacktestError,'portfolio_report'):check_report(report,requests)
        requests,report=fixture();requests['a']['market']['spot']=101.
        with self.assertRaisesRegex(BacktestError,'portfolio_report'):check_report(report,requests)

    def test_missing_surface_stays_unexplained_despite_offsetting_positions(self):
        @dataclass
        class Saved:
            manifest:dict
            ledger:list
            cost_basis:list
            risks:list
            pnl_components:list
        times=['2026-01-05T16:00:00+00:00','2026-01-06T16:00:00+00:00']
        requests,_=fixture()
        history=[dict(at=t,pricing_requests={cid:__import__('json').dumps(p) for cid,p in requests.items()},
                      spot={'bid':100.,'ask':100.}) for t in times]
        basis=dict(maturity_dates=['2026-02-04','2026-04-05'],log_moneyness=[-.1,0.,.1])
        inventory=dict(targets={'a':dict(quantity=1.,multiplier=1.),'b':dict(quantity=-1.,multiplier=1.)},hedges={},stock={})
        saved=Saved(dict(market_history=history,full_history_hash=fingerprint(history)),
            [dict(at=times[0],daily_pnl=0.,cumulative_pnl=0.),dict(at=times[1],daily_pnl=5.,cumulative_pnl=5.)],
            [dict(at=t,positions=inventory) for t in times],
            [dict(at=t,basis=basis,instruments={'a':{},'b':{}}) for t in times],
            [dict(target=0.,option_hedge=0.,stock=0.,total=0.,tolerance=1e-9,cumulative_tolerance=1e-9),
             dict(target=4.,option_hedge=0.,stock=0.,total=5.,tolerance=1e-9,cumulative_tolerance=1e-9)])
        with patch('rust_pricing_backtest.portfolio_market_risk.evaluate_portfolio') as pricing:
            report=explain_backtest(saved)
            pricing.assert_not_called()
        self.assertEqual(report['status'],'partial')
        self.assertEqual(report['covered_intervals'],0)
        row=report['rows'][1]
        self.assertEqual(row['reason']['code'],'pnl_surface')
        self.assertEqual(row['unexplained_pnl'],4.)
        self.assertEqual(row['non_asset_pnl'],1.)
        self.assertEqual(row['reconciliation_error'],0.)


if __name__=='__main__':unittest.main()
