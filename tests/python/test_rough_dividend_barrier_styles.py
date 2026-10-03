"""Independent controls for hard Barrier directions, option sides and cash order."""
import math
from itertools import product
import unittest

import numpy as np

from rough_dividend_barrier_styles import inputs
from rough_dividend_survival_reference import path_values


class BarrierStyles(unittest.TestCase):
    def test_deterministic_dividend_single_monitor_against_payoff_quadrature(self):
        _, bases, market = inputs()
        t = market['expiry_time']
        case = dict(bases['h01_n8_four'], eta=0., kappa=0.,
            times=[0., t], monitoring_indices=[1], squared_leverage=[.04]*6,
            equity_volatility_correlation=0., dividend_volatility_correlation=0.)
        market = dict(market, dividend_volatility=0., equity_dividend_correlation=0.)
        x, w = np.polynomial.legendre.leggauss(256)

        def quadrature(spot, barrier, strike, up, call, knock_in, notional, rebate, monitors, historical_hit):
            growth = market['annual_carry'] / market['annual_discount']
            funded = spot-sum(q/growth**ex for q, ex in zip(market['cash_means'], market['cash_times']))
            a = funded*growth**t
            b = sum(q*growth**(t-ex) for q, ex in zip(market['cash_means'], market['cash_times']) if ex > t)
            cash = sum(q for q, ex in zip(market['cash_means'], market['cash_times']) if ex == t)
            sd, mu = math.sqrt(.04*t), -.02*t
            # Split the direct Gaussian payoff integral at every discontinuity
            # and exercise boundary; this does not use truncated-moment formulas.
            splits = [-12., 12.]
            for cut in [(strike-b)/a, (barrier-b-(cash if up else 0))/a]:
                if cut > 0:
                    point = (math.log(cut)-mu)/sd
                    if -12 < point < 12: splits.append(point)
            splits.sort()
            value = 0.
            for left, right in zip(splits[:-1], splits[1:]):
                z = (left+right)/2 + (right-left)*x/2
                stock = a*np.exp(mu+sd*z)+b
                hit = np.full(len(z), historical_hit, dtype=bool)
                if 1 in monitors:
                    hit |= stock+cash >= barrier if up else stock <= barrier
                if 0 in monitors:
                    hit |= spot >= barrier if up else spot <= barrier
                active = hit if knock_in else ~hit
                payoff = notional*np.maximum(stock-strike if call else strike-stock, 0)*active + rebate*(~active)
                value += (right-left)/2 * (w @ (payoff*np.exp(-z*z/2)/math.sqrt(2*math.pi)))
            return value*market['annual_discount']**market['payment_time']

        for notional, rebate in [(1., 0.), (2., 7.)]:
            for strike, up, call, knock_in, monitors, barrier, historical_hit in product(
                    (1., 80., 110., 1000.), (True, False), (True, False), (True, False),
                    ([], [1], [0], [0, 1]), (95., 105.), (False, True)):
                with self.subTest(notional=notional, rebate=rebate, strike=strike, up=up,
                                  call=call, knock_in=knock_in, monitors=monitors, barrier=barrier, historical_hit=historical_hit):
                    actual = path_values(dict(case, monitoring_indices=monitors),
                        dict(market, barrier=barrier, strike=strike),
                        np.zeros((1,1,3)), np.full((1,1),.5),
                        direction='up' if up else 'down', side='call' if call else 'put',
                        style='knock_in' if knock_in else 'knock_out', notional=notional, rebate=rebate, historical_hit=historical_hit)[0]
                    price = lambda spot: quadrature(spot, barrier, strike, up, call, knock_in, notional, rebate, monitors, historical_hit)
                    spot, bump = market['spot'], .01
                    coarse = (price(spot+bump)-price(spot-bump))/(2*bump)
                    fine = (price(spot+bump/2)-price(spot-bump/2))/bump
                    expected = [price(spot), (4*fine-coarse)/3]
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=8e-9)

    def test_analytic_tangents_for_every_contract_variant(self):
        cfg, bases, market = inputs("rough-barrier-rebates-reference.json")
        rng = np.random.Generator(np.random.PCG64(930))
        z, u = rng.standard_normal((32,8,3)), rng.random((32,8))
        for row in cfg['cases']:
            contract = row['contract']
            m = dict(market, barrier=contract['barrier'], strike=contract['strike'])
            kwargs = {k: contract[k] for k in ('direction','side','style','notional','rebate')}
            base = bases[row['base_case']]
            analytic = path_values(base,m,z,u,**kwargs)[:,1]
            for bump in (1e-3,5e-4):
                fd = (path_values(base,m,z,u,spot=100+bump,**kwargs)[:,0]
                      -path_values(base,m,z,u,spot=100-bump,**kwargs)[:,0])/(2*bump)
                np.testing.assert_allclose(analytic,fd,rtol=0,atol=2e-6)

    def test_retained_batch_errors_and_precision(self):
        for filename in ("rough-barrier-styles-reference.json", "rough-barrier-rebates-reference.json"):
            cfg, _, _ = inputs(filename)
            for row in cfg['cases']:
                means = np.asarray(row['batch_means'])
                self.assertEqual(means.shape, (cfg['sampling']['batches'],2))
                se = means.std(axis=0,ddof=1)/np.sqrt(len(means))
                for j,q in enumerate(('price','delta')):
                    self.assertAlmostEqual(means[:,j].mean(),row[q],delta=2e-13)
                    self.assertAlmostEqual(se[j],row[q+'_se'],delta=2e-13)
                    self.assertLess(se[j],cfg['acceptance']['reference_'+q+'_se'])


if __name__ == '__main__':
    unittest.main()
