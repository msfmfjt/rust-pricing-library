from copy import deepcopy
import math
import unittest
import numpy as np
from rust_pricing_backtest.domain import BacktestError
from rust_pricing_backtest.discrete_smile import TwoStepSmileCalibrator


def payload():
    n = 33
    return dict(market=dict(type='equity', spot=100.,
                           discount_curve=dict(times=[0., 1.], discount_factors=[1., 1.]),
                           dividend_curve=dict(times=[0., 1.], discount_factors=[1., 1.])),
                model=dict(type='local_volatility',
                           local_variance_grid=dict(time_nodes=[0., .2, .4],
                               log_forward_moneyness_nodes=np.linspace(-.8, .8, n).tolist(),
                               values=[.04]*(3*n), floor=1e-8, cap=4.),
                           reporting_iv_basis=dict(maturity_nodes=[.2, .4],
                               log_forward_moneyness_nodes=[-.2, -.1, 0., .1, .2],
                               implied_volatilities=[.2]*10)), product=dict(history=[98.]))


class DiscreteSmileTests(unittest.TestCase):
    def test_nonlinear_quote_fit_preserves_first_marginal_and_source(self):
        p = payload(); before = deepcopy(p); c = TwoStepSmileCalibrator(p)
        for direction in (c.quote_nodes, c.quote_nodes**2, np.eye(5)[2]):
            for sign in (-1., 1.):
                out, audit = c.calibrate(sign*.003*direction)
                self.assertEqual(p, before); self.assertEqual(out['product'], p['product'])
                self.assertEqual(out['model']['local_variance_grid']['values'][:33], [.04]*33)
                self.assertEqual(out['model']['reporting_iv_basis']['implied_volatilities'][:5], [.2]*5)
                self.assertLessEqual(audit['maximum_verified_price_error_bound'], 1e-10)
                self.assertFalse(audit['clipping_applied']); self.assertTrue(audit['diagnostic_only'])
                self.assertEqual(audit['full_market_iv_bucket_acceptance'], 'not_established')
                self.assertNotEqual(audit['source_request_hash'], audit['recalibrated_request_hash'])

    def test_nonflat_base_recalibration_and_price_roundtrip(self):
        c = TwoStepSmileCalibrator(payload())
        shift = .015*c.quote_nodes+.02*c.quote_nodes**2
        out, _ = c.calibrate(shift)
        self.assertGreater(np.ptp(out['model']['local_variance_grid']['values'][33:66]), .001)
        back, audit = TwoStepSmileCalibrator(out).calibrate(-shift)
        np.testing.assert_allclose(back['model']['reporting_iv_basis']['implied_volatilities'], .2, atol=1e-15)
        self.assertLessEqual(audit['maximum_verified_price_error_bound'], 1e-10)
        # Underdetermined local variance need not roundtrip: only quotes do.
        TwoStepSmileCalibrator(back)

    def test_calibrated_price_differences_match_frozen_tangent(self):
        c = TwoStepSmileCalibrator(payload(), price_tolerance=1e-12); u, audit = c.terminal_iv_tangent()
        self.assertEqual(audit['rank'], 5); self.assertLess(audit['iv_response_residual'], 1e-12)
        for direction in (c.quote_nodes, c.quote_nodes**2, np.eye(5)[2]):
            approximations = []
            for bump in (2e-6, 1e-6):
                rows = [np.array(c.calibrate(sign*bump*direction)[0]['model']['local_variance_grid']['values'])[33:66]
                        for sign in (1., -1.)]
                approximations.append((rows[0]-rows[1])/(2*bump))
            self.assertLess(np.max(abs(approximations[1]-u@direction)), 2e-6)
            self.assertLessEqual(np.linalg.norm(approximations[1]-u@direction),
                                 .4*np.linalg.norm(approximations[0]-u@direction)+1e-8)

    def test_unreachable_first_marginal_carry_and_dividends_reject(self):
        mutations = [lambda p: p['market']['discount_curve']['discount_factors'].__setitem__(1, .99),
                     lambda p: p['market'].update(discrete_dividends=[dict(ex_time=.1, amount=1.)]),
                     lambda p: p['model']['reporting_iv_basis']['implied_volatilities'].__setitem__(0, .201),
                     lambda p: p['model']['local_variance_grid']['values'].__setitem__(0, .05),
                     lambda p: p['model']['reporting_iv_basis']['maturity_nodes'].__setitem__(0, .1)]
        for mutate in mutations:
            p = payload(); mutate(p)
            with self.assertRaises(BacktestError): TwoStepSmileCalibrator(p)

    def test_tangent_matches_independent_gaussian_bridge_hat_expectations(self):
        c = TwoStepSmileCalibrator(payload()); inverse, _ = c.terminal_iv_tangent()
        total = .04*c.times[2]; conditional_sd = math.sqrt(c.v0*(total-c.v0)/total)
        rows = []
        for log_strike in c.quote_nodes:
            mean = c.v0/total*log_strike
            # Integrate each affine hat using truncated-normal first moments.
            cuts = np.r_[-math.inf, c.nodes, math.inf]; hats = np.zeros(len(c.nodes))
            for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
                za, zb = (a-mean)/conditional_sd, (b-mean)/conditional_sd
                probability = .5*(math.erfc(-zb/math.sqrt(2))-math.erfc(-za/math.sqrt(2)))
                first = mean*probability+conditional_sd*(math.exp(-za*za/2)-math.exp(-zb*zb/2))/math.sqrt(2*math.pi)
                if i == 0: hats[0] += probability
                elif i == len(c.nodes): hats[-1] += probability
                else:
                    hats[i-1] += (b*probability-first)/(b-a)
                    hats[i] += (first-a*probability)/(b-a)
            rows.append(hats*c.dt/(2*.2*c.times[2]))
        np.testing.assert_allclose(np.array(rows)@inverse, np.eye(5), atol=2e-13)

    def test_malformed_shapes_and_nonfinite_values_reject(self):
        for name, values in [('values', [.04]*98), ('values', [float('nan')]*99),
                             ('log_forward_moneyness_nodes', [0., 0.]), ('shape', [2, 33])]:
            p = payload(); p['model']['local_variance_grid'][name] = values
            with self.assertRaises(BacktestError): TwoStepSmileCalibrator(p)
        c = TwoStepSmileCalibrator(payload())
        for shift in ([0.], [[0.]*5], [float('inf')]*5, [-.21]*5):
            with self.assertRaises(BacktestError): c.calibrate(shift)
        p = payload(); p['model']['reporting_iv_basis']['implied_volatilities'][7] += .001
        with self.assertRaisesRegex(BacktestError, 'base terminal quotes'): TwoStepSmileCalibrator(p)

    def test_arbitrage_rank_floor_cap_and_nonconvergence_fail_closed(self):
        c = TwoStepSmileCalibrator(payload())
        with self.assertRaisesRegex(BacktestError, 'fixed first marginal'): c.calibrate([-.1]*5)
        with self.assertRaisesRegex(BacktestError, 'convexity'): c.calibrate([0., 0., .2, 0., 0.])
        with self.assertRaisesRegex(BacktestError, 'iteration limit'):
            TwoStepSmileCalibrator(payload(), maximum_iterations=1).calibrate([0., 0., .003, 0., 0.])
        p = payload(); p['model']['local_variance_grid']['cap'] = .04001
        with self.assertRaisesRegex(BacktestError, 'nonconvergence'):
            TwoStepSmileCalibrator(p).calibrate([.003]*5)
        p = payload(); p['model']['reporting_iv_basis']['log_forward_moneyness_nodes'] = np.linspace(-1e-6, 1e-6, 5).tolist()
        with self.assertRaisesRegex(BacktestError, 'rank deficient'):
            TwoStepSmileCalibrator(p).calibrate([0.]*5)


if __name__ == '__main__': unittest.main()
