from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
from rust_pricing_backtest.three_step_smile import ThreeStepTerminalSmileCalibrator
from rust_pricing_backtest.domain import BacktestError

ROOT=Path(__file__).resolve().parents[2]
def payload():return json.loads((ROOT/'fixtures/local-vol/three-step-gamma-v0.1.json').read_text())['request']

class ThreeStepSmileTests(unittest.TestCase):
    def test_terminal_fit_preserves_earlier_rows_quotes_and_context(self):
        p=payload();before=deepcopy(p);c=ThreeStepTerminalSmileCalibrator(p)
        for shift in [-.003,.003]:
            q,a=c.calibrate([shift]*5);ThreeStepTerminalSmileCalibrator(q)
            self.assertEqual(q['model']['local_variance_grid']['values'][:66],p['model']['local_variance_grid']['values'][:66])
            self.assertEqual(q['model']['reporting_iv_basis']['implied_volatilities'][:10],p['model']['reporting_iv_basis']['implied_volatilities'][:10])
            self.assertLess(a['maximum_verified_price_error_bound'],1e-10);self.assertFalse(a['clipping_applied'])
        self.assertEqual(p,before);self.assertEqual(c.payload,before)

    def test_terminal_tangent_matches_nonlinear_calibration(self):
        c=ThreeStepTerminalSmileCalibrator(payload());U,a=c.terminal_iv_tangent()
        self.assertLess(a['iv_response_residual'],1e-9)
        for direction in [np.eye(5)[2],np.array([.2,-.3,.4,.1,-.2])]:
            errors=[]
            for h in [1e-5,5e-6,2e-6,1e-6]:
                rows=[np.array(c.calibrate(sign*h*direction)[0]['model']['local_variance_grid']['values'])[66:99] for sign in [-1,1]]
                errors.append(np.max(abs((rows[1]-rows[0])/(2*h)-U@direction)))
            # The ill-conditioned quote-to-node map needs smaller bumps than
            # price-risk checks; verify second-order convergence, then accuracy.
            self.assertLess(errors[1],.3*errors[0]);self.assertLess(errors[3],.3*errors[2])
            self.assertLess(errors[-1],2e-6)

    def test_source_probability_martingale_and_no_renormalization(self):
        c=ThreeStepTerminalSmileCalibrator(payload());a=c.source_audit
        self.assertAlmostEqual(a['source_probability'],1.,places=12);self.assertAlmostEqual(a['source_first_spot_moment'],100.,places=10)
        self.assertFalse(a['renormalized']);self.assertGreater(a['source_spot_tail_bound'],0.)
        self.assertLess(a['source_spot_tail_bound'],1e-20)

    def test_scope_shape_nonfinite_and_base_mismatch_fail(self):
        p=payload()
        for change in ['first','terminal_quote','shape','time']:
            q=deepcopy(p)
            if change=='first':q['model']['local_variance_grid']['values'][0]+=.001
            elif change=='terminal_quote':q['model']['reporting_iv_basis']['implied_volatilities'][-1]+=.001
            elif change=='shape':q['model']['local_variance_grid']['shape']=[3,33]
            else:q['model']['local_variance_grid']['time_nodes'][-1]=float('nan')
            with self.assertRaises(BacktestError):ThreeStepTerminalSmileCalibrator(q)
        for order in [15,65,1.5]:
            with self.assertRaises(BacktestError):ThreeStepTerminalSmileCalibrator(p,quadrature_order=order)

    def test_unreachable_and_nonconvergent_shifts_are_not_clipped(self):
        c=ThreeStepTerminalSmileCalibrator(payload())
        for shift in [[0.]*4,[float('nan')]*5,[-.1]*5,[-.3]*5,[10.]*5]:
            with self.assertRaises(BacktestError):c.calibrate(shift)
        with self.assertRaises(BacktestError):ThreeStepTerminalSmileCalibrator(payload(),maximum_iterations=1).calibrate([.003]*5)

if __name__=='__main__':unittest.main()
