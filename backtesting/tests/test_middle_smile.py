from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
from rust_pricing_backtest.three_step_smile import ThreeStepTerminalSmileCalibrator
from rust_pricing_backtest.domain import BacktestError

ROOT=Path(__file__).resolve().parents[2]
def payload(): return json.loads((ROOT/'fixtures/local-vol/three-step-gamma-v0.1.json').read_text())['request']

class MiddleTerminalSmileTests(unittest.TestCase):
    def test_joint_fit_preserves_first_and_holds_unshifted_terminal_quotes(self):
        p=payload();before=deepcopy(p);c=ThreeStepTerminalSmileCalibrator(p)
        for direction in [np.array([.001,-.0003,.0002,-.0001,.0005]),np.full(5,-.003)]:
            q,a=c.calibrate_joint(direction,np.zeros(5));ThreeStepTerminalSmileCalibrator(q)
            self.assertEqual(q['model']['local_variance_grid']['values'][:33],p['model']['local_variance_grid']['values'][:33])
            self.assertEqual(q['model']['reporting_iv_basis']['implied_volatilities'][10:],p['model']['reporting_iv_basis']['implied_volatilities'][10:])
            self.assertGreater(max(abs(np.array(q['model']['local_variance_grid']['values'])[66:99]-c.variance[2])),1e-4)
            for key in ['middle_calibration','terminal_calibration']:
                self.assertLess(a[key]['maximum_verified_price_error_bound'],1e-10)
            self.assertFalse(a['clipping_applied'])
        self.assertEqual(p,before);self.assertEqual(c.payload,before)

    def test_joint_tangent_and_compensation_match_nonlinear_recalibration(self):
        c=ThreeStepTerminalSmileCalibrator(payload());T,a=c.joint_iv_tangent()
        self.assertEqual(T.shape,(66,10));self.assertLess(a['iv_response_residual'],1e-9)
        self.assertLess(a['cross_jacobian_refinement_with_tail'],1e-9)
        np.testing.assert_array_equal(T[:33,5:],0.)
        np.testing.assert_allclose(T[33:,:5],T[33:,5:]@a['terminal_compensation_quote_weights'],atol=1e-12,rtol=0)
        for direction in [np.eye(10)[2],np.array([.2,-.3,.4,.1,-.2,.3,.1,-.2,.4,-.1])]:
            errors=[]
            for h in [2e-6,1e-6]:
                rows=[]
                for sign in [-1,1]:
                    q,_=c.calibrate_joint(sign*h*direction[:5],sign*h*direction[5:]);rows.append(np.array(q['model']['local_variance_grid']['values'])[33:99])
                errors.append(max(abs((rows[1]-rows[0])/(2*h)-T@direction)))
            self.assertLess(errors[1],.4*errors[0]);self.assertLess(errors[1],2e-6)

    def test_rejected_joint_shift_cannot_return_intermediate_unfitted_state(self):
        c=ThreeStepTerminalSmileCalibrator(payload());before=deepcopy(c.payload)
        for middle,terminal in [([0]*4,[0]*5),([0]*5,[0]*4),([float('nan')]*5,[0]*5),([0]*5,[-.3]*5),([.1]*5,[0]*5)]:
            with self.assertRaises(BacktestError):c.calibrate_joint(middle,terminal)
        self.assertEqual(c.payload,before)
        with self.assertRaises(BacktestError):ThreeStepTerminalSmileCalibrator(payload(),maximum_iterations=1).calibrate_joint([.003]*5,[0]*5)

if __name__=='__main__':unittest.main()
