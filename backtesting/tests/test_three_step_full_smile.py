from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
from rust_pricing_backtest.three_step_smile import ThreeStepTerminalSmileCalibrator
from rust_pricing_backtest.domain import BacktestError
ROOT=Path(__file__).resolve().parents[2]
def model():return json.loads((ROOT/'fixtures/local-vol/three-step-gamma-v0.1.json').read_text())['request']

class AllMaturitySmileTests(unittest.TestCase):
    def test_first_IV_recalibration_holds_both_later_markets(self):
        p=model();before=deepcopy(p);c=ThreeStepTerminalSmileCalibrator(p)
        for shift in [-.003,.003]:
            q,a=c.calibrate_all_maturities(shift,[0.]*5,[0.]*5);ThreeStepTerminalSmileCalibrator(q)
            self.assertEqual(q['model']['reporting_iv_basis']['implied_volatilities'][5:],p['model']['reporting_iv_basis']['implied_volatilities'][5:])
            np.testing.assert_array_equal(np.array(q['model']['local_variance_grid']['values'])[:33],(.2+shift)**2)
            for stage in ['prefix_calibration','terminal_calibration']:self.assertLess(a[stage]['maximum_verified_price_error_bound'],1e-10)
            self.assertFalse(a['clipping_applied'])
        self.assertEqual(c.payload,before);self.assertEqual(p,before)

    def test_all_maturity_tangent_retains_ten_coordinates_and_matches_recalibration(self):
        c=ThreeStepTerminalSmileCalibrator(model());T,a=c.all_maturity_iv_tangent();L,_=c.joint_iv_tangent()
        self.assertEqual(T.shape,(67,11));np.testing.assert_array_equal(T[1:,1:],L);np.testing.assert_array_equal(T[0,1:],0.)
        self.assertLess(a['first_terminal_iv_response_residual'],1e-9);self.assertLess(a['first_response_refinement_with_tail'],1e-9)
        self.assertEqual(a['first_maturity_individual_quote_shifts'],'unsupported')
        for direction in [np.eye(11)[0],np.array([.5,.2,-.3,.4,.1,-.2,.3,.1,-.2,.4,-.1])]:
            rows=[];h=1e-6
            for sign in [-1,1]:
                q,_=c.calibrate_all_maturities(sign*h*direction[0],sign*h*direction[1:6],sign*h*direction[6:]);v=np.array(q['model']['local_variance_grid']['values']);rows.append(np.r_[v[0],v[33:99]])
            np.testing.assert_allclose((rows[1]-rows[0])/(2*h),T@direction,atol=2e-6,rtol=0)

    def test_invalid_or_unreachable_first_shift_never_returns_a_partial_fit(self):
        c=ThreeStepTerminalSmileCalibrator(model());before=deepcopy(c.payload)
        for first,middle,last in [(float('nan'),[0]*5,[0]*5),([0]*5,[0]*5,[0]*5),(-.3,[0]*5,[0]*5),(.1,[0]*5,[0]*5),(0.,[0]*4,[0]*5),(0.,[0]*5,[0]*4)]:
            with self.assertRaises(BacktestError):c.calibrate_all_maturities(first,middle,last)
        self.assertEqual(c.payload,before)
        with self.assertRaises(BacktestError):ThreeStepTerminalSmileCalibrator(model(),maximum_iterations=1).calibrate_all_maturities(.003,[0]*5,[0]*5)

if __name__=='__main__':unittest.main()
