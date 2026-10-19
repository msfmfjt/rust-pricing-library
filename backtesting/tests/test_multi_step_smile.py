import json
from pathlib import Path
from copy import deepcopy
import unittest
import numpy as np
from rust_pricing_backtest.multi_step_smile import MultiStepTerminalSmileCalibrator
from rust_pricing_backtest.domain import BacktestError
ROOT=Path(__file__).resolve().parents[2]
def models():return json.loads((ROOT/'fixtures/local-vol/multi-step-gamma-v0.1.json').read_text())['models']
class MultiStepSmileTests(unittest.TestCase):
    def test_nonflat_first_market_is_preserved_under_terminal_recalibration(self):
        for case in models():
            p=case['request'];before=deepcopy(p);c=MultiStepTerminalSmileCalibrator(p)
            self.assertGreater(np.ptp(c.iv[0]),1e-4);self.assertEqual(c.indices[0],2)
            for sign in [-1,1]:
                q,a=c.calibrate(sign*.003*np.ones(5));MultiStepTerminalSmileCalibrator(q)
                self.assertLess(a['maximum_verified_price_error_bound'],1e-10);self.assertFalse(a['clipping_applied'])
                self.assertEqual(q['model']['local_variance_grid']['values'][:-66],p['model']['local_variance_grid']['values'][:-66])
                self.assertEqual(q['model']['reporting_iv_basis']['implied_volatilities'][:-5],p['model']['reporting_iv_basis']['implied_volatilities'][:-5])
            self.assertEqual(p,before);self.assertEqual(c.payload,before)
    def test_tangent_matches_nonlinear_calibration(self):
        for case in models():
            c=MultiStepTerminalSmileCalibrator(case['request']);D,a=c.terminal_iv_tangent();self.assertLess(a['iv_response_residual'],1e-9)
            for direction in [np.eye(5)[2],np.array([.3,-.2,.1,.4,-.5])]:
                vals=[];h=1e-6
                for sign in [-1,1]:
                    q,_=c.calibrate(sign*h*direction);vals.append(q['model']['local_variance_grid']['values'][-66:-33])
                np.testing.assert_allclose((np.array(vals[1])-vals[0])/(2*h),D@direction,atol=2e-6,rtol=0)
    def test_invalid_quotes_grid_and_failed_fit_are_explicit(self):
        p=models()[0]['request'];c=MultiStepTerminalSmileCalibrator(p)
        for shifts in [[0]*4,[float('nan')]*5,[-.3]*5,[0,.9,0,0,0]]:
            with self.assertRaises(BacktestError):c.calibrate(shifts)
        q=deepcopy(p);q['model']['reporting_iv_basis']['implied_volatilities'][0]+=.001
        with self.assertRaises(BacktestError):MultiStepTerminalSmileCalibrator(q)
        for row,value in [(0,1e-32),(1,1e-16)]:
            q=deepcopy(p);q['model']['local_variance_grid']['floor']=1e-40;q['model']['local_variance_grid']['values'][row*33:(row+1)*33]=[value]*33
            with self.assertRaises(BacktestError):MultiStepTerminalSmileCalibrator(q)
        with self.assertRaises(BacktestError):MultiStepTerminalSmileCalibrator(p,maximum_iterations=1).calibrate([.003]*5)
if __name__=='__main__':unittest.main()
