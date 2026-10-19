import json
from pathlib import Path
from copy import deepcopy
import unittest
import numpy as np
from rust_pricing_backtest.multi_step_full_smile import MultiStepSmileCalibrator
from rust_pricing_backtest.domain import BacktestError

ROOT=Path(__file__).resolve().parents[2]
def model(n):
    return deepcopy(next(c['request'] for c in json.loads((ROOT/'fixtures/local-vol/multi-step-gamma-v0.1.json').read_text())['models'] if c['intervals']==n))

class MultiStepFullSmileTests(unittest.TestCase):
    def test_first_smile_moves_and_later_quotes_are_held_fixed(self):
        for n in [4,6]:
            p=model(n);before=deepcopy(p);c=MultiStepSmileCalibrator(p)
            shifts=np.zeros_like(c.iv);shifts[0]=[.0003,-.0002,.0004,-.0001,.0002]
            q,a=c.calibrate_all_maturities(shifts);fitted=MultiStepSmileCalibrator(q)
            np.testing.assert_allclose(fitted.iv,c.iv+shifts,atol=1e-16,rtol=0)
            self.assertEqual(p,before);self.assertFalse(a['clipping_applied'])
            for row in a['fixed_variance_rows']:np.testing.assert_array_equal(fitted.variance[row],c.variance[row])
            for row in c.calibrated_rows:self.assertGreater(np.max(abs(fitted.variance[row]-c.variance[row])),1e-6)
    def test_coupled_tangent_matches_recalibration_and_terminal_block(self):
        for n in [4,6]:
            c=MultiStepSmileCalibrator(model(n));t,a=c.all_maturity_iv_tangent();self.assertLess(a['iv_response_residual'],1e-9)
            last,_=c.terminal_iv_tangent();np.testing.assert_allclose(t[-33:,-5:],last,atol=1e-11,rtol=0)
            direction=np.linspace(-.2,.3,c.iv.size).reshape(c.iv.shape);h=1e-6;v=[]
            for sign in [-1,1]:
                q,_=c.calibrate_all_maturities(sign*h*direction)
                v.append(np.array(q['model']['local_variance_grid']['values']).reshape(c.variance.shape)[c.calibrated_rows].ravel())
            np.testing.assert_allclose((v[1]-v[0])/(2*h),t@direction.ravel(),atol=2e-6,rtol=0)
    def test_invalid_unreachable_and_nonconvergent_targets_fail_without_mutation(self):
        p=model(4);c=MultiStepSmileCalibrator(p);before=deepcopy(c.payload)
        for shifts in [[0]*5, [['bad']*5]*2, np.full_like(c.iv,np.nan),np.full_like(c.iv,-.5)]:
            with self.assertRaises(BacktestError):c.calibrate_all_maturities(shifts)
        s=np.zeros_like(c.iv);s[0,2]=.2
        with self.assertRaises(BacktestError):c.calibrate_all_maturities(s)
        with self.assertRaises(BacktestError):MultiStepSmileCalibrator(p,maximum_iterations=1).calibrate_all_maturities(np.full_like(c.iv,.003))
        self.assertEqual(c.payload,before)

if __name__=='__main__':unittest.main()
