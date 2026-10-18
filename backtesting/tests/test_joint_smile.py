from copy import deepcopy
import unittest
import numpy as np
from rust_pricing_backtest.discrete_smile import TwoStepSmileCalibrator
from rust_pricing_backtest.domain import BacktestError
from test_discrete_smile import payload

class JointSmileTests(unittest.TestCase):
    def model(self):
        c=TwoStepSmileCalibrator(payload(),price_tolerance=1e-12)
        return c.calibrate(.015*c.quote_nodes+.02*c.quote_nodes**2)[0]

    def test_first_move_preserves_terminal_quotes_and_source_context(self):
        p=self.model();before=deepcopy(p);c=TwoStepSmileCalibrator(p);old=c.calibrate([0.]*5)
        for shift in [-.003,.003]:
            q,a=c.calibrate_joint(shift,[0.]*5)
            np.testing.assert_array_equal(q['model']['reporting_iv_basis']['implied_volatilities'][5:],p['model']['reporting_iv_basis']['implied_volatilities'][5:])
            np.testing.assert_allclose(q['model']['local_variance_grid']['values'][:33],(.2+shift)**2,atol=1e-16)
            self.assertLessEqual(a['maximum_verified_price_error_bound'],1e-10)
            self.assertEqual(a['first_marginal'],'shifted_flat_lognormal');TwoStepSmileCalibrator(q)
        self.assertEqual(p,before);self.assertEqual(c.calibrate([0.]*5),old)

    def test_joint_tangent_matches_independent_calibration_differences(self):
        c=TwoStepSmileCalibrator(self.model(),price_tolerance=1e-12);u,a=c.joint_iv_tangent()
        self.assertEqual(u.shape,(34,6));self.assertLess(a['maximum_joint_iv_response_residual'],1e-10)
        self.assertEqual(a['coordinate_order'][0],'first_maturity_parallel_iv')
        for direction in [np.eye(6)[0],np.array([.4,.1,-.2,.3,-.2,.1])]:
            errors=[]
            for bump in [2e-6,1e-6]:
                rows=[]
                for sign in [1,-1]:
                    q,_=c.calibrate_joint(sign*bump*direction[0],sign*bump*direction[1:]);v=q['model']['local_variance_grid']['values'];rows.append(np.r_[v[0],v[33:66]])
                errors.append(np.max(abs((rows[0]-rows[1])/(2*bump)-u@direction)))
            self.assertLess(errors[-1],2e-6)
            # At tiny bumps the nonlinear residual / bump roundoff floor can
            # dominate the O(h²) term; do not demand monotonicity below it.
            self.assertLessEqual(errors[-1],max(.5*errors[0],1e-7))

    def test_first_column_is_combination_of_independent_terminal_basis(self):
        c=TwoStepSmileCalibrator(self.model());u,a=c.joint_iv_tangent();old,_=c.terminal_iv_tangent()
        np.testing.assert_array_equal(u[1:,1:],old)
        np.testing.assert_allclose(u[1:,0],old@a['first_recalibration_terminal_basis_weights'],atol=1e-15)
        self.assertTrue(np.all(u[0,1:]==0));self.assertAlmostEqual(u[0,0],.4)

    def test_joint_shifts_reject_nonfinite_unreachable_and_nonconvergent_requests(self):
        c=TwoStepSmileCalibrator(self.model())
        for shift in [float('nan'),float('inf'),-.21,[.001,.002],100.]:
            with self.assertRaises(BacktestError):c.calibrate_joint(shift,[0.]*5)
        with self.assertRaisesRegex(BacktestError,'fixed first marginal'):c.calibrate_joint(.1,[0.]*5)
        with self.assertRaises(BacktestError):c.calibrate_joint(0.,[float('nan')]*5)
        with self.assertRaises(BacktestError):TwoStepSmileCalibrator(self.model(),maximum_iterations=1).calibrate_joint(.003,[.001]*5)

if __name__=='__main__':unittest.main()
