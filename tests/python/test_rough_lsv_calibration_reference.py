"""Analytic controls for the independent finite-particle calibration oracle."""
import json
import unittest

import numpy as np

from rough_lsv_calibration_reference import (
    FIXTURE, case_result, conditional_moments, variance_multipliers, verify_fixture,
)


class IndependentRoughCalibration(unittest.TestCase):
    def test_all_retained_controls_and_96_quote_calibrations(self):
        verify_fixture()

    def test_brownian_limit_ignores_newest_cell_residual(self):
        fixture=json.loads(FIXTURE.read_text())
        times=np.array([0.,.03,.08,.16,.3,.45,.62,.8,1.])
        normals=np.asarray(fixture['normals'])
        z=normals.reshape(64,3,8)
        rho,eta=-.7,.8
        brownian=np.column_stack([np.zeros(64),np.cumsum(np.sqrt(np.diff(times))*(rho*z[:,0]+np.sqrt(1-rho*rho)*z[:,1]),axis=1)])
        expected=np.exp(eta*brownian/2-eta*eta*times/4)
        np.testing.assert_allclose(variance_multipliers(times,.5,eta,rho,normals),expected,rtol=2e-15,atol=2e-15)
        changed=z.copy();changed[:,2]*=1000.
        np.testing.assert_allclose(variance_multipliers(times,.5,eta,rho,changed.reshape(64,24)),expected,rtol=2e-15,atol=2e-15)

    def test_zero_vol_of_vol_moments_and_leverage_are_exact(self):
        fixture=json.loads(FIXTURE.read_text())
        case=next(c for c in fixture['cases'] if c['id']=='zero_vol_of_vol')
        result=case_result(case,fixture['normals'])
        self.assertEqual(result['squared_leverage'],case['target_variances'])
        for row in result['moments']:
            self.assertEqual([row[k] for k in ['second','third','fourth']],[1.]*3)
            self.assertEqual(row['effective_samples'],64.)
            self.assertFalse(row['extrapolated'])

    def test_compact_kernel_boundary_and_nearest_donor_tie(self):
        # Kernel weight is zero at |u|=1; each node sees only its own particle.
        moments,ess,donors=conditional_moments(np.array([-1.,0.,1.]),np.array([1.,2.,3.]),np.array([-1.,0.,1.]),1.,1.)
        np.testing.assert_array_equal(moments,np.array([[1.,1.,1.],[4.,8.,16.],[9.,27.,81.]]))
        np.testing.assert_array_equal(ess,[1.,1.,1.])
        np.testing.assert_array_equal(donors,[0,1,2])
        # Equal-distance donor ties choose the lower node. Borrow moments only;
        # the unsupported middle node retains ESS zero rather than donor ESS 2.
        moments,ess,donors=conditional_moments(np.array([-1.,-1.,1.,1.]),np.array([1.,2.,3.,4.]),np.array([-1.,0.,1.]),.2,2.)
        np.testing.assert_array_equal(donors,[0,0,2])
        np.testing.assert_array_equal(ess,[2.,0.,2.])
        np.testing.assert_array_equal(moments[1],[2.5,4.5,8.5])
        self.assertIsNone(conditional_moments(np.array([0.,0.]),np.ones(2),np.array([2.,3.]),.2,1.))
        fixture=json.loads(FIXTURE.read_text())
        case=next(c for c in fixture['cases'] if c['id']=='interior_donor_tie')
        row=case_result(case,fixture['normals'])['moments'][4]
        self.assertEqual(row['source_node'],0)
        self.assertTrue(row['extrapolated'])
        self.assertLess(row['effective_samples'],case['minimum_effective_samples'])


if __name__=='__main__':
    unittest.main()
