"""Keep reference-comparison roundoff allowance out of calculated quantities."""
from copy import deepcopy
import math
import unittest
from check_extension_provider import compare_audit


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.audit=dict(gamma_bin_boundaries=[80.,90.,100.,110.,120.],request_fingerprint='request',
            plan_fingerprint='plan',native_price=dict(mean=7.,standard_error=.01),
            joint_sample_covariance=[1.,.5,.5,2.],estimate=[dict(mean=.123,standard_error=.004)])

    def test_identity_changes_leave_calculations_exact(self):
        a=deepcopy(self.audit);a.update(request_fingerprint='ordinary-request',plan_fingerprint='ordinary-plan')
        result=compare_audit(a,self.audit)
        self.assertTrue(result['numerical_audit_exact']);self.assertEqual(result['gamma_bin_maximum_ulp_difference'],0.)

    def test_only_one_ulp_of_bin_label_roundoff_is_accepted(self):
        a=deepcopy(self.audit);a['gamma_bin_boundaries'][3]=100.*1.1
        result=compare_audit(a,self.audit)
        self.assertTrue(result['numerical_audit_exact']);self.assertTrue(result['gamma_bin_roundoff_passed'])
        self.assertEqual(result['gamma_bin_maximum_ulp_difference'],1.)
        a['gamma_bin_boundaries'][3]=math.nextafter(a['gamma_bin_boundaries'][3],math.inf)
        self.assertFalse(compare_audit(a,self.audit)['gamma_bin_roundoff_passed'])

    def test_even_one_ulp_in_price_risk_or_covariance_is_rejected(self):
        for key,index in [('native_price','mean'),('joint_sample_covariance',0),('estimate',0)]:
            a=deepcopy(self.audit)
            if key=='estimate':a[key][index]['mean']=math.nextafter(a[key][index]['mean'],math.inf)
            else:a[key][index]=math.nextafter(a[key][index],math.inf)
            with self.subTest(key=key):self.assertFalse(compare_audit(a,self.audit)['numerical_audit_exact'])

    def test_nonfinite_and_missing_bins_are_rejected(self):
        for bins in [[80.], [80.,90.,100.,float('nan'),120.]]:
            a=deepcopy(self.audit);a['gamma_bin_boundaries']=bins
            with self.assertRaises(ValueError):compare_audit(a,self.audit)


if __name__=='__main__':unittest.main()
