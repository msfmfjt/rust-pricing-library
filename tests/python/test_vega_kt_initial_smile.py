"""Initial nonflat continuation Gamma has separate refinement, tails and MC checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import rust_pricing as rp
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from check_initial_smile_gamma import evaluate,reference
from check_path_gamma_recovery import request
from test_vega_kt_selection import remove_plan_fingerprints
from test_vega_kt_nonflat_gamma import model,compile
BOUNDS=[60.,80.,90.,100.,110.,120.,140.]


class InitialSmileGammaTests(unittest.TestCase):
    def test_flat_limit_matches_black_total_variance_and_is_not_reported_as_mc(self):
        p=request('call',samples=8192,nodes=9);d=evaluate(p,BOUNDS)
        t=p['model']['local_variance_grid']['time_nodes'];v=.04*t[-1];expected=100*np.exp(-v/8)/np.sqrt(2*np.pi*v)*t[1]/2
        self.assertEqual(d['status'],'initial_variance_functional_passed')
        self.assertAlmostEqual(d['estimate'],expected,delta=d['functional_tolerance'])
        self.assertEqual(d['fine']['subdivisions'],65536)
        self.assertEqual(d['fine']['interpolation_nodes'],2*d['coarse']['interpolation_nodes']-1)
        self.assertNotIn('standard_error',d['fine'])
        self.assertEqual(d['full_market_iv_bucket_acceptance'],'not_established')
        self.assertIn('finite_log_window',d['gamma_mass_domain'])

    def test_negative_gamma_atom_at_variance_slope_jump_is_retained(self):
        p=model('call',samples=8192);g=p['model']['local_variance_grid'];x=np.array(g['log_forward_moneyness_nodes']);g['values'][9:]=(.06-.02*abs(x)).tolist()*2
        bounds=[99.999,100.,100.001];d=evaluate(p,bounds);ref=reference(p,'call',bounds,192)
        self.assertLess(d['fine']['gamma_bin_masses'][2],-.01)
        np.testing.assert_allclose(d['fine']['gamma_bin_masses'],ref['gamma_bin_masses'],atol=d['mass_tolerance'],rtol=0)
        self.assertAlmostEqual(d['fine']['recovered_variance_sensitivity'],ref['recovered_variance_sensitivity'],delta=d['functional_tolerance'])
        self.assertTrue(d['recovery_passed'])

    def test_workers_notional_and_native_price_replay(self):
        p=model('lookback');plan=compile(p);before=plan.evaluate().to_json()
        a=evaluate(p,BOUNDS);b=evaluate(p,BOUNDS,workers=4)
        self.assertEqual(remove_plan_fingerprints(a),remove_plan_fingerprints(b));self.assertEqual(before,plan.evaluate().to_json())
        p['product']['notional']=2.5;c=evaluate(p,BOUNDS)
        np.testing.assert_allclose(c['fine']['gamma_bin_masses'],2.5*np.array(a['fine']['gamma_bin_masses']),atol=1e-8,rtol=1e-7)
        self.assertAlmostEqual(c['estimate'],2.5*a['estimate'],delta=1e-7)
        self.assertAlmostEqual(c['native_initial_variance_sensitivity']['sample_variance'],6.25*a['native_initial_variance_sensitivity']['sample_variance'],places=8)

    def test_coarse_grid_and_too_strict_tolerance_withhold_estimate(self):
        p=model()
        for kwargs in [dict(subdivisions=64),dict(relative_tolerance=1e-12)]:
            d=evaluate(p,BOUNDS,**kwargs);self.assertEqual(d['status'],'recovery_failed');self.assertIsNone(d['estimate'])
            self.assertGreater(d['fine']['recovered_variance_sensitivity'],1.)
            self.assertFalse(d['recovery_passed'])

    def test_native_mc_failure_withholds_even_a_converged_gamma_estimate(self):
        d=evaluate(model('asian',samples=2,seed=4),BOUNDS)
        self.assertTrue(d['recovery_passed']);self.assertFalse(d['validation_passed'])
        self.assertEqual(d['status'],'validation_failed');self.assertIsNone(d['estimate'])
        self.assertEqual(d['independent_sampling_units'],2)
        self.assertGreater(d['fine']['recovered_variance_sensitivity'],1.)

    def test_scope_and_precision_requests_rejected(self):
        p=model()
        for kwargs in [dict(subdivisions=63),dict(subdivisions=32769),dict(relative_tolerance=0.),dict(relative_tolerance=float('nan'))]:
            with self.assertRaises(rp.PricingError):evaluate(p,BOUNDS,**kwargs)
        for b in [[],[100.,100.],[0.],[float('inf')]]:
            with self.assertRaises(rp.PricingError):evaluate(p,b)
        q=deepcopy(p);q['model']['local_variance_grid']['values'][0]+=.001
        with self.assertRaises(rp.PricingError):evaluate(q,BOUNDS)
        q=deepcopy(p);q['model']['local_variance_grid']['time_nodes'][1]+=.001
        with self.assertRaises(rp.PricingError):evaluate(q,BOUNDS)
        with self.assertRaises(rp.PricingError):compile(p).evaluate_vega_kt_initial_gamma_diagnostic(BOUNDS)

    def test_brownian_bridge_and_unpaired_sampling_leave_deterministic_gamma_unchanged(self):
        p=model('fixed_asian',samples=32768);a=evaluate(p,BOUNDS)
        p['engine']['variance_reduction'].update(antithetic=False,brownian_bridge=True);b=evaluate(p,BOUNDS)
        self.assertEqual(a['fine'],b['fine']);self.assertEqual(a['quadrature_variance_sensitivity'],b['quadrature_variance_sensitivity'])
        self.assertEqual(b['sampling_unit'],'path');self.assertTrue(b['validation_passed'])
        self.assertGreater(b['native_initial_variance_sensitivity']['standard_error'],0.)

    def test_bin_sums_and_tail_bounds_are_explicit_and_do_not_erase_signed_error(self):
        d=evaluate(model(),BOUNDS)
        for r in [d['coarse'],d['fine']]:
            self.assertAlmostEqual(sum(r['gamma_bin_masses']),r['total_gamma_mass'],places=12)
            self.assertAlmostEqual(sum(r['gamma_bin_variance_sensitivities']),r['recovered_variance_sensitivity'],places=11)
        self.assertGreater(d['price_tail_bound'],0.);self.assertLess(d['variance_tail_bound'],1e-20)
        self.assertAlmostEqual(d['fine']['recovered_variance_sensitivity']-d['native_initial_variance_sensitivity']['mean'],d['variance_difference']['mean'],places=11)
        json.dumps(d,allow_nan=False)


if __name__=='__main__':unittest.main()
