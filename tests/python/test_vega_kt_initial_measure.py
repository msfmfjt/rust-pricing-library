"""Initial path Gamma: fixed history, exact conditioning, signed measures and scope."""
import json
import math
from pathlib import Path
import sys
import unittest
import rust_pricing as rp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from check_path_gamma_recovery import request, evaluate as convolution
from check_initial_gamma_measure import evaluate, reference


class InitialMeasureTests(unittest.TestCase):
    def test_actual_payoff_ad_price_and_worker_replay(self):
        for kind in ['call','asian','fixed_asian','lookback']:
            p=request(kind,samples=512,nodes=9)
            plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=1,reduction_block_size=256)
            before=plan.evaluate();d=evaluate(p,[85.,100.,115.])
            old=convolution(p,[70.,85.,100.,115.,135.],.01)
            self.assertEqual(d['native_initial_local_vega'],old['unrecovered_initial_local_vega'])
            self.assertAlmostEqual(d['native_price']['mean'],before.value,places=12)
            self.assertAlmostEqual(d['native_price']['standard_error'],before.standard_error,places=12)
            self.assertEqual(plan.evaluate().to_json(),before.to_json())
            self.assertLess(abs(d['payoff_reconstruction_difference']['mean']),1e-12)
            self.assertLess(d['payoff_reconstruction_difference']['standard_error'],1e-12)
            counts=d['gamma_bin_nonzero_sampling_units']
            self.assertTrue(any(n>0 for n in counts))
            for n,m in zip(counts,d['gamma_bin_masses']):
                self.assertTrue(0<=n<=d['independent_sampling_units'])
                if n==0:
                    self.assertEqual(m['mean'],0.)
                    self.assertEqual(m['sample_variance'],0.)
            self.assertEqual(d['iv_bucket_acceptance'],'not_established')
            self.assertEqual(d['initial_gamma_status'],'conditional_payoff_measure')
            other=evaluate(p,[85.,100.,115.],workers=4)
            d.pop('plan_fingerprint');other.pop('plan_fingerprint');self.assertEqual(d,other)
            json.dumps(d,allow_nan=False)

    def test_binning_never_relocates_gamma_or_drops_tails(self):
        p=request('fixed_asian',samples=512)
        a=evaluate(p,[85.,100.,115.]);b=evaluate(p,[1.,2.])
        for key in ['total_gamma_mass','recovered_initial_local_vega','conditional_price','initial_local_vega_difference']:
            self.assertEqual(a[key],b[key])
        self.assertEqual(b['gamma_bin_masses'][-1],b['total_gamma_mass'])
        self.assertEqual(b['gamma_bin_first_step_vegas'][-1],b['recovered_initial_local_vega'])
        for bins,total in [('gamma_bin_masses','total_gamma_mass'),('gamma_bin_first_step_vegas','recovered_initial_local_vega')]:
            self.assertAlmostEqual(math.fsum(m['mean'] for m in a[bins]),a[total]['mean'],places=12)

    def test_notional_statistics_and_put_affine_remainder(self):
        p=request('fixed_asian',samples=512)
        a=evaluate(p,[80.,100.,120.]);p['product']['notional']=2.5;b=evaluate(p,[80.,100.,120.])
        for key in ['total_gamma_mass','recovered_initial_local_vega','initial_local_vega_difference','conditional_price','price_difference']:
            self.assertAlmostEqual(b[key]['mean'],2.5*a[key]['mean'],places=10)
            self.assertAlmostEqual(b[key]['sample_variance'],6.25*a[key]['sample_variance'],delta=1e-9*max(1.,b[key]['sample_variance']))
        p['product']['notional']=1.;p['product']['side']={'type':'put'};put=evaluate(p,[80.,100.,120.])
        self.assertAlmostEqual(put['cash']['mean'],75.5,places=12)
        self.assertAlmostEqual(put['slope']['mean'],-a['total_gamma_mass']['mean'],places=12)
        self.assertAlmostEqual(put['recovered_initial_local_vega']['mean'],a['recovered_initial_local_vega']['mean'],places=12)

    def test_one_interval_is_an_exact_dirac_at_the_boundary(self):
        p=request('call',samples=128,nodes=9)
        grid=p['model']['local_variance_grid'];grid['time_nodes']=[0.,grid['time_nodes'][-1]]
        grid['shape']=[2,9];grid['values']=[.04]*18
        p['risk']={'vega':True,'delta':False,'smile_dynamics':{'type':'sticky_log_moneyness'}}
        d=evaluate(p,[90.,100.,110.]);self.assertEqual([m['mean'] for m in d['gamma_bin_masses']],[0.,0.,1.,0.])
        self.assertEqual(d['total_gamma_mass']['sample_variance'],0.)
        self.assertEqual(d['recovered_initial_local_vega']['sample_variance'],0.)
        self.assertEqual(d['conditional_price']['sample_variance'],0.)
        t=p['model']['local_variance_grid']['time_nodes'][-1];sd=.2*math.sqrt(t)
        expected=100.*math.exp(-sd*sd/8)/math.sqrt(2*math.pi)*math.sqrt(t)
        self.assertAlmostEqual(d['recovered_initial_local_vega']['mean'],expected,places=12)

    def test_affine_history_has_zero_gamma_without_zeroing_mc_noise(self):
        p=request('fixed_asian',samples=1024);p['product']['observations'][0]['value']['fixing']=800.
        d=evaluate(p,[80.,100.,120.]);self.assertEqual(d['total_gamma_mass']['mean'],0.)
        self.assertEqual(d['recovered_initial_local_vega']['mean'],0.)
        self.assertEqual(d['cash']['mean'],100.)
        self.assertGreater(d['native_initial_local_vega']['standard_error'],0.)
        self.assertEqual(d['native_initial_local_vega']['sample_variance'],d['initial_local_vega_difference']['sample_variance'])

    def test_brownian_bridge_and_independent_paths_match_reference(self):
        p=request('asian',samples=4096)
        p['engine']['variance_reduction']={'antithetic':False,'brownian_bridge':True}
        d=evaluate(p,[85.,100.,115.]);self.assertEqual(d['sampling_unit'],'independent_path')
        g=p['model']['local_variance_grid'];t=g['time_nodes'];r=reference('asian',[85.,100.,115.],.04*t[1],.04*(t[2]-t[1]),.2,t[1])
        self.assertLess(abs(d['recovered_initial_local_vega']['mean']-r['recovered_initial_local_vega']),6*d['recovered_initial_local_vega']['standard_error']+.002)
        self.assertLess(abs(d['initial_local_vega_difference']['mean']),6*d['initial_local_vega_difference']['standard_error']+.002)

    def test_valuation_date_observation_remains_fixed_history(self):
        p=request('fixed_asian',samples=512)
        p['product']['observations'][0]={'date':p['valuation_date'],'weight':.25,'value':{'type':'unknown'}}
        a=evaluate(p,[85.,100.,115.])
        p['product']['observations'][0]['value']={'type':'known','fixing':100.}
        b=evaluate(p,[85.,100.,115.])
        for key in ['total_gamma_mass','recovered_initial_local_vega','conditional_price']:
            self.assertEqual(a[key],b[key])
        self.assertLess(abs(a['payoff_reconstruction_difference']['mean']),1e-12)

    def test_scope_and_invalid_boundaries(self):
        p=request('asian',samples=16,nodes=9)
        for bounds in [[],[0.,100.],[100.,100.],[100.,90.],[float('inf')],[float('nan')],list(range(1,131))]:
            with self.assertRaises(rp.PricingError):evaluate(p,bounds)
        cases=[]
        nonflat=json.loads(json.dumps(p));nonflat['model']['local_variance_grid']['values'][1]=.05;cases.append(nonflat)
        carry=json.loads(json.dumps(p));carry['market']['discount_curve']['discount_factors']=[1.,.98];cases.append(carry)
        rqmc=json.loads(json.dumps(p));rqmc['engine']={'type':'randomized_quasi_monte_carlo','points_per_scramble':16,'scramble_count':2,'master_scramble_seed':7,'variance_reduction':{'antithetic':True,'brownian_bridge':False}};cases.append(rqmc)
        digital=request('call',samples=16,nodes=9);expiry=digital['product']['expiry'];digital['product']={'type':'digital','underlying_id':1,'currency_id':2,'expiry':expiry,'strike':100.,'payout':1.,'side':{'type':'call'},'payout_kind':{'type':'cash'},'payment_date':expiry};digital['risk']={'vega':False,'delta':False,'smile_dynamics':{'type':'sticky_log_moneyness'}};cases.append(digital)
        for case in cases:
            with self.assertRaises(rp.PricingError):evaluate(case,[100.])


if __name__=='__main__':unittest.main()
