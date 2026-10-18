"""Expose finite-Euler smile calibration error separately from Gamma recovery.

At flat baseline, perturb market IV by epsilon*log(K/S0). The analytic Dupire
surface is rebuilt for both signs. Its discrete Euler quote-price derivative is
(1-1/N) times the continuous market quote derivative on N equal intervals.
This probe records the missing quote response as an OPEN calibration gap; a
successful diagnostic check never promotes the smile-bucket acceptance gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backtesting'))
from rust_pricing_backtest.numerics import flat_request, black_scholes_call


def rebuilt_skew_request(base, steps, epsilon, seed, samples):
    p=json.loads(flat_request(base,seed=seed,samples=samples,steps=steps,nodes=33,strike=110.,risk=False).to_json())
    g=p['model']['local_variance_grid'];sigma=base['volatility'];x=np.array(g['log_forward_moneyness_nodes'])
    implied=sigma+epsilon*x;values=[]
    # w=t*(sigma+epsilon*x)^2. The exact Dupire denominator simplifies because
    # the O(t*epsilon^2) terms cancel; its t=0 limit is defined analytically.
    for t in g['time_nodes']:
        denominator=(sigma/implied)**2-.25*t*t*implied**2*epsilon**2
        variance=implied**2/denominator
        if not np.isfinite(variance).all() or np.any(variance<=g['floor']) or np.any(variance>=g['cap']):
            raise ValueError('skew rebuild outside variance bounds')
        values.extend(variance.tolist())
    g['values']=values
    p['model']['reporting_iv_basis']['implied_volatilities']=list(implied)*steps
    return p


def run(policy):
    import rust_pricing as rp
    base=json.loads((ROOT/'fixtures/backtesting/numerical-policy-v0.1.json').read_text())
    t=base['expiry_days']/365;market=black_scholes_call(base['spot'],110.,t,base['volatility'])['vega']*math.log(110./base['spot'])
    rows=[];checks=[]
    for n in policy['time_steps']:
        values=[]
        for seed in policy['seeds']:
            prices=[]
            for sign in [1,-1]:
                p=rebuilt_skew_request(base,n,sign*policy['iv_skew_bump'],seed,policy['samples'])
                plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=1,reduction_block_size=256)
                prices.append(plan.evaluate().value)
            values.append((prices[0]-prices[1])/(2*policy['iv_skew_bump']))
        mean=float(np.mean(values));se=float(np.std(values,ddof=1)/math.sqrt(len(values)));discrete=(1-1/n)*market
        error=abs(mean-discrete);limit=policy['statistical_multiplier']*se+policy['absolute_tolerance']
        checks.append(dict(check='native_rebuilt_smile_matches_discrete_quote_tangent',steps=n,observed=error,limit=limit,passed=error<=limit))
        rows.append(dict(steps=n,native_derivative=mean,paired_batch_standard_error=se,batch_derivatives=values,
            continuous_market_quote_derivative=market,discrete_quote_derivative=discrete,
            deterministic_missing_quote_fraction=1/n,market_iv_calibration_status='not_exact_at_finite_steps'))
    module=getattr(rp,'rust_pricing',rp)
    return dict(document_kind='discrete_smile_quote_calibration_gap',schema_version=1,diagnostic_status='passed' if all(c['passed'] for c in checks) else 'failed',
        full_market_iv_bucket_acceptance='not_established',policy=policy,observations=rows,checks=checks,
        native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    r=run(json.loads(args.policy.read_text()));args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['diagnostic_status'],checks=len(r['checks']),remaining_gap='finite_step_smile_quote_recalibration')))
    return int(r['diagnostic_status']!='passed')

if __name__=='__main__':raise SystemExit(main())
