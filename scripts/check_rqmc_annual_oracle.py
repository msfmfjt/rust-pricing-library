"""Independent recalibrated price differences for the first annual all_rows book."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import sha,sealed
from check_portfolio_market import sources,write,policy_for
from market_acceptance_cases import reference_prices
from check_multi_step_gamma_chain import independent_quote_prices
from multi_step_reference import black
from rust_pricing_backtest.domain import require,fingerprint
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.serialization import load_input
from rust_pricing_backtest.numerics import threshold_check


def run(original,output):
    require(not output.exists(),'annual_oracle_output','fresh output required')
    frozen=sources();script_hash=sha(__file__);before=time.perf_counter()
    snapshot=load_input(original.read_text())['history'][0]
    instruments={cid:json.loads(raw) for cid,raw in snapshot.pricing_requests.items()};ids=sorted(instruments)
    base=instruments[ids[0]];precision=policy_for('regression_6')
    c=MultiStepExtensionCalibrator(base,extension='all_rows');tangent,ta=c.all_maturity_iv_tangent();q=c.iv.size
    checks=[]
    def check(name,observed,limit,**ctx):checks.append(threshold_check(name,float(observed),float(limit),**ctx))
    def oracle(market):
        ps=deepcopy(instruments)
        for p in ps.values():p['model']=deepcopy(market['model']);p['market']=deepcopy(market['market'])
        return reference_prices(ps,power=precision['sobol_power'],scrambles=precision['sobol_scrambles'])
    bp=oracle(base);differences=[];calibrations=[]
    check('tangent_IV_identity',ta['iv_response_residual'],1e-9)
    for bump in precision['iv_bumps']:
        columns=[]
        for k in range(q):
            pair=[];variances=[]
            for sign in (1,-1):
                shift=np.zeros_like(c.iv);shift.flat[k]=sign*bump
                moved,a=c.calibrate_all_maturities(shift)
                targets=np.array([black(c.spot,c.strikes,t*iv**2)[0] for t,iv in zip(c.maturities,c.iv+shift)])
                check('independent_quote_fit',np.max(abs(independent_quote_prices(moved)-targets)),precision['quote_price_tolerance'],coordinate=k,bump=bump,sign=sign)
                v=np.array(moved['model']['local_variance_grid']['values']).reshape(c.variance.shape)
                check('terminal_row_unchanged',np.max(abs(v[-1]-c.variance[-1])),0)
                variances.append(v[:-1].ravel());pair.append(oracle(moved));calibrations.append(a)
            check('tangent_vs_recalibration',np.max(abs((variances[0]-variances[1])/(2*bump)-tangent[:,k])),precision['tangent_tolerance'],coordinate=k,bump=bump)
            columns.append((pair[0]-pair[1])/(2*bump))
            print(json.dumps(dict(bump=bump,coordinate=k,failed=sum(not x['passed'] for x in checks))),flush=True)
        differences.append(np.stack(columns,axis=2))
    diffs=np.array(differences)
    for j,cid in enumerate(ids):
        for k in range(q):
            d=diffs[0,:,j,k]-diffs[1,:,j,k]
            check('paired_bump_refinement',abs(d.mean()),6*d.std(ddof=1)/np.sqrt(len(d))+precision['absolute_tolerance'],instrument=cid,coordinate=k)
    require(frozen==sources() and script_hash==sha(__file__),'annual_oracle_source','source changed during calculation')
    market_hashes={cid:fingerprint({k:v for k,v in p.items() if k!='engine'}) for cid,p in instruments.items()}
    r=dict(document_kind='annual_all_rows_independent_oracle',schema_version=1,
        status='passed' if all(x['passed'] for x in checks) else 'failed',checks=checks,
        source_sha256=frozen,checker_sha256=script_hash,input_sha256=sha(original),precision=precision,
        engine_independent_request_hashes=market_hashes,instrument_order=ids,prices=bp.tolist(),
        differences=diffs.tolist(),calibration_audits=calibrations,elapsed_seconds=time.perf_counter()-before)
    r['report_hash']=fingerprint(r);write(output,r)
    print(json.dumps(dict(status=r['status'],checks=len(checks),elapsed_seconds=r['elapsed_seconds'])))
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.input,a.output);raise SystemExit(r['status']!='passed')
