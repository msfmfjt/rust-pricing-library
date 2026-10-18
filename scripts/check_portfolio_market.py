"""Numerical acceptance across declared markets, joint risk and Delta/IV P&L."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backtesting'),str(ROOT/'scripts')]
from market_acceptance_cases import cases,requests,reference_prices
from multi_step_reference import black
from check_multi_step_gamma_chain import independent_quote_prices
from rust_pricing_backtest.domain import fingerprint, require, BacktestError
from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
from rust_pricing_backtest.named_variance_basis import named_variance_basis
from rust_pricing_backtest.portfolio_market_risk import evaluate_portfolio,sampling_covariance,explain_pnl,recalibrate_market_move
from rust_pricing_backtest.numerics import threshold_check
POLICY=ROOT/'fixtures/backtesting/portfolio-market-policy-v0.1.json'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,allow_nan=False)+'\n');temp.replace(path)


def sources():
    paths=list((ROOT/'crates').rglob('*.rs'))+list((ROOT/'backtesting/rust_pricing_backtest').rglob('*.py'))
    paths += [ROOT/'scripts'/n for n in ['check_portfolio_market.py','market_acceptance_cases.py','multi_step_cases.py',
        'multi_step_reference.py','check_custom_variance_basis.py','custom_basis_cases.py','extension_reference.py',
        'extension_put_reference.py','check_multi_step_gamma_chain.py']]
    paths += [ROOT/'fixtures/local-vol/multi-step-gamma-v0.1.json']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def policy_for(case_id):
    full=json.loads(POLICY.read_text())
    require(case_id in full['cases'],'acceptance_case','unknown case')
    effective={k:v for k,v in full.items() if k!='cases' and not k.endswith('_by_case')}
    effective['samples']=full['samples'][case_id]
    for key,value in full.items():
        if key.endswith('_by_case'):
            target=key[:-8]
            if case_id in value:effective[target]=value[case_id]
    if 'engine' in effective:effective.pop('samples')
    return effective


def run(case_id,output,*,resume=False):
    import rust_pricing as rp
    policy=policy_for(case_id)
    frozen=sources();binary=hashlib.sha256(Path(getattr(rp,'rust_pricing',rp).__file__).read_bytes()).hexdigest()
    case=next(c for c in cases() if c['id']==case_id);instruments=requests(case);ids=sorted(instruments)
    for cid,p in instruments.items():
        if 'engine' in policy:p['engine']=dict(deepcopy(policy['engine']),master_scramble_seed=policy['seeds'][0])
        else:p['engine']['independent_sampling_units']=policy['samples']
        if cid in policy.get('points_per_scramble_by_instrument',{}):
            p['engine']['points_per_scramble']=policy['points_per_scramble_by_instrument'][cid]
        if cid in policy.get('brownian_bridge_by_instrument',{}):
            p['engine']['variance_reduction']['brownian_bridge']=policy['brownian_bridge_by_instrument'][cid]
    base=instruments[ids[0]]
    basis=named_variance_basis(base,'coupled_row_hats')[0] if case_id.startswith('regression_') else None
    args=dict(extension='custom',variance_basis=basis) if basis is not None else dict(extension='all_rows')
    iv_bumps=policy['iv_bumps']
    c=MultiStepExtensionCalibrator(base,price_tolerance=policy.get('calibration_price_tolerance',1e-10),**args);tangent,ta=c.all_maturity_iv_tangent();q=c.iv.size
    protected={ROOT/n for n in frozen}|{POLICY}
    for destination in (output,output.with_suffix('.partial.json'),output.with_suffix(output.suffix+'.tmp'),output.with_suffix('.partial.json.tmp')):
        require(all(destination.resolve()!=p.resolve() and not (destination.exists() and destination.samefile(p)) for p in protected),
                'acceptance_output','output aliases a numerical input or source')
    common=dict(document_kind='portfolio_market_acceptance',schema_version=1,case=case_id,policy=policy,
        source_sha256=frozen,native_binary_sha256=binary,request_hashes={cid:fingerprint(p) for cid,p in instruments.items()})
    r=dict(common,status='in_progress',checks=[],phase='prices',observations=[],calibration_audits=[],pnl=[])
    if resume:
        r=json.loads(output.with_suffix('.partial.json').read_text())
        digest=r.pop('report_hash');require(digest==fingerprint(r),'acceptance_resume','corrupted saved report')
        require(all(r.get(k)==v for k,v in common.items()) and r['phase'] in ('native','complete'),
                'acceptance_resume','changed inputs/sources or incomplete independent prices')
    def check(name,value,limit,**ctx):r['checks'].append(threshold_check(name,float(value),float(limit),**ctx))
    def save():
        require(sources()==frozen and policy_for(case_id)==policy,'acceptance_source_changed','sources or effective case policy changed during validation')
        body={k:v for k,v in r.items() if k!='report_hash'};body['report_hash']=fingerprint(body)
        write(output.with_suffix('.partial.json'),body)
        return body
    def oracle(market):
        ps=deepcopy(instruments)
        for p in ps.values():p['model']=deepcopy(market['model']);p['market']=deepcopy(market['market'])
        return reference_prices(ps,power=policy['sobol_power'],scrambles=policy['sobol_scrambles'])
    if not resume:
        check('tangent_IV_identity',ta['iv_response_residual'],1e-9)
        bp=oracle(base);derivatives=[]
        for bump in iv_bumps:
            columns=[]
            for coordinate in range(q):
                prices=[];vs=[]
                for sign in (1,-1):
                    shift=np.zeros(q);shift[coordinate]=sign*bump
                    moved,a=c.calibrate_all_maturities(shift.reshape(c.iv.shape))
                    targets=np.array([black(c.spot,c.strikes,t*iv**2)[0] for t,iv in zip(c.maturities,c.iv+shift.reshape(c.iv.shape))])
                    check('independent_quote_fit',np.max(abs(independent_quote_prices(moved)-targets)),policy['quote_price_tolerance'],coordinate=coordinate,bump=bump,sign=sign)
                    variance=np.array(moved['model']['local_variance_grid']['values']).reshape(c.variance.shape)
                    check('unused_terminal_row_fixed',np.max(abs(variance[-1]-c.variance[-1])),0)
                    vs.append(variance[:-1].ravel());prices.append(oracle(moved));r['calibration_audits'].append(a)
                columns.append((prices[0]-prices[1])/(2*bump))
                check('tangent_vs_recalibration',np.max(abs((vs[0]-vs[1])/(2*bump)-tangent[:,coordinate])),policy['tangent_tolerance'])
            derivatives.append(np.stack(columns,axis=2))
            print(json.dumps(dict(case=case_id,phase='independent_prices',bump=bump,failed=sum(not x['passed'] for x in r['checks']))),flush=True)
        diffs=np.array(derivatives)
        for j,cid in enumerate(ids):
            for k in range(q):
                d=diffs[0,:,j,k]-diffs[1,:,j,k]
                check('paired_bump_refinement',abs(d.mean()),6*d.std(ddof=1)/np.sqrt(len(d))+policy['absolute_tolerance'],instrument=cid,coordinate=k)
        for sign in (1,-1):
            shift=np.zeros_like(c.iv);shift[0]=sign*policy['stress_iv_bump'];moved,a=c.calibrate_all_maturities(shift)
            target=np.array([black(c.spot,c.strikes,t*iv*iv)[0] for t,iv in zip(c.maturities,c.iv+shift)])
            check('finite_stress_quote_fit',np.max(abs(independent_quote_prices(moved)-target)),policy['quote_price_tolerance'])
            r['calibration_audits'].append(a)
        r['price_record']=dict(instrument_order=ids,prices=bp.tolist(),differences=diffs.tolist())
        r['phase']='native';save()
    bp=np.asarray(r['price_record']['prices']);diffs=np.asarray(r['price_record']['differences'])
    means=diffs.mean(axis=1);se=diffs.std(axis=1,ddof=1)/np.sqrt(policy['sobol_scrambles'])
    weights={cid:float(((-1)**j)*(j+1)/10) for j,cid in enumerate(ids)}
    partitions=({cid:policy['subdivisions_by_instrument'].get(cid,policy['subdivisions']) for cid in ids}
        if 'subdivisions_by_instrument' in policy else policy['subdivisions'])
    for seed in policy['seeds']:
        if seed in [o['seed'] for o in r['observations']]:continue
        for p in instruments.values():p['engine']['master_seed' if p['engine']['type']=='pseudo_monte_carlo' else 'master_scramble_seed']=seed
        joint=evaluate_portfolio(instruments,weights,**args,worker_threads=4,retain_samples=True,
            subdivisions=partitions,relative_tolerance=policy['relative_tolerance'],
            gamma_relative_standard_error_limit=policy['gamma_relative_standard_error_limit'])
        samples=np.asarray(joint.pop('sampling_unit_values'));joint.pop('report_hash')
        joint['report_hash']=fingerprint(joint)
        cov=np.asarray(joint['market_sample_covariance']);n=joint['independent_sampling_units']
        check('joint_covariance_PSD',max(0.,-np.linalg.eigvalsh(cov).min()),1e-8)
        direct=np.einsum('niq,i->nq',samples,np.array([weights[cid] for cid in ids]))
        check('portfolio_covariance_vs_direct_samples',np.max(abs(sampling_covariance(direct)/n-np.array(joint['net_price_and_market_mean_covariance']))),1e-10)
        transform=np.kron(np.array([[weights[cid] for cid in ids]]),np.eye(q))
        check('joint_projection_vs_direct_samples',np.max(abs(transform@cov@transform.T/n-np.array(joint['net_price_and_market_mean_covariance'])[1:,1:])),1e-10)
        check('off_diagonal_instrument_covariance_present',not np.any(abs(cov[:q,q:2*q])>1e-6),0)
        for j,cid in enumerate(ids):
            a=joint['audits'][cid]
            check('price_vs_independent',abs(a['native_price']['mean']-bp[:,j].mean()),6*np.hypot(a['native_price']['standard_error'],bp[:,j].std(ddof=1)/np.sqrt(len(bp)))+policy['absolute_tolerance'],instrument=cid,seed=seed)
            for k,m in enumerate(a['estimate']):
                for b,bump in enumerate(iv_bumps):
                    check('Gamma_vs_independent_price_difference',abs(m['mean']-means[b,j,k]),6*np.hypot(m['standard_error'],se[b,j,k])+policy['absolute_tolerance'],instrument=cid,seed=seed,coordinate=k,bump=bump)
        if seed==policy['seeds'][0]:
            w=np.array([weights[cid] for cid in ids]);opening=bp@w
            # Signed smile twist plus simultaneous spot move, and each alone.
            for mode in ('spot','iv','combined'):
                for scale in (1.,.5):
                    ds=scale*policy['pnl_spot_shock'] if mode!='iv' else 0.
                    shift=scale*policy['pnl_iv_shock']*np.linspace(-.5,1.,q).reshape(c.iv.shape) if mode!='spot' else np.zeros_like(c.iv)
                    moved,a=recalibrate_market_move(base,c.spot+ds,c.iv+shift,**args)
                    prices=oracle(moved)@w;actual=prices-opening
                    explanation=explain_pnl(joint,instruments,current_spot=c.spot+ds,current_iv=c.iv+shift,actual_asset_pnl=float(actual.mean()))
                    sigma=np.hypot(explanation['vega_and_delta_correction_standard_error'],actual.std(ddof=1)/np.sqrt(len(actual)))
                    check('Pnl_vs_independent_revaluation',abs(explanation['unexplained_pnl']),6*sigma+policy['absolute_tolerance'],mode=mode,scale=scale)
                    check('Pnl_reconciliation',abs(explanation['available_greek_pnl']+explanation['unexplained_pnl']-actual.mean()),1e-12,mode=mode,scale=scale)
                    r['pnl'].append(dict(mode=mode,scale=scale,explanation=explanation,recalibration=a,independent_pnl_batches=actual.tolist()))
        r['observations'].append(dict(seed=seed,report=joint));save()
        print(json.dumps(dict(case=case_id,phase='native',seed=seed,panels=len(ids),failed=sum(not x['passed'] for x in r['checks']))),flush=True)
    require([o['seed'] for o in r['observations']]==policy['seeds'] and len(r['pnl'])==6,'acceptance_coverage','incomplete native/P&L coverage')
    r.update(phase='complete',status='passed' if all(x['passed'] for x in r['checks']) else 'failed')
    result=save();write(output,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    try:r=run(a.case,a.output,resume=a.resume)
    except (BacktestError,ValueError) as error:
        print(str(error),file=sys.stderr);return 1
    print(json.dumps(dict(status=r['status'],checks=len(r['checks']),failed=sum(not x['passed'] for x in r['checks']))),flush=True)
    return int(r['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
