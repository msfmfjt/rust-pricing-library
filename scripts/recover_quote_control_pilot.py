"""Recover an output lost to ENOSPC from sealed native audits; no new Gamma claim.

This diagnostic wrapper is confined to this recovery script. Production code
never reads saved native caches. Fresh ordinary price/Delta evaluation, native
argument fingerprints, every Gamma gate and the full saved joint hash must match.
"""
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_quote_control_provider import provider
from check_portfolio_holdout import sha,sealed
from check_portfolio_market import sources,write
from rust_pricing_backtest import run_backtest,BacktestError
from rust_pricing_backtest.domain import require,fingerprint,canonical
from rust_pricing_backtest.serialization import load_input,input_json
from rust_pricing_backtest.validation import validate_document
from rust_pricing_backtest.portfolio_market_risk import implementation_identity
import rust_pricing as rp


def gzip_text(path,text):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temporary,'wt',encoding='utf-8',compresslevel=6) as stream:stream.write(text)
    require(gzip.decompress(temporary.read_bytes()).decode()==text,'recovery_storage','compressed output differs')
    temporary.replace(path)


def run(directory):
    frozen=sources();before=time.perf_counter()
    require(not (directory/'report.json').exists(),'recovery_output','fresh completion record required')
    prefill=sealed(directory/'prefill-1.json');opening=sealed(directory/'prefill-0.json')
    expected=sealed(directory/'accepted-benchmark-001-joint.json')
    package=sealed(directory.parent/'packaged/report.json')
    require(prefill['status']==opening['status']=='passed' and prefill['source_sha256']==opening['source_sha256']==frozen
            and expected['implementation_identity']==implementation_identity()
            and package['helper_sha256']==sha(ROOT/'scripts/check_quote_control_provider.py'),
            'recovery_source','unchanged original implementation and successful original native audits required')
    args=load_input((directory/'first/input.json').read_text())
    checkpoint=json.loads((directory/'first/checkpoint.json').read_text())
    require(prefill['snapshot_hash']==fingerprint(args['history'][1]),'recovery_request','wrong retained date')
    audits={a['request_fingerprint']:a for a in prefill['native_audits'].values()}
    require(len(audits)==31 and all(a['status']=='path_gamma_chain_passed' and a['recovery_passed'] and a['validation_passed'] for a in audits.values()),
            'recovery_native','all 31 original raw native gates required')
    original_plan=rp.PricingPlan;calls=[]
    class RetainedAuditPlan:
        def __init__(self,real,request):self.real=real;self.request=request
        @staticmethod
        def compile(request,**kwargs):
            require(kwargs==dict(worker_threads=4,reduction_block_size=256),'recovery_policy','execution policy changed')
            return RetainedAuditPlan(original_plan.compile(request,**kwargs),request)
        def evaluate(self):return self.real.evaluate()
        def evaluate_vega_kt_multi_step_gamma_chain_audit(self,rows,boundaries,**kwargs):
            key=self.request.fingerprint
            require(key in audits,'recovery_request','no native evidence for this exact request')
            a=audits[key]
            require(rows==[r['variance_row'] for r in a['rows']] and boundaries==a['gamma_bin_boundaries']
                    and kwargs==dict(variance_risk_directions=a['variance_risk_directions'],subdivisions=a['subdivisions'],
                        relative_tolerance=a['relative_tolerance'],gamma_relative_standard_error_limit=a['gamma_relative_standard_error_limit'],retain_samples=True),
                    'recovery_policy','native Gamma arguments differ from the retained run')
            require(self.real.request_fingerprint==key,'recovery_request','compiled request fingerprint differs')
            calls.append(key);return deepcopy(a)
    p=provider()
    with patch.object(rp,'PricingPlan',RetainedAuditPlan):
        result=run_backtest(**args,provider=p,checkpoint=checkpoint,stop_after=2,on_error='return')
    require(result.status=='partial' and len(result.ledger)==2,'recovery_replay',str(result.failure))
    require(len(calls)==len(set(calls))==31,'recovery_native','unexpected duplicate or missing audit')
    require(result.risks[1]['joint_market']['report']==expected,'recovery_equality','recomputed complete joint report is not byte-content identical')
    # The first committed state is preserved by the real checkpoint path.
    first_risks=result.risks[:1]
    require(first_risks[0]['joint_market']['report']['report_hash']==sealed(directory/'accepted-benchmark-000-joint.json')['report_hash'],
            'recovery_prefix','opening joint risk changed')
    output=directory/'resumed';output.mkdir(exist_ok=True)
    incomplete=output/'result.json'
    require(incomplete.exists() and incomplete.stat().st_size==0,'recovery_storage','unexpected partial output needs separate review')
    body=result.to_json();require(validate_document(json.loads(body))['status']=='valid','recovery_schema','result')
    gzip_text(output/'result.json.gz',body);incomplete.unlink();del body
    text=input_json(result);require(text==(directory/'first/input.json').read_text(),'recovery_input','input changed')
    os.link(directory/'first/input.json',output/'input.json');del text
    gzip_text(output/'checkpoint.json.gz',canonical(result.checkpoint)+'\n')
    with patch('rust_pricing_backtest.quote_control.evaluate_portfolio',side_effect=BacktestError('portfolio_gamma_failure','injected audit failure')):
        rejected=run_backtest(**args,provider=provider(),checkpoint=checkpoint,stop_after=2,on_error='return')
    require(rejected.status=='failed' and rejected.failure['code']=='portfolio_gamma_failure' and rejected.checkpoint==checkpoint,
            'recovery_rollback','failed native operation changed the saved checkpoint')
    schemas={}
    for part in ('first','resumed'):
        schemas[part]={}
        for name in ('input.json','result.json','checkpoint.json'):
            path=directory/part/name
            payload=path.read_text() if path.exists() else gzip.decompress(path.with_suffix(path.suffix+'.gz').read_bytes()).decode()
            schemas[part][name]=validate_document(json.loads(payload))['status']
    require(sources()==frozen,'recovery_source','source changed during recovery')
    r=dict(document_kind='market_quote_control_provider_pilot',schema_version=1,status='passed',failed=0,
        seed=100129,source_sha256=frozen,implementation_identity=implementation_identity(),
        checker_sha256=sha(ROOT/'scripts/check_quote_control_provider.py'),recovery_checker_sha256=sha(__file__),
        original_input_sha256=sha(ROOT/'benchmark-results/vega-kt-rqmc-provider/annual-pilot/backtest/input.json'),
        processed_snapshots=2,actual_contracts=30,hedge_candidates=20,annual_completed=False,
        production_provider=True,fresh_provider_resume=True,unchanged_committed_prefix=True,
        failed_native_resume_atomic=True,joint_did_not_repeat_native_audits=True,
        native_unique_requests=[len(opening['native_audits']),len(prefill['native_audits'])],
        control_requests=[row['joint_market']['report']['control_variate']['request_hash'] for row in result.risks],
        control_strikes=[row['joint_market']['report']['control_variate']['request']['product']['strike'] for row in result.risks],
        held=[row['joint_market']['held']['max_bucket_standard_error'] for row in result.risks],
        proposed=[d['joint_market']['max_bucket_standard_error'] for d in result.decisions],schemas=schemas,
        elapsed_seconds=sum(max(e['seconds'] for e in stage['events']) for stage in (opening,prefill)),
        elapsed_seconds_scope='not an end-to-end throughput measurement; original execution log retains per-contract timings',
        sampling_error_limit=10.,storage_recovery=dict(cause='ENOSPC while writing result.json after the second joint report',
            original_native_generation='fresh normal provider on both dates before the storage failure',
            serialized_continuation='reconstructed using strictly matched retained native audits; ordinary price/Delta freshly recalculated',
            additional_native_gamma_evaluations=0,reused_native_audits=31,complete_joint_report_identical=True,
            native_prefill_sha256=[sha(directory/f'prefill-{i}.json') for i in (0,1)],
            prior_failure_log_sha256=sha(directory.parent/'pilot.log'),compressed_result=True,compressed_checkpoint=True,
            input_lossless_hardlink=True,recovery_seconds=time.perf_counter()-before))
    r['report_hash']=fingerprint(r);write(directory/'report.json',r)
    print(json.dumps({k:r[k] for k in ('status','held','proposed','storage_recovery')}),flush=True)
    return r


if __name__=='__main__':run(ROOT/'benchmark-results/vega-kt-quote-control-provider-24/pilot')
