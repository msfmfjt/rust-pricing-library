"""Replay one decision from a sealed native panel, without provider promotion."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'backtesting')]
from check_portfolio_holdout import sealed, sha
from check_portfolio_market import sources, write
from rust_pricing_backtest import Risk, run_backtest
from rust_pricing_backtest.domain import fingerprint, require
from rust_pricing_backtest.serialization import load_input
from rust_pricing_backtest.validation import validate_document
from rust_pricing_backtest.portfolio_market_risk import check_report, implementation_identity


class SavedPanelProvider:
    """Exact request/snapshot binding; unavailable dates fail rather than reprice."""
    validation = dict(status='request_specific_native_panel_replay', diagnostic_only=True,
        scope='one audited annual snapshot; normal RustPricingProvider policy unchanged')

    def __init__(self, report, snapshot):
        self.report = deepcopy(report)
        self.snapshot_hash = fingerprint(snapshot)
        self.identity = 'sealed_native_panel:'+report['report_hash']+':'+self.snapshot_hash
        self.validation = dict(type(self).validation, estimator=report.get('market_risk_estimator','raw_native_Gamma'))
        self.requests = {cid:json.loads(raw) for cid,raw in snapshot.pricing_requests.items()}
        check_report(report, self.requests)
        if 'control_variate' in report:
            require(report['control_variate']['estimator_source_sha256']==sha(ROOT/'scripts/market_quote_control.py'),
                'precision_replay','control estimator source changed')
        require(all(a['status']=='path_gamma_chain_passed' and a['recovery_passed'] and a['validation_passed']
            for a in report['audits'].values()), 'precision_replay', 'every native gate must pass')

    def evaluate(self, snapshot, contract):
        require(fingerprint(snapshot)==self.snapshot_hash, 'precision_replay', 'saved panel cannot price another snapshot')
        cid = contract.contract_id; p = self.requests[cid]; item = self.report['instruments'][cid]
        require(p['product']['strike']==contract.strike and p['product']['expiry']==contract.expiry.isoformat()
            and p['product']['side']=={'type':contract.side} and p['product']['notional']==1.,
            'precision_replay', 'unit contract differs from saved native payoff')
        audit = self.report['audits'][cid]; buckets = tuple(item['buckets'])
        return Risk(item['price'], item['frozen_grid_delta'], buckets, snapshot.basis.basis_id,
            pre_projection=math.fsum(buckets), price_standard_error=audit['native_price']['standard_error'],
            provenance=fingerprint(dict(report=self.report['report_hash'],contract=contract,snapshot=snapshot)),
            bucket_standard_errors=tuple(item.get('bucket_standard_errors',[m['standard_error'] for m in audit['estimate']])),
            native_policy=self.report.get('market_risk_estimator','sealed_native_panel_replay'), payment_date=contract.payment_day,
            gamma=item['gamma'], reporting_iv=tuple(p['model']['reporting_iv_basis']['implied_volatilities']),
            reporting_forwards=(snapshot.spot.mid,)*len(snapshot.basis.maturity_dates))

    def joint_market_risk(self, snapshot, contracts, risks):
        require(set(risks)==set(self.requests) and all(self.evaluate(snapshot,contracts[cid])==r for cid,r in risks.items()),
            'precision_replay', 'all thirty decision risks must match the native panel')
        return deepcopy(self.report)


def run(panel_path, pilot, output):
    require(not output.exists(), 'precision_replay_output', 'fresh directory required')
    panel = sealed(panel_path)
    require(panel['status']=='passed' and panel['universe']=='full' and panel['instruments']==30,
        'precision_replay', 'passing full thirty-instrument panel required')
    native = panel['native_report']
    require(native['implementation_identity']==implementation_identity() and sources()==panel['source_sha256'],
        'precision_replay', 'native panel source differs')
    args = load_input((pilot/'backtest/input.json').read_text())
    args['config'] = replace(args['config'], solver_iterations=100_000)
    original = args['history'][0]; requests = {}
    for cid,raw in original.pricing_requests.items():
        p=json.loads(raw); p['engine']=deepcopy(native['instrument_engines'][cid]); requests[cid]=json.dumps(p,allow_nan=False)
    snapshot=replace(original,pricing_requests=requests)
    args['history']=[snapshot,*args['history'][1:]]
    provider=SavedPanelProvider(native,snapshot)
    result=run_backtest(**args,provider=provider,on_error='return',stop_after=1)
    result.write(output/'backtest')
    schema={name:validate_document(json.loads((output/'backtest'/name).read_text()))['status']
        for name in ('input.json','result.json','checkpoint.json')}
    passed=(result.status=='partial' and len(result.ledger)==len(result.decisions)==len(result.orders)==1
        and len(result.decisions[0]['candidates'])==20)
    stale_rejection=None
    if passed:
        failed=run_backtest(**args,provider=provider,on_error='return',stop_after=2,checkpoint=result.checkpoint)
        require(failed.status=='failed' and failed.failure['code']=='precision_replay'
            and failed.checkpoint==result.checkpoint and failed.orders==result.orders and failed.ledger==result.ledger,
            'precision_replay','next date must reject stale risk and roll back orders and ledger')
        stale_rejection=dict(code=failed.failure['code'],checkpoint_unchanged=True,orders_unchanged=True,ledger_unchanged=True)
    report=dict(document_kind='rqmc_precision_saved_panel_decision',schema_version=1,
        status='passed' if passed else 'failed',failure=result.failure,
        role='saved_native_panel_replay_not_normal_provider_or_annual_acceptance',
        panel_sha256=sha(panel_path),checker_sha256=sha(__file__),implementation_identity=implementation_identity(),
        processed_snapshots=len(result.ledger),orders=len(result.orders),schemas=schema,
        stale_snapshot_rejection=stale_rejection,
        held=None if not result.risks else result.risks[0]['joint_market']['held'],
        proposed=None if not result.decisions else result.decisions[0]['joint_market'],
        solver=None if not result.decisions else result.decisions[0]['solver'])
    report['report_hash']=fingerprint(report);write(output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','failure','processed_snapshots','orders')}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panel','pilot','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=run(a.panel,a.pilot,a.output);raise SystemExit(r['status']!='passed')
