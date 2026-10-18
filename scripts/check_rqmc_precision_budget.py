"""Check the unchanged native work cap for the exact six-step European book."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import sealed,sha
from check_portfolio_market import write
from rust_pricing_backtest.serialization import load_input
from rust_pricing_backtest.domain import require,fingerprint
from rust_pricing_backtest.multi_step_provider import gamma_boundaries
from rust_pricing_backtest.portfolio_market_risk import implementation_identity


def run(output):
    import rust_pricing as rp
    require(not output.exists(),'precision_budget_output','fresh output required')
    prior=ROOT/'benchmark-results/vega-kt-rqmc-provider'
    reference=sealed(prior/'held-512-probe.json')['native_report']
    require(reference['implementation_identity']==implementation_identity(),'precision_budget_source','same implementation required')
    args=load_input((prior/'annual-pilot/backtest/input.json').read_text())
    p=json.loads(args['history'][0].pricing_requests['target-00'])
    require(p['product']['type']=='european_vanilla' and len(p['model']['local_variance_grid']['time_nodes'])==7,
        'precision_budget_scope','exact six-interval European required')
    bounds=gamma_boundaries(p,p['market']['spot'],put_knots=True)
    require(len(bounds)==5,'precision_budget_scope','six Gamma bins required')
    p['engine']['points_per_scramble']=1024
    p['risk'].pop('vega_kt',None);p['risk']['vega']=False
    plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),worker_threads=4,reduction_block_size=256)
    rejection=None
    try:
        plan.evaluate_vega_kt_multi_step_gamma_chain_audit(list(range(6)),bounds,
            variance_risk_directions=np.array(reference['calibration_tangent']).T.reshape(15,6,33).tolist(),
            subdivisions=512,relative_tolerance=1e-6,gamma_relative_standard_error_limit=.25)
    except rp.PricingError as error:
        rejection=str(error)
    require(rejection is not None and 'sample, storage or continuation budget exceeded' in rejection,
        'precision_budget_rejection','unchanged native cap must reject 1024 x 16')
    # The European source graph has seven reachable nodes, all retained by the
    # compiler. This is a workload-specific estimate, not a general budget API.
    entries=[]
    for points,count in ((256,16),(512,16),(1024,12),(1024,16)):
        work=points*count*2*sum(range(1,7))*4*(512+33+6+3)*7
        entries.append(dict(points_per_scramble=points,scramble_count=count,
            continuation_work=work,within_continuation_cap=work<=10_000_000_000))
    dependencies=[Path(__file__),ROOT/'crates/pricing/src/engine/payoff/compile.rs',
        ROOT/'crates/pricing/src/engine/risk/vegakt_path_chain.rs',prior/'held-512-probe.json',
        prior/'annual-pilot/backtest/input.json']
    r=dict(document_kind='rqmc_precision_work_budget',schema_version=1,status='passed',
        implementation_identity=implementation_identity(),limit=10_000_000_000,
        native_rejected_design=dict(points_per_scramble=1024,scramble_count=16,error=rejection),
        workload_estimates=entries,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in dependencies},
        scope='workload-specific continuation cost; storage and conditional AD budgets remain native checks')
    r['report_hash']=fingerprint(r);write(output,r);print(json.dumps(entries))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
