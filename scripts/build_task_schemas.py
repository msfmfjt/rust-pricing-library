"""Generate self-contained task/report schemas without changing frozen contracts."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from build_specification_schema import obj, tag, ref, array
ROOT=Path(__file__).resolve().parents[1]

def slot(x): return {'oneOf':[tag('absent'),tag('present',dict(value=x))]}
def build():
    spec=json.loads((ROOT/'schemas/specification-v2/pricing_specification.schema.json').read_text())
    defs=copy.deepcopy(spec['$defs'])
    defs['specification']={k:v for k,v in spec.items() if k not in ('$defs','$schema','$id','title')}
    number={'type':'number'}; nn={'type':'number','minimum':0}; positive={'type':'number','exclusiveMinimum':0}
    rho={'type':'number','minimum':-1,'maximum':1}; u32={'type':'integer','minimum':0,'maximum':2**32-1}; u64={'type':'integer','minimum':0,'maximum':2**64-1}
    date={'type':'string','format':'date'}; matrix=array(array(number)); matrices=array(matrix)
    defs['task_operation']={'oneOf':[tag('evaluate'),tag('aad',{'gamma_relative_bump':{'type':'number','exclusiveMinimum':0,'exclusiveMaximum':1}},optional=('gamma_relative_bump',)),tag('local_variance_risk')]}
    defs['correlation_tolerances']=obj({k:nn for k in ('symmetry_abs_tol','diagonal_abs_tol','psd_abs_tol','psd_rel_tol','zero_pivot_abs_tol','zero_pivot_rel_tol')})
    defs['correlation']=obj(dict(underlying_ids=array(u32),entries=array(obj(dict(date=date,matrix=matrix))),tolerances=ref('correlation_tolerances')))
    defs['basket_lsv']=obj(dict(factor=ref('volatility_factor'),particles=ref('particles')))
    defs['basket_hw']=obj(dict(mean_reversion=nn,volatility_times=array(nn),volatilities=array(nn),rate_correlations=array(rho),lsv_targets=array(slot(ref('forward_density')))))
    defs['local_correlation']=obj(dict(basket_weights=array(positive),target=ref('local_variance_grid'),second_correlation=ref('correlation'),particles=ref('particles'),feasibility={'enum':['reject','project_and_report']},minimum_variance_span=positive,second_driver_correlations=matrices,hull_white_target=ref('forward_density')),optional=('second_driver_correlations','hull_white_target'))
    defs['dividends']=obj(dict(mean_reversion=nn,equity_linkage={'type':'number','minimum':0,'maximum':1},dividend_volatility=nn,equity_dividend_correlation=rho))
    nums=copy.deepcopy(defs['numerics']);nums['required'].append('maximum_step') if 'maximum_step' not in nums['required'] else None
    defs['task_source']={'oneOf':[
        tag('single_asset',dict(specification=ref('specification'))),
        tag('stochastic_dividend',dict(equity=ref('specification'),dividends=ref('dividends'),volatility_correlations=array(rho,0,2),rate_correlation=rho,maximum_step=positive),optional=('rate_correlation',)),
        tag('multi_asset',dict(valuation_date=date,product=ref('graph_product'),markets=array(ref('market')),models=array(ref('model')),correlation=ref('correlation'),lsv=array(slot(ref('basket_lsv'))),driver_correlations=matrices,hull_white=ref('basket_hw'),local_correlation=ref('local_correlation'),engine=ref('engine'),numerics=nums),optional=('driver_correlations','hull_white','local_correlation'))]}
    task=obj(dict(document_kind={'const':'pricing_task'},schema_version={'const':1},operation=ref('task_operation'),source=ref('task_source')))
    task.update({'$schema':'https://json-schema.org/draft/2020-12/schema','$id':'https://rust-pricing-library.dev/schemas/task-v1/pricing_task.schema.json','title':'Complete pricing task v1','$defs':defs})
    rd=json.loads((ROOT/'schemas/v3/pricing_result.schema.json').read_text())['$defs']
    unit_names=['spot','spot_squared','unit_volatility','variance','log_discount_factor','log_density','unit_parameter','one_percent_spot','one_percent_spot_squared','volatility_point','basis_point']
    rd['report_axis']=obj(dict(name={'type':'string','minLength':1},values=array(number)))
    rd['reported_risk']=obj(dict(name={'type':'string','minLength':1},underlying_ids=array(u32,0),axes=array(ref('report_axis'),0),values=array(number),standard_errors=array(nn),unit={'enum':['currency_per_'+u for u in unit_names]},method={'type':'string','minLength':1},calibration={'enum':['not_applicable','recalibrated','fixed_calibration','fixed_exercise_policy']},convention={'type':'string','minLength':1}),optional=('standard_errors',))
    report=obj(dict(document_kind={'const':'valuation_report'},schema_version={'const':1},task_fingerprint=ref('fingerprint'),plan_fingerprint=ref('fingerprint'),library_version={'type':'string','minLength':1},platform={'type':'string','minLength':1},currency_id={'type':'integer','minimum':0,'maximum':65535},value=number,standard_error=nn,estimator=ref('estimator'),independent_sampling_units={'type':'integer','minimum':1,'maximum':2**64-1},evaluated_paths={'type':'string','pattern':'^[1-9][0-9]{0,38}$'},uncertainty_scope={'enum':['pricing_sampling','conditional_on_calibration','conditional_on_exercise_policy']},calibration_seeds=array(u64,0),risks=array(ref('reported_risk'),0),vega_kt=ref('vega_kt_report')),optional=('vega_kt',))
    report.update({'$schema':'https://json-schema.org/draft/2020-12/schema','$id':'https://rust-pricing-library.dev/schemas/report-v1/valuation_report.schema.json','title':'Common valuation report v1','$defs':rd})
    return {'schemas/task-v1/pricing_task.schema.json':task,'schemas/report-v1/valuation_report.schema.json':report}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    for path,schema in build().items():
        content=json.dumps(schema,indent=2,ensure_ascii=False)+'\n';p=ROOT/path
        if args.check:
            if not p.exists() or p.read_text()!=content: raise SystemExit(f'outdated schema: {path}')
        else:
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
    print('task/report schema generation matches' if args.check else 'task/report schemas generated')
if __name__=='__main__':main()
