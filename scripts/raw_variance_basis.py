"""Versioned raw basis input for a frozen numerical market fixture."""
import json
from pathlib import Path
import numpy as np
from rust_pricing_backtest.domain import BacktestError,fingerprint,require

ORDER='transition_row_then_log_moneyness'
METRIC='euclidean_parameter_norm_without_column_normalization'


def matrix_document(payload,basis):
    grid=payload['model']['local_variance_grid']
    return dict(document_kind='variance_basis_matrix',schema_version=1,
        source_request_hash=fingerprint(payload),variance_coordinate_order=ORDER,parameter_metric=METRIC,
        time_nodes=list(grid['time_nodes']),log_forward_moneyness_nodes=list(grid['log_forward_moneyness_nodes']),
        values=np.asarray(basis,dtype=float).tolist())


def load_matrix(path,payload):
    path=Path(path)
    require(path.stat().st_size<=4*1024*1024,'basis_input_size','basis JSON must be at most 4 MiB')
    def object_pairs(pairs):
        obj={}
        for k,v in pairs:
            require(k not in obj,'basis_input_fields','duplicate JSON key: '+k);obj[k]=v
        return obj
    try:
        doc=json.loads(path.read_text(),object_pairs_hook=object_pairs,
            parse_constant=lambda value:require(False,'basis_input_values','nonfinite JSON number'))
    except (ValueError,UnicodeError) as error:
        if isinstance(error,BacktestError):raise
        raise BacktestError('basis_input_json','invalid basis JSON') from error
    expected=matrix_document(payload,[])
    require(isinstance(doc,dict) and set(doc)==set(expected),'basis_input_fields','unexpected or missing basis fields')
    require(type(doc['schema_version']) is int and doc['schema_version']==1
            and all(doc[k]==expected[k] for k in expected if k!='values'),
            'basis_input_binding','basis must bind the exact request, grid, coordinate order and parameter metric')
    rows=(len(expected['time_nodes'])-1)*len(expected['log_forward_moneyness_nodes']);values=doc['values']
    require(isinstance(values,list) and len(values)==rows and all(isinstance(r,list) for r in values),
            'basis_input_shape','one basis row per transition cell required')
    width=len(values[0])
    require(1<=width<=rows and all(len(r)==width for r in values),
            'basis_input_shape','rectangular cell-by-parameter matrix required')
    require(all(type(v) in (int,float) for r in values for v in r),'basis_input_values','numeric basis entries required')
    try:b=np.array(values,dtype=float,copy=True)
    except (ValueError,OverflowError) as error:raise BacktestError('basis_input_values','unrepresentable basis entry') from error
    require(np.isfinite(b).all(),'basis_input_values','finite basis entries required')
    b.setflags(write=False)
    return b,dict(source_request_hash=doc['source_request_hash'],basis_hash=fingerprint(b.tolist()),
        basis_shape=list(b.shape),variance_coordinate_order=ORDER,parameter_metric=METRIC,
        diagnostic_only=True,full_market_iv_bucket_acceptance='not_established')


def coupled_fixture_basis(payload):
    """Nonseparable metric in the existing coarse row-hat parameter span.

    A parameter affects its own cell and adjacent moneyness cells in the next
    transition. The coefficients vary with both time and moneyness. This is a
    declared acceptance fixture, not an automatically accepted ledger policy.
    """
    from custom_basis_cases import basis_for
    base,_=basis_for(payload,'row_hats');n=len(payload['model']['local_variance_grid']['time_nodes'])-1
    k=base.shape[1]//n;c=np.eye(base.shape[1])
    for row in range(1,n):
        for j in range(k):
            c[row*k+j,(row-1)*k+j]=.2+.1*j/(k-1)+.03*row
            if j+1<k:c[row*k+j,(row-1)*k+j+1]=-.12*(1+j/(k-1))
    return base@c
