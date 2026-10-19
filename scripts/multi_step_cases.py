"""Declared fixtures and payoffs for variable-interval Gamma validation."""
from copy import deepcopy
from datetime import date,timedelta
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def models():return json.loads((ROOT/'fixtures/local-vol/multi-step-gamma-v0.1.json').read_text())['models']
def model(intervals=4,kind='call',samples=8192,seed=7):
    p=deepcopy(next(m['request'] for m in models() if m['intervals']==intervals));p['engine'].update(independent_sampling_units=samples,master_seed=seed)
    return with_product(p,kind)
def with_product(p,kind):
    p=deepcopy(p);count=len(p['model']['local_variance_grid']['time_nodes'])-1;start=date.fromisoformat(p['valuation_date'])
    dates=[str(start+timedelta(days=round(t*365))) for t in p['model']['local_variance_grid']['time_nodes'][1:]]
    common=dict(underlying_id=1,currency_id=2,strike=100.,notional=1.,side=dict(type='call'))
    if kind=='call':p['product']=dict(type='european_vanilla',expiry=dates[-1],**common)
    elif kind in ['asian','fixed_asian']:
        weight=1/(count+(kind=='fixed_asian'));obs=[dict(date=d,weight=weight,value=dict(type='unknown')) for d in dates]
        if kind=='fixed_asian':obs.insert(0,dict(date=str(start-timedelta(days=1)),weight=weight,value=dict(type='known',fixing=98.)))
        p['product']=dict(type='arithmetic_asian',observations=obs,payment_date=dates[-1],**common)
    elif kind=='lookback':p['product']=dict(type='fixed_lookback',monitoring_dates=dates,payment_date=dates[-1],**common)
    else:raise ValueError('unsupported test payoff')
    return p
