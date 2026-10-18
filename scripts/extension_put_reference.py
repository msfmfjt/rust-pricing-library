"""Independent Put payoff oracle; integrate the final lognormal shock exactly.

Uses direct negative-normal Black probabilities, not Call-side sensitivities,
native pricing/AD or the calibration Jacobian. The declared fixture strike is 100.
"""
from copy import deepcopy
from datetime import date,timedelta
from functools import lru_cache
import numpy as np
from scipy.special import ndtr,ndtri
from scipy.stats import qmc
from multi_step_cases import with_product

PRODUCTS=('put','asian','fixed_asian','lookback','lookback_past','lookback_today')


def request(base,kind):
    p=with_product(base,'call' if kind=='put' else 'lookback' if kind.startswith('lookback') else kind)
    p['product']['side']={'type':'put'}
    if kind in ('lookback_past','lookback_today'):
        today=date.fromisoformat(p['valuation_date'])
        p['product']['monitoring_dates'].insert(0,str(today-timedelta(days=1)))
        p['product']['historical_extremum']=98.
        if kind=='lookback_today':
            p['product']['monitoring_dates'].insert(1,str(today))
            p['product']['valuation_date_fixing']=90.
    return p


def conditional_prices(history,variance):
    """Columns are declared payoffs conditional on every pre-terminal spot."""
    history=np.asarray(history);s=history[-1];n=len(history)+1
    total=np.sum(history,axis=0);minimum=np.min(history,axis=0)
    minima=[minimum,np.minimum(minimum,98.),np.minimum(minimum,90.)]
    strikes=[np.full_like(s,100.),n*100.-total,(n+1)*100.-98.-total]
    strikes += [np.minimum(m,100.) for m in minima]
    factors=[1.,1/n,1/(n+1),1.,1.,1.]
    cash=[0.,0.,0.,*[np.maximum(100.-m,0.) for m in minima]]
    values=[];sd=np.sqrt(variance)
    for strike,factor,constant in zip(strikes,factors,cash):
        valid=strike>0.;k=np.where(valid,strike,1.)
        d1=np.log(s/k)/sd+sd/2;put=k*ndtr(-d1+sd)-s*ndtr(-d1)
        values.append(factor*np.where(valid,put,0.)+constant)
    return np.stack(values,axis=-1)


@lru_cache(maxsize=4)
def _normals(n,power,scrambles):
    batches=[]
    for seed in range(7001,7001+scrambles):
        z=ndtri(qmc.Sobol(n-1,scramble=True,seed=seed).random_base2(power))
        z.setflags(write=False);batches.append(z)
    return tuple(batches)


def path_prices(payload,*,power=16,scrambles=8):
    g=payload['model']['local_variance_grid'];t=np.array(g['time_nodes']);nodes=np.array(g['log_forward_moneyness_nodes'])
    rows=np.array(g['values']).reshape(len(t),-1);n=len(t)-1;spot=payload['market']['spot'];batches=[]
    for normals in _normals(n,power,scrambles):
        x=np.zeros(len(normals));history=[]
        for row in range(n-1):
            variance=np.interp(x,nodes,rows[row])*(t[row+1]-t[row])
            x+=-.5*variance+np.sqrt(variance)*normals[:,row];history.append(spot*np.exp(x))
        variance=np.interp(x,nodes,rows[-2])*(t[-1]-t[-2])
        batches.append(conditional_prices(history,variance).mean(axis=0))
    return np.asarray(batches)
