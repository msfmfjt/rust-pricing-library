"""Declared broader market cases and an independent terminal-conditioned oracle."""
from copy import deepcopy
from datetime import date, timedelta
from functools import lru_cache
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
from multi_step_cases import models
from check_custom_variance_basis import PRODUCTS, product_request
from check_multi_step_gamma_chain import independent_quote_prices
from multi_step_reference import black

SPECS = (
    dict(id='short_3_left_skew', days=[0,7,21,45], nodes=17, width=.6, quotes=[-.1,0.,.1], reporting=[2,3], level=.045, skew=-.15, curve=.35, strike=95.),
    dict(id='uneven_3_right_skew', days=[0,21,85,180], nodes=25, width=.8, quotes=[-.18,-.08,0.,.08,.18], reporting=[2,3], level=.055, skew=.16, curve=.2, strike=105.),
    dict(id='uneven_5_smile', days=[0,30,80,150,240,365], nodes=33, width=1.1, quotes=[-.2,-.08,0.,.08,.2], reporting=[2,5], level=.065, skew=-.10, curve=.3, strike=110.),
    dict(id='long_8_smile', days=[0,30,90,180,270,365,450,600,730], nodes=33, width=1.2, quotes=[-.15,0.,.15], reporting=[2,4,6,8], level=.05, skew=.08, curve=.25, strike=100.),
)


def cases():
    out=[]
    for m in models():
        out.append(dict(id='regression_'+str(m['intervals']), request=deepcopy(m['request']), strike=100.))
    for spec in SPECS:
        p=deepcopy(models()[0]['request']); t=np.array(spec['days'])/365; x=np.linspace(-spec['width'],spec['width'],spec['nodes'])
        rows=np.array([spec['level']*(1+.12*i/(len(t)-1))*(1+spec['skew']*x+spec['curve']*x*x) for i in range(len(t))])
        g=p['model']['local_variance_grid'];g.update(time_nodes=t.tolist(),log_forward_moneyness_nodes=x.tolist(),
            values=rows.ravel().tolist(),shape=list(rows.shape))
        b=p['model']['reporting_iv_basis'];b.update(maturity_nodes=t[spec['reporting']].tolist(),log_forward_moneyness_nodes=spec['quotes'],
            implied_volatilities=[.2]*(len(spec['reporting'])*len(spec['quotes'])),shape=[len(spec['reporting']),len(spec['quotes'])])
        start=date.fromisoformat(p['valuation_date']);p['product']['expiry']=(start+timedelta(days=spec['days'][-1])).isoformat()
        # Price inversion uses an independent dense density integral, not the calibrator Jacobian.
        target=independent_quote_prices(p);strikes=100*np.exp(spec['quotes']);ivs=[]
        for maturity,values in zip(b['maturity_nodes'],target):
            lo=np.full(len(strikes),1e-6);hi=np.full(len(strikes),4.)
            for _ in range(64):
                mid=(lo+hi)/2;less=black(100,strikes,maturity*mid*mid)[0]<values
                lo=np.where(less,mid,lo);hi=np.where(less,hi,mid)
            ivs.extend(((lo+hi)/2).tolist())
        b['implied_volatilities']=ivs
        p['risk'].pop('vega_kt',None);p['risk']['vega']=False
        out.append(dict(id=spec['id'],request=p,strike=spec['strike']))
    return out


def requests(case):
    result={}
    for j,name in enumerate(PRODUCTS):
        p=product_request(case['request'],j);p['product']['strike']=case['strike']
        if not case['id'].startswith('regression_') and p['product']['type']=='arithmetic_asian':
            obs=p['product']['observations'];w=np.arange(1,len(obs)+1,dtype=float);w/=w.sum()
            for observation,weight in zip(obs,w):observation['weight']=float(weight)
        result[name]=p
    return result


@lru_cache(maxsize=8)
def normals(n,power,scrambles):
    return tuple(ndtri(qmc.Sobol(n-1,scramble=True,seed=seed).random_base2(power)) for seed in range(7001,7001+scrambles))


def reference_prices(instruments, *, power=16, scrambles=8):
    """No native pricing, adjoints or calibration derivatives enter the oracle."""
    ids=sorted(instruments);p=instruments[ids[0]];g=p['model']['local_variance_grid']
    t=np.asarray(g['time_nodes']);nodes=np.asarray(g['log_forward_moneyness_nodes']);v=np.asarray(g['values']).reshape(len(t),-1)
    n=len(t)-1;spot=p['market']['spot'];start=date.fromisoformat(p['valuation_date'])
    dates=[(start+timedelta(days=round(day*365))).isoformat() for day in t]
    out=[]
    for z in normals(n,power,scrambles):
        x=np.zeros(len(z));history={dates[0]:np.full(len(z),spot)}
        for row in range(n-1):
            variance=np.interp(x,nodes,v[row])*(t[row+1]-t[row]);x+=-variance/2+np.sqrt(variance)*z[:,row]
            history[dates[row+1]]=spot*np.exp(x)
        s=spot*np.exp(x);variance=np.interp(x,nodes,v[n-1])*(t[n]-t[n-1]);sd=np.sqrt(variance)
        values=[]
        for cid in ids:
            product=instruments[cid]['product'];kind=product['type'];strike=product['strike'];put=product['side']['type']=='put';cash=0.;weight=1.
            if kind=='arithmetic_asian':
                partial=np.zeros(len(s));weight=0.
                for observation in product['observations']:
                    day=observation['date'];w=observation['weight'];value=observation['value']
                    if value['type']=='known':partial+=w*value['fixing']
                    elif day==dates[-1]:weight+=w
                    else:partial+=w*history[day]
                assert weight>0
                strike=(strike-partial)/weight
            elif kind=='fixed_lookback':
                rows=[history[d] for d in product['monitoring_dates'] if d in history and d!=dates[0]]
                if 'historical_extremum' in product:rows.append(np.full(len(s),product['historical_extremum']))
                if dates[0] in product['monitoring_dates']:rows.append(np.full(len(s),product['valuation_date_fixing']))
                extremum=np.min(rows,axis=0) if put else np.max(rows,axis=0)
                cash=np.maximum(strike-extremum,0) if put else np.maximum(extremum-strike,0)
                strike=np.minimum(strike,extremum) if put else np.maximum(strike,extremum)
            else:assert kind=='european_vanilla'
            k=np.maximum(strike,1e-50);d1=np.log(s/k)/sd+sd/2
            option=k*ndtr(-d1+sd)-s*ndtr(-d1) if put else s*ndtr(d1)-k*ndtr(d1-sd)
            option=np.where(np.asarray(strike)>0,option,0. if put else s-strike)
            values.append(float(np.mean((weight*option+cash)*product['notional'])))
        out.append(values)
    return np.asarray(out)
