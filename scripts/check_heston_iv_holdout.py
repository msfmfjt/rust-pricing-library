"""Independent, retained holdout targets. Never import the pricing extension.

Markov targets: adaptive DOP853 affine dynamics + P1/P2 Gauss-Legendre inversion,
checked at two quadratures; Black IV uses independent erfc/Brent inversion.
SSVI targets: direct power-law formula with theta(T)=0.04*T.
This checks synthetic interpolation/extrapolation, not future market prediction.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre
from check_heston_parameter_risk import ode, P, WEIGHTS, RATES
from check_heston_iv_calibration import implied

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT/'fixtures/rough-volatility/iv-holdout.json'
PARENT = ROOT/'fixtures/rough-volatility/iv-calibration.json'
PARENT_SHA = 'e61d8d6f1fc40484ca708ffd59b9ccbbe2127cf0f1a4b010ba2397db59a01f7c'
PROTOCOL = dict(time_steps=128, integration_intervals=512, cutoff=128.,
    max_iterations=60, max_evaluations=100, residual_tolerance=2e-6,
    interpolation_fit_tolerance=5e-4, extrapolation_fit_tolerance=5e-3,
    grid_tolerance=2.5e-5, max_stages=1,
    independent_price_agreement=2e-7, retained_iv_tolerance=2e-8,
    legendre_orders=[160,240], cutoffs=[192.,256.])


def markov_prices(t, weights, rates, order, cutoff):
    nodes, qw = roots_legendre(order)
    u=(nodes+1)*cutoff/2
    z=np.concatenate([1j*u,1+1j*u])
    lm,_=ode(z,t,P,weights,rates,tangents=False)
    m=np.exp(lm).reshape(2,order)
    return [float(.97*((100-k)/2+np.dot(qw,
        (np.exp(1j*u*math.log(100/k))*(100*m[1]-k*m[0])).imag/u)*cutoff/(2*math.pi)))
        for k in [90.,110.]]


def ssvi_iv(t,k):
    theta=.04*t; phi=.35/(theta**.5*(1+theta)**.5); rho=-.5
    y=phi*math.log(k/100)
    w=.5*theta*(1+rho*y+math.sqrt((y+rho)**2+1-rho*rho))
    return math.sqrt(w/t)


def generate():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest()!=PARENT_SHA:
        raise ValueError('parent training fixture changed')
    parent=json.loads(PARENT.read_text())
    rows=[]
    for family,w,r in [('heston',[1.],[0.]),('lift',WEIGHTS,RATES)]:
        for t in [.25,.5,.75,1.]:
            a=markov_prices(t,w,r,160,192.)
            b=markov_prices(t,w,r,240,256.)
            for j,k in enumerate([90.,110.]):
                if abs(a[j]-b[j])>PROTOCOL['independent_price_agreement']:
                    raise ValueError(('independent price agreement',family,t,k,a[j],b[j]))
                rows.append(dict(family=family,region='interpolation',maturity=t,strike=k,
                    target_volatility=implied(b[j],100.,k,.97,t)))
    for t in [.4,.6,1.,1.25]:
        for k in [87.5,95.,105.,112.5]:
            rows.append(dict(family='ssvi',region='interpolation',maturity=t,strike=k,target_volatility=ssvi_iv(t,k)))
    for t in [.25,.75,1.5]:
        for k in [92.5,107.5]:
            rows.append(dict(family='ssvi',region='interpolation',maturity=t,strike=k,target_volatility=ssvi_iv(t,k)))
    for t in [.125,2.]:
        for k in [85.,100.,115.]:
            rows.append(dict(family='ssvi',region='extrapolation',maturity=t,strike=k,target_volatility=ssvi_iv(t,k)))
    for t in [.25,.75,1.5]:
        for k in [80.,120.]:
            rows.append(dict(family='ssvi',region='extrapolation',maturity=t,strike=k,target_volatility=ssvi_iv(t,k)))
    return dict(version=1,parent_sha256=PARENT_SHA,protocol=PROTOCOL,
        starts=parent['protocol']['starts'], bounds=parent['protocol']['bounds'],
        parameters=P,lift_weights=WEIGHTS,lift_rates=RATES,forward=100.,discount=.97,
        rows=rows, row_counts={'heston':8,'lift':8,'ssvi':34})


def compare(saved,fresh,path=''):
    if isinstance(fresh,dict):
        if not isinstance(saved,dict) or set(saved)!=set(fresh):raise ValueError(('keys',path))
        for k,v in fresh.items():compare(saved[k],v,path+'/'+k)
    elif isinstance(fresh,list):
        if not isinstance(saved,list) or len(saved)!=len(fresh):raise ValueError(('length',path))
        for i,(a,b) in enumerate(zip(saved,fresh)):compare(a,b,path+f'/{i}')
    elif isinstance(fresh,float):
        tol=PROTOCOL['retained_iv_tolerance'] if path.endswith('/target_volatility') else 0.
        if type(saved) not in (float,int) or not math.isfinite(saved) or abs(saved-fresh)>tol:
            raise ValueError(('value',path,saved,fresh))
    elif type(saved) is not type(fresh) or saved!=fresh:raise ValueError(('constant',path))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    fresh=generate()
    if args.generate:FILE.write_text(json.dumps(fresh,indent=2)+'\n')
    else:compare(json.loads(FILE.read_text()),fresh)
    print('PASS: 16 independent Markov IV targets, 34 direct SSVI holdout targets; fixed protocol')
