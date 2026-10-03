"""Independent forward-Greek references, without the pricing extension.

Delta uses a share-measure probability (Re(z)=1); Gamma uses its density,
not production's differentiated half-moment Black-control formula. Ordinary
Heston is closed form; the 3-factor lift uses DOP853. Transform definitions
reuse independent reference code, not production paths, kernels or bindings.
Cutoffs 200/300 and P1/P2 price bumps supply additional cross-checks.
Only --generate writes the fixture. Budgets are fixed before numerical runs.
"""
from __future__ import annotations
import argparse
import cmath
import json
import math
from functools import lru_cache
from pathlib import Path
from scipy.integrate import quad
from scipy.special import ndtr
from check_heston_fourier import P, heston_log, ode_log

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'fixtures/rough-volatility/fourier-greeks.json'
WEIGHTS = [0.2, 0.4, 0.5]
RATES = [0.1, 1.0, 8.0]
PROTOCOL = dict(
    delta_abs_tolerance=2e-5, gamma_abs_tolerance=2e-6,
    reference_cutoff_tolerance=2e-8, reference_bump_delta_tolerance=2e-7,
    reference_bump_gamma_tolerance=2e-7, reference_bump_sizes=[0.02,0.01],
    time_steps=1024, integration_intervals=1024, cutoff=192.0,
    refinement_delta_tolerance=5e-5, refinement_gamma_tolerance=1e-5,
    refinement_time_steps=[512,1024], refinement_cutoffs=[128.0,256.0],
    refinement_intervals=[512,1024], refinement_maturities=[0.25,1.0],
    bump_sizes=[0.04,0.02], bump_delta_tolerance=3e-6,
    bump_gamma_tolerance=1e-6,
)

@lru_cache(maxsize=40000)
def log_transform(family, z, t):
    if family == 'heston': return heston_log(z,t)
    if family == 'lift': return ode_log(z,t,WEIGHTS,RATES)
    raise ValueError(family)

def integral(fn, cutoff):
    value,error = quad(fn,0.0,cutoff,epsabs=2e-10,epsrel=2e-10,limit=200)
    if not math.isfinite(value) or error>1e-7:
        raise ValueError(('quadrature',value,error))
    return value

def reference(family,t,f,k,d,cutoff):
    x=math.log(f/k)
    def phased(u,p):
        return cmath.exp(1j*u*x + log_transform(family,p+1j*u,t))
    p1=0.5+integral(lambda u:phased(u,1).imag/u,cutoff)/math.pi
    p2=0.5+integral(lambda u:phased(u,0).imag/u,cutoff)/math.pi
    gamma=d/(math.pi*f)*integral(lambda u:phased(u,1).real,cutoff)
    return dict(call=d*(f*p1-k*p2),call_forward_delta=d*p1,
                put_forward_delta=d*(p1-1),forward_gamma=gamma)

def black(f,k,d,w):
    sd=math.sqrt(w);d1=math.log(f/k)/sd+sd/2;d2=d1-sd
    return dict(call=float(d*(f*ndtr(d1)-k*ndtr(d2))),
                call_forward_delta=float(d*ndtr(d1)),
                put_forward_delta=float(-d*ndtr(-d1)),
                forward_gamma=d*math.exp(-d1*d1/2)/(f*sd*math.sqrt(2*math.pi)))

def generate():
    rows=[]
    for family in ['heston','lift']:
        for t in [0.25,1.0]:
            for k in [80.0,100.0,120.0]:
                a=reference(family,t,100.0,k,0.97,200.0)
                b=reference(family,t,100.0,k,0.97,300.0)
                for key in b:
                    if abs(a[key]-b[key])>PROTOCOL['reference_cutoff_tolerance']:
                        raise ValueError(('cutoff refinement',family,t,k,key,a[key],b[key]))
                for h in PROTOCOL['reference_bump_sizes']:
                    up=reference(family,t,100.0+h,k,0.97,300.0)['call']
                    down=reference(family,t,100.0-h,k,0.97,300.0)['call']
                    delta=(up-down)/(2*h);gamma=(up-2*b['call']+down)/(h*h)
                    if abs(delta-b['call_forward_delta'])>PROTOCOL['reference_bump_delta_tolerance']:
                        raise ValueError(('reference delta bump',family,t,k,h,delta,b))
                    if abs(gamma-b['forward_gamma'])>PROTOCOL['reference_bump_gamma_tolerance']:
                        raise ValueError(('reference gamma bump',family,t,k,h,gamma,b))
                rows.append(dict(family=family,maturity=t,strike=k,**b))
    black_rows=[]
    for t in [0.25,1.0]:
        for k in [80.0,100.0,120.0]:
            for kind in ['constant','mean_reverting']:
                w=0.04*t if kind=='constant' else 0.055*t+(0.04-0.055)*(-math.expm1(-0.7*t))/0.7
                black_rows.append(dict(kind=kind,maturity=t,strike=k,integrated_variance=w,
                                       **black(100.0,k,0.97,w)))
    return dict(version=1,parameters=P,lift_weights=WEIGHTS,lift_rates=RATES,
                forward=100.0,discount=0.97,protocol=PROTOCOL,rows=rows,black_rows=black_rows)

def compare(saved,expected,path=''):
    if isinstance(expected,dict):
        if not isinstance(saved,dict) or set(saved)!=set(expected): raise ValueError(('keys',path))
        for k,v in expected.items():compare(saved[k],v,path+'/'+k)
    elif isinstance(expected,list):
        if not isinstance(saved,list) or len(saved)!=len(expected):raise ValueError(('length',path))
        for i,(s,e) in enumerate(zip(saved,expected)):compare(s,e,path+f'/{i}')
    elif isinstance(expected,float):
        isref=path.startswith(('/rows/','/black_rows/')) and path.rsplit('/',1)[-1] in {
            'call','call_forward_delta','put_forward_delta','forward_gamma'}
        tol=2e-8 if isref else 0.0
        if type(saved) not in (int,float) or not math.isfinite(saved) or abs(saved-expected)>tol:
            raise ValueError(('value',path,saved,expected))
    elif type(saved) is not type(expected) or saved!=expected:raise ValueError(('protocol',path))

def check():
    before=FIXTURE.read_bytes();expected=generate();compare(json.loads(before),expected)
    if before!=FIXTURE.read_bytes():raise ValueError('fixture modified')
    print(f"independent forward-Greek references passed: {len(expected['rows'])} stochastic, "
          f"{len(expected['black_rows'])} deterministic")

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true')
    if parser.parse_args().generate:
        FIXTURE.write_text(json.dumps(generate(),indent=2)+'\n');print(FIXTURE)
    else:check()
