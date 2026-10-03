"""Independent Fourier/Riccati references; never import the pricing extension.

Rough CF: 70-digit fractional power series, independently summed at two orders.
Markov CF/prices: SciPy adaptive ODE and closed Heston; prices use P1/P2 inversion,
not the production half-moment/control-variate/Simpson formula.
Sources: https://arxiv.org/abs/1609.02108 and https://arxiv.org/abs/1810.04868.
Default execution checks retained fixtures; only --generate writes them.
"""
from __future__ import annotations
import argparse
import json
import cmath
from pathlib import Path
import math
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp, quad

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'fixtures/rough-volatility/fourier.json'
P = dict(initial_variance=0.04, mean_reversion=0.7, long_run_variance=0.055,
         vol_of_vol=0.18, correlation=-0.65)

def heston_log(z, t, p=P):
    v0,k,th,nu,rho = (p[x] for x in P)
    b=k-rho*nu*z
    d=cmath.sqrt(b*b-nu*nu*(z*z-z))
    g=(b-d)/(b+d)
    e=cmath.exp(-d*t)
    h=(b-d)/(nu*nu)*(1-e)/(1-g*e)
    a=k*th/(nu*nu)*((b-d)*t-2*cmath.log((1-g*e)/(1-g)))
    return v0*h+a

def ode_log(z, t, weights, rates, method='DOP853'):
    w=np.asarray(weights); r=np.asarray(rates)
    def rhs(_, y):
        psi=np.dot(w,y[:-1]); q=(z*z-z)/2+(P['correlation']*P['vol_of_vol']*z-P['mean_reversion'])*psi+P['vol_of_vol']**2/2*psi**2
        return np.append(-r*y[:-1]+q, P['initial_variance']*q+P['mean_reversion']*P['long_run_variance']*psi)
    sol=solve_ivp(rhs,(0,t),np.zeros(len(w)+1,dtype=complex),method=method,rtol=2e-11,atol=2e-13)
    if not sol.success: raise ValueError(sol.message)
    return complex(sol.y[-1,-1])

def rough_series(z, t, h, order):
    with mp.workdps(70):
        alpha=mp.mpf(str(h))+mp.mpf('0.5'); t=mp.mpf(str(t)); z=mp.mpc(z.real,z.imag)
        v0,k,th,nu,rho=(mp.mpf(str(P[x])) for x in P)
        c=(z*z-z)/2; b=rho*nu*z-k; d=nu*nu/2
        coeff=[mp.mpc(0)]
        answer=v0*c*t
        a1=c/mp.gamma(1+alpha); coeff.append(a1)
        for n in range(1,order+1):
            conv=sum((coeff[i]*coeff[n-i] for i in range(1,n)),mp.mpc(0))
            rhs=b*coeff[n]+d*conv
            answer += (v0*rhs+k*th*coeff[n])*t**(alpha*n+1)/(alpha*n+1)
            coeff.append(mp.gamma(1+alpha*n)/mp.gamma(1+alpha*(n+1))*rhs)
        return answer

def price_p1p2(log_transform, t, strike, cutoff):
    def probability(p):
        value,err=quad(lambda u: (cmath.exp(1j*u*math.log(100/strike)+log_transform(p+1j*u,t))).imag/u,
                       0,cutoff,epsabs=1e-10,epsrel=1e-10,limit=160)
        if err>1e-7: raise ValueError(('integration error',err))
        return 0.5+value/math.pi
    return 0.97*(100*probability(1)-strike*probability(0))

def generate():
    transforms=[]
    for h in [0.1,0.3]:
        for t in [0.25,1.0]:
            for z in [0.5+0.7j,2j,1+0.7j]:
                a=rough_series(z,t,h,120); b=rough_series(z,t,h,180)
                if abs(a-b)>mp.mpf('1e-30'): raise ValueError(('series not converged',h,t,z,abs(a-b)))
                transforms.append(dict(hurst=h,maturity=t,exponent=[z.real,z.imag],log_transform=[float(b.real),float(b.imag)]))
    w=[0.2,0.4,0.5]; r=[0.1,1.0,8.0]
    markov=[]
    for t in [0.25,1.0]:
        for z in [0.5+0.7j,2j,1+0.7j,0.5+32j]:
            closed=heston_log(z,t); ode=ode_log(z,t,[1.0],[0.0]); lift=ode_log(z,t,w,r); lift_bdf=ode_log(z,t,w,r,'BDF')
            if abs(closed-ode)>2e-9 or abs(lift-lift_bdf)>2e-8: raise ValueError('independent ODE check failed')
            markov.append(dict(maturity=t,exponent=[z.real,z.imag],heston_log=[closed.real,closed.imag],lift_log=[lift.real,lift.imag]))
    prices=[]
    for family,fun in [('heston',heston_log),('lift',lambda z,t:ode_log(z,t,w,r))]:
        for t in [0.25,1.0]:
            for strike in [80,100,120]:
                a=price_p1p2(fun,t,strike,200); b=price_p1p2(fun,t,strike,300)
                if abs(a-b)>2e-7: raise ValueError(('Fourier tail crosscheck',family,t,strike,a,b))
                prices.append(dict(family=family,maturity=t,strike=strike,call=b))
    return dict(version=1,parameters=P,lift_weights=w,lift_rates=r,rough_transforms=transforms,
                markov_transforms=markov,prices=prices,forward=100.0,discount=0.97,
                protocol=dict(rough_cf_abs_tolerance=5e-5,markov_cf_abs_tolerance=2e-5,
                              price_abs_tolerance=0.001,refinement_abs_tolerance=0.003,
                              mc_seed=[91,1973],mc_steps=512,mc_rough_h01_steps=1024,mc_scrambles=16,mc_points=16384,
                              mc_combined_budget=0.10,mc_se_cap=0.015))

def compare(saved, fresh, path=''):
    """Check shape and fixed inputs exactly; permit only reference rounding noise."""
    if isinstance(fresh, dict):
        if not isinstance(saved, dict) or set(saved) != set(fresh):
            raise ValueError(('keys', path))
        for key in fresh:
            compare(saved[key], fresh[key], path+'/'+key)
    elif isinstance(fresh, list):
        if not isinstance(saved, list) or len(saved) != len(fresh):
            raise ValueError(('length', path))
        for i, (x, y) in enumerate(zip(saved, fresh)):
            compare(x, y, path+f'/{i}')
    elif isinstance(fresh, float):
        # Only computed values, not model parameters, exponents or tolerances,
        # may differ by the stated platform-rounding tolerance.
        is_reference = any('/'+key+'/' in path for key in
                           ('log_transform', 'heston_log', 'lift_log')) or path.endswith('/call')
        tolerance = 2e-8 if is_reference else 0.0
        if type(saved) not in (float, int) or not math.isfinite(saved) or abs(saved-fresh) > tolerance:
            raise ValueError(('reference/input mismatch', path, saved, fresh))
    elif type(saved) is not type(fresh) or saved != fresh:
        raise ValueError(('protocol mismatch', path))


def check():
    saved = json.loads(FIXTURE.read_text())
    compare(saved, generate())
    print(f"independent Fourier references passed: {len(saved['rough_transforms'])} rough CF, "
          f"{len(saved['markov_transforms'])} Markov CF, {len(saved['prices'])} prices")

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    if args.generate:
        FIXTURE.write_text(json.dumps(generate(),indent=2)+'\n');print(FIXTURE)
    else: check()
