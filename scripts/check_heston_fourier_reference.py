"""Independent continuous-time references; never imports the pricing extension.

Standard Heston: closed-form transform; lifted: adaptive complex DOP853 ODE.
Rough: high-precision fractional power series at short maturities (within its
observed convergence region). Rechecks a retained fixture, never overwrites it
unless --write is explicitly supplied during development.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import cmath
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import ndtr, gamma

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'fixtures/rough-volatility/fourier.json'
PARAMETERS={'initial_variance':0.04,'mean_reversion':0.7,'long_run_variance':0.055,'vol_of_vol':0.18,'correlation':-0.65}
LIFT={'weights':[0.6,1.1,0.8],'rates':[0.2,1.5,12.0]}

def classical_transform(z:complex,t:float,p:dict)->complex:
    if z in (0,1): return 1+0j
    v0,k,th,nu,rho=(p[n] for n in PARAMETERS)
    b=k-rho*nu*z
    d=cmath.sqrt(b*b-nu*nu*(z*z-z))
    g=(b-d)/(b+d)
    e=cmath.exp(-d*t)
    h=(b-d)/(nu*nu)*(1-e)/(1-g*e)
    a=k*th/(nu*nu)*((b-d)*t-2*cmath.log((1-g*e)/(1-g)))
    return cmath.exp(a+v0*h)

def lifted_transform(z:complex,t:float,p:dict,lift:dict,tolerance:float=2e-12)->complex:
    v0,k,th,nu,rho=(p[n] for n in PARAMETERS)
    w=np.array(lift['weights']); x=np.array(lift['rates'])
    def rhs(_,state):
        h=w@state[:-1]
        q=0.5*(z*z-z)+(rho*nu*z-k)*h+0.5*nu*nu*h*h
        return np.r_[q-x*state[:-1],v0*q+k*th*h]
    sol=solve_ivp(rhs,(0,t),np.zeros(len(w)+1,dtype=complex),method='DOP853',rtol=tolerance,atol=tolerance/100)
    if not sol.success: raise AssertionError(sol.message)
    return cmath.exp(sol.y[-1,-1])

def fractional_series(z:complex,t:float,p:dict,hurst:float,terms:int=100)->complex:
    # h(t)=sum a_n t^(alpha*n); q coefficients follow a convolution.
    with mp.workdps(80):
        alpha=mp.mpf(str(hurst))+mp.mpf('0.5')
        zz=mp.mpc(z.real,z.imag)
        v0,k,th,nu,rho=(mp.mpf(str(p[n])) for n in PARAMETERS)
        c=(zz*zz-zz)/2; b=rho*nu*zz-k; a=nu*nu/2
        h=[mp.mpc(0)]; q=[c]
        for n in range(terms):
            h.append(q[n]*mp.gamma(alpha*n+1)/mp.gamma(alpha*(n+1)+1))
            power=n+1
            q.append(b*h[power]+a*sum(h[j]*h[power-j] for j in range(1,power)))
        tt=mp.mpf(str(t))
        ih=sum(h[n]*tt**(alpha*n+1)/(alpha*n+1) for n in range(1,terms+1))
        iq=sum(q[n]*tt**(alpha*n+1)/(alpha*n+1) for n in range(terms))
        return complex(mp.exp(v0*iq+k*th*ih))

def black(f,k,variance):
    if variance==0:return max(f-k,0.)
    sd=math.sqrt(variance); d1=math.log(f/k)/sd+sd/2
    return f*ndtr(d1)-k*ndtr(d1-sd)

def price(transform,t,strike,discount=0.97,cutoff=180.):
    f=100.; variance=PARAMETERS['initial_variance']*t
    def integrand(u):
        z=0.5+1j*u
        control=cmath.exp((z*z-z)*variance/2)
        return (cmath.exp(1j*u*math.log(f/strike))*(control-transform(z))).real/(u*u+0.25)
    value,error=quad(integrand,0,cutoff,epsabs=1e-10,epsrel=1e-10,limit=400)
    assert error<1e-8,error
    return discount*(black(f,strike,variance)+math.sqrt(f*strike)/math.pi*value)

def rough_adams_transform(hurst,t,frequencies,steps):
    """Independent explicit predictor/corrector, vectorized over frequencies.

    Unlike Rust: no implicit quadratic root and no final trapezoidal integral
    of h. The terminal functional integrates interpolated q against g0 exactly.
    """
    alpha=hurst+0.5; dt=t/steps
    u=np.asarray(frequencies); z=0.5+1j*u
    v0,k,th,nu,rho=(PARAMETERS[n] for n in PARAMETERS)
    c=(z*z-z)/2; b=rho*nu*z-k; a=nu*nu/2
    q=np.empty((steps+1,len(u)),dtype=complex); q[0]=c
    r=np.arange(steps+2,dtype=float)
    predictor=dt**alpha/gamma(alpha+1)*np.diff(r**alpha)
    interior=dt**alpha/gamma(alpha+2)*np.diff(r**(alpha+1),n=2)
    newest=dt**alpha/gamma(alpha+2)
    for n in range(1,steps+1):
        hp=predictor[:n][::-1]@q[:n]
        first=newest*((n-1)**(alpha+1)-(n-1-alpha)*n**alpha)
        history=first*q[0]
        if n>1: history=history+interior[:n-1][::-1]@q[1:n]
        h=history+newest*(c+b*hp+a*hp*hp)
        q[n]=c+b*h+a*h*h
    assert np.all(np.isfinite(q)), 'Adams reference instability'
    # Integrated input curve: v0 + k*theta*(T-s)^alpha/Gamma(alpha+1).
    # Exact integration of the linear interpolant of q for this functional.
    beta=alpha+1
    r=np.arange(steps,dtype=float)
    d0=((r+1)**beta-r**beta)/beta
    d1=((r+1)**(beta+1)-r**(beta+1))/(beta+1)
    left=dt**beta/gamma(beta)*(d1-r*d0)
    right=dt**beta/gamma(beta)*((r+1)*d0-d1)
    fractional=left[::-1]@q[:-1]+right[::-1]@q[1:]
    ordinary=dt*(q.sum(axis=0)-0.5*(q[0]+q[-1]))
    return np.exp(v0*ordinary+k*th*fractional)

def rough_prices(hurst,steps,intervals=1024,cutoff=160.):
    u=np.linspace(0,cutoff,intervals+1); z=0.5+1j*u
    m=rough_adams_transform(hurst,1.,u,steps)
    black_transform=np.exp((z*z-z)*PARAMETERS['initial_variance']/2)
    w=np.ones(intervals+1); w[1:-1:2]=4; w[2:-1:2]=2
    prices=[]
    for strike in [80.,100.,120.]:
        integrand=(np.exp(1j*u*math.log(100./strike))*(black_transform-m)).real/(u*u+0.25)
        integral=(cutoff/intervals)/3*(w@integrand)
        prices.append(0.97*(black(100.,strike,PARAMETERS['initial_variance'])+math.sqrt(100.*strike)/math.pi*integral))
    return prices

@lru_cache(maxsize=1)
def compute():
    data={'parameters':PARAMETERS,'lifted':LIFT,'transforms':[],'prices':[],
          'protocol':{'transform_tolerance':3e-6,'price_tolerance':0.003,'time_steps':1024,'integration_intervals':1024,'cutoff':160.0,'rough_reference_steps':[2048,4096],'rough_reference_grid_change_cap':0.0001}}
    for family in ['classical','lifted','rough_0.1','rough_0.3']:
        t=0.1 if family.startswith('rough') else 1.0
        for damping,frequency in [(0.,0.),(1.,0.),(0.,1.),(0.5,0.75),(0.5,3.),(0.5,5.)]:
            z=complex(damping,frequency)
            if family=='classical': value=classical_transform(z,t,PARAMETERS)
            elif family=='lifted':
                value=lifted_transform(z,t,PARAMETERS,LIFT)
                assert abs(value-lifted_transform(z,t,PARAMETERS,LIFT,2e-13))<2e-11
            else:
                h=float(family.removeprefix('rough_'))
                value=fractional_series(z,t,PARAMETERS,h)
                assert abs(value-fractional_series(z,t,PARAMETERS,h,120))<1e-13
            data['transforms'].append({'family':family,'maturity':t,'damping':damping,'frequency':frequency,'real':value.real,'imag':value.imag})
    for family in ['classical','lifted']:
        t=1.0
        transform=(lambda z:classical_transform(z,t,PARAMETERS)) if family=='classical' else (lambda z:lifted_transform(z,t,PARAMETERS,LIFT))
        for strike in [80.,100.,120.]:
            value=price(transform,t,strike)
            longer=price(transform,t,strike,cutoff=240.)
            assert abs(value-longer)<2e-9,(value,longer)
            data['prices'].append({'family':family,'maturity':t,'forward':100.,'strike':strike,'discount':0.97,'call':value})
    for h in [0.1,0.3]:
        coarse=rough_prices(h,2048)
        fine=rough_prices(h,4096)
        # Separate inversion check, at the fine reference time grid.
        extended=rough_prices(h,4096,intervals=2560,cutoff=200.)
        for strike,c,f,e in zip([80.,100.,120.],coarse,fine,extended,strict=True):
            assert abs(c-f)<0.0001,(h,strike,c,f)
            assert abs(f-e)<2e-7,(h,strike,f,e)
            data['prices'].append({'family':f'rough_{h}','maturity':1.0,'forward':100.,'strike':strike,'discount':0.97,'call':e,'reference_time_change':f-c,'reference_inversion_change':e-f})
    return data

def check(data):
    expected=compute()
    for key in ['parameters','lifted','protocol']:
        assert data[key]==expected[key],key
    assert len(data['transforms'])==len(expected['transforms'])==24
    assert len(data['prices'])==len(expected['prices'])==12
    for kind,fields,tolerance in [('transforms',('real','imag'),2e-11),('prices',('call','reference_time_change','reference_inversion_change'),2e-9)]:
        for actual,reference in zip(data[kind],expected[kind],strict=True):
            for k,v in reference.items():
                if k in fields:
                    assert math.isfinite(actual[k]) and abs(actual[k]-v)<tolerance,(kind,k,actual,reference)
                else: assert actual[k]==v,(kind,k)
    print('Independent continuous-time references: 24 transforms and 12 prices passed')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--write',action='store_true'); args=parser.parse_args()
    if args.write: FIXTURE.write_text(json.dumps(compute(),indent=2)+'\n')
    else: check(json.loads(FIXTURE.read_text()))
