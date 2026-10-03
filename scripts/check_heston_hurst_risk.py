"""Independent Hurst references: 60-digit fractional series and kernel integrals.

No pricing extension or production path/Riccati/kernel code is imported. Fixtures
are written only with --generate; normal execution recomputes and verifies them.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'fixtures/rough-volatility/hurst-risk.json'
P = [.04, .7, .055, .18, -.65]
HURSTS = [.01, .1, .3, .5]
PROTOCOL = dict(transform_abs_tolerance=5e-5, deterministic_price_abs_tolerance=1e-4,
                refinement_abs_tolerance=1e-4, price_bump_tolerance=5e-6,
                transform_bump_tolerance=2e-8, kernel_abs_tolerance=3e-13,
                time_steps=[512, 1024], cutoffs=[128.0, 256.0],
                integration_intervals=[512, 1024], series_orders=[100, 140])

def series(z, t, h, order):
    alpha = h+mp.mpf('.5')
    v0,k,th,nu,rho = [mp.mpf(str(v)) for v in P]
    c=(z*z-z)/2; b=rho*nu*z-k; a=nu*nu/2
    coeff=[mp.mpc(0),c/mp.gamma(1+alpha)]
    exponent=v0*c*t
    for n in range(1,order+1):
        f=b*coeff[n]+a*sum((coeff[j]*coeff[n-j] for j in range(1,n)),mp.mpc(0))
        exponent+=(v0*f+k*th*coeff[n])*t**(alpha*n+1)/(alpha*n+1)
        coeff.append(mp.gamma(1+alpha*n)/mp.gamma(1+alpha*(n+1))*f)
    return exponent

def deterministic_variance(t, h, order):
    # Integral of the Mittag-Leffler mean-reverting deterministic variance.
    v0,k,th,_,_=[mp.mpf(str(v)) for v in P]
    alpha=h+mp.mpf('.5')
    return th*t+(v0-th)*sum(((-k)**j*t**(alpha*j+1)/mp.gamma(alpha*j+2)
                            for j in range(order)),mp.mpf(0))

def kernel_reference(h, steps, lag):
    alpha=h+mp.mpf('.5');dt=1/mp.mpf(steps)
    common=dt**alpha/mp.gamma(alpha)
    def kernel(y):
        if y==0:return mp.mpf(0) # endpoint value irrelevant to the improper integral
        return common*y**(alpha-1)*(mp.log(dt)+mp.log(y)-mp.digamma(alpha))
    initial=mp.quad(lambda y: kernel(y)*(y-(lag-1)),[lag-1,lag])
    interior=initial+mp.quad(lambda y:kernel(y)*(lag+1-y),[lag,lag+1])
    endpoint=mp.quad(lambda y: kernel(y)*(1-y),[0,1])
    return [initial,interior,endpoint]

def generate():
    transforms=[];deterministic=[];kernels=[]
    with mp.workdps(60):
        for h in HURSTS:
            hh=mp.mpf(str(h))
            for t in [.25,1.0]:
                tt=mp.mpf(str(t))
                for z in [.5+.7j,2j,1+.7j]:
                    zz=mp.mpc(str(z.real),str(z.imag))
                    values=[mp.diff(lambda x:series(zz,tt,x,n),hh) for n in PROTOCOL['series_orders']]
                    if abs(values[0]-values[1])>mp.mpf('1e-25'):
                        raise ValueError(('series refinement',h,t,z,values))
                    # Independent high-precision symmetric difference, or inward
                    # second-order difference at the upper H boundary.
                    step=mp.mpf('1e-7')
                    primal=lambda x:series(zz,tt,x,140)
                    if h==.5:
                        bump=(3*primal(hh)-4*primal(hh-step)+primal(hh-2*step))/(2*step)
                    else:bump=(primal(hh+step)-primal(hh-step))/(2*step)
                    if abs(bump-values[1])>mp.mpf('2e-13'):raise ValueError('series derivative bump')
                    transforms.append(dict(hurst=h,maturity=t,exponent=[z.real,z.imag],
                                           derivative=[float(values[1].real),float(values[1].imag)]))
                w=deterministic_variance(tt,hh,140)
                dw=mp.diff(lambda x:deterministic_variance(tt,x,140),hh)
                if abs(w-deterministic_variance(tt,hh,100))>mp.mpf('1e-25'):raise ValueError('Mittag-Leffler refinement')
                for k in [80.,100.,120.]:
                    d1=mp.log(100/k)/mp.sqrt(w)+mp.sqrt(w)/2
                    d2=d1-mp.sqrt(w);cdf=lambda x:(1+mp.erf(x/mp.sqrt(2)))/2
                    call=mp.mpf('.97')*(100*cdf(d1)-k*cdf(d2))
                    dh=mp.mpf('.97')*100*mp.exp(-d1*d1/2)/mp.sqrt(2*mp.pi)/(2*mp.sqrt(w))*dw
                    deterministic.append(dict(hurst=h,maturity=t,strike=k,variance=float(w),
                                              variance_derivative=float(dw),call=float(call),derivative=float(dh)))
            for steps in [128,8192]:
                for lag in sorted(set([1,2,16,128,steps])):
                    values=kernel_reference(hh,steps,lag)
                    kernels.append(dict(hurst=h,steps=steps,lag=lag,derivatives=[float(v) for v in values]))
    return dict(version=1,parameters=P,hursts=HURSTS,forward=100.,discount=.97,protocol=PROTOCOL,
                transforms=transforms,deterministic=deterministic,kernels=kernels)

def compare(saved,fresh,path=''):
    if isinstance(fresh,dict):
        if not isinstance(saved,dict) or set(saved)!=set(fresh):raise ValueError(('keys',path))
        for k,v in fresh.items():compare(saved[k],v,path+'/'+k)
    elif isinstance(fresh,list):
        if not isinstance(saved,list) or len(saved)!=len(fresh):raise ValueError(('length',path))
        for j,(a,b) in enumerate(zip(saved,fresh)):compare(a,b,path+f'/{j}')
    elif isinstance(fresh,float):
        computed= any(x in path for x in ['/derivative','/variance']) or path.endswith('/call')
        tolerance=2e-12 if computed else 0.0
        if type(saved) not in (int,float) or not math.isfinite(saved) or abs(saved-fresh)>tolerance:
            raise ValueError(('value',path,saved,fresh))
    elif type(saved) is not type(fresh) or saved!=fresh:raise ValueError(('protocol',path))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    fresh=generate()
    if args.generate:FIXTURE.write_text(json.dumps(fresh,indent=2)+'\n')
    else:compare(json.loads(FIXTURE.read_text()),fresh)
    print('Independent Hurst references: 24 complex derivatives, 24 deterministic prices/risks, 36 kernel rows')
if __name__=='__main__':main()
