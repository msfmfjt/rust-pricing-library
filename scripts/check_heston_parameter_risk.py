"""Independent continuous-time parameter references; no pricing extension imports.

Adaptive DOP853 for Markov affine models; P1/P2 Legendre inversion (two contours),
not production half-moment Simpson/control derivatives. Rough transforms use a
60-digit fractional series. Only --generate writes fixtures.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'fixtures/rough-volatility/parameter-risk.json'
NAMES = ['initial_variance', 'mean_reversion', 'long_run_variance', 'vol_of_vol', 'correlation']
P = [0.04, 0.7, 0.055, 0.18, -0.65]
WEIGHTS, RATES = [0.2, 0.4, 0.5], [0.1, 1.0, 8.0]
PROTOCOL = dict(price_abs_tolerance=2e-4, transform_abs_tolerance=5e-4,
                refinement_abs_tolerance=5e-4, independent_bump_tolerance=2e-5,
                independent_quadrature_tolerance=2e-7,
                time_steps=[512, 1024], integration_intervals=[512, 1024],
                cutoffs=[128.0, 256.0], bump_sizes=[1e-5, 5e-6])

def ode(z, t, p, weights, rates, tangents=True):
    z = np.asarray(z, dtype=complex)
    v0, k, th, nu, rho = p
    nf, nz = len(weights), len(z)
    w = np.asarray(weights)[:, None]
    r = np.asarray(rates)[:, None]
    groups = 6 if tangents else 1
    def rhs(_, flat):
        y = flat.reshape(groups, nf+1, nz)
        psi = (w * y[0, :nf]).sum(axis=0)
        b = rho*nu*z-k
        f = (z*z-z)/2 + b*psi + nu*nu/2*psi*psi
        dy = np.zeros_like(y)
        dy[0, :nf] = -r*y[0, :nf]+f
        dy[0, nf] = v0*f+k*th*psi
        if tangents:
            dpsi = (w[None, :, :]*y[1:, :nf]).sum(axis=1)
            explicit = np.array([0*z, -psi, 0*z, rho*z*psi+nu*psi*psi, nu*z*psi])
            df = explicit+(b+nu*nu*psi)*dpsi
            dy[1:, :nf] = -r[None, :, :]*y[1:, :nf]+df[:, None, :]
            dy[1:, nf] = v0*df+k*th*dpsi
            dy[1, nf] += f
            dy[2, nf] += th*psi
            dy[3, nf] += k*psi
        return dy.ravel()
    sol = solve_ivp(rhs, (0, t), np.zeros(groups*(nf+1)*nz, dtype=complex),
                    t_eval=[t], method='DOP853', rtol=2e-11, atol=2e-13)
    if not sol.success:
        raise ValueError(sol.message)
    out = sol.y[:, -1].reshape(groups, nf+1, nz)[:, nf]
    return out[0], out[1:]

def prices(t, p, weights, rates, order, cutoff, tangents=True):
    nodes, quad_weights = roots_legendre(order)
    u = (nodes+1)*cutoff/2
    z = np.concatenate([1j*u, 1+1j*u])
    log_m, dl = ode(z, t, p, weights, rates, tangents)
    m = np.exp(log_m).reshape(2, order)
    output = []
    for strike in [80.0, 100.0, 120.0]:
        phase = np.exp(1j*u*math.log(100/strike))
        call = .97*((100-strike)/2 + np.dot(quad_weights,
                    (phase*(100*m[1]-strike*m[0])).imag/u)*cutoff/(2*math.pi))
        d = None
        if tangents:
            dm = (dl*np.exp(log_m)).reshape(5, 2, order)
            d = .97*np.dot((phase*(100*dm[:, 1]-strike*dm[:, 0])).imag/u, quad_weights)*cutoff/(2*math.pi)
        output.append((call, d))
    return output

def rough_series(z, t, h, p, order):
    alpha=mp.mpf(str(h))+mp.mpf('.5');t=mp.mpf(str(t));z=mp.mpc(z)
    v0,k,th,nu,rho=p
    c=(z*z-z)/2;b=rho*nu*z-k;a=nu*nu/2
    coeff=[mp.mpc(0),c/mp.gamma(1+alpha)]
    exponent=v0*c*t
    for n in range(1,order+1):
        f=b*coeff[n]+a*sum((coeff[j]*coeff[n-j] for j in range(1,n)),mp.mpc(0))
        exponent+=(v0*f+k*th*coeff[n])*t**(alpha*n+1)/(alpha*n+1)
        coeff.append(mp.gamma(1+alpha*n)/mp.gamma(1+alpha*(n+1))*f)
    return exponent

def rough_derivatives(h, t, z, order):
    with mp.workdps(60):
        p=[mp.mpf(str(v)) for v in P]
        values=[]
        for q in range(5):
            def fun(x):
                pp=p.copy();pp[q]=x
                return rough_series(z, t, h, pp, order)
            values.append(mp.diff(fun,p[q]))
        return values

def generate():
    rows=[]
    for kind, (w,r) in [('heston',([1.0],[0.0])),('lift',(WEIGHTS,RATES))]:
        for t in [0.25,1.0]:
            a=prices(t,P,w,r,160,192.0)
            b=prices(t,P,w,r,240,256.0)
            for j,strike in enumerate([80.0,100.0,120.0]):
                if abs(a[j][0]-b[j][0])>2e-7 or np.max(np.abs(a[j][1]-b[j][1]))>PROTOCOL['independent_quadrature_tolerance']:
                    raise ValueError(('independent quadrature refinement',kind,t,strike,a[j],b[j]))
            # Finite differences of independent P1/P2 prices, NOT tangent code.
            for q in range(5):
                for h in PROTOCOL['bump_sizes']:
                    up=P.copy();down=P.copy();up[q]+=h;down[q]-=h
                    u=prices(t,up,w,r,240,256.0,False)
                    d=prices(t,down,w,r,240,256.0,False)
                    for j in range(3):
                        fd=(u[j][0]-d[j][0])/(2*h)
                        if abs(fd-b[j][1][q])>PROTOCOL['independent_bump_tolerance']:
                            raise ValueError(('independent derivative bump',kind,t,q,j,h,fd,b[j][1][q]))
            for j,strike in enumerate([80.0,100.0,120.0]):
                rows.append(dict(family=kind,maturity=t,strike=strike,call=b[j][0],derivatives=b[j][1].tolist()))
    rough=[]
    for h in [.1,.3]:
        for t in [.25,1.0]:
            for z in [.5+.7j,2j,1+.7j]:
                a=rough_derivatives(h,t,z,100)
                b=rough_derivatives(h,t,z,140)
                if max(abs(x-y) for x,y in zip(a,b))>mp.mpf('1e-25'):
                    raise ValueError(('rough derivative series refinement',h,t,z))
                rough.append(dict(hurst=h,maturity=t,exponent=[z.real,z.imag],
                                  derivatives=[[float(v.real),float(v.imag)] for v in b]))
    return dict(version=1,names=NAMES,parameters=P,lift_weights=WEIGHTS,lift_rates=RATES,
                forward=100.0,discount=0.97,protocol=PROTOCOL,rows=rows,rough_transforms=rough)

def compare(saved, fresh, path=''):
    if isinstance(fresh,dict):
        if not isinstance(saved,dict) or set(saved)!=set(fresh): raise ValueError(('keys',path))
        for k,v in fresh.items(): compare(saved[k],v,path+'/'+k)
    elif isinstance(fresh,list):
        if not isinstance(saved,list) or len(saved)!=len(fresh): raise ValueError(('length',path))
        for j,(a,b) in enumerate(zip(saved,fresh)): compare(a,b,path+f'/{j}')
    elif isinstance(fresh,float):
        computed='/derivatives/' in path or path.endswith('/call')
        tol=2e-8 if computed else 0.0
        if type(saved) not in (float,int) or not math.isfinite(saved) or abs(saved-fresh)>tol:
            raise ValueError(('value',path,saved,fresh))
    elif type(saved) is not type(fresh) or saved!=fresh: raise ValueError(('protocol',path))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    fresh=generate()
    if args.generate:
        FIXTURE.write_text(json.dumps(fresh,indent=2)+'\n')
    else: compare(json.loads(FIXTURE.read_text()),fresh)
    print('independent parameter references: 12 price rows / 60 derivatives; 12 rough transforms / 60 complex derivatives')
if __name__=='__main__': main()
