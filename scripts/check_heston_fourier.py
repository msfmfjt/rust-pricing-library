"""Independent Fourier/Riccati references; never import the pricing extension.

Rough CF: 70-digit fractional power series, independently summed at two orders.
Markov CF/prices: SciPy adaptive ODE and closed Heston; prices use P1/P2 inversion,
not the production half-moment/control-variate/Simpson formula.
Rough prices: an explicit fractional Adams predictor/corrector and an independently
integrated terminal functional. Their half-moment/Simpson inversion is shared in
form with production; the six retained prices are numerical cross-checks, not
certified continuum prices. The reference grids and budgets come from PR #123.
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
from scipy.special import gamma, ndtr

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'fixtures/rough-volatility/fourier.json'
P = dict(initial_variance=0.04, mean_reversion=0.7, long_run_variance=0.055,
         vol_of_vol=0.18, correlation=-0.65)
ROUGH_PRICE_PROTOCOL = dict(
    price_abs_tolerance=0.003, time_steps=1024, integration_intervals=1024,
    cutoff=160.0, reference_time_steps=[2048, 4096],
    reference_fourier_grids=[[1024, 160.0], [2560, 200.0]],
    reference_time_change_cap=1e-4, reference_inversion_change_cap=2e-7,
)

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


def rough_adams_transform(hurst, maturity, frequencies, steps):
    """Explicit fractional predictor/corrector, independent of the Rust solver.

    Retained from b41b767d2b17e1445f9fd53f877632bda110a170 (PR #123).
    No implicit quadratic root or production kernel helper is used. Integrate
    the piecewise-linear Riccati forcing against g0 for the terminal functional,
    instead of applying production's final trapezoidal integral to psi.
    """
    alpha = hurst + 0.5
    dt = maturity / steps
    z = 0.5 + 1j * np.asarray(frequencies)
    v0, kappa, theta, nu, rho = (P[key] for key in P)
    c = (z * z - z) / 2
    b = rho * nu * z - kappa
    a = nu * nu / 2
    q = np.empty((steps + 1, len(z)), dtype=complex)
    q[0] = c
    r = np.arange(steps + 2, dtype=float)
    predictor = dt**alpha / gamma(alpha + 1) * np.diff(r**alpha)
    interior = dt**alpha / gamma(alpha + 2) * np.diff(r**(alpha + 1), n=2)
    newest = dt**alpha / gamma(alpha + 2)
    for n in range(1, steps + 1):
        predicted = predictor[:n][::-1] @ q[:n]
        first = newest * ((n - 1)**(alpha + 1) - (n - 1 - alpha) * n**alpha)
        history = first * q[0]
        if n > 1:
            history = history + interior[:n - 1][::-1] @ q[1:n]
        psi = history + newest * (c + b * predicted + a * predicted * predicted)
        q[n] = c + b * psi + a * psi * psi
    if not np.all(np.isfinite(q)):
        raise ValueError('Adams reference instability')
    # g0(T-s) = v0 + kappa*theta*(T-s)^alpha/Gamma(alpha+1).
    # Integrate the linear interpolant of q against this input curve exactly.
    beta = alpha + 1
    r = np.arange(steps, dtype=float)
    d0 = ((r + 1)**beta - r**beta) / beta
    d1 = ((r + 1)**(beta + 1) - r**(beta + 1)) / (beta + 1)
    left = dt**beta / gamma(beta) * (d1 - r * d0)
    right = dt**beta / gamma(beta) * ((r + 1) * d0 - d1)
    fractional = left[::-1] @ q[:-1] + right[::-1] @ q[1:]
    ordinary = dt * (q.sum(axis=0) - 0.5 * (q[0] + q[-1]))
    return np.exp(v0 * ordinary + kappa * theta * fractional)


def rough_adams_prices(hurst, steps, intervals, cutoff):
    """One-year calls at the fixed retained strikes, forward 100 and discount .97."""
    frequencies = np.linspace(0, cutoff, intervals + 1)
    z = 0.5 + 1j * frequencies
    transform = rough_adams_transform(hurst, 1.0, frequencies, steps)
    variance = P['initial_variance']
    control = np.exp((z * z - z) * variance / 2)
    weights = np.ones(intervals + 1)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    prices = []
    for strike in [80.0, 100.0, 120.0]:
        integrand = (np.exp(1j * frequencies * math.log(100.0 / strike))
                     * (control - transform)).real / (frequencies * frequencies + 0.25)
        integral = (cutoff / intervals) / 3 * (weights @ integrand)
        sd = math.sqrt(variance)
        d1 = math.log(100.0 / strike) / sd + sd / 2
        black = 100.0 * ndtr(d1) - strike * ndtr(d1 - sd)
        prices.append(float(0.97 * (black + math.sqrt(100.0 * strike) / math.pi * integral)))
    return prices


def rough_price_references():
    protocol = ROUGH_PRICE_PROTOCOL
    coarse_steps, fine_steps = protocol['reference_time_steps']
    (intervals, cutoff), (extended_intervals, extended_cutoff) = protocol['reference_fourier_grids']
    rows = []
    for hurst in [0.1, 0.3]:
        coarse = rough_adams_prices(hurst, coarse_steps, intervals, cutoff)
        fine = rough_adams_prices(hurst, fine_steps, intervals, cutoff)
        extended = rough_adams_prices(hurst, fine_steps, extended_intervals, extended_cutoff)
        for strike, c, f, e in zip([80.0, 100.0, 120.0], coarse, fine, extended, strict=True):
            if abs(f - c) >= protocol['reference_time_change_cap']:
                raise ValueError(('Adams time-grid crosscheck', hurst, strike, c, f))
            if abs(e - f) >= protocol['reference_inversion_change_cap']:
                raise ValueError(('Adams inversion crosscheck', hurst, strike, f, e))
            rows.append(dict(hurst=hurst, maturity=1.0, strike=strike, call=e,
                             reference_time_change=f - c, reference_inversion_change=e - f))
    return rows


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
                rough_prices=rough_price_references(),rough_price_protocol=ROUGH_PRICE_PROTOCOL,
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
        is_rough_price_reference = path.startswith('/rough_prices/') and path.endswith(
            ('/call', '/reference_time_change', '/reference_inversion_change'))
        # Preserve PR #123's tighter rounding allowance for its retained prices
        # and reference-grid changes; the existing references keep their budget.
        tolerance = 2e-9 if is_rough_price_reference else (2e-8 if is_reference else 0.0)
        if type(saved) not in (float, int) or not math.isfinite(saved) or abs(saved-fresh) > tolerance:
            raise ValueError(('reference/input mismatch', path, saved, fresh))
    elif type(saved) is not type(fresh) or saved != fresh:
        raise ValueError(('protocol mismatch', path))


def check():
    saved = json.loads(FIXTURE.read_text())
    compare(saved, generate())
    print(f"independent Fourier references passed: {len(saved['rough_transforms'])} rough CF, "
          f"{len(saved['markov_transforms'])} Markov CF, {len(saved['prices'])} Markov prices, "
          f"{len(saved['rough_prices'])} fractional Adams prices")

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    if args.generate:
        FIXTURE.write_text(json.dumps(generate(),indent=2)+'\n');print(FIXTURE)
    else: check()
