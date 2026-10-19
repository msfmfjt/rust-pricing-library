"""Independent two-step QRH expectations; never imports production bindings.

The last asset Gaussian is integrated analytically by Black, leaving two
Gauss-Hermite coordinates: the first asset draw and its newest-cell residual.
Analytic parameter tangents are checked against price-only numerical stencils.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import digamma, ndtr, roots_hermitenorm

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'fixtures/rough-volatility/quadratic-parameter.json'
PARAMETERS = [0.1, 0.7, 0.4, 0.5, 0.2, 0.03, 0.1]
NAMES = ['initial_state','mean_reversion','vol_of_vol','quadratic','shift','variance_floor','hurst']
# Fixed reference and MC budgets. Not time-grid or model error bounds.
ORDERS = (64, 96)
REFERENCE_TOLERANCE = 2e-7
MC_SLACK = 0.003
MC_SE_CAP = 0.2


def expectation(p, order, *, derivatives=True):
    z0,kappa,nu,a,b,c,h = map(float, p)
    dt = 0.5
    nodes, w = roots_hermitenorm(order)
    weights = w[:,None] * w[None,:] / (2 * math.pi)
    x, y = nodes[:,None], nodes[None,:]
    alpha = h + 0.5
    gamma = math.gamma(alpha)
    loading = dt**(h-0.5)/(alpha*gamma)
    scale = 1/(math.sqrt(2*h)*gamma)
    ratio = (0.5-h)/alpha
    residual = scale * dt**h * ratio
    innovation = loading * math.sqrt(dt)*x + residual*y
    v0 = a*(z0-b)**2+c
    root = math.sqrt(v0)
    z1 = z0-kappa*z0*loading*dt + kappa*nu*root*innovation
    f1 = 100*np.exp(-0.5*v0*dt+math.sqrt(v0*dt)*x)
    v1 = a*(z1-b)**2+c
    total = v1*dt
    sd = np.sqrt(total)
    d1 = (np.log(f1/100)+0.5*total)/sd
    d2 = d1-sd
    value = f1*ndtr(d1)-100*ndtr(d2)
    price = float(np.sum(weights*value))
    if not derivatives:
        return price
    dv0 = np.array([2*a*(z0-b),0.,0.,(z0-b)**2,-2*a*(z0-b),1.,0.])
    dl = loading*(math.log(dt)-1/alpha-digamma(alpha))
    dr = scale*dt**h*(ratio*(math.log(dt)-0.5/h-digamma(alpha))-1/alpha**2)
    delta = ndtr(d1)
    variance_slope = f1*np.exp(-0.5*d1*d1)/math.sqrt(2*math.pi)/(2*sd)
    gradients = []
    for j in range(7):
        dz1 = kappa*nu*dv0[j]/(2*root)*innovation
        if j == 0: dz1 = dz1 + 1-kappa*loading*dt
        elif j == 1: dz1 = dz1 - z0*loading*dt + nu*root*innovation
        elif j == 2: dz1 = dz1 + kappa*root*innovation
        elif j == 6: dz1 = dz1-kappa*z0*dl*dt+kappa*nu*root*(dl*math.sqrt(dt)*x+dr*y)
        df1 = f1*(-0.5*dt+0.5*math.sqrt(dt)/root*x)*dv0[j]
        dv1 = 2*a*(z1-b)*(dz1-(1.0 if j == 4 else 0.0))
        if j == 3: dv1 += (z1-b)**2
        elif j == 5: dv1 += 1
        gradients.append(float(np.sum(weights*(delta*df1+variance_slope*dt*dv1))))
    return [price,*gradients]


def generate():
    rows=[]
    max_resolution = max_stencil = 0.0
    for h in [0.1,0.3,0.5]:
        p=PARAMETERS.copy();p[6]=h
        low=expectation(p,ORDERS[0]);high=expectation(p,ORDERS[1])
        for v,q in zip(low,high):
            gap=abs(v-q);max_resolution=max(gap,max_resolution)
            assert gap <= REFERENCE_TOLERANCE*(1+abs(q)), (h,'resolution',v,q)
        for j,adj in enumerate(high[1:]):
            def price(d):
                q=p.copy();q[j]+=d
                return expectation(q,ORDERS[1],derivatives=False)
            for e in [2e-6,1e-6]:
                fd=((3*price(0)-4*price(-e)+price(-2*e))/(2*e) if j==6 and h==0.5
                    else (price(e)-price(-e))/(2*e))
                gap=abs(fd-adj);max_stencil=max(gap,max_stencil)
                assert gap <= 2e-6*(1+abs(adj)), (h,j,e,adj,fd)
        rows.append({'parameters':p, 'price':high[0], 'adjoints':high[1:]})
    data={'law':'two-step-fixed-driver-quadratic-heston','parameter_names':NAMES,
          'orders':list(ORDERS),'mc_slack':MC_SLACK,'mc_se_cap':MC_SE_CAP,
          'rows':rows}
    print(json.dumps({'reference_max_resolution_gap':max_resolution,'reference_max_stencil_gap':max_stencil}))
    return data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    data=generate()
    if args.write:
        PATH.write_text(json.dumps(data,indent=2)+'\n')
    else:
        old=json.loads(PATH.read_text())
        assert all(old[k]==data[k] for k in ['law','parameter_names','orders','mc_slack','mc_se_cap'])
        for row,new in zip(old['rows'],data['rows'],strict=True):
            assert row['parameters']==new['parameters']
            for a,b in zip([row['price'],*row['adjoints']],[new['price'],*new['adjoints']],strict=True):
                assert abs(a-b)<=1e-10*(1+abs(b))

if __name__=='__main__': main()
