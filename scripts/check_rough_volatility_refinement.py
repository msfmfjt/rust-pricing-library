"""Independent Gaussian-coupling and geometric lift reference values.

The coarse newest-cell integral is conditioned on fine Brownian increments
and fine newest-cell residuals. Two distinct quadratures cross-check all
nontrivial cross-cell integrals; no production pricing code is imported.

Geometric lift kernels are compared at strictly positive lags, NOT at zero
or in a continuous-time price norm. The ratio shrinks with factor count.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import roots_jacobi

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/rough-volatility/refinement.json"


def cell(h: float, m: int) -> dict:
    alpha, p = h + 0.5, h - 0.5
    k = math.sqrt(2 * h)
    sigma = (0.5 - h) / (0.5 + h)
    coarse_sigma = m**h * sigma
    loading = k * m ** (h - 0.5) / alpha
    a, b, cross = [], [], []
    nodes, weights = roots_jacobi(128, 0.0, p)
    u = (nodes + 1.0) / 2.0
    for j in range(m):
        lag = m - j - 1
        integral = k * ((lag + 1.0)**alpha - lag**alpha) / alpha
        if lag == 0:
            product = 1.0
        else:
            # Endpoint-weighted QUADPACK versus Gauss-Jacobi quadrature.
            product = 2*h*quad(lambda x: (lag+x)**p, 0, 1,
                weight="alg", wvar=(p, 0.0), epsabs=2e-13, epsrel=2e-13)[0]
            alternate = 2*h * float(np.dot(weights, (lag+u)**p)) / 2**alpha
            assert abs(product-alternate) < 2e-12, (h, m, j)
        cross.append(product)
        a.append((integral-loading) / coarse_sigma)
        b.append((product - k/alpha*integral) / sigma / coarse_sigma)
    remaining = 1.0 - sum(x*x for x in a+b)
    assert remaining > 0 and abs(sum(a)) < 2e-12
    return dict(hurst=h, ratio=m, brownian_coefficients=a,
                residual_coefficients=b, extra_coefficient=math.sqrt(remaining),
                normalized_integral_cross_covariances=cross)


def lift(h: float, n: int) -> dict:
    # Shrinking log-bin width and expanding tails: r_n -> 1, n*log(r_n) -> inf.
    # A test protocol, not an optimized or tolerance-certified production rule.
    ratio = math.exp(3.0 / math.sqrt(n))
    alpha, p = h+0.5, 0.5-h
    lower = ratio ** (np.arange(n)-n/2)
    weights = lower**p * (ratio**p-1) / (math.gamma(alpha)*math.gamma(2-alpha))
    rates = lower * p/(p+1) * (ratio**(p+1)-1)/(ratio**p-1)
    lags = [1/128, 1/32, 0.125, 0.5, 1.0]
    values = [float(np.dot(weights, np.exp(-rates*t))) for t in lags]
    target = [t**(h-0.5)/math.gamma(alpha) for t in lags]
    errors = [abs(v-k)/k for v,k in zip(values,target)]
    return dict(hurst=h, factors=n, ratio=ratio, lags=lags,
                kernel_values=values, relative_errors=errors)


def reference() -> dict:
    return {"version":1,
        "scope":"pairwise-dyadic-Gaussian-coupling-and-positive-lag-kernel",
        "cells":[cell(h,m) for h in [0.1,0.3] for m in [2,4,8]],
        "lifts":[lift(h,n) for h in [0.1,0.3] for n in [8,16,32,64,128,256]]}


def compare(a, b, path="root") -> None:
    if isinstance(b,dict):
        assert a.keys()==b.keys(), path
        for key in b:
            compare(a[key],b[key],path+"."+key)
    elif isinstance(b,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):
            compare(x,y,f"{path}[{i}]")
    elif isinstance(b,float):
        assert abs(a-b)<=2e-11*max(1,abs(b)), (path,a,b)
    else:
        assert a==b,(path,a,b)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--write",action="store_true")
    args=parser.parse_args()
    expected=reference()
    if args.write:
        FIXTURE.write_text(json.dumps(expected,indent=2,allow_nan=False)+"\n")
    else:
        compare(json.loads(FIXTURE.read_text()),expected)
    for h in [0.1,0.3]:
        rows=[r for r in expected["lifts"] if r["hurst"]==h]
        for previous,current in zip(rows,rows[1:]):
            assert max(current["relative_errors"]) < max(previous["relative_errors"])
        assert max(rows[-1]["relative_errors"]) < 0.025
    print("6 Gaussian cell couplings: independent quadratures agree")
    print("12 lift kernels: retained values checked; positive-lag error decreases")

if __name__=="__main__":
    main()
