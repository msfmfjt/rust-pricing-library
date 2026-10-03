"""Independent finite-grid lifted-Heston kernel references.

This compares rational, semi-implicit kernels, not exp(-x*t) kernels or
continuous-time option prices. Production Rust/Python pricing is not imported.
The infinite-Laplace-measure integral is evaluated by endpoint-weighted
QUADPACK and cross-checked with a beta/gamma identity.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.integrate import quad
from scipy.special import betaln, gammaln

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/rough-volatility/lifted-factor-prices.json"
FACTORS = (8, 16, 32, 64, 128, 256)
STEPS = 64
LAGS = (1, 2, 8, 32, 64)


def infinite_kernel(alpha: float, dt: float, lag: int) -> float:
    """Integrate over y=x*dt/(1+x*dt) in (0,1)."""
    integral, error = quad(
        lambda _: 1.0, 0.0, 1.0, weight="alg",
        wvar=(-alpha, lag+alpha-2.0), epsabs=2e-12, epsrel=2e-12,
    )
    factor = dt**(alpha-1) / (math.gamma(alpha)*math.gamma(1-alpha))
    value = factor*integral
    alternate = math.exp((alpha-1)*math.log(dt)
                         + betaln(1-alpha, lag+alpha-1)
                         - gammaln(alpha)-gammaln(1-alpha))
    assert error*factor <= 2e-10*max(1.0, value), (alpha, dt, lag, error)
    assert abs(value-alternate) < 2e-11*max(1.0, value), (alpha, dt, lag)
    return float(value)


def finite_kernel(alpha: float, dt: float, lag: int, n: int) -> float:
    # Direct bin endpoint expressions, independent of Rust's exp_m1 factory.
    ratio = math.exp(3/math.sqrt(n))
    p = 1-alpha
    terms = []
    for j in range(n):
        low, high = ratio**(j-n/2), ratio**(j+1-n/2)
        mass = (high**p-low**p)/(math.gamma(alpha)*math.gamma(2-alpha))
        rate = p/(p+1)*(high**(p+1)-low**(p+1))/(high**p-low**p)
        terms.append(mass*math.exp(-lag*math.log1p(rate*dt)))
    return math.fsum(terms)


def reference() -> dict:
    rows = []
    for h in (0.1, 0.3):
        for maturity in (0.25, 1.0):
            dt, alpha = maturity/STEPS, h+0.5
            target = [infinite_kernel(alpha, dt, lag) for lag in LAGS]
            rows.append(dict(hurst=h, maturity=maturity, steps=STEPS, lags=list(LAGS),
                infinite_kernel=target,
                finite_kernels=[dict(factors=n, ratio=math.exp(3/math.sqrt(n)),
                    values=[finite_kernel(alpha,dt,lag,n) for lag in LAGS]) for n in FACTORS]))
    return dict(version=1, scope="fixed-grid-semi-implicit-infinite-factor-limit",
        protocol=dict(hursts=[0.1,0.3], maturities=[0.25,1.0], steps=STEPS,
            factors=list(FACTORS), ratio_rule="exp(3/sqrt(factors))", seeds=[91,1973],
            independent_antithetic_units=8192, strikes=[80.0,100.0,120.0],
            spot=100.0, initial_variance=0.04, mean_reversion=0.7,
            long_run_variance=0.055, vol_of_vol=0.18, correlation=-0.65,
            final_factors=256, paired_price_budget=0.025, paired_se_cap=0.002,
            forward_sampling_sigmas=5.0, forward_se_cap=0.15),
        kernels=rows)


def compare(actual, expected, path="root") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key, value in expected.items():
            compare(actual[key], value, path+"."+key)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (x,y) in enumerate(zip(actual,expected)):
            compare(x,y,f"{path}[{i}]")
    elif isinstance(expected,float):
        assert isinstance(actual,(float,int)) and not isinstance(actual,bool), path
        assert math.isfinite(actual), (path,actual)
        assert abs(actual-expected) <= 2e-11*max(1,abs(expected)), (path,actual,expected)
    else:
        assert type(actual) is type(expected) and actual == expected, (path,actual,expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true",help="explicitly generate reviewed reference data")
    args = parser.parse_args()
    expected = reference()
    if args.write:
        FIXTURE.write_text(json.dumps(expected,indent=2,allow_nan=False)+"\n")
    else:
        compare(json.loads(FIXTURE.read_text()),expected)
    print("20 infinite-factor kernel values: independent integral and beta identity agree")
    print("24 finite-factor kernels and all fixed price-protocol inputs checked")


if __name__ == "__main__":
    main()
