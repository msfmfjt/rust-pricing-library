#!/usr/bin/env python3
"""Independent closed-form Gaussian moments for S^2-weighted native log hats."""
import argparse
import json
from decimal import Decimal as D, localcontext
from pathlib import Path
from check_local_vol_gamma_numerics import cdf, PI

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/local-vol/density-mass-v0.1.json'


def masses(spot, variance, nodes):
    sd = variance.sqrt()
    z = [(x-D('1.5')*variance)/sd for x in nodes]
    pdf = lambda x: (-x*x/2).exp()/(2*PI).sqrt()
    out = [D(0)]*len(nodes)
    out[0] = cdf(z[0])
    out[-1] = 1-cdf(z[-1])
    for i, (a, b) in enumerate(zip(z, z[1:])):
        probability = cdf(b)-cdf(a)
        first = pdf(a)-pdf(b)
        out[i] += (b*probability-first)/(b-a)
        out[i+1] += (first-a*probability)/(b-a)
    return [spot*spot*variance.exp()*v for v in out]


def artifact():
    cases = [
        ('central', .04, [-.5, -.12, 0., .14, .5]),
        ('underresolved', .04*(8/365), [-.2, -.1, 0., .1, .2]),
        ('left_tail', .04, [-1.6, -1.5, -1.4]),
        ('right_tail', .04, [1.5, 1.6, 1.7]),
        ('narrow', .04, [0., 1e-8, 2e-8]),
        ('narrow_tail', .04, [1.5, 1.50000001, 1.50000002]),
        ('short_variance', 1e-20, [-2e-10, 0., 2e-10]),
        ('clamped_endpoints', .04, [-.02, .03]),
    ]
    return {'fixture_version': 1, 'reference': '120-digit Decimal Gaussian power series and exact truncated first moments; binary64 inputs',
            'cases': [{'id': name, 'spot': 100., 'variance': v, 'nodes': x,
                       'expected': [float(m) for m in masses(D(100), D(v), list(map(D, x)))]}
                      for name, v, x in cases]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true')
    a=p.parse_args()
    with localcontext() as ctx:
        ctx.prec=120
        payload=json.dumps(artifact(), indent=2, allow_nan=False)+'\n'
    if a.write:
        FIXTURE.write_text(payload)
    elif FIXTURE.read_text()!=payload:
        raise AssertionError('Density mass references changed')
    print('Eight independent integrated-density reference cases verified')


if __name__=='__main__':
    main()
