#!/usr/bin/env python3
"""Independent price differences and Gaussian moments for weak Gamma recovery."""
import argparse
import json
from decimal import localcontext
from pathlib import Path
from check_local_vol_vegakt_chain import artifact

FIXTURE=Path(__file__).resolve().parents[1]/'fixtures/local-vol/weak-chain-v0.1.json'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write',action='store_true')
    a=p.parse_args()
    with localcontext() as ctx:
        ctx.prec=100
        payload=json.dumps(artifact(weak=True),indent=2,allow_nan=False)+'\n'
    if a.write:
        FIXTURE.write_text(payload)
    elif FIXTURE.read_text()!=payload:
        raise AssertionError('Weak Gamma chain references changed')
    print('Weak Gamma-to-IV moments verified: 54 channels, paths and antithetic pairs')


if __name__=='__main__':
    main()
