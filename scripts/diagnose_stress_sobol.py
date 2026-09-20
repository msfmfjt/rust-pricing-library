#!/usr/bin/env python3
"""Inspect the exact digital-net projection implicated in the 2F stress case.

This reads the bundled Joe--Kuo directions; it does not approximate densities
from random draws. GF(2) ranks determine occupancy of dyadic cells for a 2**m
point prefix. Gray-code ordering is an invertible transform of the input bits.
The production lower-triangular linear scramble preserves leading-row rank;
the digital shift changes which cells are occupied, not how many.

This is a sampling diagnostic, not a price-accuracy acceptance test.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "crates/pricing/data/joe-kuo-6.21201-u32be.bin"


def gf2_rank(vectors):
    pivots = {}
    for value in vectors:
        while value:
            bit = value.bit_length() - 1
            if bit in pivots:
                value ^= pivots[bit]
            else:
                pivots[bit] = value
                break
    return len(pivots)


def leading_rows(data, dimension, m):
    columns = struct.unpack_from(">32I", data, 20 + dimension * 128)
    return [
        sum(((column >> (31 - bit)) & 1) << k for k, column in enumerate(columns[:m]))
        for bit in range(m)
    ]


def projection(data, dimensions, m):
    first, second = (leading_rows(data, d, m) for d in dimensions)
    strength = next(
        q
        for q in range(m, -1, -1)
        if all(gf2_rank(first[:a] + second[: q - a]) == q for a in range(q + 1))
    )
    rank = gf2_rank(first[:6] + second[:1])
    occupied = 2**rank
    return {
        "dimensions_zero_based": dimensions,
        "points": 2**m,
        "two_dimensional_t_value": m - strength,
        "dyadic_grid": [64, 2],
        "leading_row_rank": rank,
        "occupied_cells": occupied,
        "empty_cells": 128 - occupied,
        "points_per_occupied_cell": 2 ** (m - rank),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=309)
    args = parser.parse_args()
    data = ASSET.read_bytes()
    magic, version, dimensions, bits = struct.unpack_from(">8sIII", data)
    assert (magic, version, bits) == (b"JK621201", 1, 32)
    assert len(data) == 20 + dimensions * 128
    assert 0 < 4 * args.steps < dimensions
    # An independent known property of the first two Sobol coordinates.
    assert projection(data, [0, 1], 15)["two_dimensional_t_value"] == 0
    cases = {
        "2f_stress_factor_major": [args.steps, 4 * args.steps],
        "2f_bridge_rank_major": [1, 4],
        "bs_stress_factor_major": [args.steps, 2 * args.steps],
        "2f_one_year_factor_major": [128, 512],
    }
    print(json.dumps({
        "direction_asset_sha256": hashlib.sha256(data).hexdigest(),
        "actual_stress_steps": args.steps,
        "results": [
            {"case": name, **projection(data, pair, m)}
            for name, pair in cases.items()
            for m in [11, 13, 15, 16]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
