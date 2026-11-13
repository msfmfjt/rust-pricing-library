"""Repeat calibration, hedge sizing and out-of-sample valuation of the 100-note book."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from statistics import fmean, stdev

import rust_pricing as rp
from cross_currency_autocall_quote_risk import QUOTES, seasoned_basket

NOTES = 100
SHARES_PER_CONTRACT = 100
INNER_REPLICATIONS = 8
POINTS = 1024
SCRAMBLES = 4
BUMP = .005
# The native engine advances its master seed by this odd constant per stream.
# Each sizing replication uses three streams and each holdout uses another three.
STREAM_STEP = 0x9e3779b97f4a7c15
REFERENCE_SEED = 29737


def run_cycle(index, worker_threads):
    # Index zero reproduces the earlier sizing seed. Positive indices are disjoint
    # from that reference and one another even if portfolio quantities coincide.
    seed = (REFERENCE_SEED + index*6*INNER_REPLICATIONS*STREAM_STEP) % (1 << 64)
    engine = rp.Engine.randomized_quasi_monte_carlo(
        POINTS, seed, scramble_count=SCRAMBLES, antithetic=True, brownian_bridge=True)
    report = seasoned_basket().bergomi_quote_hedge(
        QUOTES, engine, note_quantity=NOTES,
        contract_sizes={stock: [SHARES_PER_CONTRACT]*2 for stock in QUOTES},
        whole_contracts=True, bump=BUMP, replications=INNER_REPLICATIONS, worker_threads=worker_threads)
    ideal, actual = report.theoretical_hedged.vegas, report.hedged.vegas
    theoretical, contracts = report.theoretical_quantities, report.quantities
    rows = []
    for stock in QUOTES:
        for i, (expiry, side, strike, _, _) in enumerate(QUOTES[stock]):
            rows.append(dict(stock=stock, expiry=expiry, side=side, strike=strike,
                             theoretical=theoretical[stock][i],
                             contracts=contracts[stock][i],
                             ideal=ideal[stock][i].value, ideal_se=ideal[stock][i].standard_error,
                             actual=actual[stock][i].value, actual_se=actual[stock][i].standard_error))
    return dict(index=index, seed=seed, rows=rows)


def summarize(cycles):
    """One cluster per independently sized book. All residuals here are for 100 notes."""
    if len(cycles) < 2:
        raise ValueError("at least two independent sizing cycles are required")
    if len({cycle["seed"] for cycle in cycles}) != len(cycles):
        raise ValueError("repeated seed in sizing cycles")
    count = len(cycles)
    summaries = []
    for bucket in zip(*(cycle["rows"] for cycle in cycles), strict=True):
        ids = {(row["stock"], row["expiry"]) for row in bucket}
        if len(ids) != 1:
            raise ValueError("quote buckets differ between sizing cycles")
        stock, expiry = next(iter(ids))
        result = dict(stock=stock, expiry=expiry, quantity_mean=fmean(r["theoretical"] for r in bucket))
        for kind in ("ideal", "actual"):
            residuals = [r[kind] for r in bucket]
            # Typical SE for ONE complete experiment when holding its chosen trades
            # fixed vs the SD across experiments that also repeat the sizing stage.
            within = sqrt(fmean(r[kind+"_se"]**2 for r in bucket))
            complete = stdev(residuals)
            result[kind] = dict(mean=fmean(residuals), fixed_se=within, full_se=complete,
                                mean_se=complete/sqrt(count), negative=sum(x < 0 for x in residuals))
        differences = [r["actual"]-r["ideal"] for r in bucket]
        result["rounding"] = dict(mean=fmean(differences), mean_se=stdev(differences)/sqrt(count),
                                  mean_absolute=fmean(abs(delta) for delta in differences))
        frequencies = {}
        for row in bucket:
            quantity = int(row["contracts"])
            frequencies[quantity] = frequencies.get(quantity, 0) + 1
        result["contracts"] = frequencies
        summaries.append(result)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=16, help="independent new sizing and holdout cycles")
    parser.add_argument("--first-cycle", type=int, default=1,
                        help="first independent cycle index (use 17 for the second batch)")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--worker-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cycles < 2 or min(args.first_cycle, args.jobs, args.worker_threads) < 1:
        parser.error("use at least two cycles, a positive first cycle and positive worker counts")
    # The first batch also emits the original reference, outside the summary.
    start = 0 if args.first_cycle == 1 else args.first_cycle
    indices = range(start, args.first_cycle + args.cycles)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = []
        for cycle in pool.map(lambda i: run_cycle(i, args.worker_threads), indices):
            print(json.dumps(cycle), flush=True)
            if cycle["index"]:
                results.append(cycle)
    print(json.dumps(dict(cycles=len(results), summary=summarize(results))), flush=True)


if __name__ == "__main__":
    main()
