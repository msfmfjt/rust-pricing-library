"""Measure the installed Python facade without introducing benchmark dependencies."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np
import rust_pricing as rp

SAMPLING_UNITS = 16_384
COMPILE_SAMPLES = 20
EVALUATION_SAMPLES = 5
GETTER_SAMPLES = 100_000


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python_european_bs.py <output.json>")

    request = build_request()
    plan = rp.PricingPlan.compile(
        request, worker_threads=2, reduction_block_size=256
    )
    result = plan.evaluate()

    compile_seconds = sample(
        COMPILE_SAMPLES,
        lambda: rp.PricingPlan.compile(
            request, worker_threads=2, reduction_block_size=256
        ),
    )
    evaluate_seconds = sample(EVALUATION_SAMPLES, plan.evaluate)
    getter_seconds = sample(GETTER_SAMPLES, lambda: result.value)

    report = {
        "schema_version": 1,
        "benchmark_kind": "python_european_black_scholes",
        "library_version": rp.__version__,
        "configuration": {
            "engine": "pseudo_monte_carlo",
            "sampling_units": SAMPLING_UNITS,
            "antithetic": True,
            "evaluated_paths": SAMPLING_UNITS * 2,
            "worker_threads": 2,
            "reduction_block_size": 256,
        },
        "measurements": {
            "compile_full_risk_from_python": summarize(compile_seconds),
            "evaluate_full_risk_from_python": summarize(
                evaluate_seconds, SAMPLING_UNITS * 2
            ),
            "result_value_getter": summarize(getter_seconds),
        },
        "notes": [
            "Compile/evaluate release the GIL; timings include the Python-to-Rust call boundary.",
            "The full-risk kernel includes AAD and its CRN bump validations.",
            "Results are an optimization baseline and not a latency SLA.",
        ],
    }
    Path(sys.argv[1]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_request() -> rp.PricingRequest:
    discount = rp.DiscountCurve(1, np.array([0.0, 1.0]), np.exp([0.0, -0.05]))
    dividend = rp.DiscountCurve(2, np.array([0.0, 1.0]), np.exp([0.0, -0.02]))
    return rp.PricingRequest(
        "2026-09-04",
        rp.Product.european_vanilla(1, 1, "2027-09-04", 100.0, 1.0, "call"),
        rp.Market.equity(1, 1, 100.0, discount, dividend),
        rp.Model.black_scholes(0.2),
        rp.Engine.pseudo_monte_carlo(
            0x0123_4567_89AB_CDEF, SAMPLING_UNITS, antithetic=True
        ),
        rp.RiskRequest(
            delta=True,
            gamma_relative_bump=0.01,
            vega=True,
            checkpoint_interval=16,
            aad_tile_capacity=128,
        ),
    )


def sample(count: int, operation: Callable[[], object]) -> list[float]:
    durations: list[float] = []
    for _ in range(count):
        start = perf_counter()
        operation()
        durations.append(perf_counter() - start)
    return durations


def summarize(seconds: list[float], paths: int | None = None) -> dict[str, object]:
    median = statistics.median(seconds)
    return {
        "samples": len(seconds),
        "median_seconds": median,
        "minimum_seconds": min(seconds),
        "maximum_seconds": max(seconds),
        "evaluated_paths_per_sample": paths,
        "median_paths_per_second": None if paths is None else paths / median,
    }


if __name__ == "__main__":
    main()
