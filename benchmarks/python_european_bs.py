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
SPOT = 100.0
VOLATILITY = 0.2
VALIDATION_SPOT_BUMP = 1.0
VALIDATION_VOLATILITY_BUMP = 1.0e-4


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python_european_bs.py <output.json>")

    request = build_request(SPOT, VOLATILITY, full_risk=True)
    bump_requests = [
        build_request(SPOT, VOLATILITY, full_risk=False),
        build_request(SPOT - VALIDATION_SPOT_BUMP, VOLATILITY, full_risk=False),
        build_request(SPOT + VALIDATION_SPOT_BUMP, VOLATILITY, full_risk=False),
        build_request(
            SPOT, VOLATILITY - VALIDATION_VOLATILITY_BUMP, full_risk=False
        ),
        build_request(
            SPOT, VOLATILITY + VALIDATION_VOLATILITY_BUMP, full_risk=False
        ),
    ]
    plan = rp.PricingPlan.compile(
        request, worker_threads=2, reduction_block_size=256
    )
    result = plan.evaluate()
    bump_plans = [
        rp.PricingPlan.compile(item, worker_threads=2, reduction_block_size=256)
        for item in bump_requests
    ]
    for bump_plan in bump_plans:
        bump_plan.evaluate()

    compile_seconds = sample(
        COMPILE_SAMPLES,
        lambda: rp.PricingPlan.compile(
            request, worker_threads=2, reduction_block_size=256
        ),
    )
    evaluate_seconds = sample(EVALUATION_SAMPLES, plan.evaluate)
    compile_bump_seconds = sample(
        COMPILE_SAMPLES,
        lambda: [
            rp.PricingPlan.compile(
                item, worker_threads=2, reduction_block_size=256
            )
            for item in bump_requests
        ],
    )
    evaluate_bump_seconds = sample(
        EVALUATION_SAMPLES,
        lambda: [bump_plan.evaluate() for bump_plan in bump_plans],
    )
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
            "compile_crn_bump_validation_from_python": summarize(
                compile_bump_seconds
            ),
            "evaluate_crn_bump_validation_from_python": summarize(
                evaluate_bump_seconds, SAMPLING_UNITS * 2 * len(bump_plans)
            ),
            "result_value_getter": summarize(getter_seconds),
        },
        "notes": [
            "Compile/evaluate release the GIL; timings include the Python-to-Rust call boundary.",
            "The full-risk kernel includes AAD and its CRN bump validations.",
            "The standalone bump case evaluates base, Spot-down/up, and volatility-down/up Price-only plans with common random numbers.",
            "Results are an optimization baseline and not a latency SLA.",
        ],
    }
    serialized = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    Path(sys.argv[1]).write_bytes(serialized.encode("utf-8"))


def build_request(
    spot: float, volatility: float, *, full_risk: bool
) -> rp.PricingRequest:
    discount = rp.DiscountCurve(1, np.array([0.0, 1.0]), np.exp([0.0, -0.05]))
    dividend = rp.DiscountCurve(2, np.array([0.0, 1.0]), np.exp([0.0, -0.02]))
    return rp.PricingRequest(
        "2026-09-04",
        rp.Product.european_vanilla(1, 1, "2027-09-04", 100.0, 1.0, "call"),
        rp.Market.equity(1, 1, spot, discount, dividend),
        rp.Model.black_scholes(volatility),
        rp.Engine.pseudo_monte_carlo(
            0x0123_4567_89AB_CDEF, SAMPLING_UNITS, antithetic=True
        ),
        (
            rp.RiskRequest(
                delta=True,
                gamma_relative_bump=0.01,
                vega=True,
                checkpoint_interval=16,
                aad_tile_capacity=128,
            )
            if full_risk
            else rp.RiskRequest()
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
