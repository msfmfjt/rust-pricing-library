"""Compare complete public Gamma-chain audits with an installed baseline wheel.

Run this same script in each wheel's isolated Python environment. The baseline
report passed with --reference must match every input, ordinary price, and audit
field exactly; timing data and installation paths are intentionally separate.
This bounded benchmark supplements, and does not replace, acceptance tests.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backtesting"))
from multi_step_cases import model
from rust_pricing_backtest.multi_step_full_smile import MultiStepSmileCalibrator


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1 or args.repetitions < 1:
        parser.error("samples and repetitions must be positive")
    reference = None if args.reference is None else json.loads(args.reference.read_text())
    cases = []
    for intervals in [4, 6]:
        for kind in ["asian", "lookback"]:
            payload = model(intervals, kind, samples=args.samples, seed=7)
            payload["risk"] = dict(
                delta=False, vega=False,
                smile_dynamics=dict(type="sticky_log_moneyness"),
            )
            calibrator = MultiStepSmileCalibrator(payload)
            tangent, _ = calibrator.all_maturity_iv_tangent()
            directions = tangent.T.reshape(5 * intervals // 2, intervals // 2, 33).tolist()
            request = rp.PricingRequest.from_json(json.dumps(payload))
            plan = rp.PricingPlan.compile(request, worker_threads=4, reduction_block_size=256)
            before = json.loads(plan.evaluate().to_json())
            audit = None
            timings = []
            for _ in range(args.repetitions):
                started = time.perf_counter()
                observed = plan.evaluate_vega_kt_multi_step_gamma_chain_audit(
                    calibrator.calibrated_rows, [80., 90., 100., 110., 120.],
                    variance_risk_directions=directions, subdivisions=512,
                    gamma_relative_standard_error_limit=0.5,
                )
                timings.append(time.perf_counter() - started)
                if observed["status"] != "path_gamma_chain_passed":
                    raise AssertionError(f"{intervals}/{kind}: Gamma acceptance failed")
                if audit is not None and digest(observed) != digest(audit):
                    raise AssertionError(f"{intervals}/{kind}: repeated audit changed")
                audit = observed
            if digest(json.loads(plan.evaluate().to_json())) != digest(before):
                raise AssertionError(f"{intervals}/{kind}: ordinary price changed after Gamma")
            result = dict(
                intervals=intervals, kind=kind, payload=payload, directions=directions,
                ordinary_price=before, audit=audit, audit_sha256=digest(audit),
                elapsed_seconds=timings, median_seconds=statistics.median(timings),
            )
            if reference is not None:
                prior = reference["cases"][len(cases)]
                for key in ["intervals", "kind", "payload", "directions", "ordinary_price", "audit"]:
                    if digest(prior[key]) != digest(result[key]):
                        raise AssertionError(f"{intervals}/{kind}: baseline mismatch in {key}")
                result["baseline_speedup"] = prior["median_seconds"] / result["median_seconds"]
            cases.append(result)
            print(json.dumps({k: result[k] for k in [
                "intervals", "kind", "audit_sha256", "elapsed_seconds", "median_seconds",
            ]}), flush=True)
    if reference is not None and len(reference["cases"]) != len(cases):
        raise AssertionError("baseline case inventory differs")
    native_dir = Path(rp.__file__).parent
    native_files = sorted([*native_dir.glob("*.so"), *native_dir.glob("*.pyd")])
    report = dict(
        captured_at=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        python=sys.version, module=str(rp.__file__), samples=args.samples,
        repetitions=args.repetitions, subdivisions=512, worker_threads=4,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in native_files},
        thread_environment={k: os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]},
        reference=None if args.reference is None else str(args.reference),
        exact_baseline_parity=reference is not None, cases=cases,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
