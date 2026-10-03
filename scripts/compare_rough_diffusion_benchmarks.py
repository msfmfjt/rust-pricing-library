"""Compare matched path benchmarks, enforcing output identity but no timing gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

SCHEMA = "rough-diffusion-cache-benchmark/v1"
EXPECTED = {(family, h, steps, grid)
            for family in ("rough_heston", "quadratic_rough_heston")
            for h in (0.1, 0.3, 0.5)
            for steps in (64, 256, 1024)
            for grid in ("uniform", "quadratic_time")}


def rows(data: dict) -> dict:
    if data.get("schema") != SCHEMA or data.get("seed") != 91:
        raise ValueError("unexpected schema or seed")
    result = {}
    for row in data["rows"]:
        key = (row["family"], row["hurst"], row["steps"], row["grid"])
        if key in result or key not in EXPECTED:
            raise ValueError("duplicate or unexpected case")
        if type(row["paths"]) is not int or not 1 <= row["paths"] <= 4096:
            raise ValueError("invalid path count")
        seconds = row["seconds"]
        if not seconds or not all(math.isfinite(x) and x > 0 for x in seconds):
            raise ValueError("invalid timings")
        for field in ("digest", "plan_fingerprint"):
            value = row[field]
            if field == "plan_fingerprint":
                prefix = "blake3-256:"
                if not value.startswith(prefix):
                    raise ValueError("missing fingerprint algorithm prefix")
                value = value[len(prefix):]
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid output/fingerprint digest")
        if type(row["negative_nodes"]) is not int or row["negative_nodes"] < 0:
            raise ValueError("invalid negative-node count")
        result[key] = row
    if set(result) != EXPECTED:
        raise ValueError("missing case")
    return result


def compare(baseline: dict, candidate: dict) -> dict:
    if baseline.get("scope") != candidate.get("scope"):
        raise ValueError("timed scope changed")
    left, right = rows(baseline), rows(candidate)
    output = []
    for key in sorted(EXPECTED):
        a, b = left[key], right[key]
        for field in ("paths", "digest", "plan_fingerprint", "scheme", "negative_nodes"):
            if a[field] != b[field]:
                raise ValueError(f"{key}: {field} changed")
        if len(a["seconds"]) != len(b["seconds"]):
            raise ValueError("repetition count changed")
        old = statistics.median(a["seconds"])
        new = statistics.median(b["seconds"])
        output.append(dict(zip(("family", "hurst", "steps", "grid"), key),
                           baseline_seconds=old, candidate_seconds=new, speedup=old/new))
    return {"bitwise_equal_cases": len(output), "rows": output,
            "geometric_mean_speedup": math.exp(statistics.mean(math.log(r["speedup"]) for r in output)),
            "minimum_speedup": min(r["speedup"] for r in output),
            "maximum_speedup": max(r["speedup"] for r in output),
            "scope": baseline["scope"],
            "timing_gate": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(json.loads(args.baseline.read_text()),
                             json.loads(args.candidate.read_text())), indent=2))


if __name__ == "__main__":
    main()
