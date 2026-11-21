#!/usr/bin/env python3
"""Run the four registered application cases, using the shared benchmark runner."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def check(actual, expected, path):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"fields changed at {path}")
        for k, v in expected.items():
            check(actual[k], v, f"{path}/{k}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"shape changed at {path}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            check(a, e, f"{path}/{i}")
    elif isinstance(expected, float):
        if not math.isfinite(actual) or abs(actual - expected) > 2e-10 * (1 + abs(expected)):
            raise ValueError(f"{path}: {actual} expected {expected}")
    elif actual != expected:
        raise ValueError(f"{path}: {actual} expected {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aad-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.aad_root.resolve() / "benchmarks"))
    import run
    binaries = run.build(HERE / "Cargo.toml", "model_matrix", "1.98.1", HERE / "target/matrix")
    env = run.host()
    env.update({"revision":run.output("git","rev-parse","HEAD",cwd=HERE),
                "dirty":bool(run.output("git","status","--porcelain",cwd=HERE)),
                "rustc":run.output("rustc","+1.98.1","-Vv"),"profile":"release",
                "lock_sha256":run.sha256(HERE / "Cargo.lock"),
                "binary_sha256":{k:run.sha256(p) for k,p in zip(("timing","allocations"),binaries)},
                "build_env":{k:v for k,v in os.environ.items() if k in ("RUSTFLAGS","CARGO_ENCODED_RUSTFLAGS") or k.startswith(("CARGO_PROFILE_RELEASE_", "CARGO_TARGET_"))}})
    frozen = json.loads((HERE / "model_fixtures.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as staging:
        path = Path(staging) / "models.jsonl"
        with path.open("w") as stream:
            for name, expected in frozen.items():
                (timing, peak), (counts, _) = [run.observe([binary, name]) for binary in binaries]
                for record in (timing, counts):
                    if (record["workload"], record["schema"], record["backend"]) != (name, 1, "pricing-native"):
                        raise ValueError(f"invalid record for {name}")
                    check({k:v for k,v in record["result"].items() if k != "phases"}, expected, name)
                times, allocations = timing["result"]["phases"], counts["result"]["phases"]
                if times.keys() != allocations.keys() or any(
                    len(t["samples_ns"]) != 11 or t["units"] != allocations[k]["units"] for k,t in times.items()
                ):
                    raise ValueError(f"inconsistent samples for {name}")
                timing.update({"environment":env,"peak_rss_bytes":peak,"allocation_phases":allocations})
                print(json.dumps(timing,allow_nan=False,separators=(",",":")),file=stream)
        os.replace(path,args.output)


if __name__ == "__main__":
    main()
