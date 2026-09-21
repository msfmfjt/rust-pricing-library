"""Paired Linux release measurements; no production allocator instrumentation.

The identical public-API example is overlaid onto the baseline checkout. Timing
is native; DHAT is a separate one-operation process, including plan setup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import time

EXAMPLE = Path("crates/pricing/examples/benchmark_model_boundaries.rs")
CASES = ["bergomi_1f", "bergomi_2f", "rough", "hw_1f", "hw_2f", "hw_rough",
         "multi_hw_mixed", "local_correlation_hw"]


def command(args, cwd=None, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    median = statistics.median(values)
    return {"min": min(values), "median": median, "max": max(values),
            "mad": statistics.median(abs(v - median) for v in values)}


def build(root, output, label):
    target = output / f"target-{label}"
    if target.exists():
        raise RuntimeError(f"clean target required: {target}")
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    # Fetch before either timed build; both builds still compile dependencies.
    start = time.perf_counter()
    subprocess.run(["cargo", "build", "--locked", "--release", "-p", "pricing",
                    "--example", "benchmark_model_boundaries"], cwd=root, env=env, check=True)
    elapsed = time.perf_counter() - start
    binary = target / "release/examples/benchmark_model_boundaries"
    return binary, {"clean_build_seconds": elapsed, "executable_bytes": binary.stat().st_size,
                    "executable_sha256": digest(binary)}


def native(binary, case, phase, settings, output, tag):
    rss = output / f"{tag}.rss"
    args = [str(binary), case, phase, *map(str, settings), "1", "3"]
    text = command(["/usr/bin/time", "-f", "%M", "-o", str(rss), *args])
    data = json.loads(text)
    data["process_peak_rss_bytes"] = int(rss.read_text().strip()) * 1024
    rss.unlink()
    assert len(data["operation_ns"]) == 3 and min(data["operation_ns"]) > 0
    return data


def paired(binaries, case, phase, settings, output, rounds, start=0):
    rows = []
    for i in range(start, start + rounds):
        pair = {}
        order = ["baseline", "candidate"] if i % 2 == 0 else ["candidate", "baseline"]
        for label in order:
            pair[label] = native(binaries[label], case, phase, settings, output, f"{label}-{i}")
        if pair["baseline"]["checksum"] != pair["candidate"]["checksum"]:
            raise AssertionError(f"exact replay mismatch: {case}/{phase}/{settings}")
        pair["order"] = order
        pair["ratio"] = statistics.median(pair["candidate"]["operation_ns"]) / statistics.median(pair["baseline"]["operation_ns"])
        rows.append(pair)
    checksums = {p[label]["checksum"] for p in rows for label in binaries}
    assert len(checksums) == 1, f"non-reproducible result: {case}/{phase}"
    return rows


def heap(binary, case, phase, settings, output, label):
    prefix = output / f"{label}-{case}-{phase}"
    log = prefix.with_suffix(".dhat.log")
    profile = prefix.with_suffix(".dhat.json")
    args = ["valgrind", "--tool=dhat", "--num-callers=4", f"--dhat-out-file={profile}",
            f"--log-file={log}", str(binary), case, phase, *map(str, settings), "0", "1"]
    result = json.loads(command(args))
    text = log.read_text()
    def counts(name):
        match = re.search(re.escape(name) + r"\s*([\d,]+) bytes in ([\d,]+) blocks", text)
        if not match:
            raise RuntimeError(f"missing DHAT {name}: {log}")
        return [int(v.replace(",", "")) for v in match.groups()]
    total_bytes, allocations = counts("Total:")
    peak_bytes, peak_blocks = counts("At t-gmax:")
    return {"allocated_bytes": total_bytes, "allocation_blocks": allocations,
            "peak_heap_bytes": peak_bytes, "blocks_at_peak": peak_blocks,
            "checksum": result["checksum"], "profile": profile.name,
            "scope": "entire process: inputs, one plan, one operation, output; no warmup"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if platform.system() != "Linux" or args.rounds < 3:
        raise SystemExit("Linux and at least three paired rounds required")
    roots = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "baseline": {}, "candidate": {}, "measurements": [],
              "metadata": {"platform": platform.platform(), "cpu": command(["lscpu"]),
                           "rustc": command(["rustc", "-vV"]), "cargo": command(["cargo", "-V"]),
                           "valgrind": command(["valgrind", "--version"]),
                           "environment": {k: v for k, v in os.environ.items() if k in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"] or k.startswith("CARGO_PROFILE_")},
                           "runner_run_id": os.environ.get("GITHUB_RUN_ID"),
                           "example_sha256": digest(roots["candidate"] / EXAMPLE),
                           "warmups_per_process": 1, "timed_operations_per_process": 3,
                           "rounds": args.rounds, "profile": "release; codegen-units=1; lto=thin",
                           "timing_scope": "compile: plan construction; other phases: existing plan evaluation; destruction/hash/output excluded",
                           "rss_scope": "whole native process, including inputs, setup and warmup",
                           "allocation_scope": "separate DHAT whole-process run; no native-time comparison to DHAT"}}
    try:
        for label, root in roots.items():
            report[label].update({"commit": command(["git", "rev-parse", "HEAD"], root),
                                  "tree": command(["git", "rev-parse", "HEAD^{tree}"], root),
                                  "cargo_lock_sha256": digest(root / "Cargo.lock")})
        assert report["baseline"]["cargo_lock_sha256"] == report["candidate"]["cargo_lock_sha256"]
        assert (roots["baseline"] / "rust-toolchain.toml").read_bytes() == (roots["candidate"] / "rust-toolchain.toml").read_bytes()
        shutil.copy2(roots["candidate"] / EXAMPLE, roots["baseline"] / EXAMPLE)
        for root in roots.values():
            subprocess.run(["cargo", "fetch", "--locked"], cwd=root, check=True)
        binaries = {}
        for label, root in roots.items():
            binary, metadata = build(root, output, label)
            binaries[label] = binary
            report[label].update(metadata)
        # S6 representative panel. The larger case changes particles, time steps
        # and workers together; this is not an isolated scaling attribution.
        for settings in [(1, 8, 128, 512), (2, 16, 256, 1024)]:
            for case in CASES:
                phases = ["compile", "price", "aad"]
                if case.startswith("hw_"):
                    phases.append("aad_vegakt")
                for phase in phases:
                    pairs = paired(binaries, case, phase, settings, output, args.rounds)
                    first_ratio = statistics.median(p["ratio"] for p in pairs)
                    # Investigate a concrete candidate slowdown with fresh,
                    # opposite-order pairs; this is evidence, not a CI speed gate.
                    extra = []
                    if first_ratio > 1.10:
                        extra = paired(binaries, case, phase, settings, output, args.rounds, args.rounds)
                    row = {"case": case, "phase": phase, "settings": settings,
                           "pairs": pairs, "confirmation_pairs": extra,
                           "paired_time_ratio": stats([p["ratio"] for p in pairs]),
                           "confirmation_time_ratio": stats([p["ratio"] for p in extra]) if extra else None,
                           "native_checksum": pairs[0]["baseline"]["checksum"]}
                    for label in binaries:
                        row[label] = {"operation_ns": stats([statistics.median(p[label]["operation_ns"]) for p in pairs]),
                                      "process_peak_rss_bytes": stats([p[label]["process_peak_rss_bytes"] for p in pairs])}
                    # Profile the same small workload with one operation. DHAT
                    # changes CPU capabilities/timing; compare its pair directly.
                    if settings[0] == 1:
                        row["heap"] = {label: heap(binary, case, phase, settings, output, label)
                                       for label, binary in binaries.items()}
                        assert row["heap"]["baseline"]["checksum"] == row["heap"]["candidate"]["checksum"], f"DHAT replay: {case}/{phase}"
                    report["measurements"].append(row)
                    print(f"{case}/{phase}/{settings}: time ratio {first_ratio:.3f}, exact replay passed", flush=True)
        report["complete"] = True
    finally:
        (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    # The full paired observations remain downloadable and available from job
    # logs, so a durable source-linked report can be recorded after the run.
    print("MODEL_BOUNDARY_REPORT " + json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
