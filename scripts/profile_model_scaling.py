"""S7 coupled-HW scaling: independent axes, cold/warm AAD and DHAT stacks.

Both disposable checkouts receive the identical public-API example. Native
timing/RSS use seven alternating-order pairs. DHAT is a separate measurement.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess

from compare_model_boundaries import EXAMPLE, build, command, digest, stats


# workers, steps, particles, independent units, assets. Change one axis at a time.
BASE = (1, 8, 128, 512, 2)
AXES = {"workers": (0, [2, 4]), "steps": (1, [16, 32]),
        "particles": (2, [256, 512]), "units": (3, [2048, 8192]),
        "assets": (4, [3, 4])}
CASE = "local_correlation_hw"


def workloads():
    yield "base", BASE
    for axis, (index, values) in AXES.items():
        for value in values:
            settings = list(BASE)
            settings[index] = value
            yield f"{axis}-{value}", tuple(settings)


def invocation(binary, phase, settings, warmups, repeats):
    return [str(binary), CASE, phase, *map(str, settings[:4]),
            str(warmups), str(repeats), str(settings[4])]


def native(binary, phase, settings, output):
    rss = output / "native.rss"
    result = json.loads(command(["/usr/bin/time", "-f", "%M", "-o", str(rss),
                                 *invocation(binary, phase, settings, 1, 3)]))
    result["process_peak_rss_bytes"] = int(rss.read_text().strip()) * 1024
    rss.unlink()
    assert len(result["operation_ns"]) == 3 and min(result["operation_ns"]) > 0
    return result


def pairs(binaries, phase, settings, output, rounds, start=0):
    result = []
    for i in range(start, start + rounds):
        order = ["baseline", "candidate"] if i % 2 == 0 else ["candidate", "baseline"]
        row = {label: native(binaries[label], phase, settings, output) for label in order}
        assert row["baseline"]["checksum"] == row["candidate"]["checksum"], (phase, settings)
        row["order"] = order
        row["ratio"] = statistics.median(row["candidate"]["operation_ns"]) / statistics.median(row["baseline"]["operation_ns"])
        result.append(row)
    assert len({r[label]["checksum"] for r in result for label in binaries}) == 1
    return result


def profile(binary, phase, settings, output, label, name, repeats):
    prefix = output / f"{label}-{name}-{phase}-{repeats}"
    log = prefix.with_suffix(".dhat.log")
    path = prefix.with_suffix(".dhat.json")
    result = json.loads(command([
        "valgrind", "--tool=dhat", "--num-callers=16", f"--dhat-out-file={path}",
        f"--log-file={log}", *invocation(binary, phase, settings, 0, repeats)]))
    data = json.loads(path.read_text())
    points = data.get("pps", data.get("aps"))
    if not points or not data.get("ftbl"):
        raise RuntimeError(f"unsupported DHAT profile: {path}")
    totals = {"allocated_bytes": sum(p["tb"] for p in points),
              "allocation_blocks": sum(p["tbk"] for p in points),
              "peak_heap_bytes": sum(p["gb"] for p in points),
              "blocks_at_peak": sum(p["gbk"] for p in points)}
    for heading, keys in [("Total:", ("allocated_bytes", "allocation_blocks")),
                          ("At t-gmax:", ("peak_heap_bytes", "blocks_at_peak"))]:
        match = re.search(re.escape(heading) + r"\s*([\d,]+) bytes in ([\d,]+) blocks", log.read_text())
        if not match or [int(v.replace(",", "")) for v in match.groups()] != [totals[k] for k in keys]:
            raise RuntimeError(f"DHAT JSON/log totals disagree: {path}")
    def leaders(metric):
        return [{"allocated_bytes": p["tb"], "allocation_blocks": p["tbk"],
                 "peak_heap_bytes": p["gb"], "blocks_at_peak": p["gbk"],
                 "frames": [data["ftbl"][i] for i in p["fs"]]}
                for p in sorted(points, key=lambda p: p[metric], reverse=True)[:12]]
    return dict(totals, checksum=result["checksum"], profile=path.name,
                profile_sha256=digest(path), repeats=repeats,
                allocation_leaders=leaders("tbk"), peak_heap_leaders=leaders("gb"))


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
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Optimized code with line information for allocation-site attribution.
    os.environ["CARGO_PROFILE_RELEASE_DEBUG"] = "1"
    report = {"schema_version": 1, "baseline": {}, "candidate": {}, "native": [], "heap": [],
              "metadata": {"platform": platform.platform(), "cpu": command(["lscpu"]),
                           "rustc": command(["rustc", "-vV"]), "cargo": command(["cargo", "-V"]),
                           "valgrind": command(["valgrind", "--version"]),
                           "environment": {k: v for k, v in os.environ.items() if k in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"] or k.startswith("CARGO_PROFILE_")},
                           "runner_run_id": os.environ.get("GITHUB_RUN_ID"),
                           "example_sha256": digest(roots["candidate"] / EXAMPLE),
                           "rounds": args.rounds, "warmups_per_process": 1,
                           "timed_operations_per_process": 3,
                           "profile": "release; thin LTO; codegen-units=1; debug=1",
                           "settings_order": ["workers", "steps", "particles", "independent_units", "assets"],
                           "timing_scope": "compile: plan build; aad_cold: first AAD on each newly built plan, build excluded; aad: reuse one plan with first AAD excluded as warmup",
                           "heap_scope": "whole DHAT process; one plan; no warmup; one or two evaluations; separate from native timing"}}
    try:
        for label, root in roots.items():
            report[label] = {"commit": command(["git", "rev-parse", "HEAD"], root),
                             "tree": command(["git", "rev-parse", "HEAD^{tree}"], root),
                             "cargo_lock_sha256": digest(root / "Cargo.lock")}
        assert report["baseline"]["cargo_lock_sha256"] == report["candidate"]["cargo_lock_sha256"]
        assert (roots["baseline"] / "rust-toolchain.toml").read_bytes() == (roots["candidate"] / "rust-toolchain.toml").read_bytes()
        shutil.copy2(roots["candidate"] / EXAMPLE, roots["baseline"] / EXAMPLE)
        for root in roots.values():
            subprocess.run(["cargo", "fetch", "--locked"], cwd=root, check=True)
        binaries = {}
        for label, root in roots.items():
            binary, meta = build(root, out, label)
            binaries[label] = binary
            report[label].update(meta)
        for name, settings in workloads():
            checksums = {}
            for phase in ["compile", "price", "aad_cold", "aad"]:
                initial = pairs(binaries, phase, settings, out, args.rounds)
                ratio = statistics.median(p["ratio"] for p in initial)
                extra = pairs(binaries, phase, settings, out, args.rounds, args.rounds) if ratio > 1.10 else []
                row = {"name": name, "settings": settings, "phase": phase, "pairs": initial,
                       "confirmation_pairs": extra, "paired_time_ratio": stats([p["ratio"] for p in initial]),
                       "confirmation_time_ratio": stats([p["ratio"] for p in extra]) if extra else None}
                checksums[phase] = initial[0]["baseline"]["checksum"]
                for label in binaries:
                    row[label] = {"operation_ns": stats([statistics.median(p[label]["operation_ns"]) for p in initial]),
                                  "process_peak_rss_bytes": stats([p[label]["process_peak_rss_bytes"] for p in initial])}
                report["native"].append(row)
                print(f"{name}/{phase}: paired median {ratio:.4f}; exact replay", flush=True)
            assert checksums["aad"] == checksums["aad_cold"], f"cached AAD mismatch: {name}"
        # Independent axes most likely to stress calibration/trace storage.
        for name, settings in workloads():
            if name not in ["base", "steps-32", "particles-512", "assets-4"]:
                continue
            phases = [("compile", 1), ("price", 1), ("aad", 1), ("aad", 2)] if name == "base" else [("aad", 1)]
            first_aad = None
            for phase, repeats in phases:
                values = {label: profile(binary, phase, settings, out, label, name, repeats)
                          for label, binary in binaries.items()}
                assert values["baseline"]["checksum"] == values["candidate"]["checksum"], (name, phase, repeats)
                if phase == "aad":
                    checksum = values["baseline"]["checksum"]
                    assert first_aad is None or checksum == first_aad, f"DHAT cached AAD mismatch: {name}"
                    first_aad = checksum
                report["heap"].append({"name": name, "settings": settings, "phase": phase,
                                       "repeats": repeats, **values})
                print(f"{name}/{phase}/{repeats}: heap captured; exact replay", flush=True)
        report["complete"] = True
    except Exception as error:
        report["failure"] = repr(error)
        raise
    finally:
        (out / "scaling.json").write_text(json.dumps(report, indent=2) + "\n")
        print("MODEL_SCALING_REPORT " + json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
