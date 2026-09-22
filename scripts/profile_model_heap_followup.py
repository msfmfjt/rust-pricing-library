"""Explain whole-process AAD heap peaks using every live allocation stack.

Read the original S7 pair, then repeat only the base/two-evaluation workload.
No production code or benchmark behavior is changed by this measurement.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess

from compare_model_boundaries import EXAMPLE, build, command, digest, stats
from profile_model_scaling import BASE, profile


ORIGINAL_RUN = 35708535272
ORIGINAL_BASELINE = "ea759bf5d73b06e4c480665e59c7daa37e6667be"
ORIGINAL_CANDIDATE = "5886ac6834676ea6c10cd89aa087a58c81625ae6"
RETAINED = Path("design/validation/local-correlation-heap-followup-2026-09-22.json")
METRICS = {"allocated_bytes": "tb", "allocation_blocks": "tbk",
           "peak_heap_bytes": "gb", "blocks_at_peak": "gbk",
           "end_heap_bytes": "eb", "blocks_at_end": "ebk"}


def display_frame(frame):
    """Omit load addresses and executable directories; retain symbol/source."""
    frame = re.sub(r"^0x[0-9A-Fa-f]+: ", "", frame)
    return re.sub(r"\(in .*/([^/]+)\)", r"(in \1)", frame)


def logical_frame(frame):
    # Compare groups of call stacks, not individual allocation-site identities.
    # Line offsets and compiler-generated impl numbering change after extraction.
    frame = re.sub(r"(\.rs):\d+(?::\d+)?\)", r"\1)", display_frame(frame))
    return re.sub(r"impl#\d+", "impl#", frame)


def summarize_profile(path, log, frames):
    data = json.loads(path.read_text())
    points = data.get("pps", data.get("aps"))
    if not points or not data.get("ftbl"):
        raise ValueError(f"unsupported DHAT profile: {path}")
    totals = {name: sum(p.get(key, 0) for p in points)
              for name, key in METRICS.items()}
    for label, keys in [("Total:", ("allocated_bytes", "allocation_blocks")),
                        ("At t-gmax:", ("peak_heap_bytes", "blocks_at_peak")),
                        ("At t-end:", ("end_heap_bytes", "blocks_at_end"))]:
        match = re.search(re.escape(label) + r"\s*([\d,]+) bytes in ([\d,]+) blocks",
                          log.read_text())
        if not match or [int(v.replace(",", "")) for v in match.groups()] != [totals[k] for k in keys]:
            raise ValueError(f"DHAT JSON/log totals disagree: {path}: {label}")
    peak = []
    for point in points:
        if not point.get("gb", 0) and not point.get("gbk", 0):
            continue
        stack = []
        for index in point["fs"]:
            frame = display_frame(data["ftbl"][index])
            stack.append(frames.setdefault(frame, len(frames)))
        peak.append({"bytes": point["gb"], "blocks": point["gbk"], "frames": stack})
    assert sum(p["bytes"] for p in peak) == totals["peak_heap_bytes"]
    assert sum(p["blocks"] for p in peak) == totals["blocks_at_peak"]
    return dict(totals, peak_points=peak, profile=path.name, profile_sha256=digest(path))


def compare_peak(baseline, candidate, frames):
    texts = list(frames)
    grouped = {}
    for label, value in [("baseline", baseline), ("candidate", candidate)]:
        groups = defaultdict(lambda: {"bytes": 0, "blocks": 0, "points": []})
        for i, point in enumerate(value["peak_points"]):
            key = tuple(logical_frame(texts[f]) for f in point["frames"])
            group = groups[key]
            group["bytes"] += point["bytes"]
            group["blocks"] += point["blocks"]
            group["points"].append(i)
        grouped[label] = groups
    differences = []
    for key in sorted(grouped["baseline"].keys() | grouped["candidate"].keys()):
        b, c = (grouped[label][key] for label in ("baseline", "candidate"))
        if (b["bytes"], b["blocks"]) != (c["bytes"], c["blocks"]):
            differences.append({"baseline": b, "candidate": c,
                                "bytes_delta": c["bytes"] - b["bytes"],
                                "blocks_delta": c["blocks"] - b["blocks"]})
    byte_delta = candidate["peak_heap_bytes"] - baseline["peak_heap_bytes"]
    block_delta = candidate["blocks_at_peak"] - baseline["blocks_at_peak"]
    assert sum(d["bytes_delta"] for d in differences) == byte_delta
    assert sum(d["blocks_delta"] for d in differences) == block_delta
    return {"bytes_delta": byte_delta, "blocks_delta": block_delta,
            "groups": sorted(differences, key=lambda d: abs(d["bytes_delta"]), reverse=True)}


def original_pair(directory, candidate, frames):
    retained = candidate / RETAINED
    if retained.is_file():
        previous = json.loads(retained.read_text())
        assert previous["original"]["run_id"] == ORIGINAL_RUN
        assert not frames
        frames.update({frame: i for i, frame in enumerate(previous["frames"])})
        result = previous["original"]
        result["comparison"] = compare_peak(result["baseline"], result["candidate"], frames)
        return result
    reports = list(directory.rglob("scaling.json"))
    if len(reports) != 1:
        raise ValueError("one original scaling.json is required")
    report = json.loads(reports[0].read_text())
    assert report["metadata"]["runner_run_id"] == str(ORIGINAL_RUN)
    assert report["baseline"]["commit"] == ORIGINAL_BASELINE
    assert report["candidate"]["commit"] == ORIGINAL_CANDIDATE
    row, = [r for r in report["heap"] if (r["name"], r["phase"], r["repeats"]) == ("base", "aad", 2)]
    result = {"run_id": ORIGINAL_RUN}
    for label in ("baseline", "candidate"):
        reference = row[label]
        paths = list(directory.rglob(reference["profile"]))
        if len(paths) != 1 or digest(paths[0]) != reference["profile_sha256"]:
            raise ValueError(f"original profile hash mismatch: {label}")
        value = summarize_profile(paths[0], paths[0].with_suffix(".log"), frames)
        for key in ("allocated_bytes", "allocation_blocks", "peak_heap_bytes", "blocks_at_peak"):
            assert value[key] == reference[key]
        value["checksum"] = reference["checksum"]
        value["source"] = report[label]
        result[label] = value
    assert result["baseline"]["checksum"] == result["candidate"]["checksum"]
    result["comparison"] = compare_peak(result["baseline"], result["candidate"], frames)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--original-directory", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if platform.system() != "Linux" or args.rounds < 3:
        raise SystemExit("Linux and at least three fresh-process pairs required")
    roots = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["CARGO_PROFILE_RELEASE_DEBUG"] = "1"
    frames = {}
    report = {"schema_version": 1, "baseline": {}, "candidate": {}, "pairs": [],
              "metadata": {"platform": platform.platform(), "cpu": command(["lscpu"]),
                           "rustc": command(["rustc", "-vV"]), "cargo": command(["cargo", "-V"]),
                           "valgrind": command(["valgrind", "--version"]),
                           "runner_run_id": os.environ.get("GITHUB_RUN_ID"),
                           "example_sha256": digest(roots["candidate"] / EXAMPLE),
                           "settings": list(BASE), "phase": "aad", "repeats": 2,
                           "rounds": args.rounds, "warmups": 0,
                           "profile": "release; thin LTO; codegen-units=1; debug=1",
                           "environment": {k: v for k, v in os.environ.items() if k in ["RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RAYON_NUM_THREADS"] or k.startswith("CARGO_PROFILE_")},
                           "scope": "whole DHAT process; one plan and two AAD evaluations; no timing inference",
                           "frame_storage": "symbol/source frames, with load addresses and executable directories omitted",
                           "comparison": "all live-at-global-peak points; group call stacks ignoring source line offsets and generated impl numbers; point indices retain original source locations"}}
    try:
        report["original"] = original_pair(args.original_directory, roots["candidate"], frames)
        print("Original peak bytes delta:", report["original"]["comparison"]["bytes_delta"], flush=True)
        for label, root in roots.items():
            report[label] = {"commit": command(["git", "rev-parse", "HEAD"], root),
                             "tree": command(["git", "rev-parse", "HEAD^{tree}"], root),
                             "cargo_lock_sha256": digest(root / "Cargo.lock")}
        assert report["baseline"]["commit"] == ORIGINAL_BASELINE
        assert report["baseline"]["cargo_lock_sha256"] == report["candidate"]["cargo_lock_sha256"]
        assert (roots["baseline"] / "rust-toolchain.toml").read_bytes() == (roots["candidate"] / "rust-toolchain.toml").read_bytes()
        shutil.copy2(roots["candidate"] / EXAMPLE, roots["baseline"] / EXAMPLE)
        binaries = {}
        for label, root in roots.items():
            subprocess.run(["cargo", "fetch", "--locked"], cwd=root, check=True)
        for label, root in roots.items():
            binaries[label], meta = build(root, out, label)
            report[label].update(meta)
        for index in range(args.rounds):
            order = ["baseline", "candidate"] if index % 2 == 0 else ["candidate", "baseline"]
            pair = {"order": order}
            for label in order:
                measured = profile(binaries[label], "aad", BASE, out, label, f"pair-{index}", 2)
                path = out / measured["profile"]
                value = summarize_profile(path, path.with_suffix(".log"), frames)
                value["checksum"] = measured["checksum"]
                assert value["checksum"] == report["original"][label]["checksum"]
                pair[label] = value
            pair["comparison"] = compare_peak(pair["baseline"], pair["candidate"], frames)
            report["pairs"].append(pair)
            print(f"Pair {index + 1}: peak delta {pair['comparison']['bytes_delta']} bytes; exact replay", flush=True)
        report["summary"] = {"paired_peak_bytes_delta": stats([p["comparison"]["bytes_delta"] for p in report["pairs"]])}
        for label in roots:
            report["summary"][label] = {metric: stats([p[label][metric] for p in report["pairs"]]) for metric in METRICS}
        report["complete"] = True
    except Exception as error:
        report["failure"] = repr(error)
        raise
    finally:
        report["frames"] = list(frames)
        (out / "heap-followup.json").write_text(json.dumps(report, indent=2) + "\n")
        print("MODEL_HEAP_FOLLOWUP_REPORT " + json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
