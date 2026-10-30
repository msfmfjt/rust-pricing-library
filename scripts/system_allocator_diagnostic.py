"""Isolate system malloc/free concurrency; never waive pricing acceptance gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

from smoke_test_wheel import run_test_process

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/diagnostics/system_allocator_probe.c"


def validate_events(path: Path, workers: int, rounds: int, seconds: int) -> dict:
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if len(events) != rounds * 4 + 2:
        raise ValueError("incomplete system allocator event sequence")
    first, last = events[0], events[-1]
    if (first.get("event") != "start" or first.get("workers") != workers
            or first.get("rounds") != rounds or first.get("seconds") != seconds
            or last != {"event": "completed", "cases": rounds * 2}):
        raise ValueError("system allocator configuration/completion mismatch")
    total = 0
    for index in range(rounds * 2):
        start, case = events[1 + index * 2:3 + index * 2]
        counts = case.get("iterations", [])
        if (start != {"event": "case-start", "round": index // 2, "bytes": (8, 64)[index % 2]}
                or case.get("event") != "case" or case.get("round") != index // 2
                or case.get("bytes") != (8, 64)[index % 2]
                or not isinstance(case.get("elapsed_seconds"), (int, float))
                or not seconds - 0.001 <= case["elapsed_seconds"] < 1000
                or len(counts) != workers
                or any(type(count) is not int or count <= 0 for count in counts)):
            raise ValueError("incomplete or inconsistent system allocator case")
        total += sum(counts)
    return {"completed_cases": rounds * 2, "allocation_free_pairs": total,
            "nano_mode": first.get("nano_mode"),
            "secure_allocator": first.get("secure_allocator")}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--seconds", type=int, default=10)
    args = parser.parse_args(argv)
    if (not 1 <= args.workers <= 64 or not 1 <= args.rounds <= 64
            or not 1 <= args.seconds <= 30 or args.rounds * 2 * args.seconds > 900):
        parser.error("workers/rounds must be 1..64, seconds 1..30, and total duration <=900s")
    directory = args.report_directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "host": platform.platform(), "workers": args.workers, "rounds": args.rounds,
              "seconds_per_case": args.seconds, "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "limitations": "Standalone system allocator diagnostic; no pricing, Python, Rust or sanitizer in the child. A negative result does not prove repair or identify the original corrupting operation."}

    def save():
        (directory / "system-allocator.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        if platform.system() != "Darwin":
            raise RuntimeError("this diagnostic targets macOS")
        overrides = {k: v for k, v in os.environ.items()
                     if k.startswith("Malloc") or k in ("DYLD_INSERT_LIBRARIES", "ASAN_OPTIONS")}
        if overrides:
            record["allocator_overrides"] = overrides
            raise RuntimeError("system allocator diagnostic requires no allocator overrides")
        record["source_head"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        record["compiler"] = subprocess.check_output(["xcrun", "clang", "--version"], text=True).strip()
        record["sdk"] = subprocess.check_output(["xcrun", "--show-sdk-version"], text=True).strip()
        binary = directory / "system-allocator-probe"
        compile_command = ["xcrun", "clang", "-std=c11", "-O2", "-g", "-Wall", "-Wextra", "-Werror",
                           "-fno-builtin-malloc", "-fno-builtin-free", "-pthread", str(SOURCE), "-o", str(binary)]
        record["compile_command"] = compile_command
        save()
        subprocess.run(compile_command, check=True, timeout=60)
        record["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        events = directory / "system-allocator-events.jsonl"
        events.unlink(missing_ok=True)
        command = [str(binary), str(args.workers), str(args.rounds), str(args.seconds), str(events)]
        failure = run_test_process(command, "diagnostic", "system-allocator-process", directory, "lldb")
        if failure:
            raise RuntimeError(failure)
        record.update(validate_events(events, args.workers, args.rounds, args.seconds))
        record["status"] = "completed_without_reproduction"
    except Exception as error:
        record.update(status="failed_or_incomplete", error=str(error))
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
