"""Run a bounded supplemental allocator experiment; preserve first-stop evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.smoke_test_wheel import run_test_process


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--tasks", type=int, default=256)
    parser.add_argument("--idle-seconds", type=int, default=10)
    parser.add_argument("--direct", action="store_true", help="Local diagnostic validation only; CI retains LLDB first-stop capture.")
    args = parser.parse_args()
    out = args.report_directory.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert 1 <= args.workers <= 16 and 1 <= args.seconds <= 600
    assert 1 <= args.tasks <= 512 and 1 <= args.idle_seconds <= 30
    forbidden = ("MallocNanoZone", "MallocScribble", "MallocStackLogging", "MallocGuardEdges", "DYLD_INSERT_LIBRARIES")
    assert not any(name in os.environ for name in forbidden), "Allocator or injection override is set"
    binary = out / "allocator-dispatch-probe"
    source = ROOT / "scripts/diagnostics/allocator_dispatch_probe.c"
    assert binary.is_file()
    assert not (out / "environment.json").exists(), "Do not silently retry or overwrite an experiment"
    environment = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "host": platform.platform(),
        "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
        "source_sha256": sha(source),
        "binary_sha256": sha(binary),
        "binary_uuids": subprocess.check_output(["xcrun", "dwarfdump", "--uuid", str(binary)], text=True),
        "linked_libraries": subprocess.check_output(["otool", "-L", str(binary)], text=True),
        "compiler": subprocess.check_output(["xcrun", "clang", "--version"], text=True),
        "sdk": subprocess.check_output(["xcrun", "--show-sdk-version"], text=True).strip(),
        "workers": args.workers, "seconds": args.seconds, "tasks_per_burst": args.tasks,
        "idle_seconds": args.idle_seconds, "debugger": None if args.direct else "lldb",
        "scope": "Standalone valid C allocation and libdispatch activity; the target imports neither pricing, Python nor Rust. A negative run cannot establish repair or waive any production failure.",
    }
    if not args.direct:
        dwarf = out / "allocator-dispatch-probe.dSYM/Contents/Resources/DWARF/allocator-dispatch-probe"
        assert dwarf.is_file(), "Missing retained debug symbols"
        uuids = subprocess.check_output(["xcrun", "dwarfdump", "--uuid", str(dwarf)], text=True)
        binary_uuid = re.search(r"UUID: ([A-F0-9-]+) [(]arm64[)]", environment["binary_uuids"])
        dwarf_uuid = re.search(r"UUID: ([A-F0-9-]+) [(]arm64[)]", uuids)
        assert binary_uuid and dwarf_uuid and binary_uuid[1] == dwarf_uuid[1]
        info = subprocess.check_output(["xcrun", "dwarfdump", "--debug-info", str(dwarf)], text=True)
        assert "DW_TAG_compile_unit" in info and "allocator_dispatch_probe.c" in info
        environment["debug_symbols"] = {"uuid": dwarf_uuid[1], "sha256": sha(dwarf), "compile_unit_verified": True}
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (out / source.name).write_bytes(source.read_bytes())
    command = [str(binary), str(args.workers), str(args.seconds), str(args.tasks),
               str(args.idle_seconds), str(out / "events.jsonl")]
    failure = run_test_process(command, "diagnostic", "allocator-dispatch", out, None if args.direct else "lldb")
    assert sha(source) == environment["source_sha256"] and sha(binary) == environment["binary_sha256"]
    if failure:
        raise RuntimeError(failure)
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    first, last = events[0], events[-1]
    bursts = events[1:-1]
    assert first == {"event": "start", "workers": args.workers, "seconds": args.seconds,
                     "tasks_per_burst": args.tasks, "idle_seconds": args.idle_seconds,
                     "live_slots_per_worker": 256}
    assert last["event"] == "completed" and len(bursts) == last["bursts"] > 0
    for index, event in enumerate(bursts, 1):
        assert event["event"] == "burst" and event["burst"] == index
        assert event["tasks_completed"] == index * args.tasks
    assert last["tasks_completed"] == len(bursts) * args.tasks
    assert len(last["worker_operations"]) == args.workers and all(type(n) is int and n > 0 for n in last["worker_operations"])
    assert last["elapsed_seconds"] >= args.seconds
    result = {"status": "completed_without_reproduction", "counts_as_production_acceptance": False,
              "source_sha256": environment["source_sha256"], "binary_sha256": environment["binary_sha256"],
              "workers": args.workers, "bursts": len(bursts), "dispatch_tasks": last["tasks_completed"],
              "worker_operations": last["worker_operations"], "elapsed_seconds": last["elapsed_seconds"],
              "events_sha256": sha(out / "events.jsonl"), "production_repair_proven": False}
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
