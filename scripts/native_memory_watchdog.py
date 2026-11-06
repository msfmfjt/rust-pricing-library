"""Observe an unchanged diagnostic command and retain incomplete outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class DiagnosticCancelled(Exception):
    """Cancellation must not be mistaken for a recoverable observer error."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(directory: Path, record: dict) -> None:
    temporary = directory / "supervisor.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "supervisor.json")


def observe(pid: int, directory: Path, index: int, sample: bool) -> dict:
    # Do not inject the target's optional sanitizer into the observer tools.
    env = {key: value for key, value in os.environ.items()
           if key not in ("DYLD_INSERT_LIBRARIES", "ASAN_OPTIONS", "__PYVENV_LAUNCHER__")}
    output = subprocess.check_output(
        ["/bin/ps", "-axo", "pid=,ppid=,pcpu=,rss=,etime=,comm="],
        env=env, text=True, timeout=10)
    processes = {}
    for line in output.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6:
            child, parent, cpu, rss, elapsed, executable = fields
            processes[int(child)] = dict(pid=int(child), ppid=int(parent),
                                         cpu_percent=float(cpu), rss_kib=int(rss),
                                         elapsed=elapsed, executable=executable)
    depths = {pid: 0}
    while True:
        children = {child: depths[row["ppid"]] + 1 for child, row in processes.items()
                    if child not in depths and row["ppid"] in depths}
        if not children:
            break
        depths.update(children)
    rows = [dict(processes[child], depth=depth) for child, depth in depths.items()
            if child in processes]
    record = dict(at=now(), status="observed", processes=rows)
    if sample and rows:
        target = max(rows, key=lambda row: (row["depth"], row["cpu_percent"]))
        path = directory / f"sample-{index:04d}-pid-{target['pid']}.txt"
        try:
            result = subprocess.run(
                ["/usr/bin/sample", str(target["pid"]), "1", "10", "-mayDie", "-file", str(path)],
                env=env, text=True, capture_output=True, timeout=20)
            captured = result.returncode == 0 and path.is_file() and path.stat().st_size > 0
            record["sample"] = dict(pid=target["pid"], returncode=result.returncode,
                                    status="captured" if captured else "unavailable",
                                    file=path.name if path.exists() else None,
                                    output=result.stdout + result.stderr)
        except (OSError, subprocess.SubprocessError) as error:
            record["sample"] = dict(pid=target["pid"], status="unavailable", error=str(error))
    return record


def signal_group(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin returns EPERM for a group containing only unreaped zombies.
        # Do not ignore a real denial while any group member is still alive.
        output = subprocess.check_output(["/bin/ps", "-axo", "pgid=,stat="],
                                         text=True, timeout=10)
        for line in output.splitlines():
            group, state = line.split()
            if int(group) == pid and not state.startswith("Z"):
                raise


def stop_group(process: subprocess.Popen) -> None:
    # Keep the leader unreaped until the final group signal, reserving its PID.
    signal_group(process.pid, signal.SIGTERM)
    time.sleep(1)
    signal_group(process.pid, signal.SIGKILL)
    process.wait()


def supervise(command: list[str], directory: Path, timeout_seconds: float = 16800,
              snapshot_seconds: float = 60, sample_seconds: float = 900) -> int:
    intervals = (timeout_seconds, snapshot_seconds, sample_seconds)
    if not command or any(not math.isfinite(value) or value <= 0 for value in intervals):
        raise ValueError("a command and positive observation/deadline intervals are required")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = dict(status="running", command=command, started_at=now(),
                  timeout_seconds=timeout_seconds, observation_errors=0, sampling_failures=0,
                  scope="Diagnostic observation only; partial execution is never a pass.")
    save(directory, record)
    process = None
    index, next_snapshot, next_sample = 0, 0.0, min(60.0, sample_seconds)
    try:
        process = subprocess.Popen(command, start_new_session=True)
        record["pid"] = process.pid
        save(directory, record)
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= timeout_seconds:
                record["status"] = "timed_out"
                break
            if elapsed >= next_snapshot:
                try:
                    snapshot = observe(process.pid, directory, index, elapsed >= next_sample)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    snapshot = dict(at=now(), status="observation_failed", error=str(error))
                    record["observation_errors"] += 1
                if snapshot.get("sample", {}).get("status") == "unavailable":
                    record["sampling_failures"] += 1
                with (directory / "observations.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(snapshot) + "\n")
                print(json.dumps(dict(watchdog_elapsed_seconds=round(elapsed, 3),
                                      observation=snapshot)), flush=True)
                index += 1
                next_snapshot = time.monotonic() - started + snapshot_seconds
                if elapsed >= next_sample:
                    next_sample = time.monotonic() - started + sample_seconds
                save(directory, record)
            time.sleep(min(1, snapshot_seconds, max(0, timeout_seconds - (time.monotonic() - started))))
        if record["status"] == "timed_out":
            stop_group(process)
            return 124
        record["status"] = "passed" if process.returncode == 0 else "failed"
        return process.returncode if process.returncode >= 0 else 128 - process.returncode
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, DiagnosticCancelled))
                      else "supervisor_error", error=str(error))
        if process is not None:
            stop_group(process)
        raise
    finally:
        record.update(finished_at=now(), elapsed_seconds=time.monotonic() - started,
                      child_returncode=process.returncode if process is not None else None,
                      observation_count=index)
        save(directory, record)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-directory", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=16800)
    parser.add_argument("--snapshot-seconds", type=float, default=60)
    parser.add_argument("--sample-seconds", type=float, default=900)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command

    def cancelled(signum, frame):
        raise DiagnosticCancelled(f"received {signal.Signals(signum).name}")

    previous = signal.signal(signal.SIGTERM, cancelled)
    try:
        return supervise(command, args.report_directory.resolve(), args.timeout_seconds,
                         args.snapshot_seconds, args.sample_seconds)
    except DiagnosticCancelled as error:
        print(str(error), file=sys.stderr)
        return 128 + signal.SIGTERM
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
