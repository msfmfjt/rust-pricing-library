"""Run Rust and installed-wheel benchmarks and record host/build metadata."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def main() -> None:
    output = Path("benchmark-results")
    output.mkdir(exist_ok=True)
    rust_report = output / "rust.json"
    local_vol_rust_report = output / "local-volatility-rust.json"
    python_report = output / "python.json"
    replay_report = output / "replay.json"
    local_vol_replay_report = output / "local-volatility-replay.json"

    command_peak_memory_bytes: dict[str, int] = {}

    command_peak_memory_bytes["rust_european_black_scholes"] = run_measured(
        [
            "cargo",
            "run",
            "--locked",
            "--release",
            "-p",
            "pricing",
            "--example",
            "benchmark_european_bs",
            "--",
            str(rust_report),
        ]
    )
    add_process_peak_memory(rust_report, command_peak_memory_bytes["rust_european_black_scholes"])
    command_peak_memory_bytes["rust_local_volatility_vegakt"] = run_measured(
        [
            "cargo",
            "run",
            "--locked",
            "--release",
            "-p",
            "pricing",
            "--example",
            "benchmark_local_vol",
            "--",
            str(local_vol_rust_report),
        ]
    )
    add_process_peak_memory(
        local_vol_rust_report, command_peak_memory_bytes["rust_local_volatility_vegakt"]
    )
    command_peak_memory_bytes["replay_european_black_scholes"] = run_measured(
        [
            "cargo",
            "run",
            "--locked",
            "--release",
            "-p",
            "pricing",
            "--example",
            "replay_european_bs",
            "--",
            str(replay_report),
        ]
    )
    run([sys.executable, "scripts/check_replay_fixture.py", str(replay_report)])
    if local_volatility_platform_name() is not None:
        command_peak_memory_bytes["replay_local_volatility"] = run_measured(
            [
                "cargo",
                "run",
                "--locked",
                "--release",
                "-p",
                "pricing",
                "--example",
                "replay_local_vol",
                "--",
                str(local_vol_replay_report),
            ]
        )
        if local_volatility_fixture_exists():
            run([sys.executable, "scripts/check_replay_fixture.py", str(local_vol_replay_report)])
        else:
            print(
                f"generated unfrozen Local Volatility replay evidence at {local_vol_replay_report}"
            )
    wheel_python = Path(
        os.environ.get(
            "WHEEL_SMOKE_PYTHON",
            str(
                Path(".wheel-smoke-venv")
                / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            ),
        )
    )
    command_peak_memory_bytes["python_european_black_scholes"] = run_measured(
        [str(wheel_python), "benchmarks/python_european_bs.py", str(python_report)]
    )
    add_process_peak_memory(
        python_report, command_peak_memory_bytes["python_european_black_scholes"]
    )

    rustc = capture(["rustc", "-vV"])
    cargo = capture(["cargo", "-V"])
    metadata = {
        "schema_version": 1,
        "git_sha": os.environ.get("GITHUB_SHA"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": sys.version,
        "python_abi": sys.implementation.cache_tag,
        "rustc": rustc,
        "cargo": cargo,
        "target_triple": rustc_host(rustc),
        "enabled_features": {
            "rust_benchmarks": [],
            "python_wheel": ["pricing-python/extension-module"],
        },
        "cargo_lock_sha256": file_sha256(Path("Cargo.lock")),
        "peak_memory_bytes": max(command_peak_memory_bytes.values()),
        "command_peak_memory_bytes": command_peak_memory_bytes,
        "allocation_count": None,
        "unavailable_metrics": [
            "Allocation counting requires an instrumented allocator.",
        ],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run([sys.executable, "scripts/check_benchmark_reports.py", str(output)])


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def run_measured(command: list[str]) -> int:
    process = subprocess.Popen(command)
    peak = 0
    while True:
        peak = max(peak, process_peak_rss_bytes(process.pid))
        if process.poll() is not None:
            peak = max(peak, process_peak_rss_bytes(process.pid))
            break
        time.sleep(0.05)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    if peak <= 0:
        raise RuntimeError(f"could not observe peak RSS for command: {command}")
    return peak


def process_peak_rss_bytes(pid: int) -> int:
    system = platform.system()
    if system == "Linux":
        return linux_process_peak_rss_bytes(pid)
    if system == "Darwin":
        return darwin_process_rss_bytes(pid)
    if system == "Windows":
        return windows_process_peak_rss_bytes(pid)
    return 0


def linux_process_peak_rss_bytes(pid: int) -> int:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return 0
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:") or line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    return 0


def darwin_process_rss_bytes(pid: int) -> int:
    import ctypes

    class ProcTaskInfo(ctypes.Structure):
        _fields_ = [
            ("pti_virtual_size", ctypes.c_uint64),
            ("pti_resident_size", ctypes.c_uint64),
            ("pti_total_user", ctypes.c_uint64),
            ("pti_total_system", ctypes.c_uint64),
            ("pti_threads_user", ctypes.c_uint64),
            ("pti_threads_system", ctypes.c_uint64),
            ("pti_policy", ctypes.c_int32),
            ("pti_faults", ctypes.c_int32),
            ("pti_pageins", ctypes.c_int32),
            ("pti_cow_faults", ctypes.c_int32),
            ("pti_messages_sent", ctypes.c_int32),
            ("pti_messages_received", ctypes.c_int32),
            ("pti_syscalls_mach", ctypes.c_int32),
            ("pti_syscalls_unix", ctypes.c_int32),
            ("pti_csw", ctypes.c_int32),
            ("pti_threadnum", ctypes.c_int32),
            ("pti_numrunning", ctypes.c_int32),
            ("pti_priority", ctypes.c_int32),
        ]

    proc_pidtaskinfo = 4
    info = ProcTaskInfo()
    result = ctypes.CDLL("/usr/lib/libproc.dylib").proc_pidinfo(
        pid, proc_pidtaskinfo, 0, ctypes.byref(info), ctypes.sizeof(info)
    )
    if result <= 0:
        return 0
    return int(info.pti_resident_size)


def windows_process_peak_rss_bytes(pid: int) -> int:
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    process_query_limited_information = 0x1000
    process_vm_read = 0x0010
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information | process_vm_read, False, pid
    )
    if not handle:
        return 0
    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if not ok:
            return 0
        return int(max(counters.PeakWorkingSetSize, counters.WorkingSetSize))
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def add_process_peak_memory(path: Path, peak_memory_bytes: int) -> None:
    document = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_json_constant,
    )
    if not isinstance(document, dict):
        raise RuntimeError(f"{path}: top-level JSON value must be an object")
    capabilities = document.setdefault("capabilities", {})
    if not isinstance(capabilities, dict):
        raise RuntimeError(f"{path}: capabilities must be an object")
    capabilities["peak_memory_available_in_process"] = True
    document["process_peak_memory_bytes"] = peak_memory_bytes
    path.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


def local_volatility_fixture_exists() -> bool:
    platform_name = local_volatility_platform_name()
    if platform_name is None:
        return False
    return (Path("fixtures/replay") / f"local_volatility-{platform_name}.json").is_file()


def local_volatility_platform_name() -> str | None:
    return {
        ("Darwin", "arm64"): "macos-aarch64",
        ("Windows", "AMD64"): "windows-x86_64",
        ("Windows", "x86_64"): "windows-x86_64",
    }.get((platform.system(), platform.machine()))


def capture(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def rustc_host(rustc_version: str) -> str:
    for line in rustc_version.splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ")
    raise RuntimeError("rustc -vV output did not include host target")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
