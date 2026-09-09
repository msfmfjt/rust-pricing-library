"""Run Rust and installed-wheel benchmarks and record host/build metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main() -> None:
    output = Path("benchmark-results")
    output.mkdir(exist_ok=True)
    rust_report = output / "rust.json"
    local_vol_rust_report = output / "local-volatility-rust.json"
    python_report = output / "python.json"
    replay_report = output / "replay.json"
    local_vol_replay_report = output / "local-volatility-replay.json"

    run(
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
    run(
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
    run(
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
    if local_volatility_fixture_exists():
        run(
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
        run([sys.executable, "scripts/check_replay_fixture.py", str(local_vol_replay_report)])
    wheel_python = Path(".wheel-smoke-venv") / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    run([str(wheel_python), "benchmarks/python_european_bs.py", str(python_report)])

    metadata = {
        "schema_version": 1,
        "git_sha": os.environ.get("GITHUB_SHA"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python": sys.version,
        "rustc": capture(["rustc", "-vV"]),
        "cargo": capture(["cargo", "-V"]),
        "peak_memory_bytes": None,
        "allocation_count": None,
        "unavailable_metrics": [
            "Per-process peak memory requires a platform-specific harness.",
            "Allocation counting requires an instrumented allocator.",
        ],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run([sys.executable, "scripts/check_benchmark_reports.py", str(output)])


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def local_volatility_fixture_exists() -> bool:
    platform_name = {
        ("Darwin", "arm64"): "macos-aarch64",
        ("Windows", "AMD64"): "windows-x86_64",
        ("Windows", "x86_64"): "windows-x86_64",
    }.get((platform.system(), platform.machine()))
    if platform_name is None:
        return False
    return (Path("fixtures/replay") / f"local_volatility-{platform_name}.json").is_file()


def capture(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


if __name__ == "__main__":
    main()
