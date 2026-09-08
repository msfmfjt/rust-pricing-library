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
    python_report = output / "python.json"

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


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def capture(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


if __name__ == "__main__":
    main()
