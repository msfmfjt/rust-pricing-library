"""Install the single built wheel into a clean venv and run the Python smoke suite."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import venv


def main() -> None:
    wheels = sorted(Path("dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel in dist, found {len(wheels)}")

    environment = Path(".wheel-smoke-venv")
    venv.EnvBuilder(with_pip=True, clear=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "numpy>=2.0,<3.0",
            str(wheels[0].resolve()),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests/python",
            "-v",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
