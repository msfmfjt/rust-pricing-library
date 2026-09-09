"""Install the single built wheel into a clean venv and run the Python smoke suite."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import venv
from zipfile import ZipFile


def main() -> None:
    wheels = sorted(Path("dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel in dist, found {len(wheels)}")

    with ZipFile(wheels[0]) as archive:
        members = set(archive.namelist())
    if "rust_pricing/__init__.pyi" not in members:
        raise RuntimeError("wheel does not contain the rust_pricing.pyi type stub")
    if not any(Path(member).name == "py.typed" for member in members):
        raise RuntimeError("wheel does not contain the py.typed marker")

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
    subprocess.run([str(python), "examples/python/european_bs.py"], check=True)
    subprocess.run([str(python), "examples/python/local_vol_vegakt.py"], check=True)
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
