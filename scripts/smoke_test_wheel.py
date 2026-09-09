"""Install the single built wheel into a clean venv and run the Python smoke suite."""

from __future__ import annotations

import ast
from email.message import Message
from email.parser import Parser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv
from zipfile import ZipFile


def main() -> None:
    wheel = selected_wheel()

    with ZipFile(wheel) as archive:
        members = set(archive.namelist())
        if "rust_pricing/__init__.pyi" not in members:
            raise RuntimeError("wheel does not contain the rust_pricing.pyi type stub")
        stub = archive.read("rust_pricing/__init__.pyi")
        metadata = Parser().parsestr(read_dist_info_text(archive, members, "METADATA"))
        wheel_metadata = Parser().parsestr(read_dist_info_text(archive, members, "WHEEL"))
        record = read_dist_info_text(archive, members, "RECORD")
    if not any(Path(member).name == "py.typed" for member in members):
        raise RuntimeError("wheel does not contain the py.typed marker")
    verify_wheel_metadata(members, metadata, wheel_metadata, record)
    expected_stub = Path("rust_pricing.pyi").read_bytes()
    if stub != expected_stub:
        raise RuntimeError("wheel type stub does not match rust_pricing.pyi")
    stub_api = exported_stub_api(expected_stub)

    environment = Path(".wheel-smoke-venv")
    create_environment(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "numpy>=2.0,<3.0",
            str(wheel.resolve()),
        ],
        check=True,
    )
    verify_runtime_symbols(python, stub_api, metadata["Version"])
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


def selected_wheel() -> Path:
    if len(sys.argv) == 2:
        wheel = Path(sys.argv[1])
        if not wheel.is_file():
            raise RuntimeError(f"wheel does not exist: {wheel}")
        if wheel.suffix != ".whl":
            raise RuntimeError(f"expected a .whl file, got: {wheel}")
        return wheel
    if len(sys.argv) != 1:
        raise RuntimeError("usage: smoke_test_wheel.py [wheel]")

    wheels = sorted(Path("dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel in dist, found {len(wheels)}")
    return wheels[0]


def read_dist_info_text(archive: ZipFile, members: set[str], filename: str) -> str:
    matches = [
        member
        for member in members
        if member.startswith("rust_pricing-") and member.endswith(f".dist-info/{filename}")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one dist-info/{filename}, found {len(matches)}")
    return archive.read(matches[0]).decode("utf-8")


def verify_wheel_metadata(
    members: set[str],
    metadata: Message,
    wheel_metadata: Message,
    record: str,
) -> None:
    if metadata["Name"] != "rust-pricing":
        raise RuntimeError(f"unexpected wheel name: {metadata['Name']}")
    if metadata["Version"] != "0.1.0":
        raise RuntimeError(f"unexpected wheel version: {metadata['Version']}")
    if metadata["Requires-Python"] != ">=3.12":
        raise RuntimeError(f"unexpected Python requirement: {metadata['Requires-Python']}")
    if wheel_metadata["Root-Is-Purelib"] != "false":
        raise RuntimeError("wheel must be a platform-specific extension wheel")
    tags = wheel_metadata.get_all("Tag") or []
    if not tags or any(tag.endswith("-none-any") for tag in tags):
        raise RuntimeError(f"wheel must carry platform tags, got: {tags}")
    if not any(member.endswith(".dist-info/sboms/pricing-python.cyclonedx.json") for member in members):
        raise RuntimeError("wheel does not contain the generated CycloneDX SBOM")
    if "rust_pricing/__init__.pyi," not in record or "rust_pricing/py.typed," not in record:
        raise RuntimeError("wheel RECORD does not list stub and py.typed entries")


def create_environment(environment: Path) -> None:
    try:
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
    except subprocess.CalledProcessError:
        uv = shutil.which("uv")
        if uv is None:
            raise
        subprocess.run([uv, "venv", "--seed", str(environment)], check=True)


def exported_stub_api(stub: bytes) -> dict[str, object]:
    tree = ast.parse(stub.decode("utf-8"))
    symbols = []
    class_members: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            symbols.append(node.name)
            if isinstance(node, ast.ClassDef):
                class_members[node.name] = [
                    member.name
                    for member in node.body
                    if isinstance(member, ast.FunctionDef)
                ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.append(node.target.id)
    return {"symbols": symbols, "class_members": class_members}


def verify_runtime_symbols(python: Path, stub_api: dict[str, object], version: str) -> None:
    code = """
import json
import rust_pricing

api = json.loads(input())
missing = [name for name in api["symbols"] if not hasattr(rust_pricing, name)]
for cls_name, members in api["class_members"].items():
    cls = getattr(rust_pricing, cls_name, None)
    if cls is not None:
        missing.extend(
            f"{cls_name}.{member}"
            for member in members
            if not hasattr(cls, member)
        )
if rust_pricing.__version__ != api["version"]:
    missing.append("__version__")
if rust_pricing.version() != api["version"]:
    missing.append("version()")
raise SystemExit("missing runtime symbols: " + ", ".join(missing) if missing else 0)
"""
    subprocess.run(
        [str(python), "-c", code],
        input=json.dumps({**stub_api, "version": version}),
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
