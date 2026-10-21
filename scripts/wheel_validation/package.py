"""Apply the same package and installed API gates in every CLI phase."""
from __future__ import annotations
from email.parser import Parser
import os
from pathlib import Path
import shutil
import subprocess
import venv
from zipfile import ZipFile
from .archive import (
    expected_project_metadata, locked_registry_packages, locked_dependency_graph,
    verify_wheel_archive_members, verify_wheel_text_members, verify_wheel_member_layout,
    read_dist_info_text, parse_wheel_filename, verify_wheel_python_abi_matches_interpreter,
    verify_wheel_metadata,
)
from .stub import exported_stub_api
from .runtime import verify_runtime_symbols

def prepare_wheel(wheel: Path) -> Path:
    """Keep every archive, metadata, install and runtime check common to all phases."""
    expected_metadata = expected_project_metadata()
    expected_readme_payload = Path("README.md").read_text("utf-8") + "\n"
    locked_packages = locked_registry_packages()
    locked_dependencies = locked_dependency_graph()

    with ZipFile(wheel) as archive:
        verify_wheel_archive_members(archive.infolist())
        member_order = [member for member in archive.namelist() if not member.endswith("/")]
        members = set(member_order)
        member_bytes = {member: archive.read(member) for member in members}
        verify_wheel_text_members(member_bytes)
        verify_wheel_member_layout(members)
        if "rust_pricing/__init__.pyi" not in members:
            raise RuntimeError("wheel does not contain the rust_pricing.pyi type stub")
        stub = archive.read("rust_pricing/__init__.pyi")
        metadata = Parser().parsestr(read_dist_info_text(archive, members, "METADATA"))
        wheel_metadata = Parser().parsestr(read_dist_info_text(archive, members, "WHEEL"))
        record = read_dist_info_text(archive, members, "RECORD")
    if "rust_pricing/py.typed" not in members:
        raise RuntimeError("wheel does not contain the py.typed marker")
    _, _, filename_tags = parse_wheel_filename(wheel.name)
    verify_wheel_python_abi_matches_interpreter(wheel, filename_tags)
    verify_wheel_metadata(
        wheel,
        member_order,
        members,
        metadata,
        wheel_metadata,
        record,
        member_bytes,
        expected_metadata,
        expected_readme_payload,
        locked_packages,
        locked_dependencies,
    )
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
            "./backtesting[validation,constraints]",
        ],
        check=True,
    )
    verify_runtime_symbols(python, stub_api, metadata["Version"])
    return python


def create_environment(environment: Path) -> None:
    try:
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
    except subprocess.CalledProcessError:
        uv = shutil.which("uv")
        if uv is None:
            raise
        subprocess.run([uv, "venv", "--seed", str(environment)], check=True)
