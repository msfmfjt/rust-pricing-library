"""Install the single built wheel into a clean venv and run the Python smoke suite."""

from __future__ import annotations

import ast
import base64
import csv
from email.message import Message
from email.parser import Parser
import hashlib
import json
import os
from pathlib import PurePosixPath
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from urllib.parse import unquote
import uuid
import venv
from zipfile import ZipFile
from zipfile import ZipInfo

EXPECTED_METADATA_FIELDS = (
    "Metadata-Version",
    "Name",
    "Version",
    "Summary",
    "Author",
    "Requires-Python",
    "Description-Content-Type",
)
EXPECTED_WHEEL_FIELDS = (
    "Wheel-Version",
    "Generator",
    "Root-Is-Purelib",
    "Tag",
)
EXPECTED_INIT_PY = (
    "from .rust_pricing import *\n\n"
    "__doc__ = rust_pricing.__doc__\n"
    "if hasattr(rust_pricing, \"__all__\"):\n"
    "    __all__ = rust_pricing.__all__"
)
EXPECTED_WORKSPACE_SBOM_DEPENDENCIES = {
    "pricing-core": set(),
    "pricing-numerics": {"pricing-core"},
    "pricing-aad": {"pricing-core", "pricing-numerics"},
    "pricing-market": {"pricing-core", "pricing-numerics"},
    "pricing-product": {"pricing-core"},
    "pricing-models": {"pricing-core", "pricing-market", "pricing-numerics"},
    "pricing-mc": {
        "pricing-aad",
        "pricing-core",
        "pricing-market",
        "pricing-models",
        "pricing-numerics",
        "pricing-product",
    },
    "pricing-risk": {
        "pricing-aad",
        "pricing-core",
        "pricing-market",
        "pricing-mc",
        "pricing-models",
        "pricing-numerics",
        "pricing-product",
    },
    "pricing": {
        "pricing-aad",
        "pricing-core",
        "pricing-market",
        "pricing-mc",
        "pricing-models",
        "pricing-numerics",
        "pricing-product",
        "pricing-risk",
    },
    "pricing-python": {"pricing"},
}


def main() -> None:
    wheel = selected_wheel()
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


def verify_wheel_archive_members(member_infos: list[ZipInfo]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for info in member_infos:
        name = info.filename
        if not name:
            raise RuntimeError("wheel contains an empty member path")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"wheel contains an unsafe member path: {name}")
        mode = (info.external_attr >> 16) & 0o777
        if name.endswith("/"):
            if mode & 0o002:
                raise RuntimeError(f"wheel directory must not be world-writable: {name}")
            if mode & 0o111 != 0o111:
                raise RuntimeError(f"wheel directory must be searchable: {name}")
        elif is_wheel_extension_member(name):
            if mode & 0o111 == 0:
                raise RuntimeError(f"wheel extension member must be executable: {name}")
            if mode & 0o002:
                raise RuntimeError(f"wheel extension member must not be world-writable: {name}")
        else:
            if mode & 0o111:
                raise RuntimeError(f"wheel data member must not be executable: {name}")
            if mode & 0o002:
                raise RuntimeError(f"wheel data member must not be world-writable: {name}")
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise RuntimeError(f"wheel contains duplicate member paths: {sorted(duplicates)}")


def is_wheel_extension_member(name: str) -> bool:
    return name.startswith("rust_pricing/") and (
        name.endswith(".so") or name.endswith(".pyd") or name.endswith(".dll")
    )


def verify_wheel_member_layout(members: set[str]) -> None:
    expected_package_members = {
        "rust_pricing/__init__.py",
        "rust_pricing/__init__.pyi",
        "rust_pricing/py.typed",
    }
    missing_package_members = sorted(expected_package_members.difference(members))
    if missing_package_members:
        raise RuntimeError(f"wheel is missing package members: {missing_package_members}")

    dist_info_dirs = {
        member.split(".dist-info/", 1)[0] + ".dist-info"
        for member in members
        if ".dist-info/" in member
    }
    if len(dist_info_dirs) != 1:
        raise RuntimeError(f"wheel must contain exactly one dist-info directory, found {dist_info_dirs}")
    dist_info_dir = next(iter(dist_info_dirs))
    if not dist_info_dir.startswith("rust_pricing-"):
        raise RuntimeError(f"unexpected dist-info directory: {dist_info_dir}")

    extensions = [
        member
        for member in members
        if is_wheel_extension_member(member)
    ]
    if len(extensions) != 1:
        raise RuntimeError(f"wheel must contain exactly one extension module, found {extensions}")
    extension_name = Path(extensions[0]).name
    if not extension_name.startswith("rust_pricing."):
        raise RuntimeError(f"unexpected extension module name: {extension_name}")

    for filename in ["METADATA", "WHEEL", "RECORD"]:
        matches = [
            member for member in members if member == f"{dist_info_dir}/{filename}"
        ]
        if len(matches) != 1:
            raise RuntimeError(f"wheel must contain exactly one dist-info/{filename}")

    allowed_prefixes = ("rust_pricing/", f"{dist_info_dir}/")
    unexpected = sorted(member for member in members if not member.startswith(allowed_prefixes))
    if unexpected:
        raise RuntimeError(f"wheel contains unexpected top-level members: {unexpected}")


def verify_wheel_text_members(member_bytes: dict[str, bytes]) -> None:
    for member, data in sorted(member_bytes.items()):
        if not is_wheel_text_member(member):
            continue
        if data.startswith(b"\xef\xbb\xbf"):
            raise RuntimeError(f"wheel text member has a UTF-8 BOM: {member}")
        if b"\r" in data:
            raise RuntimeError(f"wheel text member must use LF newlines: {member}")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"wheel text member must be valid UTF-8: {member}") from exc


def is_wheel_text_member(member: str) -> bool:
    if member in {"rust_pricing/__init__.py", "rust_pricing/__init__.pyi", "rust_pricing/py.typed"}:
        return True
    return (
        member.endswith(".dist-info/METADATA")
        or member.endswith(".dist-info/WHEEL")
        or member.endswith(".dist-info/RECORD")
        or member.endswith(".dist-info/sboms/pricing-python.cyclonedx.json")
    )


def verify_wheel_metadata(
    wheel: Path,
    member_order: list[str],
    members: set[str],
    metadata: Message,
    wheel_metadata: Message,
    record: str,
    member_bytes: dict[str, bytes],
    expected_metadata: dict[str, str],
    expected_readme_payload: str,
    locked_packages: dict[tuple[str, str], str],
    locked_dependencies: dict[tuple[str, str], set[tuple[str, str]]],
) -> None:
    verify_message_fields(metadata, EXPECTED_METADATA_FIELDS, "METADATA")
    verify_message_fields(wheel_metadata, EXPECTED_WHEEL_FIELDS, "WHEEL")
    for field, expected_value in expected_metadata.items():
        if metadata[field] != expected_value:
            raise RuntimeError(
                f"unexpected wheel {field}: {metadata[field]} != {expected_value}"
            )
    filename_distribution, filename_version, filename_tags = parse_wheel_filename(wheel.name)
    if filename_distribution != "rust_pricing":
        raise RuntimeError(f"unexpected wheel filename distribution: {filename_distribution}")
    if filename_version != metadata["Version"]:
        raise RuntimeError(
            f"wheel filename version {filename_version} does not match METADATA version {metadata['Version']}"
        )
    expected_dist_info = f"rust_pricing-{metadata['Version']}.dist-info"
    actual_dist_info = dist_info_dir(members)
    if actual_dist_info != expected_dist_info:
        raise RuntimeError(f"unexpected wheel dist-info directory: {actual_dist_info}")
    init_py = member_bytes["rust_pricing/__init__.py"].decode("utf-8")
    if init_py != EXPECTED_INIT_PY:
        raise RuntimeError("wheel __init__.py content changed")
    if member_bytes["rust_pricing/py.typed"] != b"":
        raise RuntimeError("wheel py.typed marker must be empty")
    content_type = metadata["Description-Content-Type"]
    if not (
        isinstance(content_type, str)
        and content_type.startswith("text/markdown")
        and "charset=UTF-8" in content_type
    ):
        raise RuntimeError(f"unexpected wheel Description-Content-Type: {content_type}")
    for field in ["License", "License-Expression", "License-File"]:
        if metadata.get_all(field):
            raise RuntimeError(f"wheel metadata must not declare {field}")
    if metadata.get_payload() != expected_readme_payload:
        raise RuntimeError("wheel metadata README payload does not match README.md")
    if wheel_metadata["Wheel-Version"] != "1.0":
        raise RuntimeError(f"unexpected wheel metadata version: {wheel_metadata['Wheel-Version']}")
    generator = wheel_metadata["Generator"]
    if not isinstance(generator, str) or not generator.startswith("maturin "):
        raise RuntimeError(f"unexpected wheel metadata generator: {generator}")
    if wheel_metadata["Root-Is-Purelib"] != "false":
        raise RuntimeError("wheel must be a platform-specific extension wheel")
    tags = wheel_metadata.get_all("Tag") or []
    if not tags or any(not is_platform_cpython_tag(tag) for tag in tags):
        raise RuntimeError(f"wheel must carry platform tags, got: {tags}")
    if set(tags) != filename_tags:
        raise RuntimeError(
            f"wheel filename tags {sorted(filename_tags)} do not match WHEEL tags {sorted(tags)}"
        )
    verify_cyclonedx_sbom(
        members,
        member_bytes,
        expected_metadata["Version"],
        locked_packages,
        locked_dependencies,
    )
    record_members = verify_wheel_record(record, member_order, members, member_bytes)
    if "rust_pricing/__init__.pyi" not in record_members or "rust_pricing/py.typed" not in record_members:
        raise RuntimeError("wheel RECORD does not list stub and py.typed entries")


def verify_message_fields(message: Message, expected_fields: tuple[str, ...], name: str) -> None:
    actual_fields = [field for field, _ in message.items()]
    if actual_fields != list(expected_fields):
        raise RuntimeError(f"wheel {name} fields changed: {actual_fields}")
    for field in expected_fields:
        if len(message.get_all(field) or []) != 1:
            raise RuntimeError(f"wheel {name} field {field} must appear exactly once")


def parse_wheel_filename(filename: str) -> tuple[str, str, set[str]]:
    if not filename.endswith(".whl"):
        raise RuntimeError(f"expected a .whl file, got: {filename}")
    stem = filename.removesuffix(".whl")
    parts = stem.split("-")
    if len(parts) not in {5, 6} or any(not part for part in parts):
        raise RuntimeError(f"invalid wheel filename: {filename}")
    distribution = parts[0]
    version = parts[1]
    python_tags, abi_tags, platform_tags = (part.split(".") for part in parts[-3:])
    tags = {
        f"{python_tag}-{abi_tag}-{platform_tag}"
        for python_tag in python_tags
        for abi_tag in abi_tags
        for platform_tag in platform_tags
    }
    if not tags:
        raise RuntimeError(f"wheel filename has no tags: {filename}")
    return distribution, version, tags


def dist_info_dir(members: set[str]) -> str:
    dist_info_dirs = {
        member.split(".dist-info/", 1)[0] + ".dist-info"
        for member in members
        if ".dist-info/" in member
    }
    if len(dist_info_dirs) != 1:
        raise RuntimeError(f"wheel must contain exactly one dist-info directory, found {dist_info_dirs}")
    return next(iter(dist_info_dirs))


def is_platform_cpython_tag(tag: str) -> bool:
    parts = tag.split("-")
    if len(parts) != 3:
        return False
    python_tag, abi_tag, platform_tag = parts
    return (
        python_tag.startswith("cp")
        and abi_tag.startswith("cp")
        and platform_tag != "any"
        and all(parts)
    )


def verify_wheel_python_abi_matches_interpreter(wheel: Path, tags: set[str]) -> None:
    expected = f"cp{sys.version_info.major}{sys.version_info.minor}"
    for tag in tags:
        parts = tag.split("-")
        if len(parts) == 3 and parts[0] == expected and parts[1] == expected:
            return
    raise RuntimeError(
        f"wheel {wheel.name} is not built for this Python interpreter "
        f"({expected}); run this smoke test with a matching Python"
    )


def verify_cyclonedx_sbom(
    members: set[str],
    member_bytes: dict[str, bytes],
    version: str,
    locked_packages: dict[tuple[str, str], str],
    locked_dependencies: dict[tuple[str, str], set[tuple[str, str]]],
) -> None:
    matches = [
        member
        for member in members
        if member.endswith(".dist-info/sboms/pricing-python.cyclonedx.json")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one generated CycloneDX SBOM, found {len(matches)}")
    try:
        sbom = json.loads(member_bytes[matches[0]].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("wheel CycloneDX SBOM must be valid UTF-8 JSON") from exc
    if not isinstance(sbom, dict):
        raise RuntimeError("wheel CycloneDX SBOM root must be an object")
    if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") != "1.5":
        raise RuntimeError("wheel CycloneDX SBOM must use CycloneDX 1.5")
    if sbom.get("version") != 1:
        raise RuntimeError("wheel CycloneDX SBOM version must be 1")
    serial_number = sbom.get("serialNumber")
    if not isinstance(serial_number, str) or not serial_number.startswith("urn:uuid:"):
        raise RuntimeError("wheel CycloneDX SBOM serialNumber must be a UUID URN")
    try:
        uuid.UUID(serial_number.removeprefix("urn:uuid:"))
    except ValueError as exc:
        raise RuntimeError("wheel CycloneDX SBOM serialNumber must be a valid UUID") from exc
    metadata = sbom.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("wheel CycloneDX SBOM metadata must be an object")
    component = metadata.get("component")
    if not isinstance(component, dict):
        raise RuntimeError("wheel CycloneDX SBOM metadata.component must be an object")
    root_ref = component.get("bom-ref")
    if not isinstance(root_ref, str) or not root_ref:
        raise RuntimeError("wheel CycloneDX SBOM root component has no bom-ref")
    if component.get("name") != "pricing-python" or component.get("version") != version:
        raise RuntimeError("wheel CycloneDX SBOM root component metadata mismatch")
    if component.get("type") != "library" or component.get("scope") != "required":
        raise RuntimeError("wheel CycloneDX SBOM root component type/scope mismatch")
    if component.get("author") != "Masafumi Fujita":
        raise RuntimeError("wheel CycloneDX SBOM root component author mismatch")
    purl = component.get("purl")
    if purl != f"pkg:cargo/pricing-python@{version}?download_url=file://.":
        raise RuntimeError("wheel CycloneDX SBOM root component purl mismatch")
    external_references = component.get("externalReferences")
    if not isinstance(external_references, list) or {
        "type": "vcs",
        "url": "https://github.com/msfmfjt/rust-pricing-library",
    } not in external_references:
        raise RuntimeError("wheel CycloneDX SBOM root component must reference the VCS URL")

    components = sbom.get("components")
    if not isinstance(components, list):
        raise RuntimeError("wheel CycloneDX SBOM components must be an array")
    known_refs = {root_ref}
    names_by_ref = {root_ref: "pricing-python"}
    packages_by_ref = {root_ref: ("pricing-python", version)}
    workspace_refs = {"pricing-python": root_ref}
    required_workspace_components = set(EXPECTED_WORKSPACE_SBOM_DEPENDENCIES) - {
        "pricing-python"
    }
    component_names = set()
    registry_components: dict[tuple[str, str], dict[str, object]] = {}
    for index, component in enumerate(components, start=1):
        if not isinstance(component, dict):
            raise RuntimeError(f"wheel CycloneDX SBOM component {index} must be an object")
        bom_ref = component.get("bom-ref")
        if not isinstance(bom_ref, str) or not bom_ref:
            raise RuntimeError(f"wheel CycloneDX SBOM component {index} has no bom-ref")
        if bom_ref in known_refs:
            raise RuntimeError(f"wheel CycloneDX SBOM component bom-ref duplicated: {bom_ref}")
        known_refs.add(bom_ref)
        name = component.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"wheel CycloneDX SBOM component {index} has no name")
        component_version = component.get("version")
        if not isinstance(component_version, str) or not component_version:
            raise RuntimeError(f"wheel CycloneDX SBOM component {name} has no version")
        names_by_ref[bom_ref] = name
        packages_by_ref[bom_ref] = (name, component_version)
        if name in required_workspace_components:
            if name in component_names:
                raise RuntimeError(f"wheel CycloneDX SBOM workspace component duplicated: {name}")
            component_names.add(name)
            workspace_refs[name] = bom_ref
            verify_workspace_sbom_component(component, name, version)
        else:
            key = (name, component_version)
            if key in registry_components:
                raise RuntimeError(
                    f"wheel CycloneDX SBOM registry component duplicated: {name} {component_version}"
                )
            registry_components[key] = component
    missing = sorted(required_workspace_components.difference(component_names))
    if missing:
        raise RuntimeError(f"wheel CycloneDX SBOM is missing components: {missing}")
    actual_registry_packages = set(registry_components)
    expected_registry_packages = set(locked_packages)
    if actual_registry_packages != expected_registry_packages:
        missing = sorted(expected_registry_packages - actual_registry_packages)
        unexpected = sorted(actual_registry_packages - expected_registry_packages)
        raise RuntimeError(
            "wheel CycloneDX SBOM registry components do not match Cargo.lock: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for (name, component_version), component in registry_components.items():
        verify_registry_sbom_component(
            component,
            name,
            component_version,
            locked_packages[(name, component_version)],
        )

    dependencies = sbom.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise RuntimeError("wheel CycloneDX SBOM dependencies must be a non-empty array")
    dependency_refs = set()
    dependency_edges: dict[str, set[str]] = {}
    for index, dependency in enumerate(dependencies, start=1):
        if not isinstance(dependency, dict):
            raise RuntimeError(f"wheel CycloneDX SBOM dependency {index} must be an object")
        ref = dependency.get("ref")
        if not isinstance(ref, str) or not ref:
            raise RuntimeError(f"wheel CycloneDX SBOM dependency {index} has no ref")
        if ref in dependency_refs:
            raise RuntimeError(f"wheel CycloneDX SBOM dependency ref duplicated: {ref}")
        dependency_refs.add(ref)
        if ref not in known_refs:
            raise RuntimeError(f"wheel CycloneDX SBOM dependency ref is unknown: {ref}")
        depends_on = dependency.get("dependsOn")
        if depends_on is not None and (not isinstance(depends_on, list) or not all(
            isinstance(item, str) and item for item in depends_on
        )):
            raise RuntimeError(
                f"wheel CycloneDX SBOM dependency {index} must list string dependsOn refs"
            )
        if len(depends_on or []) != len(set(depends_on or [])):
            raise RuntimeError(
                f"wheel CycloneDX SBOM dependency {index} has duplicate dependsOn refs"
            )
        dependency_edges[ref] = set(depends_on or [])
        for dependency_ref in depends_on or []:
            if dependency_ref not in known_refs:
                raise RuntimeError(
                    f"wheel CycloneDX SBOM dependency dependsOn ref is unknown: {dependency_ref}"
                )
    if dependency_refs != known_refs:
        missing = sorted(known_refs.difference(dependency_refs))
        raise RuntimeError(f"wheel CycloneDX SBOM dependencies missing refs: {missing}")
    for name, expected_dependencies in EXPECTED_WORKSPACE_SBOM_DEPENDENCIES.items():
        ref = workspace_refs[name]
        actual_dependencies = {
            names_by_ref[dependency_ref]
            for dependency_ref in dependency_edges[ref]
            if names_by_ref[dependency_ref] in EXPECTED_WORKSPACE_SBOM_DEPENDENCIES
        }
        if actual_dependencies != expected_dependencies:
            raise RuntimeError(
                f"wheel CycloneDX SBOM workspace dependencies for {name} mismatch: "
                f"{sorted(actual_dependencies)} != {sorted(expected_dependencies)}"
            )
    if set(packages_by_ref.values()) != set(locked_dependencies):
        raise RuntimeError("wheel CycloneDX SBOM dependency graph packages mismatch")
    for ref, package in packages_by_ref.items():
        actual_dependencies = {
            packages_by_ref[dependency_ref] for dependency_ref in dependency_edges[ref]
        }
        expected_dependencies = locked_dependencies[package]
        if actual_dependencies != expected_dependencies:
            raise RuntimeError(
                "wheel CycloneDX SBOM dependency graph mismatch for "
                f"{package[0]} {package[1]}: "
                f"{sorted(actual_dependencies)} != {sorted(expected_dependencies)}"
            )


def verify_workspace_sbom_component(
    component: dict[str, object],
    name: str,
    version: str,
) -> None:
    if component.get("type") != "library":
        raise RuntimeError(f"wheel CycloneDX SBOM component {name} must be a library")
    if component.get("scope") != "required":
        raise RuntimeError(f"wheel CycloneDX SBOM component {name} scope mismatch")
    if component.get("author") != "Masafumi Fujita":
        raise RuntimeError(f"wheel CycloneDX SBOM component {name} author mismatch")
    if component.get("version") != version:
        raise RuntimeError(f"wheel CycloneDX SBOM component {name} version mismatch")
    purl = component.get("purl")
    if not workspace_dependency_purl_matches(purl, name, version):
        raise RuntimeError(f"wheel CycloneDX SBOM component {name} purl mismatch")
    external_references = component.get("externalReferences")
    if not isinstance(external_references, list) or {
        "type": "vcs",
        "url": "https://github.com/msfmfjt/rust-pricing-library",
    } not in external_references:
        raise RuntimeError(f"wheel CycloneDX SBOM component {name} must reference the VCS URL")


def workspace_dependency_purl_matches(purl: object, name: str, version: str) -> bool:
    prefix = f"pkg:cargo/{name}@{version}?download_url=file://"
    if not isinstance(purl, str) or not purl.startswith(prefix):
        return False
    relative_path = unquote(purl.removeprefix(prefix)).replace("\\", "/")
    return relative_path == f"../{name}"


def verify_registry_sbom_component(
    component: dict[str, object],
    name: str,
    version: str,
    checksum: str,
) -> None:
    if component.get("type") != "library" or component.get("scope") not in {
        "required",
        "excluded",
    }:
        raise RuntimeError(
            f"wheel CycloneDX SBOM registry component {name} type/scope mismatch"
        )
    if component.get("purl") != f"pkg:cargo/{name}@{version}":
        raise RuntimeError(
            f"wheel CycloneDX SBOM registry component {name} purl mismatch"
        )
    expected_ref = (
        "registry+https://github.com/rust-lang/crates.io-index#"
        f"{name}@{version}"
    )
    if component.get("bom-ref") != expected_ref:
        raise RuntimeError(
            f"wheel CycloneDX SBOM registry component {name} bom-ref mismatch"
        )
    if component.get("hashes") != [{"alg": "SHA-256", "content": checksum}]:
        raise RuntimeError(
            f"wheel CycloneDX SBOM registry component {name} checksum mismatch"
        )


def locked_registry_packages() -> dict[tuple[str, str], str]:
    workspace_names = set(EXPECTED_WORKSPACE_SBOM_DEPENDENCIES)
    registry_packages: dict[tuple[str, str], str] = {}
    expected_source = "registry+https://github.com/rust-lang/crates.io-index"
    for package in locked_package_entries():
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Cargo.lock package name must be a non-empty string")
        if not isinstance(version, str) or not version:
            raise RuntimeError(f"Cargo.lock package {name} version must be a string")
        if name in workspace_names:
            continue
        if package.get("source") != expected_source:
            raise RuntimeError(f"Cargo.lock package {name} must use crates.io")
        checksum = package.get("checksum")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise RuntimeError(f"Cargo.lock package {name} checksum must be SHA-256")
        try:
            int(checksum, 16)
        except ValueError as exc:
            raise RuntimeError(
                f"Cargo.lock package {name} checksum must be SHA-256"
            ) from exc
        key = (name, version)
        if key in registry_packages:
            raise RuntimeError(f"Cargo.lock package duplicated: {name} {version}")
        registry_packages[key] = checksum
    return registry_packages


def locked_dependency_graph() -> dict[tuple[str, str], set[tuple[str, str]]]:
    packages = locked_package_entries()
    packages_by_key: dict[tuple[str, str], dict[str, object]] = {}
    versions_by_name: dict[str, set[str]] = {}
    for package in packages:
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise RuntimeError("Cargo.lock package identity must contain strings")
        key = (name, version)
        if key in packages_by_key:
            raise RuntimeError(f"Cargo.lock package duplicated: {name} {version}")
        packages_by_key[key] = package
        versions_by_name.setdefault(name, set()).add(version)

    graph: dict[tuple[str, str], set[tuple[str, str]]] = {}
    workspace_names = set(EXPECTED_WORKSPACE_SBOM_DEPENDENCIES)
    for key, package in packages_by_key.items():
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list) or not all(
            isinstance(dependency, str) for dependency in dependencies
        ):
            raise RuntimeError(
                f"Cargo.lock package {key[0]} dependencies must be strings"
            )
        resolved = {
            resolve_locked_dependency(dependency, packages_by_key, versions_by_name)
            for dependency in dependencies
        }
        if key[0] in workspace_names:
            dev_names = workspace_dev_dependency_names(key[0])
            resolved = {
                dependency for dependency in resolved if dependency[0] not in dev_names
            }
        graph[key] = resolved
    return graph


def resolve_locked_dependency(
    dependency: str,
    packages_by_key: dict[tuple[str, str], dict[str, object]],
    versions_by_name: dict[str, set[str]],
) -> tuple[str, str]:
    parts = dependency.split()
    if len(parts) == 1:
        versions = versions_by_name.get(parts[0], set())
        if len(versions) != 1:
            raise RuntimeError(f"Cargo.lock dependency is ambiguous: {dependency}")
        key = (parts[0], next(iter(versions)))
    elif len(parts) == 2:
        key = (parts[0], parts[1])
    else:
        raise RuntimeError(f"Cargo.lock dependency format is unsupported: {dependency}")
    if key not in packages_by_key:
        raise RuntimeError(f"Cargo.lock dependency is missing a package: {dependency}")
    return key


def workspace_dev_dependency_names(crate_name: str) -> set[str]:
    manifest = tomllib.loads(
        Path(f"crates/{crate_name}/Cargo.toml").read_text("utf-8")
    )
    dev_dependencies = manifest.get("dev-dependencies", {})
    if not isinstance(dev_dependencies, dict):
        raise RuntimeError(
            f"crates/{crate_name}/Cargo.toml dev-dependencies must be a table"
        )
    names = set()
    for dependency_name, specification in dev_dependencies.items():
        if not isinstance(dependency_name, str):
            raise RuntimeError(f"{crate_name} dev-dependency name must be a string")
        if isinstance(specification, dict):
            package_name = specification.get("package", dependency_name)
            if not isinstance(package_name, str) or not package_name:
                raise RuntimeError(
                    f"{crate_name} dev-dependency package must be a string"
                )
            names.add(package_name)
        else:
            names.add(dependency_name)
    return names


def locked_package_entries() -> list[dict[str, object]]:
    lock = tomllib.loads(Path("Cargo.lock").read_text("utf-8"))
    if lock.get("version") != 4:
        raise RuntimeError("Cargo.lock must use lockfile format version 4")
    packages = lock.get("package")
    if not isinstance(packages, list) or not all(
        isinstance(package, dict) for package in packages
    ):
        raise RuntimeError("Cargo.lock package entries must be tables")
    return packages


def expected_project_metadata() -> dict[str, str]:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text("utf-8"))
    cargo_manifest = tomllib.loads(Path("Cargo.toml").read_text("utf-8"))
    build_system = required_table(pyproject, "build-system", "pyproject.toml")
    if build_system.get("requires") != ["maturin>=1.9,<2.0"]:
        raise RuntimeError("pyproject.toml build-system.requires mismatch")
    if build_system.get("build-backend") != "maturin":
        raise RuntimeError("pyproject.toml build-system.build-backend mismatch")
    project = required_table(pyproject, "project", "pyproject.toml")
    workspace = required_table(cargo_manifest, "workspace", "Cargo.toml")
    workspace_package = required_table(workspace, "package", "Cargo.toml workspace")

    name = required_string(project, "name", "pyproject.toml project")
    description = required_string(project, "description", "pyproject.toml project")
    readme = required_string(project, "readme", "pyproject.toml project")
    if readme != "README.md":
        raise RuntimeError("pyproject.toml project.readme must be README.md")
    authors = project.get("authors")
    if authors != [{"name": "Masafumi Fujita"}]:
        raise RuntimeError("pyproject.toml project.authors mismatch")
    for key in ["license", "license-files"]:
        if key in project:
            raise RuntimeError(
                f"pyproject.toml project.{key} must stay absent until licensing is decided"
            )
    if project.get("dynamic") != ["version"]:
        raise RuntimeError("pyproject.toml project.dynamic must derive version dynamically")
    maturin = required_table(pyproject, "tool", "pyproject.toml").get("maturin")
    if not isinstance(maturin, dict):
        raise RuntimeError("pyproject.toml must contain a [tool.maturin] table")
    expected_maturin = {
        "manifest-path": "crates/pricing-python/Cargo.toml",
        "module-name": "rust_pricing",
        "features": ["extension-module"],
    }
    for key, expected_value in expected_maturin.items():
        if maturin.get(key) != expected_value:
            raise RuntimeError(f"pyproject.toml tool.maturin.{key} mismatch")
    version = required_string(workspace_package, "version", "Cargo.toml workspace.package")
    requires_python = required_string(
        project,
        "requires-python",
        "pyproject.toml project",
    )
    return {
        "Name": name,
        "Version": version,
        "Summary": description,
        "Author": "Masafumi Fujita",
        "Requires-Python": requires_python,
    }


def required_table(document: dict[str, object], key: str, label: str) -> dict[str, object]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a [{key}] table")
    return value


def required_string(document: dict[str, object], key: str, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must contain a non-empty {key!r} string")
    return value


def verify_wheel_record(
    record: str,
    wheel_member_order: list[str],
    wheel_members: set[str],
    member_bytes: dict[str, bytes],
) -> set[str]:
    if not record.endswith("\n"):
        raise RuntimeError("wheel RECORD must end with LF")
    if record.endswith("\n\n"):
        raise RuntimeError("wheel RECORD must end with exactly one LF")
    rows = list(csv.reader(record.splitlines()))
    entries: list[tuple[int, str, str, str]] = []
    members: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if len(row) != 3:
            raise RuntimeError(f"wheel RECORD row {index} must have three fields")
        path, digest, size = row
        if not path:
            raise RuntimeError(f"wheel RECORD row {index} has an empty path")
        if path in members:
            raise RuntimeError(f"wheel RECORD lists {path!r} more than once")
        members.add(path)
        entries.append((index, path, digest, size))
    if not any(member.endswith(".dist-info/RECORD") for member in members):
        raise RuntimeError("wheel RECORD does not list itself")
    record_order = [path for _, path, _, _ in entries]
    if record_order != wheel_member_order:
        raise RuntimeError("wheel RECORD row order must match archive member order")
    if not record_order[-1].endswith(".dist-info/RECORD"):
        raise RuntimeError("wheel RECORD entry must be last")

    missing_from_record = sorted(wheel_members.difference(members))
    if missing_from_record:
        raise RuntimeError(f"wheel RECORD is missing entries: {missing_from_record[:10]}")
    missing_from_wheel = sorted(members.difference(wheel_members))
    if missing_from_wheel:
        raise RuntimeError(f"wheel RECORD lists missing files: {missing_from_wheel[:10]}")

    for index, path, digest, size in entries:
        if path.endswith(".dist-info/RECORD"):
            if digest or size:
                raise RuntimeError("wheel RECORD entry must omit its own digest and size")
            continue
        if not digest.startswith("sha256="):
            raise RuntimeError(f"wheel RECORD row {index} must use a sha256 digest")
        expected_digest = base64.urlsafe_b64encode(
            hashlib.sha256(member_bytes[path]).digest()
        ).decode("ascii").rstrip("=")
        actual_digest = digest.removeprefix("sha256=")
        if actual_digest != expected_digest:
            raise RuntimeError(f"wheel RECORD row {index} has an invalid digest for {path}")
        if not size.isdecimal():
            raise RuntimeError(f"wheel RECORD row {index} has an invalid size")
        if int(size) != len(member_bytes[path]):
            raise RuntimeError(f"wheel RECORD row {index} has the wrong size for {path}")
    return members


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
    verify_stub_static_shape(tree)
    symbols = []
    class_members: dict[str, list[str]] = {}
    class_methods: dict[str, dict[str, ast.FunctionDef]] = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            symbols.append(node.name)
            if isinstance(node, ast.ClassDef):
                class_members[node.name] = [
                    member.name
                    for member in node.body
                    if isinstance(member, ast.FunctionDef)
                ]
                class_methods[node.name] = {
                    member.name: member
                    for member in node.body
                    if isinstance(member, ast.FunctionDef)
                }
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.append(node.target.id)
    return {
        "symbols": symbols,
        "class_members": class_members,
        "static_methods": sorted(decorated_members(class_methods, "staticmethod")),
        "properties": sorted(decorated_members(class_methods, "property")),
    }


def verify_stub_static_shape(tree: ast.Module) -> None:
    imported_names: set[str] = set()
    top_level_names: list[str] = []
    top_level_functions: dict[str, ast.FunctionDef] = {}
    class_bases: dict[str, list[str]] = {}
    class_members: dict[str, list[str]] = {}
    class_methods: dict[str, dict[str, ast.FunctionDef]] = {}
    assignments: dict[str, ast.expr] = {}

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Assign):
            top_level_names.extend(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top_level_names.append(node.target.id)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            top_level_names.append(node.name)
            if isinstance(node, ast.ClassDef):
                class_bases[node.name] = class_base_names(node)
                class_members[node.name] = [
                    member.name
                    for member in node.body
                    if isinstance(member, ast.FunctionDef)
                ]
                class_methods[node.name] = {
                    member.name: member
                    for member in node.body
                    if isinstance(member, ast.FunctionDef)
                }
            else:
                top_level_functions[node.name] = node

    duplicates = sorted(duplicates_in(top_level_names))
    if duplicates:
        raise RuntimeError(f"wheel type stub has duplicate top-level definitions: {duplicates}")

    expected_top_level_names = {
        "AsianObservation",
        "BarrierDirection",
        "BarrierStyle",
        "DateLike",
        "DiagnosticEstimate",
        "Diagnostics",
        "DigitalPayout",
        "DiscountCurve",
        "DividendEvent",
        "Engine",
        "EssviSlice",
        "Market",
        "Model",
        "OptionSide",
        "PricingError",
        "PricingPlan",
        "PricingRequest",
        "PricingResult",
        "PricingWarning",
        "Product",
        "RiskEstimate",
        "RiskRequest",
        "RiskUnit",
        "RiskValidation",
        "SmileDynamics",
        "ValidationError",
        "ValidationIssue",
        "ValidationPhase",
        "VegaKtBucketEstimate",
        "VegaKtCoordinate",
        "VegaKtCovarianceLayout",
        "VegaKtProjection",
        "VegaKtReportingStats",
        "VegaKtResidualDiagnostics",
        "VegaKtResult",
        "VegaKtUnit",
        "__version__",
        "request_json_schema",
        "result_json_schema",
        "version",
    }
    missing_top_level_names = sorted(expected_top_level_names.difference(top_level_names))
    if missing_top_level_names:
        raise RuntimeError(
            f"wheel type stub is missing top-level definitions: {missing_top_level_names}"
        )
    unexpected_top_level_names = sorted(set(top_level_names).difference(expected_top_level_names))
    if unexpected_top_level_names:
        raise RuntimeError(
            f"wheel type stub has unexpected top-level definitions: {unexpected_top_level_names}"
        )

    for class_name, members in sorted(class_members.items()):
        duplicate_members = sorted(duplicates_in(members))
        if duplicate_members:
            raise RuntimeError(
                f"wheel type stub has duplicate members in {class_name}: {duplicate_members}"
            )

    for function in [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    ]:
        if function.returns is None:
            raise RuntimeError(
                f"wheel type stub {function.name} is missing a return annotation"
            )
        arguments = (
            function.args.posonlyargs + function.args.args + function.args.kwonlyargs
        )
        for argument in arguments:
            if argument.arg in {"self", "cls"}:
                continue
            if argument.annotation is None:
                raise RuntimeError(
                    f"wheel type stub {function.name}.{argument.arg} "
                    "is missing an argument annotation"
                )

    known_names = {
        "None",
        "RuntimeError",
        "ValueError",
        "bool",
        "dict",
        "float",
        "int",
        "list",
        "object",
        "property",
        "staticmethod",
        "str",
        "tuple",
    }.union(imported_names, top_level_names)
    referenced_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    unresolved = sorted(referenced_names.difference(known_names))
    if unresolved:
        raise RuntimeError(f"wheel type stub has unresolved names: {unresolved}")

    expected_literals = {
        "BarrierDirection": {
            "down",
            "up",
        },
        "BarrierStyle": {
            "knock_in",
            "knock_out",
        },
        "DigitalPayout": {
            "asset",
            "cash",
        },
        "OptionSide": {
            "call",
            "put",
        },
        "RiskUnit": {
            "delta_one_percent_spot",
            "delta_raw",
            "gamma_one_percent_spot_squared",
            "gamma_raw",
            "vega_one_vol_point",
            "vega_raw",
        },
        "SmileDynamics": {
            "sticky_delta",
            "sticky_log_moneyness",
            "sticky_strike",
        },
        "ValidationPhase": {
            "current_schema",
            "declared_schema",
            "domain",
            "migration",
            "syntax_and_limits",
        },
        "VegaKtCovarianceLayout": {
            "price_and_bucket_variance_only",
            "full_bucket_matrix_row_major",
        },
        "VegaKtUnit": {
            "currency_per_unit_absolute_volatility",
            "currency_per_volatility_point",
        },
    }
    for alias, expected in sorted(expected_literals.items()):
        actual = literal_alias_values(assignments.get(alias))
        if actual != expected:
            raise RuntimeError(
                f"wheel type stub {alias} values must be {sorted(expected)}, "
                f"found {sorted(actual)}"
            )

    expected_return_names = {
        ("ValidationIssue", "phase"): "ValidationPhase",
    }
    for (class_name, method_name), expected in sorted(expected_return_names.items()):
        actual = function_return_name(class_methods.get(class_name, {}).get(method_name))
        if actual != expected:
            raise RuntimeError(
                f"wheel type stub {class_name}.{method_name} must return {expected}, "
                f"found {actual}"
            )

    expected_signature_shapes = {
        ("AsianObservation", "known"): {
            "positional": ["date", "weight", "fixing"],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("AsianObservation", "unknown"): {
            "positional": ["date", "weight"],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("DiscountCurve", "__init__"): {
            "positional": ["self", "curve_id", "times", "discount_factors"],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("DividendEvent", "fixed_cash"): {
            "positional": ["event_id", "ex_time", "amount"],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("DividendEvent", "fixed_cash_and_proportional"): {
            "positional": ["event_id", "ex_time", "fixed_cash", "beta"],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("DividendEvent", "proportional"): {
            "positional": ["event_id", "ex_time", "beta"],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("Engine", "pseudo_monte_carlo"): {
            "positional": ["master_seed", "independent_sampling_units"],
            "positional_defaults": {},
            "keyword_only": ["antithetic", "brownian_bridge"],
            "required_keyword_only": [],
            "keyword_only_defaults": {
                "antithetic": False,
                "brownian_bridge": False,
            },
        },
        ("Engine", "randomized_quasi_monte_carlo"): {
            "positional": ["points_per_scramble", "master_scramble_seed"],
            "positional_defaults": {},
            "keyword_only": ["scramble_count", "antithetic", "brownian_bridge"],
            "required_keyword_only": [],
            "keyword_only_defaults": {
                "scramble_count": 16,
                "antithetic": False,
                "brownian_bridge": True,
            },
        },
        ("EssviSlice", "__init__"): {
            "positional": ["self", "time", "theta", "psi", "rho_psi"],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("Market", "equity"): {
            "positional": [
                "currency_id",
                "underlying_id",
                "spot",
                "discount_curve",
                "dividend_curve",
            ],
            "positional_defaults": {},
            "keyword_only": ["discrete_dividends"],
            "required_keyword_only": [],
            "keyword_only_defaults": {"discrete_dividends": None},
        },
        ("Model", "black_76"): {
            "positional": ["volatility"],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("Model", "black_scholes"): {
            "positional": ["volatility"],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("Model", "local_volatility_from_essvi"): {
            "positional": [
                "slices",
                "terminal_theta_slope",
                "time_nodes",
                "log_forward_moneyness_nodes",
                "floor",
                "cap",
            ],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("Model", "local_volatility_from_grid"): {
            "positional": [
                "time_nodes",
                "log_forward_moneyness_nodes",
                "local_variances",
                "floor",
                "cap",
            ],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("Model", "local_volatility_from_grid_with_reporting_basis"): {
            "positional": [
                "time_nodes",
                "log_forward_moneyness_nodes",
                "local_variances",
                "floor",
                "cap",
                "reporting_maturity_nodes",
                "reporting_log_forward_moneyness_nodes",
                "reporting_implied_volatilities",
            ],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("Model", "local_volatility_from_standard_ssvi_heston_like"): {
            "positional": [
                "theta_times",
                "theta_values",
                "terminal_theta_slope",
                "rho",
                "lambda_",
                "time_nodes",
                "log_forward_moneyness_nodes",
                "floor",
                "cap",
            ],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("Model", "local_volatility_from_standard_ssvi_power_law"): {
            "positional": [
                "theta_times",
                "theta_values",
                "terminal_theta_slope",
                "rho",
                "eta",
                "gamma",
                "time_nodes",
                "log_forward_moneyness_nodes",
                "floor",
                "cap",
            ],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("PricingPlan", "compile"): {
            "positional": ["request"],
            "positional_defaults": {},
            "keyword_only": ["worker_threads", "reduction_block_size"],
            "required_keyword_only": ["worker_threads"],
            "keyword_only_defaults": {"reduction_block_size": None},
        },
        ("PricingRequest", "__init__"): {
            "positional": [
                "self",
                "valuation_date",
                "product",
                "market",
                "model",
                "engine",
                "risk",
            ],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("PricingRequest", "from_json"): {
            "positional": ["json"],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("PricingResult", "from_json"): {
            "positional": ["json"],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("Product", "arithmetic_asian"): {
            "positional": [
                "underlying_id",
                "currency_id",
                "strike",
                "notional",
                "side",
                "observations",
                "payment_date",
            ],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("Product", "barrier"): {
            "positional": [
                "underlying_id",
                "currency_id",
                "expiry",
                "strike",
                "barrier",
                "notional",
                "side",
                "direction",
                "style",
                "monitoring_dates",
                "payment_date",
            ],
            "positional_defaults": {},
            "keyword_only": ["rebate"],
            "required_keyword_only": [],
            "keyword_only_defaults": {"rebate": None},
        },
        ("Product", "digital"): {
            "positional": [
                "underlying_id",
                "currency_id",
                "expiry",
                "strike",
                "payout",
                "side",
                "payout_kind",
            ],
            "positional_defaults": {},
            "keyword_only": ["payment_date"],
            "required_keyword_only": [],
            "keyword_only_defaults": {"payment_date": None},
        },
        ("Product", "european_vanilla"): {
            "positional": [
                "underlying_id",
                "currency_id",
                "expiry",
                "strike",
                "notional",
                "side",
            ],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        ("Product", "fixed_lookback"): {
            "positional": [
                "underlying_id",
                "currency_id",
                "strike",
                "notional",
                "side",
                "monitoring_dates",
                "payment_date",
            ],
            "positional_defaults": {},
            "keyword_only": ["historical_extremum"],
            "required_keyword_only": [],
            "keyword_only_defaults": {"historical_extremum": None},
        },
        ("RiskRequest", "__init__"): {
            "positional": ["self"],
            "positional_defaults": {},
            "keyword_only": [
                "delta",
                "gamma_relative_bump",
                "gamma_absolute_bump",
                "vega",
                "vega_kt_maturity_nodes",
                "vega_kt_log_forward_moneyness_nodes",
                "vega_kt_relative_density_threshold",
                "vega_kt_full_bucket_covariance",
                "smile_dynamics",
                "checkpoint_interval",
                "aad_tile_capacity",
            ],
            "required_keyword_only": [],
            "keyword_only_defaults": {
                "delta": False,
                "gamma_relative_bump": None,
                "gamma_absolute_bump": None,
                "vega": False,
                "vega_kt_maturity_nodes": None,
                "vega_kt_log_forward_moneyness_nodes": None,
                "vega_kt_relative_density_threshold": None,
                "vega_kt_full_bucket_covariance": False,
                "smile_dynamics": "sticky_log_moneyness",
                "checkpoint_interval": None,
                "aad_tile_capacity": None,
            },
        },
    }
    for (class_name, method_name), expected in sorted(expected_signature_shapes.items()):
        actual = function_signature_shape(class_methods[class_name][method_name])
        if actual != expected:
            raise RuntimeError(
                f"wheel type stub {class_name}.{method_name} signature changed: "
                f"{actual} != {expected}"
            )

    expected_top_level_signature_shapes = {
        "request_json_schema": {
            "positional": [],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        "result_json_schema": {
            "positional": [],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
        "version": {
            "positional": [],
            "positional_defaults": {},
            "keyword_only": [],
            "required_keyword_only": [],
            "keyword_only_defaults": {},
        },
    }
    for function_name, expected in sorted(expected_top_level_signature_shapes.items()):
        actual = function_signature_shape(top_level_functions[function_name])
        if actual != expected:
            raise RuntimeError(
                f"wheel type stub {function_name} signature changed: "
                f"{actual} != {expected}"
            )

    expected_class_members = {
        "AsianObservation": {
            "__repr__",
            "date",
            "fixing",
            "known",
            "unknown",
            "weight",
        },
        "DiscountCurve": {
            "__init__",
            "__repr__",
            "curve_id",
        },
        "DiagnosticEstimate": {
            "__repr__",
            "value",
            "standard_error",
            "confidence_interval",
            "estimator",
            "effective_sampling_units",
        },
        "Diagnostics": {
            "__repr__",
            "master_seed",
            "estimator",
            "scramble_count",
            "direction_checksum",
            "scramble_checksum",
            "policy_version",
            "worker_threads",
            "reduction_block_size",
            "aad_tile_policy_version",
            "aad_tile_capacity",
            "checkpoint_policy_version",
            "checkpoint_interval",
            "antithetic",
            "discount_region",
            "dividend_region",
            "payoff_fingerprint",
            "delta_method",
            "gamma_method",
            "vega_method",
            "gamma_spot_bump",
            "validation_spot_bump",
            "validation_volatility_bump",
            "bump_policy_version",
            "delta_validation",
            "gamma_validation",
            "vega_validation",
            "warnings",
        },
        "DividendEvent": {
            "__repr__",
            "event_id",
            "ex_time",
            "fixed_cash",
            "fixed_cash_and_proportional",
            "proportional",
        },
        "Engine": {
            "__repr__",
            "pseudo_monte_carlo",
            "randomized_quasi_monte_carlo",
        },
        "EssviSlice": {
            "__init__",
            "__repr__",
            "psi",
            "rho_psi",
            "theta",
            "time",
        },
        "Market": {
            "__repr__",
            "equity",
        },
        "Model": {
            "__repr__",
            "black_76",
            "black_scholes",
            "local_volatility_from_essvi",
            "local_volatility_from_grid",
            "local_volatility_from_grid_with_reporting_basis",
            "local_volatility_from_standard_ssvi_heston_like",
            "local_volatility_from_standard_ssvi_power_law",
        },
        "PricingError": set(),
        "PricingPlan": {
            "__repr__",
            "compile",
            "evaluate",
            "request_fingerprint",
            "plan_fingerprint",
            "worker_threads",
            "reduction_block_size",
        },
        "PricingRequest": {
            "__init__",
            "__repr__",
            "from_json",
            "to_json",
            "to_pretty_json",
            "fingerprint",
        },
        "PricingResult": {
            "__repr__",
            "from_json",
            "value",
            "standard_error",
            "confidence_interval",
            "estimate",
            "delta",
            "delta_raw",
            "delta_market_scaled",
            "gamma",
            "gamma_raw",
            "gamma_market_scaled",
            "vega",
            "vega_raw",
            "vega_market_scaled",
            "vega_kt",
            "sampling_variance",
            "estimator_variance",
            "independent_sampling_units",
            "evaluated_paths",
            "diagnostics",
            "warnings",
            "replay_schema_version",
            "replay_request_fingerprint",
            "replay_library_version",
            "replay_platform",
            "to_json",
            "to_pretty_json",
        },
        "PricingWarning": {
            "__repr__",
            "code",
            "message",
        },
        "Product": {
            "__repr__",
            "arithmetic_asian",
            "barrier",
            "digital",
            "european_vanilla",
            "fixed_lookback",
        },
        "RiskEstimate": {
            "__repr__",
            "raw",
            "market_scaled",
            "raw_unit",
            "market_scaled_unit",
        },
        "RiskRequest": {
            "__init__",
            "__repr__",
        },
        "RiskValidation": {
            "__repr__",
            "bump_and_revalue",
            "bump_minus_primary",
        },
        "ValidationError": set(),
        "ValidationIssue": {
            "__repr__",
            "pointer",
            "instance_path",
            "phase",
            "schema_version",
            "document_kind",
            "code",
            "message",
            "to_dict",
            "__eq__",
            "__ne__",
        },
        "VegaKtBucketEstimate": {
            "__repr__",
            "raw_mean",
            "market_scaled_mean",
            "sample_variance",
            "price_covariance",
        },
        "VegaKtCoordinate": {
            "__repr__",
            "maturity",
            "log_moneyness",
            "implied_volatility",
        },
        "VegaKtProjection": {
            "__repr__",
            "scalar_vega",
            "signed_residual",
            "pre_projection",
            "reporting_stats",
        },
        "VegaKtReportingStats": {
            "__repr__",
            "left_edge_count",
            "right_edge_count",
            "left_edge_sensitivity",
            "right_edge_sensitivity",
        },
        "VegaKtResidualDiagnostics": {
            "__repr__",
            "active_domain_start_index",
            "active_domain_end_index",
            "active_domain_forward_index",
            "excluded_probability_mass",
            "signed_residual",
            "pre_projection",
            "reporting_stats",
        },
        "VegaKtResult": {
            "__repr__",
            "coordinates",
            "estimates",
            "raw_buckets",
            "full_bucket_covariance",
            "covariance_layout",
            "projection",
            "residual_diagnostics",
            "raw_unit",
            "market_scaled_unit",
            "policy_label",
            "truncation_order",
        },
    }
    unexpected_classes = sorted(set(class_members).difference(expected_class_members))
    if unexpected_classes:
        raise RuntimeError(f"wheel type stub has unexpected classes: {unexpected_classes}")
    missing_classes = sorted(set(expected_class_members).difference(class_members))
    if missing_classes:
        raise RuntimeError(f"wheel type stub is missing classes: {missing_classes}")
    for class_name, expected in sorted(expected_class_members.items()):
        actual = set(class_members.get(class_name, []))
        if actual != expected:
            raise RuntimeError(
                f"wheel type stub {class_name} members changed: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )

    expected_class_bases = {
        "ValidationError": ["ValueError"],
        "PricingError": ["RuntimeError"],
    }
    for class_name in sorted(expected_class_members):
        expected = expected_class_bases.get(class_name, [])
        actual = class_bases.get(class_name, [])
        if actual != expected:
            raise RuntimeError(
                f"wheel type stub {class_name} bases changed: {actual} != {expected}"
            )

    expected_static_methods = {
        ("AsianObservation", "known"),
        ("AsianObservation", "unknown"),
        ("DividendEvent", "fixed_cash"),
        ("DividendEvent", "fixed_cash_and_proportional"),
        ("DividendEvent", "proportional"),
        ("Engine", "pseudo_monte_carlo"),
        ("Engine", "randomized_quasi_monte_carlo"),
        ("Market", "equity"),
        ("Model", "black_76"),
        ("Model", "black_scholes"),
        ("Model", "local_volatility_from_essvi"),
        ("Model", "local_volatility_from_grid"),
        ("Model", "local_volatility_from_grid_with_reporting_basis"),
        ("Model", "local_volatility_from_standard_ssvi_heston_like"),
        ("Model", "local_volatility_from_standard_ssvi_power_law"),
        ("PricingPlan", "compile"),
        ("PricingRequest", "from_json"),
        ("PricingResult", "from_json"),
        ("Product", "arithmetic_asian"),
        ("Product", "barrier"),
        ("Product", "digital"),
        ("Product", "european_vanilla"),
        ("Product", "fixed_lookback"),
    }
    expected_properties = {
        ("AsianObservation", "date"),
        ("AsianObservation", "fixing"),
        ("AsianObservation", "weight"),
        ("DiscountCurve", "curve_id"),
        ("DiagnosticEstimate", "confidence_interval"),
        ("DiagnosticEstimate", "effective_sampling_units"),
        ("DiagnosticEstimate", "estimator"),
        ("DiagnosticEstimate", "standard_error"),
        ("DiagnosticEstimate", "value"),
        ("Diagnostics", "aad_tile_capacity"),
        ("Diagnostics", "aad_tile_policy_version"),
        ("Diagnostics", "antithetic"),
        ("Diagnostics", "bump_policy_version"),
        ("Diagnostics", "checkpoint_interval"),
        ("Diagnostics", "checkpoint_policy_version"),
        ("Diagnostics", "delta_method"),
        ("Diagnostics", "delta_validation"),
        ("Diagnostics", "discount_region"),
        ("Diagnostics", "direction_checksum"),
        ("Diagnostics", "dividend_region"),
        ("Diagnostics", "estimator"),
        ("Diagnostics", "gamma_method"),
        ("Diagnostics", "gamma_spot_bump"),
        ("Diagnostics", "gamma_validation"),
        ("Diagnostics", "master_seed"),
        ("Diagnostics", "payoff_fingerprint"),
        ("Diagnostics", "policy_version"),
        ("Diagnostics", "reduction_block_size"),
        ("Diagnostics", "scramble_checksum"),
        ("Diagnostics", "scramble_count"),
        ("Diagnostics", "validation_spot_bump"),
        ("Diagnostics", "validation_volatility_bump"),
        ("Diagnostics", "vega_method"),
        ("Diagnostics", "vega_validation"),
        ("Diagnostics", "warnings"),
        ("Diagnostics", "worker_threads"),
        ("DividendEvent", "event_id"),
        ("DividendEvent", "ex_time"),
        ("EssviSlice", "psi"),
        ("EssviSlice", "rho_psi"),
        ("EssviSlice", "theta"),
        ("EssviSlice", "time"),
        ("PricingPlan", "plan_fingerprint"),
        ("PricingPlan", "reduction_block_size"),
        ("PricingPlan", "request_fingerprint"),
        ("PricingPlan", "worker_threads"),
        ("PricingRequest", "fingerprint"),
        ("PricingResult", "confidence_interval"),
        ("PricingResult", "delta"),
        ("PricingResult", "delta_market_scaled"),
        ("PricingResult", "delta_raw"),
        ("PricingResult", "diagnostics"),
        ("PricingResult", "estimator_variance"),
        ("PricingResult", "estimate"),
        ("PricingResult", "evaluated_paths"),
        ("PricingResult", "gamma"),
        ("PricingResult", "gamma_market_scaled"),
        ("PricingResult", "gamma_raw"),
        ("PricingResult", "independent_sampling_units"),
        ("PricingResult", "replay_library_version"),
        ("PricingResult", "replay_platform"),
        ("PricingResult", "replay_request_fingerprint"),
        ("PricingResult", "replay_schema_version"),
        ("PricingResult", "sampling_variance"),
        ("PricingResult", "standard_error"),
        ("PricingResult", "value"),
        ("PricingResult", "vega"),
        ("PricingResult", "vega_kt"),
        ("PricingResult", "vega_market_scaled"),
        ("PricingResult", "vega_raw"),
        ("PricingResult", "warnings"),
        ("PricingWarning", "code"),
        ("PricingWarning", "message"),
        ("RiskEstimate", "market_scaled"),
        ("RiskEstimate", "market_scaled_unit"),
        ("RiskEstimate", "raw"),
        ("RiskEstimate", "raw_unit"),
        ("RiskValidation", "bump_and_revalue"),
        ("RiskValidation", "bump_minus_primary"),
        ("ValidationIssue", "code"),
        ("ValidationIssue", "document_kind"),
        ("ValidationIssue", "instance_path"),
        ("ValidationIssue", "message"),
        ("ValidationIssue", "phase"),
        ("ValidationIssue", "pointer"),
        ("ValidationIssue", "schema_version"),
        ("VegaKtBucketEstimate", "market_scaled_mean"),
        ("VegaKtBucketEstimate", "price_covariance"),
        ("VegaKtBucketEstimate", "raw_mean"),
        ("VegaKtBucketEstimate", "sample_variance"),
        ("VegaKtCoordinate", "implied_volatility"),
        ("VegaKtCoordinate", "log_moneyness"),
        ("VegaKtCoordinate", "maturity"),
        ("VegaKtProjection", "pre_projection"),
        ("VegaKtProjection", "reporting_stats"),
        ("VegaKtProjection", "scalar_vega"),
        ("VegaKtProjection", "signed_residual"),
        ("VegaKtReportingStats", "left_edge_count"),
        ("VegaKtReportingStats", "left_edge_sensitivity"),
        ("VegaKtReportingStats", "right_edge_count"),
        ("VegaKtReportingStats", "right_edge_sensitivity"),
        ("VegaKtResidualDiagnostics", "active_domain_end_index"),
        ("VegaKtResidualDiagnostics", "active_domain_forward_index"),
        ("VegaKtResidualDiagnostics", "active_domain_start_index"),
        ("VegaKtResidualDiagnostics", "excluded_probability_mass"),
        ("VegaKtResidualDiagnostics", "pre_projection"),
        ("VegaKtResidualDiagnostics", "reporting_stats"),
        ("VegaKtResidualDiagnostics", "signed_residual"),
        ("VegaKtResult", "coordinates"),
        ("VegaKtResult", "covariance_layout"),
        ("VegaKtResult", "estimates"),
        ("VegaKtResult", "full_bucket_covariance"),
        ("VegaKtResult", "market_scaled_unit"),
        ("VegaKtResult", "policy_label"),
        ("VegaKtResult", "projection"),
        ("VegaKtResult", "raw_buckets"),
        ("VegaKtResult", "raw_unit"),
        ("VegaKtResult", "residual_diagnostics"),
        ("VegaKtResult", "truncation_order"),
    }
    for class_name, method_name in sorted(expected_static_methods):
        decorators = decorator_names(class_methods[class_name][method_name])
        if "staticmethod" not in decorators:
            raise RuntimeError(
                f"wheel type stub {class_name}.{method_name} must be a staticmethod"
            )
    for class_name, method_name in sorted(expected_properties):
        decorators = decorator_names(class_methods[class_name][method_name])
        if "property" not in decorators:
            raise RuntimeError(
                f"wheel type stub {class_name}.{method_name} must be a property"
            )
    actual_static_methods = decorated_members(class_methods, "staticmethod")
    if actual_static_methods != expected_static_methods:
        raise RuntimeError(
            "wheel type stub staticmethod set changed: "
            f"missing={sorted(expected_static_methods - actual_static_methods)}, "
            f"unexpected={sorted(actual_static_methods - expected_static_methods)}"
        )
    actual_properties = decorated_members(class_methods, "property")
    if actual_properties != expected_properties:
        raise RuntimeError(
            "wheel type stub property set changed: "
            f"missing={sorted(expected_properties - actual_properties)}, "
            f"unexpected={sorted(actual_properties - expected_properties)}"
        )


def duplicates_in(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def literal_alias_values(node: ast.expr | None) -> set[str]:
    if not isinstance(node, ast.Subscript):
        return set()
    if not isinstance(node.value, ast.Name) or node.value.id != "Literal":
        return set()
    slice_value = node.slice
    elements = slice_value.elts if isinstance(slice_value, ast.Tuple) else [slice_value]
    values = set()
    for element in elements:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            values.add(element.value)
    return values


def function_return_name(node: ast.FunctionDef | None) -> str | None:
    if node is None:
        return None
    annotation = node.returns
    if isinstance(annotation, ast.Name):
        return annotation.id
    return None


def function_signature_shape(node: ast.FunctionDef) -> dict[str, object]:
    arguments = node.args
    if arguments.vararg is not None or arguments.kwarg is not None:
        raise RuntimeError(f"wheel type stub {node.name} uses variadic arguments")
    positional_names = [
        argument.arg for argument in arguments.posonlyargs + arguments.args
    ]
    positional_default_names = (
        positional_names[-len(arguments.defaults) :] if arguments.defaults else []
    )
    positional_defaults = {
        name: stub_default_value(default)
        for name, default in zip(positional_default_names, arguments.defaults)
    }
    keyword_only_names = [argument.arg for argument in arguments.kwonlyargs]
    required_keyword_only = [
        name
        for name, default in zip(keyword_only_names, arguments.kw_defaults)
        if default is None
    ]
    keyword_only_defaults = {
        name: stub_default_value(default)
        for name, default in zip(keyword_only_names, arguments.kw_defaults)
        if default is not None
    }
    return {
        "positional": positional_names,
        "positional_defaults": positional_defaults,
        "keyword_only": keyword_only_names,
        "required_keyword_only": required_keyword_only,
        "keyword_only_defaults": keyword_only_defaults,
    }


def stub_default_value(node: ast.expr) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    raise RuntimeError(f"wheel type stub uses unsupported default {ast.unparse(node)}")


def decorator_names(node: ast.FunctionDef) -> set[str]:
    return {
        decorator.id
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Name)
    }


def decorated_members(
    class_methods: dict[str, dict[str, ast.FunctionDef]],
    decorator_name: str,
) -> set[tuple[str, str]]:
    return {
        (class_name, method_name)
        for class_name, methods in class_methods.items()
        for method_name, method in methods.items()
        if decorator_name in decorator_names(method)
    }


def class_base_names(node: ast.ClassDef) -> list[str]:
    bases: list[str] = []
    for base in node.bases:
        if not isinstance(base, ast.Name):
            raise RuntimeError(f"wheel type stub {node.name} uses an unsupported base")
        bases.append(base.id)
    return bases


def verify_runtime_symbols(python: Path, stub_api: dict[str, object], version: str) -> None:
    code = """
import json
import rust_pricing

api = json.loads(input())
missing = [name for name in api["symbols"] if not hasattr(rust_pricing, name)]
expected_runtime_symbols = set(api["symbols"])
actual_runtime_symbols = {
    name for name in dir(rust_pricing) if not name.startswith("_")
}
actual_runtime_symbols.discard("rust_pricing")
if hasattr(rust_pricing, "__version__"):
    actual_runtime_symbols.add("__version__")
unexpected = sorted(actual_runtime_symbols.difference(expected_runtime_symbols))
if unexpected:
    missing.append("unexpected runtime symbols: " + ", ".join(unexpected))
for cls_name, members in api["class_members"].items():
    cls = getattr(rust_pricing, cls_name, None)
    if cls is not None:
        missing.extend(
            f"{cls_name}.{member}"
            for member in members
            if not hasattr(cls, member)
        )
        runtime_declared_members = set(cls.__dict__)
        expected_special_members = {
            member
            for member in members
            if member in {"__eq__", "__ne__", "__repr__"}
        }
        missing.extend(
            f"missing runtime special member {cls_name}.{member}"
            for member in sorted(expected_special_members)
            if member not in runtime_declared_members
        )
        expected_members = {
            member for member in members if not member.startswith("__")
        }
        actual_members = {
            member for member in runtime_declared_members if not member.startswith("_")
        }
        extra_members = sorted(actual_members.difference(expected_members))
        if extra_members:
            missing.append(
                f"unexpected runtime members on {cls_name}: "
                + ", ".join(extra_members)
            )
for cls_name, member in api["static_methods"]:
    cls = getattr(rust_pricing, cls_name, None)
    if cls is not None and not isinstance(cls.__dict__.get(member), staticmethod):
        missing.append(f"runtime member must be staticmethod: {cls_name}.{member}")
for cls_name, member in api["properties"]:
    cls = getattr(rust_pricing, cls_name, None)
    descriptor = cls.__dict__.get(member) if cls is not None else None
    if descriptor is not None and callable(descriptor):
        missing.append(f"runtime property member must not be callable: {cls_name}.{member}")
if rust_pricing.__version__ != api["version"]:
    missing.append("__version__")
if rust_pricing.version() != api["version"]:
    missing.append("version()")
if not isinstance(rust_pricing.__version__, str):
    missing.append("__version__ type")
if not isinstance(rust_pricing.version(), str):
    missing.append("version() type")
if not issubclass(rust_pricing.ValidationError, ValueError):
    missing.append("ValidationError base")
if not issubclass(rust_pricing.PricingError, RuntimeError):
    missing.append("PricingError base")
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
