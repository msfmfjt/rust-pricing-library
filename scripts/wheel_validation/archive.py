"""Wheel archive, metadata, lockfile, RECORD and SBOM validation."""
from __future__ import annotations
import base64
import csv
from email.message import Message
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tomllib
from urllib.parse import unquote
import uuid
from zipfile import ZipFile, ZipInfo

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
    "pricing-numerics": set(),
    "pricing": {"pricing-numerics"},
    "pricing-python": {"pricing"},
}


def selected_wheel(wheel_argument: str | None = None) -> Path:
    if wheel_argument is not None:
        wheel = Path(wheel_argument)
        if not wheel.is_file():
            raise RuntimeError(f"wheel does not exist: {wheel}")
        if wheel.suffix != ".whl":
            raise RuntimeError(f"expected a .whl file, got: {wheel}")
        return wheel
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
