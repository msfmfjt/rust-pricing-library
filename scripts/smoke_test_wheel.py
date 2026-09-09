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
import venv
from zipfile import ZipFile


def main() -> None:
    wheel = selected_wheel()
    expected_metadata = expected_project_metadata()

    with ZipFile(wheel) as archive:
        verify_wheel_archive_members(archive.namelist())
        members = {member for member in archive.namelist() if not member.endswith("/")}
        member_bytes = {member: archive.read(member) for member in members}
        verify_wheel_member_layout(members)
        if "rust_pricing/__init__.pyi" not in members:
            raise RuntimeError("wheel does not contain the rust_pricing.pyi type stub")
        stub = archive.read("rust_pricing/__init__.pyi")
        metadata = Parser().parsestr(read_dist_info_text(archive, members, "METADATA"))
        wheel_metadata = Parser().parsestr(read_dist_info_text(archive, members, "WHEEL"))
        record = read_dist_info_text(archive, members, "RECORD")
    if "rust_pricing/py.typed" not in members:
        raise RuntimeError("wheel does not contain the py.typed marker")
    verify_wheel_metadata(
        wheel,
        members,
        metadata,
        wheel_metadata,
        record,
        member_bytes,
        expected_metadata,
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


def verify_wheel_archive_members(member_names: list[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in member_names:
        if not name:
            raise RuntimeError("wheel contains an empty member path")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"wheel contains an unsafe member path: {name}")
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise RuntimeError(f"wheel contains duplicate member paths: {sorted(duplicates)}")


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
        if member.startswith("rust_pricing/")
        and (member.endswith(".so") or member.endswith(".pyd") or member.endswith(".dll"))
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


def verify_wheel_metadata(
    wheel: Path,
    members: set[str],
    metadata: Message,
    wheel_metadata: Message,
    record: str,
    member_bytes: dict[str, bytes],
    expected_metadata: dict[str, str],
) -> None:
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
    if "from .rust_pricing import *" not in init_py:
        raise RuntimeError("wheel __init__.py must re-export the extension module")
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
    if "Rust Pricing Library" not in metadata.get_payload():
        raise RuntimeError("wheel metadata does not include the README payload")
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
    verify_cyclonedx_sbom(members, member_bytes, expected_metadata["Version"])
    record_members = verify_wheel_record(record, members, member_bytes)
    if "rust_pricing/__init__.pyi" not in record_members or "rust_pricing/py.typed" not in record_members:
        raise RuntimeError("wheel RECORD does not list stub and py.typed entries")


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


def verify_cyclonedx_sbom(
    members: set[str],
    member_bytes: dict[str, bytes],
    version: str,
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
    metadata = sbom.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("wheel CycloneDX SBOM metadata must be an object")
    component = metadata.get("component")
    if not isinstance(component, dict):
        raise RuntimeError("wheel CycloneDX SBOM metadata.component must be an object")
    if component.get("name") != "pricing-python" or component.get("version") != version:
        raise RuntimeError("wheel CycloneDX SBOM root component metadata mismatch")

    components = sbom.get("components")
    if not isinstance(components, list):
        raise RuntimeError("wheel CycloneDX SBOM components must be an array")
    component_names = {
        component.get("name")
        for component in components
        if isinstance(component, dict) and isinstance(component.get("name"), str)
    }
    required_workspace_components = {
        "pricing",
        "pricing-aad",
        "pricing-core",
        "pricing-market",
        "pricing-mc",
        "pricing-models",
        "pricing-numerics",
        "pricing-product",
        "pricing-risk",
    }
    missing = sorted(required_workspace_components.difference(component_names))
    if missing:
        raise RuntimeError(f"wheel CycloneDX SBOM is missing components: {missing}")

    dependencies = sbom.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise RuntimeError("wheel CycloneDX SBOM dependencies must be a non-empty array")
    for index, dependency in enumerate(dependencies, start=1):
        if not isinstance(dependency, dict):
            raise RuntimeError(f"wheel CycloneDX SBOM dependency {index} must be an object")
        if not isinstance(dependency.get("ref"), str) or not dependency["ref"]:
            raise RuntimeError(f"wheel CycloneDX SBOM dependency {index} has no ref")
        depends_on = dependency.get("dependsOn")
        if depends_on is not None and (not isinstance(depends_on, list) or not all(
            isinstance(item, str) and item for item in depends_on
        )):
            raise RuntimeError(
                f"wheel CycloneDX SBOM dependency {index} must list string dependsOn refs"
            )


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
    wheel_members: set[str],
    member_bytes: dict[str, bytes],
) -> set[str]:
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


def verify_stub_static_shape(tree: ast.Module) -> None:
    imported_names: set[str] = set()
    top_level_names: list[str] = []
    class_members: dict[str, list[str]] = {}
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
                class_members[node.name] = [
                    member.name
                    for member in node.body
                    if isinstance(member, ast.FunctionDef)
                ]

    duplicates = sorted(duplicates_in(top_level_names))
    if duplicates:
        raise RuntimeError(f"wheel type stub has duplicate top-level definitions: {duplicates}")

    for class_name, members in sorted(class_members.items()):
        duplicate_members = sorted(duplicates_in(members))
        if duplicate_members:
            raise RuntimeError(
                f"wheel type stub has duplicate members in {class_name}: {duplicate_members}"
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

    expected_class_members = {
        "DiagnosticEstimate": {
            "value",
            "standard_error",
            "confidence_interval",
            "estimator",
            "effective_sampling_units",
        },
        "Diagnostics": {
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
        "PricingPlan": {
            "compile",
            "evaluate",
            "request_fingerprint",
            "plan_fingerprint",
            "worker_threads",
            "reduction_block_size",
        },
        "PricingRequest": {
            "__init__",
            "from_json",
            "to_json",
            "to_pretty_json",
            "fingerprint",
        },
        "PricingResult": {
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
            "code",
            "message",
        },
        "RiskEstimate": {
            "raw",
            "market_scaled",
            "raw_unit",
            "market_scaled_unit",
        },
        "RiskValidation": {
            "bump_and_revalue",
            "bump_minus_primary",
        },
        "ValidationIssue": {
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
            "raw_mean",
            "market_scaled_mean",
            "sample_variance",
            "price_covariance",
        },
        "VegaKtCoordinate": {
            "maturity",
            "log_moneyness",
            "implied_volatility",
        },
        "VegaKtProjection": {
            "scalar_vega",
            "signed_residual",
            "pre_projection",
            "reporting_stats",
        },
        "VegaKtReportingStats": {
            "left_edge_count",
            "right_edge_count",
            "left_edge_sensitivity",
            "right_edge_sensitivity",
        },
        "VegaKtResidualDiagnostics": {
            "active_domain_start_index",
            "active_domain_end_index",
            "active_domain_forward_index",
            "excluded_probability_mass",
            "signed_residual",
            "pre_projection",
            "reporting_stats",
        },
        "VegaKtResult": {
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
    for class_name, expected in sorted(expected_class_members.items()):
        actual = set(class_members.get(class_name, []))
        if not expected.issubset(actual):
            raise RuntimeError(
                f"wheel type stub {class_name} is missing members: "
                f"{sorted(expected - actual)}"
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
