#!/usr/bin/env python3
"""Reject workspace dependencies that point upward through the architecture."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


LEVEL = {
    "pricing-numerics": 0,
    "pricing": 1,
    "pricing-python": 2,
}

EXPECTED_DEPENDENCIES = {
    "pricing-numerics": set(),
    "pricing": {"pricing-numerics"},
    "pricing-python": {"pricing"},
}

REMOVED_CRATES = {
    "pricing-core", "pricing-aad", "pricing-market", "pricing-product",
    "pricing-models", "pricing-mc", "pricing-risk",
}

EXPECTED_MANIFEST_PATHS = {
    name: f"crates/{name}/Cargo.toml" for name in LEVEL
}


def main() -> int:
    metadata = load_metadata()
    workspace_ids = metadata["workspace_members"]
    workspace_root = Path(metadata["workspace_root"])
    packages = workspace_packages(metadata["packages"], workspace_ids, workspace_root)

    missing = sorted(set(LEVEL) - set(packages))
    unexpected = sorted(set(packages) - set(LEVEL))
    errors = []

    if missing:
        errors.append(f"missing workspace crates: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unclassified workspace crates: {', '.join(unexpected)}")

    for crate_name in sorted(set(packages) & set(LEVEL)):
        package = packages[crate_name]
        seen_dependencies = set()
        for dependency in package["dependencies"]:
            dependency_name = dependency.get("name")
            if not isinstance(dependency_name, str) or not dependency_name:
                errors.append(f"{crate_name} has a dependency without a non-empty string name")
                continue
            if dependency_name in seen_dependencies:
                errors.append(f"{crate_name} lists dependency {dependency_name} more than once")
                continue
            seen_dependencies.add(dependency_name)
            if dependency_name not in LEVEL:
                continue
            if dependency_name not in packages:
                errors.append(
                    f"{crate_name} depends on missing workspace crate {dependency_name}"
                )
                continue
            check_workspace_dependency(
                crate_name,
                dependency,
                packages[dependency_name],
                workspace_root,
                errors,
            )
            if LEVEL[dependency_name] >= LEVEL[crate_name]:
                errors.append(
                    f"{crate_name} (level {LEVEL[crate_name]}) depends upward on "
                    f"{dependency_name} (level {LEVEL[dependency_name]})"
                )

    for name, package in packages.items():
        dependencies = {entry["name"] for entry in package["dependencies"]}
        if dependencies & REMOVED_CRATES:
            errors.append(f"{name} depends on removed crates: {sorted(dependencies & REMOVED_CRATES)}")
        if name in EXPECTED_DEPENDENCIES and dependencies & set(LEVEL) != EXPECTED_DEPENDENCIES[name]:
            errors.append(f"{name} must depend on exactly {sorted(EXPECTED_DEPENDENCIES[name])}")
        if name != "pricing-python" and dependencies & {"pyo3", "numpy"}:
            errors.append(f"{name} must not depend on Python bindings")
    check_internal_boundaries(workspace_root, errors)

    if errors:
        for error in errors:
            print(f"dependency-direction error: {error}", file=sys.stderr)
        return 1

    print("workspace dependency direction is valid")
    return 0


def check_internal_boundaries(workspace_root: Path, errors: list[str]) -> None:
    """Small source guard in addition to Rust privacy, not a general Rust parser.

    Compatibility re-exports in mod.rs are intentional. Domain definitions may
    not import execution state; MC/process/sampling/payoff implementations may
    not call report assembly. Inherent compatibility methods live in engine.
    """
    import re

    source = workspace_root / "crates/pricing/src"
    if re.search(r"(?m)^pub(?:\([^)]*\))?\s+mod\s+engine", (source / "lib.rs").read_text()):
        errors.append("engine module must remain private")
    for folder in ["core", "market", "models", "product", "risk"]:
        for path in (source / folder).rglob("*.rs"):
            if path.name == "mod.rs":
                continue  # Existing public compatibility re-exports.
            code = re.sub(r"//[^\n]*", "", path.read_text())
            if re.search(r"crate::engine::|\b(?:DeterministicExecutor|SoaWorkspace)\b", code):
                errors.append(f"domain definition imports execution state: {path.relative_to(source)}")
    for folder in ["mc", "sampling", "processes", "payoff"]:
        for path in (source / "engine" / folder).rglob("*.rs"):
            code = re.sub(r"//[^\n]*", "", path.read_text())
            if re.search(r"crate::engine::risk::|\b(?:RiskReport|VegaKtResult)\b|\.build_(?:risk_report|vega_kt_result)\(", code):
                errors.append(f"lower execution layer assembles risk reports: {path.relative_to(source)}")


def load_metadata() -> dict[str, Any]:
    try:
        metadata = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"dependency-direction error: invalid cargo metadata JSON: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise SystemExit("dependency-direction error: cargo metadata root must be an object")
    workspace_members = metadata.get("workspace_members")
    if not isinstance(workspace_members, list) or not all(
        isinstance(member, str) for member in workspace_members
    ):
        raise SystemExit(
            "dependency-direction error: workspace_members must be a string array"
        )
    if len(workspace_members) != len(set(workspace_members)):
        raise SystemExit(
            "dependency-direction error: workspace_members contains duplicate package IDs"
        )
    packages = metadata.get("packages")
    if not isinstance(packages, list) or not all(
        isinstance(package, dict) for package in packages
    ):
        raise SystemExit("dependency-direction error: packages must be an object array")
    workspace_root = metadata.get("workspace_root")
    if not isinstance(workspace_root, str) or not workspace_root:
        raise SystemExit("dependency-direction error: workspace_root must be a string")
    return metadata


def workspace_packages(
    metadata_packages: list[dict[str, Any]],
    workspace_ids: list[str],
    workspace_root: Path,
) -> dict[str, dict[str, Any]]:
    workspace_id_set = set(workspace_ids)
    packages_by_id = {}
    duplicate_ids = set()
    for package in metadata_packages:
        package_id = package.get("id")
        if not isinstance(package_id, str):
            raise SystemExit("dependency-direction error: package id must be a string")
        if package_id in packages_by_id:
            duplicate_ids.add(package_id)
        packages_by_id[package_id] = package
    if duplicate_ids:
        raise SystemExit(
            "dependency-direction error: packages contains duplicate package IDs: "
            + ", ".join(sorted(duplicate_ids))
        )

    selected: dict[str, dict[str, Any]] = {}
    duplicate_names = set()
    missing_ids = []
    for package_id in workspace_ids:
        package = packages_by_id.get(package_id)
        if package is None:
            missing_ids.append(package_id)
            continue
        name = package.get("name")
        dependencies = package.get("dependencies")
        if not isinstance(name, str) or not name:
            raise SystemExit(
                f"dependency-direction error: {package_id}: "
                "package name must be a non-empty string"
            )
        if not isinstance(dependencies, list) or not all(
            isinstance(dependency, dict) for dependency in dependencies
        ):
            raise SystemExit(
                f"dependency-direction error: {name}: dependencies must be an object array"
            )
        if name in selected:
            duplicate_names.add(name)
        expected_manifest = EXPECTED_MANIFEST_PATHS.get(name)
        if expected_manifest is not None:
            manifest_path = package.get("manifest_path")
            if not isinstance(manifest_path, str) or not manifest_path:
                raise SystemExit(
                    f"dependency-direction error: {name}: "
                    "manifest_path must be a non-empty string"
                )
            actual_manifest = relative_manifest_path(manifest_path, workspace_root)
            if actual_manifest != expected_manifest:
                raise SystemExit(
                    f"dependency-direction error: {name}: manifest_path mismatch: "
                    f"{actual_manifest} != {expected_manifest}"
                )
        selected[name] = package
    if missing_ids:
        raise SystemExit(
            "dependency-direction error: workspace_members references missing packages: "
            + ", ".join(missing_ids)
        )
    if duplicate_names:
        raise SystemExit(
            "dependency-direction error: multiple workspace packages share crate names: "
            + ", ".join(sorted(duplicate_names))
        )
    if len(selected) != len(workspace_id_set):
        raise SystemExit(
            "dependency-direction error: workspace package selection is inconsistent"
        )
    return selected


def check_workspace_dependency(
    crate_name: str,
    dependency: dict[str, Any],
    target_package: dict[str, Any],
    workspace_root: Path,
    errors: list[str],
) -> None:
    dependency_name = dependency["name"]
    if dependency.get("source") is not None:
        errors.append(
            f"{crate_name} dependency {dependency_name} must use a local workspace source"
        )

    dependency_path = dependency.get("path")
    if not isinstance(dependency_path, str) or not dependency_path:
        errors.append(
            f"{crate_name} dependency {dependency_name} must declare a local path"
        )
    else:
        actual_path = relative_dependency_path(dependency_path, workspace_root)
        expected_path = f"crates/{dependency_name}"
        if actual_path != expected_path:
            errors.append(
                f"{crate_name} dependency {dependency_name} path mismatch: "
                f"{actual_path} != {expected_path}"
            )

    target_version = target_package.get("version")
    if not isinstance(target_version, str) or not target_version:
        errors.append(f"workspace crate {dependency_name} has an invalid version")
        return
    expected_requirement = f"^{target_version}"
    if dependency.get("req") != expected_requirement:
        errors.append(
            f"{crate_name} dependency {dependency_name} version requirement mismatch: "
            f"{dependency.get('req')!r} != {expected_requirement!r}"
        )


def relative_manifest_path(manifest_path: str, workspace_root: Path) -> str:
    try:
        return Path(manifest_path).resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError as exc:
        raise SystemExit(
            f"dependency-direction error: manifest_path escapes workspace_root: {manifest_path}"
        ) from exc


def relative_dependency_path(dependency_path: str, workspace_root: Path) -> str:
    try:
        return Path(dependency_path).resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError as exc:
        raise SystemExit(
            "dependency-direction error: dependency path escapes workspace_root: "
            f"{dependency_path}"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
