#!/usr/bin/env python3
"""Reject workspace dependencies that point upward through the architecture."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


LEVEL = {
    "pricing-core": 0,
    "pricing-numerics": 1,
    "pricing-product": 1,
    "pricing-aad": 2,
    "pricing-market": 2,
    "pricing-models": 3,
    "pricing-mc": 4,
    "pricing-risk": 5,
    "pricing": 6,
    "pricing-python": 7,
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
            if LEVEL[dependency_name] >= LEVEL[crate_name]:
                errors.append(
                    f"{crate_name} (level {LEVEL[crate_name]}) depends upward on "
                    f"{dependency_name} (level {LEVEL[dependency_name]})"
                )

    if errors:
        for error in errors:
            print(f"dependency-direction error: {error}", file=sys.stderr)
        return 1

    print("workspace dependency direction is valid")
    return 0


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


def relative_manifest_path(manifest_path: str, workspace_root: Path) -> str:
    try:
        return Path(manifest_path).resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError as exc:
        raise SystemExit(
            f"dependency-direction error: manifest_path escapes workspace_root: {manifest_path}"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
