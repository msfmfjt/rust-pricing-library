#!/usr/bin/env python3
"""Reject workspace dependencies that point upward through the architecture."""

from __future__ import annotations

import json
import sys


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


def main() -> int:
    metadata = json.load(sys.stdin)
    workspace_ids = set(metadata["workspace_members"])
    packages = {
        package["name"]: package
        for package in metadata["packages"]
        if package["id"] in workspace_ids
    }

    missing = sorted(set(LEVEL) - set(packages))
    unexpected = sorted(set(packages) - set(LEVEL))
    errors = []

    if missing:
        errors.append(f"missing workspace crates: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unclassified workspace crates: {', '.join(unexpected)}")

    for crate_name in sorted(set(packages) & set(LEVEL)):
        package = packages[crate_name]
        for dependency in package["dependencies"]:
            dependency_name = dependency["name"]
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


if __name__ == "__main__":
    raise SystemExit(main())
