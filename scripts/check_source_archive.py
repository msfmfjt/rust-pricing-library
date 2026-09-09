"""Validate the retained Rust source archive contents."""

from __future__ import annotations

from pathlib import PurePosixPath
import sys
import tarfile


REQUIRED_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "rust-toolchain.toml",
    "pyproject.toml",
    "crates/pricing/Cargo.toml",
    "crates/pricing-core/Cargo.toml",
    "crates/pricing-python/Cargo.toml",
    "schemas/v1/pricing_request.schema.json",
    "schemas/v1/pricing_result.schema.json",
    "rust_pricing.pyi",
}

FORBIDDEN_PARTS = {
    ".git",
    "target",
    "dist",
    ".venv",
    ".wheel-smoke-venv",
    "benchmark-results",
    "__pycache__",
}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_source_archive.py <archive.tar.gz>")

    archive = sys.argv[1]
    with tarfile.open(archive, "r:gz") as package:
        names = {member.name for member in package.getmembers() if member.isfile()}

    missing = sorted(REQUIRED_FILES.difference(names))
    if missing:
        raise SystemExit(f"{archive}: missing required source files: {missing}")

    forbidden = sorted(name for name in names if has_forbidden_part(name))
    if forbidden:
        raise SystemExit(f"{archive}: archive contains generated/private files: {forbidden[:10]}")

    return 0


def has_forbidden_part(name: str) -> bool:
    return any(part in FORBIDDEN_PARTS for part in PurePosixPath(name).parts)


if __name__ == "__main__":
    raise SystemExit(main())
