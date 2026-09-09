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
    "crates/pricing-aad/Cargo.toml",
    "crates/pricing-core/Cargo.toml",
    "crates/pricing-market/Cargo.toml",
    "crates/pricing-mc/Cargo.toml",
    "crates/pricing-models/Cargo.toml",
    "crates/pricing-numerics/Cargo.toml",
    "crates/pricing-product/Cargo.toml",
    "crates/pricing-python/Cargo.toml",
    "crates/pricing-risk/Cargo.toml",
    "crates/pricing/Cargo.toml",
    "schemas/v1/pricing_request.schema.json",
    "schemas/v1/pricing_result.schema.json",
    "rust_pricing.pyi",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "crates/pricing-aad/src/lib.rs",
    "crates/pricing-core/src/lib.rs",
    "crates/pricing-market/src/lib.rs",
    "crates/pricing-mc/src/lib.rs",
    "crates/pricing-models/src/lib.rs",
    "crates/pricing-numerics/src/lib.rs",
    "crates/pricing-product/src/lib.rs",
    "crates/pricing-python/src/lib.rs",
    "crates/pricing-risk/src/lib.rs",
    "crates/pricing/src/lib.rs",
    "examples/python/european_bs.py",
    "examples/python/local_vol_vegakt.py",
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
        names = set()
        for member in package.getmembers():
            name = member.name
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise SystemExit(f"{archive}: unsafe archive member path: {name}")
            if member.isfile():
                names.add(name)

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
