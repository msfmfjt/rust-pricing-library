"""Validate the retained Rust source archive contents."""

from __future__ import annotations

from pathlib import PurePosixPath
import sys
import tarfile
import tomllib


REQUIRED_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "rust-toolchain.toml",
    "pyproject.toml",
    "docs/architecture-v0.1.md",
    "docs/benchmarking-v0.1.md",
    "docs/european-bs-conformance-v0.1.md",
    "docs/european-bs-diagnostics-v0.1.md",
    "docs/european-bs-roadmap-v0.1.md",
    "docs/local-vol-vegakt-conformance-v0.1.md",
    "docs/local-vol-vegakt-diagnostics-v0.1.md",
    "docs/local-vol-vegakt-numerical-contracts-v0.1.md",
    "docs/local-vol-vegakt-roadmap-v0.1.md",
    "docs/requirements-v1.0.md",
    "docs/wire-schema-compatibility.md",
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
    "fixtures/acceptance/README.md",
    "fixtures/acceptance/european_bs_analytical.csv",
    "fixtures/local-vol/README.md",
    "fixtures/local-vol/reference-cases-v0.1.json",
    "fixtures/replay/README.md",
    "fixtures/replay/european_bs-macos-aarch64.json",
    "fixtures/replay/european_bs-windows-x86_64.json",
    "fixtures/replay/local_volatility-macos-aarch64.json",
    "fixtures/replay/local_volatility-windows-x86_64.json",
    "fixtures/v1/pricing_request.golden.json",
    "fixtures/v1/pricing_result.golden.json",
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

CRATE_MANIFESTS = {
    name for name in REQUIRED_FILES if name.startswith("crates/") and name.endswith("/Cargo.toml")
}

INTERNAL_WORKSPACE_DEPENDENCIES = {
    "pricing-core": "crates/pricing-core",
    "pricing-numerics": "crates/pricing-numerics",
    "pricing-aad": "crates/pricing-aad",
    "pricing-market": "crates/pricing-market",
    "pricing-product": "crates/pricing-product",
    "pricing-models": "crates/pricing-models",
    "pricing-mc": "crates/pricing-mc",
    "pricing-risk": "crates/pricing-risk",
    "pricing": "crates/pricing",
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

        check_cargo_manifests(package, archive)

    return 0


def has_forbidden_part(name: str) -> bool:
    return any(part in FORBIDDEN_PARTS for part in PurePosixPath(name).parts)


def check_cargo_manifests(package: tarfile.TarFile, archive: str) -> None:
    workspace = read_toml(package, "Cargo.toml")
    workspace_package = workspace.get("workspace", {}).get("package", {})
    version = workspace_package.get("version")
    if version != "0.1.0":
        raise SystemExit(f"{archive}: workspace package version must be 0.1.0")
    if workspace_package.get("repository") != "https://github.com/msfmfjt/rust-pricing-library":
        raise SystemExit(f"{archive}: workspace repository URL mismatch")

    workspace_dependencies = workspace.get("workspace", {}).get("dependencies", {})
    for dependency, expected_path in INTERNAL_WORKSPACE_DEPENDENCIES.items():
        spec = workspace_dependencies.get(dependency)
        if not isinstance(spec, dict):
            raise SystemExit(f"{archive}: missing workspace dependency {dependency}")
        if spec.get("version") != version or spec.get("path") != expected_path:
            raise SystemExit(
                f"{archive}: workspace dependency {dependency} must declare "
                f"version {version} and path {expected_path}"
            )

    for manifest in sorted(CRATE_MANIFESTS):
        document = read_toml(package, manifest)
        package_section = document.get("package", {})
        if package_section.get("publish") is not False:
            raise SystemExit(f"{archive}: {manifest} must declare publish = false")
        for key in ["version", "edition", "rust-version", "authors", "repository"]:
            value = package_section.get(key)
            if not isinstance(value, dict) or value.get("workspace") is not True:
                raise SystemExit(f"{archive}: {manifest} package.{key} must use workspace")


def read_toml(package: tarfile.TarFile, name: str) -> dict[str, object]:
    member = package.extractfile(name)
    if member is None:
        raise SystemExit(f"missing TOML member: {name}")
    try:
        document = tomllib.loads(member.read().decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"{name}: invalid TOML: {exc}") from exc
    if not isinstance(document, dict):
        raise SystemExit(f"{name}: TOML root must be a table")
    return document


if __name__ == "__main__":
    raise SystemExit(main())
