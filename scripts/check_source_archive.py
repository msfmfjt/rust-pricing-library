"""Validate the retained Rust source archive contents."""

from __future__ import annotations

from pathlib import PurePosixPath
from pathlib import Path
import sys
import tarfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "rust-toolchain.toml",
    "pyproject.toml",
    ".github/workflows/ci.yml",
    "docs/architecture-v0.1.md",
    "docs/benchmarking-v0.1.md",
    "docs/european-bs-conformance-v0.1.md",
    "docs/european-bs-diagnostics-v0.1.md",
    "docs/european-bs-roadmap-v0.1.md",
    "docs/local-vol-vegakt-conformance-v0.1.md",
    "docs/local-vol-vegakt-diagnostics-v0.1.md",
    "docs/local-vol-vegakt-numerical-contracts-v0.1.md",
    "docs/local-vol-vegakt-roadmap-v0.1.md",
    "docs/release-readiness-v0.1.md",
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
    "crates/pricing-mc/data/README.md",
    "crates/pricing-mc/data/joe-kuo-6.21201-u32be.bin",
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
    "scripts/check_benchmark_reports.py",
    "scripts/check_dependency_direction.py",
    "scripts/check_local_vol_reference_fixture.py",
    "scripts/check_markdown_links.py",
    "scripts/check_replay_fixture.py",
    "scripts/check_schemas.py",
    "scripts/check_source_archive.py",
    "scripts/run_benchmark_suite.py",
    "scripts/smoke_test_wheel.py",
    "benchmarks/python_european_bs.py",
    "examples/python/european_bs.py",
    "examples/python/local_vol_vegakt.py",
    "tests/python/test_smoke.py",
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

REQUIRED_CI_SNIPPETS = {
    "python scripts/check_local_vol_reference_fixture.py",
    "python scripts/check_schemas.py",
    "python scripts/check_markdown_links.py",
    "cargo fmt --all --check",
    "cargo clippy --locked --workspace --all-targets --all-features -- -D warnings",
    "cargo test --locked --workspace --all-features --exclude pricing-python",
    "cargo test --locked -p pricing-python",
    "cargo test --locked -p pricing --test statistical_acceptance -- --ignored --nocapture",
    "cargo doc --locked --workspace --all-features --no-deps",
    "python -m maturin build --locked --release --out dist",
    "git archive --format=tar.gz --output",
    "python scripts/check_source_archive.py",
    "python scripts/smoke_test_wheel.py",
    "python scripts/run_benchmark_suite.py",
    "python scripts/check_benchmark_reports.py benchmark-results",
    "actions/upload-artifact@v4",
}

FORBIDDEN_PARTS = {
    ".git",
    "target",
    "dist",
    "wheelhouse",
    ".venv",
    ".wheel-smoke-venv",
    "benchmark-results",
    "__pycache__",
    ".idea",
    ".vscode",
}

FORBIDDEN_NAMES = {
    ".DS_Store",
    "uv.lock",
}

FORBIDDEN_SUFFIXES = {
    ".egg-info",
    ".pyc",
    ".pyo",
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
            elif not member.isdir():
                raise SystemExit(
                    f"{archive}: unsupported archive member type for {name}: {member.type!r}"
                )

        forbidden = sorted(name for name in names if has_forbidden_part(name))
        if forbidden:
            raise SystemExit(f"{archive}: archive contains generated/private files: {forbidden[:10]}")

        expected = REQUIRED_FILES.union(repository_source_files())
        missing = sorted(expected.difference(names))
        if missing:
            raise SystemExit(f"{archive}: missing required source files: {missing}")

        check_cargo_manifests(package, archive)
        check_ci_workflow(package, archive)

    return 0


def has_forbidden_part(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        any(part in FORBIDDEN_PARTS for part in path.parts)
        or path.name in FORBIDDEN_NAMES
        or any(path.name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)
    )


def repository_source_files() -> set[str]:
    patterns = [
        ("crates", "*.rs"),
        ("benchmarks", "*.py"),
        ("examples/python", "*.py"),
        ("tests/python", "*.py"),
        ("scripts", "*.py"),
    ]
    files: set[str] = set()
    for root, pattern in patterns:
        files.update(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / root).rglob(pattern)
            if path.is_file()
        )
    return files


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


def check_ci_workflow(package: tarfile.TarFile, archive: str) -> None:
    workflow = read_text(package, ".github/workflows/ci.yml")
    missing = sorted(snippet for snippet in REQUIRED_CI_SNIPPETS if snippet not in workflow)
    if missing:
        raise SystemExit(f"{archive}: CI workflow is missing required gates: {missing}")


def read_text(package: tarfile.TarFile, name: str) -> str:
    member = package.extractfile(name)
    if member is None:
        raise SystemExit(f"missing text member: {name}")
    try:
        return member.read().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"{name}: invalid UTF-8: {exc}") from exc


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
