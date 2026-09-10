"""Validate the retained Rust source archive contents."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from pathlib import Path
import sys
import tarfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = {
    ".gitattributes",
    ".gitignore",
    "Cargo.lock",
    "Cargo.toml",
    "CONTRIBUTING.md",
    "rust-toolchain.toml",
    "rustfmt.toml",
    "pyproject.toml",
    ".github/workflows/ci.yml",
    "docs/adr/0000-template.md",
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
    "docs/requirements-change-template.md",
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
    "python scripts/check_replay_fixture.py benchmark-results/replay.json",
    "python scripts/check_replay_fixture.py benchmark-results/local-volatility-replay.json",
    "python scripts/check_benchmark_reports.py benchmark-results",
    "cargo metadata --locked --format-version 1 --no-deps | python scripts/check_dependency_direction.py",
    'python-version: "3.12"',
    "target: aarch64-apple-darwin",
    "target: x86_64-pc-windows-msvc",
    "actions/upload-artifact@v4",
    "name: rust-pricing-${{ matrix.target }}-cp312",
    "name: rust-pricing-source-${{ matrix.target }}",
    "name: benchmark-${{ matrix.target }}",
    "retention-days: 14",
}

REQUIRED_CI_SNIPPET_COUNTS = {
    'python-version: "3.12"': 2,
    "python -m maturin build --locked --release --out dist": 2,
    "python scripts/smoke_test_wheel.py": 2,
    "actions/upload-artifact@v4": 3,
    "retention-days: 14": 3,
}

REQUIRED_README_SNIPPETS = {
    "python3 scripts/check_local_vol_reference_fixture.py",
    "python3 scripts/check_schemas.py",
    "python3 scripts/check_markdown_links.py",
    "git archive --format=tar.gz --output /tmp/rust-pricing-source-check.tar.gz HEAD",
    "python3 scripts/check_source_archive.py /tmp/rust-pricing-source-check.tar.gz",
    "cargo fmt --all --check",
    "cargo clippy --locked --workspace --all-targets --all-features -- -D warnings",
    "cargo test --locked --workspace --all-features --exclude pricing-python",
    "cargo test --locked -p pricing-python",
    "cargo test --locked -p pricing --test statistical_acceptance -- --ignored --nocapture",
    "cargo doc --locked --workspace --all-features --no-deps",
    "cargo metadata --locked --format-version 1 --no-deps | python3 scripts/check_dependency_direction.py",
    "python -m maturin develop --locked",
    "python -m unittest discover -s tests/python -v",
    "python -m maturin build --locked --release --out dist",
    "python scripts/smoke_test_wheel.py",
    "Run `scripts/smoke_test_wheel.py` with the same CPython ABI as the built wheel",
    "The smoke test rejects ABI\nmismatches before installation.",
    "python scripts/run_benchmark_suite.py",
    "python scripts/check_replay_fixture.py benchmark-results/replay.json",
    "python scripts/check_replay_fixture.py benchmark-results/local-volatility-replay.json",
    "python scripts/check_benchmark_reports.py benchmark-results",
}

REQUIRED_CONTRIBUTING_SNIPPETS = REQUIRED_README_SNIPPETS

REQUIRED_WHEEL_SMOKE_SNIPPETS = {
    "verify_runtime_symbols(python, stub_api, metadata[\"Version\"])",
    "verify_wheel_text_members(member_bytes)",
    "verify_wheel_archive_members(archive.infolist())",
    "EXPECTED_METADATA_FIELDS = (",
    "EXPECTED_WHEEL_FIELDS = (",
    "verify_message_fields(metadata, EXPECTED_METADATA_FIELDS, \"METADATA\")",
    "wheel {name} fields changed",
    "must appear exactly once",
    "wheel extension member must be executable",
    "wheel data member must not be executable",
    "wheel data member must not be world-writable",
    "wheel text member must be valid UTF-8",
    "wheel text member has a UTF-8 BOM",
    "unexpected runtime symbols",
    "unexpected runtime members on",
    "expected_class_members",
    "decorator_names",
    "decorated_members",
    "wheel type stub staticmethod set changed",
    "wheel type stub property set changed",
    "class_base_names",
    "wheel type stub {class_name} bases changed",
    "ValidationError base",
    "PricingError base",
    "function_signature_shape",
    "wheel type stub {class_name}.{method_name} signature changed",
    "expected_top_level_signature_shapes",
    "wheel type stub {function_name} signature changed",
    "must be a staticmethod",
    "must be a property",
    "\"__repr__\"",
    "missing runtime special member",
    "is missing a return annotation",
    "is missing an argument annotation",
    "wheel type stub has unresolved names",
    "wheel type stub has duplicate top-level definitions",
    "wheel RECORD row {index} must use a sha256 digest",
    "wheel RECORD must end with LF",
    "wheel RECORD row order must match archive member order",
    "wheel RECORD entry must be last",
    "wheel CycloneDX SBOM must use CycloneDX 1.5",
    "wheel CycloneDX SBOM version must be 1",
    "wheel CycloneDX SBOM serialNumber must be a UUID URN",
    "wheel CycloneDX SBOM root component purl mismatch",
    "wheel CycloneDX SBOM root component must reference the VCS URL",
}

REQUIRED_SCHEMA_CHECK_SNIPPETS = {
    "check_schema_version_fields(schema, path)",
    "check_const_schemas_are_typed(schema, path)",
    "check_array_schemas_are_typed_and_sized(schema, path)",
    "EXPECTED_SCHEMA_TOP_LEVEL_KEYS = [",
    "EXPECTED_SCHEMA_DEF_ORDER = {",
    "top-level key order changed",
    "$defs order changed",
    "EXPECTED_REPLAY_KEYS = [",
    "top-level golden field order changed",
    "replay field order changed",
    "EXPECTED_REQUIRED_PROPERTIES = {",
    "required field contracts changed",
    "EXPECTED_OPTIONAL_PROPERTIES = {",
    "optional field contracts changed",
    "check_tagged_union_discriminators(document_kind, schema, path)",
    "EXPECTED_TAGGED_UNIONS = {",
    "tagged union variant order changed",
    "tagged union locations changed",
    "required fields must lead properties in order",
    "const schema must declare its JSON type",
    "array schema must declare object items",
    "non-empty array schema must declare positive minItems",
    "union variant must list its type discriminator first",
    "union variant must declare its type discriminator first",
    "schema_version must match its Rust integer width",
}

REQUIRED_REPLAY_FIXTURE_CHECK_SNIPPETS = {
    "if generated_text != expected_text:",
    "difflib.unified_diff",
    "require_exact_keys(path, document, REPLAY_DOCUMENT_KEYS, \"replay document\")",
    "key order changed",
    "case_names != expected_case_names",
    "replay case order changed",
    "plan/result request fingerprints do not match",
    "result.replay.library_version must match Cargo workspace version",
    "result.replay.platform must match artifact platform",
    "price-only case must not carry VegaKT",
    "full_bucket_matrix_row_major",
    "must contain 36 row-major entries",
}

REQUIRED_BENCHMARK_CHECK_SNIPPETS = {
    "EXPECTED_ARTIFACTS = {",
    "OPTIONAL_ARTIFACTS = {\"local-volatility-replay.json\"}",
    "REPORT_KEYS = (",
    "BASE_CONFIGURATION_KEYS = (",
    "BASE_CONFIGURATION = {",
    "LOCAL_VOL_CONFIGURATION = {",
    "PYTHON_CONFIGURATION = {",
    "PYTHON_GETTER_SAMPLES = 100_000",
    "LOCAL_VOL_CAPABILITY_KEYS = (",
    "BASE_CAPABILITIES = {",
    "LOCAL_VOL_CAPABILITIES = {",
    "PYTHON_CAPABILITIES = {",
    "BASE_NOTES = (",
    "LOCAL_VOL_NOTES = (",
    "PYTHON_NOTES = (",
    "METADATA_KEYS = (",
    "ENABLED_FEATURE_KEYS = (",
    "COMMAND_PEAK_KEYS = (",
    "COMMAND_PEAK_REPORTS = {",
    "UNAVAILABLE_METRICS = (",
    "unexpected benchmark artifacts",
    "missing measurements",
    "unexpected measurements",
    "measurement order changed",
    "samples mismatch",
    "evaluated_paths_per_sample mismatch",
    "median_paths_per_second mismatch",
    "evaluated_paths must match sampling_units and antithetic",
    "configuration mismatch",
    "capabilities mismatch",
    "notes mismatch",
    "cargo_lock_sha256 must match Cargo.lock",
    "must match {report_name}",
    "peak_memory_bytes must match the maximum command peak",
    "unavailable_metrics mismatch",
    "check_replay_report(root / \"replay.json\"",
    "replay case order changed",
    "key order changed",
    "validate_local_vol_case",
}

REQUIRED_DEPENDENCY_DIRECTION_SNIPPETS = {
    "\"pricing-core\": 0",
    "\"pricing-python\": 7",
    "missing workspace crates",
    "unclassified workspace crates",
    "depends upward on",
    "workspace_members references missing packages",
    "multiple workspace packages share crate names",
}

REQUIRED_LOCAL_VOL_REFERENCE_CHECK_SNIPPETS = {
    "getcontext().prec = 70",
    "STANDARD_SSVI_CASE_IDS = {",
    "\"power_regular\"",
    "\"heston_like_regular\"",
    "\"heston_like_small_theta\"",
    "essvi_interpolation.id must be midpoint_regular",
    "JSON artifact must end with LF",
    "non-standard JSON constant",
    "duplicate object key",
}

REQUIRED_MARKDOWN_LINK_CHECK_SNIPPETS = {
    "MARKDOWN_ROOTS = [ROOT / \"README.md\", ROOT / \"CONTRIBUTING.md\", ROOT / \"docs\", ROOT / \"fixtures\"]",
    "LINK_PATTERN = re.compile",
    "escapes repository root",
    "target does not exist",
    "anchor does not exist",
    "markdown_anchors(resolved)",
    "github_heading_slug",
    "re.match(r\"^[a-zA-Z][a-zA-Z0-9+.-]*:\", target)",
}

REQUIRED_SOURCE_ARCHIVE_CHECK_SNIPPETS = {
    "PurePosixPath(name).is_absolute()",
    "\"..\" in PurePosixPath(name).parts",
    "duplicate archive member path",
    "unsupported archive member type",
    "check_file_member_mode(archive, member)",
    "check_directory_member_mode(archive, member)",
    "source file must not be executable",
    "source file must not be world-writable",
    "directory must not be world-writable",
    "directory must be searchable",
    "archive contains generated/private files",
    "archive contains unexpected source files",
    "REQUIRED_FILES.union(repository_source_files())",
    "\"target\"",
    "\"dist\"",
    "\".wheel-smoke-venv\"",
    "\"benchmark-results\"",
    "\"uv.lock\"",
    "\".pyc\"",
    "TEXT_SOURCE_SUFFIXES = {",
    "TEXT_SOURCE_NAMES = {",
    "check_text_member(package, archive, name)",
    "must not start with a UTF-8 BOM",
    "must use LF line endings",
    "must end with LF",
    "must be valid UTF-8",
    "(ROOT / root).rglob(pattern)",
}

REQUIRED_RELEASE_READINESS_SNIPPETS = {
    "Status: code and private artifact gates ready; external publication decisions open",
    "platform-specific CPython wheels for Apple Silicon macOS and Windows x86-64",
    "the retained Rust source archive for the exact commit",
    "benchmark and replay artifacts for the supported platforms",
    "Local Volatility reference fixture validation",
    "JSON Schema validation",
    "local Markdown link validation",
    "retained source archive validation",
    "Rust formatting, Clippy, unit/integration tests, and statistical acceptance",
    "dependency-direction validation",
    "Rust API documentation generation",
    "Python extension build, wheel smoke test, Python smoke suite, benchmark run",
    "benchmark artifact validation",
    "private artifact repository location and access policy",
    "internal licensing terms for private consumers",
    "whether a public open-source license will ever be selected",
}

REQUIRED_ARCHITECTURE_SNIPPETS = {
    "`syntax_and_limits`, `declared_schema`, `migration`, `current_schema`, or `domain`",
}

REQUIRED_WIRE_SCHEMA_COMPATIBILITY_SNIPPETS = {
    'covariance_layout.type = "full_bucket_matrix_row_major"',
    "`full_bucket_covariance`",
    'covariance_layout.type =\n"price_and_bucket_variance_only"',
    "A present\n`null`, a missing full matrix under the full-matrix layout, or an unexpected\nmatrix under the compact layout is invalid schema/domain input.",
}

REQUIRED_LOCAL_VOL_DIAGNOSTICS_SNIPPETS = {
    "the full-matrix layout requires `full_bucket_covariance`",
    "the compact\nprice-and-bucket variance layout omits that field",
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
TEXT_SOURCE_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_SOURCE_NAMES = {
    ".gitattributes",
    ".gitignore",
}

JOE_KUO_DIRECTION_DATA = "crates/pricing-mc/data/joe-kuo-6.21201-u32be.bin"
JOE_KUO_DIRECTION_DATA_README = "crates/pricing-mc/data/README.md"
JOE_KUO_DIRECTION_DATA_MAGIC = b"JK621201"
JOE_KUO_DIRECTION_DATA_VERSION = 1
JOE_KUO_DIRECTION_DATA_DIMENSIONS = 21_201
JOE_KUO_DIRECTION_DATA_BITS = 32
JOE_KUO_DIRECTION_DATA_SHA256 = (
    "189f65c4e4fcf7455efb7618f3dafbbbaf70303fc35ecd380fa28a67ab896900"
)
REQUIRED_DIRECTION_DATA_README_SNIPPETS = {
    "`joe-kuo-6.21201-u32be.bin` embeds the complete 21,201-dimensional",
    "ASCII magic `JK621201`",
    "big-endian format version `1`",
    "big-endian dimension count `21201`",
    "big-endian bit count `32`",
    "row-major, big-endian `u32` direction words",
    "SciPy's `_sobol_direction_numbers.npz`",
    "new-joe-kuo-6.21201",
    "updated 5 January 2010",
    JOE_KUO_DIRECTION_DATA_SHA256,
    "BSD 3-Clause License",
}
REQUIRED_THIRD_PARTY_NOTICE_SNIPPETS = {
    "SciPy Sobol direction-number data",
    JOE_KUO_DIRECTION_DATA,
    "generated from SciPy's `_sobol_direction_numbers.npz`",
    "Frances Y. Kuo's UNSW Sobol sequence resource",
    "BSD 3-Clause License",
    "Redistribution and use in source and binary forms",
}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_source_archive.py <archive.tar.gz>")

    archive = sys.argv[1]
    with tarfile.open(archive, "r:gz") as package:
        names = set()
        seen_members = set()
        for member in package.getmembers():
            name = member.name
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise SystemExit(f"{archive}: unsafe archive member path: {name}")
            if name in seen_members:
                raise SystemExit(f"{archive}: duplicate archive member path: {name}")
            seen_members.add(name)
            if member.isfile():
                check_file_member_mode(archive, member)
                names.add(name)
            elif not member.isdir():
                raise SystemExit(
                    f"{archive}: unsupported archive member type for {name}: {member.type!r}"
                )
            else:
                check_directory_member_mode(archive, member)

        forbidden = sorted(name for name in names if has_forbidden_part(name))
        if forbidden:
            raise SystemExit(f"{archive}: archive contains generated/private files: {forbidden[:10]}")

        expected = REQUIRED_FILES.union(repository_source_files())
        missing = sorted(expected.difference(names))
        if missing:
            raise SystemExit(f"{archive}: missing required source files: {missing}")
        unexpected = sorted(names.difference(expected))
        if unexpected:
            raise SystemExit(
                f"{archive}: archive contains unexpected source files: {unexpected[:10]}"
            )
        for name in sorted(names):
            if is_text_source(name):
                check_text_member(package, archive, name)

        check_cargo_manifests(package, archive)
        check_pyproject(package, archive)
        check_ci_workflow(package, archive)
        check_wheel_smoke_gate(package, archive)
        check_schema_validation_gate(package, archive)
        check_replay_fixture_gate(package, archive)
        check_benchmark_validation_gate(package, archive)
        check_dependency_direction_gate(package, archive)
        check_local_vol_reference_gate(package, archive)
        check_markdown_link_gate(package, archive)
        check_source_archive_gate(package, archive)
        check_readme_release_gates(package, archive)
        check_contributing_release_gates(package, archive)
        check_architecture_contract(package, archive)
        check_release_readiness(package, archive)
        check_wire_schema_compatibility(package, archive)
        check_local_vol_diagnostics(package, archive)
        check_direction_data(package, archive)
        check_direction_data_readme(package, archive)
        check_third_party_notices(package, archive)

    return 0


def has_forbidden_part(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        any(part in FORBIDDEN_PARTS for part in path.parts)
        or path.name in FORBIDDEN_NAMES
        or any(path.name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)
    )


def check_file_member_mode(archive: str, member: tarfile.TarInfo) -> None:
    if member.mode & 0o111:
        raise SystemExit(f"{archive}: source file must not be executable: {member.name}")
    if member.mode & 0o002:
        raise SystemExit(f"{archive}: source file must not be world-writable: {member.name}")


def check_directory_member_mode(archive: str, member: tarfile.TarInfo) -> None:
    if member.mode & 0o002:
        raise SystemExit(f"{archive}: directory must not be world-writable: {member.name}")
    if member.mode & 0o111 != 0o111:
        raise SystemExit(f"{archive}: directory must be searchable: {member.name}")


def is_text_source(name: str) -> bool:
    path = PurePosixPath(name)
    return path.suffix in TEXT_SOURCE_SUFFIXES or path.name in TEXT_SOURCE_NAMES


def check_text_member(package: tarfile.TarFile, archive: str, name: str) -> None:
    raw = read_bytes(package, name)
    if raw.startswith(b"\xef\xbb\xbf"):
        raise SystemExit(f"{archive}: {name} must not start with a UTF-8 BOM")
    if b"\r" in raw:
        raise SystemExit(f"{archive}: {name} must use LF line endings")
    if not raw.endswith(b"\n"):
        raise SystemExit(f"{archive}: {name} must end with LF")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(f"{archive}: {name} must be valid UTF-8: {exc}") from exc


def repository_source_files() -> set[str]:
    patterns = [
        ("crates", "*.rs"),
        ("benchmarks", "*.py"),
        ("docs", "*.md"),
        ("examples/python", "*.py"),
        ("fixtures", "*.csv"),
        ("fixtures", "*.json"),
        ("fixtures", "*.md"),
        ("schemas", "*.json"),
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


def check_pyproject(package: tarfile.TarFile, archive: str) -> None:
    pyproject = read_toml(package, "pyproject.toml")
    python_manifest = read_toml(package, "crates/pricing-python/Cargo.toml")
    build_system = pyproject.get("build-system", {})
    if build_system.get("requires") != ["maturin>=1.9,<2.0"]:
        raise SystemExit(f"{archive}: pyproject build-system.requires mismatch")
    if build_system.get("build-backend") != "maturin":
        raise SystemExit(f"{archive}: pyproject build-system.build-backend mismatch")

    project = pyproject.get("project", {})
    if project.get("name") != "rust-pricing":
        raise SystemExit(f"{archive}: pyproject project.name mismatch")
    if project.get("description") != "Deterministic Rust and Python derivatives-pricing library.":
        raise SystemExit(f"{archive}: pyproject project.description mismatch")
    if project.get("readme") != "README.md":
        raise SystemExit(f"{archive}: pyproject project.readme mismatch")
    if project.get("requires-python") != ">=3.12":
        raise SystemExit(f"{archive}: pyproject project.requires-python mismatch")
    if project.get("authors") != [{"name": "Masafumi Fujita"}]:
        raise SystemExit(f"{archive}: pyproject project.authors mismatch")
    for key in ["license", "license-files"]:
        if key in project:
            raise SystemExit(
                f"{archive}: pyproject project.{key} must stay absent until licensing is decided"
            )
    if project.get("dynamic") != ["version"]:
        raise SystemExit(f"{archive}: pyproject must derive version dynamically")

    tool = pyproject.get("tool")
    if not isinstance(tool, dict):
        raise SystemExit(f"{archive}: pyproject must contain a [tool] table")
    maturin = tool.get("maturin")
    if not isinstance(maturin, dict):
        raise SystemExit(f"{archive}: pyproject must contain a [tool.maturin] table")
    expected = {
        "manifest-path": "crates/pricing-python/Cargo.toml",
        "module-name": "rust_pricing",
        "features": ["extension-module"],
    }
    for key, expected_value in expected.items():
        if maturin.get(key) != expected_value:
            raise SystemExit(f"{archive}: pyproject tool.maturin.{key} mismatch")

    python_package = python_manifest.get("package", {})
    if python_package.get("name") != "pricing-python":
        raise SystemExit(f"{archive}: pricing-python package.name mismatch")
    if python_package.get("version") != {"workspace": True}:
        raise SystemExit(f"{archive}: pricing-python package.version must use workspace")
    python_lib = python_manifest.get("lib", {})
    if python_lib.get("name") != "rust_pricing":
        raise SystemExit(f"{archive}: pricing-python lib.name mismatch")
    if python_lib.get("crate-type") != ["cdylib", "rlib"]:
        raise SystemExit(f"{archive}: pricing-python lib.crate-type mismatch")
    features = python_manifest.get("features", {})
    if features.get("extension-module") != ["pyo3/extension-module"]:
        raise SystemExit(f"{archive}: pricing-python extension-module feature mismatch")


def check_ci_workflow(package: tarfile.TarFile, archive: str) -> None:
    workflow = read_text(package, ".github/workflows/ci.yml")
    missing = sorted(snippet for snippet in REQUIRED_CI_SNIPPETS if snippet not in workflow)
    if missing:
        raise SystemExit(f"{archive}: CI workflow is missing required gates: {missing}")
    for snippet, expected_count in sorted(REQUIRED_CI_SNIPPET_COUNTS.items()):
        actual_count = workflow.count(snippet)
        if actual_count != expected_count:
            raise SystemExit(
                f"{archive}: CI workflow must contain {snippet!r} "
                f"{expected_count} times, found {actual_count}"
            )


def check_wheel_smoke_gate(package: tarfile.TarFile, archive: str) -> None:
    smoke = read_text(package, "scripts/smoke_test_wheel.py")
    missing = sorted(
        snippet for snippet in REQUIRED_WHEEL_SMOKE_SNIPPETS if snippet not in smoke
    )
    if missing:
        raise SystemExit(
            f"{archive}: wheel smoke test is missing required gates: {missing}"
        )


def check_schema_validation_gate(package: tarfile.TarFile, archive: str) -> None:
    schema_check = read_text(package, "scripts/check_schemas.py")
    missing = sorted(
        snippet
        for snippet in REQUIRED_SCHEMA_CHECK_SNIPPETS
        if snippet not in schema_check
    )
    if missing:
        raise SystemExit(
            f"{archive}: schema validation is missing required gates: {missing}"
        )


def check_replay_fixture_gate(package: tarfile.TarFile, archive: str) -> None:
    replay_check = read_text(package, "scripts/check_replay_fixture.py")
    missing = sorted(
        snippet
        for snippet in REQUIRED_REPLAY_FIXTURE_CHECK_SNIPPETS
        if snippet not in replay_check
    )
    if missing:
        raise SystemExit(
            f"{archive}: replay fixture validation is missing required gates: {missing}"
        )


def check_benchmark_validation_gate(package: tarfile.TarFile, archive: str) -> None:
    benchmark_check = read_text(package, "scripts/check_benchmark_reports.py")
    missing = sorted(
        snippet
        for snippet in REQUIRED_BENCHMARK_CHECK_SNIPPETS
        if snippet not in benchmark_check
    )
    if missing:
        raise SystemExit(
            f"{archive}: benchmark validation is missing required gates: {missing}"
        )


def check_dependency_direction_gate(package: tarfile.TarFile, archive: str) -> None:
    dependency_check = read_text(package, "scripts/check_dependency_direction.py")
    missing = sorted(
        snippet
        for snippet in REQUIRED_DEPENDENCY_DIRECTION_SNIPPETS
        if snippet not in dependency_check
    )
    if missing:
        raise SystemExit(
            f"{archive}: dependency-direction validation is missing required gates: {missing}"
        )


def check_local_vol_reference_gate(package: tarfile.TarFile, archive: str) -> None:
    local_vol_check = read_text(package, "scripts/check_local_vol_reference_fixture.py")
    missing = sorted(
        snippet
        for snippet in REQUIRED_LOCAL_VOL_REFERENCE_CHECK_SNIPPETS
        if snippet not in local_vol_check
    )
    if missing:
        raise SystemExit(
            f"{archive}: Local Volatility reference validation is missing required gates: {missing}"
        )


def check_markdown_link_gate(package: tarfile.TarFile, archive: str) -> None:
    markdown_check = read_text(package, "scripts/check_markdown_links.py")
    missing = sorted(
        snippet
        for snippet in REQUIRED_MARKDOWN_LINK_CHECK_SNIPPETS
        if snippet not in markdown_check
    )
    if missing:
        raise SystemExit(
            f"{archive}: Markdown link validation is missing required gates: {missing}"
        )


def check_source_archive_gate(package: tarfile.TarFile, archive: str) -> None:
    source_archive_check = read_text(package, "scripts/check_source_archive.py")
    missing = sorted(
        snippet
        for snippet in REQUIRED_SOURCE_ARCHIVE_CHECK_SNIPPETS
        if snippet not in source_archive_check
    )
    if missing:
        raise SystemExit(
            f"{archive}: source archive validation is missing required gates: {missing}"
        )


def check_readme_release_gates(package: tarfile.TarFile, archive: str) -> None:
    readme = read_text(package, "README.md")
    missing = sorted(snippet for snippet in REQUIRED_README_SNIPPETS if snippet not in readme)
    if missing:
        raise SystemExit(f"{archive}: README is missing documented release gates: {missing}")


def check_contributing_release_gates(package: tarfile.TarFile, archive: str) -> None:
    contributing = read_text(package, "CONTRIBUTING.md")
    missing = sorted(
        snippet for snippet in REQUIRED_CONTRIBUTING_SNIPPETS if snippet not in contributing
    )
    if missing:
        raise SystemExit(
            f"{archive}: CONTRIBUTING is missing documented release gates: {missing}"
        )


def check_architecture_contract(package: tarfile.TarFile, archive: str) -> None:
    architecture = read_text(package, "docs/architecture-v0.1.md")
    missing = sorted(
        snippet for snippet in REQUIRED_ARCHITECTURE_SNIPPETS if snippet not in architecture
    )
    if missing:
        raise SystemExit(f"{archive}: architecture is missing required contracts: {missing}")


def check_release_readiness(package: tarfile.TarFile, archive: str) -> None:
    readiness = read_text(package, "docs/release-readiness-v0.1.md")
    missing = sorted(
        snippet for snippet in REQUIRED_RELEASE_READINESS_SNIPPETS if snippet not in readiness
    )
    if missing:
        raise SystemExit(f"{archive}: release readiness is missing required statements: {missing}")


def check_wire_schema_compatibility(package: tarfile.TarFile, archive: str) -> None:
    compatibility = read_text(package, "docs/wire-schema-compatibility.md")
    missing = sorted(
        snippet
        for snippet in REQUIRED_WIRE_SCHEMA_COMPATIBILITY_SNIPPETS
        if snippet not in compatibility
    )
    if missing:
        raise SystemExit(
            f"{archive}: wire schema compatibility is missing required statements: {missing}"
        )


def check_local_vol_diagnostics(package: tarfile.TarFile, archive: str) -> None:
    diagnostics = read_text(package, "docs/local-vol-vegakt-diagnostics-v0.1.md")
    missing = sorted(
        snippet
        for snippet in REQUIRED_LOCAL_VOL_DIAGNOSTICS_SNIPPETS
        if snippet not in diagnostics
    )
    if missing:
        raise SystemExit(
            f"{archive}: Local Volatility diagnostics are missing required statements: {missing}"
        )


def check_direction_data(package: tarfile.TarFile, archive: str) -> None:
    direction_data = read_bytes(package, JOE_KUO_DIRECTION_DATA)
    expected_size = 20 + JOE_KUO_DIRECTION_DATA_DIMENSIONS * JOE_KUO_DIRECTION_DATA_BITS * 4
    if len(direction_data) != expected_size:
        raise SystemExit(
            f"{archive}: Joe-Kuo direction data size mismatch: "
            f"expected {expected_size} bytes, found {len(direction_data)}"
        )
    if direction_data[:8] != JOE_KUO_DIRECTION_DATA_MAGIC:
        raise SystemExit(
            f"{archive}: Joe-Kuo direction data magic mismatch: {direction_data[:8]!r}"
        )
    version = int.from_bytes(direction_data[8:12], "big")
    if version != JOE_KUO_DIRECTION_DATA_VERSION:
        raise SystemExit(
            f"{archive}: Joe-Kuo direction data version mismatch: {version}"
        )
    dimensions = int.from_bytes(direction_data[12:16], "big")
    if dimensions != JOE_KUO_DIRECTION_DATA_DIMENSIONS:
        raise SystemExit(
            f"{archive}: Joe-Kuo direction data dimension count mismatch: {dimensions}"
        )
    bits = int.from_bytes(direction_data[16:20], "big")
    if bits != JOE_KUO_DIRECTION_DATA_BITS:
        raise SystemExit(f"{archive}: Joe-Kuo direction data bit count mismatch: {bits}")
    actual_digest = hashlib.sha256(direction_data).hexdigest()
    if actual_digest != JOE_KUO_DIRECTION_DATA_SHA256:
        raise SystemExit(
            f"{archive}: Joe-Kuo direction data SHA-256 mismatch: {actual_digest}"
        )


def check_direction_data_readme(package: tarfile.TarFile, archive: str) -> None:
    readme = read_text(package, JOE_KUO_DIRECTION_DATA_README)
    missing = sorted(
        snippet for snippet in REQUIRED_DIRECTION_DATA_README_SNIPPETS if snippet not in readme
    )
    if missing:
        raise SystemExit(f"{archive}: direction data README is missing: {missing}")


def check_third_party_notices(package: tarfile.TarFile, archive: str) -> None:
    notices = read_text(package, "THIRD_PARTY_NOTICES.md")
    missing = sorted(
        snippet for snippet in REQUIRED_THIRD_PARTY_NOTICE_SNIPPETS if snippet not in notices
    )
    if missing:
        raise SystemExit(f"{archive}: third-party notices are missing: {missing}")


def read_bytes(package: tarfile.TarFile, name: str) -> bytes:
    member = package.extractfile(name)
    if member is None:
        raise SystemExit(f"missing binary member: {name}")
    return member.read()


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
