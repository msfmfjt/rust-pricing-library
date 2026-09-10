"""Validate benchmark reports before retaining CI artifacts."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tomllib
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_replay_fixture import EXPECTED_CASE_NAMES
from check_replay_fixture import FINGERPRINT
from check_replay_fixture import REPLAY_CASE_KEYS
from check_replay_fixture import REPLAY_DOCUMENT_KEYS
from check_replay_fixture import REPLAY_EXECUTION_KEYS
from check_replay_fixture import REPLAY_METADATA_KEYS
from check_replay_fixture import REPLAY_MONTE_CARLO_KEYS
from check_replay_fixture import REPLAY_PLAN_KEYS
from check_replay_fixture import REPLAY_REQUEST_KEYS
from check_replay_fixture import REPLAY_RESULT_KEYS
from check_replay_fixture import SUPPORTED_PLATFORMS as SUPPORTED_REPLAY_PLATFORMS
from check_replay_fixture import validate_local_vol_case
from check_replay_fixture import validate_risk_methods
from check_replay_fixture import validate_risk_validation


ROOT = Path(__file__).resolve().parents[1]
EUROPEAN_MEASUREMENTS = (
    "compile_price_only",
    "evaluate_price_only",
    "compile_full_risk",
    "evaluate_aad_with_crn_bump_validation",
    "compile_crn_bump_validation",
    "evaluate_crn_bump_validation_price_only",
)

LOCAL_VOL_MEASUREMENTS = (
    "compile_price_only",
    "evaluate_price_only",
    "compile_aad_local_vega",
    "evaluate_aad_local_vega",
    "compile_vega_kt_decomposition",
    "evaluate_vega_kt_decomposition",
    "compile_selected_crn_bump_validation",
    "evaluate_selected_crn_bump_validation_price_only",
)

PYTHON_MEASUREMENTS = (
    "compile_full_risk_from_python",
    "evaluate_full_risk_from_python",
    "compile_crn_bump_validation_from_python",
    "evaluate_crn_bump_validation_from_python",
    "result_value_getter",
)
REPORT_KEYS = (
    "benchmark_kind",
    "capabilities",
    "configuration",
    "library_version",
    "measurements",
    "notes",
    "process_peak_memory_bytes",
    "schema_version",
)
BASE_CONFIGURATION_KEYS = (
    "antithetic",
    "compile_samples",
    "engine",
    "evaluated_paths",
    "evaluation_samples",
    "reduction_block_size",
    "sampling_units",
    "worker_threads",
)
BASE_CONFIGURATION = {
    "antithetic": True,
    "compile_samples": 20,
    "engine": "pseudo_monte_carlo",
    "evaluated_paths": 32_768,
    "evaluation_samples": 5,
    "reduction_block_size": 256,
    "sampling_units": 16_384,
    "worker_threads": 2,
}
LOCAL_VOL_CONFIGURATION_KEYS = (
    "aad_tile_capacity",
    "antithetic",
    "checkpoint_interval",
    "compile_samples",
    "engine",
    "evaluated_paths",
    "evaluation_samples",
    "local_variance_grid_shape",
    "reduction_block_size",
    "reporting_iv_basis_shape",
    "sampling_units",
    "vega_kt_covariance_layout",
    "worker_threads",
)
LOCAL_VOL_CONFIGURATION = {
    "aad_tile_capacity": 256,
    "antithetic": True,
    "checkpoint_interval": 16,
    "compile_samples": 20,
    "engine": "pseudo_monte_carlo",
    "evaluated_paths": 16_384,
    "evaluation_samples": 5,
    "local_variance_grid_shape": [3, 5],
    "reduction_block_size": 256,
    "reporting_iv_basis_shape": [2, 3],
    "sampling_units": 8_192,
    "vega_kt_covariance_layout": "full_bucket_matrix_row_major",
    "worker_threads": 2,
}
PYTHON_CONFIGURATION_KEYS = (
    "antithetic",
    "engine",
    "evaluated_paths",
    "reduction_block_size",
    "sampling_units",
    "worker_threads",
)
PYTHON_CONFIGURATION = {
    "antithetic": True,
    "engine": "pseudo_monte_carlo",
    "evaluated_paths": 32_768,
    "reduction_block_size": 256,
    "sampling_units": 16_384,
    "worker_threads": 2,
}
BASE_CAPABILITY_KEYS = (
    "aad_and_bump_timing_separable",
    "allocation_count_available",
    "peak_memory_available_in_process",
    "standalone_bump_timing_available",
)
BASE_CAPABILITIES = {
    "aad_and_bump_timing_separable": False,
    "allocation_count_available": False,
    "peak_memory_available_in_process": True,
    "standalone_bump_timing_available": True,
}
LOCAL_VOL_CAPABILITY_KEYS = (
    "aad_local_vega_timing_available",
    "allocation_count_available",
    "peak_memory_available_in_process",
    "standalone_spot_and_local_variance_bump_timing_available",
    "vega_kt_decomposition_timing_available",
)
LOCAL_VOL_CAPABILITIES = {
    "aad_local_vega_timing_available": True,
    "allocation_count_available": False,
    "peak_memory_available_in_process": True,
    "standalone_spot_and_local_variance_bump_timing_available": True,
    "vega_kt_decomposition_timing_available": True,
}
PYTHON_CAPABILITY_KEYS = ("peak_memory_available_in_process",)
PYTHON_CAPABILITIES = {"peak_memory_available_in_process": True}
BASE_NOTES = (
    "The full-risk kernel computes AAD Delta/Vega, central-bumped AAD Delta Gamma, and CRN bump validations in one execution.",
    "The standalone bump case evaluates base, Spot-down/up, and volatility-down/up Price-only plans with common random numbers.",
    "Results are an optimization baseline and not a latency SLA.",
)
LOCAL_VOL_NOTES = (
    "The AAD Local Vega workload computes scalar Local Volatility Vega without VegaKT reporting projection.",
    "The VegaKT workload computes Delta, bumped Gamma, scalar Vega, VegaKT buckets, and full bucket covariance.",
    "The standalone bump case evaluates base, Spot-down/up, and uniform local-variance-down/up Price-only plans with common random numbers.",
    "Results are an optimization baseline and not a latency SLA.",
)
PYTHON_NOTES = (
    "Compile/evaluate release the GIL; timings include the Python-to-Rust call boundary.",
    "The full-risk kernel includes AAD and its CRN bump validations.",
    "The standalone bump case evaluates base, Spot-down/up, and volatility-down/up Price-only plans with common random numbers.",
    "Results are an optimization baseline and not a latency SLA.",
)

EXPECTED_ARTIFACTS = {
    "rust.json",
    "local-volatility-rust.json",
    "python.json",
    "replay.json",
    "metadata.json",
}
OPTIONAL_ARTIFACTS = {"local-volatility-replay.json"}
METADATA_KEYS = (
    "allocation_count",
    "cargo",
    "cargo_lock_sha256",
    "command_peak_memory_bytes",
    "enabled_features",
    "git_sha",
    "machine",
    "peak_memory_bytes",
    "platform",
    "processor",
    "python",
    "python_abi",
    "runner_arch",
    "runner_os",
    "rustc",
    "schema_version",
    "target_triple",
    "unavailable_metrics",
)
ENABLED_FEATURE_KEYS = ("python_wheel", "rust_benchmarks")
COMMAND_PEAK_KEYS = (
    "python_european_black_scholes",
    "replay_european_black_scholes",
    "rust_european_black_scholes",
    "rust_local_volatility_vegakt",
)
UNAVAILABLE_METRICS = (
    "Allocation counting requires an instrumented allocator.",
)
OPTIONAL_COMMAND_PEAKS = {
    "local-volatility-replay.json": "replay_local_volatility",
}
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_benchmark_reports.py <benchmark-results-dir>")

    root = Path(sys.argv[1])
    library_version = workspace_package_version()
    artifacts = check_artifact_set(root)
    check_report(
        root / "rust.json",
        "rust_european_black_scholes",
        EUROPEAN_MEASUREMENTS,
        library_version,
    )
    check_report(
        root / "local-volatility-rust.json",
        "rust_local_volatility_vegakt",
        LOCAL_VOL_MEASUREMENTS,
        library_version,
        local_volatility=True,
    )
    check_python_report(root / "python.json", library_version)
    check_replay_report(root / "replay.json", "european_black_scholes_replay", library_version)
    local_volatility_replay = root / "local-volatility-replay.json"
    if local_volatility_replay.is_file():
        check_replay_report(
            local_volatility_replay,
            "local_volatility_replay",
            library_version,
        )
    check_metadata(root / "metadata.json", artifacts)
    print(f"benchmark reports are valid in {root}")


def check_artifact_set(root: Path) -> set[str]:
    actual = {path.name for path in root.glob("*.json")}
    missing = sorted(EXPECTED_ARTIFACTS.difference(actual))
    if missing:
        raise SystemExit(f"{root}: missing benchmark artifacts: {missing}")
    expected = EXPECTED_ARTIFACTS.union(OPTIONAL_ARTIFACTS)
    unexpected = sorted(actual.difference(expected))
    if unexpected:
        raise SystemExit(f"{root}: unexpected benchmark artifacts: {unexpected}")
    return actual


def workspace_package_version() -> str:
    manifest = tomllib.loads((ROOT / "Cargo.toml").read_text("utf-8"))
    workspace = manifest.get("workspace")
    if not isinstance(workspace, dict):
        raise SystemExit(f"{ROOT / 'Cargo.toml'}: missing workspace table")
    package = workspace.get("package")
    if not isinstance(package, dict):
        raise SystemExit(f"{ROOT / 'Cargo.toml'}: missing workspace.package table")
    version = package.get("version")
    if not isinstance(version, str) or not version:
        raise SystemExit(f"{ROOT / 'Cargo.toml'}: workspace package version must be a string")
    return version


def check_report(
    path: Path,
    benchmark_kind: str,
    required_measurements: Sequence[str],
    library_version: str,
    *,
    local_volatility: bool = False,
) -> None:
    document = load_object(path)
    require_exact_keys(
        document,
        REPORT_KEYS,
        path,
        "report",
    )
    require(document.get("schema_version") == 1, path, "schema_version must be 1")
    require(document.get("benchmark_kind") == benchmark_kind, path, "unexpected benchmark_kind")
    require(
        document.get("library_version") == library_version,
        path,
        "library_version must match Cargo workspace version",
    )
    configuration = require_object(document.get("configuration"), path, "configuration")
    expected_configuration_keys = (
        LOCAL_VOL_CONFIGURATION_KEYS if local_volatility else BASE_CONFIGURATION_KEYS
    )
    require_exact_keys(configuration, expected_configuration_keys, path, "configuration")
    require(configuration.get("engine") == "pseudo_monte_carlo", path, "unexpected engine")
    require(configuration.get("antithetic") is True, path, "antithetic must be true")
    require_positive_int(configuration.get("sampling_units"), path, "sampling_units")
    require_positive_int(configuration.get("evaluated_paths"), path, "evaluated_paths")
    check_antithetic_path_count(configuration, path)
    require_positive_int(configuration.get("worker_threads"), path, "worker_threads")
    require_positive_int(
        configuration.get("reduction_block_size"), path, "reduction_block_size"
    )
    require_positive_int(configuration.get("compile_samples"), path, "compile_samples")
    require_positive_int(configuration.get("evaluation_samples"), path, "evaluation_samples")
    if local_volatility:
        require(
            configuration.get("local_variance_grid_shape") == [3, 5],
            path,
            "unexpected local_variance_grid_shape",
        )
        require(
            configuration.get("reporting_iv_basis_shape") == [2, 3],
            path,
            "unexpected reporting_iv_basis_shape",
        )
        require_positive_int(
            configuration.get("checkpoint_interval"), path, "checkpoint_interval"
        )
        require_positive_int(configuration.get("aad_tile_capacity"), path, "aad_tile_capacity")
        require(
            configuration.get("vega_kt_covariance_layout") == "full_bucket_matrix_row_major",
            path,
            "unexpected vega_kt_covariance_layout",
        )
    expected_configuration = LOCAL_VOL_CONFIGURATION if local_volatility else BASE_CONFIGURATION
    require(configuration == expected_configuration, path, "configuration mismatch")

    measurements = require_object(document.get("measurements"), path, "measurements")
    required_measurement_set = set(required_measurements)
    missing = sorted(required_measurement_set.difference(measurements))
    require(not missing, path, f"missing measurements: {missing}")
    unexpected = sorted(set(measurements).difference(required_measurement_set))
    require(not unexpected, path, f"unexpected measurements: {unexpected}")
    require(
        list(measurements) == list(required_measurements),
        path,
        "measurement order changed",
    )
    for name in required_measurements:
        check_measurement(
            require_object(measurements.get(name), path, name),
            path,
            name,
            expected_paths_for_measurement(name, configuration["evaluated_paths"]),
        )
    require_positive_int(
        document.get("process_peak_memory_bytes"), path, "process_peak_memory_bytes"
    )

    capabilities = require_object(document.get("capabilities"), path, "capabilities")
    expected_capability_keys = (
        LOCAL_VOL_CAPABILITY_KEYS if local_volatility else BASE_CAPABILITY_KEYS
    )
    require_exact_keys(capabilities, expected_capability_keys, path, "capabilities")
    expected_capabilities = LOCAL_VOL_CAPABILITIES if local_volatility else BASE_CAPABILITIES
    require(capabilities == expected_capabilities, path, "capabilities mismatch")
    notes = document.get("notes")
    require_string_array(notes, path, "notes")
    expected_notes = LOCAL_VOL_NOTES if local_volatility else BASE_NOTES
    require(notes == list(expected_notes), path, "notes mismatch")


def check_replay_report(path: Path, fixture_kind: str, library_version: str) -> None:
    document = load_object(path)
    require_exact_keys(
        document,
        REPLAY_DOCUMENT_KEYS,
        path,
        "replay report",
    )
    require(document.get("schema_version") == 1, path, "schema_version must be 1")
    require(document.get("fixture_kind") == fixture_kind, path, "unexpected fixture_kind")
    require(
        isinstance(document.get("platform"), str) and document["platform"],
        path,
        "missing platform",
    )
    require(
        document["platform"] in SUPPORTED_REPLAY_PLATFORMS,
        path,
        f"unsupported replay platform: {document['platform']!r}",
    )
    cases = document.get("cases")
    require(isinstance(cases, list) and len(cases) > 0, path, "cases must be a non-empty array")
    seen_case_names: list[str] = []
    for index, case in enumerate(cases):
        case_path = f"cases[{index}]"
        case_object = require_object(case, path, case_path)
        require_exact_keys(case_object, REPLAY_CASE_KEYS, path, case_path)
        case_name = case_object.get("name")
        require(
            isinstance(case_name, str) and case_name,
            path,
            f"{case_path}.name must be a non-empty string",
        )
        require(
            case_name not in seen_case_names,
            path,
            f"duplicate replay case name: {case_name!r}",
        )
        seen_case_names.append(case_name)
        plan = require_object(case_object.get("plan"), path, f"{case_path}.plan")
        require_exact_keys(plan, REPLAY_PLAN_KEYS, path, f"{case_path}.plan")
        require_fingerprint(
            plan.get("plan_fingerprint"), path, f"{case_path}.plan.plan_fingerprint"
        )
        require_fingerprint(
            plan.get("request_fingerprint"), path, f"{case_path}.plan.request_fingerprint"
        )
        require_positive_int(
            plan.get("reduction_block_size"), path, f"{case_path}.plan.reduction_block_size"
        )
        require_positive_int(
            plan.get("worker_threads"), path, f"{case_path}.plan.worker_threads"
        )
        request = require_object(case_object.get("request"), path, f"{case_path}.request")
        require_exact_keys(request, REPLAY_REQUEST_KEYS, path, f"{case_path}.request")
        require(
            request.get("document_kind") == "pricing_request",
            path,
            f"{case_path}.request.document_kind",
        )
        require(
            request.get("schema_version") == 1,
            path,
            f"{case_path}.request.schema_version",
        )
        request_engine = require_object(
            request.get("engine"), path, f"{case_path}.request.engine"
        )
        result = require_object(case_object.get("result"), path, f"{case_path}.result")
        require_exact_keys(result, REPLAY_RESULT_KEYS, path, f"{case_path}.result")
        require(
            result.get("document_kind") == "pricing_result",
            path,
            f"{case_path}.result.document_kind",
        )
        require(
            result.get("schema_version") == 1,
            path,
            f"{case_path}.result.schema_version",
        )
        replay = require_object(result.get("replay"), path, f"{case_path}.result.replay")
        require_exact_keys(replay, REPLAY_METADATA_KEYS, path, f"{case_path}.result.replay")
        require(
            replay.get("schema_version") == 1,
            path,
            f"{case_path}.result.replay.schema_version must be 1",
        )
        require(
            replay.get("library_version") == library_version,
            path,
            f"{case_path}.result.replay.library_version must match Cargo workspace version",
        )
        require(
            replay.get("platform") == document["platform"],
            path,
            f"{case_path}.result.replay.platform must match artifact platform",
        )
        require_fingerprint(
            replay.get("request_fingerprint"), path, f"{case_path}.result.replay.request_fingerprint"
        )
        require(
            replay.get("request_fingerprint") == plan.get("request_fingerprint"),
            path,
            f"{case_path} request fingerprints must match",
        )
        execution = require_object(case_object.get("execution"), path, f"{case_path}.execution")
        require_exact_keys(execution, REPLAY_EXECUTION_KEYS, path, f"{case_path}.execution")
        monte_carlo = require_object(
            execution.get("monte_carlo"), path, f"{case_path}.execution.monte_carlo"
        )
        require_exact_keys(
            monte_carlo,
            REPLAY_MONTE_CARLO_KEYS,
            path,
            f"{case_path}.execution.monte_carlo",
        )
        require_positive_int(
            monte_carlo.get("reduction_block_size"),
            path,
            f"{case_path}.execution.monte_carlo.reduction_block_size",
        )
        require_positive_int(
            monte_carlo.get("worker_threads"),
            path,
            f"{case_path}.execution.monte_carlo.worker_threads",
        )
        require(
            monte_carlo.get("reduction_block_size") == plan.get("reduction_block_size"),
            path,
            f"{case_path} reduction_block_size must match plan",
        )
        require(
            monte_carlo.get("worker_threads") == plan.get("worker_threads"),
            path,
            f"{case_path} worker_threads must match plan",
        )
        risk_methods = require_object(
            execution.get("risk_methods"),
            path,
            f"{case_path}.execution.risk_methods",
        )
        validate_risk_methods(path, f"{case_path}.execution.risk_methods", risk_methods)
        risk_validation = require_object(
            execution.get("risk_validation"),
            path,
            f"{case_path}.execution.risk_validation",
        )
        validate_risk_validation(
            path,
            f"{case_path}.execution.risk_validation",
            risk_validation,
        )
        check_replay_sampling(
            execution,
            monte_carlo,
            request_engine,
            path,
            case_path,
        )
        if fixture_kind == "local_volatility_replay":
            validate_local_vol_case(
                path,
                case_path,
                case_name,
                request,
                result,
            )
    expected_case_names = set(EXPECTED_CASE_NAMES[fixture_kind])
    actual_case_names = set(seen_case_names)
    require(
        actual_case_names == expected_case_names,
        path,
        "replay case set mismatch: "
        f"missing={sorted(expected_case_names - actual_case_names)}, "
        f"unexpected={sorted(actual_case_names - expected_case_names)}",
    )
    require(
        tuple(seen_case_names) == EXPECTED_CASE_NAMES[fixture_kind],
        path,
        "replay case order changed",
    )

def check_python_report(path: Path, library_version: str) -> None:
    document = load_object(path)
    require_exact_keys(
        document,
        REPORT_KEYS,
        path,
        "python report",
    )
    require(document.get("schema_version") == 1, path, "schema_version must be 1")
    require(
        document.get("benchmark_kind") == "python_european_black_scholes",
        path,
        "unexpected benchmark_kind",
    )
    require(
        document.get("library_version") == library_version,
        path,
        "library_version must match Cargo workspace version",
    )
    configuration = require_object(document.get("configuration"), path, "configuration")
    require_exact_keys(
        configuration,
        PYTHON_CONFIGURATION_KEYS,
        path,
        "configuration",
    )
    require(configuration.get("engine") == "pseudo_monte_carlo", path, "unexpected engine")
    require(configuration.get("antithetic") is True, path, "antithetic must be true")
    require_positive_int(configuration.get("sampling_units"), path, "sampling_units")
    require_positive_int(configuration.get("evaluated_paths"), path, "evaluated_paths")
    check_antithetic_path_count(configuration, path)
    require_positive_int(configuration.get("worker_threads"), path, "worker_threads")
    require_positive_int(
        configuration.get("reduction_block_size"), path, "reduction_block_size"
    )
    require(configuration == PYTHON_CONFIGURATION, path, "configuration mismatch")

    measurements = require_object(document.get("measurements"), path, "measurements")
    python_measurement_set = set(PYTHON_MEASUREMENTS)
    missing = sorted(python_measurement_set.difference(measurements))
    require(not missing, path, f"missing measurements: {missing}")
    unexpected = sorted(set(measurements).difference(python_measurement_set))
    require(not unexpected, path, f"unexpected measurements: {unexpected}")
    require(
        list(measurements) == list(PYTHON_MEASUREMENTS),
        path,
        "measurement order changed",
    )
    for name in PYTHON_MEASUREMENTS:
        check_measurement(
            require_object(measurements.get(name), path, name),
            path,
            name,
            expected_paths_for_measurement(name, configuration["evaluated_paths"]),
        )
    require_positive_int(
        document.get("process_peak_memory_bytes"), path, "process_peak_memory_bytes"
    )
    capabilities = require_object(document.get("capabilities"), path, "capabilities")
    require_exact_keys(
        capabilities,
        PYTHON_CAPABILITY_KEYS,
        path,
        "capabilities",
    )
    require(capabilities == PYTHON_CAPABILITIES, path, "capabilities mismatch")
    notes = document.get("notes")
    require_string_array(notes, path, "notes")
    require(notes == list(PYTHON_NOTES), path, "notes mismatch")


def check_measurement(
    document: dict[str, Any],
    path: Path,
    name: str,
    expected_paths_per_sample: int | None,
) -> None:
    require_exact_keys(
        document,
        {
            "evaluated_paths_per_sample",
            "maximum_seconds",
            "median_paths_per_second",
            "median_seconds",
            "minimum_seconds",
            "samples",
        },
        path,
        name,
    )
    require_positive_int(document.get("samples"), path, f"{name}.samples")
    require_positive_float(document.get("median_seconds"), path, f"{name}.median_seconds")
    require_non_negative_float(document.get("minimum_seconds"), path, f"{name}.minimum_seconds")
    require_positive_float(document.get("maximum_seconds"), path, f"{name}.maximum_seconds")
    require(
        document["minimum_seconds"] <= document["median_seconds"] <= document["maximum_seconds"],
        path,
        f"{name} timings are not ordered",
    )
    paths = document.get("evaluated_paths_per_sample")
    paths_per_second = document.get("median_paths_per_second")
    if expected_paths_per_sample is None:
        require(paths is None, path, f"{name}.evaluated_paths_per_sample must be null")
        require(paths_per_second is None, path, f"{name} paths/sec must be null")
    else:
        require_positive_int(paths, path, f"{name}.evaluated_paths_per_sample")
        require(
            paths == expected_paths_per_sample,
            path,
            f"{name}.evaluated_paths_per_sample mismatch",
        )
        require_positive_float(paths_per_second, path, f"{name}.median_paths_per_second")


def expected_paths_for_measurement(name: str, evaluated_paths: int) -> int | None:
    if name.startswith("compile_") or name == "result_value_getter":
        return None
    if "crn_bump_validation" in name and "price_only" in name:
        return evaluated_paths * 5
    if name == "evaluate_crn_bump_validation_from_python":
        return evaluated_paths * 5
    return evaluated_paths


def check_antithetic_path_count(configuration: dict[str, Any], path: Path) -> None:
    sampling_units = configuration["sampling_units"]
    evaluated_paths = configuration["evaluated_paths"]
    multiplicity = 2 if configuration["antithetic"] else 1
    require(
        evaluated_paths == sampling_units * multiplicity,
        path,
        "evaluated_paths must match sampling_units and antithetic",
    )


def check_metadata(path: Path, artifacts: set[str]) -> None:
    document = load_object(path)
    require_exact_keys(document, METADATA_KEYS, path, "metadata")
    require(document.get("schema_version") == 1, path, "schema_version must be 1")
    for key in [
        "platform",
        "machine",
        "python",
        "python_abi",
        "rustc",
        "cargo",
        "target_triple",
        "cargo_lock_sha256",
    ]:
        require_non_empty_string(document.get(key), path, key)
    for key in ["git_sha", "runner_os", "runner_arch", "processor"]:
        require_optional_non_empty_string(document.get(key), path, key)
    if document.get("git_sha") is not None:
        require(
            GIT_SHA.fullmatch(document["git_sha"]) is not None,
            path,
            "git_sha must be a lowercase 40-character commit SHA",
        )
    require(
        f"host: {document['target_triple']}" in document["rustc"].splitlines(),
        path,
        "target_triple must match rustc host",
    )
    require(
        len(document["cargo_lock_sha256"]) == 64
        and all(character in "0123456789abcdef" for character in document["cargo_lock_sha256"]),
        path,
        "cargo_lock_sha256 must be lowercase SHA-256 hex",
    )
    require(
        document["cargo_lock_sha256"] == file_sha256(ROOT / "Cargo.lock"),
        path,
        "cargo_lock_sha256 must match Cargo.lock",
    )
    enabled_features = require_object(document.get("enabled_features"), path, "enabled_features")
    require_exact_keys(enabled_features, ENABLED_FEATURE_KEYS, path, "enabled_features")
    expected_features = {
        "python_wheel": ["pricing-python/extension-module"],
        "rust_benchmarks": [],
    }
    missing_feature_sets = sorted(set(expected_features).difference(enabled_features))
    require(
        not missing_feature_sets,
        path,
        f"missing enabled feature sets: {missing_feature_sets}",
    )
    unexpected_feature_sets = sorted(set(enabled_features).difference(expected_features))
    require(
        not unexpected_feature_sets,
        path,
        f"unexpected enabled feature sets: {unexpected_feature_sets}",
    )
    for key, expected in expected_features.items():
        features = enabled_features.get(key)
        require_string_array(features, path, f"enabled_features.{key}")
        require(features == expected, path, f"enabled_features.{key} mismatch")
    require_positive_int(document.get("peak_memory_bytes"), path, "peak_memory_bytes")
    command_peaks = require_object(
        document.get("command_peak_memory_bytes"), path, "command_peak_memory_bytes"
    )
    expected_command_peaks = expected_command_peak_keys(artifacts)
    require_exact_keys(
        command_peaks,
        expected_command_peaks,
        path,
        "command_peak_memory_bytes",
    )
    for name, peak in command_peaks.items():
        require(isinstance(name, str) and name, path, "command peak name must be non-empty")
        require_positive_int(peak, path, f"command_peak_memory_bytes.{name}")
    require(
        document["peak_memory_bytes"] == max(command_peaks.values()),
        path,
        "peak_memory_bytes must match the maximum command peak",
    )
    require(document.get("allocation_count") is None, path, "allocation_count must be null")
    require_string_array(document.get("unavailable_metrics"), path, "unavailable_metrics")
    require(
        document["unavailable_metrics"] == list(UNAVAILABLE_METRICS),
        path,
        "unavailable_metrics mismatch",
    )


def expected_command_peak_keys(artifacts: set[str]) -> tuple[str, ...]:
    expected_command_peaks = set(COMMAND_PEAK_KEYS)
    for artifact, command_name in OPTIONAL_COMMAND_PEAKS.items():
        if artifact in artifacts:
            expected_command_peaks.add(command_name)
    return tuple(sorted(expected_command_peaks))


def load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing benchmark artifact: {path}")
    raw = path.read_bytes()
    require(not raw.startswith(b"\xef\xbb\xbf"), path, "UTF-8 BOM is not allowed")
    require(b"\r" not in raw, path, "CR or CRLF line endings are not allowed")
    require(raw.endswith(b"\n"), path, "JSON artifact must end with LF")
    require(not raw.endswith(b"\n\n"), path, "JSON artifact must end with exactly one LF")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_json_constant,
        )
    except ValueError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise SystemExit(f"{path}: top-level JSON value must be an object")
    return document


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def require_object(value: Any, path: Path, name: str) -> dict[str, Any]:
    require(isinstance(value, dict), path, f"{name} must be an object")
    return value


def require_exact_keys(
    document: dict[str, Any],
    expected_keys: set[str] | Sequence[str],
    path: Path,
    name: str,
) -> None:
    expected_key_set = set(expected_keys)
    actual_keys = set(document)
    missing = sorted(expected_key_set.difference(actual_keys))
    require(not missing, path, f"{name} missing keys: {missing}")
    unexpected = sorted(actual_keys.difference(expected_key_set))
    require(not unexpected, path, f"{name} unexpected keys: {unexpected}")
    if not isinstance(expected_keys, set):
        require(list(document) == list(expected_keys), path, f"{name} key order changed")


def require_non_empty_string(value: Any, path: Path, name: str) -> None:
    require(isinstance(value, str) and value, path, f"{name} must be a non-empty string")


def require_optional_non_empty_string(value: Any, path: Path, name: str) -> None:
    require(
        value is None or (isinstance(value, str) and value),
        path,
        f"{name} must be null or a non-empty string",
    )


def require_fingerprint(value: Any, path: Path, name: str) -> None:
    require(
        isinstance(value, str) and FINGERPRINT.fullmatch(value) is not None,
        path,
        f"{name} must be a blake3-256 fingerprint",
    )


def check_replay_sampling(
    execution: dict[str, Any],
    monte_carlo: dict[str, Any],
    request_engine: dict[str, Any],
    path: Path,
    case_path: str,
) -> None:
    sampling_units = execution.get("independent_sampling_units")
    require_positive_int(
        sampling_units, path, f"{case_path}.execution.independent_sampling_units"
    )
    evaluated_paths = require_positive_decimal_int(
        execution.get("evaluated_paths"), path, f"{case_path}.execution.evaluated_paths"
    )
    antithetic = monte_carlo.get("antithetic")
    require(
        isinstance(antithetic, bool),
        path,
        f"{case_path}.execution.monte_carlo.antithetic",
    )
    multiplicity = 2 if antithetic else 1
    scramble_count = monte_carlo.get("scramble_count")
    if scramble_count is None:
        expected_paths = sampling_units * multiplicity
    else:
        require_positive_int(
            scramble_count, path, f"{case_path}.execution.monte_carlo.scramble_count"
        )
        require(
            request_engine.get("scramble_count") == scramble_count,
            path,
            f"{case_path} scramble_count must match request",
        )
        require(
            sampling_units == scramble_count,
            path,
            f"{case_path} independent_sampling_units must match scramble_count",
        )
        points_per_scramble = request_engine.get("points_per_scramble")
        require_positive_int(
            points_per_scramble, path, f"{case_path}.request.engine.points_per_scramble"
        )
        expected_paths = points_per_scramble * scramble_count * multiplicity
    require(
        evaluated_paths == expected_paths,
        path,
        f"{case_path} evaluated_paths must match the sampling policy",
    )


def require_string_array(value: Any, path: Path, name: str) -> None:
    require(isinstance(value, list), path, f"{name} must be a string array")
    for index, item in enumerate(value):
        require(isinstance(item, str), path, f"{name}[{index}] must be a string")


def require_positive_int(value: Any, path: Path, name: str) -> None:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        path,
        f"{name} must be a positive integer",
    )


def require_positive_decimal_int(value: Any, path: Path, name: str) -> int:
    require(
        isinstance(value, str) and value.isdecimal() and int(value) > 0,
        path,
        f"{name} must be a positive decimal integer string",
    )
    return int(value)


def require_positive_float(value: Any, path: Path, name: str) -> None:
    require(
        is_finite_number(value) and value > 0.0,
        path,
        f"{name} must be a positive number",
    )


def require_non_negative_float(value: Any, path: Path, name: str) -> None:
    require(
        is_finite_number(value) and value >= 0.0,
        path,
        f"{name} must be a non-negative number",
    )


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def require(condition: bool, path: Path, message: str) -> None:
    if not condition:
        raise SystemExit(f"{path}: {message}")


if __name__ == "__main__":
    main()
