"""Validate benchmark reports before retaining CI artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from typing import Any


EUROPEAN_MEASUREMENTS = {
    "compile_price_only",
    "evaluate_price_only",
    "compile_full_risk",
    "evaluate_aad_with_crn_bump_validation",
    "compile_crn_bump_validation",
    "evaluate_crn_bump_validation_price_only",
}

LOCAL_VOL_MEASUREMENTS = {
    "compile_price_only",
    "evaluate_price_only",
    "compile_aad_local_vega",
    "evaluate_aad_local_vega",
    "compile_vega_kt_decomposition",
    "evaluate_vega_kt_decomposition",
    "compile_selected_crn_bump_validation",
    "evaluate_selected_crn_bump_validation_price_only",
}

PYTHON_MEASUREMENTS = {
    "compile_full_risk_from_python",
    "evaluate_full_risk_from_python",
    "compile_crn_bump_validation_from_python",
    "evaluate_crn_bump_validation_from_python",
    "result_value_getter",
}

EXPECTED_ARTIFACTS = {
    "rust.json",
    "local-volatility-rust.json",
    "python.json",
    "replay.json",
    "metadata.json",
}
OPTIONAL_ARTIFACTS = {"local-volatility-replay.json"}
EXPECTED_COMMAND_PEAKS = {
    "rust_european_black_scholes",
    "rust_local_volatility_vegakt",
    "replay_european_black_scholes",
    "python_european_black_scholes",
}
OPTIONAL_COMMAND_PEAKS = {
    "local-volatility-replay.json": "replay_local_volatility",
}
SUPPORTED_REPLAY_PLATFORMS = {
    "macos-aarch64",
    "windows-x86_64",
}
FINGERPRINT = re.compile(r"^blake3-256:[0-9a-f]{64}$")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_benchmark_reports.py <benchmark-results-dir>")

    root = Path(sys.argv[1])
    artifacts = check_artifact_set(root)
    check_report(root / "rust.json", "rust_european_black_scholes", EUROPEAN_MEASUREMENTS)
    check_report(
        root / "local-volatility-rust.json",
        "rust_local_volatility_vegakt",
        LOCAL_VOL_MEASUREMENTS,
        local_volatility=True,
    )
    check_python_report(root / "python.json")
    check_replay_report(root / "replay.json", "european_black_scholes_replay")
    local_volatility_replay = root / "local-volatility-replay.json"
    if local_volatility_replay.is_file():
        check_replay_report(local_volatility_replay, "local_volatility_replay")
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


def check_report(
    path: Path,
    benchmark_kind: str,
    required_measurements: set[str],
    *,
    local_volatility: bool = False,
) -> None:
    document = load_object(path)
    require(document.get("schema_version") == 1, path, "schema_version must be 1")
    require(document.get("benchmark_kind") == benchmark_kind, path, "unexpected benchmark_kind")
    require(isinstance(document.get("library_version"), str), path, "missing library_version")
    configuration = require_object(document.get("configuration"), path, "configuration")
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

    measurements = require_object(document.get("measurements"), path, "measurements")
    missing = sorted(required_measurements.difference(measurements))
    require(not missing, path, f"missing measurements: {missing}")
    unexpected = sorted(set(measurements).difference(required_measurements))
    require(not unexpected, path, f"unexpected measurements: {unexpected}")
    for name in sorted(required_measurements):
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
    require(
        capabilities.get("peak_memory_available_in_process") is True,
        path,
        "peak memory capability must be true",
    )
    if local_volatility:
        require(
            capabilities.get("aad_local_vega_timing_available") is True,
            path,
            "AAD Local Vega timing capability must be true",
        )
        require(
            capabilities.get("vega_kt_decomposition_timing_available") is True,
            path,
            "VegaKT timing capability must be true",
        )
    require_string_array(document.get("notes"), path, "notes")


def check_replay_report(path: Path, fixture_kind: str) -> None:
    document = load_object(path)
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
    seen_case_names: set[str] = set()
    for index, case in enumerate(cases):
        case_path = f"cases[{index}]"
        case_object = require_object(case, path, case_path)
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
        seen_case_names.add(case_name)
        plan = require_object(case_object.get("plan"), path, f"{case_path}.plan")
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
        require_fingerprint(
            replay.get("request_fingerprint"), path, f"{case_path}.result.replay.request_fingerprint"
        )
        require(
            replay.get("request_fingerprint") == plan.get("request_fingerprint"),
            path,
            f"{case_path} request fingerprints must match",
        )
        execution = require_object(case_object.get("execution"), path, f"{case_path}.execution")
        monte_carlo = require_object(
            execution.get("monte_carlo"), path, f"{case_path}.execution.monte_carlo"
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
        check_replay_sampling(
            execution,
            monte_carlo,
            request_engine,
            path,
            case_path,
        )


def check_python_report(path: Path) -> None:
    document = load_object(path)
    require(document.get("schema_version") == 1, path, "schema_version must be 1")
    require(
        document.get("benchmark_kind") == "python_european_black_scholes",
        path,
        "unexpected benchmark_kind",
    )
    require(isinstance(document.get("library_version"), str), path, "missing library_version")
    configuration = require_object(document.get("configuration"), path, "configuration")
    require(configuration.get("engine") == "pseudo_monte_carlo", path, "unexpected engine")
    require(configuration.get("antithetic") is True, path, "antithetic must be true")
    require_positive_int(configuration.get("sampling_units"), path, "sampling_units")
    require_positive_int(configuration.get("evaluated_paths"), path, "evaluated_paths")
    check_antithetic_path_count(configuration, path)
    require_positive_int(configuration.get("worker_threads"), path, "worker_threads")
    require_positive_int(
        configuration.get("reduction_block_size"), path, "reduction_block_size"
    )

    measurements = require_object(document.get("measurements"), path, "measurements")
    missing = sorted(PYTHON_MEASUREMENTS.difference(measurements))
    require(not missing, path, f"missing measurements: {missing}")
    unexpected = sorted(set(measurements).difference(PYTHON_MEASUREMENTS))
    require(not unexpected, path, f"unexpected measurements: {unexpected}")
    for name in sorted(PYTHON_MEASUREMENTS):
        check_measurement(
            require_object(measurements.get(name), path, name),
            path,
            name,
            expected_paths_for_measurement(name, configuration["evaluated_paths"]),
        )
    require_positive_int(
        document.get("process_peak_memory_bytes"), path, "process_peak_memory_bytes"
    )
    require_string_array(document.get("notes"), path, "notes")


def check_measurement(
    document: dict[str, Any],
    path: Path,
    name: str,
    expected_paths_per_sample: int | None,
) -> None:
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
    require(
        len(document["cargo_lock_sha256"]) == 64
        and all(character in "0123456789abcdef" for character in document["cargo_lock_sha256"]),
        path,
        "cargo_lock_sha256 must be lowercase SHA-256 hex",
    )
    enabled_features = require_object(document.get("enabled_features"), path, "enabled_features")
    for key in ["rust_benchmarks", "python_wheel"]:
        features = enabled_features.get(key)
        require_string_array(features, path, f"enabled_features.{key}")
    require_positive_int(document.get("peak_memory_bytes"), path, "peak_memory_bytes")
    command_peaks = require_object(
        document.get("command_peak_memory_bytes"), path, "command_peak_memory_bytes"
    )
    expected_command_peaks = set(EXPECTED_COMMAND_PEAKS)
    for artifact, command_name in OPTIONAL_COMMAND_PEAKS.items():
        if artifact in artifacts:
            expected_command_peaks.add(command_name)
    missing = sorted(expected_command_peaks.difference(command_peaks))
    require(not missing, path, f"missing command_peak_memory_bytes entries: {missing}")
    unexpected = sorted(set(command_peaks).difference(expected_command_peaks))
    require(not unexpected, path, f"unexpected command_peak_memory_bytes entries: {unexpected}")
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


def require_non_empty_string(value: Any, path: Path, name: str) -> None:
    require(isinstance(value, str) and value, path, f"{name} must be a non-empty string")


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


def require(condition: bool, path: Path, message: str) -> None:
    if not condition:
        raise SystemExit(f"{path}: {message}")


if __name__ == "__main__":
    main()
