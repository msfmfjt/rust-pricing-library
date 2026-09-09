"""Validate benchmark reports before retaining CI artifacts."""

from __future__ import annotations

import json
from pathlib import Path
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


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_benchmark_reports.py <benchmark-results-dir>")

    root = Path(sys.argv[1])
    check_artifact_set(root)
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
    check_metadata(root / "metadata.json")
    print(f"benchmark reports are valid in {root}")


def check_artifact_set(root: Path) -> None:
    actual = {path.name for path in root.glob("*.json")}
    missing = sorted(EXPECTED_ARTIFACTS.difference(actual))
    if missing:
        raise SystemExit(f"{root}: missing benchmark artifacts: {missing}")
    expected = EXPECTED_ARTIFACTS.union(OPTIONAL_ARTIFACTS)
    unexpected = sorted(actual.difference(expected))
    if unexpected:
        raise SystemExit(f"{root}: unexpected benchmark artifacts: {unexpected}")


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
        check_measurement(require_object(measurements.get(name), path, name), path, name)
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
    require(isinstance(document.get("notes"), list), path, "notes must be an array")


def check_replay_report(path: Path, fixture_kind: str) -> None:
    document = load_object(path)
    require(document.get("fixture_kind") == fixture_kind, path, "unexpected fixture_kind")
    require(isinstance(document.get("platform"), str), path, "missing platform")
    cases = document.get("cases")
    require(isinstance(cases, list) and len(cases) > 0, path, "cases must be a non-empty array")
    for index, case in enumerate(cases):
        case_path = f"cases[{index}]"
        case_object = require_object(case, path, case_path)
        require(isinstance(case_object.get("name"), str), path, f"{case_path}.name")
        require_object(case_object.get("request"), path, f"{case_path}.request")
        require_object(case_object.get("result"), path, f"{case_path}.result")
        execution = require_object(case_object.get("execution"), path, f"{case_path}.execution")
        require_object(execution.get("monte_carlo"), path, f"{case_path}.execution.monte_carlo")


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
        check_measurement(require_object(measurements.get(name), path, name), path, name)
    require_positive_int(
        document.get("process_peak_memory_bytes"), path, "process_peak_memory_bytes"
    )
    require(isinstance(document.get("notes"), list), path, "notes must be an array")


def check_measurement(document: dict[str, Any], path: Path, name: str) -> None:
    require_positive_int(document.get("samples"), path, f"{name}.samples")
    require_positive_float(document.get("median_seconds"), path, f"{name}.median_seconds")
    require_positive_float(document.get("minimum_seconds"), path, f"{name}.minimum_seconds")
    require_positive_float(document.get("maximum_seconds"), path, f"{name}.maximum_seconds")
    require(
        document["minimum_seconds"] <= document["median_seconds"] <= document["maximum_seconds"],
        path,
        f"{name} timings are not ordered",
    )
    paths = document.get("evaluated_paths_per_sample")
    paths_per_second = document.get("median_paths_per_second")
    if paths is None:
        require(paths_per_second is None, path, f"{name} paths/sec must be null")
    else:
        require_positive_int(paths, path, f"{name}.evaluated_paths_per_sample")
        require_positive_float(paths_per_second, path, f"{name}.median_paths_per_second")


def check_metadata(path: Path) -> None:
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
        require(isinstance(document.get(key), str), path, f"missing {key}")
    require(
        len(document["cargo_lock_sha256"]) == 64
        and all(character in "0123456789abcdef" for character in document["cargo_lock_sha256"]),
        path,
        "cargo_lock_sha256 must be lowercase SHA-256 hex",
    )
    enabled_features = require_object(document.get("enabled_features"), path, "enabled_features")
    for key in ["rust_benchmarks", "python_wheel"]:
        features = enabled_features.get(key)
        require(
            isinstance(features, list) and all(isinstance(feature, str) for feature in features),
            path,
            f"enabled_features.{key} must be a string array",
        )
    require_positive_int(document.get("peak_memory_bytes"), path, "peak_memory_bytes")
    command_peaks = require_object(
        document.get("command_peak_memory_bytes"), path, "command_peak_memory_bytes"
    )
    for name, peak in command_peaks.items():
        require(isinstance(name, str) and name, path, "command peak name must be non-empty")
        require_positive_int(peak, path, f"command_peak_memory_bytes.{name}")
    require(document.get("allocation_count") is None, path, "allocation_count must be null")
    require(
        isinstance(document.get("unavailable_metrics"), list),
        path,
        "unavailable_metrics must be an array",
    )


def load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing benchmark artifact: {path}")
    raw = path.read_bytes()
    require(not raw.startswith(b"\xef\xbb\xbf"), path, "UTF-8 BOM is not allowed")
    require(b"\r" not in raw, path, "CR or CRLF line endings are not allowed")
    require(raw.endswith(b"\n"), path, "JSON artifact must end with LF")
    require(not raw.endswith(b"\n\n"), path, "JSON artifact must end with exactly one LF")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
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


def require_object(value: Any, path: Path, name: str) -> dict[str, Any]:
    require(isinstance(value, dict), path, f"{name} must be an object")
    return value


def require_positive_int(value: Any, path: Path, name: str) -> None:
    require(isinstance(value, int) and value > 0, path, f"{name} must be a positive integer")


def require_positive_float(value: Any, path: Path, name: str) -> None:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0.0,
        path,
        f"{name} must be a positive number",
    )


def require(condition: bool, path: Path, message: str) -> None:
    if not condition:
        raise SystemExit(f"{path}: {message}")


if __name__ == "__main__":
    main()
