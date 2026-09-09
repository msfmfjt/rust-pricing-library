"""Compare generated replay evidence with the frozen supported-platform fixture."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PREFIXES = {
    "european_black_scholes_replay": "european_bs",
    "local_volatility_replay": "local_volatility",
}
EXPECTED_CASE_NAMES = {
    "european_black_scholes_replay": {
        "pseudo_mc_full_risk",
        "rqmc_full_risk",
    },
    "local_volatility_replay": {
        "pseudo_mc_price_only",
        "rqmc_price_only",
        "pseudo_mc_delta_gamma_vega_vegakt",
        "rqmc_delta_gamma_vega_vegakt",
    },
}
SUPPORTED_PLATFORMS = {
    "macos-aarch64",
    "windows-x86_64",
}
FINGERPRINT = re.compile(r"^blake3-256:[0-9a-f]{64}$")
REPLAY_DOCUMENT_KEYS = {"cases", "fixture_kind", "platform", "schema_version"}
REPLAY_CASE_KEYS = {"execution", "name", "plan", "request", "result"}
REPLAY_PLAN_KEYS = {
    "plan_fingerprint",
    "reduction_block_size",
    "request_fingerprint",
    "worker_threads",
}
REPLAY_EXECUTION_KEYS = {
    "estimator_variance_bits",
    "evaluated_paths",
    "independent_sampling_units",
    "monte_carlo",
    "risk_methods",
    "risk_validation",
    "sampling_variance_bits",
}
REPLAY_MONTE_CARLO_KEYS = {
    "aad_tile_capacity",
    "aad_tile_policy_version",
    "antithetic",
    "checkpoint_interval",
    "checkpoint_policy_version",
    "direction_checksum",
    "discount_region",
    "dividend_region",
    "estimator",
    "master_seed",
    "payoff_fingerprint",
    "policy_version",
    "reduction_block_size",
    "scramble_checksum",
    "scramble_count",
    "worker_threads",
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_replay_fixture.py <generated.json>")

    generated = Path(sys.argv[1])
    library_version = workspace_package_version()
    generated_text, document = load_object(generated)
    fixture_kind, platform = replay_identity(generated, document, library_version)
    fixture_prefix = FIXTURE_PREFIXES[fixture_kind]

    expected = Path("fixtures/replay") / f"{fixture_prefix}-{platform}.json"
    if not expected.is_file():
        raise SystemExit(f"no frozen replay fixture for {platform}: {expected}")

    expected_text, expected_document = load_object(expected)
    expected_fixture_kind, expected_platform = replay_identity(
        expected,
        expected_document,
        library_version,
    )
    if expected_fixture_kind != fixture_kind or expected_platform != platform:
        raise SystemExit(
            f"{expected}: fixture identity "
            f"{expected_fixture_kind!r}/{expected_platform!r} does not match generated "
            f"{fixture_kind!r}/{platform!r}"
        )

    if generated_text != expected_text:
        diff = difflib.unified_diff(
            expected_text.splitlines(),
            generated_text.splitlines(),
            fromfile=str(expected),
            tofile=str(generated),
            lineterm="",
        )
        raise SystemExit("replay fixture mismatch:\n" + "\n".join(diff))

    print(f"replay fixture matches {expected}")


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


def replay_identity(
    path: Path,
    document: dict[str, object],
    library_version: str,
) -> tuple[str, str]:
    require_exact_keys(path, document, REPLAY_DOCUMENT_KEYS, "replay document")
    schema_version = document.get("schema_version")
    if schema_version != 1:
        raise SystemExit(f"{path}: schema_version must be 1")
    platform = document.get("platform")
    if not isinstance(platform, str) or not platform:
        raise SystemExit(f"{path}: replay evidence has no non-empty string platform field")
    if platform not in SUPPORTED_PLATFORMS:
        raise SystemExit(f"{path}: unsupported replay platform: {platform!r}")
    fixture_kind = document.get("fixture_kind")
    if not isinstance(fixture_kind, str):
        raise SystemExit(f"{path}: replay evidence has no string fixture_kind field")
    if fixture_kind not in FIXTURE_PREFIXES:
        raise SystemExit(f"{path}: unsupported replay fixture_kind: {fixture_kind!r}")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"{path}: replay evidence cases must be a non-empty array")
    case_names = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise SystemExit(f"{path}: cases[{index}] must be an object")
        require_exact_keys(path, case, REPLAY_CASE_KEYS, f"cases[{index}]")
        name = case.get("name")
        if not isinstance(name, str) or not name:
            raise SystemExit(f"{path}: cases[{index}].name must be a non-empty string")
        if name in case_names:
            raise SystemExit(f"{path}: duplicate replay case name: {name}")
        case_names.add(name)
        validate_case(path, index, case, fixture_kind, platform, library_version)
    expected_case_names = EXPECTED_CASE_NAMES[fixture_kind]
    if case_names != expected_case_names:
        raise SystemExit(
            f"{path}: replay case set mismatch; "
            f"missing={sorted(expected_case_names - case_names)}, "
            f"unexpected={sorted(case_names - expected_case_names)}"
        )
    return fixture_kind, platform


def validate_case(
    path: Path,
    index: int,
    case: dict[str, object],
    fixture_kind: str,
    platform: str,
    library_version: str,
) -> None:
    case_path = f"cases[{index}]"
    plan = require_object(path, case.get("plan"), f"{case_path}.plan")
    require_exact_keys(path, plan, REPLAY_PLAN_KEYS, f"{case_path}.plan")
    request = require_object(path, case.get("request"), f"{case_path}.request")
    result = require_object(path, case.get("result"), f"{case_path}.result")
    execution = require_object(path, case.get("execution"), f"{case_path}.execution")
    require_exact_keys(path, execution, REPLAY_EXECUTION_KEYS, f"{case_path}.execution")
    monte_carlo = require_object(
        path, execution.get("monte_carlo"), f"{case_path}.execution.monte_carlo"
    )
    require_exact_keys(
        path,
        monte_carlo,
        REPLAY_MONTE_CARLO_KEYS,
        f"{case_path}.execution.monte_carlo",
    )

    require_fingerprint(path, plan.get("plan_fingerprint"), f"{case_path}.plan.plan_fingerprint")
    request_fingerprint = require_fingerprint(
        path, plan.get("request_fingerprint"), f"{case_path}.plan.request_fingerprint"
    )
    if request.get("document_kind") != "pricing_request":
        raise SystemExit(f"{path}: {case_path}.request.document_kind must be pricing_request")
    if request.get("schema_version") != 1:
        raise SystemExit(f"{path}: {case_path}.request.schema_version must be 1")
    if result.get("document_kind") != "pricing_result":
        raise SystemExit(f"{path}: {case_path}.result.document_kind must be pricing_result")
    if result.get("schema_version") != 1:
        raise SystemExit(f"{path}: {case_path}.result.schema_version must be 1")
    replay = require_object(path, result.get("replay"), f"{case_path}.result.replay")
    result_request_fingerprint = require_fingerprint(
        path,
        replay.get("request_fingerprint"),
        f"{case_path}.result.replay.request_fingerprint",
    )
    if result_request_fingerprint != request_fingerprint:
        raise SystemExit(
            f"{path}: {case_path} plan/result request fingerprints do not match"
        )
    if replay.get("schema_version") != 1:
        raise SystemExit(f"{path}: {case_path}.result.replay.schema_version must be 1")
    replay_library_version = replay.get("library_version")
    if replay_library_version != library_version:
        raise SystemExit(
            f"{path}: {case_path}.result.replay.library_version must match Cargo workspace version"
        )
    if replay.get("platform") != platform:
        raise SystemExit(
            f"{path}: {case_path}.result.replay.platform must match artifact platform"
        )
    if fixture_kind == "local_volatility_replay":
        validate_local_vol_case(path, case_path, case["name"], request, result)


def validate_local_vol_case(
    path: Path,
    case_path: str,
    name: object,
    request: dict[str, object],
    result: dict[str, object],
) -> None:
    risks = require_object(path, result.get("risks"), f"{case_path}.result.risks")
    request_risk = require_object(path, request.get("risk"), f"{case_path}.request.risk")
    if name in {"pseudo_mc_price_only", "rqmc_price_only"}:
        if "vega_kt" in risks or "vega_kt" in request_risk:
            raise SystemExit(f"{path}: {case_path} price-only case must not carry VegaKT")
        return

    request_vega_kt = require_object(
        path, request_risk.get("vega_kt"), f"{case_path}.request.risk.vega_kt"
    )
    if request_vega_kt.get("full_bucket_covariance") is not True:
        raise SystemExit(
            f"{path}: {case_path}.request.risk.vega_kt.full_bucket_covariance must be true"
        )
    result_vega_kt = require_object(
        path, risks.get("vega_kt"), f"{case_path}.result.risks.vega_kt"
    )
    layout = require_object(
        path,
        result_vega_kt.get("covariance_layout"),
        f"{case_path}.result.risks.vega_kt.covariance_layout",
    )
    if layout.get("type") != "full_bucket_matrix_row_major":
        raise SystemExit(
            f"{path}: {case_path}.result.risks.vega_kt must use full covariance layout"
        )
    covariance = result_vega_kt.get("full_bucket_covariance")
    if not isinstance(covariance, list) or len(covariance) != 36:
        raise SystemExit(
            f"{path}: {case_path}.result.risks.vega_kt.full_bucket_covariance "
            "must contain 36 row-major entries"
        )


def require_object(path: Path, value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: {field} must be an object")
    return value


def require_exact_keys(
    path: Path,
    document: dict[str, object],
    expected_keys: set[str],
    field: str,
) -> None:
    actual_keys = set(document)
    missing = sorted(expected_keys.difference(actual_keys))
    if missing:
        raise SystemExit(f"{path}: {field} missing keys: {missing}")
    unexpected = sorted(actual_keys.difference(expected_keys))
    if unexpected:
        raise SystemExit(f"{path}: {field} unexpected keys: {unexpected}")


def require_fingerprint(path: Path, value: object, field: str) -> str:
    if not isinstance(value, str) or FINGERPRINT.fullmatch(value) is None:
        raise SystemExit(f"{path}: {field} must be a BLAKE3-256 fingerprint")
    return value


def load_object(path: Path) -> tuple[str, dict[str, object]]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise SystemExit(f"{path}: UTF-8 BOM is not allowed")
    if b"\r" in raw:
        raise SystemExit(f"{path}: CR or CRLF line endings are not allowed")
    if not raw.endswith(b"\n"):
        raise SystemExit(f"{path}: JSON artifact must end with LF")
    if raw.endswith(b"\n\n"):
        raise SystemExit(f"{path}: JSON artifact must end with exactly one LF")
    try:
        text = raw.decode("utf-8")
        document = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_json_constant,
        )
    except ValueError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise SystemExit(f"{path}: top-level JSON value must be an object")
    return text, document


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate key: {key}")
        document[key] = value
    return document


def reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


if __name__ == "__main__":
    main()
