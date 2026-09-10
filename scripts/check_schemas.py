"""Validate bundled JSON Schema golden artifacts.

This check intentionally uses only the Python standard library so schema
regression coverage stays available before project dependencies are installed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = ROOT / "schemas" / "v1"
GOLDEN_ROOT = ROOT / "fixtures" / "v1"
DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
EXPECTED_SCHEMAS = {
    "pricing_request": SCHEMA_ROOT / "pricing_request.schema.json",
    "pricing_result": SCHEMA_ROOT / "pricing_result.schema.json",
}
EXPECTED_GOLDENS = {
    "pricing_request": GOLDEN_ROOT / "pricing_request.golden.json",
    "pricing_result": GOLDEN_ROOT / "pricing_result.golden.json",
}
EXPECTED_SCHEMA_DEFS = {
    "pricing_request": {
        "asian_observation",
        "asian_observation_value",
        "barrier_direction",
        "barrier_style",
        "curve",
        "digital_payout",
        "dividend_event",
        "dividend_quote",
        "engine",
        "gamma",
        "local_variance_grid",
        "market",
        "model",
        "product",
        "reporting_iv_basis",
        "risk",
        "side",
        "smile_dynamics",
        "spot_bump",
        "variance_reduction",
        "vega_kt",
    },
    "pricing_result": {
        "diagnostics",
        "estimate",
        "estimator",
        "reporting_iv_projection_stats",
        "risk_estimate",
        "risk_report",
        "risk_unit",
        "vega_kt_bucket_estimate",
        "vega_kt_bucket_unit",
        "vega_kt_coordinate",
        "vega_kt_covariance_entry",
        "vega_kt_covariance_layout",
        "vega_kt_projection",
        "vega_kt_report",
        "vega_kt_residual_diagnostics",
        "warning",
    },
}
EXPECTED_TOP_LEVEL_REQUIRED = {
    "pricing_request": [
        "document_kind",
        "schema_version",
        "valuation_date",
        "product",
        "market",
        "model",
        "engine",
        "risk",
    ],
    "pricing_result": [
        "document_kind",
        "schema_version",
        "value",
        "risks",
        "diagnostics",
        "replay",
    ],
}
EXPECTED_SCHEMA_TOP_LEVEL_KEYS = [
    "$schema",
    "$id",
    "title",
    "type",
    "additionalProperties",
    "required",
    "properties",
    "$defs",
]
EXPECTED_SCHEMA_DEF_ORDER = {
    "pricing_request": [
        "side",
        "product",
        "asian_observation",
        "asian_observation_value",
        "digital_payout",
        "barrier_direction",
        "barrier_style",
        "curve",
        "market",
        "dividend_event",
        "dividend_quote",
        "model",
        "local_variance_grid",
        "reporting_iv_basis",
        "variance_reduction",
        "engine",
        "risk",
        "gamma",
        "spot_bump",
        "vega_kt",
        "smile_dynamics",
    ],
    "pricing_result": [
        "estimate",
        "risk_report",
        "risk_estimate",
        "vega_kt_report",
        "diagnostics",
        "warning",
        "estimator",
        "risk_unit",
        "vega_kt_covariance_layout",
        "vega_kt_bucket_unit",
        "vega_kt_coordinate",
        "vega_kt_bucket_estimate",
        "vega_kt_covariance_entry",
        "vega_kt_projection",
        "vega_kt_residual_diagnostics",
        "reporting_iv_projection_stats",
    ],
}
EXPECTED_GOLDEN_KEYS = {
    "pricing_request": set(EXPECTED_TOP_LEVEL_REQUIRED["pricing_request"]),
    "pricing_result": set(EXPECTED_TOP_LEVEL_REQUIRED["pricing_result"]),
}
EXPECTED_REQUIRED_PROPERTIES = {
    "pricing_request": {
        (): EXPECTED_TOP_LEVEL_REQUIRED["pricing_request"],
        ("$defs", "side", "oneOf", 0): ["type"],
        ("$defs", "side", "oneOf", 1): ["type"],
        ("$defs", "product", "oneOf", 0): [
            "type",
            "underlying_id",
            "currency_id",
            "expiry",
            "strike",
            "notional",
            "side",
        ],
        ("$defs", "product", "oneOf", 1): [
            "type",
            "underlying_id",
            "currency_id",
            "expiry",
            "strike",
            "payout",
            "side",
            "payout_kind",
        ],
        ("$defs", "product", "oneOf", 2): [
            "type",
            "underlying_id",
            "currency_id",
            "expiry",
            "strike",
            "barrier",
            "notional",
            "side",
            "direction",
            "style",
            "monitoring_dates",
            "payment_date",
        ],
        ("$defs", "product", "oneOf", 3): [
            "type",
            "underlying_id",
            "currency_id",
            "strike",
            "notional",
            "side",
            "observations",
            "payment_date",
        ],
        ("$defs", "product", "oneOf", 4): [
            "type",
            "underlying_id",
            "currency_id",
            "strike",
            "notional",
            "side",
            "monitoring_dates",
            "payment_date",
        ],
        ("$defs", "asian_observation"): ["date", "weight", "value"],
        ("$defs", "asian_observation_value", "oneOf", 0): ["type", "fixing"],
        ("$defs", "asian_observation_value", "oneOf", 1): ["type"],
        ("$defs", "digital_payout", "oneOf", 0): ["type"],
        ("$defs", "digital_payout", "oneOf", 1): ["type"],
        ("$defs", "barrier_direction", "oneOf", 0): ["type"],
        ("$defs", "barrier_direction", "oneOf", 1): ["type"],
        ("$defs", "barrier_style", "oneOf", 0): ["type"],
        ("$defs", "barrier_style", "oneOf", 1): ["type"],
        ("$defs", "curve"): ["curve_id", "times", "discount_factors"],
        ("$defs", "market"): [
            "type",
            "currency_id",
            "underlying_id",
            "spot",
            "discount_curve",
            "dividend_curve",
        ],
        ("$defs", "dividend_event"): ["event_id", "ex_time", "quote"],
        ("$defs", "dividend_quote", "oneOf", 0): ["type", "amount"],
        ("$defs", "dividend_quote", "oneOf", 1): ["type", "beta"],
        ("$defs", "dividend_quote", "oneOf", 2): [
            "type",
            "fixed_cash",
            "beta",
        ],
        ("$defs", "model", "oneOf", 0): ["type", "volatility"],
        ("$defs", "model", "oneOf", 1): ["type", "volatility"],
        ("$defs", "model", "oneOf", 2): ["type", "local_variance_grid"],
        ("$defs", "local_variance_grid"): [
            "time_nodes",
            "log_forward_moneyness_nodes",
            "shape",
            "values",
            "floor",
            "cap",
        ],
        ("$defs", "reporting_iv_basis"): [
            "maturity_nodes",
            "log_forward_moneyness_nodes",
            "shape",
            "implied_volatilities",
        ],
        ("$defs", "variance_reduction"): ["antithetic", "brownian_bridge"],
        ("$defs", "engine", "oneOf", 0): [
            "type",
            "master_seed",
            "independent_sampling_units",
            "variance_reduction",
        ],
        ("$defs", "engine", "oneOf", 1): [
            "type",
            "points_per_scramble",
            "scramble_count",
            "master_scramble_seed",
            "variance_reduction",
        ],
        ("$defs", "risk"): ["delta", "vega", "smile_dynamics"],
        ("$defs", "gamma"): ["bump"],
        ("$defs", "spot_bump", "oneOf", 0): ["type", "value"],
        ("$defs", "spot_bump", "oneOf", 1): ["type", "value"],
        ("$defs", "vega_kt"): [
            "maturity_nodes",
            "log_forward_moneyness_nodes",
            "relative_density_threshold",
            "full_bucket_covariance",
        ],
        ("$defs", "smile_dynamics", "oneOf", 0): ["type"],
        ("$defs", "smile_dynamics", "oneOf", 1): ["type"],
        ("$defs", "smile_dynamics", "oneOf", 2): ["type"],
    },
    "pricing_result": {
        (): EXPECTED_TOP_LEVEL_REQUIRED["pricing_result"],
        ("properties", "replay"): [
            "schema_version",
            "request_fingerprint",
            "library_version",
            "platform",
        ],
        ("$defs", "estimate"): [
            "value",
            "standard_error",
            "confidence_lower",
            "confidence_upper",
            "estimator",
            "effective_sampling_units",
        ],
        ("$defs", "risk_estimate"): [
            "raw",
            "market_scaled",
            "raw_unit",
            "market_scaled_unit",
        ],
        ("$defs", "vega_kt_report"): [
            "coordinates",
            "estimates",
            "raw_buckets",
            "covariance_layout",
            "projection",
            "residual_diagnostics",
            "raw_unit",
            "market_scaled_unit",
            "policy_label",
            "truncation_order",
        ],
        ("$defs", "vega_kt_report", "allOf", 0, "if"): ["covariance_layout"],
        ("$defs", "vega_kt_report", "allOf", 0, "then"): [
            "full_bucket_covariance"
        ],
        ("$defs", "vega_kt_report", "allOf", 1, "if"): ["covariance_layout"],
        ("$defs", "vega_kt_report", "allOf", 1, "then", "not"): [
            "full_bucket_covariance"
        ],
        ("$defs", "diagnostics"): ["warnings"],
        ("$defs", "warning"): ["code", "message"],
        ("$defs", "estimator", "oneOf", 0): ["type"],
        ("$defs", "estimator", "oneOf", 1): ["type"],
        ("$defs", "estimator", "oneOf", 2): ["type"],
        ("$defs", "risk_unit", "oneOf", 0): ["type"],
        ("$defs", "risk_unit", "oneOf", 1): ["type"],
        ("$defs", "risk_unit", "oneOf", 2): ["type"],
        ("$defs", "risk_unit", "oneOf", 3): ["type"],
        ("$defs", "risk_unit", "oneOf", 4): ["type"],
        ("$defs", "risk_unit", "oneOf", 5): ["type"],
        ("$defs", "vega_kt_covariance_layout", "oneOf", 0): ["type"],
        ("$defs", "vega_kt_covariance_layout", "oneOf", 1): ["type"],
        ("$defs", "vega_kt_bucket_unit", "oneOf", 0): ["type"],
        ("$defs", "vega_kt_bucket_unit", "oneOf", 1): ["type"],
        ("$defs", "vega_kt_coordinate"): [
            "maturity",
            "log_moneyness",
            "implied_volatility",
        ],
        ("$defs", "vega_kt_bucket_estimate"): [
            "raw_mean",
            "market_scaled_mean",
        ],
        ("$defs", "vega_kt_covariance_entry", "oneOf", 0): ["type", "value"],
        ("$defs", "vega_kt_covariance_entry", "oneOf", 1): ["type"],
        ("$defs", "vega_kt_projection"): [
            "scalar_vega",
            "signed_residual",
            "pre_projection",
            "reporting_stats",
        ],
        ("$defs", "vega_kt_residual_diagnostics"): [
            "active_domain_start_index",
            "active_domain_end_index",
            "active_domain_forward_index",
            "excluded_probability_mass",
            "signed_residual",
            "pre_projection",
            "reporting_stats",
        ],
        ("$defs", "reporting_iv_projection_stats"): [
            "left_edge_count",
            "right_edge_count",
            "left_edge_sensitivity",
            "right_edge_sensitivity",
        ],
    },
}
EXPECTED_OPTIONAL_PROPERTIES = {
    "pricing_request": {
        ("$defs", "product", "oneOf", 1): ["payment_date"],
        ("$defs", "product", "oneOf", 2): ["rebate"],
        ("$defs", "product", "oneOf", 4): ["historical_extremum"],
        ("$defs", "market"): ["discrete_dividends"],
        ("$defs", "model", "oneOf", 2): ["reporting_iv_basis"],
        ("$defs", "risk"): [
            "gamma",
            "vega_kt",
            "checkpoint_interval",
            "aad_tile_capacity",
        ],
    },
    "pricing_result": {
        ("$defs", "risk_report"): ["delta", "gamma", "vega", "vega_kt"],
        ("$defs", "vega_kt_report"): ["full_bucket_covariance"],
        ("$defs", "vega_kt_report", "allOf", 0, "if", "properties", "covariance_layout"): [
            "type"
        ],
        ("$defs", "vega_kt_report", "allOf", 1, "if", "properties", "covariance_layout"): [
            "type"
        ],
        ("$defs", "vega_kt_bucket_estimate"): [
            "sample_variance",
            "price_covariance",
        ],
    },
}
WIRE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
DATE_PATTERN = "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
DATE_STRING_FIELDS = {"date", "expiry", "payment_date", "valuation_date"}
DATE_STRING_ARRAY_FIELDS = {"monitoring_dates"}
ID_FIELD_MAXIMUMS = {
    "currency_id": 65_535,
    "curve_id": 4_294_967_295,
    "event_id": 4_294_967_295,
    "underlying_id": 4_294_967_295,
}
REQUEST_INTEGER_LIMITS = {
    "aad_tile_capacity": (1, 4_294_967_295),
    "checkpoint_interval": (1, 4_294_967_295),
    "independent_sampling_units": (1, 18_446_744_073_709_551_615),
    "master_scramble_seed": (0, 18_446_744_073_709_551_615),
    "master_seed": (0, 18_446_744_073_709_551_615),
    "points_per_scramble": (1, 4_294_967_296),
    "scramble_count": (1, 4_294_967_295),
}
RESULT_INTEGER_LIMITS = {
    "active_domain_end_index": (0, 18_446_744_073_709_551_615),
    "active_domain_forward_index": (0, 18_446_744_073_709_551_615),
    "active_domain_start_index": (0, 18_446_744_073_709_551_615),
    "effective_sampling_units": (1, 18_446_744_073_709_551_615),
    "left_edge_count": (0, 18_446_744_073_709_551_615),
    "right_edge_count": (0, 18_446_744_073_709_551_615),
}
SHAPE_DIMENSION_MAXIMUM = 18_446_744_073_709_551_615
OPTIONAL_EMPTY_ARRAY_PATHS = {
    ("$defs", "market", "properties", "discrete_dividends"),
    ("$defs", "diagnostics", "properties", "warnings"),
}
EXPECTED_TAGGED_UNIONS = {
    "pricing_request": {
        ("$defs", "asian_observation_value"): ("known", "unknown"),
        ("$defs", "barrier_direction"): ("up", "down"),
        ("$defs", "barrier_style"): ("knock_in", "knock_out"),
        ("$defs", "digital_payout"): ("cash", "asset"),
        ("$defs", "dividend_quote"): (
            "fixed_cash",
            "proportional",
            "fixed_cash_and_proportional",
        ),
        ("$defs", "engine"): (
            "pseudo_monte_carlo",
            "randomized_quasi_monte_carlo",
        ),
        ("$defs", "model"): (
            "black_scholes",
            "black_76",
            "local_volatility",
        ),
        ("$defs", "product"): (
            "european_vanilla",
            "digital",
            "barrier",
            "arithmetic_asian",
            "fixed_lookback",
        ),
        ("$defs", "side"): ("call", "put"),
        ("$defs", "smile_dynamics"): (
            "sticky_log_moneyness",
            "sticky_strike",
            "sticky_delta",
        ),
        ("$defs", "spot_bump"): ("absolute", "relative"),
    },
    "pricing_result": {
        ("$defs", "estimator"): (
            "analytical",
            "pseudo_monte_carlo",
            "randomized_quasi_monte_carlo",
        ),
        ("$defs", "risk_unit"): (
            "delta_raw",
            "delta_one_percent_spot",
            "gamma_raw",
            "gamma_one_percent_spot_squared",
            "vega_raw",
            "vega_one_vol_point",
        ),
        ("$defs", "vega_kt_bucket_unit"): (
            "currency_per_unit_absolute_volatility",
            "currency_per_volatility_point",
        ),
        ("$defs", "vega_kt_covariance_entry"): ("value", "unavailable"),
        ("$defs", "vega_kt_covariance_layout"): (
            "price_and_bucket_variance_only",
            "full_bucket_matrix_row_major",
        ),
    },
}


class SchemaError(Exception):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def load_schema(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    check_text_file_encoding(raw, path, "schema file")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_json_constant,
        )
    except SchemaError:
        raise
    except Exception as exc:  # noqa: BLE001 - report parser failures as validation failures.
        raise SchemaError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise SchemaError(f"{path}: schema root must be an object")
    return document


def load_golden(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    check_text_file_encoding(raw, path, "golden JSON")
    require(
        b"\n" not in raw[:-1],
        f"{path}: golden JSON must use compact single-line canonical form",
    )
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_json_constant,
        )
    except SchemaError:
        raise
    except Exception as exc:  # noqa: BLE001 - report parser failures as validation failures.
        raise SchemaError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise SchemaError(f"{path}: golden JSON root must be an object")
    compact = json.dumps(document, separators=(",", ":"), ensure_ascii=False) + "\n"
    require(
        raw.decode("utf-8") == compact,
        f"{path}: golden JSON must match canonical compact encoding",
    )
    return document


def check_text_file_encoding(raw: bytes, path: Path, label: str) -> None:
    require(not raw.startswith(b"\xef\xbb\xbf"), f"{path}: UTF-8 BOM is not allowed")
    require(b"\r" not in raw, f"{path}: CR or CRLF line endings are not allowed")
    require(raw.endswith(b"\n"), f"{path}: {label} must end with LF")
    require(not raw.endswith(b"\n\n"), f"{path}: {label} must end with exactly one LF")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{path}: invalid UTF-8: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def reject_json_constant(value: str) -> Any:
    raise SchemaError(f"non-standard JSON constant: {value}")


def pointer(path: tuple[str | int, ...]) -> str:
    if not path:
        return ""
    parts = []
    for part in path:
        token = str(part).replace("~", "~0").replace("/", "~1")
        parts.append(token)
    return "/" + "/".join(parts)


def walk(value: Any, path: tuple[str | int, ...] = ()) -> list[tuple[tuple[str | int, ...], Any]]:
    items = [(path, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            items.extend(walk(child, (*path, key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            items.extend(walk(child, (*path, index)))
    return items


def collect_defs(schema: dict[str, Any]) -> set[str]:
    defs = schema.get("$defs", {})
    require(isinstance(defs, dict), f"{schema.get('$id', '<unknown>')}: $defs must be an object")
    return set(defs)


def check_refs(schema: dict[str, Any], path: Path) -> None:
    defs = collect_defs(schema)
    for location, value in walk(schema):
        if isinstance(value, dict) and "$ref" in value:
            ref = value["$ref"]
            require(isinstance(ref, str), f"{path}:{pointer(location)}: $ref must be a string")
            require(ref.startswith("#/$defs/"), f"{path}:{pointer(location)}: $ref must be local to $defs")
            target = ref.removeprefix("#/$defs/")
            require(target in defs, f"{path}:{pointer(location)}: unresolved $ref {ref}")


def check_no_json_null(schema: dict[str, Any], path: Path) -> None:
    for location, value in walk(schema):
        require(value is not None, f"{path}:{pointer(location)}: JSON null is not part of schema v1")


def check_wire_names(schema: dict[str, Any], path: Path) -> None:
    for location, value in walk(schema):
        if isinstance(value, dict):
            const = value.get("const")
            if isinstance(const, str) and (*location, "const")[-2:] != ("$schema", "const"):
                require(WIRE_NAME.fullmatch(const) is not None, f"{path}:{pointer((*location, 'const'))}: invalid wire name {const!r}")
            properties = value.get("properties")
            if isinstance(properties, dict):
                for field in properties:
                    if field not in {"$schema", "$id"}:
                        require(WIRE_NAME.fullmatch(field) is not None, f"{path}:{pointer((*location, 'properties', field))}: invalid field name")


def check_required_properties(document_kind: str, schema: dict[str, Any], path: Path) -> None:
    actual_required: dict[tuple[str | int, ...], list[str]] = {}
    for location, value in walk(schema):
        if not isinstance(value, dict):
            continue
        required = value.get("required")
        properties = value.get("properties")
        if required is None:
            continue
        require(isinstance(required, list), f"{path}:{pointer((*location, 'required'))}: required must be an array")
        actual_required[location] = required
        require(isinstance(properties, dict), f"{path}:{pointer(location)}: object with required must define properties")
        seen_required: set[str] = set()
        for field in required:
            require(isinstance(field, str), f"{path}:{pointer((*location, 'required'))}: required entry must be a string")
            require(field not in seen_required, f"{path}:{pointer((*location, 'required'))}: duplicate required field {field!r}")
            seen_required.add(field)
            require(field in properties, f"{path}:{pointer((*location, 'required'))}: required field {field!r} missing from properties")
        property_names = list(properties)
        require(
            property_names[: len(required)] == required,
            f"{path}:{pointer(location)}: required fields must lead properties in order",
        )
    require(
        actual_required == EXPECTED_REQUIRED_PROPERTIES[document_kind],
        f"{path}: required field contracts changed",
    )


def check_optional_properties(document_kind: str, schema: dict[str, Any], path: Path) -> None:
    actual_optional: dict[tuple[str | int, ...], list[str]] = {}
    for location, value in walk(schema):
        if not isinstance(value, dict):
            continue
        properties = value.get("properties")
        if not isinstance(properties, dict):
            continue
        required = value.get("required", [])
        require(
            isinstance(required, list),
            f"{path}:{pointer((*location, 'required'))}: required must be an array",
        )
        optional_fields = [field for field in properties if field not in required]
        if optional_fields:
            actual_optional[location] = optional_fields
    require(
        actual_optional == EXPECTED_OPTIONAL_PROPERTIES[document_kind],
        f"{path}: optional field contracts changed",
    )


def check_schema_version_fields(schema: dict[str, Any], path: Path) -> None:
    for location, value in walk(schema):
        if not isinstance(value, dict):
            continue
        properties = value.get("properties")
        if not isinstance(properties, dict):
            continue
        definition = properties.get("schema_version")
        if definition is None:
            continue
        require(
            definition
            == {"type": "integer", "const": 1, "minimum": 1, "maximum": 4_294_967_295},
            f"{path}:{pointer((*location, 'properties', 'schema_version'))}: schema_version must match its Rust integer width",
        )


def check_const_schemas_are_typed(schema: dict[str, Any], path: Path) -> None:
    for location, value in walk(schema):
        if not isinstance(value, dict) or "const" not in value:
            continue
        require(
            "type" in value,
            f"{path}:{pointer(location)}: const schema must declare its JSON type",
        )


def check_date_fields(schema: dict[str, Any], path: Path) -> None:
    for location, value in walk(schema):
        if not isinstance(value, dict):
            continue
        properties = value.get("properties")
        if not isinstance(properties, dict):
            continue
        for field, definition in properties.items():
            if not isinstance(definition, dict):
                continue
            field_location = (*location, "properties", field)
            if field in DATE_STRING_FIELDS:
                require(
                    definition.get("type") == "string"
                    and definition.get("pattern") == DATE_PATTERN,
                    f"{path}:{pointer(field_location)}: date field must use the schema date pattern",
                )
            if field in DATE_STRING_ARRAY_FIELDS:
                items = definition.get("items")
                require(
                    definition.get("type") == "array"
                    and isinstance(items, dict)
                    and items.get("type") == "string"
                    and items.get("pattern") == DATE_PATTERN,
                    f"{path}:{pointer(field_location)}: date array must use the schema date pattern",
                )
            if field == "maturity_nodes" and "vega_kt" in location:
                items = definition.get("items")
                require(
                    definition.get("type") == "array"
                    and isinstance(items, dict)
                    and items.get("type") == "string"
                    and items.get("pattern") == DATE_PATTERN,
                    f"{path}:{pointer(field_location)}: VegaKT maturity nodes must use the schema date pattern",
                )


def check_id_fields(schema: dict[str, Any], path: Path) -> None:
    for location, value in walk(schema):
        if not isinstance(value, dict):
            continue
        properties = value.get("properties")
        if not isinstance(properties, dict):
            continue
        for field, maximum in ID_FIELD_MAXIMUMS.items():
            definition = properties.get(field)
            if definition is None:
                continue
            field_location = (*location, "properties", field)
            require(
                isinstance(definition, dict)
                and definition.get("type") == "integer"
                and definition.get("minimum") == 0
                and definition.get("maximum") == maximum,
                f"{path}:{pointer(field_location)}: {field} must match its Rust integer width",
            )


def check_shape_fields(schema: dict[str, Any], path: Path) -> None:
    for location, value in walk(schema):
        if not isinstance(value, dict):
            continue
        properties = value.get("properties")
        if not isinstance(properties, dict) or "shape" not in properties:
            continue
        definition = properties["shape"]
        items = definition.get("items") if isinstance(definition, dict) else None
        field_location = (*location, "properties", "shape")
        require(
            isinstance(definition, dict)
            and definition.get("type") == "array"
            and definition.get("minItems") == 2
            and definition.get("maxItems") == 2
            and isinstance(items, dict)
            and items.get("type") == "integer"
            and items.get("minimum") == 2
            and items.get("maximum") == SHAPE_DIMENSION_MAXIMUM,
            f"{path}:{pointer(field_location)}: shape must be a two-dimensional bounded integer array",
        )


def check_array_schemas_are_typed_and_sized(schema: dict[str, Any], path: Path) -> None:
    for location, value in walk(schema):
        if not isinstance(value, dict) or value.get("type") != "array":
            continue
        require(
            isinstance(value.get("items"), dict),
            f"{path}:{pointer(location)}: array schema must declare object items",
        )
        min_items = value.get("minItems")
        if location in OPTIONAL_EMPTY_ARRAY_PATHS:
            require(
                min_items is None,
                f"{path}:{pointer(location)}: optional empty array must not declare minItems",
            )
        else:
            require(
                isinstance(min_items, int) and min_items >= 1,
                f"{path}:{pointer(location)}: non-empty array schema must declare positive minItems",
            )
        max_items = value.get("maxItems")
        if max_items is not None:
            require(
                isinstance(max_items, int)
                and isinstance(min_items, int)
                and max_items >= min_items,
                f"{path}:{pointer(location)}: maxItems must be an integer no smaller than minItems",
            )


def check_integer_fields_are_bounded(schema: dict[str, Any], path: Path) -> None:
    for location, value in walk(schema):
        if not isinstance(value, dict) or value.get("type") != "integer":
            continue
        require(
            isinstance(value.get("minimum"), int)
            and isinstance(value.get("maximum"), int)
            and value["minimum"] <= value["maximum"],
            f"{path}:{pointer(location)}: integer schema must declare finite minimum and maximum",
        )


def check_request_integer_limits(schema: dict[str, Any], path: Path) -> None:
    if path != EXPECTED_SCHEMAS["pricing_request"]:
        return
    check_integer_limits(schema, path, REQUEST_INTEGER_LIMITS)


def check_result_integer_limits(schema: dict[str, Any], path: Path) -> None:
    if path != EXPECTED_SCHEMAS["pricing_result"]:
        return
    check_integer_limits(schema, path, RESULT_INTEGER_LIMITS)


def check_integer_limits(
    schema: dict[str, Any],
    path: Path,
    expected_limits: dict[str, tuple[int, int]],
) -> None:
    for location, value in walk(schema):
        if not isinstance(value, dict):
            continue
        properties = value.get("properties")
        if not isinstance(properties, dict):
            continue
        for field, (minimum, maximum) in expected_limits.items():
            definition = properties.get(field)
            if definition is None:
                continue
            field_location = (*location, "properties", field)
            require(
                isinstance(definition, dict)
                and definition.get("type") == "integer"
                and definition.get("minimum") == minimum
                and definition.get("maximum") == maximum,
                f"{path}:{pointer(field_location)}: {field} must match its Rust integer width",
            )


def check_request_digital_payment_date_contract(schema: dict[str, Any], path: Path) -> None:
    if path != EXPECTED_SCHEMAS["pricing_request"]:
        return
    defs = schema.get("$defs", {})
    require(isinstance(defs, dict), f"{path}: $defs must be an object")
    product = defs.get("product")
    require(isinstance(product, dict), f"{path}: product definition must be an object")
    variants = product.get("oneOf")
    require(isinstance(variants, list), f"{path}: product definition must use oneOf")
    digital = None
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        properties = variant.get("properties")
        if not isinstance(properties, dict):
            continue
        discriminator = properties.get("type")
        if isinstance(discriminator, dict) and discriminator.get("const") == "digital":
            digital = variant
            break
    require(digital is not None, f"{path}: product definition must include digital")
    required = digital.get("required")
    properties = digital.get("properties")
    require(isinstance(required, list), f"{path}: digital.required must be an array")
    require(isinstance(properties, dict), f"{path}: digital.properties must be an object")
    require(
        "payment_date" not in required,
        f"{path}: digital payment_date must remain optional for legacy request JSON",
    )
    payment_date = properties.get("payment_date")
    require(
        isinstance(payment_date, dict)
        and payment_date.get("type") == "string"
        and payment_date.get("pattern") == DATE_PATTERN,
        f"{path}: optional digital payment_date must use the schema date pattern",
    )


def check_vega_kt_result_arrays(schema: dict[str, Any], path: Path) -> None:
    defs = schema.get("$defs", {})
    if not isinstance(defs, dict):
        return
    report = defs.get("vega_kt_report")
    if not isinstance(report, dict):
        return
    properties = report.get("properties")
    require(isinstance(properties, dict), f"{path}: vega_kt_report properties must be an object")
    for field in ["coordinates", "estimates", "raw_buckets", "full_bucket_covariance"]:
        definition = properties.get(field)
        require(
            isinstance(definition, dict)
            and definition.get("type") == "array"
            and definition.get("minItems") == 1
            and "items" in definition,
            f"{path}: vega_kt_report.{field} must be a non-empty typed array",
        )
    all_of = report.get("allOf")
    require(isinstance(all_of, list), f"{path}: vega_kt_report must define layout invariants")
    require(
        any(
            requires_vega_kt_full_covariance_for_layout(
                item, "full_bucket_matrix_row_major"
            )
            for item in all_of
            if isinstance(item, dict)
        ),
        f"{path}: full_bucket_matrix_row_major must require full_bucket_covariance",
    )
    require(
        any(
            forbids_vega_kt_full_covariance_for_layout(
                item, "price_and_bucket_variance_only"
            )
            for item in all_of
            if isinstance(item, dict)
        ),
        f"{path}: price_and_bucket_variance_only must forbid full_bucket_covariance",
    )
    check_vega_kt_covariance_entry(defs, path)


def check_vega_kt_covariance_entry(defs: dict[str, Any], path: Path) -> None:
    entry = defs.get("vega_kt_covariance_entry")
    require(isinstance(entry, dict), f"{path}: vega_kt_covariance_entry must be defined")
    one_of = entry.get("oneOf")
    require(
        isinstance(one_of, list) and len(one_of) == 2,
        f"{path}: vega_kt_covariance_entry must have exactly value and unavailable variants",
    )
    variants = {}
    for index, variant in enumerate(one_of):
        require(
            isinstance(variant, dict),
            f"{path}: vega_kt_covariance_entry oneOf[{index}] must be an object schema",
        )
        properties = variant.get("properties")
        require(
            isinstance(properties, dict),
            f"{path}: vega_kt_covariance_entry oneOf[{index}] must define properties",
        )
        discriminator = properties.get("type")
        require(
            isinstance(discriminator, dict) and isinstance(discriminator.get("const"), str),
            f"{path}: vega_kt_covariance_entry oneOf[{index}] must use a string type tag",
        )
        variants[discriminator["const"]] = variant

    require(
        set(variants) == {"value", "unavailable"},
        f"{path}: vega_kt_covariance_entry variants must be value and unavailable",
    )
    value_variant = variants["value"]
    value_properties = value_variant.get("properties")
    require(
        isinstance(value_properties, dict),
        f"{path}: value covariance entry must define properties",
    )
    require(
        value_variant.get("required") == ["type", "value"]
        and value_properties.get("value") == {"type": "number"},
        f"{path}: value covariance entry must require a numeric value",
    )
    unavailable_variant = variants["unavailable"]
    require(
        unavailable_variant.get("required") == ["type"]
        and set(unavailable_variant.get("properties", {})) == {"type"},
        f"{path}: unavailable covariance entry must carry only its type tag",
    )


def requires_vega_kt_full_covariance_for_layout(item: dict[str, Any], layout: str) -> bool:
    then = item.get("then")
    return item_matches_vega_kt_covariance_layout(item.get("if"), layout) and isinstance(
        then, dict
    ) and "full_bucket_covariance" in then.get("required", [])


def forbids_vega_kt_full_covariance_for_layout(item: dict[str, Any], layout: str) -> bool:
    then = item.get("then")
    not_schema = then.get("not") if isinstance(then, dict) else None
    return (
        item_matches_vega_kt_covariance_layout(item.get("if"), layout)
        and isinstance(not_schema, dict)
        and "full_bucket_covariance" in not_schema.get("required", [])
    )


def item_matches_vega_kt_covariance_layout(item: Any, layout: str) -> bool:
    if not isinstance(item, dict):
        return False
    properties = item.get("properties")
    if not isinstance(properties, dict):
        return False
    covariance_layout = properties.get("covariance_layout")
    if not isinstance(covariance_layout, dict):
        return False
    layout_properties = covariance_layout.get("properties")
    if not isinstance(layout_properties, dict):
        return False
    tag = layout_properties.get("type")
    return isinstance(tag, dict) and tag.get("const") == layout


def check_result_replay_metadata(schema: dict[str, Any], path: Path) -> None:
    if path != EXPECTED_SCHEMAS["pricing_result"]:
        return
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{path}: root properties must be an object")
    replay = properties.get("replay")
    require(isinstance(replay, dict), f"{path}: replay must be an object schema")
    replay_properties = replay.get("properties")
    require(isinstance(replay_properties, dict), f"{path}: replay properties must be an object")
    require(
        replay_properties.get("schema_version")
        == {"type": "integer", "const": 1, "minimum": 1, "maximum": 4_294_967_295},
        f"{path}:/properties/replay/properties/schema_version: replay schema_version must match its Rust integer width",
    )
    require(
        replay_properties.get("request_fingerprint")
        == {"type": "string", "pattern": "^blake3-256:[0-9a-f]{64}$"},
        f"{path}:/properties/replay/properties/request_fingerprint: replay request_fingerprint must use the canonical digest pattern",
    )
    for field in ["library_version", "platform"]:
        require(
            replay_properties.get(field) == {"type": "string", "minLength": 1},
            f"{path}:/properties/replay/properties/{field}: replay {field} must be a non-empty string",
        )


def check_no_unstructured_objects(schema: dict[str, Any], path: Path) -> None:
    for location, value in walk(schema):
        if not isinstance(value, dict) or value.get("type") != "object":
            continue
        if "properties" in value or "oneOf" in value:
            continue
        require(
            False,
            f"{path}:{pointer(location)}: object schema must define properties or oneOf",
        )


def check_strict_objects(schema: dict[str, Any], path: Path) -> None:
    for location, value in walk(schema):
        if not isinstance(value, dict) or value.get("type") != "object" or "properties" not in value:
            continue
        if location == ("properties", "diagnostics"):
            continue
        require(
            value.get("additionalProperties") is False or value.get("unevaluatedProperties") is False,
            f"{path}:{pointer(location)}: schema object must reject unknown fields",
        )


def check_tagged_union_discriminators(
    document_kind: str,
    schema: dict[str, Any],
    path: Path,
) -> None:
    actual_union_locations: set[tuple[str | int, ...]] = set()
    for location, value in walk(schema):
        if not isinstance(value, dict):
            continue
        one_of = value.get("oneOf")
        if not isinstance(one_of, list):
            continue
        actual_union_locations.add(location)
        tags: dict[str, int] = {}
        for index, variant in enumerate(one_of):
            variant_path = (*location, "oneOf", index)
            require(
                isinstance(variant, dict),
                f"{path}:{pointer(variant_path)}: union variant must be an object schema",
            )
            require(
                variant.get("type") == "object",
                f"{path}:{pointer(variant_path)}: union variant must be an object",
            )
            required = variant.get("required")
            require(
                isinstance(required, list) and "type" in required,
                f"{path}:{pointer(variant_path)}: union variant must require its type discriminator",
            )
            require(
                required[0] == "type",
                f"{path}:{pointer((*variant_path, 'required'))}: union variant must list its type discriminator first",
            )
            properties = variant.get("properties")
            require(
                isinstance(properties, dict),
                f"{path}:{pointer(variant_path)}: union variant must define properties",
            )
            require(
                next(iter(properties), None) == "type",
                f"{path}:{pointer((*variant_path, 'properties'))}: union variant must declare its type discriminator first",
            )
            discriminator = properties.get("type")
            require(
                isinstance(discriminator, dict),
                f"{path}:{pointer((*variant_path, 'properties', 'type'))}: union discriminator must be an object",
            )
            require(
                discriminator.get("type") == "string",
                f"{path}:{pointer((*variant_path, 'properties', 'type'))}: union discriminator must be a string",
            )
            tag = discriminator.get("const")
            require(
                isinstance(tag, str),
                f"{path}:{pointer((*variant_path, 'properties', 'type', 'const'))}: union discriminator must be a string const",
            )
            previous = tags.get(tag)
            require(
                previous is None,
                f"{path}:{pointer((*location, 'oneOf', index, 'properties', 'type', 'const'))}: "
                f"duplicate union discriminator {tag!r} also appears in oneOf[{previous}]",
            )
            tags[tag] = index
        expected_tags = EXPECTED_TAGGED_UNIONS[document_kind].get(location)
        if expected_tags is not None:
            require(
                tuple(tags) == expected_tags,
                f"{path}:{pointer(location)}: tagged union variant order changed",
            )
    expected_union_locations = set(EXPECTED_TAGGED_UNIONS[document_kind])
    require(
        actual_union_locations == expected_union_locations,
        f"{path}: tagged union locations changed",
    )


def check_top_level(document_kind: str, path: Path, schema: dict[str, Any]) -> None:
    expected_id = f"urn:rust-pricing-library:schema:v1:{document_kind}"
    expected_title = "".join(part.title() for part in document_kind.split("_")) + " schema v1"
    require(list(schema) == EXPECTED_SCHEMA_TOP_LEVEL_KEYS, f"{path}: top-level key order changed")
    require(schema.get("$schema") == DRAFT_2020_12, f"{path}: $schema must be Draft 2020-12")
    require(schema.get("$id") == expected_id, f"{path}: $id must be {expected_id}")
    require(schema.get("title") == expected_title, f"{path}: title must be {expected_title!r}")
    require(schema.get("type") == "object", f"{path}: root type must be object")
    require(schema.get("additionalProperties") is False, f"{path}: root must reject unknown fields")
    require(
        schema.get("required") == EXPECTED_TOP_LEVEL_REQUIRED[document_kind],
        f"{path}: root required fields changed",
    )
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{path}: root properties must be an object")
    require(
        set(properties) == set(EXPECTED_TOP_LEVEL_REQUIRED[document_kind]),
        f"{path}: root properties changed",
    )
    require(
        properties.get("document_kind") == {"type": "string", "const": document_kind},
        f"{path}: document_kind const mismatch",
    )
    require(
        properties.get("schema_version")
        == {"type": "integer", "const": 1, "minimum": 1, "maximum": 4_294_967_295},
        f"{path}: schema_version const mismatch",
    )
    defs = schema.get("$defs")
    require(isinstance(defs, dict), f"{path}: $defs must be an object")
    require(
        list(defs) == EXPECTED_SCHEMA_DEF_ORDER[document_kind],
        f"{path}: $defs order changed",
    )
    require(
        set(defs) == EXPECTED_SCHEMA_DEFS[document_kind],
        f"{path}: $defs names changed",
    )


def check_golden(document_kind: str, path: Path) -> None:
    require(path.exists(), f"missing golden JSON {path}")
    document = load_golden(path)
    require(
        set(document) == EXPECTED_GOLDEN_KEYS[document_kind],
        f"{path}: top-level golden fields changed",
    )
    require(
        document.get("document_kind") == document_kind,
        f"{path}: document_kind must be {document_kind!r}",
    )
    require(document.get("schema_version") == 1, f"{path}: schema_version must be 1")
    for location, value in walk(document):
        require(value is not None, f"{path}:{pointer(location)}: JSON null is not permitted")
    if document_kind == "pricing_result":
        replay = document.get("replay")
        require(isinstance(replay, dict), f"{path}: replay must be an object")
        require(replay.get("schema_version") == 1, f"{path}: replay.schema_version must be 1")
        require(
            isinstance(replay.get("request_fingerprint"), str)
            and re.fullmatch(r"blake3-256:[0-9a-f]{64}", replay["request_fingerprint"])
            is not None,
            f"{path}: replay.request_fingerprint must be canonical blake3-256 hex",
        )
        for field in ["library_version", "platform"]:
            require(
                isinstance(replay.get(field), str) and bool(replay[field]),
                f"{path}: replay.{field} must be a non-empty string",
            )


def check_schema(document_kind: str, path: Path) -> None:
    require(path.exists(), f"missing schema {path}")
    schema = load_schema(path)
    check_top_level(document_kind, path, schema)
    check_no_json_null(schema, path)
    check_refs(schema, path)
    check_wire_names(schema, path)
    check_required_properties(document_kind, schema, path)
    check_optional_properties(document_kind, schema, path)
    check_schema_version_fields(schema, path)
    check_const_schemas_are_typed(schema, path)
    check_date_fields(schema, path)
    check_id_fields(schema, path)
    check_shape_fields(schema, path)
    check_array_schemas_are_typed_and_sized(schema, path)
    check_integer_fields_are_bounded(schema, path)
    check_request_integer_limits(schema, path)
    check_result_integer_limits(schema, path)
    check_request_digital_payment_date_contract(schema, path)
    check_vega_kt_result_arrays(schema, path)
    check_result_replay_metadata(schema, path)
    check_no_unstructured_objects(schema, path)
    check_strict_objects(schema, path)
    check_tagged_union_discriminators(document_kind, schema, path)


def main() -> int:
    try:
        actual = set(SCHEMA_ROOT.glob("*.schema.json"))
        expected = set(EXPECTED_SCHEMAS.values())
        require(actual == expected, f"schema file set mismatch: expected {sorted(map(str, expected))}, got {sorted(map(str, actual))}")
        actual_goldens = set(GOLDEN_ROOT.glob("*.golden.json"))
        expected_goldens = set(EXPECTED_GOLDENS.values())
        require(
            actual_goldens == expected_goldens,
            f"golden JSON file set mismatch: expected {sorted(map(str, expected_goldens))}, got {sorted(map(str, actual_goldens))}",
        )
        for document_kind, path in EXPECTED_SCHEMAS.items():
            check_schema(document_kind, path)
        for document_kind, path in EXPECTED_GOLDENS.items():
            check_golden(document_kind, path)
    except SchemaError as exc:
        print(f"schema check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
