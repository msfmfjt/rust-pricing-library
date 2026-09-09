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


def check_required_properties(schema: dict[str, Any], path: Path) -> None:
    for location, value in walk(schema):
        if not isinstance(value, dict):
            continue
        required = value.get("required")
        properties = value.get("properties")
        if required is None:
            continue
        require(isinstance(required, list), f"{path}:{pointer((*location, 'required'))}: required must be an array")
        require(isinstance(properties, dict), f"{path}:{pointer(location)}: object with required must define properties")
        seen_required: set[str] = set()
        for field in required:
            require(isinstance(field, str), f"{path}:{pointer((*location, 'required'))}: required entry must be a string")
            require(field not in seen_required, f"{path}:{pointer((*location, 'required'))}: duplicate required field {field!r}")
            seen_required.add(field)
            require(field in properties, f"{path}:{pointer((*location, 'required'))}: required field {field!r} missing from properties")


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
            definition == {"const": 1},
            f"{path}:{pointer((*location, 'properties', 'schema_version'))}: schema_version must be const 1",
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
            and items.get("minimum") == 2,
            f"{path}:{pointer(field_location)}: shape must be a two-dimensional integer array",
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
        replay_properties.get("schema_version") == {"const": 1},
        f"{path}:/properties/replay/properties/schema_version: replay schema_version must be const 1",
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


def check_tagged_union_discriminators(schema: dict[str, Any], path: Path) -> None:
    for location, value in walk(schema):
        if not isinstance(value, dict):
            continue
        one_of = value.get("oneOf")
        if not isinstance(one_of, list):
            continue
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
            properties = variant.get("properties")
            require(
                isinstance(properties, dict),
                f"{path}:{pointer(variant_path)}: union variant must define properties",
            )
            discriminator = properties.get("type")
            require(
                isinstance(discriminator, dict),
                f"{path}:{pointer((*variant_path, 'properties', 'type'))}: union discriminator must be an object",
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


def check_top_level(document_kind: str, path: Path, schema: dict[str, Any]) -> None:
    expected_id = f"urn:rust-pricing-library:schema:v1:{document_kind}"
    require(schema.get("$schema") == DRAFT_2020_12, f"{path}: $schema must be Draft 2020-12")
    require(schema.get("$id") == expected_id, f"{path}: $id must be {expected_id}")
    require(schema.get("type") == "object", f"{path}: root type must be object")
    require(schema.get("additionalProperties") is False, f"{path}: root must reject unknown fields")
    properties = schema.get("properties")
    require(isinstance(properties, dict), f"{path}: root properties must be an object")
    require(properties.get("document_kind") == {"const": document_kind}, f"{path}: document_kind const mismatch")
    require(properties.get("schema_version") == {"const": 1}, f"{path}: schema_version const mismatch")


def check_golden(document_kind: str, path: Path) -> None:
    require(path.exists(), f"missing golden JSON {path}")
    document = load_golden(path)
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
    check_required_properties(schema, path)
    check_schema_version_fields(schema, path)
    check_date_fields(schema, path)
    check_id_fields(schema, path)
    check_shape_fields(schema, path)
    check_request_digital_payment_date_contract(schema, path)
    check_vega_kt_result_arrays(schema, path)
    check_result_replay_metadata(schema, path)
    check_no_unstructured_objects(schema, path)
    check_strict_objects(schema, path)
    check_tagged_union_discriminators(schema, path)


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
