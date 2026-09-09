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
DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
EXPECTED_SCHEMAS = {
    "pricing_request": SCHEMA_ROOT / "pricing_request.schema.json",
    "pricing_result": SCHEMA_ROOT / "pricing_result.schema.json",
}
WIRE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


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
    require(not raw.startswith(b"\xef\xbb\xbf"), f"{path}: UTF-8 BOM is not allowed")
    require(b"\r" not in raw, f"{path}: CR or CRLF line endings are not allowed")
    require(raw.endswith(b"\n"), f"{path}: schema file must end with LF")
    require(not raw.endswith(b"\n\n"), f"{path}: schema file must end with exactly one LF")
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
        for field in required:
            require(isinstance(field, str), f"{path}:{pointer((*location, 'required'))}: required entry must be a string")
            require(field in properties, f"{path}:{pointer((*location, 'required'))}: required field {field!r} missing from properties")


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


def check_schema(document_kind: str, path: Path) -> None:
    require(path.exists(), f"missing schema {path}")
    schema = load_schema(path)
    check_top_level(document_kind, path, schema)
    check_no_json_null(schema, path)
    check_refs(schema, path)
    check_wire_names(schema, path)
    check_required_properties(schema, path)
    check_strict_objects(schema, path)


def main() -> int:
    try:
        actual = set(SCHEMA_ROOT.glob("*.schema.json"))
        expected = set(EXPECTED_SCHEMAS.values())
        require(actual == expected, f"schema file set mismatch: expected {sorted(map(str, expected))}, got {sorted(map(str, actual))}")
        for document_kind, path in EXPECTED_SCHEMAS.items():
            check_schema(document_kind, path)
    except SchemaError as exc:
        print(f"schema check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
