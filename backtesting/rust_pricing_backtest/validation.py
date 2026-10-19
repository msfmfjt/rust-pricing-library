"""Offline structural validation of current backtest wire documents.

Install the optional validation extra. This does not price, replay, check hashes,
follow result paths, fetch schemas, or certify the financial calculations.
"""

import argparse
from datetime import date, datetime
from functools import lru_cache
from importlib.resources import files
import json
import math
from pathlib import Path

from .domain import BacktestError, canonical, fingerprint, require


VERSIONS = {
    "hedging_backtest_request": {5: "request-v5", 6: "request-v6", 7: "request-v7", 8: "request-v8", 9: "request-v9"},
    "hedging_backtest_result": {9: "result-v9", 10: "result-v10", 11: "result-v11", 12: "result-v12", 13: "result-v13"},
    "hedging_backtest_checkpoint": {7: "checkpoint-v7", 8: "checkpoint-v8", 9: "checkpoint-v9", 10: "checkpoint-v10", 11: "checkpoint-v11"},
    "hedging_backtest_ensemble_request": {1: "ensemble-request-v1"},
    "hedging_backtest_ensemble": {1: "ensemble-v1", 2: "ensemble-v2", 3: "ensemble-v3", 4: "ensemble-v4", 5: "ensemble-v5"},
}
SCHEMA_FILES = ("definitions-v1.schema.json", "definitions-v2.schema.json", "definitions-v3.schema.json", "definitions-v4.schema.json", "definitions-v5.schema.json",
                *(f"{name}.schema.json" for versions in VERSIONS.values() for name in versions.values()))


def _pairs(items):
    value = {}
    for key, item in items:
        require(key not in value, "invalid_json", f"duplicate object key: {key}")
        value[key] = item
    return value


def loads_document(payload: str):
    """Reject duplicate keys, nonstandard constants and overflowed JSON numbers."""
    try:
        value = json.loads(payload, object_pairs_hook=_pairs,
                           parse_constant=lambda x: require(False, "invalid_json", f"nonfinite number: {x}"))
    except (ValueError, RecursionError) as error:
        if isinstance(error, BacktestError):
            raise
        raise BacktestError("invalid_json", str(error)[:250]) from error
    _check_json(value)
    return value


def _check_json(value, path=""):
    if isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), "invalid_json", f"{path}: object keys must be strings")
        for key, item in value.items():
            _check_json(item, f"{path}/{key.replace('~', '~0').replace('/', '~1')}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _check_json(item, f"{path}/{i}")
    else:
        require(value is None or type(value) in (str, bool, int, float), "invalid_json", f"{path}: JSON value required")
        require(not isinstance(value, float) or math.isfinite(value), "invalid_json", f"{path}: nonfinite number")


def schema_documents():
    """Return bundled current/historical schemas; no network or checkout needed."""
    directory = files("rust_pricing_backtest").joinpath("schemas")
    return {name: json.loads(directory.joinpath(name).read_text(encoding="utf-8")) for name in SCHEMA_FILES}


@lru_cache(maxsize=1)
def _validators():
    try:
        from jsonschema import Draft202012Validator, FormatChecker
        from referencing import Registry, Resource
        from referencing.exceptions import NoSuchResource
    except ImportError as error:
        raise BacktestError("missing_validation_dependency",
                            "install rust-pricing-backtest[validation] to run schema validation") from error

    def no_retrieval(uri):
        raise NoSuchResource(ref=uri)

    checker = FormatChecker()

    @checker.checks("date", raises=ValueError)
    def calendar_date(value):
        return not isinstance(value, str) or date.fromisoformat(value).isoformat() == value

    @checker.checks("date-time", raises=ValueError)
    def aware_timestamp(value):
        return not isinstance(value, str) or datetime.fromisoformat(value).utcoffset() is not None

    documents = schema_documents()
    registry = Registry(retrieve=no_retrieval)
    for document in documents.values():
        Draft202012Validator.check_schema(document)
        registry = registry.with_resource(document["$id"], Resource.from_contents(document))
    validators = {name: Draft202012Validator(documents[f"{name}.schema.json"], registry=registry, format_checker=checker)
                  for versions in VERSIONS.values() for name in versions.values()}
    return validators, fingerprint(documents)


def _embedded_json(history, path):
    # Only the documented snapshot field contains native request strings.
    # Arbitrary provider metadata may legitimately use the same key name.
    for i, snapshot in enumerate(history):
        for contract, payload in snapshot.get("pricing_requests", {}).items():
            try:
                embedded = loads_document(payload)
                require(isinstance(embedded, dict), "invalid_json", "native request must be a JSON object")
            except BacktestError as error:
                raise BacktestError("invalid_embedded_json", f"{path}/{i}/pricing_requests/{contract}: {error.message}") from error


def validate_document(document) -> dict:
    """Check current structural schemas and embedded JSON syntax, not semantics."""
    _check_json(document)
    require(isinstance(document, dict), "schema_fields", "document must be an object")
    header = document.get("manifest", document)
    require(isinstance(header, dict), "schema_fields", "/manifest: object required")
    kind = header.get("document_kind")
    if kind is None and "state" in document and "identity_hash" in document:
        kind = "hedging_backtest_checkpoint"
    require(kind in VERSIONS, "schema_version", "unsupported document kind")
    version = header.get("schema_version")
    require(type(version) is int and version in VERSIONS[kind],
            "schema_version", f"{kind}: supported schema versions are {list(VERSIONS[kind])}; older readers are separate")
    name = VERSIONS[kind][version]
    validators, bundle_hash = _validators()
    validator = validators[name]
    error = next(validator.iter_errors(document), None)
    if error is not None:
        path = "/" + "/".join(str(x).replace("~", "~0").replace("/", "~1") for x in error.absolute_path)
        raise BacktestError("schema_fields", f"{path}: {error.validator}: {error.message[:250]}")
    if kind == "hedging_backtest_request":
        _embedded_json(document["market_history"], "/market_history")
    elif kind == "hedging_backtest_result":
        _embedded_json(header["market_history"], "/manifest/market_history")
    return {"document_kind": kind, "schema_version": version, "schema_id": validator.schema["$id"],
            "schema_bundle_hash": bundle_hash, "risk_validation": header.get("risk_validation"),
            "status": "valid", "validation_scope": "structure_and_embedded_json_syntax",
            "semantic_checks": "not_run", "financial_acceptance": "not_evaluated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="*", type=Path)
    parser.add_argument("--export-schemas", type=Path, help="copy the offline schema bundle for external validators")
    args = parser.parse_args()
    if not args.documents and args.export_schemas is None:
        parser.error("provide JSON document paths or --export-schemas")
    if args.export_schemas is not None:
        args.export_schemas.mkdir(parents=True, exist_ok=True)
        for name, schema in schema_documents().items():
            (args.export_schemas / name).write_text(json.dumps(schema, indent=2)+"\n", encoding="utf-8")
    failed = False
    for path in args.documents:
        try:
            report = validate_document(loads_document(path.read_text(encoding="utf-8")))
        except (BacktestError, OSError, RecursionError) as error:
            report = {"status": "invalid", "code": getattr(error, "code", "input_error"), "message": str(error)[:500]}
            failed = True
        print(canonical({"path": str(path), **report}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
