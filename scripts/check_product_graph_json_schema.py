"""Offline schema checks for graph products and specification v2 compatibility."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]


def validator(path):
    schema = json.loads((ROOT / path).read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def main():
    graph_schema = validator("schemas/product-graph-v1/product_graph.schema.json")
    source_schema = validator("schemas/specification-v2/pricing_specification.schema.json")
    old_source = validator("schemas/specification-v1/pricing_specification.schema.json")
    graph = json.loads((ROOT / "fixtures/product-graph-v1/coupon_and_capped_call.golden.json").read_text())
    source = json.loads((ROOT / "fixtures/specification-v2/product_graph.golden.json").read_text())
    graph_schema.validate(graph)
    source_schema.validate(source)
    for fixture in (ROOT / "fixtures/specification-v1").glob("*.golden.json"):
        value = json.loads(fixture.read_text())
        old_source.validate(value)
        value["schema_version"] = 2
        source_schema.validate(value)
    old_graph = copy.deepcopy(source)
    old_graph["schema_version"] = 1
    assert not old_source.is_valid(old_graph), "v1 must not accept graph products"

    rejected = 0

    def reject(schema, original, path, value):
        nonlocal rejected
        changed = copy.deepcopy(original)
        parent = changed
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
        assert not schema.is_valid(changed), f"schema accepted {path}: {value}"
        rejected += 1

    for path, value in [
        (["unknown"], 1), (["schema_version"], 2), (["document_kind"], "pricing_request"),
        (["product", "currency_id"], 65536), (["product", "underlying_ids"], []),
        (["product", "underlying_ids"], [1, 1]), (["product", "nodes"], []),
        (["product", "nodes", 0, "id"], -1),
        (["product", "nodes", 0, "operation", "underlying_id"], 2**32),
        (["product", "nodes", 0, "operation", "observation_date"], "2027-02-30"),
        (["product", "nodes", 0, "operation", "unknown"], 1),
        (["product", "nodes", 0, "operation", "type"], "callback"),
        (["product", "cashflows"], []),
        (["product", "cashflows"], graph["product"]["cashflows"] * 33),
        (["product", "cashflows", 0, "payment_date"], None),
        (["product", "cashflows", 0, "output_node"], 1.5),
    ]:
        reject(graph_schema, graph, path, value)
    for path, value in [
        (["schema_version"], 1), (["schema_version"], 3),
        (["product", "graph", "underlying_ids"], [1, 2]),
        (["product", "graph", "unknown"], 1),
        (["risk", "payoff_smoothing"], {"type": "compact_c2", "half_width": .1}),
        (["risk", "payoff_smoothing_width_ladder"], [1., .5]),
        (["lsm"], {}),
    ]:
        reject(source_schema, source, path, value)
    # Each supported operation is checked independently of Rust graph semantics.
    operations = graph_schema.schema["$defs"]["graph_operation"]["oneOf"]
    for shape in operations:
        operation = {}
        for field, spec in shape["properties"].items():
            operation[field] = spec.get("const", "2027-03-04" if field == "observation_date"
                                        else .1 if field == "half_width" else 0)
        changed = copy.deepcopy(graph)
        changed["product"]["nodes"][0]["operation"] = operation
        graph_schema.validate(changed)
        reject(graph_schema, changed, ["product", "nodes", 0, "operation", "extra"], True)
    print(f"Graph schemas: 2 fixtures, 4 legacy upgrades, {len(operations)} opcodes, {rejected} rejections passed")


if __name__ == "__main__":
    main()
