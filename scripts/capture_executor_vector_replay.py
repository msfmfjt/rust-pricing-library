"""Capture complete installed-wheel results for a byte-exact before/after comparison.

Run with each wheel installed in a separate Python environment, then compare the
two output files with cmp. This supplements the numerical acceptance tests; it
does not replace their independent oracles or alter their sampling/tolerances.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[1]


def execution_source(task: dict) -> dict:
    source = task["source"]
    if source["type"] == "single_asset":
        return source["specification"]
    if source["type"] == "stochastic_dividend":
        return source["equity"]
    return source


def task_cases():
    sources = {
        path.stem: json.loads(path.read_text())
        for path in sorted((ROOT / "fixtures/task-v1").glob("*.golden.json"))
    }
    for name in ("bergomi_hull_white", "bergomi_lsv", "rough_heston"):
        source = rp.PricingSpecification.from_json(
            (ROOT / f"fixtures/specification-v1/{name}.golden.json").read_text()
        )
        sources[name] = json.loads(rp.PricingTask.from_specification(source).to_json())
    for name, original in sources.items():
        for variant in ("base", "workers_3", "block_7", "antithetic_off", "bridge_off", "bridge_on"):
            task = copy.deepcopy(original)
            source = execution_source(task)
            if variant == "workers_3":
                source["numerics"]["worker_threads"] = 3
            elif variant == "block_7":
                source["numerics"]["reduction_block_size"] = 7
            elif variant == "antithetic_off":
                source["engine"]["variance_reduction"]["antithetic"] = False
            elif variant.startswith("bridge_"):
                source["engine"]["variance_reduction"]["brownian_bridge"] = variant == "bridge_on"
            typed = rp.PricingTask.from_json(json.dumps(task, allow_nan=False))
            yield {
                "id": f"{name}/{variant}",
                "task_json": typed.to_json(),
                "report_json": typed.compile().evaluate().report.to_json(),
            }


def covariance_cases():
    # Reuse the unchanged public test's request and complete sampling matrix.
    sys.path.insert(0, str(ROOT / "tests/python"))
    from test_rough_dividend_continuous import FIXTURE, compile_plan, reporting_request

    properties = {
        node.name: [
            member.name for member in node.body
            if isinstance(member, ast.FunctionDef)
            and any(isinstance(d, ast.Name) and d.id == "property" for d in member.decorator_list)
        ]
        for node in ast.parse((ROOT / "rust_pricing.pyi").read_text()).body
        if isinstance(node, ast.ClassDef)
    }

    def snapshot(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, (list, tuple)):
            return [snapshot(item) for item in value]
        fields = properties[type(value).__name__]
        if not fields:
            raise ValueError(f"no public result fields: {type(value).__name__}")
        return {name: snapshot(getattr(value, name)) for name in fields}

    case = FIXTURE["cases"][0]
    for qmc in (False, True):
        for antithetic in (False, True):
            data = json.loads(reporting_request(case, points=16).to_json())
            if not qmc:
                data["engine"] = dict(type="pseudo_monte_carlo", independent_sampling_units=64, master_seed=193)
            data["engine"]["variance_reduction"] = dict(antithetic=antithetic, brownian_bridge=True)
            request = rp.PricingRequest.from_json(json.dumps(data))
            for workers in (1, 3):
                plan = compile_plan(case, request, reduction_block_size=8, worker_threads=workers)
                for full_covariance in (False, True):
                    result = plan.evaluate_reporting_iv_projection(
                        local_volatility_bump=.01,
                        relative_density_threshold=1e-8,
                        full_covariance=full_covariance,
                    )
                    yield {
                        "id": f"continuous/qmc_{qmc}/antithetic_{antithetic}/workers_{workers}/covariance_{full_covariance}",
                        "request_json": request.to_json(),
                        "result": snapshot(result),
                    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = list(task_cases()) + list(covariance_cases())
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("duplicate replay case")
    data = (json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.write_bytes(data)
    print(f"{len(records)} complete results: sha256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
