"""Offline JSON Schema validation of source documents and composition boundaries."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def main():
    schema = json.loads((ROOT / "schemas/specification-v1/pricing_specification.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    fixtures = {p.stem.removesuffix(".golden"): json.loads(p.read_text())
                for p in (ROOT / "fixtures/specification-v1").glob("*.golden.json")}
    assert set(fixtures) == {"black_scholes", "bergomi_hull_white", "bergomi_lsv", "rough_heston"}
    for name, document in fixtures.items():
        validator.validate(document)
    valid = copy.deepcopy(fixtures["bergomi_hull_white"])
    valid["model"]["rates"] = {"type": "deterministic"}
    validator.validate(valid)
    valid = copy.deepcopy(fixtures["bergomi_hull_white"])
    valid["model"]["equity"]["factor"] = {
        "type": "bergomi_two_factor", "mean_reversions": [0.5, 2.0], "vol_of_vol": 0.3,
        "mixing_weight": 0.4, "spot_correlations": [-0.3, -0.2], "factor_correlation": 0.1,
    }
    valid["model"]["rates"]["vol_rate_correlations"] = [0.0, 0.0]
    validator.validate(valid)
    valid = copy.deepcopy(fixtures["black_scholes"])
    valid["model"]["rates"] = copy.deepcopy(fixtures["bergomi_hull_white"]["model"]["rates"])
    valid["model"]["rates"]["vol_rate_correlations"] = []
    valid["numerics"]["maximum_step"] = 0.25
    validator.validate(valid)
    valid = copy.deepcopy(fixtures["rough_heston"])
    valid["model"]["equity"]["model"] = {
        "type": "rough_sabr", "hurst": 0.2, "vol_of_vol": 0.3, "correlation": -0.4,
        "beta": 1.0, "forward_variance": {"type": "exponential", "initial": 0.04, "growth": 0.1},
    }
    validator.validate(valid)
    valid = copy.deepcopy(fixtures["bergomi_lsv"])
    valid["model"]["rates"] = copy.deepcopy(fixtures["bergomi_hull_white"]["model"]["rates"])
    valid["model"]["equity"]["forward_density"] = {
        "local_variance_grid": copy.deepcopy(valid["model"]["equity"]["target"]["local_variance_grid"]),
        "log_densities": [0.0] * 15,
    }
    validator.validate(valid)

    invalid = []

    def changed(name, path, value):
        document = copy.deepcopy(fixtures[name])
        parent = document
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
        invalid.append((f"{name}:{'.'.join(path)}", document))

    for name in fixtures:
        changed(name, ["schema_version"], 2)
        changed(name, ["document_kind"], "pricing_request")
        changed(name, ["unknown"], True)
        changed(name, ["numerics", "worker_threads"], 0)
        changed(name, ["numerics", "reduction_block_size"], None)
        changed(name, ["model", "equity", "unknown"], True)
    changed("black_scholes", ["model", "rates", "unknown"], True)
    changed("black_scholes", ["numerics", "maximum_step"], 0.25)
    changed("bergomi_lsv", ["numerics", "maximum_step"], 0.25)
    changed("bergomi_lsv", ["model", "rates"], fixtures["bergomi_hull_white"]["model"]["rates"])
    changed("bergomi_lsv", ["model", "equity", "particles", "particle_count"], 1)
    changed("bergomi_lsv", ["model", "equity", "particles", "minimum_effective_samples"], 0.5)
    changed("bergomi_lsv", ["model", "equity", "forward_density"], valid["model"]["equity"]["forward_density"])
    changed("rough_heston", ["model", "rates"], fixtures["bergomi_hull_white"]["model"]["rates"])
    changed("rough_heston", ["model", "equity", "model", "hurst"], 0.0)
    changed("bergomi_hull_white", ["model", "rates", "vol_rate_correlations"], [])
    changed("bergomi_hull_white", ["model", "rates", "vol_rate_correlations"], [0.0, 0.0])
    changed("bergomi_hull_white", ["numerics", "maximum_step"], 0.0)
    for name in ["bergomi_hull_white", "bergomi_lsv", "rough_heston"]:
        for field, value in [("delta", True), ("vega", True),
                             ("checkpoint_interval", 4), ("aad_tile_capacity", 4)]:
            changed(name, ["risk", field], value)
    for name in ["bergomi_hull_white", "rough_heston"]:
        document = copy.deepcopy(fixtures[name])
        del document["numerics"]["maximum_step"]
        invalid.append((name + ":missing step", document))
    for label, document in invalid:
        assert not validator.is_valid(document), f"schema accepted invalid source: {label}"
    print(f"Specification schema: 4 fixtures, 5 valid compositions, {len(invalid)} rejection cases passed")


if __name__ == "__main__":
    main()
