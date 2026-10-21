"""Generate the independent, self-contained pricing_specification v1 schema."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "schemas/specification-v1/pricing_specification.schema.json"


def obj(properties, optional=()):
    return dict(type="object", additionalProperties=False,
                required=[name for name in properties if name not in optional], properties=properties)


def tag(name, properties=None, optional=()):
    return obj({"type": {"type": "string", "const": name}, **(properties or {})}, optional)


def ref(name):
    return {"$ref": f"#/$defs/{name}"}


def array(items, minimum=1, maximum=None):
    result = dict(type="array", items=items, minItems=minimum)
    if maximum is not None:
        result["maxItems"] = maximum
    return result


def build_schema():
    schema = json.loads((ROOT / "schemas/v3/pricing_request.schema.json").read_text())
    number = {"type": "number"}
    nonnegative = {"type": "number", "minimum": 0}
    positive = {"type": "number", "exclusiveMinimum": 0}
    rho = {"type": "number", "minimum": -1, "maximum": 1}
    unit = {"type": "number", "minimum": 0, "maximum": 1}
    hurst = {"type": "number", "exclusiveMinimum": 0, "maximum": 0.5}
    integer = lambda minimum, maximum: dict(type="integer", minimum=minimum, maximum=maximum)
    u64 = integer(0, 2**64 - 1)
    definitions = schema["$defs"]
    definitions["volatility_factor"] = {"oneOf": [
        tag("bergomi", dict(mean_reversion=nonnegative, vol_of_vol=nonnegative, correlation=rho)),
        tag("bergomi_two_factor", dict(mean_reversions=array(nonnegative, 2, 2), vol_of_vol=nonnegative,
            mixing_weight=unit, spot_correlations=array(rho, 2, 2), factor_correlation=rho)),
        tag("rough_bergomi", dict(hurst=hurst, vol_of_vol=nonnegative, correlation=rho)),
    ]}
    definitions["forward_variance"] = {"oneOf": [
        tag("knots", dict(times=array(nonnegative), values=array(nonnegative))),
        tag("exponential", dict(initial=nonnegative, growth=number)),
    ]}
    heston = dict(initial_variance=nonnegative, mean_reversion=nonnegative,
                  long_run_variance=nonnegative, vol_of_vol=nonnegative, correlation=rho)
    definitions["rough_model"] = {"oneOf": [
        tag("rough_heston", dict(hurst=hurst, **heston)),
        tag("lifted_heston", dict(**heston, weights=array(positive, 1, 256), rates=array(nonnegative, 1, 256))),
        tag("quadratic_rough_heston", dict(hurst=hurst, initial_state=number, mean_reversion=nonnegative,
            vol_of_vol=nonnegative, quadratic=nonnegative, shift=number, variance_floor=nonnegative)),
        tag("mixed_rough_bergomi", dict(hurst=hurst, correlation=rho, weights=array(nonnegative, 1, 256),
            vol_of_vols=array(nonnegative, 1, 256), forward_variance=ref("forward_variance"))),
        tag("rough_sabr", dict(hurst=hurst, vol_of_vol=nonnegative, correlation=rho,
            beta=unit, forward_variance=ref("forward_variance"))),
        tag("rfsv", dict(hurst=hurst, mean_reversion=positive, vol_of_log_vol=nonnegative,
            mean_log_vol=number, initial_log_vol=number), optional=("initial_log_vol",)),
    ]}
    definitions["local_target"] = obj(dict(local_variance_grid=ref("local_variance_grid"),
        reporting_iv_basis=ref("reporting_iv_basis")), optional=("reporting_iv_basis",))
    definitions["particles"] = obj(dict(particle_count=integer(2, 2**64 - 1), seed=u64,
        log_bandwidth=positive, minimum_effective_samples={"type": "number", "minimum": 1},
        retain_reverse_trace={"type": "boolean"}))
    definitions["market_iv_source"] = obj(dict(maturity_nodes=array(positive),
        log_moneyness_nodes=array(number), implied_volatilities=array(positive)))
    definitions["forward_density"] = obj(dict(local_variance_grid=ref("local_variance_grid"),
        log_densities=array(nonnegative), market_iv=ref("market_iv_source")), optional=("market_iv",))
    definitions["equity_model"] = {"oneOf": [
        tag("standard", dict(model=ref("model"))),
        tag("stochastic_volatility", dict(initial_volatility=nonnegative, factor=ref("volatility_factor"))),
        tag("local_stochastic_volatility", dict(target=ref("local_target"), factor=ref("volatility_factor"),
            particles=ref("particles"), forward_density=ref("forward_density")), optional=("forward_density",)),
        tag("rough_volatility", dict(model=ref("rough_model"))),
        tag("rough_local_stochastic_volatility", dict(target=ref("local_target"),
            model=ref("rough_model"), particles=ref("particles"))),
    ]}
    definitions["rate_model"] = {"oneOf": [
        tag("deterministic"),
        tag("hull_white", dict(mean_reversion=nonnegative, volatility_times=array(nonnegative),
            volatilities=array(nonnegative), equity_rate_correlation=rho, vol_rate_correlations=array(rho, 0, 2))),
    ]}
    composition = obj(dict(equity=ref("equity_model"), rates=ref("rate_model")))
    equity_is = lambda *names: {"properties": {"equity": {"properties": {"type": {"enum": list(names)}}}}}
    hybrid = {"properties": {"rates": {"properties": {"type": {"const": "hull_white"}}}}}
    composition["allOf"] = [
        {"if": hybrid, "then": {"allOf": [
            equity_is("standard", "stochastic_volatility", "local_stochastic_volatility"),
            {"if": equity_is("standard"), "then": {"properties": {
                "equity": {"properties": {"model": {"properties": {"type": {"const": "black_scholes"}}}}},
                "rates": {"properties": {"vol_rate_correlations": {"maxItems": 0}}},
            }}, "else": {"if": {"properties": {"equity": {"properties": {"factor": {"properties": {"type": {"const": "bergomi_two_factor"}}}}}}},
                "then": {"properties": {"rates": {"properties": {"vol_rate_correlations": {"minItems": 2, "maxItems": 2}}}}},
                "else": {"properties": {"rates": {"properties": {"vol_rate_correlations": {"minItems": 1, "maxItems": 1}}}}}}},
            {"if": equity_is("local_stochastic_volatility"),
             "then": {"properties": {"equity": {"required": ["forward_density"]}}}},
        ]}, "else": {"properties": {"equity": {"not": {"required": ["forward_density"]}}}}},
    ]
    definitions["model_composition"] = composition
    definitions["numerics"] = obj(dict(worker_threads=integer(1, 2**32 - 1),
        reduction_block_size=integer(1, 2**64 - 1), maximum_step=positive), optional=("maximum_step",))
    schema["$id"] = "urn:rust-pricing-library:schema:specification-v1:pricing_specification"
    schema["title"] = "PricingSpecification schema v1"
    schema["properties"]["document_kind"]["const"] = "pricing_specification"
    schema["properties"]["schema_version"]["const"] = 1
    schema["properties"]["model"] = ref("model_composition")
    schema["properties"]["numerics"] = ref("numerics")
    schema["required"].append("numerics")
    requires_step = {"properties": {"model": {"anyOf": [
        equity_is("stochastic_volatility", "rough_volatility"),
        {"allOf": [hybrid, equity_is("standard")]},
    ]}}}
    schema["allOf"].append({"if": requires_step,
        "then": {"properties": {"numerics": {"required": ["maximum_step"]}}},
        "else": {"properties": {"numerics": {"not": {"required": ["maximum_step"]}}}}})
    extension = {"properties": {"model": {"anyOf": [hybrid, {"not": equity_is("standard")}]}}}
    schema["allOf"].append({"if": extension, "then": {
        "not": {"required": ["lsm"]},
        "properties": {"risk": {"properties": {"delta": {"const": False}, "vega": {"const": False}},
            "not": {"anyOf": [{"required": [key]} for key in (
                "gamma", "vega_kt", "payoff_smoothing_width_ladder", "checkpoint_interval", "aad_tile_capacity")]}}},
    }})
    return copy.deepcopy(schema)


def schema_text():
    return json.dumps(build_schema(), indent=2, ensure_ascii=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = schema_text()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            raise SystemExit("pricing specification schema is stale; run scripts/build_specification_schema.py")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text)


if __name__ == "__main__":
    main()
