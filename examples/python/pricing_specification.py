"""Replay complete model inputs, including calibration and execution settings."""
from pathlib import Path

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]


def main():
    for name in ["black_scholes", "bergomi_hull_white", "bergomi_lsv", "rough_heston"]:
        source = rp.PricingSpecification.from_json(
            (ROOT / f"fixtures/specification-v1/{name}.golden.json").read_text()
        )
        saved = source.to_pretty_json()
        restored = rp.PricingSpecification.from_json(saved)
        plan = restored.compile()
        result = plan.evaluate()
        assert source.fingerprint == result.specification_fingerprint
        print(name, result.value, result.standard_error, plan.plan_fingerprint)


if __name__ == "__main__":
    main()
