"""Regression guards for fixed factor-price inputs and mandatory CI evidence."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tarfile
import unittest

import check_lifted_heston_factors as reference
import check_source_archive as archive

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/lifted-heston-factors.yml"
GATES = (
    "python scripts/check_lifted_heston_factors.py",
    "python -m unittest discover -s scripts -p 'test_lifted_heston_factors.py'",
    "cargo test --locked -p pricing --test lifted_heston_factor_prices",
    "cargo test --locked --no-default-features -p pricing --test lifted_heston_factor_prices",
    "cargo test --locked --release -p pricing --test lifted_heston_factor_prices -- --include-ignored --nocapture",
    "os: [ubuntu-24.04, macos-15, windows-2025]",
    "name: lifted-heston-factor-prices-${{ matrix.os }}",
    "path: lifted-heston-factor-prices.log",
    "if-no-files-found: error",
)


def check_workflow(text: str) -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as package:
        data = text.encode("utf-8")
        item = tarfile.TarInfo(WORKFLOW)
        item.size = len(data)
        package.addfile(item, io.BytesIO(data))
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode="r") as package:
        archive.check_lifted_factor_workflow(package, "mutated-workflow.tar")


class FactorPriceContracts(unittest.TestCase):
    def test_retained_reference_matches_independent_integrals(self) -> None:
        reference.compare(json.loads(reference.FIXTURE.read_text()), reference.reference())

    def test_reference_and_protocol_corruptions_are_rejected(self) -> None:
        expected = reference.reference()
        candidates = []
        for value in [0.0, float("nan"), float("inf"), -1.0]:
            modified = copy.deepcopy(expected)
            modified["kernels"][0]["infinite_kernel"][0] = value
            candidates.append(modified)
        finite = copy.deepcopy(expected)
        finite["kernels"][0]["finite_kernels"][0]["values"][0] += 0.001
        candidates.append(finite)
        missing = copy.deepcopy(expected)
        missing["kernels"].pop()
        candidates.append(missing)
        duplicate = copy.deepcopy(expected)
        duplicate["kernels"].append(copy.deepcopy(duplicate["kernels"][0]))
        candidates.append(duplicate)
        for key in ["vol_of_vol", "paired_price_budget", "independent_antithetic_units"]:
            modified = copy.deepcopy(expected)
            modified["protocol"][key] = 0
            candidates.append(modified)
        for i, modified in enumerate(candidates):
            with self.subTest(case=i), self.assertRaises(AssertionError):
                reference.compare(modified,expected)

    def test_every_ci_gate_is_required(self) -> None:
        text = (ROOT/WORKFLOW).read_text()
        check_workflow(text)
        for gate in GATES:
            with self.subTest(gate=gate):
                self.assertIn(gate,text)
                with self.assertRaises(SystemExit):
                    check_workflow(text.replace(gate,"REMOVED_FACTOR_PRICE_GATE"))

    def test_source_archive_requires_factor_price_inputs(self) -> None:
        required = {
            "crates/pricing/tests/lifted_heston_factor_prices.rs",
            "scripts/check_lifted_heston_factors.py",
            "scripts/test_lifted_heston_factors.py",
            "fixtures/rough-volatility/lifted-factor-prices.json",
            "design/validation/lifted-heston-factor-prices.md", WORKFLOW,
        }
        self.assertTrue(required <= archive.REQUIRED_FILES)


if __name__ == "__main__":
    unittest.main()
