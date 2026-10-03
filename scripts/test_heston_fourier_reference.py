"""Mutation guards for independent Fourier references and mandatory CI evidence."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tarfile
import unittest

import check_heston_fourier_reference as reference
import check_source_archive as archive

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/heston-fourier.yml"
GATES = (
    "python scripts/check_heston_fourier_reference.py",
    "python -m unittest discover -s scripts -p 'test_heston_fourier_reference.py'",
    "cargo test --locked -p pricing --test heston_fourier",
    "cargo test --locked --no-default-features -p pricing --test heston_fourier",
    "cargo test --locked --release -p pricing --test heston_fourier -- --include-ignored --nocapture",
    "os: [ubuntu-24.04, macos-15, windows-2025]",
    "name: heston-fourier-${{ matrix.os }}",
    "path: heston-fourier.log",
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
        archive.check_heston_fourier_workflow(package, "mutated-workflow.tar")


class FourierReferenceContracts(unittest.TestCase):
    def test_retained_reference_matches_independent_calculations(self) -> None:
        reference.check(json.loads(reference.FIXTURE.read_text()))

    def test_corruptions_and_protocol_changes_are_rejected(self) -> None:
        expected = reference.compute()
        candidates = []
        for kind, field in [("transforms", "real"), ("prices", "call")]:
            for value in [float("nan"), float("inf"), -1.0]:
                modified = copy.deepcopy(expected)
                modified[kind][0][field] = value
                candidates.append(modified)
            for operation in ["drop", "duplicate"]:
                modified = copy.deepcopy(expected)
                if operation == "drop":
                    modified[kind].pop()
                else:
                    modified[kind].append(copy.deepcopy(modified[kind][0]))
                candidates.append(modified)
        for key in ["price_tolerance", "transform_tolerance", "time_steps"]:
            modified = copy.deepcopy(expected)
            modified["protocol"][key] *= 2
            candidates.append(modified)
        modified = copy.deepcopy(expected)
        modified["parameters"]["correlation"] = 0.0
        candidates.append(modified)
        modified = copy.deepcopy(expected)
        modified["transforms"][1]["damping"] = 0.25
        candidates.append(modified)
        modified = copy.deepcopy(expected)
        modified["prices"][-1]["reference_time_change"] = 0.1
        candidates.append(modified)
        for i, modified in enumerate(candidates):
            with self.subTest(case=i), self.assertRaises(AssertionError):
                reference.check(modified)

    def test_every_ci_gate_is_required(self) -> None:
        text = (ROOT / WORKFLOW).read_text()
        check_workflow(text)
        for gate in GATES:
            with self.subTest(gate=gate):
                self.assertIn(gate, text)
                with self.assertRaises(SystemExit):
                    check_workflow(text.replace(gate, "REMOVED_FOURIER_GATE"))

    def test_source_archive_requires_fourier_inputs(self) -> None:
        required = {
            "crates/pricing-numerics/src/complex.rs",
            "crates/pricing/src/engine/analytic/heston_fourier/mod.rs",
            "crates/pricing/src/engine/analytic/heston_fourier/riccati.rs",
            "crates/pricing/tests/heston_fourier.rs",
            "scripts/check_heston_fourier_reference.py",
            "scripts/test_heston_fourier_reference.py",
            "fixtures/rough-volatility/fourier.json",
            "tests/python/test_heston_fourier.py",
            "examples/python/heston_fourier.py",
            "docs/models/heston-fourier.md",
            "design/validation/heston-fourier.md",
            WORKFLOW,
        }
        self.assertTrue(required <= archive.REQUIRED_FILES)
        for filename in required:
            self.assertTrue((ROOT / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
