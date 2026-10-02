"""Regression guards for retained refinement data and its release-CI gates."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tarfile
import unittest

import check_rough_volatility_refinement as reference
import check_source_archive as archive

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/rough-volatility.yml"
GATES = (
    "python scripts/check_rough_volatility_refinement.py",
    "python -m unittest discover -s scripts -p 'test_rough_volatility_refinement.py'",
    "cargo test --locked -p pricing --test rough_volatility_refinement",
    "cargo test --locked --no-default-features -p pricing --test rough_volatility_refinement",
    "cargo test --locked --release -p pricing --test rough_volatility_refinement -- --include-ignored --nocapture",
    "name: rough-volatility-refinement-${{ matrix.os }}",
    "path: rough-volatility-refinement.log",
)


def check_workflow(text: str) -> None:
    """Exercise the real archive checker against a tiny in-memory archive."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as package:
        data = text.encode("utf-8")
        member = tarfile.TarInfo(WORKFLOW)
        member.size = len(data)
        package.addfile(member, io.BytesIO(data))
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode="r") as package:
        archive.check_rough_refinement_workflow(package, "mutation-fixture.tar")


class RefinementContracts(unittest.TestCase):
    def test_retained_fixture_matches_independent_reconstruction(self) -> None:
        reference.compare(json.loads(reference.FIXTURE.read_text()), reference.reference())

    def test_corrupt_or_truncated_fixtures_are_rejected(self) -> None:
        expected = reference.reference()
        candidates = []
        for section, field in (
            ("cells", "extra_coefficient"),
            ("cells", "normalized_integral_cross_covariances"),
            ("lifts", "kernel_values"),
            ("lifts", "relative_errors"),
            ("lifts", "ratio"),
        ):
            changed = copy.deepcopy(expected)
            value = changed[section][0][field]
            if isinstance(value, list):
                value[0] += 1e-5
            else:
                changed[section][0][field] += 1e-5
            candidates.append((field, changed))
        missing = copy.deepcopy(expected)
        missing["cells"].pop()
        candidates.append(("missing-cell", missing))
        duplicate = copy.deepcopy(expected)
        duplicate["lifts"].append(copy.deepcopy(duplicate["lifts"][0]))
        candidates.append(("extra-lift", duplicate))
        for label, changed in candidates:
            with self.subTest(label=label), self.assertRaises(AssertionError):
                reference.compare(changed, expected)

    def test_each_refinement_ci_gate_is_required(self) -> None:
        text = (ROOT / WORKFLOW).read_text()
        check_workflow(text)
        for gate in GATES:
            with self.subTest(gate=gate):
                self.assertIn(gate, text)
                with self.assertRaises(SystemExit):
                    check_workflow(text.replace(gate, "REMOVED_REFINEMENT_GATE"))

    def test_source_archive_requires_all_refinement_inputs(self) -> None:
        required = {
            "crates/pricing/tests/rough_volatility_refinement.rs",
            "scripts/check_rough_volatility_refinement.py",
            "scripts/test_rough_volatility_refinement.py",
            "fixtures/rough-volatility/refinement.json",
            "design/validation/rough-volatility-refinement.md",
            WORKFLOW,
        }
        self.assertTrue(required <= archive.REQUIRED_FILES)


if __name__ == "__main__":
    unittest.main()
