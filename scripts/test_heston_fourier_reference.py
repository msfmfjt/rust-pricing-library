"""Guard retained mathematical references, fixed protocol and mandatory CI evidence."""
from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tarfile
import unittest

import check_heston_fourier as reference
import check_source_archive as archive

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = '.github/workflows/heston-fourier.yml'
GATES = (
    'python scripts/check_heston_fourier.py',
    "python -m unittest discover -s scripts -p 'test_heston_fourier_reference.py'",
    'cargo test --locked -p pricing --test heston_fourier',
    'cargo test --locked --no-default-features -p pricing --test heston_fourier',
    'cargo test --locked --release -p pricing --test heston_fourier -- --include-ignored --nocapture',
    'os: [ubuntu-24.04, macos-15, windows-2025]',
    'name: heston-fourier-${{ matrix.os }}',
    'path: heston-fourier.log',
    'if-no-files-found: error',
)


def check_workflow(workflow: str, smoke: str) -> None:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w') as package:
        for name, text in [(WORKFLOW, workflow), ('scripts/smoke_test_wheel.py', smoke)]:
            item = tarfile.TarInfo(name)
            encoded = text.encode()
            item.size = len(encoded)
            package.addfile(item, io.BytesIO(encoded))
    data.seek(0)
    with tarfile.open(fileobj=data, mode='r') as package:
        archive.check_heston_fourier_workflow(package, 'guard.tar')


class FourierReferenceGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = reference.generate()

    def test_independent_reference_without_writing_fixture(self):
        before = reference.FIXTURE.read_bytes()
        reference.compare(json.loads(before), self.expected)
        self.assertEqual(before, reference.FIXTURE.read_bytes())

    def test_reference_and_exact_protocol_mutations(self):
        candidates = []
        for value in [0.0, float('nan'), float('inf'), '0', False]:
            mutated = copy.deepcopy(self.expected)
            mutated['rough_transforms'][0]['log_transform'][0] = value
            candidates.append(mutated)
        for key in ['rough_transforms', 'markov_transforms', 'prices']:
            mutated = copy.deepcopy(self.expected)
            mutated[key].pop()
            candidates.append(mutated)
        for key in ['mc_points', 'mc_steps', 'mc_rough_h01_steps', 'mc_combined_budget', 'mc_se_cap']:
            mutated = copy.deepcopy(self.expected)
            mutated['protocol'][key] *= 2
            candidates.append(mutated)
        for key in ['initial_variance', 'vol_of_vol', 'correlation']:
            mutated = copy.deepcopy(self.expected)
            mutated['parameters'][key] += 1e-10
            candidates.append(mutated)
        mutated = copy.deepcopy(self.expected)
        mutated['prices'][0]['call'] += .001
        candidates.append(mutated)
        for i, value in enumerate(candidates):
            with self.subTest(i=i), self.assertRaises(ValueError):
                reference.compare(value, self.expected)

    def test_ci_and_wheel_example_are_mandatory(self):
        text = (ROOT / WORKFLOW).read_text()
        smoke = (ROOT / 'scripts/smoke_test_wheel.py').read_text()
        check_workflow(text, smoke)
        for gate in GATES:
            self.assertIn(gate, text)
            with self.subTest(gate=gate), self.assertRaises(SystemExit):
                check_workflow(text.replace(gate, 'REMOVED'), smoke)
        with self.assertRaises(SystemExit):
            check_workflow(text, smoke.replace('"examples/python/heston_fourier.py"', '"REMOVED"'))

    def test_archive_requires_new_sources_and_references(self):
        files = {
            'crates/pricing-numerics/src/complex.rs',
            'crates/pricing/src/engine/analytic/heston_fourier/mod.rs',
            'crates/pricing/src/engine/analytic/heston_fourier/riccati.rs',
            'crates/pricing-python/src/heston_fourier.rs',
            'crates/pricing/tests/heston_fourier.rs',
            'fixtures/rough-volatility/fourier.json',
            'scripts/check_heston_fourier.py',
            'scripts/test_heston_fourier_reference.py',
            'tests/python/test_heston_fourier.py',
            'examples/python/heston_fourier.py',
            'docs/models/rough-heston-fourier.md',
            'design/validation/rough-heston-fourier.md', WORKFLOW,
        }
        self.assertTrue(files <= archive.REQUIRED_FILES)


if __name__ == '__main__':
    unittest.main()
