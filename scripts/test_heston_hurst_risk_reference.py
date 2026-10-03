"""Mutations must not weaken reference inputs, budgets or retained CI evidence."""
from __future__ import annotations
import copy
import io
import json
from pathlib import Path
import tarfile
import unittest
import check_heston_hurst_risk as reference
import check_source_archive as archive

ROOT=Path(__file__).resolve().parents[1]
WORKFLOW='.github/workflows/heston-hurst-risk.yml'
GATES=(
    'cargo test --locked -p pricing --lib hurst',
    'cargo test --locked -p pricing-numerics',
    'python scripts/check_heston_hurst_risk.py',
    "python -m unittest discover -s scripts -p 'test_heston_hurst_risk_reference.py'",
    'cargo test --locked -p pricing --test heston_hurst_risk',
    'cargo test --locked --no-default-features -p pricing --test heston_hurst_risk',
    'cargo test --locked --release -p pricing --test heston_hurst_risk -- --include-ignored --nocapture',
    'os: [ubuntu-24.04, macos-15, windows-2025]',
    'name: heston-hurst-risk-${{ matrix.os }}',
    'path: heston-hurst-risk.log',
    'if-no-files-found: error',
)

def check_workflow(workflow,smoke):
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w') as package:
        for name,text in [(WORKFLOW,workflow),('scripts/smoke_test_wheel.py',smoke)]:
            entry=tarfile.TarInfo(name);content=text.encode();entry.size=len(content)
            package.addfile(entry,io.BytesIO(content))
    data.seek(0)
    with tarfile.open(fileobj=data,mode='r') as package:
        archive.check_heston_hurst_risk_workflow(package,'guard.tar')

class HurstReferenceGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.expected=reference.generate()

    def test_independent_references_without_modifying_saved_file(self):
        before=reference.FIXTURE.read_bytes()
        reference.compare(json.loads(before),self.expected)
        self.assertEqual(reference.FIXTURE.read_bytes(),before)

    def test_reference_input_and_budget_mutations_fail(self):
        changes=[]
        for value in [0.0,float('nan'),float('inf'),'0',False]:
            item=copy.deepcopy(self.expected);item['deterministic'][0]['derivative']=value;changes.append(item)
        for key in ['deterministic','transforms','kernels']:
            item=copy.deepcopy(self.expected);item[key].pop();changes.append(item)
        for key in ['deterministic_price_abs_tolerance','transform_abs_tolerance','refinement_abs_tolerance']:
            item=copy.deepcopy(self.expected);item['protocol'][key]*=2;changes.append(item)
        item=copy.deepcopy(self.expected);item['parameters'][0]+=1e-10;changes.append(item)
        item=copy.deepcopy(self.expected);item['hursts'].reverse();changes.append(item)
        item=copy.deepcopy(self.expected);item['transforms'][0]['derivative'][0]+=1e-4;changes.append(item)
        for item in changes:
            with self.assertRaises(ValueError):reference.compare(item,self.expected)

    def test_ci_evidence_and_executable_example_mandatory(self):
        workflow=(ROOT/WORKFLOW).read_text();smoke=(ROOT/'scripts/smoke_test_wheel.py').read_text()
        check_workflow(workflow,smoke)
        for gate in GATES:
            self.assertIn(gate,workflow)
            with self.subTest(gate=gate),self.assertRaises(SystemExit):
                check_workflow(workflow.replace(gate,'REMOVED'),smoke)
        with self.assertRaises(SystemExit):
            check_workflow(workflow,smoke.replace('"examples/python/heston_hurst_risk.py"','"REMOVED"'))

    def test_new_files_required_in_source_archive(self):
        new_files={WORKFLOW,
            'crates/pricing/src/engine/analytic/heston_fourier/hurst_risk.rs',
            'crates/pricing/tests/heston_hurst_risk.rs',
            'fixtures/rough-volatility/hurst-risk.json',
            'scripts/check_heston_hurst_risk.py',
            'scripts/test_heston_hurst_risk_reference.py',
            'tests/python/test_heston_hurst_risk.py',
            'examples/python/heston_hurst_risk.py',
            'docs/models/heston-hurst-risk.md',
            'design/validation/heston-hurst-risk.md'}
        self.assertTrue(new_files <= archive.REQUIRED_FILES)

if __name__=='__main__':unittest.main()
