"""Protocol, reference, artifact and CI-gate mutation controls."""
import copy
import io
import json
from pathlib import Path
import tarfile
import unittest
import check_heston_calibration_protocol as protocol
import check_source_archive as archive
ROOT=Path(__file__).resolve().parents[1]
WORKFLOW='.github/workflows/heston-calibration.yml'
GATES=('cargo test --locked -p pricing-numerics', 'cargo test --locked -p pricing --test heston_calibration', 'cargo test --locked --no-default-features -p pricing --test heston_calibration', 'cargo test --locked --release -p pricing --test heston_calibration -- --include-ignored --nocapture', 'python scripts/check_heston_fourier.py', 'python scripts/check_heston_calibration_protocol.py', "python -m unittest discover -s scripts -p 'test_heston_calibration_protocol.py'", 'python scripts/compare_heston_calibration_solvers.py', 'os: [ubuntu-24.04, macos-15, windows-2025]', 'name: heston-calibration-${{ matrix.os }}', 'path: heston-calibration.log', 'if-no-files-found: error', 'name: heston-calibration-solvers', 'path: heston-calibration-solvers.log')
MEMBERS={'tests/python/test_heston_calibration.py', 'design/validation/heston-calibration.md', 'docs/models/heston-calibration.md', '.github/workflows/heston-calibration.yml', 'scripts/test_heston_calibration_protocol.py', 'fixtures/rough-volatility/calibration.json', 'crates/pricing-numerics/src/least_squares.rs', 'scripts/check_heston_calibration_protocol.py', 'crates/pricing/src/engine/analytic/heston_fourier/calibration.rs', 'examples/python/heston_calibration.py', 'crates/pricing-python/src/heston_calibration.rs', 'scripts/compare_heston_calibration_solvers.py', 'crates/pricing/tests/heston_calibration.rs'}

def check_workflow(workflow,smoke):
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w') as package:
        for name,text in [(WORKFLOW,workflow),('scripts/smoke_test_wheel.py',smoke)]:
            content=text.encode();entry=tarfile.TarInfo(name);entry.size=len(content)
            package.addfile(entry,io.BytesIO(content))
    data.seek(0)
    with tarfile.open(fileobj=data,mode='r') as package:archive.check_heston_calibration_workflow(package,'guard.tar')

class CalibrationProtocolGuards(unittest.TestCase):
    def test_retained_protocol_and_reference_file(self):protocol.main()
    def test_protocol_and_target_mutations_fail(self):
        ref=(ROOT/protocol.EXPECTED['reference_file']).read_bytes()
        for key in ['max_iterations','max_evaluations','residual_tolerance','reprice_budget']:
            bad=copy.deepcopy(protocol.EXPECTED);bad[key]*=2
            with self.subTest(key=key),self.assertRaises(ValueError):protocol.check(bad,ref)
        for bad in [ref+b' ', b'{}']:
            with self.assertRaises(ValueError):protocol.check(protocol.EXPECTED,bad)
    def test_evidence_gates_and_example_cannot_be_removed(self):
        workflow=(ROOT/WORKFLOW).read_text();smoke=(ROOT/'scripts/smoke_test_wheel.py').read_text()
        check_workflow(workflow,smoke)
        for gate in GATES:
            with self.subTest(gate=gate),self.assertRaises(SystemExit):check_workflow(workflow.replace(gate,'REMOVED'),smoke)
        with self.assertRaises(SystemExit):check_workflow(workflow,smoke.replace('"examples/python/heston_calibration.py"','"REMOVED"'))
    def test_new_archive_members_are_mandatory(self):self.assertTrue(MEMBERS <= archive.REQUIRED_FILES)
if __name__=='__main__':unittest.main()
