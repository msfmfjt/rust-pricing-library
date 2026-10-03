"""Retained IV targets, protocol and CI/evidence mutation tests."""
import copy
import io
import json
from pathlib import Path
import tarfile
import unittest
import check_heston_iv_calibration as protocol
import check_source_archive as archive
ROOT=Path(__file__).resolve().parents[1]
WORKFLOW='.github/workflows/heston-iv-calibration.yml'
GATES=('cargo test --locked -p pricing --lib black_', 'cargo test --locked -p pricing --test heston_iv_calibration', 'cargo test --locked --no-default-features -p pricing --test heston_iv_calibration', 'cargo test --locked --release -p pricing --test heston_iv_calibration -- --include-ignored --nocapture', 'python scripts/check_heston_iv_calibration.py', "python -m unittest discover -s scripts -p 'test_heston_iv_calibration_reference.py'", 'os: [ubuntu-24.04, macos-15, windows-2025]', 'name: heston-iv-calibration-${{ matrix.os }}', 'path: heston-iv-calibration.log', 'if-no-files-found: error')
MEMBERS={'.github/workflows/heston-iv-calibration.yml', 'tests/python/test_heston_iv_calibration.py', 'crates/pricing/src/engine/analytic/heston_fourier/iv_calibration.rs', 'crates/pricing/tests/heston_iv_calibration.rs', 'examples/python/heston_iv_calibration.py', 'crates/pricing-python/src/heston_iv_calibration.rs', 'docs/models/heston-iv-calibration.md', 'scripts/check_heston_iv_calibration.py', 'scripts/test_heston_iv_calibration_reference.py', 'fixtures/rough-volatility/iv-calibration.json', 'design/validation/heston-iv-calibration.md'}

def check_workflow(workflow,smoke):
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w') as package:
        for name,text in [(WORKFLOW,workflow),('scripts/smoke_test_wheel.py',smoke)]:
            content=text.encode();entry=tarfile.TarInfo(name);entry.size=len(content)
            package.addfile(entry,io.BytesIO(content))
    data.seek(0)
    with tarfile.open(fileobj=data,mode='r') as package:archive.check_heston_iv_calibration_workflow(package,'guard.tar')

class IvCalibrationReferenceGuards(unittest.TestCase):
    def test_retained_protocol_and_reference_file(self):protocol.main()
    def test_protocol_and_target_mutations_fail(self):
        data=json.loads(protocol.FILE.read_text())
        for section,key in [('protocol','residual_tolerance'),('protocol','fine_iv_budget'),('ssvi','rho')]:
            bad=copy.deepcopy(data);bad[section][key]*=2
            with self.subTest(section=section,key=key),self.assertRaises(ValueError):protocol.check(bad)
        for key in ['black','ssvi_quotes','markov_quotes']:
            bad=copy.deepcopy(data);bad[key].pop()
            with self.subTest(key=key),self.assertRaises(ValueError):protocol.check(bad)
        for key,field in [('black','vega'),('ssvi_quotes','target_volatility'),('markov_quotes','target_volatility')]:
            bad=copy.deepcopy(data);bad[key][0][field]*=1.01
            with self.subTest(key=key),self.assertRaises(ValueError):protocol.check(bad)
        bad=copy.deepcopy(data);bad['parent_sha256']='bad'
        with self.assertRaises(ValueError):protocol.check(bad)
    def test_evidence_gates_and_example_cannot_be_removed(self):
        workflow=(ROOT/WORKFLOW).read_text();smoke=(ROOT/'scripts/smoke_test_wheel.py').read_text()
        check_workflow(workflow,smoke)
        for gate in GATES:
            with self.subTest(gate=gate),self.assertRaises(SystemExit):check_workflow(workflow.replace(gate,'REMOVED'),smoke)
        with self.assertRaises(SystemExit):check_workflow(workflow,smoke.replace('"examples/python/heston_iv_calibration.py"','"REMOVED"'))
    def test_new_archive_members_are_mandatory(self):self.assertTrue(MEMBERS <= archive.REQUIRED_FILES)
if __name__=='__main__':unittest.main()
