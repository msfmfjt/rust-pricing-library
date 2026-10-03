"""Keep experiment inputs, parent reference and mandatory CI/source members fixed."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import unittest
import check_source_archive as archive

ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/'fixtures/rough-volatility/iv-refinement.json'
PARENT=ROOT/'fixtures/rough-volatility/iv-calibration.json'
EXPECTED={
    'time_steps':32,'integration_intervals':256,'cutoff':64.,
    'fit_tolerance':5e-4,'grid_tolerance':1e-4,'max_stages':2,
    'max_iterations':60,'max_evaluations':100,'residual_tolerance':2e-6,
    'warm_start_grid_tolerance':1e-5,'warm_start_time_steps':64,
}
MEMBERS={'.github/workflows/heston-iv-refinement.yml',
'crates/pricing/src/engine/analytic/heston_fourier/iv_refinement.rs',
'crates/pricing-python/src/heston_iv_refinement.rs',
'crates/pricing/tests/heston_iv_refinement.rs',
'tests/python/test_heston_iv_refinement.py',
'examples/python/heston_iv_refinement.py',
'fixtures/rough-volatility/iv-refinement.json',
'scripts/test_heston_iv_refinement_protocol.py',
'docs/models/heston-iv-refinement.md','design/validation/heston-iv-refinement.md'}

def check(data):
    parent=json.loads(PARENT.read_text())
    expected={
        'version':1,'parent_sha256':hashlib.sha256(PARENT.read_bytes()).hexdigest(),
        'protocol':EXPECTED,'starts':parent['protocol']['starts'],'bounds':parent['protocol']['bounds'],
        'probe_names':['base','time','frequency','cutoff','joint'],
        'ssvi':{'times':[.25,.75,1.5],'theta':[.01,.03,.06],'slope':.04,'rho':-.5,'eta':.35,'gamma':.5,'strikes':[85.,100.,115.]},
        'case_count':8,
    }
    if data!=expected: raise ValueError('IV refinement fixed protocol or parent reference changed')

def workflow_check(workflow,smoke):
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w') as package:
        for name,text in [('.github/workflows/heston-iv-refinement.yml',workflow),('scripts/smoke_test_wheel.py',smoke)]:
            content=text.encode();item=tarfile.TarInfo(name);item.size=len(content)
            package.addfile(item,io.BytesIO(content))
    data.seek(0)
    with tarfile.open(fileobj=data,mode='r') as package:
        archive.check_heston_iv_refinement_workflow(package,'mutation.tar')

class RefinementProtocolTests(unittest.TestCase):
    def test_retained_protocol_and_parent_hash(self):
        check(json.loads(FILE.read_text()))
    def test_input_mutations_are_rejected(self):
        data=json.loads(FILE.read_text())
        for key in EXPECTED:
            bad=copy.deepcopy(data);bad['protocol'][key]*=2
            with self.subTest(key=key),self.assertRaises(ValueError):check(bad)
        for key in ['starts','bounds','probe_names']:
            bad=copy.deepcopy(data);bad[key].pop()
            with self.subTest(key=key),self.assertRaises(ValueError):check(bad)
        bad=copy.deepcopy(data);bad['parent_sha256']='bad'
        with self.assertRaises(ValueError):check(bad)
    def test_missing_evidence_gates_fail(self):
        workflow=(ROOT/'.github/workflows/heston-iv-refinement.yml').read_text()
        smoke=(ROOT/'scripts/smoke_test_wheel.py').read_text()
        workflow_check(workflow,smoke)
        for token in ['mpmath>=1.3,<2', '--include-ignored --nocapture','--no-default-features','check_heston_iv_calibration.py',
                      'test_heston_iv_refinement_protocol.py','os: [ubuntu-24.04, macos-15, windows-2025]',
                      'name: heston-iv-refinement-${{ matrix.os }}','path: heston-iv-refinement.log',
                      'if-no-files-found: error','contents: read','set -o pipefail']:
            with self.subTest(token=token),self.assertRaises(SystemExit):
                workflow_check(workflow.replace(token,'REMOVED'),smoke)
        with self.assertRaises(SystemExit):
            workflow_check(workflow,smoke.replace('"examples/python/heston_iv_refinement.py"','"REMOVED"'))
    def test_new_source_members_required(self):
        self.assertTrue(MEMBERS <= archive.REQUIRED_FILES)

if __name__=='__main__': unittest.main()
