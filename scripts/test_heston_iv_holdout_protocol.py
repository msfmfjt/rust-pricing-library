"""Protect retained targets, separation, acceptance budgets and required evidence."""
import copy
import io
import json
from pathlib import Path
import tarfile
import unittest
import check_source_archive as archive
import check_heston_iv_holdout as ref

ROOT=Path(__file__).resolve().parents[1]
MEMBERS={'.github/workflows/heston-iv-holdout.yml','crates/pricing/tests/heston_iv_holdout.rs',
    'tests/python/test_heston_iv_holdout.py','examples/python/heston_iv_holdout.py',
    'fixtures/rough-volatility/iv-holdout.json','scripts/check_heston_iv_holdout.py',
    'scripts/test_heston_iv_holdout_protocol.py','docs/models/heston-iv-holdout.md',
    'design/validation/heston-iv-holdout.md'}


def workflow_check(workflow,smoke):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w') as tar:
        for path,text in [('.github/workflows/heston-iv-holdout.yml',workflow),('scripts/smoke_test_wheel.py',smoke)]:
            body=text.encode();entry=tarfile.TarInfo(path);entry.size=len(body)
            tar.addfile(entry,io.BytesIO(body))
    stream.seek(0)
    with tarfile.open(fileobj=stream,mode='r') as tar:
        archive.check_heston_iv_holdout_workflow(tar,'mutation.tar')


class HoldoutProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved=json.loads(ref.FILE.read_text())
        cls.fresh=ref.generate()

    def test_independent_targets_and_disjoint_sites(self):
        ref.compare(self.saved,self.fresh)
        parent=json.loads(ref.PARENT.read_text())
        for family,count in self.saved['row_counts'].items():
            rows=[q for q in self.saved['rows'] if q['family']==family]
            sites={(q['maturity'],q['strike']) for q in rows}
            self.assertEqual(len(rows),count);self.assertEqual(len(sites),count)
            fitting={(t,k) for t in [.25,.75,1.5] for k in [85.,100.,115.]} if family=='ssvi' else {
                (q['maturity'],q['strike']) for q in parent['markov_quotes'] if q['family']==family}
            self.assertFalse(sites & fitting)
        self.assertEqual(sum(q['region']=='extrapolation' for q in self.saved['rows']),12)

    def test_target_protocol_and_coordinate_mutations_are_rejected(self):
        mutations=[]
        for name in ref.PROTOCOL:
            bad=copy.deepcopy(self.saved)
            if isinstance(bad['protocol'][name],list):bad['protocol'][name].pop()
            else:bad['protocol'][name]*=2
            mutations.append(bad)
        for field in ['target_volatility','strike','maturity']:
            bad=copy.deepcopy(self.saved);bad['rows'][0][field]+=0.01;mutations.append(bad)
        for field in ['starts','bounds','rows']:
            bad=copy.deepcopy(self.saved);bad[field].pop();mutations.append(bad)
        bad=copy.deepcopy(self.saved);bad['rows'][0]['target_volatility']=float('nan');mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(ValueError):ref.compare(bad,self.fresh)

    def test_evidence_and_dependency_mutations_are_rejected(self):
        wf=(ROOT/'.github/workflows/heston-iv-holdout.yml').read_text()
        smoke=(ROOT/'scripts/smoke_test_wheel.py').read_text()
        workflow_check(wf,smoke)
        for gate in ['mpmath>=1.3,<2','numpy>=2,<3','scipy>=1.14,<2','check_heston_iv_holdout.py',
            'test_heston_iv_holdout_protocol.py','--include-ignored --nocapture','--no-default-features',
            'os: [ubuntu-24.04, macos-15, windows-2025]','name: heston-iv-holdout-${{ matrix.os }}',
            'path: heston-iv-holdout.log','if-no-files-found: error','contents: read','set -o pipefail']:
            with self.subTest(gate=gate),self.assertRaises(SystemExit):workflow_check(wf.replace(gate,'REMOVED'),smoke)
        with self.assertRaises(SystemExit):workflow_check(wf,smoke.replace('"examples/python/heston_iv_holdout.py"','"REMOVED"'))

    def test_archive_members_are_required(self):
        self.assertTrue(MEMBERS <= archive.REQUIRED_FILES)
        for path in MEMBERS:self.assertTrue((ROOT/path).is_file(),path)

if __name__=='__main__':unittest.main()
