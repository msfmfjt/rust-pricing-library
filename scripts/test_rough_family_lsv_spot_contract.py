"""Required artifacts and explicit Delta API/CI contracts, without loading the wheel."""
import ast
from pathlib import Path
import unittest
import check_source_archive as archive
import smoke_test_wheel as wheel

ROOT=Path(__file__).resolve().parents[1]
MEMBERS={'examples/python/rough_family_lsv_spot_delta.py', 'docs/models/rough-family-lsv-spot-delta.md', 'scripts/test_rough_family_lsv_spot_contract.py', 'design/validation/rough-family-lsv-spot-delta.md', '.github/workflows/rough-family-lsv-spot-delta.yml', 'tests/python/test_rough_family_lsv_spot_delta.py', 'crates/pricing/tests/rough_family_lsv_spot_delta.rs', 'crates/pricing/src/engine/risk/lsv/rough_families/spot_delta.rs'}
GATES=["contents: read", "os: [ubuntu-24.04, macos-15, windows-2025]",
       "cargo test --locked -p pricing --test rough_family_lsv_spot_delta",
       "cargo test --locked --no-default-features -p pricing --test rough_family_lsv_spot_delta",
       "cargo test --locked --release -p pricing --test rough_family_lsv_spot_delta -- --include-ignored --nocapture",
       "set -o pipefail", "if-no-files-found: error", "path: rough-family-lsv-spot-delta.log"]
def check_workflow(text):
    missing=[x for x in GATES if x not in text]
    if missing: raise ValueError(missing)

class SpotContract(unittest.TestCase):
    def test_members_and_workflow(self):
        self.assertTrue(MEMBERS <= archive.REQUIRED_FILES)
        for f in MEMBERS: self.assertTrue((ROOT/f).is_file(),f)
        text=(ROOT/'.github/workflows/rough-family-lsv-spot-delta.yml').read_text()
        check_workflow(text)
        for gate in GATES:
            with self.assertRaises(ValueError): check_workflow(text.replace(gate,'REMOVED'))
        self.assertIn('"examples/python/rough_family_lsv_spot_delta.py"',(ROOT/'scripts/smoke_test_wheel.py').read_text())
    def test_removed_class_or_members_fail(self):
        text=(ROOT/'rust_pricing.pyi').read_text()
        wheel.exported_stub_api(text.encode())
        for name,member in [('RoughFamilyLsvPlan','evaluate_frozen_leverage_delta'),
                            ('RoughFamilyLsvPlan','evaluate_sticky_moneyness_delta'),
                            ('RoughFamilyLsvDelta','convention'),('RoughFamilyLsvDelta','delta_standard_error')]:
            tree=ast.parse(text)
            cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name==name)
            cls.body=[c for c in cls.body if not isinstance(c,ast.FunctionDef) or c.name!=member]
            with self.assertRaises(RuntimeError): wheel.verify_stub_static_shape(tree)
    def test_reference_protocol_and_semantics(self):
        text=(ROOT/'crates/pricing/tests/rough_family_lsv_spot_delta.rs').read_text()
        for x in ['0.539827837277029','7.965567455405804','1973','3e-8','5e-5',
                  'initial_spatial_kink_is_not_silently_a_two_sided_frozen_delta']:
            self.assertIn(x,text)
        doc=(ROOT/'docs/models/rough-family-lsv-spot-delta.md').read_text()
        for x in ['NOT sticky-strike market-IV','S_ref/R_ref','q_j * f_j / R_ref','conditional']:
            self.assertIn(x,doc)

if __name__=='__main__': unittest.main()
