"""Protect the new coordinate contract, numerical gates and source inventory."""
import ast
import unittest
from pathlib import Path
import check_source_archive as archive
import smoke_test_wheel as wheel
ROOT=Path(__file__).resolve().parents[1]
class ShapeContract(unittest.TestCase):
    def test_shape_methods_are_required(self):
        text=(ROOT/'rust_pricing.pyi').read_text()
        wheel.exported_stub_api(text.encode())
        for cls in ['RoughVolatilityPlan','RoughFamilyLsvPlan']:
            tree=ast.parse(text)
            c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
            c.body=[f for f in c.body if not isinstance(f,ast.FunctionDef) or f.name!='evaluate_mixed_bergomi_shape_risk']
            with self.assertRaises(RuntimeError):wheel.verify_stub_static_shape(tree)
    def test_source_and_ci_require_all_evidence(self):
        files=['crates/pricing/src/engine/processes/rough_volatility/mixed_shape.rs', 'crates/pricing/tests/mixed_bergomi_shape_risk.rs', 'tests/python/test_mixed_bergomi_shape_risk.py', 'scripts/check_mixed_bergomi_shape_reference.py', 'scripts/test_mixed_bergomi_shape_contract.py', 'fixtures/rough-volatility/mixed-shape.json', 'docs/models/mixed-bergomi-shape-risk.md', 'design/validation/mixed-bergomi-shape-risk.md', 'examples/python/mixed_bergomi_shape_risk.py']
        for path in files:
            self.assertIn(path,archive.REQUIRED_FILES)
            self.assertTrue((ROOT/path).is_file())
        text=(ROOT/'.github/workflows/mixed-bergomi-parameter-risk.yml').read_text()
        for s in ['check_mixed_bergomi_shape_reference.py','test_mixed_bergomi_shape_contract.py',
                  'mixed_shape_total_scramble_errors_preserve_direct_calibration_covariance',
                  '--test mixed_bergomi_shape_risk -- --include-ignored --nocapture',
                  'mixed-bergomi-shape-risk.log','if-no-files-found: error','RUSTDOCFLAGS: -D warnings']:
            self.assertIn(s,text)
if __name__=='__main__':unittest.main()
