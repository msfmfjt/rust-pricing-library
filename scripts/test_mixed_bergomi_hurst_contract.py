"""New method and independent Hurst evidence cannot disappear silently."""
import ast
import unittest
from pathlib import Path
import check_source_archive as archive
import smoke_test_wheel as wheel
ROOT=Path(__file__).resolve().parents[1]
class HurstContract(unittest.TestCase):
    def test_methods_are_mandatory(self):
        text=(ROOT/'rust_pricing.pyi').read_text()
        wheel.exported_stub_api(text.encode())
        for cls in ['RoughVolatilityPlan','RoughFamilyLsvPlan']:
            tree=ast.parse(text)
            c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
            c.body=[f for f in c.body if not isinstance(f,ast.FunctionDef) or f.name!='evaluate_mixed_bergomi_parameter_risk_with_hurst']
            with self.assertRaises(RuntimeError):wheel.verify_stub_static_shape(tree)
    def test_reference_and_ci_gates(self):
        for p in ['fixtures/rough-volatility/mixed-hurst.json','crates/pricing/tests/mixed_bergomi_hurst_risk.rs',
                  'scripts/check_mixed_bergomi_hurst_reference.py','tests/python/test_mixed_bergomi_hurst_risk.py']:
            self.assertIn(p,archive.REQUIRED_FILES);self.assertTrue((ROOT/p).is_file())
        text=(ROOT/'.github/workflows/mixed-bergomi-parameter-risk.yml').read_text()
        for s in ['mpmath','check_mixed_bergomi_hurst_reference.py','test_mixed_bergomi_hurst_contract.py',
                  'mixed_hurst_normalized_kernel_matches_high_precision',
                  'mixed_hurst_total_scramble_errors_preserve_direct_calibration_covariance',
                  '--test mixed_bergomi_hurst_risk -- --include-ignored --nocapture','mixed-bergomi-hurst-risk.log',
                  'if-no-files-found: error']:
            self.assertIn(s,text)
if __name__=='__main__':unittest.main()
