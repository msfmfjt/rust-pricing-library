"""New API removals and omitted numerical evidence must be detected."""
import ast
from pathlib import Path
import unittest
import smoke_test_wheel as wheel
import check_source_archive as archive
ROOT=Path(__file__).resolve().parents[1]

class Contract(unittest.TestCase):
    def test_new_members_are_mandatory(self):
        text=(ROOT/'rust_pricing.pyi').read_text()
        wheel.exported_stub_api(text.encode())
        targets=[('RoughVolatilityPlan','evaluate_mixed_bergomi_parameter_risk'),
            ('RoughFamilyLsvPlan','evaluate_mixed_bergomi_parameter_risk')]
        for cls,fields in [('MixedBergomiMcParameterRisk',['price','parameter_names','parameter_adjoints','standard_errors']),
            ('MixedBergomiLsvParameterRisk',['price','parameter_names','parameter_adjoints','direct_adjoints','calibration_adjoints','standard_errors'])]:
            targets.extend((cls,f) for f in fields)
        for cls,field in targets:
            tree=ast.parse(text)
            c=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls)
            c.body=[f for f in c.body if not isinstance(f,ast.FunctionDef) or f.name!=field]
            with self.assertRaises(RuntimeError):wheel.verify_stub_static_shape(tree)

    def test_source_and_numerical_checks_retained(self):
        paths=['crates/pricing/src/engine/processes/rough_volatility/mixed_parameter.rs',
            'crates/pricing/tests/mixed_bergomi_parameter_risk.rs',
            'tests/python/test_mixed_bergomi_parameter_risk.py','fixtures/rough-volatility/mixed-parameter.json']
        for p in paths:self.assertIn(p,archive.REQUIRED_FILES);self.assertTrue((ROOT/p).is_file())
        s=(ROOT/'.github/workflows/mixed-bergomi-parameter-risk.yml').read_text()
        for text in ['contents: read','os: [ubuntu-24.04, macos-15, windows-2025]',
            '--no-default-features','--include-ignored --nocapture','if-no-files-found: error',
            'mixed_total_scramble_errors_preserve_direct_calibration_covariance']:
            self.assertIn(text,s)

if __name__=='__main__':unittest.main()
