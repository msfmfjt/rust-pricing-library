"""Protect QRH API, evidence and distribution contracts, without production imports."""
import ast
import unittest
from pathlib import Path
import check_source_archive as archive
import smoke_test_wheel as wheel
ROOT=Path(__file__).resolve().parents[1]
FILES=['crates/pricing/src/engine/processes/rough_volatility/quadratic_parameter.rs', 'crates/pricing/src/engine/risk/rough_volatility/quadratic_parameter.rs', 'crates/pricing/src/engine/calibration/lsv/rough_families/quadratic_parameter.rs', 'crates/pricing/src/engine/risk/lsv/rough_families/quadratic_parameter.rs', 'crates/pricing/src/engine/risk/lsv/rough_families/quadratic_parameter_tests.rs', 'crates/pricing/tests/quadratic_heston_parameter_risk.rs', 'tests/python/test_quadratic_heston_parameter_risk.py', 'scripts/check_quadratic_heston_reference.py', 'scripts/test_quadratic_heston_contract.py', 'fixtures/rough-volatility/quadratic-parameter.json', 'docs/models/quadratic-heston-parameter-risk.md', 'design/validation/quadratic-heston-parameter-risk.md', 'examples/python/quadratic_heston_parameter_risk.py', '.github/workflows/quadratic-heston-parameter-risk.yml']
class QuadraticContract(unittest.TestCase):
    def test_public_api_is_required(self):
        text=(ROOT/'rust_pricing.pyi').read_text()
        wheel.exported_stub_api(text.encode())
        for name in ['RoughVolatilityPlan','RoughFamilyLsvPlan']:
            tree=ast.parse(text)
            cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==name)
            cls.body=[x for x in cls.body if not isinstance(x,ast.FunctionDef) or x.name!='evaluate_quadratic_heston_parameter_risk']
            with self.assertRaises(RuntimeError):wheel.verify_stub_static_shape(tree)
        for name in ['QuadraticHestonMcParameterRisk','QuadraticHestonLsvParameterRisk']:
            tree=ast.parse(text)
            tree.body=[x for x in tree.body if not isinstance(x,ast.ClassDef) or x.name!=name]
            with self.assertRaises(RuntimeError):wheel.verify_stub_static_shape(tree)

    def test_source_and_evidence_gates_are_present(self):
        for path in FILES:
            self.assertIn(path,archive.REQUIRED_FILES)
            self.assertTrue((ROOT/path).is_file(),path)
        text=(ROOT/'.github/workflows/quadratic-heston-parameter-risk.yml').read_text()
        for s in ['check_quadratic_heston_reference.py','test_quadratic_heston_contract.py','--lib quadratic_',
                  '--no-default-features','--test quadratic_heston_parameter_risk -- --include-ignored --nocapture',
                  'if-no-files-found: error','RUSTDOCFLAGS: -D warnings']:
            self.assertIn(s,text)
        protocol=(ROOT/'design/validation/quadratic-heston-parameter-risk.md').read_text()
        for s in ['896 path comparisons','42 calibration comparisons','168 fully rebuilt price comparisons','abs(gap)<=5SE+.003','<=.2']:
            self.assertIn(s,protocol)

if __name__=='__main__':unittest.main()
