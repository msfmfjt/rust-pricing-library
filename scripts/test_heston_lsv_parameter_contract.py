"""Retain explicit fixed-target risk contracts and executable numerical gates."""
import ast
from pathlib import Path
import unittest
import check_source_archive as archive
import smoke_test_wheel as wheel
ROOT=Path(__file__).resolve().parents[1]
class Contract(unittest.TestCase):
    def test_required_source_and_workflow_gates(self):
        for path in ['crates/pricing/src/engine/calibration/lsv/rough_families/heston_parameter.rs',
                     'crates/pricing/src/engine/risk/lsv/rough_families/heston_parameter_tests.rs',
                     'crates/pricing/tests/heston_lsv_parameter_risk.rs',
                     'tests/python/test_heston_lsv_parameter_risk.py','examples/python/heston_lsv_parameter_risk.py']:
            self.assertIn(path,archive.REQUIRED_FILES);self.assertTrue((ROOT/path).is_file())
        text=(ROOT/'.github/workflows/heston-lsv-parameter-risk.yml').read_text()
        gates=['contents: read','os: [ubuntu-24.04, macos-15, windows-2025]',
               'cargo test --locked -p pricing --test heston_lsv_parameter_risk',
               'cargo test --locked --no-default-features -p pricing --test heston_lsv_parameter_risk',
               'cargo test --locked -p pricing --lib total_scramble_errors_preserve_direct_calibration_covariance',
               'cargo test --locked --release -p pricing --test heston_lsv_parameter_risk -- --include-ignored --nocapture',
               'path: heston-lsv-parameter-risk.log','if-no-files-found: error']
        def check(t):
            if any(g not in t for g in gates):raise ValueError('missing gate')
        check(text)
        for g in gates:
            with self.assertRaises(ValueError):check(text.replace(g,'removed'))
    def test_api_removals_fail(self):
        text=(ROOT/'rust_pricing.pyi').read_text();wheel.exported_stub_api(text.encode())
        for cls,member in [('RoughFamilyLsvPlan','evaluate_heston_parameter_risk'),('HestonLsvParameterRisk','parameter_adjoints'),
                           ('HestonLsvParameterRisk','calibration_adjoints'),('HestonLsvParameterRisk','direct_adjoints'),
                           ('HestonLsvParameterRisk','standard_errors')]:
            t=ast.parse(text);c=next(c for c in t.body if isinstance(c,ast.ClassDef) and c.name==cls)
            c.body=[n for n in c.body if not isinstance(n,ast.FunctionDef) or n.name!=member]
            with self.assertRaises(RuntimeError):wheel.verify_stub_static_shape(t)
    def test_hurst_is_explicit_and_defaults_to_fixed_kernel(self):
        tree=ast.parse((ROOT/'rust_pricing.pyi').read_text())
        cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='RoughFamilyLsvPlan')
        method=next(m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name=='evaluate_heston_parameter_risk')
        self.assertEqual([a.arg for a in method.args.kwonlyargs],['include_hurst'])
        self.assertIs(method.args.kw_defaults[0].value,False)
        method.args.kw_defaults[0]=ast.Constant(value=True)
        with self.assertRaises(RuntimeError):wheel.verify_stub_static_shape(tree)
if __name__=='__main__':unittest.main()
