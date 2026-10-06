"""Guard new API, declared method scope, source inventory and numerical CI gates."""
import ast
from pathlib import Path
import unittest
import check_source_archive as archive
import smoke_test_wheel as wheel
ROOT=Path(__file__).resolve().parents[1]
GATES=['cargo test --locked -p pricing --test rough_family_gamma_bump',
       'cargo test --locked --no-default-features -p pricing --test rough_family_gamma_bump',
       'cargo test --locked --release -p pricing --test rough_family_gamma_bump -- --include-ignored --nocapture',
       'path: rough-family-gamma.log','if-no-files-found: error','set -o pipefail',
       'os: [ubuntu-24.04, macos-15, windows-2025]','contents: read']
def gates(text):
    if any(x not in text for x in GATES):raise ValueError('missing Gamma gate')
class GammaContract(unittest.TestCase):
    def test_source_and_ci(self):
        for x in ['crates/pricing/src/engine/risk/gamma_bump.rs','crates/pricing/tests/rough_family_gamma_bump.rs',
                  'tests/python/test_rough_family_gamma_bump.py','docs/models/rough-family-gamma.md',
                  'examples/python/rough_family_gamma_bump.py','.github/workflows/rough-family-gamma.yml']:
            self.assertIn(x,archive.REQUIRED_FILES);self.assertTrue((ROOT/x).is_file())
        text=(ROOT/'.github/workflows/rough-family-gamma.yml').read_text();gates(text)
        for x in GATES:
            with self.assertRaises(ValueError):gates(text.replace(x,'REMOVED'))
    def test_stub_removal_rejected(self):
        text=(ROOT/'rust_pricing.pyi').read_text();wheel.exported_stub_api(text.encode())
        for cls,method in [('RoughVolatilityPlan','evaluate_gamma_bump'),
                           ('RoughFamilyLsvPlan','evaluate_frozen_leverage_gamma_bump'),
                           ('RoughFamilyLsvPlan','evaluate_sticky_moneyness_gamma_bump'),
                           ('RoughVolatilityGamma','half_bump_gamma'),
                           ('RoughFamilyLsvGamma','bump_difference_standard_error')]:
            t=ast.parse(text);c=next(c for c in t.body if isinstance(c,ast.ClassDef) and c.name==cls)
            c.body=[v for v in c.body if not isinstance(v,ast.FunctionDef) or v.name!=method]
            with self.assertRaises(RuntimeError):wheel.verify_stub_static_shape(t)
    def test_numerical_policy_and_disclosure(self):
        t=(ROOT/'crates/pricing/tests/rough_family_gamma_bump.rs').read_text()
        for x in ['0.01984762737385059','0.019844113046403322','4096','1973','se <= 0.002','covariance_effect > 1e-4']:
            self.assertIn(x,t)
        d=(ROOT/'docs/models/rough-family-gamma.md').read_text()
        for x in ['not second-order AAD','not a remaining-bias bound','payoff_evaluations','h/S>=1e-5']:
            self.assertIn(x,d)
if __name__=='__main__':unittest.main()
