"""Guard Hurst API/packaging and retain existing scalar-risk gates."""
import ast
from pathlib import Path
import unittest
import check_source_archive as archive
import smoke_test_wheel as wheel
ROOT=Path(__file__).resolve().parents[1]
class Contract(unittest.TestCase):
    def test_required_sources_and_ci_gates(self):
        for name in ['fixtures/rough-volatility/mc-hurst.json','scripts/check_heston_mc_hurst_reference.py',
                     'crates/pricing/src/engine/processes/rough_volatility/hurst.rs',
                     'crates/pricing/src/engine/risk/rough_volatility/hurst.rs',
                     'tests/python/test_heston_mc_hurst_risk.py','examples/python/heston_mc_hurst_risk.py']:
            self.assertIn(name,archive.REQUIRED_FILES);self.assertTrue((ROOT/name).is_file())
        text=(ROOT/'.github/workflows/heston-mc-parameter-risk.yml').read_text()
        gates=['mpmath>=1.3,<2','python scripts/check_heston_mc_hurst_reference.py',
               'cargo test --locked -p pricing --test heston_mc_hurst_risk',
               'cargo test --locked --no-default-features -p pricing --test heston_mc_hurst_risk',
               'cargo test --locked --release -p pricing --test heston_mc_hurst_risk -- --include-ignored --nocapture',
               'path: heston-mc-hurst-risk.log','if-no-files-found: error','contents: read',
               'cargo test --locked --release -p pricing --test heston_mc_parameter_risk -- --include-ignored --nocapture']
        def check(t):
            if any(g not in t for g in gates):raise ValueError('missing gate')
        check(text)
        for g in gates:
            with self.assertRaises(ValueError):check(text.replace(g,'removed'))
    def test_required_api_removal_fails(self):
        text=(ROOT/'rust_pricing.pyi').read_text();wheel.exported_stub_api(text.encode())
        for cls,member in [('RoughVolatilityPlan','evaluate_hurst_risk'),('RoughHestonMcHurstRisk','hurst_sensitivity'),('RoughHestonMcHurstRisk','standard_error')]:
            t=ast.parse(text);c=next(c for c in t.body if isinstance(c,ast.ClassDef) and c.name==cls)
            c.body=[n for n in c.body if not isinstance(n,ast.FunctionDef) or n.name!=member]
            with self.assertRaises(RuntimeError):wheel.verify_stub_static_shape(t)
    def test_reference_matches_high_precision_recalculation(self):
        from check_heston_mc_hurst_reference import check
        self.assertEqual(len(check()['black']),3)
if __name__=='__main__':unittest.main()
