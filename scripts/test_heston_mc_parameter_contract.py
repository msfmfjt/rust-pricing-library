"""New parameter risk contract, source completeness, and numerical CI gates."""
import ast
from pathlib import Path
import unittest
import smoke_test_wheel as wheel
import check_source_archive as archive
ROOT=Path(__file__).resolve().parents[1]
GATES=["cargo test --locked -p pricing --test heston_mc_parameter_risk",
 "cargo test --locked --no-default-features -p pricing --test heston_mc_parameter_risk",
 "cargo test --locked --release -p pricing --test heston_mc_parameter_risk -- --include-ignored --nocapture",
 "os: [ubuntu-24.04, macos-15, windows-2025]","contents: read","if-no-files-found: error",
 "set -o pipefail","path: heston-mc-parameter-risk.log"]
class Contract(unittest.TestCase):
    def test_required_sources_and_gates(self):
        for name in ['crates/pricing/src/engine/processes/rough_volatility/heston_parameter.rs',
                     'crates/pricing/src/engine/risk/rough_volatility/heston_parameter.rs',
                     'tests/python/test_heston_mc_parameter_risk.py','docs/models/heston-mc-parameter-risk.md',
                     'examples/python/heston_mc_parameter_risk.py','.github/workflows/heston-mc-parameter-risk.yml']:
            self.assertIn(name,archive.REQUIRED_FILES);self.assertTrue((ROOT/name).is_file())
        text=(ROOT/'.github/workflows/heston-mc-parameter-risk.yml').read_text()
        def check(t):
            if any(g not in t for g in GATES):raise ValueError('missing parameter risk gate')
        check(text)
        for g in GATES:
            with self.assertRaises(ValueError):check(text.replace(g,'removed'))
    def test_required_api_removal_fails(self):
        text=(ROOT/'rust_pricing.pyi').read_text();wheel.exported_stub_api(text.encode())
        for cls,m in [('RoughVolatilityPlan','evaluate_heston_parameter_risk'),('HestonMcParameterRisk','standard_errors'),('HestonMcParameterRisk','parameter_adjoints')]:
            t=ast.parse(text);c=next(c for c in t.body if isinstance(c,ast.ClassDef) and c.name==cls)
            c.body=[n for n in c.body if not isinstance(n,ast.FunctionDef) or n.name!=m]
            with self.assertRaises(RuntimeError):wheel.verify_stub_static_shape(t)
    def test_fixed_boundaries_and_reference_budget_retained(self):
        text=(ROOT/'crates/pricing/tests/heston_mc_parameter_risk.rs').read_text()
        for s in ['99.23813686925295','1973','4096','negative > 50','2e-5','3e-5']:
            self.assertIn(s,text)
        text=(ROOT/'docs/models/heston-mc-parameter-risk.md').read_text()
        for s in ['v0>0 and |rho|<1','marginal sampling errors','not volatility Vega','Exact zero']:
            self.assertIn(s,text)
if __name__=='__main__':unittest.main()
