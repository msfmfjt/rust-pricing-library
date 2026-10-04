"""Market-IV API inventory, source archive and CI regression guards."""
import ast
from pathlib import Path
import unittest
import check_source_archive as archive
import smoke_test_wheel as wheel
ROOT=Path(__file__).resolve().parents[1]
MEMBERS=['crates/pricing/src/engine/risk/lsv/rough_families/market_iv.rs', 'crates/pricing/tests/rough_family_market_iv.rs', 'scripts/test_rough_family_market_iv_contract.py', 'examples/python/rough_family_market_iv.py', 'docs/models/rough-family-market-iv.md', 'design/validation/rough-family-market-iv.md', 'tests/python/test_rough_family_market_iv.py']

GATES=["cargo test --locked -p pricing --test rough_family_market_iv",
       "cargo test --locked --no-default-features -p pricing --test rough_family_market_iv",
       "cargo test --locked --release -p pricing --test rough_family_market_iv -- --include-ignored --nocapture",
       "path: rough-family-market-iv.log", "if-no-files-found: error", "set -o pipefail",
       "os: [ubuntu-24.04, macos-15, windows-2025]", "contents: read"]
def check_gates(text):
    if any(g not in text for g in GATES):raise ValueError('missing market-IV gate')
class MarketIvContract(unittest.TestCase):
    def test_archive_and_ci_gates(self):
        self.assertTrue(set(MEMBERS)<=archive.REQUIRED_FILES)
        for m in MEMBERS:self.assertTrue((ROOT/m).is_file(),m)
        text=(ROOT/'.github/workflows/rough-family-aad-lsv.yml').read_text()
        check_gates(text)
        for g in GATES:
            with self.assertRaises(ValueError):check_gates(text.replace(g,'REMOVED'))
    def test_api_removal_is_detected(self):
        text=(ROOT/'rust_pricing.pyi').read_text()
        wheel.exported_stub_api(text.encode())
        for cls,member in [('RoughFamilyLsvPlan','market_iv_risk_plan'),
                           ('RoughFamilyLsvMarketIvRiskPlan','evaluate'),
                           ('RoughFamilyLsvMarketIvRisk','quote_adjoints'),
                           ('RoughFamilyLsvMarketIvRisk','parallel_standard_error')]:
            tree=ast.parse(text); c=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls)
            c.body=[v for v in c.body if not isinstance(v,ast.FunctionDef) or v.name!=member]
            with self.assertRaises(RuntimeError):wheel.verify_stub_static_shape(tree)
    def test_numerical_and_coordinate_contract(self):
        text=(ROOT/'crates/pricing/tests/rough_family_market_iv.rs').read_text()
        for x in ['39.69525474770118','1973','5e-7','3e-6','covariance_effect > 1e-4']:
            self.assertIn(x,text)
        doc=(ROOT/'docs/models/rough-family-market-iv.md').read_text()
        for x in ['bit-for-bit','parallel_vega','cross-node covariance','None','NOT included']:
            self.assertIn(x,doc)
if __name__=='__main__':unittest.main()
