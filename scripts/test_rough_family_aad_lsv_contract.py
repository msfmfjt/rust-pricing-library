"""Archive/CI/stub mutation guards; no runtime extension is imported."""
import ast
import io
from pathlib import Path
import tarfile
import unittest
import check_source_archive as archive
import smoke_test_wheel as wheel

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = {'crates/pricing/src/engine/risk/lsv/rough_families.rs', 'scripts/test_rough_family_aad_lsv_contract.py', 'crates/pricing/src/engine/risk/rough_volatility/delta.rs', 'crates/pricing/src/engine/processes/rough_volatility/lsv.rs', '.github/workflows/rough-family-aad-lsv.yml', 'examples/python/rough_family_aad_lsv.py', 'crates/pricing/tests/rough_family_aad_lsv.rs', 'design/validation/rough-family-aad-lsv.md', 'crates/pricing/src/engine/calibration/lsv/rough_families.rs', 'docs/models/rough-family-aad-lsv.md', 'tests/python/test_rough_family_aad_lsv.py', 'crates/pricing/src/engine/processes/rough_volatility/reverse.rs'}

def workflow_check(workflow, smoke):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as package:
        for path, text in [(".github/workflows/rough-family-aad-lsv.yml", workflow),
                           ("scripts/smoke_test_wheel.py", smoke)]:
            body = text.encode()
            entry = tarfile.TarInfo(path)
            entry.size = len(body)
            package.addfile(entry, io.BytesIO(body))
    stream.seek(0)
    with tarfile.open(fileobj=stream, mode="r") as package:
        archive.check_rough_family_aad_lsv_workflow(package, "mutation.tar")

class RoughFamilyContractTests(unittest.TestCase):
    def test_workflow_and_all_gates(self):
        wf = (ROOT/".github/workflows/rough-family-aad-lsv.yml").read_text()
        smoke = (ROOT/"scripts/smoke_test_wheel.py").read_text()
        workflow_check(wf, smoke)
        for gate in ['contents: read', "python -m unittest discover -s scripts -p 'test_rough_family_aad_lsv_contract.py'", 'cargo test --locked -p pricing --test rough_family_aad_lsv', 'cargo test --locked --no-default-features -p pricing --test rough_family_aad_lsv', 'cargo test --locked --release -p pricing --test rough_family_aad_lsv -- --include-ignored --nocapture', 'os: [ubuntu-24.04, macos-15, windows-2025]', 'set -o pipefail', 'name: rough-family-aad-lsv-${{ matrix.os }}', 'path: rough-family-aad-lsv.log', 'if-no-files-found: error']:
            with self.subTest(gate=gate), self.assertRaises(SystemExit):
                workflow_check(wf.replace(gate, "REMOVED"), smoke)
        with self.assertRaises(SystemExit):
            workflow_check(wf, smoke.replace('"examples/python/rough_family_aad_lsv.py"', '"REMOVED"'))

    def test_source_members_required_and_present(self):
        self.assertTrue(MEMBERS <= archive.REQUIRED_FILES)
        for member in MEMBERS:
            self.assertTrue((ROOT/member).is_file(), member)

    def test_explicit_risk_api_contract(self):
        source = (ROOT/"rust_pricing.pyi").read_text()
        wheel.exported_stub_api(source.encode())
        for name in ["RoughVolatilityDelta", "RoughFamilyLsvPlan"]:
            tree = ast.parse(source)
            tree.body = [n for n in tree.body if not isinstance(n, ast.ClassDef) or n.name != name]
            with self.assertRaises(RuntimeError):
                wheel.verify_stub_static_shape(tree)

    def test_numerical_and_semantic_protocol_retained(self):
        source = (ROOT/"crates/pricing/tests/rough_family_aad_lsv.rs").read_text()
        for required in ["six_family_black_limit_price_delta_and_lsv_node_risk", "1973",
                         "7.965567455405804", "0.539827837277029", "99.23813686925295",
                         "public_lsv_native_rqmc_layout_matches_independent_payoff_reconstruction",
                         "lsv_keeps_future_cash_reserve_without_extending_the_target_horizon"]:
            self.assertIn(required, source)
        docs = (ROOT/"docs/models/rough-family-aad-lsv.md").read_text()
        self.assertIn("Local variance nodes are not market implied-volatility quotes", docs)
        self.assertIn("fixed-driver LSV extension", docs)
        self.assertIn("Two explicit high-level LSV Spot Delta conventions", docs)

if __name__ == "__main__":
    unittest.main()
