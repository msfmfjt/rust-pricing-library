"""Keep the release-wheel API gate strict as the Bass facade is extended."""

import ast
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("wheel_gate", ROOT / "scripts/smoke_test_wheel.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class WheelContractTests(unittest.TestCase):
    def test_current_stub_and_runtime_export_contract(self):
        stub = (ROOT / "rust_pricing.pyi").read_bytes()
        api = gate.exported_stub_api(stub)
        self.assertIn("BassMarketIvModel", api["symbols"])
        self.assertIn("bass_local_volatility", api["class_members"]["Model"])

    def test_bass_contract_rejects_missing_extra_and_misdecorated_api(self):
        original = ast.parse((ROOT / "rust_pricing.pyi").read_text())
        for mutation in ("missing_class", "missing_method", "extra_method", "wrong_decorator"):
            tree = copy.deepcopy(original)
            model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BassLvModel")
            if mutation == "missing_class":
                tree.body.remove(model)
            elif mutation == "missing_method":
                model.body = [n for n in model.body if not isinstance(n, ast.FunctionDef) or n.name != "calibrate"]
            elif mutation == "extra_method":
                model.body.append(ast.parse("def unexpected(self) -> float: ...").body[0])
            else:
                method = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "calibrate")
                method.decorator_list = []
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                gate.verify_stub_static_shape(tree)


if __name__ == "__main__":
    unittest.main()
