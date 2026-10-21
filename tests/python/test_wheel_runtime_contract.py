"""Check the installed wheel, including behavior behind advertised defaults."""
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.wheel_validation.runtime import verify_runtime_api
from scripts.wheel_validation.stub import exported_stub_api


class InstalledWheelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = exported_stub_api((ROOT / 'rust_pricing.pyi').read_bytes())

    def test_every_installed_callable_matches_the_stub(self):
        count = verify_runtime_api(rp, self.api, rp.__version__)
        self.assertEqual(count, len(self.api['callables']))
        self.assertGreater(count, 300)

    def test_runtime_gate_rejects_declared_default_drift(self):
        api = copy.deepcopy(self.api)
        parameters = api['callables']['Engine.randomized_quasi_monte_carlo']['parameters']
        next(p for p in parameters if p['name'] == 'scramble_count')['default']['value'] = 8
        with self.assertRaisesRegex(RuntimeError, 'Engine.randomized_quasi_monte_carlo signature mismatch'):
            verify_runtime_api(rp, api, rp.__version__)

    def test_runtime_subprocess_works_outside_the_source_directory(self):
        # Model gates also import the checker by path; CWD must not choose it.
        code = '''import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts.wheel_validation.runtime import verify_runtime_symbols
from scripts.wheel_validation.stub import exported_stub_api
api = exported_stub_api((Path(sys.argv[1]) / 'rust_pricing.pyi').read_bytes())
verify_runtime_symbols(Path(sys.executable), api, sys.argv[2])
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-c', code, str(ROOT), rp.__version__],
                                    cwd=directory, text=True, capture_output=True, check=True)
        self.assertIn(f'verified callables={len(self.api["callables"])}', result.stdout)

    def test_advertised_rqmc_defaults_match_native_request_serialization(self):
        discount = rp.DiscountCurve(10, [0.0, 1.0], [1.0, 0.95])
        dividend = rp.DiscountCurve(11, [0.0, 1.0], [1.0, 0.98])
        product = rp.Product.european_vanilla(1, 2, '2027-09-04', 100.0, 1.0, 'call')
        market = rp.Market.equity(2, 1, 100.0, discount, dividend)
        model = rp.Model.black_scholes(0.2)
        risk = rp.RiskRequest()

        def request(engine):
            return rp.PricingRequest('2026-09-04', product, market, model, engine, risk)

        implicit = request(rp.Engine.randomized_quasi_monte_carlo(64, 42))
        explicit = request(rp.Engine.randomized_quasi_monte_carlo(
            64, 42, scramble_count=16, antithetic=False, brownian_bridge=True))
        self.assertEqual(implicit.to_json(), explicit.to_json())
        self.assertEqual(implicit.fingerprint, explicit.fingerprint)
        native = json.loads(implicit.to_json())['engine']
        signature = inspect.signature(rp.Engine.randomized_quasi_monte_carlo)
        self.assertEqual(native['scramble_count'], 16)
        self.assertEqual(signature.parameters['scramble_count'].default, native['scramble_count'])
        self.assertEqual(native['variance_reduction'], {'antithetic': False, 'brownian_bridge': True})

    def test_validation_issue_comparisons_accept_only_positional_values(self):
        with self.assertRaises(rp.ValidationError) as caught:
            rp.PricingRequest.from_json('{')
        issue = caught.exception.issues[0]
        self.assertTrue(issue.__eq__(issue))
        self.assertFalse(issue.__ne__(issue))
        for name in ('__eq__', '__ne__'):
            for keyword in ('value', 'other'):
                with self.subTest(method=name, keyword=keyword), self.assertRaises(TypeError):
                    getattr(issue, name)(**{keyword: issue})


if __name__ == '__main__':
    unittest.main()
