"""Freeze the provider panels/ledgers while checking their CI partition."""
from collections import Counter
from contextlib import redirect_stdout
import io
from itertools import product
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from wheel_validation import provider_shards as shards


class ProviderValidationInventoryTests(unittest.TestCase):
    def test_actual_provider_cases_match_frozen_domains_and_partition(self):
        named_products = ("call", "asian_call", "fixed_asian_call", "lookback_call", "put", "asian_put",
                          "fixed_asian_put", "lookback_put", "lookback_past_put", "lookback_today_put")
        kinds = ("arithmetic_asian", "fixed_lookback")
        for spec in shards.SUITES:
            _, cls = shards.load_suite(spec, ROOT / "tests/python")
            if spec.name in ("custom-basis", "coupled-basis"):
                policies = (("row_hats", "weighted_time_hats", "coupled_row_hats")
                            if spec.name == "custom-basis" else ("coupled_row_hats",))
                expected = {
                    "test_a_native_panels": list(product(policies, (4, 6), named_products)),
                    "test_b_dated_ledgers": list(product(policies, (4, 6), ("call", "put"), kinds)),
                }
            else:
                extensions = ("all_rows", "linear_time")
                if spec.name == "extension":
                    methods = ("test_a_native_panels_and_covariance", "test_b_fixing_day_ledgers", "test_c_six_interval_ledgers")
                    products = ("call", "asian", "fixed_asian", "lookback")
                else:
                    methods = ("test_a_native_put_panels", "test_b_fixing_day_mixed_ledgers", "test_c_six_interval_mixed_ledgers")
                    products = ("put", "asian", "fixed_asian", "lookback", "lookback_past", "lookback_today")
                expected = {methods[0]: list(product(extensions, (4, 6), products)),
                            methods[1]: list(product(extensions, kinds)),
                            methods[2]: list(product(extensions, kinds))}
            # Order matters to the legacy acceptance scripts as well as membership.
            self.assertEqual(cls.wheel_cases(), expected)
        with redirect_stdout(io.StringIO()):
            report = shards.verify_partition(ROOT / "tests/python")
        self.assertEqual(Counter(row["module"] for row in report.values()), {
            "test_vega_kt_custom_basis_provider": 7, "test_vega_kt_coupled_basis_provider": 3,
            "test_vega_kt_extension_provider": 5, "test_vega_kt_extension_put_provider": 5,
        })
        self.assertEqual(sum(case["count"] for row in report.values() for case in row["cases"]), 179)


if __name__ == "__main__":
    unittest.main()
