"""Synthetic guard tests for the matched benchmark comparator; no timing claims."""
import copy
import unittest

from compare_rough_diffusion_benchmarks import SCHEMA, EXPECTED, compare


def data():
    return {"schema": SCHEMA, "seed": 91, "scope": "synthetic guard inputs",
            "rows": [dict(zip(("family", "hurst", "steps", "grid"), key),
                          paths=128, seconds=[1., 2., 3.], digest="0"*64,
                          plan_fingerprint="blake3-256:"+"1"*64, scheme="same", negative_nodes=7)
                     for key in sorted(EXPECTED)]}


class BenchmarkContracts(unittest.TestCase):
    def test_matched_comparison_without_speed_threshold(self):
        a, b = data(), data()
        for row in b["rows"]:
            row["seconds"] = [2., 4., 6.]
        b["rows"].reverse()
        result = compare(a, b)
        self.assertEqual(result["bitwise_equal_cases"], 36)
        self.assertEqual(result["geometric_mean_speedup"], .5)
        self.assertFalse(result["timing_gate"])

    def test_rejects_identity_changes(self):
        for field, value in [("paths",129),("digest","2"*64),
                             ("plan_fingerprint","blake3-256:"+"3"*64),("scheme","other"),
                             ("negative_nodes",8)]:
            with self.subTest(field=field):
                a, b = data(), data()
                b["rows"][0][field] = value
                with self.assertRaises(ValueError):
                    compare(a,b)

    def test_rejects_missing_duplicate_or_bad_measurements(self):
        for modification in [lambda d:d["rows"].pop(),
                             lambda d:d["rows"].append(copy.deepcopy(d["rows"][0])),
                             lambda d:d.update(seed=1973),
                             lambda d:d["rows"][0].update(seconds=[]),
                             lambda d:d["rows"][0].update(seconds=[float("nan")]),
                             lambda d:d["rows"][0].update(seconds=[0.]),
                             lambda d:d["rows"][0].update(paths=0)]:
            a, b = data(), data()
            modification(b)
            with self.assertRaises(ValueError):
                compare(a,b)

    def test_repository_registers_guards(self):
        from pathlib import Path
        from check_source_archive import REQUIRED_FILES
        root = Path(__file__).resolve().parents[1]
        workflow = (root/".github/workflows/rough-volatility.yml").read_text()
        for command in ["cargo test --locked -p pricing --lib diffusion_cache",
                        "cargo test --locked --no-default-features -p pricing --lib diffusion_cache",
                        "cargo test --locked --release -p pricing --lib diffusion_cache",
                        "test_rough_diffusion_benchmark.py"]:
            self.assertIn(command, workflow)
        self.assertIn("crates/pricing/src/engine/processes/rough_volatility/cache_tests.rs", REQUIRED_FILES)
        self.assertIn("crates/pricing/examples/benchmark_rough_diffusion_cache.rs", REQUIRED_FILES)
