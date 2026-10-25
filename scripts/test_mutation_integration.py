"""Exercise the installed cargo-mutants tool on a tiny downstream-only test fixture.

Run after installing the pinned tool: python -m unittest discover -s scripts
-p 'test_mutation_integration.py' -v. No project pricing kernels are changed.
"""

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_mutation_tests as runner


class InstalledToolTests(unittest.TestCase):
    def test_downstream_detection_survivors_and_failed_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["crates/pricing", "crates/pricing-numerics"]\nresolver = "3"\n',
                encoding="utf-8",
            )
            (root / ".gitignore").write_text("/target/\n", encoding="utf-8")
            (root / ".cargo").mkdir()
            shutil.copyfile(runner.ROOT / ".cargo/mutants.toml", root / ".cargo/mutants.toml")
            for name in runner.PACKAGES:
                package = root / "crates" / name
                (package / "src").mkdir(parents=True)
                manifest = f'[package]\nname = "{name}"\nversion = "0.1.0"\nedition = "2024"\n'
                if name == "pricing":
                    manifest += '[dependencies]\npricing-numerics = { path = "../pricing-numerics" }\n'
                (package / "Cargo.toml").write_text(manifest, encoding="utf-8")
                (package / "src/lib.rs").write_text(
                    "pub use pricing_numerics::twice;\n" if name == "pricing"
                    else "pub fn twice(x: u64) -> u64 { x * 2 }\n", encoding="utf-8",
                )
            consumer = root / "crates/pricing/tests/consumer.rs"
            consumer.parent.mkdir()
            consumer.write_text("#[test]\nfn consumer() { assert_eq!(pricing::twice(3), 6); }\n", encoding="utf-8")
            for command in (
                ["cargo", "generate-lockfile"], ["git", "init", "-b", "main"],
                ["git", "add", "."],
                ["git", "-c", "user.name=Mutation fixture", "-c", "user.email=mutation@example.invalid",
                 "commit", "-m", "fixture"],
            ):
                subprocess.run(command, cwd=root, check=True, capture_output=True)

            for name, body, expected_exit in (
                ("caught", "assert_eq!(pricing::twice(3), 6);", 0),
                ("missed", "let _ = pricing::twice(3);", 2),
                ("baseline_failed", "assert_eq!(pricing::twice(3), 7);", 4),
            ):
                with self.subTest(name=name):
                    consumer.write_text(f"#[test]\nfn consumer() {{ {body} }}\n", encoding="utf-8")
                    args = runner.parse_args(["full", "--package", "pricing-numerics",
                                              "--jobs", "1", "--output", f"target/{name}"])
                    with patch.dict(runner.os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                            contextlib.redirect_stdout(io.StringIO()) as captured:
                        code = runner.run(args, root)
                    record_path = next((root / f"target/{name}").glob("*/run.json"))
                    record = json.loads(record_path.read_text())
                    self.assertEqual(code, expected_exit, captured.getvalue())
                    self.assertEqual(record["runner_exit_code"], code)
                    self.assertLessEqual(record["started_at"], record["finished_at"])
                    self.assertGreater(record["selected"], 0)
                    if name == "caught":
                        self.assertEqual(record["status"], "passed")
                        self.assertEqual(record["counts"]["caught"], record["selected"])
                        self.assertEqual(record["outstanding"], [])
                    elif name == "missed":
                        self.assertEqual(record["status"], "failed")
                        self.assertEqual(record["counts"]["missed"], record["selected"])
                        self.assertEqual(len(record["outstanding"]), record["selected"])
                        for entry in record["outstanding"]:
                            self.assertEqual((entry["outcome"], entry["phase"]), ("MissedMutant", "Test"))
                            self.assertTrue((record_path.parent / "mutants.out" / entry["log_path"]).is_file())
                            self.assertTrue((record_path.parent / "mutants.out" / entry["diff_path"]).is_file())
                    else:
                        self.assertNotEqual(record["status"], "passed")
                        self.assertNotIn("outstanding", record)

            # Fewer than 100 mutants exist in this fixture. An empty shard is
            # an explicit no-op only when the unsharded selection is valid and
            # nonempty. The still-broken consumer must not be run for this shard.
            args = runner.parse_args(["full", "--package", "pricing-numerics",
                                      "--shard", "99/100", "--output", "target/empty_shard"])
            with patch.dict(runner.os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                    contextlib.redirect_stdout(io.StringIO()) as captured:
                code = runner.run(args, root)
            record_path = next((root / "target/empty_shard").glob("*/run.json"))
            record = json.loads(record_path.read_text())
            self.assertEqual(code, 0, captured.getvalue())
            self.assertEqual(record["status"], "empty_shard")
            self.assertEqual(record["selected"], 0)
            self.assertNotIn("cargo_mutants_exit_code", record)
            self.assertGreater(len(json.loads(record_path.with_name("unsharded.json").read_text())), 0)

            # A behavior-preserving source edit creates a real changed-line
            # selection. Both shards must retain the downstream-only observer,
            # and their disjoint union must equal the unsharded selection.
            consumer.write_text("#[test]\nfn consumer() { assert_eq!(pricing::twice(3), 6); }\n", encoding="utf-8")
            (root / "crates/pricing-numerics/src/lib.rs").write_text(
                "pub fn twice(x: u64) -> u64 { x + x }\n", encoding="utf-8")
            unsharded_output = root / "target/changed_all"
            unsharded_output.mkdir(parents=True)
            args = runner.parse_args(["changed", "--base", "HEAD", "--package", "pricing-numerics"])
            selection = runner.selection_args(args, root, unsharded_output)
            unsharded = json.loads(runner.capture(["cargo", "mutants", *selection, "--list", "--json"], root))
            all_names = {mutant["name"] for mutant in unsharded}
            self.assertGreater(len(all_names), 1)
            shard_names = []
            for shard in range(2):
                args = runner.parse_args(["changed", "--base", "HEAD", "--package", "pricing-numerics",
                                          "--shard", f"{shard}/2", "--jobs", "1",
                                          "--output", f"target/changed_{shard}"])
                with patch.dict(runner.os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                        contextlib.redirect_stdout(io.StringIO()) as captured:
                    code = runner.run(args, root)
                record_path = next((root / f"target/changed_{shard}").glob("*/run.json"))
                record = json.loads(record_path.read_text())
                self.assertEqual(code, 0, captured.getvalue())
                self.assertEqual(record["status"], "passed")
                self.assertGreater(record["selected"], 0)
                self.assertEqual(record["counts"]["caught"], record["selected"])
                inventory = json.loads(record_path.with_name("selected.json").read_text())
                shard_names.append({mutant["name"] for mutant in inventory})
            self.assertFalse(shard_names[0] & shard_names[1])
            self.assertEqual(shard_names[0] | shard_names[1], all_names)

            # Losing the downstream assertion must still fail a changed shard.
            consumer.write_text("#[test]\nfn consumer() { let _ = pricing::twice(3); }\n", encoding="utf-8")
            args = runner.parse_args(["changed", "--base", "HEAD", "--package", "pricing-numerics",
                                      "--shard", "1/2", "--jobs", "1", "--output", "target/changed_missed"])
            with patch.dict(runner.os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                    contextlib.redirect_stdout(io.StringIO()) as captured:
                code = runner.run(args, root)
            record_path = next((root / "target/changed_missed").glob("*/run.json"))
            record = json.loads(record_path.read_text())
            self.assertEqual(code, 2, captured.getvalue())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["counts"]["missed"], record["selected"])


if __name__ == "__main__":
    unittest.main()
