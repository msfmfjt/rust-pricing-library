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
                    self.assertGreater(record["selected"], 0)
                    if name == "caught":
                        self.assertEqual(record["status"], "passed")
                        self.assertEqual(record["counts"]["caught"], record["selected"])
                    elif name == "missed":
                        self.assertEqual(record["status"], "failed")
                        self.assertEqual(record["counts"]["missed"], record["selected"])
                    else:
                        self.assertNotEqual(record["status"], "passed")


if __name__ == "__main__":
    unittest.main()
