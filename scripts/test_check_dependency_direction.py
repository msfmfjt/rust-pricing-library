"""Exercise source-boundary checks under non-UTF-8 platform defaults."""

from contextlib import contextmanager
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import check_dependency_direction as checker


@contextmanager
def default_text_encoding(default_encoding):
    """Model Windows cp1252 (or an ASCII locale) without requiring that OS."""
    real_open = io.open

    def open_with_default(file, mode="r", buffering=-1, encoding=None,
                          errors=None, newline=None, closefd=True, opener=None):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = default_encoding
        return real_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    with patch("io.open", side_effect=open_with_default):
        yield


class SourceEncodingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "crates/pricing/src"
        self.files = {
            "lib.rs": "mod engine;\n",
            "market/input.rs": "pub struct Input;\n",
            "engine/mc/execute.rs": "pub fn execute() {}\n",
        }
        self.write_sources()

    def write_sources(self):
        for name, code in self.files.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code, encoding="utf-8")

    def check(self):
        errors = []
        checker.check_internal_boundaries(self.root, errors)
        return errors

    def test_utf8_source_ignores_platform_default_encoding(self):
        # Match the full-width date input that triggered the Windows CI failure.
        unicode_source = 'const INVALID_DATE: &str = "２０２４-02-29";\n'
        for encoding in ("ascii", "cp1252"):
            with self.subTest(encoding=encoding), self.assertRaises(UnicodeDecodeError):
                unicode_source.encode("utf-8").decode(encoding)
            for name, original in self.files.items():
                with self.subTest(encoding=encoding, path=name):
                    self.write_sources()
                    path = self.source / name
                    path.write_text(original + unicode_source, encoding="utf-8")
                    with default_text_encoding(encoding):
                        self.assertEqual(self.check(), [])

    def test_utf8_comments_do_not_create_boundary_violations(self):
        for name, original in self.files.items():
            (self.source / name).write_text(
                original + "// 入力: crate::engine::SoaWorkspace RiskReport VegaKtResult\n",
                encoding="utf-8",
            )
        with default_text_encoding("cp1252"):
            self.assertEqual(self.check(), [])

    def test_utf8_source_still_rejects_each_forbidden_boundary(self):
        cases = [
            ("lib.rs", "pub mod engine;", "engine module must remain private"),
            ("market/input.rs", "use crate::engine::SoaWorkspace;",
             f"domain definition imports execution state: {Path('market/input.rs')}"),
            ("engine/mc/execute.rs", "use crate::engine::risk::RiskReport;",
             f"lower execution layer assembles risk reports: {Path('engine/mc/execute.rs')}"),
        ]
        for name, violation, expected in cases:
            with self.subTest(path=name):
                self.write_sources()
                (self.source / name).write_text(
                    "// 境界検証\n" + violation + "\n", encoding="utf-8",
                )
                with default_text_encoding("cp1252"):
                    self.assertEqual(self.check(), [expected])

    def test_invalid_utf8_is_not_silently_replaced_or_ignored(self):
        for name in self.files:
            with self.subTest(path=name):
                self.write_sources()
                (self.source / name).write_bytes(b"// invalid byte: \xff\n")
                with default_text_encoding("cp1252"), self.assertRaises(UnicodeDecodeError):
                    self.check()


if __name__ == "__main__":
    unittest.main()
