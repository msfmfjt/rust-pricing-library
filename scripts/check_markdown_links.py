"""Validate repository-local Markdown links used by docs."""

from __future__ import annotations

from pathlib import Path
import re
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_ROOTS = [ROOT / "README.md", ROOT / "docs", ROOT / "fixtures"]
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def main() -> int:
    missing = []
    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        for target in LINK_PATTERN.findall(text):
            normalized = local_target(target)
            if normalized is None:
                continue
            resolved = (path.parent / normalized).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                missing.append((path, target, "escapes repository root"))
                continue
            if not resolved.exists():
                missing.append((path, target, "target does not exist"))

    if missing:
        for path, target, reason in missing:
            print(f"{path.relative_to(ROOT)}: broken link {target!r}: {reason}", file=sys.stderr)
        return 1

    print("Markdown local links are valid")
    return 0


def markdown_files() -> list[Path]:
    files = []
    for root in MARKDOWN_ROOTS:
        if root.is_file():
            files.append(root)
        else:
            files.extend(sorted(root.rglob("*.md")))
    return sorted(files)


def local_target(target: str) -> str | None:
    target = target.strip()
    if not target or target.startswith("#"):
        return None
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
        return None
    without_anchor = target.split("#", 1)[0]
    without_query = without_anchor.split("?", 1)[0]
    if not without_query:
        return None
    return unquote(without_query)


if __name__ == "__main__":
    raise SystemExit(main())
