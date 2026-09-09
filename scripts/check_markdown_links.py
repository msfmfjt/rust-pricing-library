"""Validate repository-local Markdown links used by docs."""

from __future__ import annotations

from pathlib import Path
import re
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_ROOTS = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "docs", ROOT / "fixtures"]
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def main() -> int:
    missing = []
    anchor_cache: dict[Path, set[str]] = {}
    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        for target in LINK_PATTERN.findall(text):
            normalized = local_target(path, target)
            if normalized is None:
                continue
            resolved, anchor = normalized
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                missing.append((path, target, "escapes repository root"))
                continue
            if not resolved.exists():
                missing.append((path, target, "target does not exist"))
                continue
            anchors = anchor_cache.get(resolved)
            if anchors is None:
                anchors = markdown_anchors(resolved)
                anchor_cache[resolved] = anchors
            if anchor is not None and anchor not in anchors:
                missing.append((path, target, "anchor does not exist"))

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


def local_target(path: Path, target: str) -> tuple[Path, str | None] | None:
    target = target.strip()
    if not target:
        return None
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
        return None
    before_anchor, separator, raw_anchor = target.partition("#")
    without_query = before_anchor.split("?", 1)[0]
    resolved = path if not without_query else (path.parent / unquote(without_query)).resolve()
    anchor = unquote(raw_anchor) if separator else None
    return resolved, anchor


def markdown_anchors(path: Path) -> set[str]:
    if path.suffix.lower() != ".md":
        return set()
    anchors: set[str] = set()
    heading_counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", line)
        if match is None:
            continue
        base = github_heading_slug(match.group(2))
        count = heading_counts.get(base, 0)
        heading_counts[base] = count + 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def github_heading_slug(heading: str) -> str:
    lowered = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", lowered, flags=re.UNICODE)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug


if __name__ == "__main__":
    raise SystemExit(main())
