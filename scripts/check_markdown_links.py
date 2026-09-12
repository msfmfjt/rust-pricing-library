"""Validate repository-local Markdown links used by docs."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_ROOTS = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "docs", ROOT / "fixtures"]
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


class MarkdownAnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.anchors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del tag
        for name, value in attrs:
            if name.lower() in {"id", "name"} and value:
                self.anchors.append(value)


def main() -> int:
    missing = []
    files = markdown_files()
    anchor_cache = {path: markdown_anchors(path) for path in files}
    for path, (_, duplicates) in anchor_cache.items():
        for anchor in duplicates:
            missing.append((path, f"#{anchor}", "duplicate anchor"))

    for path in files:
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
            indexed = anchor_cache.get(resolved)
            if indexed is None:
                indexed = markdown_anchors(resolved)
                anchor_cache[resolved] = indexed
            anchors, _ = indexed
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


def markdown_anchors(path: Path) -> tuple[set[str], list[str]]:
    if path.suffix.lower() != ".md":
        return set(), []
    anchors: set[str] = set()
    duplicates: list[str] = []
    heading_counts: dict[str, int] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", line)
        if match is None:
            continue
        base = github_heading_slug(match.group(2))
        count = heading_counts.get(base, 0)
        heading_counts[base] = count + 1
        add_anchor(anchors, duplicates, base if count == 0 else f"{base}-{count}")

    parser = MarkdownAnchorParser()
    parser.feed(text)
    for anchor in parser.anchors:
        add_anchor(anchors, duplicates, anchor)
    return anchors, duplicates


def add_anchor(anchors: set[str], duplicates: list[str], anchor: str) -> None:
    if anchor in anchors:
        duplicates.append(anchor)
    anchors.add(anchor)


def github_heading_slug(heading: str) -> str:
    lowered = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", lowered, flags=re.UNICODE)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug


if __name__ == "__main__":
    raise SystemExit(main())
