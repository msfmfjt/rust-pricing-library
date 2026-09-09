"""Compare generated replay evidence with the frozen supported-platform fixture."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_replay_fixture.py <generated.json>")

    generated = Path(sys.argv[1])
    generated_text, document = load_object(generated)
    platform = document.get("platform")
    if not isinstance(platform, str):
        raise SystemExit("generated replay evidence has no string platform field")

    fixture_kind = document.get("fixture_kind")
    if fixture_kind == "european_black_scholes_replay":
        fixture_prefix = "european_bs"
    elif fixture_kind == "local_volatility_replay":
        fixture_prefix = "local_volatility"
    else:
        raise SystemExit(f"unsupported replay fixture_kind: {fixture_kind!r}")

    expected = Path("fixtures/replay") / f"{fixture_prefix}-{platform}.json"
    if not expected.is_file():
        raise SystemExit(f"no frozen replay fixture for {platform}: {expected}")

    expected_text, _ = load_object(expected)
    if generated_text != expected_text:
        diff = difflib.unified_diff(
            expected_text.splitlines(),
            generated_text.splitlines(),
            fromfile=str(expected),
            tofile=str(generated),
            lineterm="",
        )
        raise SystemExit("replay fixture mismatch:\n" + "\n".join(diff))

    print(f"replay fixture matches {expected}")


def load_object(path: Path) -> tuple[str, dict[str, object]]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise SystemExit(f"{path}: UTF-8 BOM is not allowed")
    if b"\r" in raw:
        raise SystemExit(f"{path}: CR or CRLF line endings are not allowed")
    if not raw.endswith(b"\n"):
        raise SystemExit(f"{path}: JSON artifact must end with LF")
    if raw.endswith(b"\n\n"):
        raise SystemExit(f"{path}: JSON artifact must end with exactly one LF")
    try:
        text = raw.decode("utf-8")
        document = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except ValueError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise SystemExit(f"{path}: top-level JSON value must be an object")
    return text, document


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate key: {key}")
        document[key] = value
    return document


if __name__ == "__main__":
    main()
