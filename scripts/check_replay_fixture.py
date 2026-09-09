"""Compare generated replay evidence with the frozen supported-platform fixture."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
import sys


FIXTURE_PREFIXES = {
    "european_black_scholes_replay": "european_bs",
    "local_volatility_replay": "local_volatility",
}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_replay_fixture.py <generated.json>")

    generated = Path(sys.argv[1])
    generated_text, document = load_object(generated)
    fixture_kind, platform = replay_identity(generated, document)
    fixture_prefix = FIXTURE_PREFIXES[fixture_kind]

    expected = Path("fixtures/replay") / f"{fixture_prefix}-{platform}.json"
    if not expected.is_file():
        raise SystemExit(f"no frozen replay fixture for {platform}: {expected}")

    expected_text, expected_document = load_object(expected)
    expected_fixture_kind, expected_platform = replay_identity(expected, expected_document)
    if expected_fixture_kind != fixture_kind or expected_platform != platform:
        raise SystemExit(
            f"{expected}: fixture identity "
            f"{expected_fixture_kind!r}/{expected_platform!r} does not match generated "
            f"{fixture_kind!r}/{platform!r}"
        )

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


def replay_identity(path: Path, document: dict[str, object]) -> tuple[str, str]:
    schema_version = document.get("schema_version")
    if schema_version != 1:
        raise SystemExit(f"{path}: schema_version must be 1")
    platform = document.get("platform")
    if not isinstance(platform, str) or not platform:
        raise SystemExit(f"{path}: replay evidence has no non-empty string platform field")
    fixture_kind = document.get("fixture_kind")
    if not isinstance(fixture_kind, str):
        raise SystemExit(f"{path}: replay evidence has no string fixture_kind field")
    if fixture_kind not in FIXTURE_PREFIXES:
        raise SystemExit(f"{path}: unsupported replay fixture_kind: {fixture_kind!r}")
    return fixture_kind, platform


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
        document = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_json_constant,
        )
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


def reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


if __name__ == "__main__":
    main()
