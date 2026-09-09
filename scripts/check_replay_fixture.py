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
    document = json.loads(generated.read_text(encoding="utf-8"))
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

    generated_text = generated.read_text(encoding="utf-8")
    expected_text = expected.read_text(encoding="utf-8")
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


if __name__ == "__main__":
    main()
