"""Measure native evidence export with exact risk/audit parity, without repricing.

Run using an installed ordinary wheel. This compares the original cold evidence
export with observation of the same real pricing call, not whole-suite speed.
Numerical settings remain the accepted four-interval Call / 4096-sample panel.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / directory) for directory in ("tests/python", "backtesting", "scripts")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "target/provider-evidence-benchmark.json")
    parser.add_argument("--native-source-revision", help="Known build revision; omit if not verified")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new evidence path")
    if os.environ.get("PRICING_TEST_ORIGINAL_EVIDENCE") == "1":
        parser.error("unset PRICING_TEST_ORIGINAL_EVIDENCE to benchmark captured evidence")

    from test_vega_kt_extension_provider import provider, panel
    from provider_evidence import capture_provider_evidence
    from rust_pricing_backtest.domain import fingerprint, data
    import rust_pricing

    binary = Path(getattr(rust_pricing, "rust_pricing", rust_pricing).__file__)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    source_paths = (Path(__file__), ROOT / "tests/python/provider_evidence.py",
                    ROOT / "tests/python/test_vega_kt_extension_provider.py")
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    report = dict(
        captured_at=datetime.now(timezone.utc).isoformat(),
        case="extension linear_time four-interval call, unchanged 4096 sampling units",
        scope="Native risk/audit parity and avoided cold evidence export; not whole-ledger acceptance or speedup",
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_worktree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        source_sha256=hashes,
        native_binary_sha256=sha(binary),
        native_source_revision=args.native_source_revision,
        native_source_verified=args.native_source_revision is not None,
        python_version=platform.python_version(), platform=platform.platform(),
    )
    p = provider("linear_time")
    c, s = panel(4, "call")
    start = time.perf_counter()
    with capture_provider_evidence(p) as captured:
        risk = p.evaluate(s, c)
    report["observed_evaluation_seconds"] = time.perf_counter() - start
    p.clear_cache()
    start = time.perf_counter()
    expected_risk = p.evaluate(s, c)
    expected_audit = p.gamma_audit(s, c)
    report["cold_evidence_export_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    actual_risk, actual_audit = captured.get(s, c)
    captured.assert_complete()
    report["recorded_evidence_export_seconds"] = time.perf_counter() - start
    if actual_risk != expected_risk or actual_risk != risk or actual_audit != expected_audit:
        raise RuntimeError("native risk/audit differs between observed and cold evidence")
    if hashes != {str(path.relative_to(ROOT)): sha(path) for path in source_paths} or sha(binary) != report["native_binary_sha256"]:
        raise RuntimeError("benchmark sources or native binary changed during measurement")
    report.update(status="passed", risk_sha256=fingerprint(data(actual_risk)), audit_sha256=fingerprint(actual_audit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
