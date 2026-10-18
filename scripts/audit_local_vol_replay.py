"""Audit the historical local-node allocation correction, without accepting a golden file.

The newer terminal replication transition has a separate v3 change audit.
This tool deliberately rejects it; pass the historical baseline explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from check_replay_fixture import load_object, replay_identity, workspace_package_version

ROOT = Path(__file__).resolve().parents[1]
CHANGE_AUDIT = ROOT / "fixtures/replay/local-vol-node-allocation-change-audit-v1.json"
DIAGNOSTIC_LABEL = "local_vol_node_allocation_diagnostic_v1"


def differences(before, after, path=""):
    if type(before) is not type(after):
        return [{"path": path, "before": before, "after": after}]
    if isinstance(before, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            child = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in before or key not in after:
                result.append({"path": child, "before": before.get(key), "after": after.get(key)})
            else:
                result.extend(differences(before[key], after[key], child))
        return result
    if isinstance(before, list):
        if len(before) != len(after):
            return [{"path": path, "before": before, "after": after}]
        return [item for i, (left, right) in enumerate(zip(before, after))
                for item in differences(left, right, path + "/" + str(i))]
    return [] if before == after else [{"path": path, "before": before, "after": after}]


def audit_candidate(baseline: Path, candidate: Path) -> dict:
    old_text, old = load_object(baseline)
    new_text, new = load_object(candidate)
    version = workspace_package_version()
    old_identity = replay_identity(baseline, old, version)
    new_identity = replay_identity(candidate, new, version)
    if old_identity != new_identity or new_identity[0] != "local_volatility_replay":
        raise ValueError("baseline and candidate must be Local Volatility captures of the same platform")
    allowed = {item[0] for item in json.loads(CHANGE_AUDIT.read_text())["differences"]}
    changes = differences(old, new)
    unexpected = [item["path"] for item in changes if item["path"] not in allowed]
    if unexpected:
        raise ValueError("changes outside the reviewed allocation correction: " + ", ".join(unexpected))
    for case in new["cases"][2:]:
        risks = case["result"]["risks"]
        kt = risks["vega_kt"]
        if kt["policy_label"] != DIAGNOSTIC_LABEL or kt["truncation_order"] != "not_applicable":
            raise ValueError("candidate must retain the diagnostic-only policy and truncation labels")
        scalar = risks["vega"]["raw"]["value"]
        bucket_sum = math.fsum(kt["raw_buckets"])
        residual = kt["residual_diagnostics"]
        projection = kt["projection"]
        reconstructed = [
            bucket_sum + report["signed_residual"]
            for report in [residual, projection]
        ]
        # Accounting only: this is not a paper-VegaKT numerical acceptance gate.
        for value in [residual["pre_projection"], projection["pre_projection"], *reconstructed]:
            if not math.isfinite(value) or not math.isclose(value, scalar, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("bucket and independently reported residual do not reconcile to scalar Vega")
        if not math.isclose(projection["scalar_vega"], bucket_sum, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("projection scalar does not reconcile to its bucket sum")
        if projection["reporting_stats"] != residual["reporting_stats"]:
            raise ValueError("projection and residual edge diagnostics disagree")
    return {
        "platform": new_identity[1],
        "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "reviewed_change_audit_sha256": hashlib.sha256(CHANGE_AUDIT.read_bytes()).hexdigest(),
        "exact_fixture_match": old_text == new_text,
        "status": "unchanged" if old_text == new_text else "requires_native_numerical_review",
        "difference_count": len(changes),
        "differences": changes,
        "paper_vegakt_accepted": False,
        "note": "No golden file is written. This audit does not replace check_replay_fixture.py or prove capture provenance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, document = load_object(args.candidate)
    kind, platform = replay_identity(args.candidate, document, workspace_package_version())
    if kind != "local_volatility_replay":
        parser.error("candidate must be a Local Volatility replay")
    baseline = args.baseline or ROOT / "fixtures/replay" / ("local_volatility-" + platform + ".json")
    output = args.output.resolve()
    if (output in {baseline.resolve(), args.candidate.resolve(), CHANGE_AUDIT.resolve()}
            or output.is_relative_to((ROOT / "fixtures").resolve())):
        parser.error("output must not overwrite a capture, baseline or repository fixture")
    try:
        report = audit_candidate(baseline, args.candidate)
    except ValueError as error:
        parser.exit(1, str(error) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ["platform", "status", "difference_count"]}))


if __name__ == "__main__":
    main()
