"""Summarize a complete, successful opt-in Bergomi quality run.

Use the release command in design/validation/bergomi-calibration-quality.md.
The source commit, compiler and platform must describe that run, including
when its log was produced on another machine. They are supplied provenance,
not facts that this parser can establish from numerical output alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = "crates/pricing/tests/bergomi_calibration_quality.rs"
RUN_PREFIX = "BERGOMI_RUN "
QUALITY_PREFIX = "BERGOMI_QUALITY "
SMILES = ("flat_20_percent", "skew_and_term_structure")
SEEDS = (42, 137, 711, 2027)
PRICING_SEED_MASK = 0xD1B54A32D192ED03
MATURITIES = (0.25, 0.5, 1.0)
COORDINATES = [(t, x) for t in MATURITIES for x in (-0.2, -0.1, 0.0, 0.1, 0.2)]
MODEL = {"mean_reversion": 2.0, "nu": 0.7, "rho": -0.5}
SETTINGS = {
    "particles": 262144,
    "bandwidth": 0.01,
    "steps": 1024,
    "spatial_intervals": 160,
    "points_per_scramble": 32768,
    "scrambles": 8,
}
NODE_FIELDS = {
    "maturity", "log_strike", "target_iv", "model_iv", "lv_iv",
    "iv_error_bp", "lv_error_bp", "pricing_se_bp", "total_se_bp",
    "seed_to_seed_sd_bp", "paired_price_residual_bp", "paired_price_se_bp",
    "worst_seed_iv_error_bp", "minimum_ess", "quote_fallbacks",
    "target_interpolation_error_bp",
}
EVIDENCE_SCOPE = (
    "Supplementary test-only one-factor Bergomi diagnostics using paired "
    "low-level paths and a test-specific bridge layout. Not public pricing-plan "
    "acceptance, market-data calibration, or multi-platform evidence."
)


class SummaryError(ValueError):
    """Incomplete, inconsistent or unsuccessful producer evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SummaryError(message)


def reject_constant(value: str) -> None:
    raise SummaryError(f"nonfinite JSON constant {value}")


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def finite_values(value: object, location: str) -> None:
    # serde_json writes nonfinite Rust floats as null; Python also accepts
    # exponent overflow (1e999) unless checked after parsing.
    require(value is not None, f"{location}: null is not a reportable metric")
    if isinstance(value, float):
        require(math.isfinite(value), f"{location}: nonfinite value")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_values(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_values(item, f"{location}[{index}]")


def payloads(log: str, prefix: str) -> list[dict]:
    found = []
    for number, line in enumerate(log.splitlines(), start=1):
        index = line.find(prefix)
        if index < 0:
            continue
        try:
            value = json.loads(
                line[index + len(prefix):],
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except (json.JSONDecodeError, SummaryError) as error:
            raise SummaryError(f"line {number}: {error}") from error
        require(isinstance(value, dict), f"line {number}: expected an object")
        finite_values(value, f"line {number}")
        found.append(value)
    return found


def numeric_fields(entry: object, fields: set[str], label: str) -> None:
    require(isinstance(entry, dict), f"{label}: expected an object")
    require(set(entry) == fields, f"{label}: missing or unexpected fields")
    for key, value in entry.items():
        require(type(value) in (int, float), f"{label}.{key}: expected a number")


def nonnegative(entry: dict, fields: tuple[str, ...], label: str) -> None:
    for key in fields:
        require(entry[key] >= 0, f"{label}.{key}: must be nonnegative")


def zero_count(value: object, label: str) -> None:
    require(type(value) is int and value == 0, f"{label}: must be integer zero")


def panel(payload: dict) -> None:
    require(payload.get("smile") in SMILES, "unknown acceptance smile")
    for key, expected in (("model", MODEL), ("settings", SETTINGS)):
        actual = payload.get(key)
        numeric_fields(actual, set(expected), key)
        require(actual == expected, f"{key} differs from the fixed acceptance panel")


def coordinates(entries: object, fields: set[str], label: str) -> None:
    require(isinstance(entries, list) and len(entries) == 15, f"{label}: expected 15 quotes")
    for entry in entries:
        numeric_fields(entry, fields, label)
    require(
        [(e["maturity"], e["log_strike"]) for e in entries] == COORDINATES,
        f"{label}: incomplete, duplicate or reordered quote coordinates",
    )


def estimates(entries: object, count: int, label: str) -> None:
    require(isinstance(entries, list) and len(entries) == count, f"{label}: expected {count} estimates")
    for entry in entries:
        numeric_fields(entry, {"value", "se"}, label)
        nonnegative(entry, ("se",), label)


def check_log(log: str, runs: list[dict], reports: list[dict]) -> None:
    # Reports are printed BEFORE Rust's numerical assertions. A report alone
    # cannot establish that the gate passed, or even that the test completed.
    lines = log.splitlines()
    summaries = [line for line in lines if line.startswith("test result:")]
    require(
        len(summaries) == 1 and re.fullmatch(
            r"test result: ok\. 2 passed; 0 failed; 0 ignored; 0 measured; 9 filtered out; finished in .+",
            summaries[0],
        ) is not None,
        "expected one successful two-test completion; failed or partial logs are not evidence",
    )
    require("Finished `release` profile" in log, "expected cargo's release-profile completion")
    require(
        log.rfind("test result:") > max(log.rfind(RUN_PREFIX), log.rfind(QUALITY_PREFIX)),
        "producer completion precedes a report",
    )
    names = re.findall(r"^test (iv_round_trip_\S+) \.\.\.", log, re.MULTILINE)
    require(
        sorted(names) == ["iv_round_trip_flat_one_factor_bergomi", "iv_round_trip_skew_one_factor_bergomi"],
        "the log must run exactly the flat and skew acceptance tests",
    )
    require(len(reports) == 2 and len(runs) == 8, "expected two reports and eight complete runs")
    report_smiles = []
    for report in reports:
        panel(report)
        report_smiles.append(report["smile"])
        coordinates(report.get("nodes"), NODE_FIELDS, "report.nodes")
        estimates(report.get("martingales"), 3, "report.martingales")
        zero_count(report.get("quote_fallbacks"), "report.quote_fallbacks")
        for node in report["nodes"]:
            nonnegative(node, (
                "pricing_se_bp", "total_se_bp", "seed_to_seed_sd_bp",
                "paired_price_se_bp", "worst_seed_iv_error_bp", "minimum_ess",
            ), "node")
            zero_count(node["quote_fallbacks"], "node.quote_fallbacks")
    require(sorted(report_smiles) == sorted(SMILES), "missing or repeated smile report")

    observed = set()
    for payload in runs:
        panel(payload)
        run = payload.get("run")
        require(isinstance(run, dict), "missing complete run")
        seed = run.get("calibration_seed")
        require(type(seed) is int and seed in SEEDS, "unknown calibration seed")
        key = (payload["smile"], seed)
        require(key not in observed, f"duplicate complete run {key}")
        observed.add(key)
        pricing_seed = run.get("pricing_seed")
        require(
            type(pricing_seed) is int and pricing_seed == (seed ^ PRICING_SEED_MASK),
            "pricing seed does not match the calibration seed",
        )
        estimates(run.get("estimates"), 48, "run.estimates")
        support = run.get("support")
        require(isinstance(support, list) and len(support) == 3, "expected three support rows")
        for row, maturity in zip(support, MATURITIES):
            numeric_fields(row, {
                "time", "minimum_quote_ess", "quote_fallbacks",
                "all_row_fallbacks", "particle_mean_f",
            }, "run.support")
            require(row["time"] == maturity, "support row maturity mismatch")
            nonnegative(row, ("minimum_quote_ess", "all_row_fallbacks"), "run.support")
            require(type(row["all_row_fallbacks"]) is int, "fallback count must be integer")
            zero_count(row["quote_fallbacks"], "run.support.quote_fallbacks")
        quotes = run.get("quote_support")
        coordinates(quotes, {"maturity", "log_strike", "minimum_ess", "fallbacks"}, "run.quote_support")
        for quote in quotes:
            nonnegative(quote, ("minimum_ess",), "run.quote_support")
            zero_count(quote["fallbacks"], "run.quote_support.fallbacks")
    require(observed == {(smile, seed) for smile in SMILES for seed in SEEDS}, "incomplete smile/seed matrix")


def build(log: str, *, source_commit: str, platform: str, compiler: str, test_source: bytes) -> dict:
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None, "source commit must be a full lowercase Git SHA")
    require(bool(platform.strip()) and bool(compiler.strip()), "platform and compiler must be explicit")
    require(bool(test_source), "test source must not be empty")
    runs = payloads(log, RUN_PREFIX)
    reports = payloads(log, QUALITY_PREFIX)
    check_log(log, runs, reports)
    return {
        "source_commit": source_commit,
        "test_file": TEST_FILE,
        "test_source_sha256": hashlib.sha256(test_source).hexdigest(),
        "log_sha256": hashlib.sha256(log.encode("utf-8")).hexdigest(),
        "platform": platform,
        "compiler": compiler,
        "mode": "release",
        "calibration_seeds": list(SEEDS),
        "model": MODEL,
        "settings": SETTINGS,
        "iv_bp": 1e-4,
        "complete_runs": runs,
        "acceptance_reports": reports,
        "evidence_scope": EVIDENCE_SCOPE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="explicit destination for newly generated evidence")
    parser.add_argument("--source-commit", required=True, help="full commit of the tested checkout")
    parser.add_argument("--platform", required=True, help="target triple of the test run")
    parser.add_argument("--compiler", required=True, help="rustc version line of the test run")
    parser.add_argument("--test-file", type=Path, default=ROOT / TEST_FILE, help="exact source used by the tested binary")
    parser.add_argument("--check", action="store_true", help="compare with existing output without writing")
    args = parser.parse_args(argv)
    try:
        document = build(
            args.log.read_bytes().decode("utf-8"),
            source_commit=args.source_commit, platform=args.platform,
            compiler=args.compiler, test_source=args.test_file.read_bytes(),
        )
        text = json.dumps(document, indent=2, allow_nan=False) + "\n"
        if args.check:
            require(args.out.exists() and args.out.read_bytes() == text.encode("utf-8"), "output is missing or stale for this log")
            print(f"{args.out} matches the complete log")
        else:
            args.out.write_bytes(text.encode("utf-8"))
            print(f"wrote {args.out} (two smiles, four complete runs each)")
    except (SummaryError, OSError, UnicodeError) as error:
        print(f"summarize_bergomi_quality: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
