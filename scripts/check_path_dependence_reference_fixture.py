#!/usr/bin/env python3
"""Independently verify the Gate P0 Path Dependence reference fixture."""

from __future__ import annotations

import json
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any


getcontext().prec = 70
D = Decimal
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "path-dependence" / "reference-cases-v0.1.json"

TOP_LEVEL_KEYS = {
    "fixture_version",
    "policy_version",
    "tolerances",
    "smoothing_cases",
    "extrema_cases",
    "bridge_cases",
    "smoothed_bridge_cases",
    "jump_cases",
}
SMOOTHING_IDS = {
    "below_band",
    "lower_boundary",
    "lower_interior",
    "center",
    "upper_interior",
    "upper_boundary",
    "above_band",
}
EXTREMA_IDS = {"interior_ordered", "equal_operands", "outside_band_reversed"}
BRIDGE_IDS = {
    "up_safe_endpoints",
    "down_safe_endpoints",
    "endpoint_touch",
    "zero_variance_safe",
}
SMOOTHED_BRIDGE_IDS = {
    "up_safe_exterior",
    "up_center_endpoint",
    "up_hit_exterior",
    "down_interior_endpoints",
    "up_center_zero_variance",
}
JUMP_IDS = {"down_crossing", "down_safe", "down_near_crossing"}


class Checks:
    def __init__(self, decimal_abs: D) -> None:
        self.decimal_abs = decimal_abs
        self.count = 0

    def decimal(self, label: str, actual: D, expected: str) -> None:
        self.count += 1
        error = abs(actual - D(expected))
        if error > self.decimal_abs:
            raise AssertionError(
                f"{label}: actual={actual} expected={expected} error={error}"
            )

    def exact(self, label: str, actual: object, expected: object) -> None:
        self.count += 1
        if actual != expected:
            raise AssertionError(f"{label}: actual={actual!r} expected={expected!r}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise AssertionError(
            f"{label}: field mismatch missing={sorted(expected-actual)} "
            f"unknown={sorted(actual-expected)}"
        )


def require_case_ids(cases: Any, expected: set[str], label: str) -> list[dict[str, Any]]:
    if not isinstance(cases, list):
        raise AssertionError(f"{label} must be an array")
    if not all(isinstance(case, dict) for case in cases):
        raise AssertionError(f"{label} entries must be objects")
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{label} contains duplicate ids")
    if set(ids) != expected:
        raise AssertionError(
            f"{label}: case mismatch missing={sorted(expected-set(ids))} "
            f"unknown={sorted(set(ids)-expected)}"
        )
    return cases


def smoothing_values(x: D, half_width: D) -> dict[str, D]:
    if half_width <= 0:
        raise AssertionError("half_width must be positive")
    if x <= -half_width:
        indicator = indicator_first = indicator_second = D(0)
        positive_part = positive_part_first = positive_part_second = D(0)
    elif x >= half_width:
        indicator = D(1)
        indicator_first = indicator_second = D(0)
        positive_part = x
        positive_part_first = D(1)
        positive_part_second = D(0)
    else:
        u = (x + half_width) / (2 * half_width)
        indicator = 6 * u**5 - 15 * u**4 + 10 * u**3
        indicator_first = 30 * u**2 * (u - 1) ** 2 / (2 * half_width)
        indicator_second = (
            120 * u**3 - 180 * u**2 + 60 * u
        ) / (4 * half_width**2)
        positive_part = 2 * half_width * (
            u**6 - 3 * u**5 + D("2.5") * u**4
        )
        positive_part_first = indicator
        positive_part_second = indicator_first
    return {
        "indicator": indicator,
        "indicator_first": indicator_first,
        "indicator_second": indicator_second,
        "positive_part": positive_part,
        "positive_part_first": positive_part_first,
        "positive_part_second": positive_part_second,
    }


def smooth_maximum(left: D, right: D, half_width: D) -> D:
    positive_part = smoothing_values(left - right, half_width)["positive_part"]
    return right + positive_part


def check_smoothing(fixture: dict[str, Any], checks: Checks) -> None:
    cases = require_case_ids(fixture["smoothing_cases"], SMOOTHING_IDS, "smoothing_cases")
    expected_keys = {
        "indicator",
        "indicator_first",
        "indicator_second",
        "positive_part",
        "positive_part_first",
        "positive_part_second",
    }
    for case in cases:
        require_keys(case, {"id", "half_width", "input", "expected"}, case["id"])
        require_keys(case["expected"], expected_keys, f"{case['id']}.expected")
        actual = smoothing_values(D(case["input"]), D(case["half_width"]))
        for field in sorted(expected_keys):
            checks.decimal(
                f"smoothing.{case['id']}.{field}",
                actual[field],
                case["expected"][field],
            )


def check_extrema(fixture: dict[str, Any], checks: Checks) -> None:
    cases = require_case_ids(fixture["extrema_cases"], EXTREMA_IDS, "extrema_cases")
    for case in cases:
        require_keys(
            case, {"id", "left", "right", "half_width", "expected"}, case["id"]
        )
        require_keys(case["expected"], {"maximum", "minimum"}, f"{case['id']}.expected")
        left = D(case["left"])
        right = D(case["right"])
        half_width = D(case["half_width"])
        maximum = smooth_maximum(left, right, half_width)
        minimum = left + right - maximum
        checks.decimal(f"extrema.{case['id']}.maximum", maximum, case["expected"]["maximum"])
        checks.decimal(f"extrema.{case['id']}.minimum", minimum, case["expected"]["minimum"])
        checks.decimal(
            f"extrema.{case['id']}.symmetry",
            maximum,
            str(smooth_maximum(right, left, half_width)),
        )
        checks.decimal(f"extrema.{case['id']}.partition", minimum + maximum, str(left + right))


def bridge_probabilities(case: dict[str, Any]) -> tuple[D, D]:
    direction = case["direction"]
    barrier = D(case["barrier"])
    start = D(case["start_spot"])
    end = D(case["end_spot"])
    variance = D(case["integrated_variance"])
    if min(barrier, start, end) <= 0:
        raise AssertionError("bridge levels must be positive")
    if variance < 0:
        raise AssertionError("integrated_variance must be non-negative")
    if direction == "up":
        touched = start >= barrier or end >= barrier
        first = (barrier / start).ln()
        second = (barrier / end).ln()
    elif direction == "down":
        touched = start <= barrier or end <= barrier
        first = (start / barrier).ln()
        second = (end / barrier).ln()
    else:
        raise AssertionError(f"unknown bridge direction {direction!r}")
    if touched:
        hit = D(1)
    elif variance == 0:
        hit = D(0)
    else:
        hit = (-2 * first * second / variance).exp()
    return hit, D(1) - hit


def check_bridges(fixture: dict[str, Any], checks: Checks) -> None:
    cases = require_case_ids(fixture["bridge_cases"], BRIDGE_IDS, "bridge_cases")
    fields = {
        "id",
        "direction",
        "barrier",
        "start_spot",
        "end_spot",
        "integrated_variance",
        "expected",
    }
    for case in cases:
        require_keys(case, fields, case["id"])
        require_keys(
            case["expected"],
            {"hit_probability", "survival_probability"},
            f"{case['id']}.expected",
        )
        hit, survival = bridge_probabilities(case)
        checks.decimal(f"bridge.{case['id']}.hit", hit, case["expected"]["hit_probability"])
        checks.decimal(
            f"bridge.{case['id']}.survival",
            survival,
            case["expected"]["survival_probability"],
        )
        checks.decimal(f"bridge.{case['id']}.partition", hit + survival, "1")


def signed_distance(direction: str, barrier: D, spot: D) -> D:
    if direction == "up":
        return spot - barrier
    if direction == "down":
        return barrier - spot
    raise AssertionError(f"unknown jump direction {direction!r}")


def smoothed_bridge_values(case: dict[str, Any]) -> dict[str, Any]:
    direction = case["direction"]
    barrier = D(case["barrier"])
    start = D(case["start_spot"])
    end = D(case["end_spot"])
    half_width = D(case["half_width"])
    variance = D(case["integrated_variance"])
    if min(barrier, start, end, half_width) <= 0:
        raise AssertionError("smoothed bridge levels and width must be positive")
    if variance < 0:
        raise AssertionError("integrated_variance must be non-negative")

    distances = [
        signed_distance(direction, barrier, start),
        signed_distance(direction, barrier, end),
    ]
    hit_weights: list[D] = []
    log_distances: list[D] = []
    for distance in distances:
        hit_weights.append(smoothing_values(distance, half_width)["indicator"])
        safe_distance = smoothing_values(-distance, half_width)["positive_part"]
        if direction == "up":
            log_distances.append(-(D(1) - safe_distance / barrier).ln())
        elif direction == "down":
            log_distances.append((D(1) + safe_distance / barrier).ln())
        else:
            raise AssertionError(f"unknown bridge direction {direction!r}")

    bridge_survival = (
        D(1)
        if variance == 0
        else D(1) - (-2 * log_distances[0] * log_distances[1] / variance).exp()
    )
    endpoint_survival = (D(1) - hit_weights[0]) * (D(1) - hit_weights[1])
    total_survival = endpoint_survival * bridge_survival
    return {
        "hit_weights": hit_weights,
        "effective_log_distances": log_distances,
        "bridge_survival": bridge_survival,
        "endpoint_survival": endpoint_survival,
        "total_survival": total_survival,
        "total_hit_weight": D(1) - total_survival,
    }


def check_smoothed_bridges(fixture: dict[str, Any], checks: Checks) -> None:
    cases = require_case_ids(
        fixture["smoothed_bridge_cases"],
        SMOOTHED_BRIDGE_IDS,
        "smoothed_bridge_cases",
    )
    fields = {
        "id",
        "direction",
        "barrier",
        "start_spot",
        "end_spot",
        "half_width",
        "integrated_variance",
        "expected",
    }
    expected_fields = {
        "start_hit_weight",
        "end_hit_weight",
        "start_effective_log_distance",
        "end_effective_log_distance",
        "bridge_survival",
        "endpoint_survival",
        "total_survival",
        "total_hit_weight",
    }
    for case in cases:
        require_keys(case, fields, case["id"])
        require_keys(case["expected"], expected_fields, f"{case['id']}.expected")
        actual = smoothed_bridge_values(case)
        pairs = {
            "start_hit_weight": actual["hit_weights"][0],
            "end_hit_weight": actual["hit_weights"][1],
            "start_effective_log_distance": actual["effective_log_distances"][0],
            "end_effective_log_distance": actual["effective_log_distances"][1],
            "bridge_survival": actual["bridge_survival"],
            "endpoint_survival": actual["endpoint_survival"],
            "total_survival": actual["total_survival"],
            "total_hit_weight": actual["total_hit_weight"],
        }
        for field, value in pairs.items():
            checks.decimal(
                f"smoothed_bridge.{case['id']}.{field}",
                value,
                case["expected"][field],
            )


def check_jumps(fixture: dict[str, Any], checks: Checks) -> None:
    cases = require_case_ids(fixture["jump_cases"], JUMP_IDS, "jump_cases")
    fields = {
        "id",
        "direction",
        "barrier",
        "pre_jump_spot",
        "post_jump_spot",
        "half_width",
        "expected",
    }
    for case in cases:
        require_keys(case, fields, case["id"])
        require_keys(case["expected"], {"signed_score", "hit_weight"}, f"{case['id']}.expected")
        barrier = D(case["barrier"])
        half_width = D(case["half_width"])
        before = signed_distance(case["direction"], barrier, D(case["pre_jump_spot"]))
        after = signed_distance(case["direction"], barrier, D(case["post_jump_spot"]))
        score = smooth_maximum(before, after, half_width)
        weight = smoothing_values(score, half_width)["indicator"]
        checks.decimal(f"jump.{case['id']}.score", score, case["expected"]["signed_score"])
        checks.decimal(f"jump.{case['id']}.weight", weight, case["expected"]["hit_weight"])


def main() -> int:
    raw = FIXTURE.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise AssertionError("JSON artifact must not start with a UTF-8 BOM")
    if b"\r" in raw:
        raise AssertionError("JSON artifact must use LF line endings")
    if not raw.endswith(b"\n"):
        raise AssertionError("JSON artifact must end with LF")
    fixture = json.loads(
        raw,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_json_constant,
    )
    if not isinstance(fixture, dict):
        raise AssertionError("fixture top level must be an object")
    require_keys(fixture, TOP_LEVEL_KEYS, "fixture")
    checks = Checks(D(fixture["tolerances"]["decimal_abs"]))
    checks.exact("fixture_version", fixture["fixture_version"], 2)
    checks.exact("policy_version", fixture["policy_version"], "path_dependence_v2")
    require_keys(fixture["tolerances"], {"decimal_abs"}, "tolerances")
    check_smoothing(fixture, checks)
    check_extrema(fixture, checks)
    check_bridges(fixture, checks)
    check_smoothed_bridges(fixture, checks)
    check_jumps(fixture, checks)
    print(f"Path Dependence reference fixture passes {checks.count} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
