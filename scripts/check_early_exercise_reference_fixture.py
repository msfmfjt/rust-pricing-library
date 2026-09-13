#!/usr/bin/env python3
"""Independently verify the Gate E0 Early Exercise reference fixture."""

from __future__ import annotations

import json
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any


getcontext().prec = 80
D = Decimal
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "early-exercise" / "reference-cases-v0.1.json"
EPSILON = D(2) ** -52

TOP_LEVEL_KEYS = {
    "fixture_version",
    "policy_version",
    "solver_version",
    "tolerances",
    "decision_cases",
    "basis_cases",
    "scaling_cases",
    "qr_cases",
}
DECISION_IDS = {
    "continue_on_tie",
    "exercise_strictly_above",
    "not_itm_on_tolerance",
    "itm_strictly_above_tolerance",
}
BASIS_IDS = {
    "constant_only",
    "one_feature_cubic",
    "two_feature_quadratic",
    "three_feature_linear",
}
SCALING_IDS = {"symmetric_active", "constant_inactive", "shifted_active"}
QR_IDS = {
    "orthogonal_full_rank",
    "larger_column_pivots_first",
    "norm_tie_uses_original_order",
    "absolute_rank_exclusion",
}


class Checks:
    def __init__(self, decimal_abs: D) -> None:
        self.decimal_abs = decimal_abs
        self.count = 0

    def decimal(self, label: str, actual: D, expected: object) -> None:
        self.count += 1
        error = abs(actual - D(str(expected)))
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
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise AssertionError(f"{label} must be an object array")
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{label} contains duplicate ids")
    if set(ids) != expected:
        raise AssertionError(
            f"{label}: case mismatch missing={sorted(expected-set(ids))} "
            f"unknown={sorted(set(ids)-expected)}"
        )
    return cases


def exponent_vectors(feature_count: int, max_degree: int) -> list[list[int]]:
    if feature_count < 0 or max_degree < 0:
        raise AssertionError("basis dimensions must be non-negative")
    if feature_count == 0:
        return [[]]
    result: list[list[int]] = []

    def visit(prefix: list[int], remaining_features: int, remaining_degree: int) -> None:
        if remaining_features == 1:
            result.append([*prefix, remaining_degree])
            return
        for exponent in range(remaining_degree, -1, -1):
            visit([*prefix, exponent], remaining_features - 1, remaining_degree - exponent)

    for total_degree in range(max_degree + 1):
        visit([], feature_count, total_degree)
    return result


def check_decisions(fixture: dict[str, Any], checks: Checks) -> None:
    cases = require_case_ids(fixture["decision_cases"], DECISION_IDS, "decision_cases")
    for case in cases:
        require_keys(
            case,
            {"id", "kind", "immediate_value", "comparison_value", "expected"},
            case["id"],
        )
        checks.exact(f"decision.{case['id']}", D(case["immediate_value"]) > D(case["comparison_value"]), case["expected"])
        checks.exact(f"decision.{case['id']}.kind", case["kind"] in {"exercise", "itm"}, True)


def check_basis(fixture: dict[str, Any], checks: Checks) -> None:
    cases = require_case_ids(fixture["basis_cases"], BASIS_IDS, "basis_cases")
    for case in cases:
        require_keys(
            case,
            {"id", "feature_count", "max_degree", "expected_exponents"},
            case["id"],
        )
        actual = exponent_vectors(case["feature_count"], case["max_degree"])
        checks.exact(f"basis.{case['id']}", actual, case["expected_exponents"])
        checks.exact(
            f"basis.{case['id']}.degrees",
            all(sum(vector) <= case["max_degree"] for vector in actual),
            True,
        )


def scaling(values: list[D]) -> dict[str, Any]:
    if not values:
        raise AssertionError("scaling requires at least one value")
    mean = sum(values, D(0)) / len(values)
    variance = sum(((value - mean) ** 2 for value in values), D(0)) / len(values)
    scale = variance.sqrt()
    feature_scale = max(D(1), *(abs(value) for value in values))
    threshold = D(64) * EPSILON * feature_scale
    inactive = scale <= threshold
    standardized = [] if inactive else [(value - mean) / scale for value in values]
    return {
        "mean": mean,
        "population_variance": variance,
        "scale": scale,
        "zero_scale_threshold": threshold,
        "inactive": inactive,
        "standardized": standardized,
    }


def check_scaling(fixture: dict[str, Any], checks: Checks) -> None:
    cases = require_case_ids(fixture["scaling_cases"], SCALING_IDS, "scaling_cases")
    expected_keys = {
        "mean",
        "population_variance",
        "scale",
        "zero_scale_threshold",
        "inactive",
        "standardized",
    }
    for case in cases:
        require_keys(case, {"id", "values", "expected"}, case["id"])
        require_keys(case["expected"], expected_keys, f"{case['id']}.expected")
        actual = scaling([D(value) for value in case["values"]])
        for field in ["mean", "population_variance", "scale", "zero_scale_threshold"]:
            checks.decimal(f"scaling.{case['id']}.{field}", actual[field], case["expected"][field])
        checks.exact(f"scaling.{case['id']}.inactive", actual["inactive"], case["expected"]["inactive"])
        checks.exact(
            f"scaling.{case['id']}.standardized_length",
            len(actual["standardized"]),
            len(case["expected"]["standardized"]),
        )
        for index, (actual_value, expected_value) in enumerate(zip(actual["standardized"], case["expected"]["standardized"], strict=True)):
            checks.decimal(f"scaling.{case['id']}.standardized[{index}]", actual_value, expected_value)


def cpqr(matrix: list[list[D]], target: list[D], abs_tol: D, rel_tol: D) -> dict[str, Any]:
    original = [row[:] for row in matrix]
    m = len(matrix)
    n = len(matrix[0]) if matrix else 0
    if m != len(target) or any(len(row) != n for row in matrix):
        raise AssertionError("invalid QR shape")
    permutation = list(range(n))
    transformed_target = target[:]
    diagonal_count = min(m, n)
    for k in range(diagonal_count):
        norms = [sum((matrix[row][column] ** 2 for row in range(k, m)), D(0)) for column in range(k, n)]
        pivot_offset = min(
            range(len(norms)),
            key=lambda offset: (-norms[offset], permutation[k + offset]),
        )
        pivot = k + pivot_offset
        if pivot != k:
            for row in matrix:
                row[k], row[pivot] = row[pivot], row[k]
            permutation[k], permutation[pivot] = permutation[pivot], permutation[k]
        norm = sum((matrix[row][k] ** 2 for row in range(k, m)), D(0)).sqrt()
        if norm == 0:
            matrix[k][k] = D(0)
            continue
        leading = matrix[k][k]
        alpha = -norm if leading >= 0 else norm
        denominator = leading - alpha
        reflector = [matrix[row][k] / denominator for row in range(k + 1, m)]
        tau = (alpha - leading) / alpha
        matrix[k][k] = alpha
        for row, value in zip(range(k + 1, m), reflector, strict=True):
            matrix[row][k] = value
        for column in range(k + 1, n):
            dot = matrix[k][column] + sum(
                (value * matrix[row][column] for row, value in zip(range(k + 1, m), reflector, strict=True)),
                D(0),
            )
            dot *= tau
            matrix[k][column] -= dot
            for row, value in zip(range(k + 1, m), reflector, strict=True):
                matrix[row][column] -= value * dot
        dot = transformed_target[k] + sum(
            (value * transformed_target[row] for row, value in zip(range(k + 1, m), reflector, strict=True)),
            D(0),
        )
        dot *= tau
        transformed_target[k] -= dot
        for row, value in zip(range(k + 1, m), reflector, strict=True):
            transformed_target[row] -= value * dot

    diagonals = [abs(matrix[index][index]) for index in range(diagonal_count)]
    rank_threshold = max(abs_tol, rel_tol * (diagonals[0] if diagonals else D(0)))
    rank = 0
    for diagonal in diagonals:
        if diagonal <= rank_threshold:
            break
        rank += 1
    pivot_coefficients = [D(0)] * n
    for row in range(rank - 1, -1, -1):
        remainder = transformed_target[row] - sum(
            (matrix[row][column] * pivot_coefficients[column] for column in range(row + 1, rank)),
            D(0),
        )
        pivot_coefficients[row] = remainder / matrix[row][row]
    coefficients = [D(0)] * n
    for pivot_position, original_column in enumerate(permutation):
        coefficients[original_column] = pivot_coefficients[pivot_position]
    residual_sum_squares = sum(
        (
            target[row]
            - sum((original[row][column] * coefficients[column] for column in range(n)), D(0))
        ) ** 2
        for row in range(m)
    )
    return {
        "pivot_order": permutation,
        "rank": rank,
        "coefficients": coefficients,
        "residual_sum_squares": residual_sum_squares,
    }


def check_qr(fixture: dict[str, Any], checks: Checks) -> None:
    cases = require_case_ids(fixture["qr_cases"], QR_IDS, "qr_cases")
    expected_keys = {"pivot_order", "rank", "coefficients", "residual_sum_squares"}
    for case in cases:
        require_keys(case, {"id", "matrix", "target", "abs_rank_tol", "rel_rank_tol", "expected"}, case["id"])
        require_keys(case["expected"], expected_keys, f"{case['id']}.expected")
        actual = cpqr(
            [[D(value) for value in row] for row in case["matrix"]],
            [D(value) for value in case["target"]],
            D(case["abs_rank_tol"]),
            D(case["rel_rank_tol"]),
        )
        checks.exact(f"qr.{case['id']}.pivot_order", actual["pivot_order"], case["expected"]["pivot_order"])
        checks.exact(f"qr.{case['id']}.rank", actual["rank"], case["expected"]["rank"])
        for index, (actual_value, expected_value) in enumerate(zip(actual["coefficients"], case["expected"]["coefficients"], strict=True)):
            checks.decimal(f"qr.{case['id']}.coefficients[{index}]", actual_value, expected_value)
        checks.decimal(f"qr.{case['id']}.residual_sum_squares", actual["residual_sum_squares"], case["expected"]["residual_sum_squares"])


def main() -> None:
    raw = FIXTURE.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise AssertionError("fixture must not start with a UTF-8 BOM")
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise AssertionError("JSON artifact must end with LF and use LF line endings")
    fixture = json.loads(
        raw,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_json_constant,
    )
    require_keys(fixture, TOP_LEVEL_KEYS, "fixture")
    if fixture["fixture_version"] != 1:
        raise AssertionError("fixture_version must be 1")
    if fixture["policy_version"] != "early_exercise_v1":
        raise AssertionError("unexpected policy_version")
    if fixture["solver_version"] != "cpqr_householder_v1":
        raise AssertionError("unexpected solver_version")
    require_keys(fixture["tolerances"], {"decimal_abs"}, "tolerances")
    checks = Checks(D(fixture["tolerances"]["decimal_abs"]))
    check_decisions(fixture, checks)
    check_basis(fixture, checks)
    check_scaling(fixture, checks)
    check_qr(fixture, checks)
    print(f"Early Exercise reference fixture passes {checks.count} checks")


if __name__ == "__main__":
    main()
