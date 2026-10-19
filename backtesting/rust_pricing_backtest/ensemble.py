"""Paired terminal-error statistics for a fixed book and observation schedule.

This report consumes results; it never fills gaps, changes a replay, certifies
independence, or upgrades the numerical validation of its pricing provider.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import dataclass
from decimal import Decimal
import json
import math
from pathlib import Path
import platform
import statistics
from typing import Mapping, Sequence

import numpy as np

from .domain import BacktestError, canonical, data, fingerprint, require
from .engine import BacktestResult


STRATEGIES = ("unhedged", "delta", "scalar_vega", "vega_kt")


@dataclass(frozen=True)
class EnsembleConfig:
    metric: str = "nominal"
    sampling_design: str = "dependent_or_unknown"
    loss_levels: tuple[float, ...] = (.95, .99)
    confidence_level: float = .95
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 20261003
    minimum_ci_samples: int = 20
    minimum_tail_observations: int = 20
    minimum_baseline_variance: float = 1e-12

    def __post_init__(self):
        require(self.metric in ("nominal", "discounted"), "invalid_ensemble_config", "metric")
        require(self.sampling_design in ("dependent_or_unknown", "independent_simulation"),
                "invalid_ensemble_config", "sampling_design")
        levels = tuple(self.loss_levels)
        require(bool(levels) and all(isinstance(x, (float, int)) and not isinstance(x, bool)
                                    and math.isfinite(x) and 0 < x < 1 for x in levels)
                and tuple(sorted(set(levels))) == levels, "invalid_ensemble_config", "sorted unique loss_levels")
        object.__setattr__(self, "loss_levels", levels)
        require(isinstance(self.confidence_level, (float, int)) and math.isfinite(self.confidence_level)
                and 0 < self.confidence_level < 1, "invalid_ensemble_config", "confidence_level")
        for name, minimum in (("bootstrap_resamples", 100), ("bootstrap_seed", 0),
                              ("minimum_ci_samples", 2), ("minimum_tail_observations", 1)):
            value = getattr(self, name)
            require(type(value) is int and value >= minimum, "invalid_ensemble_config", name)
        require(math.isfinite(self.minimum_baseline_variance) and self.minimum_baseline_variance >= 0,
                "invalid_ensemble_config", "minimum_baseline_variance")


@dataclass(frozen=True)
class ScenarioResults:
    scenario_id: str
    replication_id: str
    source: str
    results: Mapping[str, BacktestResult]
    outer_randomness: Mapping | None = None


@dataclass(frozen=True)
class EnsembleReport:
    manifest: dict
    scenarios: tuple[dict, ...]
    strategies: dict
    comparisons: dict

    def to_json(self) -> str:
        return canonical(self) + "\n"

    def write(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "ensemble.json").write_text(self.to_json(), encoding="utf-8")
        # A row for every requested scenario/strategy, including absent runs.
        rows = [{"scenario_id": s["scenario_id"], "paired": s["paired"], **r}
                for s in self.scenarios for r in s["runs"].values()]
        with (path / "scenario_runs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def _number(value, label):
    require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value),
            "invalid_ensemble_result", label)
    return float(value)


def _conditions(result):
    """Freeze all manifest conventions, excluding only path/output identities."""
    m = deepcopy(result.manifest)
    required = ("config", "contracts", "portfolio", "provider", "code_revision", "risk_validation",
                "implementation_hash", "schedule", "market_history", "full_history_hash")
    require(all(k in m for k in required), "missing_comparison_identity", "complete result manifest required")
    require(m.get("document_kind") == "hedging_backtest_result", "invalid_ensemble_result", "document_kind")
    history = m["market_history"]
    require(bool(history) and m["full_history_hash"] == fingerprint(history),
            "invalid_ensemble_result", "full market history hash")
    require(m["schedule"] == [s["at"] for s in history], "invalid_ensemble_result", "schedule/history mismatch")
    # Preserve per-date native MC seeds/path counts, requested risk settings and
    # grids across outer scenarios. Custom providers must encode their settings
    # in provider.identity, as in the replay engine's own identity contract.
    controls = []
    for snapshot in history:
        requests = {}
        for cid, payload in snapshot["pricing_requests"].items():
            request = json.loads(payload)
            model = request["model"]
            grid = model.get("local_variance_grid", {})
            requests[cid] = {"engine": request["engine"], "risk": request["risk"],
                             "model_type": model["type"],
                             "grid": {k: v for k, v in grid.items() if k != "values"}}
        controls.append({"basis": snapshot["basis"], "requests": requests})
    for key in ("full_history_hash", "history_hash", "market_history", "identity_hash", "processed_snapshots"):
        m.pop(key, None)
    m["config"].pop("strategy")
    if "numeraire_policy" in m:
        # A stochastic common numeraire can vary by outer scenario. Its initial
        # normalization must still agree, and its path must match within a pair.
        m["numeraire_policy"].pop("input_hash", None)
    m["initial_numeraire"] = history[0]["numeraire"]
    m["pricing_controls"] = controls
    return m


def _inspect(strategy, result, config):
    if result is None:
        return {"strategy": strategy, "status": "missing", "reason": "missing_strategy",
                "terminal_error": None, "observations": None, "scheduled_snapshots": None,
                "coverage_fraction": None, "last_completed_at": None, "failure_code": None,
                "execution_cost": None, "absolute_traded_value": None, "result_hash": None}
    m, s, ledger = result.manifest, result.summary, result.ledger
    require(m["config"]["strategy"] == strategy, "strategy_mismatch", strategy)
    require(result.status in ("completed", "partial", "failed"), "invalid_ensemble_result", "status")
    require(s.get("status") == result.status and s.get("observations") == len(ledger)
            and s.get("scheduled_snapshots") == len(m["schedule"])
            and [r["at"] for r in ledger] == m["schedule"][:len(ledger)],
            "invalid_ensemble_result", "summary or ledger schedule inconsistent")
    require(result.status != "completed" or len(ledger) == len(m["schedule"]),
            "invalid_ensemble_result", "completed result must cover full schedule")
    require((result.failure is not None) == (result.status == "failed"),
            "invalid_ensemble_result", "failure details inconsistent")
    eligible = result.status == "completed" and bool(ledger) and ledger[-1]["fully_settled"] is True
    require(s.get("fully_settled") == (ledger[-1]["fully_settled"] if ledger else None),
            "invalid_ensemble_result", "settlement summary inconsistent")
    metric = s if config.metric == "nominal" else s.get("discounted_pnl")
    value = metric.get("terminal_hedging_error") if metric else None
    reason = result.status if result.status != "completed" else "unsettled" if not eligible else None
    if not eligible:
        require(value is None, "invalid_ensemble_result", "unfinished result has a terminal error")
    elif value is None:
        reason = "metric_unavailable"
    else:
        value = _number(value, "terminal error")
        final = ledger[-1]["cumulative_pnl" if config.metric == "nominal" else "discounted_cumulative_pnl"]
        require(value == final, "invalid_ensemble_result", "terminal error differs from final ledger")
    return {"strategy": strategy, "status": result.status, "reason": reason, "terminal_error": value,
            "observations": len(ledger), "scheduled_snapshots": len(m["schedule"]),
            "coverage_fraction": len(ledger) / len(m["schedule"]),
            "last_completed_at": ledger[-1]["at"] if ledger else None,
            "failure_code": result.failure.get("code") if result.failure else None,
            "execution_cost": _number(s["execution_cost"], "execution_cost"),
            "absolute_traded_value": _number(s["absolute_traded_value"], "absolute_traded_value"),
            "result_hash": fingerprint(data(result))}


def _moments(values):
    if not len(values):
        return {"mean": None, "standard_deviation": None, "rmse": None}
    # Scale before squaring, including for very large but finite errors.
    scale = max(abs(float(x)) for x in values)
    unit = [float(x) / scale if scale else 0. for x in values]
    result = {"mean": statistics.fmean(unit) * scale,
              "standard_deviation": statistics.stdev(unit) * scale if len(unit) > 1 else None,
              "rmse": math.sqrt(statistics.fmean(x*x for x in unit)) * scale}
    require(all(x is None or math.isfinite(x) for x in result.values()),
            "ensemble_nonfinite", "terminal moments overflow")
    return result


def _tail(values, level):
    losses = sorted(-float(x) for x in values)
    if not losses:
        return {"quantile": None, "expected_shortfall": None, "effective_tail_observations": 0.}
    mass = Decimal(len(losses)) * (1 - Decimal(str(level)))
    whole = int(mass)
    fractional = float(mass - whole)
    # Integral of the empirical inverse CDF above alpha. Fractional boundary
    # mass prevents including every tied observation or rounding up the tail.
    boundary = losses[len(losses) - whole - 1]
    weighted = [x / float(mass) for x in losses[len(losses)-whole:]]
    if fractional:
        weighted.append(boundary * (fractional / float(mass)))
    try:
        es = math.fsum(weighted)
    except OverflowError as error:
        raise BacktestError("ensemble_nonfinite", "expected shortfall overflow") from error
    require(math.isfinite(es), "ensemble_nonfinite", "expected shortfall overflow")
    return {"quantile": boundary, "expected_shortfall": es, "effective_tail_observations": float(mass)}


def _improvement(values, baseline, floor):
    a, b = _moments(values)["standard_deviation"], _moments(baseline)["standard_deviation"]
    if a is None or b is None or b <= math.sqrt(floor):
        return None
    ratio = a / b
    result = 1 - ratio * ratio
    require(math.isfinite(result), "ensemble_nonfinite", "variance improvement overflow")
    return result


def aggregate_scenarios(scenarios: Sequence[ScenarioResults], config: EnsembleConfig | None = None) -> EnsembleReport:
    config = config or EnsembleConfig()
    scenarios = sorted(scenarios, key=lambda s: s.scenario_id)
    require(bool(scenarios), "empty_ensemble", "at least one scenario required")
    require(len({s.scenario_id for s in scenarios}) == len(scenarios), "duplicate_scenario", "scenario_id")
    rows, condition_hash, shared_conditions = [], None, None
    histories, replications, streams = set(), set(), set()
    outer_family = None
    for scenario in scenarios:
        require(all(isinstance(x, str) and x.strip() for x in (scenario.scenario_id, scenario.replication_id, scenario.source)),
                "missing_scenario_provenance", "scenario_id, replication_id and source required")
        require(set(scenario.results) <= set(STRATEGIES),
                "invalid_scenario_strategies", scenario.scenario_id)
        if config.sampling_design == "independent_simulation":
            random = scenario.outer_randomness
            require(isinstance(random, Mapping) and all(isinstance(random.get(k), str) and random[k].strip()
                                                       for k in ("algorithm", "domain"))
                    and type(random.get("seed")) is int and random["seed"] >= 0,
                    "missing_outer_randomness", scenario.scenario_id)
            family = (scenario.source, random["algorithm"], random["domain"])
            require(outer_family is None or outer_family == family, "scenario_generator_mismatch", scenario.scenario_id)
            outer_family = family
            stream = (*family, random["seed"])
            require(scenario.replication_id not in replications and stream not in streams,
                    "dependent_replications", "independent simulation requires distinct replication IDs and outer seeds")
            streams.add(stream)
        replications.add(scenario.replication_id)
        pair_history = None
        for strategy, result in sorted(scenario.results.items()):
            conditions = _conditions(result)
            current_hash = fingerprint(conditions)
            require(condition_hash is None or current_hash == condition_hash,
                    "comparison_conditions_mismatch", f"{scenario.scenario_id}/{strategy}: book, schedule or conventions differ")
            condition_hash, shared_conditions = current_hash, conditions
            history_hash = result.manifest["full_history_hash"]
            require(pair_history is None or pair_history == history_hash,
                    "paired_history_mismatch", f"{scenario.scenario_id}: strategy inputs differ")
            pair_history = history_hash
        if pair_history is not None:
            require(pair_history not in histories, "duplicate_market_history", scenario.scenario_id)
            histories.add(pair_history)
        runs = {k: _inspect(k, scenario.results.get(k), config) for k in STRATEGIES}
        paired = all(r["reason"] is None for r in runs.values())
        rows.append({"scenario_id": scenario.scenario_id, "replication_id": scenario.replication_id,
                     "source": scenario.source, "outer_randomness": data(scenario.outer_randomness),
                     "full_history_hash": pair_history, "paired": paired, "runs": runs})
    require(shared_conditions is not None, "missing_comparison_identity", "at least one supplied result required to identify the cohort")
    selected = [r for r in rows if r["paired"]]
    samples = {k: [r["runs"][k]["terminal_error"] for r in selected] for k in STRATEGIES}
    count = len(selected)
    ci_reason = ("dependence_not_modelled" if config.sampling_design != "independent_simulation" else
                 "insufficient_samples" if count < config.minimum_ci_samples else None)
    strategies, comparisons = {}, {}
    for strategy, values in samples.items():
        run_rows = [r["runs"][strategy] for r in rows]
        tails = [{"level": level, **_tail(values, level)} for level in config.loss_levels]
        for tail in tails:
            tail["status"] = ("unavailable" if not count else "estimate" if
                              tail["effective_tail_observations"] >= config.minimum_tail_observations else "reference_only")
            tail["confidence_intervals"] = None
            tail["confidence_interval_reason"] = ci_reason or ("insufficient_tail_samples" if tail["status"] != "estimate" else None)
        strategies[strategy] = {"sample_count": count, **_moments(values), "loss_tail": tails,
            "run_counts": {status: sum(r["status"] == status for r in run_rows)
                           for status in ("completed", "partial", "failed", "missing")},
            "terminal_eligible_count": sum(r["reason"] is None for r in run_rows),
            "excluded_from_pair_count": len(rows)-count,
            "mean_execution_cost_nominal": _moments([r["runs"][strategy]["execution_cost"] for r in selected])["mean"],
            "mean_absolute_traded_value_nominal": _moments([r["runs"][strategy]["absolute_traded_value"] for r in selected])["mean"],
            "confidence_intervals": None, "confidence_interval_reason": ci_reason,
            "risk_validation": deepcopy(shared_conditions["risk_validation"])}
        if strategy != "delta":
            differences = [x-y for x, y in zip(values, samples["delta"])]
            require(all(math.isfinite(x) for x in differences), "ensemble_nonfinite", "paired differences overflow")
            improvement = _improvement(values, samples["delta"], config.minimum_baseline_variance)
            comparisons[strategy] = {"baseline": "delta", "sample_count": count,
                "mean_error_difference": _moments(differences)["mean"], "variance_improvement": improvement,
                "variance_improvement_reason": None if improvement is not None else "insufficient_or_near_zero_baseline_variance",
                "confidence_intervals": None, "confidence_interval_reason": ci_reason,
                "undefined_variance_bootstrap_resamples": None}
    if ci_reason is None:
        _bootstrap(samples, strategies, comparisons, config)
    manifest = {"document_kind": "hedging_backtest_ensemble",
        "schema_version": 5 if shared_conditions.get("schema_version") == 13 else 4 if shared_conditions.get("schema_version") == 12 else 3 if shared_conditions.get("schema_version") == 11 else 2, "config": data(config),
        "status": "complete_cohort" if count == len(rows) else "incomplete_cohort",
        "scenario_count": len(rows), "paired_sample_count": count, "excluded_scenario_count": len(rows)-count,
        "paired_scenario_ids": [r["scenario_id"] for r in selected],
        "distinct_replication_count": len(replications),
        "repeated_replication_count": len(rows)-len(replications),
        "overlapping_calendar_window_pairs": len(rows)*(len(rows)-1)//2,
        "observation_schedule": shared_conditions["schedule"],
        "population": "intersection_of_four_completed_fully_settled_runs_with_requested_metric",
        "selection_caveat": "complete_case_statistics_can_be_biased_when_runs_fail",
        "independence": "caller_declared_not_inferred_from_dates_or_tested",
        "inference_scope": "outer_scenarios_conditional_on_fixed_pricing_randomness",
        "bootstrap": "paired_scenario_percentile_pcg64_linear_quantiles",
        "tail_method": "empirical_inverse_cdf_and_fractional_upper_tail_integral",
        "loss_sign": "negative_terminal_hedging_error", "standard_deviation": "sample_ddof_1",
        "interpretation": "descriptive_statistics_not_numerical_acceptance",
        "risk_validation": deepcopy(shared_conditions["risk_validation"]),
        "conditions_hash": condition_hash, "shared_conditions": shared_conditions,
        "input_hash": fingerprint({"config": config, "scenarios": rows}),
        "implementation_hash": fingerprint({p.name: p.read_text(encoding="utf-8")
                                             for p in sorted(Path(__file__).parent.glob("*.py"))}),
        "python": platform.python_version(), "numpy": np.__version__}
    return EnsembleReport(manifest, tuple(rows), strategies, comparisons)


def _bootstrap(samples, strategies, comparisons, config):
    count = len(samples["delta"])
    rng = np.random.Generator(np.random.PCG64(config.bootstrap_seed))
    draws = {k: [] for k in STRATEGIES}
    paired_draws = {k: [] for k in comparisons}
    # One index vector per replicate, shared by all strategies and metrics.
    for _ in range(config.bootstrap_resamples):
        indexes = rng.integers(0, count, size=count)
        sample = {k: [values[i] for i in indexes] for k, values in samples.items()}
        for k, values in sample.items():
            draws[k].append({**_moments(values), "tails": [_tail(values, q) for q in config.loss_levels]})
        for k in comparisons:
            paired_draws[k].append({"mean_error_difference": _moments([x-y for x, y in zip(sample[k], sample["delta"])])["mean"],
                                   "variance_improvement": _improvement(sample[k], sample["delta"], config.minimum_baseline_variance)})
    alpha = (1-config.confidence_level)/2

    def interval(values):
        if any(v is None for v in values):
            return None
        scale = max(abs(x) for x in values)
        unit = [x / scale if scale else 0. for x in values]
        return [float(x)*scale for x in np.quantile(unit, [alpha, 1-alpha], method="linear")]

    for k, record in strategies.items():
        record["confidence_intervals"] = {m: interval([r[m] for r in draws[k]]) for m in ("mean", "standard_deviation", "rmse")}
        for i, tail in enumerate(record["loss_tail"]):
            if tail["status"] == "estimate":
                tail["confidence_intervals"] = {m: interval([r["tails"][i][m] for r in draws[k]])
                                                for m in ("quantile", "expected_shortfall")}
    for k, record in comparisons.items():
        invalid = sum(r["variance_improvement"] is None for r in paired_draws[k])
        record["undefined_variance_bootstrap_resamples"] = invalid
        record["confidence_intervals"] = {m: interval([r[m] for r in paired_draws[k]])
                                           for m in ("mean_error_difference", "variance_improvement")}
        # Do not silently condition the interval on bootstrap draws whose
        # denominator happens to be nonzero.
        if invalid:
            record["variance_interval_reason"] = "undefined_baseline_variance_in_bootstrap"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="ensemble request JSON with result paths relative to this file")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        request = json.loads(args.input.read_text(encoding="utf-8"))
        require(request["document_kind"] == "hedging_backtest_ensemble_request" and request["schema_version"] == 1,
                "schema_version", "unsupported ensemble request")
        scenarios = [ScenarioResults(**{k: v for k, v in row.items() if k != "results"},
                     results={k: BacktestResult.from_json((args.input.parent / p).read_text(encoding="utf-8"))
                              for k, p in row["results"].items()}) for row in request["scenarios"]]
        report = aggregate_scenarios(scenarios, EnsembleConfig(**request["config"]))
        report.write(args.output)
    except (BacktestError, ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"ensemble input rejected: {error}\n")
    print(canonical({k: report.manifest[k] for k in ("status", "scenario_count", "paired_sample_count", "risk_validation")}))
    return 0 if report.manifest["status"] == "complete_cohort" else 1


if __name__ == "__main__":
    raise SystemExit(main())
