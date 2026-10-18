"""Small independent outer-market simulation using diagnostic native VegaKT.

This is an integration example, not evidence of hedge effectiveness or adequate
tail estimation. Inner pricing randomness is fixed and shared by all strategies.
"""

import argparse
from pathlib import Path
import subprocess

from rust_pricing_backtest import BacktestConfig, compare_strategies
from rust_pricing_backtest.domain import canonical, data
from rust_pricing_backtest.ensemble import EnsembleConfig, ScenarioResults, aggregate_scenarios
from vega_kt_backtest import build_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/vega-kt-scenarios"))
    parser.add_argument("--scenarios", type=int, default=24)
    parser.add_argument("--first-market-seed", type=int, default=1000)
    parser.add_argument("--paths", type=int, default=512)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--numeraire-rate", type=float)
    args = parser.parse_args()
    if args.scenarios < 1 or args.first_market_seed < 0 or args.paths < 1:
        parser.error("positive scenario/path counts and nonnegative first market seed required")
    policy = EnsembleConfig(sampling_design="independent_simulation",
                            metric="discounted" if args.numeraire_rate is not None else "nominal")
    config = BacktestConfig(ridge=1e-4, solver_tolerance=1e-8, option_fee=.01, stock_fee=.001,
                            max_price_standard_error=2., max_abs_projection_residual=1000., max_abs_edge_sensitivity=1000.,
                            numeraire_id="synthetic_money_market" if args.numeraire_rate is not None else None)
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    args.output.mkdir(parents=True, exist_ok=True)
    request = {"document_kind": "hedging_backtest_ensemble_request", "schema_version": 1,
               "config": data(policy), "scenarios": []}
    for seed in range(args.first_market_seed, args.first_market_seed+args.scenarios):
        scenario_id = f"market-{seed}"
        request["scenarios"].append({"scenario_id": scenario_id, "replication_id": scenario_id,
            "source": "synthetic_essvi_outer_diffusion_v1",
            "outer_randomness": {"algorithm": "numpy.PCG64.SeedSequence", "domain": "vega_kt_outer_market_v1",
                                 "seed": seed, "seed_sequence_entropy": [0x56454741, seed]},
            "results": {k: f"{scenario_id}/{k}/result.json" for k in ("unhedged", "delta", "scalar_vega", "vega_kt")}})
    # Freeze the entire planned population before running any path. An early
    # process interruption must not leave a request describing only survivors.
    (args.output / "ensemble-input.json").write_text(canonical(request)+"\n", encoding="utf-8")
    scenarios = []
    for row in request["scenarios"]:
        seed = row["outer_randomness"]["seed"]
        history, contracts, portfolio, provider = build_case(args.paths, diagnostic_only=args.diagnostic_only,
                                                            market_seed=seed, numeraire_rate=args.numeraire_rate)
        results = compare_strategies(config, history, contracts, portfolio, provider, code_revision=revision, on_error="return")
        scenario_id = row["scenario_id"]
        scenario = ScenarioResults(**{k: v for k, v in row.items() if k != "results"}, results=results)
        scenarios.append(scenario)
        for strategy, result in results.items():
            result.write(args.output / scenario_id / strategy)
        print(canonical({"scenario_id": scenario_id, "statuses": {k: r.status for k, r in results.items()}}), flush=True)
    report = aggregate_scenarios(scenarios, policy)
    report.write(args.output)
    print(canonical({k: report.manifest[k] for k in ("status", "paired_sample_count", "risk_validation")}))
    return 0 if report.manifest["status"] == "complete_cohort" else 1


if __name__ == "__main__":
    raise SystemExit(main())
