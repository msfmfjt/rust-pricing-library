"""Measure a declared synthetic backtest workload; never infer numerical acceptance."""

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .domain import BacktestConfig, Contract, Position, Quote, RiskBasis, Snapshot, fingerprint, require
from .engine import run_backtest
from .numerics import black_scholes_call
from .rust_adapter import RustPricingProvider


def build_workload(policy):
    import rust_pricing as rp
    start = date.fromisoformat(policy["valuation_date"])
    expiry = start + timedelta(days=policy["expiry_days"])
    targets = [Contract(f"target-{i:02d}", expiry, 95.25+i, lot=.1)
               for i in range(policy["target_contracts"])]
    hedges = [Contract(f"hedge-{i:02d}", expiry, 80.+2*i, lot=.1)
              for i in range(policy["hedge_candidates"])]
    contracts = targets+hedges
    history = []
    day = start
    while len(history) < policy["sessions"]:
        if day.weekday() < 5:
            i = len(history)
            at = datetime.combine(day, datetime.min.time(), timezone.utc)+timedelta(hours=16)
            spot = 100 * math.exp(.03*math.sin(i/23))
            sigma = .2+.01*math.sin(i/31)
            remaining = (expiry-day).days
            dates = tuple(day+timedelta(days=remaining*j//policy["maturity_nodes"])
                          for j in range(1, policy["maturity_nodes"]+1))
            times = np.array([(d-day).days/365 for d in dates])
            x = np.linspace(-.8, .8, policy["strike_nodes"])
            basis = RiskBasis(day, dates, tuple(x))
            model = rp.Model.local_volatility_from_grid_with_reporting_basis(
                np.array([0., *times]), x, np.full((len(times)+1)*len(x), sigma*sigma),
                1e-8, 4., times, x, np.full(len(times)*len(x), sigma))
            market = rp.Market.equity(2, 1, spot,
                rp.DiscountCurve(10, np.array([0., 2.]), np.array([1., 1.])),
                rp.DiscountCurve(11, np.array([0., 2.]), np.array([1., 1.])))
            requests, quotes = {}, {}
            for c in contracts:
                request = rp.PricingRequest(day, rp.Product.european_vanilla(1, 2, expiry, c.strike, 1., "call"),
                    market, model, rp.Engine.pseudo_monte_carlo(policy["seed"], policy["independent_sampling_units"], antithetic=True),
                    rp.RiskRequest(delta=True, vega=True, vega_kt_maturity_nodes=dates,
                                   vega_kt_log_forward_moneyness_nodes=x, vega_kt_relative_density_threshold=1e-12))
                requests[c.contract_id] = request.to_json()
                mark = black_scholes_call(spot, c.strike, remaining/365, sigma)["price"]
                quotes[c.contract_id] = Quote(max(0., mark-.01), mark+.01, at, at)
            history.append(Snapshot(f"benchmark-{i:03d}", at, Quote(spot-.005, spot+.005, at, at),
                quotes, basis, at, "synthetic_constant_vol_performance_v1", requests))
        day += timedelta(days=1)
    positions = [Position(c.contract_id, -1., history[0].option_quotes[c.contract_id].mid) for c in targets]
    # Target-only contracts must not be offered as hedge candidates: the runner
    # excludes their identical economic contracts under its default policy.
    return history, contracts, positions


class TimedProvider:
    def __init__(self, inner):
        self.inner, self.calls, self.seconds = inner, 0, 0.

    @property
    def identity(self):
        return self.inner.identity

    @property
    def validation(self):
        return self.inner.validation

    def evaluate(self, snapshot, contract):
        started = time.perf_counter()
        result = self.inner.evaluate(snapshot, contract)
        self.seconds += time.perf_counter()-started
        self.calls += 1
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    provider = TimedProvider(RustPricingProvider(diagnostic_only=args.diagnostic_only,
        worker_threads=policy["worker_threads"], reduction_block_size=policy["reduction_block_size"]))
    before = time.perf_counter()
    history, contracts, portfolio = build_workload(policy)
    built = time.perf_counter()
    result = run_backtest(BacktestConfig(strategy=policy["strategy"], ridge=policy["ridge"],
        solver_tolerance=policy["solver_tolerance"], option_position_limit=20, option_trade_limit=20,
        max_relative_spread=policy["max_relative_spread"], min_vega=policy["min_vega"]),
        history, contracts, portfolio, provider)
    ran = time.perf_counter()
    expected_calls = (policy["sessions"]-1)*len(contracts)+len(portfolio)
    require(provider.calls == expected_calls and len(result.decisions) == policy["sessions"]-1
            and all(len(d["candidates"]) == policy["hedge_candidates"] for d in result.decisions),
            "benchmark_workload", "every decision must include the declared complete hedge universe")
    result.write(args.output / "backtest")
    saved = time.perf_counter()
    peak_mib = None
    if sys.platform in ("darwin", "linux"):
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mib = usage / (1024**2 if sys.platform == "darwin" else 1024)
    report = {"document_kind": "backtest_diagnostic_performance", "schema_version": 1,
              "status": result.status, "risk_validation": provider.validation,
              "performance_acceptance": "not_assessed_no_approved_budget", "policy": policy,
              "policy_hash": fingerprint(policy), "provider": provider.identity,
              "implementation_hash": result.manifest["implementation_hash"],
              "python": sys.version.split()[0], "platform": platform.platform(),
              "workload_build_seconds": built-before, "replay_seconds": ran-built,
              "provider_seconds_including_validation_compile_and_risk": provider.seconds,
              "provider_calls": provider.calls, "workload_verified": True, "write_seconds": saved-ran,
              "total_seconds": saved-before, "process_peak_rss_mib": peak_mib,
              "output_bytes": sum(p.stat().st_size for p in (args.output/"backtest").rglob("*") if p.is_file()),
              "summary": result.summary}
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "risk_validation", "replay_seconds", "provider_calls",
                                          "total_seconds", "process_peak_rss_mib")}, indent=2))


if __name__ == "__main__":
    main()
