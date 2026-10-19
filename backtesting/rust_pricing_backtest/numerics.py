"""Independent constant-volatility acceptance panel through public pricing APIs.

Exit 1 means a numerical acceptance failure; it is never a successful backtest.
Passing this necessary panel alone does not certify the LV paper decomposition.
"""

import argparse
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from .domain import fingerprint, require
from .optimizer import solve_hedge


def black_scholes_call(spot, strike, maturity, volatility, rate=0.0):
    """Closed-form oracle independent of the native engine (zero dividends)."""
    require(min(spot, strike, maturity, volatility) > 0, "invalid_oracle", "positive inputs required")
    d1 = (math.log(spot / strike) + (rate + volatility**2 / 2) * maturity) / (volatility * math.sqrt(maturity))
    d2 = d1 - volatility * math.sqrt(maturity)
    cdf = lambda x: .5 * math.erfc(-x / math.sqrt(2))
    return {"price": spot * cdf(d1) - strike * math.exp(-rate * maturity) * cdf(d2),
            "delta": cdf(d1),
            "vega": spot * math.exp(-d1*d1/2) / math.sqrt(2*math.pi) * math.sqrt(maturity)}


def flat_request(policy, *, seed, samples, steps, nodes, strike=None, volatility=None, risk=True):
    """Shift IV and its matching constant local variance together, not isolated IV nodes."""
    import rust_pricing as rp
    start = date.fromisoformat(policy["valuation_date"])
    days = policy["expiry_days"]
    require(days % steps == 0, "invalid_panel", "time steps must divide expiry days exactly")
    dates = [start + timedelta(days=days*i//steps) for i in range(1, steps+1)]
    times = np.array([(d-start).days/365 for d in dates])
    x = np.linspace(-policy["log_moneyness_extent"], policy["log_moneyness_extent"], nodes)
    sigma = policy["volatility"] if volatility is None else volatility
    model = rp.Model.local_volatility_from_grid_with_reporting_basis(
        np.array([0., *times]), x, np.full((steps+1)*nodes, sigma*sigma), 1e-8, 4.,
        times, x, np.full(steps*nodes, sigma))
    market = rp.Market.equity(
        2, 1, policy["spot"],
        rp.DiscountCurve(10, np.array([0., 1.]), np.array([1., math.exp(-policy["rate"])])),
        rp.DiscountCurve(11, np.array([0., 1.]), np.array([1., 1.])))
    risks = (rp.RiskRequest(delta=True, vega=True, vega_kt_maturity_nodes=dates,
                           vega_kt_log_forward_moneyness_nodes=x,
                           vega_kt_relative_density_threshold=policy["density_threshold"])
             if risk else rp.RiskRequest())
    return rp.PricingRequest(start, rp.Product.european_vanilla(
        1, 2, dates[-1], policy["strike"] if strike is None else strike, 1., "call"),
        market, model, rp.Engine.pseudo_monte_carlo(seed, samples, antithetic=True), risks)


def threshold_check(name, observed, limit, **context):
    """Fail nonfinite observations rather than writing invalid JSON or accepting NaN."""
    finite = math.isfinite(observed) and math.isfinite(limit) and limit >= 0
    return {"check": name, "passed": finite and observed <= limit,
            "observed": observed if math.isfinite(observed) else None,
            "limit": limit if math.isfinite(limit) else None, **context}


def relative_distance(value, reference):
    return float(np.linalg.norm(np.asarray(value) - np.asarray(reference)) / max(np.linalg.norm(reference), 1e-12))


def run_panel(policy):
    import rust_pricing as rp
    module = getattr(rp, "rust_pricing", rp)
    binary = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    settings = (policy["base_sampling_units"], policy["base_time_steps"], policy["base_strike_nodes"])
    n0, t0, k0 = settings
    cases = sorted({(seed, n, t, k) for seed in policy["seeds"]
                    for n, t, k in ([(n, t0, k0) for n in policy["sampling_units"]]
                                    + [(n0, t, k0) for t in policy["time_steps"]]
                                    + [(n0, t0, k) for k in policy["strike_nodes"]])})
    oracle = black_scholes_call(policy["spot"], policy["strike"], policy["expiry_days"]/365,
                               policy["volatility"], policy["rate"])
    observations, checks, evaluations = [], [], []
    started = time.perf_counter()

    def evaluate(**kwargs):
        request = flat_request(policy, **kwargs)
        before = time.perf_counter()
        plan = rp.PricingPlan.compile(request, worker_threads=1, reduction_block_size=256)
        compiled = time.perf_counter()
        result = plan.evaluate()
        finished = time.perf_counter()
        evaluations.append({"request_hash": fingerprint(json.loads(request.to_json())),
                            "compile_seconds": compiled-before, "evaluate_seconds": finished-compiled,
                            **kwargs})
        return result

    for seed, n, t, k in cases:
        args = dict(seed=seed, samples=n, steps=t, nodes=k)
        result = evaluate(**args)
        kt = result.vega_kt
        buckets = list(kt.raw_buckets)
        total = math.fsum(buckets)
        residual = kt.projection.signed_residual
        observation = {**args, "native_policy": kt.policy_label, "native_truncation_order": kt.truncation_order,
                       "price": result.value, "price_se": result.standard_error,
                       "delta": result.delta_raw, "delta_se": result.delta.raw.standard_error,
                       "local_vega": result.vega_raw, "local_vega_se": result.vega.raw.standard_error,
                       "bucket_sum": total, "signed_residual": residual,
                       "pre_projection": kt.projection.pre_projection,
                       "buckets": buckets, "maturity_totals": np.array(buckets).reshape(t, k).sum(axis=1).tolist()}
        for field in ("price", "delta", "local_vega"):
            ref = oracle["vega" if field == "local_vega" else field]
            extra = (policy["parallel_vega_relative_tolerance"] * abs(ref) if field == "local_vega"
                     else policy[field+"_absolute_tolerance"])
            checks.append(threshold_check(field+"_analytic", abs(observation[field]-ref),
                          policy["statistical_multiplier"] * observation[field+"_se"] + extra, **args))
        checks.append(threshold_check("bucket_sum_vs_local_parallel_vega", relative_distance(total, result.vega_raw),
                                      policy["parallel_vega_relative_tolerance"], **args))
        checks.append(threshold_check("projection_residual", abs(residual)/max(abs(kt.projection.pre_projection), 1e-12),
                                      policy["projection_residual_relative_tolerance"], **args))

        # Quantity stability is a distinct diagnostic: passing it cannot validate
        # the buckets themselves. A short target is hedged with two different calls.
        if (n, t) == (n0, t0):
            hedge_buckets = [evaluate(**args, strike=strike).vega_kt.raw_buckets for strike in policy["hedge_strikes"]]
            m = len(hedge_buckets)
            bound = policy["hedge_position_limit"]
            solution = solve_hedge(-np.array(buckets), np.array(hedge_buckets).T, [0.]*m,
                                   [-bound]*m, [bound]*m, [0.]*m, ridge=policy["hedge_ridge"],
                                   tolerance=policy["solver_tolerance"])
            observation["hedge_quantities"] = list(solution.quantities)
        if (n, t, k) == settings:
            bumps = []
            for bump in policy["volatility_bumps"]:
                up = evaluate(**args, volatility=policy["volatility"]+bump, risk=False).value
                down = evaluate(**args, volatility=policy["volatility"]-bump, risk=False).value
                crn = (up-down)/(2*bump)
                bumps.append(crn)
                checks.append(threshold_check("local_vega_vs_crn", relative_distance(result.vega_raw, crn),
                                              policy["crn_bump_relative_tolerance"], bump=bump, **args))
                checks.append(threshold_check("bucket_sum_vs_crn", relative_distance(total, crn),
                                              policy["parallel_vega_relative_tolerance"], bump=bump, **args))
            observation["crn_parallel_vegas"] = bumps
        observations.append(observation)

    indexed = {(x["seed"], x["samples"], x["steps"], x["nodes"]): x for x in observations}
    reference = indexed[(policy["seeds"][0], *settings)]
    for seed in policy["seeds"]:
        base = indexed[(seed, *settings)]
        checks.append(threshold_check("hedge_seed_stability", relative_distance(base["hedge_quantities"], reference["hedge_quantities"]),
                                      policy["hedge_quantity_relative_tolerance"], seed=seed))
        for k in policy["strike_nodes"]:
            obs = indexed[(seed, n0, t0, k)]
            checks.append(threshold_check("bucket_sum_grid_stability", relative_distance(obs["bucket_sum"], base["bucket_sum"]),
                                          policy["spatial_refinement_relative_tolerance"], seed=seed, nodes=k))
            checks.append(threshold_check("hedge_grid_stability", relative_distance(obs["hedge_quantities"], base["hedge_quantities"]),
                                          policy["hedge_quantity_relative_tolerance"], seed=seed, nodes=k))
    low_n, high_n = min(policy["sampling_units"]), max(policy["sampling_units"])
    ratio = math.sqrt(math.fsum(indexed[(s, high_n, t0, k0)]["price_se"]**2 for s in policy["seeds"])
                      / math.fsum(indexed[(s, low_n, t0, k0)]["price_se"]**2 for s in policy["seeds"]))
    checks.append(threshold_check("price_se_sampling_refinement", ratio, policy["max_standard_error_ratio"]))
    passed = all(x["passed"] for x in checks)
    return {"document_kind": "backtest_numerical_acceptance", "schema_version": 1,
            "status": "passed" if passed else "failed", "policy": policy, "policy_hash": fingerprint(policy),
            "native_binary_hash": binary, "runner_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "execution": {"worker_threads": 1, "reduction_block_size": 256, "antithetic": True},
            "oracle": oracle, "observations": observations, "checks": checks, "evaluations": evaluations,
            "elapsed_seconds": time.perf_counter()-started,
            "limitations": ["Necessary constant-volatility panel only; not complete H2/H3 acceptance.",
                            "No cross-bucket standard error is inferred from marginal errors.",
                            "CRN shifts both flat IV and matching local variance; no independent quote-node bump.",
                            "Quantity stability cannot validate an incorrect risk definition."]}


def markdown_report(report):
    failed = [c for c in report["checks"] if not c["passed"]]
    lines = ["# VegaKT 数値検証", "", f"判定: **{report['status']}**。{len(report['checks'])}項目中{len(failed)}項目が不合格。",
             "この検証は一定ボラティリティでの必要条件であり、論文方式全体の受入を意味しません。", "",
             f"解析解の Vega: {report['oracle']['vega']:.8f}", "",
             "| Seed | パス単位 | 時間区間 | Strike節点 | Local Vega | VegaKT合計 | 残差 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for x in report["observations"]:
        lines.append(f"| {x['seed']} | {x['samples']} | {x['steps']} | {x['nodes']} | {x['local_vega']:.6f} | {x['bucket_sum']:.6f} | {x['signed_residual']:.6f} |")
    lines += ["", "不合格の検証種別: " + (", ".join(sorted({x['check'] for x in failed})) or "なし"), "",
              f"計測時間: {report['elapsed_seconds']:.3f}秒。価格計算・検証用であり、252日バックテストの性能値ではありません。", "",
              f"Native binary SHA-256: `{report['native_binary_hash']}`", "",
              f"Policy SHA-256: `{report['policy_hash']}`", "",
              "全観測値・判定閾値・入力指紋・計測内訳は `report.json` に保存しています。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    report = run_panel(policy)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    report["checkout_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (args.output/"report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(report["checks"]),
                      "failed": sum(not c["passed"] for c in report["checks"]), "output": str(args.output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
