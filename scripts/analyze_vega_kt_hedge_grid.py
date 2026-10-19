#!/usr/bin/env python3
"""Explain hedge-grid failures without editing/replacing the frozen acceptance policy.

Refining independent IV quote hats changes the hedge objective. Compare the
original raw-hat objective with a fixed family of coarse quote shocks, using
the chain rule for a nested grid. Both native MC and independent exact risks
are retained; this is an investigation, not a new acceptance certificate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backtesting'))
from rust_pricing_backtest.domain import fingerprint
from rust_pricing_backtest.numerics import black_scholes_call, flat_request, relative_distance
from rust_pricing_backtest.optimizer import solve_hedge


def hats(nodes, x):
    w = np.zeros(len(nodes))
    upper = int(np.searchsorted(nodes, x, side='right'))
    if upper == 0:
        w[0] = 1.
    elif upper == len(nodes):
        w[-1] = 1.
    else:
        alpha = (x-nodes[upper-1])/(nodes[upper]-nodes[upper-1])
        w[upper-1], w[upper] = 1-alpha, alpha
    return w


def quantities(target, hedges, policy):
    n = len(hedges)
    bound = policy['hedge_position_limit']
    return solve_hedge(-np.asarray(target), np.asarray(hedges).T, [0.]*n,
                       [-bound]*n, [bound]*n, [0.]*n,
                       ridge=policy['hedge_ridge'], tolerance=policy['solver_tolerance']).quantities


def analyze(policy):
    kmin = min(policy['strike_nodes'])
    common = np.linspace(-policy['log_moneyness_extent'], policy['log_moneyness_extent'], kmin)
    strikes = [policy['strike'], *policy['hedge_strikes']]
    t = policy['expiry_days']/365
    nsteps = policy['base_time_steps']
    rows = []
    for seed in policy['seeds']:
        for count in policy['strike_nodes']:
            nodes = np.linspace(common[0], common[-1], count)
            if any(min(abs(nodes-x)) > 1e-14 for x in common):
                raise ValueError('comparison requires nested quote nodes')
            # d sigma_fine / d sigma_common; transpose action on risk covectors.
            shock_map = np.array([hats(common, x) for x in nodes])
            native, exact, restricted_native, restricted_exact = [], [], [], []
            for strike in strikes:
                request = flat_request(policy, seed=seed, samples=policy['base_sampling_units'],
                                       steps=nsteps, nodes=count, strike=strike)
                output = rp.PricingPlan.compile(request, worker_threads=1, reduction_block_size=256).evaluate()
                if output.vega_kt.policy_label != 'flat_variance_terminal_measure_pathwise_v1':
                    raise ValueError('terminal-measure native policy required')
                v = np.array(output.vega_kt.raw_buckets).reshape(nsteps, count)
                ref = np.zeros_like(v)
                ref[-1] = black_scholes_call(policy['spot'], strike, t, policy['volatility'])['vega']*hats(nodes, math.log(strike/policy['spot']))
                native.append(v.ravel())
                exact.append(ref.ravel())
                restricted_native.append((v@shock_map).ravel())
                restricted_exact.append((ref@shock_map).ravel())
            rows.append(dict(seed=seed, nodes=count,
                native_original_quantities=quantities(native[0], native[1:], policy),
                exact_original_quantities=quantities(exact[0], exact[1:], policy),
                native_common_shock_quantities=quantities(restricted_native[0], restricted_native[1:], policy),
                exact_common_shock_quantities=quantities(restricted_exact[0], restricted_exact[1:], policy),
                common_shock_raw_risks=[v.tolist() for v in restricted_native]))
    for row in rows:
        base = next(r for r in rows if r['seed']==row['seed'] and r['nodes']==policy['base_strike_nodes'])
        common_base = next(r for r in rows if r['seed']==row['seed'] and r['nodes']==kmin)
        row['exact_original_relative_change'] = relative_distance(row['exact_original_quantities'],base['exact_original_quantities'])
        row['native_common_relative_change'] = relative_distance(row['native_common_shock_quantities'],common_base['native_common_shock_quantities'])
        row['exact_common_relative_change'] = relative_distance(row['exact_common_shock_quantities'],common_base['exact_common_shock_quantities'])
        row['native_common_max_risk_difference'] = float(np.max(np.abs(np.array(row['common_shock_raw_risks'])-np.array(common_base['common_shock_raw_risks']))))
    return dict(document_kind='vega_kt_hedge_grid_investigation', diagnostic_only=True,
                policy_hash=fingerprint(policy), original_policy_unchanged=True,
                common_quote_nodes=common.tolist(), observations=rows,
                exact_original_failure_count=sum(r['exact_original_relative_change']>policy['hedge_quantity_relative_tolerance'] for r in rows),
                max_native_common_quantity_change=max(r['native_common_relative_change'] for r in rows),
                max_native_common_risk_difference=max(r['native_common_max_risk_difference'] for r in rows),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitations=['Common shocks restrict the risk objective; they do not demonstrate hedging of independent fine-grid quote shocks.',
                             'Nested linear log-IV hats and unchanged maturity grid only.',
                             'The original six failed hedge-grid comparisons remain failed; no threshold is changed.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads((ROOT/'fixtures/backtesting/numerical-policy-v0.1.json').read_text())
    report = analyze(policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['exact_original_failure_count', 'max_native_common_quantity_change', 'max_native_common_risk_difference']}))


if __name__ == '__main__':
    main()
