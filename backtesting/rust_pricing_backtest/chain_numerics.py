"""Compare the explicit compiled-payoff paper chain with the existing flat panel.

This is a diagnostic comparison, never a substitute for the 154-check acceptance
panel. Keep native audit fields separate; do not invent a zero projection residual.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from .domain import fingerprint, require
from .numerics import black_scholes_call, flat_request, relative_distance, threshold_check


def cases(policy):
    n0, t0, k0 = (policy['base_sampling_units'], policy['base_time_steps'],
                  policy['base_strike_nodes'])
    return sorted({(s, n, t, k) for s in policy['seeds'] for n, t, k in
                   ([(n, t0, k0) for n in policy['sampling_units']]
                    + [(n0, t, k0) for t in policy['time_steps']]
                    + [(n0, t0, k) for k in policy['strike_nodes']])})


def run_comparison(policy, *, relative_bump=.04, subdivisions=2):
    import rust_pricing as rp
    require(policy['rate'] == 0, 'unsupported_chain_carry', 'unit-discount comparison required')
    settings = (policy['base_sampling_units'], policy['base_time_steps'],
                policy['base_strike_nodes'])
    oracle = black_scholes_call(policy['spot'], policy['strike'], policy['expiry_days']/365,
                               policy['volatility'], policy['rate'])
    observations, checks, errors = [], [], []
    started = time.perf_counter()
    for seed, samples, steps, nodes in cases(policy):
        args = dict(seed=seed, samples=samples, steps=steps, nodes=nodes)
        request = flat_request(policy, **args)
        plan = rp.PricingPlan.compile(request, worker_threads=1, reduction_block_size=256)
        baseline = plan.evaluate()
        strikes = (policy['spot'] * np.exp(np.linspace(-policy['log_moneyness_extent'],
                                                     policy['log_moneyness_extent'], nodes))).tolist()
        before = time.perf_counter()
        try:
            chain = plan.evaluate_vega_kt_chain_diagnostic(
                strikes, relative_bump=relative_bump, subdivisions=subdivisions)
        except rp.PricingError as error:
            errors.append({**args, 'error': str(error), 'request_fingerprint': plan.request_fingerprint})
            continue
        elapsed = time.perf_counter()-before
        raw = np.asarray(chain.raw_buckets)
        total = chain.scalar_vega
        se = chain.scalar_standard_error
        maturity_totals = raw.reshape(steps, nodes).sum(axis=1).tolist()
        checks.append(threshold_check('chain_sum_vs_local_parallel_vega',
                                      relative_distance(total, baseline.vega_raw),
                                      policy['parallel_vega_relative_tolerance'], **args))
        checks.append(threshold_check('chain_sum_vs_analytic_with_sampling_uncertainty',
                                      abs(total-oracle['vega']),
                                      policy['statistical_multiplier']*se
                                      + policy['parallel_vega_relative_tolerance']*abs(oracle['vega']), **args))
        row = {**args, 'price': baseline.value, 'local_vega': baseline.vega_raw,
               'local_vega_se': baseline.vega.raw.standard_error,
               'legacy_bucket_sum': math.fsum(baseline.vega_kt.raw_buckets),
               'legacy_signed_residual': baseline.vega_kt.projection.signed_residual,
               'chain_bucket_sum': total, 'chain_bucket_sum_se': se,
               'chain_relative_analytic_error': relative_distance(total, oracle['vega']),
               'maturity_totals': maturity_totals,
               # Exact vanilla risk belongs to its terminal expiry; this is an
               # exploratory localization diagnostic without a new pass threshold.
               'nonterminal_bucket_l1_over_analytic_vega':
                   float(np.abs(raw.reshape(steps, nodes)[:-1]).sum())/abs(oracle['vega']),
               'chain_seconds': elapsed, 'chain': chain.to_dict()}
        if (samples, steps, nodes) == settings:
            crn = []
            for bump in policy['volatility_bumps']:
                values = []
                for sigma in [policy['volatility']+bump, policy['volatility']-bump]:
                    bumped = flat_request(policy, **args, volatility=sigma, risk=False)
                    values.append(rp.PricingPlan.compile(bumped, worker_threads=1,
                                                        reduction_block_size=256).evaluate().value)
                derivative = (values[0]-values[1])/(2*bump)
                crn.append(derivative)
                checks.append(threshold_check('chain_sum_vs_crn', relative_distance(total, derivative),
                                              policy['parallel_vega_relative_tolerance'], bump=bump, **args))
            row['crn_parallel_vegas'] = crn
        observations.append(row)
    indexed = {(o['seed'], o['samples'], o['steps'], o['nodes']): o for o in observations}
    n0, t0, k0 = settings
    for seed in policy['seeds']:
        reference = indexed.get((seed, n0, t0, k0))
        if reference is None:
            continue
        for k in policy['strike_nodes']:
            o = indexed.get((seed, n0, t0, k))
            if o is not None:
                checks.append(threshold_check('chain_sum_grid_stability',
                    relative_distance(o['chain_bucket_sum'], reference['chain_bucket_sum']),
                    policy['spatial_refinement_relative_tolerance'], seed=seed, nodes=k))
    failed = any(not c['passed'] for c in checks) or bool(errors)
    module = getattr(rp, 'rust_pricing', rp)
    return {'document_kind': 'vega_kt_chain_numerical_comparison', 'schema_version': 1,
            'diagnostic_only': True,
            'status': 'diagnostic_checks_failed' if failed else 'diagnostic_checks_passed',
            'policy': policy, 'policy_hash': fingerprint(policy),
            'method_config': {'relative_bump': relative_bump, 'subdivisions': subdivisions,
                              'requested_initial_strikes': 'spot * exp(local-grid nodes)',
                              'native_refinement': 'min(relative_bump/2, sqrt(first_step_variance)/2); 1025-node budget'},
            'native_binary_hash': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            'runner_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'oracle': oracle, 'observations': observations, 'checks': checks, 'errors': errors,
            'elapsed_seconds': time.perf_counter()-started,
            'limitations': [
                'This 63-check comparison is not the unchanged 154-check native acceptance panel.',
                'No synthetic projection residual or success normalization is used.',
                'The chain is limited to flat spatial variance, zero carry and exact reporting expiries.',
                'Sampling errors do not bound weak recovery, initial Gamma or finite-domain bias.',
                'Nonterminal risk is reported without introducing a new localization acceptance threshold.',
                'No hedge-optimizer stability or complete H2/H3 acceptance is claimed.']}


def markdown_report(r):
    lines = ['# コンパイル済みペイオフのVegaKT連結比較', '',
             f"診断判定: **{r['status']}**。{len(r['checks'])}件中"
             f"{sum(not c['passed'] for c in r['checks'])}件不合格、実行エラー{len(r['errors'])}件。",
             '既存154件の本体受入とは別の比較です。', '',
             '| Seed | 標本単位 | 区間 | Strike | 従来合計 | 連結合計 | 連結SE | 解析差率 |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for o in r['observations']:
        lines.append(f"| {o['seed']} | {o['samples']} | {o['steps']} | {o['nodes']} | "
                     f"{o['legacy_bucket_sum']:.6f} | {o['chain_bucket_sum']:.6f} | "
                     f"{o['chain_bucket_sum_se']:.6f} | {o['chain_relative_analytic_error']:.2%} |")
    lines += ['', '詳細な初期Gamma・積分・tail監査と満期配賦は `report.json` に保存しています。',
              f"解析Vega: {r['oracle']['vega']:.9f}。", '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--relative-bump', type=float, default=.04)
    p.add_argument('--subdivisions', type=int, default=2)
    a = p.parse_args()
    report = run_comparison(json.loads(a.policy.read_text()), relative_bump=a.relative_bump,
                            subdivisions=a.subdivisions)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (a.output/'report.md').write_text(markdown_report(report))
    print(json.dumps({'status': report['status'], 'checks': len(report['checks']),
                      'failed': sum(not c['passed'] for c in report['checks']),
                      'errors': len(report['errors']), 'output': str(a.output)}))
    return int(report['status'] == 'diagnostic_checks_failed')


if __name__ == '__main__':
    sys.exit(main())
