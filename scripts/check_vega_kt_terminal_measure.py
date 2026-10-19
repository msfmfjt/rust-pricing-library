#!/usr/bin/env python3
"""Independent terminal-measure diagnostic comparison, NOT the native acceptance gate.

Preserves the original deterministic 3% comparison, even when it fails because
of MC noise. Separate tests check exact bucket localization and source-hat
expectations via unconditional source-state quadrature (not the native bridge).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from scipy.integrate import quad
import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backtesting'))
from rust_pricing_backtest.chain_numerics import cases
from rust_pricing_backtest.domain import fingerprint
from rust_pricing_backtest.numerics import black_scholes_call, flat_request


def source_hat_reference(spot, strike, total, row, nodes):
    u = row['source_total_variance']
    w = total-u
    sigma_dt = row['interval_volatility']*(row['destination_maturity']-row['source_maturity'])
    pdf = lambda z: math.exp(-z*z/2)/math.sqrt(2*math.pi)
    if u == 0:
        scale = sigma_dt*spot*pdf(math.log(spot/strike)/math.sqrt(w)+.5*math.sqrt(w))/math.sqrt(w)
        return [scale if n == 0 else 0. for n in nodes]  # panel has an exact zero node
    out = []
    for j in range(len(nodes)):
        intervals = []
        if j == 0:
            intervals.append((-math.inf, nodes[0], lambda x: 1.))
        else:
            a, b = nodes[j-1], nodes[j]
            intervals.append((a, b, lambda x, a=a, b=b: (x-a)/(b-a)))
        if j == len(nodes)-1:
            intervals.append((nodes[-1], math.inf, lambda x: 1.))
        else:
            a, b = nodes[j], nodes[j+1]
            intervals.append((a, b, lambda x, a=a, b=b: (b-x)/(b-a)))
        def integrand(z, hat):
            x = -.5*u+math.sqrt(u)*z
            s = spot*math.exp(x)
            gamma = pdf(math.log(s/strike)/math.sqrt(w)+.5*math.sqrt(w))/(s*math.sqrt(w))
            return sigma_dt*s*s*gamma*hat(x)*pdf(z)
        value = 0.
        for a, b, hat in intervals:
            lo, hi = max(-12., (a+.5*u)/math.sqrt(u)), min(12., (b+.5*u)/math.sqrt(u))
            if lo < hi:
                value += quad(integrand, lo, hi, args=(hat,), epsabs=1e-12, epsrel=2e-12)[0]
        out.append(value)
    return out


def run(policy):
    observations, checks, original_comparisons = [], [], []
    oracle = black_scholes_call(policy['spot'], policy['strike'], policy['expiry_days']/365,
                               policy['volatility'])['vega']
    for seed, samples, steps, nodes in cases(policy):
        args = dict(seed=seed, samples=samples, steps=steps, nodes=nodes)
        request = flat_request(policy, **args)
        plan = rp.PricingPlan.compile(request, worker_threads=1, reduction_block_size=256)
        native = plan.evaluate()
        d = plan.evaluate_vega_kt_terminal_diagnostic()
        payload = d.to_dict()
        exact = [0.]*len(d.raw_buckets)
        exact[(steps-1)*nodes+nodes//2] = oracle
        bucket_error = max(abs(a-b) for a, b in zip(d.raw_buckets, exact))
        row_error = max(abs(a-b) for row in payload['local_rows'] for a, b in zip(
            row['expected_source_hat_vegas'], source_hat_reference(
                policy['spot'], policy['strike'], payload['total_variance'], row,
                payload['source_log_moneyness_nodes'])))
        expected_local_error = abs(d.expected_local_parallel_vega-oracle)
        se = native.vega.raw.standard_error
        gap = abs(d.expected_local_parallel_vega-native.vega_raw)
        for name, observed, limit in [
            ('terminal_bucket_absolute_error', bucket_error, 2e-11),
            ('source_hat_unconditional_quadrature', row_error, 2e-11),
            ('expected_local_parallel_absolute_error', expected_local_error, 2e-11),
            # Deterministic expectation has no sampling variance; difference SE
            # is exactly native SE. This is a diagnostic, not a revised policy.
            ('expected_local_vs_native_six_se', gap, 6*se),
        ]:
            checks.append(dict(check=name, passed=observed <= limit, observed=observed, limit=limit, **args))
        relative = abs(d.scalar_vega-native.vega_raw)/abs(native.vega_raw)
        original_comparisons.append(dict(check='original_three_percent_vs_native',
            passed=relative <= policy['parallel_vega_relative_tolerance'], observed=relative,
            limit=policy['parallel_vega_relative_tolerance'], standard_errors=gap/se, **args))
        observations.append(dict(**args, terminal=payload, native_local_vega=native.vega_raw,
                                 native_local_vega_se=se, ordinary_bucket_sum=sum(native.vega_kt.raw_buckets),
                                 ordinary_signed_residual=native.vega_kt.projection.signed_residual,
                                 ordinary_policy=native.vega_kt.policy_label))
    module = getattr(rp, 'rust_pricing', rp)
    return dict(document_kind='vega_kt_terminal_measure_validation', schema_version=1,
                diagnostic_only=True, policy_hash=fingerprint(policy),
                native_binary_hash=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                runner_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                oracle_vega=oracle, checks=checks, original_three_percent_comparisons=original_comparisons,
                observations=observations,
                limitations=['Continuous piecewise-affine single-terminal payoffs and spatially flat zero-carry model only.',
                             'The original 154-check gate is evaluated separately, without modifying its policy.',
                             'This is a terminal boundary implementation, not inversion of arbitrary noisy AD rows.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads((ROOT/'fixtures/backtesting/numerical-policy-v0.1.json').read_text())
    report = run(policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    failures = sum(not c['passed'] for c in report['checks'])
    print(json.dumps(dict(cases=len(report['observations']), checks=len(report['checks']), failures=failures,
                          unchanged_three_percent_failures=sum(not c['passed'] for c in report['original_three_percent_comparisons']),
                          max_bucket_error=max(c['observed'] for c in report['checks'] if c['check']=='terminal_bucket_absolute_error'),
                          max_source_hat_error=max(c['observed'] for c in report['checks'] if c['check']=='source_hat_unconditional_quadrature'))))
    return int(failures != 0)


if __name__ == '__main__':
    raise SystemExit(main())
