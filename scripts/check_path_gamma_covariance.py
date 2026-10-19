"""Covariance acceptance with exact replay of the preceding Gamma-chain panel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from multi_step_cases import ROOT, models, with_product
from rust_pricing_backtest.multi_step_full_smile import MultiStepSmileCalibrator
from rust_pricing_backtest.numerics import threshold_check


def run(previous_path):
    import rust_pricing as rp
    previous = json.loads(previous_path.read_text())
    if previous['status'] != 'passed' or len(previous['observations']) != 16:
        raise ValueError('complete passing sixteen-condition Gamma-chain report required')
    # Reuse price differences only if all their implementation/fixture inputs
    # remain unchanged. The listed changes are covariance reporting/reduction.
    permitted_changes = {
        'crates/pricing/src/engine/risk/vegakt_path_chain.rs',
        'crates/pricing/src/risk/vegakt_path_chain.rs',
        'crates/pricing-python/src/vegakt_path_chain.rs',
        'backtesting/rust_pricing_backtest/multi_step_full_smile.py',
        'rust_pricing.pyi',
    }
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, digest in previous['source_sha256'].items():
        if name not in permitted_changes and sha(ROOT/name) != digest:
            raise ValueError('reference price implementation changed: '+name)
    paths = set(previous['source_sha256']) | permitted_changes | {
        'crates/pricing/src/engine/risk/vegakt_path_chain/statistics.rs',
        'scripts/check_path_gamma_covariance.py',
    }
    frozen = {name: sha(ROOT/name) for name in sorted(paths)}
    checks, observations = [], []
    def check(name, value, limit, **ctx):
        checks.append(threshold_check(name, float(value), float(limit), **ctx))
    def native(p, c, directions, policy):
        return rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)),
                worker_threads=1, reduction_block_size=256
            ).evaluate_vega_kt_multi_step_gamma_chain_audit(
                c.calibrated_rows, policy['gamma_bins'],
                variance_risk_directions=directions.tolist(),
                subdivisions=policy['subdivisions'],
                relative_tolerance=policy['relative_tolerance'],
                gamma_relative_standard_error_limit=policy['gamma_relative_standard_error_limit'])
    contexts = {}
    for case in models():
        c = MultiStepSmileCalibrator(case['request'])
        t, _ = c.all_maturity_iv_tangent()
        contexts[case['intervals']] = (case['request'], c, t.T.reshape(c.iv.size, len(c.indices), len(c.nodes)))
    for old in previous['observations']:
        ctx = {name: old[name] for name in ['intervals', 'profile', 'product', 'seed']}
        base, c, directions = contexts[ctx['intervals']]
        policy = previous['policy']
        p = with_product(base, ctx['product'])
        p['engine'].update(master_seed=ctx['seed'], independent_sampling_units=policy['gamma_samples'])
        a = native(p, c, directions, policy)
        # Metadata and the new covariance fields may differ; all pre-existing
        # numerical values and gates must compare exactly, not just within SE.
        keys = set(old['audit']) - {
            'uncertainty_scope', 'joint_sample_covariance', 'covariance_dimension',
            'covariance_layout', 'covariance_unit', 'covariance_scope',
        }
        identical = all(a[k] == old['audit'][k] for k in keys)
        check('previous_numerical_fields_exact_replay', not identical, 0, **ctx)
        check('gamma_chain_passed', a['status'] != 'path_gamma_chain_passed', 0, **ctx)
        if not identical:
            raise ValueError('previous numerical output changed; do not reuse price evidence')
        d = len(directions)
        cov = np.asarray(a['joint_sample_covariance']).reshape(3*d, 3*d)
        groups = ['native_directions', 'recovered_directions', 'direction_differences']
        moments = sum([a[k] for k in groups], [])
        scale = max(1., float(np.max(abs(cov))))
        limit = 1e-10*scale
        check('covariance_symmetric', np.max(abs(cov-cov.T)), 0, **ctx)
        check('covariance_positive_semidefinite', max(0., -np.linalg.eigvalsh(cov).min()), limit, **ctx)
        check('covariance_diagonal_matches_variance', np.max(abs(cov.diagonal()-[v['sample_variance'] for v in moments])), limit, **ctx)
        check('covariance_of_mean_matches_squared_SE',
              np.max(abs(cov.diagonal()/policy['gamma_samples']-[v['standard_error']**2 for v in moments])),
              limit/policy['gamma_samples'], **ctx)
        check('native_recovered_difference_cross_covariance',
              np.max(abs(cov[2*d:]-cov[d:2*d]+cov[:d])), 4*limit, **ctx)
        weights = np.array([
            np.ones(d)/np.sqrt(d),
            np.array([(-1.)**i for i in range(d)])/np.sqrt(d),
            np.eye(d)[0]-np.eye(d)[-1],
            np.zeros(d),
        ])
        composed = np.einsum('kd,drm->krm', weights, directions)
        direct = native(p, c, composed, policy)
        direct_cov = np.array(direct['joint_sample_covariance']).reshape(12, 12)
        transform = np.kron(np.eye(3), weights)
        projected_cov = transform@cov@transform.T
        projected_scale = max(1., float(np.max(abs(projected_cov))))
        check('direct_composed_full_covariance', np.max(abs(direct_cov-projected_cov)), 1e-9*projected_scale, **ctx)
        for group in groups:
            means = weights@np.array([v['mean'] for v in a[group]])
            check('direct_composed_'+group+'_means',
                  np.max(abs(means-[v['mean'] for v in direct[group]])),
                  1e-9*max(1., float(np.max(abs(means)))), **ctx)
        price = next(row for row in previous['price_records']
                     if all(row[k] == ctx[k] for k in ['intervals','profile','product']))
        gamma = weights@np.array([v['mean'] for v in a['recovered_directions']])
        variance = np.diag(projected_cov)[4:8]
        # A non-PSD result beyond roundoff fails above. Clamping here is solely
        # for sqrt of floating-point cancellation at an exact zero readout.
        se = np.sqrt(np.maximum(0., variance)/policy['gamma_samples'])
        for label in ['independent', 'native']:
            batches = np.array(price[label+'_price_difference_batches'])[1]@weights.T
            mean = batches.mean(axis=0)
            stderr = batches.std(axis=0, ddof=1)/np.sqrt(len(batches))
            for i in range(len(weights)):
                check('composed_Gamma_vs_'+label+'_price_difference',
                      abs(gamma[i]-mean[i]),
                      6*np.hypot(se[i], stderr[i])+policy['absolute_tolerance']*sum(abs(weights[i])),
                      combination=i, **ctx)
        observations.append(dict(**ctx, audit=a, combinations=weights.tolist(),
                                 direct_combinations=direct,
                                 previous_numerical_fields_exact=True))
        print(json.dumps(dict(**ctx, checks=len(checks), failed=sum(not x['passed'] for x in checks))), flush=True)
    if frozen != {name: sha(ROOT/name) for name in frozen}:
        raise RuntimeError('source changed during covariance acceptance')
    return dict(document_kind='path_gamma_joint_covariance_validation', schema_version=1,
        status='passed' if all(x['passed'] for x in checks) else 'failed',
        checks=checks, observations=observations, source_sha256=frozen,
        native_binary_sha256=sha(Path(getattr(rp, 'rust_pricing', rp).__file__)),
        previous_report_sha256=sha(previous_path), previous_checks=len(previous['checks']),
        reference_reuse='all previous numerical fields exactly replayed and unchanged price implementation hashes',
        covariance_scope='same_sampling_unit_MC_not_total_error_or_cross_instrument_covariance',
        full_market_iv_bucket_acceptance='not_established')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.previous_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']),
                         failed=sum(not x['passed'] for x in result['checks']))))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
