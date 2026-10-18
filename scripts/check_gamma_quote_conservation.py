"""Retest quote-Gamma failures using frozen calibration-direction constraints.

Same price calibration, quote accuracy targets, pilot counters and spatial/noise
gates as the preceding failure report. No reference payoff enters the inverse.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from check_discrete_smile_calibration import ROOT, base_request, reference_price
from check_path_gamma_recovery import request, operator, source_reference
from check_gamma_selection import evaluate, quality
from rust_pricing_backtest.discrete_smile import TwoStepSmileCalibrator
from rust_pricing_backtest.numerics import threshold_check


def constrained_inverse(b, ridge, directions, preserve_total=True):
    """Independent primal KKT solve, not the native Schur-complement update."""
    d = np.asarray(directions)
    if preserve_total: d = np.vstack([np.ones(b.shape[0]), d])
    c = d@b; n, k = b.shape[1], len(d)
    h = b.T@b+(ridge*np.linalg.norm(b))**2*np.eye(n)
    return np.linalg.solve(np.block([[h, c.T], [c, np.zeros((k, k))]]), np.vstack([b.T, d]))[:n]


def run(policy):
    import rust_pricing as rp
    source_paths = [Path(__file__), ROOT/'scripts/check_discrete_smile_calibration.py',
                    ROOT/'scripts/check_path_gamma_recovery.py', ROOT/'scripts/check_gamma_selection.py',
                    ROOT/'backtesting/rust_pricing_backtest/discrete_smile.py']
    source_paths += list((ROOT/'crates/pricing/src/risk').glob('vegakt_*.rs'))
    source_paths += list((ROOT/'crates/pricing/src/risk/vegakt_convolution').glob('*.rs'))
    source_paths += list((ROOT/'crates/pricing/src/engine/risk').glob('vegakt_*.rs'))
    source_paths += list((ROOT/'crates/pricing-python/src').glob('vegakt_*.rs'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    checks, observations, price_records = [], [], []
    def check(name, value, limit, **ctx): checks.append(threshold_check(name, float(value), float(limit), **ctx))
    p = base_request(policy, 'flat'); calibrator = TwoStepSmileCalibrator(p)
    tangent, tangent_audit = calibrator.terminal_iv_tangent(); directions = tangent.T/.4
    config = dict(policy['selection'], local_vega_conservation_directions=[directions.tolist()])
    B = operator(100., calibrator.v0, .04*calibrator.dt, .2*calibrator.dt, calibrator.nodes, 100*np.exp(calibrator.nodes))
    inverses = {lam: constrained_inverse(B, lam, directions) for lam in config['candidate_regularizations']}
    # Freeze all shocked models before any exotic price or Gamma is evaluated.
    pairs, calibration_audits = [], []
    for bump in policy['iv_bumps']:
        row = []
        for direction in np.eye(len(calibrator.quote_nodes)):
            pair = []
            for sign in (1., -1.):
                shifted, audit = calibrator.calibrate(sign*bump*direction)
                pair.append(shifted); calibration_audits.append(audit)
            row.append(pair)
        pairs.append(row)
    for kind in policy['products']:
        product = request(kind, samples=16)['product']; refs, batches = [], []
        for bump, models in zip(policy['iv_bumps'], pairs):
            reference = np.array([(reference_price(up, kind, 100., 96)-reference_price(down, kind, 100., 96))/(2*bump) for up, down in models])
            batch = []
            for seed in policy['batch_seeds']:
                differences = []
                for pair in models:
                    prices = []
                    for model in pair:
                        q = deepcopy(model); q['product'] = product
                        q['engine'].update(master_seed=seed, independent_sampling_units=policy['batch_samples'])
                        q['risk'] = dict(delta=False, vega=False, smile_dynamics=dict(type='sticky_log_moneyness'))
                        prices.append(rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(q)), worker_threads=1, reduction_block_size=256).evaluate().value)
                    differences.append((prices[0]-prices[1])/(2*bump))
                batch.append(differences)
            refs.append(reference); batches.append(np.array(batch))
        refs, batches = np.array(refs), np.array(batches)
        mean = batches.mean(axis=1); se = batches.std(axis=1, ddof=1)/math.sqrt(len(policy['batch_seeds']))
        for i, bump in enumerate(policy['iv_bumps']):
            check('recalibrated_native_price_reference', np.max(abs(mean[i]-refs[i])/(policy['statistical_multiplier']*se[i]+policy['derivative_absolute_tolerance'])), 1., product=kind, bump=bump)
        check('independent_price_bump_refinement', np.max(abs(refs[0]-refs[1])), policy['bump_refinement_tolerance'], product=kind)
        diff = batches[0]-batches[1]; dse = diff.std(axis=0, ddof=1)/math.sqrt(len(diff))
        check('paired_native_bump_refinement', np.max(abs(diff.mean(axis=0))/(policy['statistical_multiplier']*dse+policy['bump_refinement_tolerance'])), 1., product=kind)
        population = source_reference(kind, 100., 100., calibrator.v0, .04*calibrator.dt, .2, calibrator.nodes)
        population_ad = directions@population
        check('calibration_tangent_vs_exotic_reference', np.max(abs(population_ad-refs[1])), policy['bump_refinement_tolerance'], product=kind)
        price_records.append(dict(product=kind, reference_derivatives=refs.tolist(), batch_derivatives=batches.tolist(), derivative_means=mean.tolist(), derivative_standard_errors=se.tolist()))
        for seed in policy['gamma_seeds']:
            q = deepcopy(p); q['product'] = product; q['engine']['master_seed'] = seed
            result = evaluate(q, 100*np.exp(calibrator.nodes), config)
            ctx = dict(product=kind, seed=seed)
            check('unchanged_heldout_recovery_gate', int(result['status'] != 'positive_time_policy_passed'), 0, **ctx)
            choice = None
            for index, candidate in enumerate(result['pilot_candidates']+([result['validation']] if result['validation'] else [])):
                raw = candidate['diagnostic']; row = raw['rows'][0]; lam = raw['relative_regularization']; R = inverses[lam]
                cctx = dict(**ctx, regularization=lam, sample='pilot' if index < len(result['pilot_candidates']) else 'validation')
                b = np.array([m['mean'] for m in row['local_vega']]); masses = np.array([m['mean'] for m in row['gamma_masses']])
                check('independent_primal_kkt_recovery', np.max(abs(masses-R@b)), policy['solver_absolute_tolerance'], **cctx)
                check('all_spatial_residuals_retained', np.max(abs(B@masses-b-[m['mean'] for m in row['reconstruction_difference']])), policy['solver_absolute_tolerance'], **cctx)
                check('updated_inverse_noise_gain', abs(np.linalg.norm(R)-row['inverse_noise_gain_bound'])/np.linalg.norm(R), policy['operator_relative_tolerance'], **cctx)
                check('updated_resolution_trace', abs(np.trace(R@B)-row['resolution_trace']), policy['operator_relative_tolerance'], **cctx)
                check('pathwise_integral_conservation', abs(row['total_difference']['mean'])+6*row['total_difference']['standard_error'], policy['identity_absolute_tolerance'], **cctx)
                own = quality(row, config); check('unchanged_quality_verdict', int(own['passed'] != candidate['passed']), 0, **cctx)
                if index < len(result['pilot_candidates']) and candidate['passed'] and choice is None: choice = lam
                for i, (a, b, error) in enumerate(zip(row['local_vega_directions'], row['reconstructed_directions'], row['direction_differences'])):
                    check('pathwise_quote_direction_conservation', abs(error['mean'])+6*error['standard_error'], policy['identity_absolute_tolerance'], quote=i, **cctx)
                    check('same_sample_quote_variance_conserved', abs(a['sample_variance']-b['sample_variance'])/max(1., a['sample_variance']), policy['identity_absolute_tolerance'], quote=i, **cctx)
            check('strongest_pilot_feasible_selection', int(choice != result['selected_regularization']), 0, **ctx)
            record = dict(**ctx, audit=result)
            if result['status'] == 'positive_time_policy_passed':
                R = inverses[result['selected_regularization']]; population_gamma = directions@B@R@population
                row = result['validation']['diagnostic']['rows'][0]
                gamma = np.array([m['mean'] for m in row['reconstructed_directions']]); gse = np.array([m['standard_error'] for m in row['reconstructed_directions']])
                for i, (value, ref) in enumerate(zip(population_gamma, population_ad)):
                    check('population_gamma_quote_bias', abs(value-ref), policy['gamma_quote_absolute_target']+policy['gamma_quote_relative_target']*abs(ref), quote=i, **ctx)
                    check('recovered_gamma_vs_recalibrated_native_price', abs(gamma[i]-mean[1,i]), policy['statistical_multiplier']*math.hypot(gse[i], se[1,i])+policy['derivative_absolute_tolerance'], quote=i, **ctx)
                    check('recovered_gamma_vs_independent_price_reference', abs(gamma[i]-refs[1,i]), policy['statistical_multiplier']*gse[i]+policy['derivative_absolute_tolerance'], quote=i, **ctx)
                record.update(population_quote_ad=population_ad.tolist(), population_quote_gamma=population_gamma.tolist(), population_quote_bias=(population_gamma-population_ad).tolist(), quote_means=gamma.tolist(), quote_standard_errors=gse.tolist())
            observations.append(record)
            print(json.dumps(dict(**ctx, status=result['status'], selected=result['selected_regularization'])), flush=True)
    if hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}:
        raise RuntimeError('validation sources changed during execution')
    module = getattr(rp, 'rust_pricing', rp)
    return dict(document_kind='gamma_quote_direction_conservation_validation', schema_version=1,
                status='passed' if all(c['passed'] for c in checks) else 'failed',
                full_market_iv_bucket_acceptance='not_established', policy=policy,
                recovery_policy=config, tangent_audit=tangent_audit, checks=checks, observations=observations,
                recalibrated_price_records=price_records, calibration_audits=calibration_audits,
                native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), source_sha256=hashes,
                limitations=['Flat baseline and two-step calibration directions only; nonflat Gamma remains unsupported.',
                             'Quote risks use the declared calibration tangent, not the uncalibrated atom-chain IV bucket interpretation.',
                             'Constraints preserve given local-Vega functionals; they do not identify the full Gamma uniquely.',
                             'Quote standard errors retain same-path covariance within each direction; no full quote covariance is claimed.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--policy', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args(); r=run(json.loads(args.policy.read_text())); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(r, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=r['status'], checks=len(r['checks']), failed=sum(not c['passed'] for c in r['checks']))))
    return int(r['status'] != 'passed')


if __name__ == '__main__': raise SystemExit(main())
