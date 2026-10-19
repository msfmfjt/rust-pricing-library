"""Validate finite-step nonlinear smile calibration and expose Gamma quote bias.

Calibration uses vanilla prices only. Native price differences use independent
batch seeds. Gamma selection retains the existing pilot/held-out policy; quote
projection bias is reported separately and is NOT absorbed into a tolerance.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from check_path_gamma_recovery import ROOT, request, operator, source_reference
from check_gamma_selection import evaluate as gamma_audit
from check_conservative_gamma_market import kkt_inverse
from rust_pricing_backtest.discrete_smile import TwoStepSmileCalibrator
from rust_pricing_backtest.numerics import black_scholes_call, threshold_check


def reference_price(p, kind, strike=100., order=64):
    """Separate payoff quadrature, including product kinks, without calibrator calls."""
    g = p['model']['local_variance_grid']; spot = p['market']['spot']
    nodes = np.array(g['log_forward_moneyness_nodes']); t = np.array(g['time_nodes'])
    values = np.array(g['values']).reshape(3, len(nodes)); v0 = values[0, 0]*t[1]
    sd = math.sqrt(v0); lo, hi = -v0/2-12*sd, -v0/2+12*sd
    cuts = [lo, hi]+[x for x in nodes if lo < x < hi]
    threshold = dict(call=None, asian=2*strike, fixed_asian=4*(strike-.25*98.), lookback=strike)[kind]
    if threshold and lo < math.log(threshold/spot) < hi: cuts.append(math.log(threshold/spot))
    cuts = sorted(set(cuts)); z, w = np.polynomial.legendre.leggauss(order); out = 0.
    cdf = np.vectorize(lambda x: .5*math.erfc(-x/math.sqrt(2)))
    for a, b in zip(cuts, cuts[1:]):
        x = (a+b)/2+(b-a)/2*z; weights = w*(b-a)/2*np.exp(-.5*((x+v0/2)/sd)**2)/(sd*math.sqrt(2*math.pi))
        s = spot*np.exp(x); v = np.interp(x, nodes, values[1])*(t[2]-t[1]); v_sd = np.sqrt(v)
        k = dict(call=np.full_like(s, strike), asian=2*strike-s,
                 fixed_asian=2*strike-49.-.5*s, lookback=np.maximum(strike, s))[kind]
        valid = k > 0; d1 = np.zeros_like(s); d1[valid] = (np.log(s[valid]/k[valid])+v[valid]/2)/v_sd[valid]
        call = np.where(valid, s*cdf(d1)-k*cdf(d1-v_sd), s-k)
        payoff = call*(.5 if kind in ('asian', 'fixed_asian') else 1.)
        if kind == 'lookback': payoff += np.maximum(s-strike, 0.)
        out += float(weights@payoff)
    return out


def base_request(policy, shape):
    p = request('asian', samples=policy['gamma_samples'], seed=policy['gamma_seeds'][0], nodes=policy['local_nodes'])
    b = p['model']['reporting_iv_basis']; x = np.array(policy['quote_log_nodes'])
    b.update(log_forward_moneyness_nodes=x.tolist(), implied_volatilities=[.2]*(2*len(x)), shape=[2, len(x)])
    p['risk']['vega_kt']['log_forward_moneyness_nodes'] = x.tolist()
    if shape == 'nonflat': p, _ = TwoStepSmileCalibrator(p).calibrate(policy['base_skew']*x+policy['base_curvature']*x*x)
    return p


def run(policy):
    import rust_pricing as rp
    source_paths = list((ROOT/'backtesting/rust_pricing_backtest').glob('*.py'))+[
        ROOT/'scripts'/name for name in ('check_discrete_smile_calibration.py', 'check_path_gamma_recovery.py',
            'check_gamma_selection.py', 'check_conservative_gamma_market.py', 'check_atom_gamma_chain.py')]
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    checks, observations, calibrations, gamma_rows = [], [], [], []
    x = np.array(policy['quote_log_nodes']); directions = dict(skew=x, curvature=x*x)
    directions.update({f'quote_{i}': d for i, d in enumerate(np.eye(len(x)))})
    payoff_specs = [('call', 100*math.exp(k)) for k in x]+[(k, 100.) for k in policy['products']]
    products = {k: request(k, samples=16)['product'] for k in policy['products']}
    def check(name, value, limit, **ctx): checks.append(threshold_check(name, float(value), float(limit), **ctx))
    def native(p, kind, strike, seed):
        p = deepcopy(p); p['product'] = deepcopy(products[kind]); p['product']['strike'] = strike
        p['risk'] = {'delta': False, 'vega': False, 'smile_dynamics': {'type': 'sticky_log_moneyness'}}
        p['engine']['master_seed'] = seed; p['engine']['independent_sampling_units'] = policy['batch_samples']
        return rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p)), worker_threads=1, reduction_block_size=256).evaluate().value
    for shape in policy['base_shapes']:
        p = base_request(policy, shape); calibrator = TwoStepSmileCalibrator(p)
        tangent, tangent_audit = calibrator.terminal_iv_tangent()
        check('quote_tangent_right_inverse', tangent_audit['iv_response_residual'], policy['algebra_tolerance'], shape=shape)
        # Material finite quote changes exercise the nonlinear solver, separately
        # from small derivative bumps. Refined independent integration verifies it.
        for name, direction in directions.items():
            for sign in (1., -1.):
                out, a = calibrator.calibrate(sign*policy['stress_iv_bump']*direction)
                reference = np.array([reference_price(out, 'call', k, 96) for k in calibrator.strikes])
                check('nonlinear_quote_fit_independent_quadrature', np.max(abs(reference-a['target_prices'])), policy['quote_price_tolerance'], shape=shape, direction=name, sign=sign)
                calibrations.append(dict(shape=shape, direction=name, purpose='finite_quote_stress', audit=a))
        # Recovery is evaluated once per product/seed before seeing any exotic
        # price differences. Nonflat Gamma is explicitly outside its native scope.
        recoveries = {}
        if shape == 'flat':
            B = operator(100., calibrator.v0, .04*calibrator.dt, .2*calibrator.dt,
                         calibrator.nodes, 100*np.exp(calibrator.nodes))
            for kind in policy['products']:
                population = source_reference(kind, 100., 100., calibrator.v0, .04*calibrator.dt, .2, calibrator.nodes)
                for seed in policy['gamma_seeds']:
                    q = deepcopy(p); q['product'] = products[kind]; q['engine']['master_seed'] = seed
                    audit = gamma_audit(q, 100*np.exp(calibrator.nodes), policy['selection'])
                    check('heldout_gamma_recovery_gate', int(audit['status'] != 'positive_time_policy_passed'), 0, product=kind, seed=seed)
                    if audit['status'] != 'positive_time_policy_passed':
                        gamma_rows.append(dict(product=kind, seed=seed, audit=audit)); continue
                    R = kkt_inverse(B, audit['selected_regularization'])
                    row = audit['validation']['diagnostic']['rows'][0]
                    weights = tangent/.4
                    local = np.array([m['mean'] for m in row['local_vega']])
                    masses = np.array([m['mean'] for m in row['gamma_masses']])
                    quote_gamma = weights.T@B@masses; quote_ad = weights.T@local
                    population_ad = weights.T@population; population_gamma = weights.T@B@R@population
                    # Triangle bounds on SE do not assume independent node errors.
                    gamma_se_bound = abs(weights.T@B)@np.array([m['standard_error'] for m in row['gamma_masses']])
                    check('gamma_projection_vs_regularized_population', np.max(abs(quote_gamma-population_gamma)/(policy['statistical_multiplier']*gamma_se_bound+policy['derivative_absolute_tolerance'])), 1., product=kind, seed=seed)
                    residual = np.array([m['mean'] for m in row['reconstruction_difference']])
                    check('quote_projection_retains_spatial_residual', np.max(abs(quote_gamma-quote_ad-weights.T@residual)), policy['solver_absolute_tolerance'], product=kind, seed=seed)
                    quote_limits = policy['gamma_quote_absolute_target']+policy['gamma_quote_relative_target']*abs(population_ad)
                    gaps = [threshold_check('population_gamma_quote_bias', float(abs(a-b)), float(limit), quote=i)
                            for i, (a, b, limit) in enumerate(zip(population_gamma, population_ad, quote_limits))]
                    gamma_rows.append(dict(product=kind, seed=seed, audit=audit,
                        calibrated_quote_ad=quote_ad.tolist(), calibrated_quote_gamma=quote_gamma.tolist(),
                        gamma_quote_standard_error_upper_bounds=gamma_se_bound.tolist(),
                        population_quote_ad=population_ad.tolist(), population_quote_gamma=population_gamma.tolist(),
                        population_quote_bias=(population_gamma-population_ad).tolist(), quote_accuracy_checks=gaps))
                    recoveries[(kind, seed)] = population_ad
        for name, direction in directions.items():
            batches, refs = [], []
            for bump in policy['iv_bumps']:
                pair = []
                for sign in (1., -1.):
                    out, a = calibrator.calibrate(sign*bump*direction); pair.append(out)
                    calibrations.append(dict(shape=shape, direction=name, purpose='derivative', audit=a))
                ref = np.array([(reference_price(pair[0], kind, strike)-reference_price(pair[1], kind, strike))/(2*bump) for kind, strike in payoff_specs])
                fine = np.array([(reference_price(pair[0], kind, strike, 96)-reference_price(pair[1], kind, strike, 96))/(2*bump) for kind, strike in payoff_specs])
                check('independent_payoff_quadrature_refinement', np.max(abs(ref-fine)), policy['quadrature_derivative_tolerance'], shape=shape, direction=name, bump=bump)
                expected = []
                for i, strike in enumerate(calibrator.strikes):
                    iv = calibrator.iv[1, i]; h = bump*direction[i]
                    expected.append((black_scholes_call(100., strike, calibrator.times[2], iv+h)['price']-black_scholes_call(100., strike, calibrator.times[2], iv-h)['price'])/(2*bump))
                check('finite_smile_quotes_match_market_price_response', np.max(abs(fine[:len(x)]-expected)), policy['quadrature_derivative_tolerance'], shape=shape, direction=name, bump=bump)
                values = []
                for seed in policy['batch_seeds']:
                    values.append([(native(pair[0], kind, strike, seed)-native(pair[1], kind, strike, seed))/(2*bump) for kind, strike in payoff_specs])
                batch = np.array(values); mean = batch.mean(axis=0); se = batch.std(axis=0, ddof=1)/math.sqrt(len(batch))
                check('native_recalibrated_prices_vs_payoff_reference', np.max(abs(mean-fine)/(policy['statistical_multiplier']*se+policy['derivative_absolute_tolerance'])), 1., shape=shape, direction=name, bump=bump)
                batches.append(batch); refs.append(fine)
            differences = batches[0]-batches[1]; error = differences.mean(axis=0)
            se = differences.std(axis=0, ddof=1)/math.sqrt(len(differences))
            check('paired_native_bump_refinement', np.max(abs(error)/(policy['statistical_multiplier']*se+policy['bump_refinement_tolerance'])), 1., shape=shape, direction=name)
            check('independent_reference_bump_refinement', np.max(abs(refs[0]-refs[1])), policy['bump_refinement_tolerance'], shape=shape, direction=name)
            if shape == 'flat':
                for (kind, seed), pop in recoveries.items():
                    index = len(x)+policy['products'].index(kind)
                    check('calibration_pullback_vs_exotic_price_difference', abs(pop@direction-refs[1][index]), policy['bump_refinement_tolerance'], product=kind, seed=seed, direction=name)
            observations.append(dict(shape=shape, direction=name, payoff_specs=payoff_specs,
                reference_derivatives=np.array(refs).tolist(), batch_derivatives=np.array(batches).tolist(),
                tangent_audit=tangent_audit))
            print(json.dumps(dict(shape=shape, direction=name, checks_passed=all(c['passed'] for c in checks))), flush=True)
    module = getattr(rp, 'rust_pricing', rp)
    if source_hashes != {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}:
        raise RuntimeError('validation sources changed during this run; rerun before retaining evidence')
    gaps = [c for row in gamma_rows for c in row.get('quote_accuracy_checks', [])]
    return dict(document_kind='two_step_discrete_smile_calibration_validation', schema_version=1,
        status='passed' if all(c['passed'] for c in checks+gaps) else 'failed',
        calibration_and_consistency_status='passed' if all(c['passed'] for c in checks) else 'failed',
        full_market_iv_bucket_acceptance='not_established',
        gamma_quote_accuracy_status='failed' if any(not c['passed'] for c in gaps) else 'passed_in_tested_scope',
        gamma_quote_accuracy_failed=sum(not c['passed'] for c in gaps), gamma_quote_accuracy_checks=len(gaps),
        policy=policy, checks=checks, observations=observations, calibrations=calibrations, gamma_projections=gamma_rows,
        source_sha256=source_hashes,
        native_binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Exactly two simulation intervals, fixed first Gaussian marginal, zero carry and unit discount.',
                     'Only listed terminal quote strikes are fitted; local variance extension depends on the declared norm.',
                     'Nonflat base calibration is tested; native nonflat Gamma recovery is not implemented.',
                     'Gamma quote bias checks are separate and remain failures; diagnostic success does not waive them.',
                     'Gamma quote SE upper bounds use the triangle inequality, not an assumed joint covariance.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-only', action='store_true',
                        help='Exit status gates calibration/consistency only; Gamma quote failures remain in the report.')
    args = parser.parse_args(); report = run(json.loads(args.policy.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], calibration_and_consistency_status=report['calibration_and_consistency_status'],
                          checks=len(report['checks']), failed=sum(not c['passed'] for c in report['checks']),
                          gamma_quote_accuracy_failed=report['gamma_quote_accuracy_failed'])))
    gate = report['calibration_and_consistency_status'] if args.calibration_only else report['status']
    return int(gate != 'passed')


if __name__ == '__main__': raise SystemExit(main())
