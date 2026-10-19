#!/usr/bin/env python3
"""Read-only investigation of retained VegaKT reports; not an acceptance runner.

The deterministic terminal calculation uses Gaussian conditional moments and
SciPy adaptive quadrature, not native AD/recovery/projection. Production code,
existing reports and acceptance policies are never modified by this script.
"""
import argparse
import bisect
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import scipy
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]


def interval(a, b):
    root = math.sqrt(2)
    if a >= 0:
        return .5*(math.erfc(a/root)-math.erfc(b/root))
    if b <= 0:
        return .5*(math.erfc(-b/root)-math.erfc(-a/root))
    return 1-.5*math.erfc(-a/root)-.5*math.erfc(b/root)


def normal_hats(nodes, mean, variance):
    """Exact truncated zeroth/first normal moments, evaluated in binary64."""
    z = [(x-mean)/math.sqrt(variance) for x in nodes]
    result = [0.]*len(nodes)
    result[0] = .5*math.erfc(-z[0]/math.sqrt(2))
    result[-1] = .5*math.erfc(z[-1]/math.sqrt(2))
    pdf = lambda x: math.exp(-x*x/2)/math.sqrt(2*math.pi)
    for i, (a, b) in enumerate(zip(z, z[1:])):
        probability = interval(a, b)
        first = pdf(a)-pdf(b)
        result[i] += (b*probability-first)/(b-a)
        result[i+1] += (first-a*probability)/(b-a)
    if min(result) < 0 or abs(math.fsum(result)-1) > 2e-12:
        raise ValueError('invalid independently integrated hat measure')
    return result


def terminal_population_atm(observation, policy, exact_vega):
    """Mean weak terminal estimate for a unit ATM call, without MC noise.

    Gamma_T is a unit atom at K. Given X_T=log(K/S0)=0, X_source is normal
    with variance v_source*(v_T-v_source)/v_T. The true source-hat adjoints
    are proportional to that conditional measure. Diagonal weak inversion
    spreads the terminal atom over its conditional Gaussian width.
    """
    c = observation['chain']
    spot = policy['spot']
    sigma = policy['volatility']
    t, maturity = c['maturity_nodes'][-2:]
    v, u = sigma*sigma*maturity, sigma*sigma*t
    nodes = c['recovery_log_moneyness_nodes']
    source_hats = normal_hats(nodes, 0., u*(v-u)/v)
    target_hats = normal_hats(nodes, 1.5*v, v)
    terminal_density = math.exp(-v/8)/(spot*math.sqrt(2*math.pi*v))
    mass = spot*spot*terminal_density
    gamma = [mass*a/(spot*spot*math.exp(v)*m) if m else 0.
             for a, m in zip(source_hats, target_hats)]
    strikes = [spot*math.exp(x) for x in nodes]
    lower, upper = c['rows'][-1]['strike_bounds']
    active = [i for i, k in enumerate(strikes) if lower*(1-1e-13) <= k <= upper*(1+1e-13)]
    strikes, gamma = [strikes[i] for i in active], [gamma[i] for i in active]
    reporting = c['log_moneyness_nodes']
    center = min(range(len(reporting)), key=lambda j: abs(reporting[j]))
    left, right = reporting[center-1], reporting[center+1]

    def integrand(k):
        i = min(max(bisect.bisect_right(strikes, k)-1, 0), len(strikes)-2)
        weight = (k-strikes[i])/(strikes[i+1]-strikes[i])
        g = gamma[i]*(1-weight)+gamma[i+1]*weight
        x = math.log(k/spot)
        hat = max(0., 1-x/right if x >= 0 else 1-x/left)
        vega = spot*math.sqrt(maturity/(2*math.pi))*math.exp(-(-x/math.sqrt(v)+.5*math.sqrt(v))**2/2)
        return g*vega*hat

    knots = sorted(set(strikes+[spot*math.exp(x) for x in reporting
                               if lower < spot*math.exp(x) < upper]))
    estimates = [math.fsum(quad(integrand, a, b, epsabs=eps, epsrel=eps)[0]
                          for a, b in zip(knots, knots[1:])) for eps in [1e-10, 1e-12]]
    if abs(estimates[0]-estimates[1]) > 1e-8:
        raise ValueError('population quadrature has not converged')
    q = 2*c['coarse_subdivisions']
    native_rule = math.fsum((b-a)/q*integrand(a+(i+.5)*(b-a)/q)
                            for a, b in zip(knots, knots[1:]) for i in range(q))
    sample = c['position_iv_buckets'][-len(reporting)+center]
    return {'steps': observation['steps'], 'population_weak_atm': native_rule,
            'continuous_projection_weak_atm': estimates[1],
            'native_quadrature_bias_estimate': native_rule-estimates[1],
            'exact_atm_vega': exact_vega, 'relative_bias': native_rule/exact_vega-1,
            'sample_atm_mean': sample['mean'], 'sample_atm_standard_error': sample['standard_error'],
            'quadrature_refinement_difference': estimates[1]-estimates[0]}


def analyze(native, chain):
    p = chain['policy']
    if native['policy_hash'] != chain['policy_hash'] or p['spot'] != p['strike'] or p['rate'] != 0:
        raise ValueError('investigation requires the matching zero-carry ATM panel')
    maturity = p['expiry_days']/365
    exact = p['spot']*math.sqrt(maturity/(2*math.pi))*math.exp(-p['volatility']**2*maturity/8)
    if not math.isclose(exact, chain['oracle']['vega'], rel_tol=1e-14):
        raise ValueError('independent analytic Vega disagrees with report')
    flags, counterexamples = [], []
    for o in chain['observations']:
        c = o['chain']
        if c['policy_label'] != 'flat_variance_weak_gamma_chain_diagnostic_v1':
            raise ValueError('unexpected native estimator')
        nx, nt = len(c['log_moneyness_nodes']), len(c['maturity_nodes'])
        center = min(range(nx), key=lambda j: abs(c['log_moneyness_nodes'][j]))
        if abs(c['log_moneyness_nodes'][center]) > 1e-12:
            raise ValueError('exact ATM reporting node required')
        target = (nt-1)*nx+center
        flagged = []
        for i, m in enumerate(c['position_iv_buckets']):
            expected = exact if i == target else 0.
            # Exploratory screening, not a newly approved acceptance rule.
            envelope = p['statistical_multiplier']*m['standard_error']+p['parallel_vega_relative_tolerance']*exact
            if abs(m['mean']-expected) > envelope:
                flagged.append({'maturity_index': i//nx, 'strike_index': i % nx,
                                'mean': m['mean'], 'standard_error': m['standard_error'],
                                'exact': expected, 'screening_envelope': envelope})
        context = {k: o[k] for k in ['seed', 'samples', 'steps', 'nodes']}
        flags.append(context | {'flagged_buckets': flagged})
        error = abs(exact/o['local_vega']-1)
        if error > p['parallel_vega_relative_tolerance']:
            counterexamples.append(context | {'exact_vega': exact, 'native_local_vega': o['local_vega'],
                                             'relative_difference': error, 'native_local_standard_error': o['local_vega_se'],
                                             'difference_in_native_standard_errors': abs(exact-o['local_vega'])/o['local_vega_se']})
    population = [terminal_population_atm(o, p, exact) for o in chain['observations']
                  if (o['seed'], o['samples'], o['nodes']) == (7, 4096, 17) and o['steps'] in [2, 8, 16]]
    return {'document_kind': 'vega_kt_remaining_failure_investigation', 'date': '2026-10-04',
            'diagnostic_only': True, 'policy_unchanged': True, 'policy_hash': chain['policy_hash'],
            'native_failures': dict(Counter(c['check'] for c in native['checks'] if not c['passed'])),
            'maximum_native_accounting_difference': max(abs(o['local_vega']-o['bucket_sum']-o['signed_residual']) for o in native['observations']),
            'flagged_cases': sum(bool(o['flagged_buckets']) for o in flags),
            'flagged_bucket_count': sum(len(o['flagged_buckets']) for o in flags),
            'bucket_screening': flags, 'population_terminal_diagnostics': population,
            'exact_solution_gate_counterexamples': counterexamples,
            'limitations': ['Bucket flags are exploratory and do not amend either existing numerical panel.',
                            'Population calculation covers only the terminal ATM bucket of the flat unit European call.',
                            'Population diagnostics retain the finite recovery grid and separately record native midpoint and converged adaptive projection.',
                            'Source-hat replacement is not exact Gaussian-conditioning inversion.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, default=ROOT/'benchmark-results/vega-kt-numerics/weak-chain/report.json')
    parser.add_argument('--chain', type=Path, default=ROOT/'benchmark-results/vega-kt-pricing-chain/final-comparison/report.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    native, chain = [json.loads(p.read_text()) for p in [args.native, args.chain]]
    report = analyze(native, chain)
    report['input_sha256'] = {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.native, args.chain]}
    report['native_binary_hash'] = chain['native_binary_hash']
    report['analysis_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['scipy_version'] = scipy.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'native_failures': sum(report['native_failures'].values()),
                      'flagged_cases': report['flagged_cases'], 'flagged_buckets': report['flagged_bucket_count'],
                      'statistical_counterexamples': len(report['exact_solution_gate_counterexamples'])}))


if __name__ == '__main__':
    main()
