"""Independent NumPy rough-LSV particle calibration on shared Gaussian inputs.

Rust supplies only the retained Philox Gaussian table, checked by a separate
Rust test. NumPy computes the Volterra history, particle evolution, conditional
moments, effective sample sizes, fallback donors and leverage. No production
calibration, path, price or risk function is imported.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from rough_dividend_continuous_reference import DIRECTORY, inputs
from rough_dividend_survival_reference import hybrid_weights

FIXTURE = DIRECTORY / 'rough-lsv-calibration-reference.json'


class UnsupportedCalibrationNode(ValueError):
    def __init__(self, time_index):
        super().__init__(f'no supported calibration node at time index {time_index}')
        self.time_index = time_index


def variance_multipliers(times, hurst, eta, correlation, normals):
    z = np.asarray(normals).reshape(len(normals), 3, len(times)-1)
    weights, residual, variance = hybrid_weights(np.asarray(times), hurst)
    increments = np.sqrt(np.diff(times))*(correlation*z[:, 0]+math.sqrt(1-correlation**2)*z[:, 1])
    driver = increments @ weights.T
    driver[:, 1:] += residual*z[:, 2]
    return np.exp(eta*driver/2-eta*eta*variance/4)


def conditional_moments(log_states, multipliers, nodes, bandwidth, minimum_ess, *, exact=False):
    count = len(log_states)
    if exact:
        return np.ones((len(nodes), 3)), np.full(len(nodes), float(count)), np.arange(len(nodes))
    u = (np.asarray(log_states)[:, None]-nodes)/bandwidth
    weights = np.where(np.abs(u) < 1, (1-u*u)**2, 0.)
    totals = weights.sum(axis=0)
    squares = (weights*weights).sum(axis=0)
    ess = np.divide(totals*totals, squares, out=np.zeros_like(totals), where=squares > 0)
    powers = np.asarray(multipliers)[:, None]**np.array([2, 3, 4])
    moments = np.divide(weights.T @ powers, totals[:, None], out=np.ones((len(nodes), 3)), where=totals[:, None] > 0)
    supported = np.flatnonzero(ess >= minimum_ess)
    if not len(supported):
        return None
    donors = np.array([j if ess[j] >= minimum_ess else
                       min(supported, key=lambda k: (abs(nodes[k]-x), k))
                       for j, x in enumerate(nodes)])
    # Borrow moment ratios only; retain each recipient's original ESS.
    return moments[donors], ess, donors


def calibrate(times, nodes, target, normals, *, hurst, eta, correlation, initial_f,
              bandwidth, minimum_effective_samples):
    times, nodes = np.asarray(times), np.asarray(nodes)
    z = np.asarray(normals).reshape(len(normals), 3, len(times)-1)
    target = np.asarray(target).reshape(len(times), len(nodes))
    a = variance_multipliers(times, hurst, eta, correlation, normals)
    # Evolve normalized log residual equity directly, independently of Rust's
    # multiplicative level-state update and sorted/compensated kernel reduction.
    log_f = np.zeros(len(z))
    surface, moments, diagnostics = [], [], []
    for r, t in enumerate(times):
        result = conditional_moments(log_f, a[:, r], nodes, bandwidth,
                                     minimum_effective_samples, exact=(r == 0 or eta == 0))
        if result is None:
            raise UnsupportedCalibrationNode(r)
        ratios, ess, donors = result
        row = target[r]/ratios[:, 0]
        surface.extend(row.tolist())
        for j in range(len(nodes)):
            moments.append(dict(second=float(ratios[j, 0]), third=float(ratios[j, 1]),
                                fourth=float(ratios[j, 2]), effective_samples=float(ess[j]),
                                source_node=int(donors[j]), extrapolated=bool(donors[j] != j)))
        diagnostics.append(dict(time=float(t), minimum_effective_samples=float(ess[ess >= minimum_effective_samples].min()),
                                extrapolated_nodes=int(np.count_nonzero(donors != np.arange(len(nodes)))),
                                particle_mean_f=float(initial_f*np.exp(log_f).mean())))
        if r+1 < len(times):
            variance = np.interp(log_f, nodes, row)*a[:, r]**2
            dt = times[r+1]-t
            log_f += -variance*dt/2+np.sqrt(variance*dt)*z[:, 0, r]
    assert np.isfinite(surface).all() and (np.asarray(surface) > 0).all()
    return dict(squared_leverage=surface, moments=moments, diagnostics=diagnostics)


def case_result(case, normals):
    return calibrate(case['times'], case['log_nodes'], case['target_variances'], normals,
                     **{key: case[key] for key in ['hurst', 'eta', 'correlation', 'initial_f',
                                                  'bandwidth', 'minimum_effective_samples']})


def assert_result(actual, expected):
    np.testing.assert_allclose(actual['squared_leverage'], expected['squared_leverage'], rtol=2e-12, atol=2e-12)
    for kind in ['moments', 'diagnostics']:
        assert len(actual[kind]) == len(expected[kind])
        for a, e in zip(actual[kind], expected[kind]):
            for key in a:
                if key in ['source_node', 'extrapolated', 'extrapolated_nodes']:
                    assert a[key] == e[key]
                else:
                    np.testing.assert_allclose(a[key], e[key], rtol=2e-12, atol=2e-12)


def verify_quote_calibrations(normals):
    """Independently recalibrate all 96 previously retained quote scenarios."""
    fixture = json.loads((DIRECTORY/'rough-continuous-market-iv-reference.json').read_text())
    bases, market = inputs()
    original_times = fixture['target']['time_nodes']
    nodes = fixture['target']['log_moneyness_nodes']
    config = fixture['calibration']
    assert (config['particle_count'], config['seed']) == (len(normals), 42)
    growth = market['annual_carry']/market['annual_discount']
    initial_f = market['spot']-sum(q/growth**t for q, t in zip(market['cash_means'], market['cash_times']))
    for case in fixture['cases']:
        base = bases[case['base_case']]
        maximum_error = 0.
        for panel in case['panels']:
            for scenario in panel['scenarios']:
                original = np.asarray(scenario['target_variances']).reshape(len(original_times), len(nodes))
                target = np.column_stack([np.interp(base['times'], original_times, column) for column in original.T])
                result = calibrate(base['times'], nodes, target, normals, hurst=base['hurst'], eta=base['eta'],
                                   correlation=base['equity_volatility_correlation'], initial_f=initial_f,
                                   bandwidth=config['log_bandwidth'], minimum_effective_samples=config['minimum_effective_samples'])
                np.testing.assert_allclose(result['squared_leverage'], scenario['squared_leverage'], rtol=0, atol=2e-13)
                maximum_error = max(maximum_error, float(np.max(np.abs(np.asarray(result['squared_leverage'])-scenario['squared_leverage']))))
        print(json.dumps(dict(case=case['id'], calibration_scenarios=24, max_leverage_error=maximum_error)), flush=True)


def verify_fixture():
    fixture = json.loads(FIXTURE.read_text())
    normals = fixture['normals']
    assert np.asarray(normals).shape == (64, 3*fixture['steps'])
    assert np.isfinite(normals).all()
    for case in fixture['cases']:
        try:
            result = case_result(case, normals)
        except UnsupportedCalibrationNode as error:
            assert error.time_index == case['unsupported_time_index']
        else:
            assert 'unsupported_time_index' not in case
            assert_result(result, case['expected'])
        print(json.dumps(dict(case=case['id'], status='passed')), flush=True)
    verify_quote_calibrations(normals)


def build_fixture(normal_log, output):
    line, = [line for line in normal_log.read_text().splitlines() if line.startswith('CALIBRATION_NORMALS ')]
    fixture = json.loads(line.removeprefix('CALIBRATION_NORMALS '))
    fixture['scope'] = 'independent_finite_particle_calibration_given_shared_gaussian_inputs'
    fixture['normal_layout'] = 'particle-major; factor-major spot/orthogonal-variance/near-cell blocks'
    fixture['normal_source'] = 'Rust Philox4x32 LsvCalibration domain, seed 42; RNG inputs only, no calibration outputs'
    times = [0., .03, .08, .16, .3, .45, .62, .8, 1.]
    nodes = [-1., -.4, 0., .3, 1.]
    target = [0.035+.01*t+.002*j for t in times for j in range(len(nodes))]
    cases = []
    for name, h, eta, rho in [('h01_negative_correlation', .1, .6, -.4),
                              ('h03_positive_correlation', .3, 1.1, .5),
                              ('brownian_limit', .5, .8, -.7), ('zero_vol_of_vol', .1, 0., -.4)]:
        cases.append(dict(id=name, times=times, log_nodes=nodes, target_variances=target,
                          hurst=h, eta=eta, correlation=rho, initial_f=79., bandwidth=.35, minimum_effective_samples=5.))
    # Binary-exact symmetric node spacing makes the interior donor tie exact.
    # Short following steps retain the deliberately under-supported middle node.
    cases.append(cases[0] | dict(id='interior_donor_tie',
        times=[0., .125]+[.125+i*1e-8 for i in range(1, 8)],
        log_nodes=[-.09375, -.0625, -.03125], target_variances=[.04]*27,
        bandwidth=.004, minimum_effective_samples=1.5))
    cases.append(cases[0] | dict(id='no_supported_node', log_nodes=[2., 3.], target_variances=[.04]*18,
                                unsupported_time_index=1))
    for case in cases:
        if 'unsupported_time_index' not in case:
            case['expected'] = case_result(case, fixture['normals'])
    fixture['cases'] = cases
    output.write_text(json.dumps(fixture, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normal-log', type=Path, help='explicit Rust Gaussian input capture log')
    parser.add_argument('--build', type=Path, help='write independently computed calibration controls')
    args = parser.parse_args()
    if args.build:
        if not args.normal_log:
            parser.error('--build requires --normal-log')
        build_fixture(args.normal_log, args.build)
    else:
        verify_fixture()
