"""Independent quote/Dupire and continuous-Barrier valuation reference.

Default verification needs only NumPy and committed calibration inputs, never
rust_pricing. --build explicitly captures leverage via separate price-plan
compilations; calibration itself is not an independent implementation. Neither
mode calls a production risk estimator to generate expected values. The
separate rough_lsv_calibration_reference independently reconstructs these
retained calibration inputs using shared Gaussian observations.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from rough_dividend_continuous_reference import (
    DIRECTORY, inputs, local_volatility_bump_means as recalibrated_bump_means,
)

FIXTURE = DIRECTORY / 'rough-continuous-market-iv-reference.json'


def dupire_target(quote_times, quote_x, quotes, target_times, target_x):
    """Natural cubic total variance, analytic derivatives and linear time.

    Directly solve the spline system; no production coefficient maps, surface
    queries or finite-difference derivatives. Time-zero copies the first
    positive target time. Constant-IV time tails match the quote contract.
    """
    times, xs = np.asarray(quote_times), np.asarray(quote_x)
    q = np.asarray(quotes).reshape(len(times), len(xs))
    h = np.diff(xs)
    matrix = np.eye(len(xs))
    for j in range(1, len(xs)-1):
        matrix[j, j-1:j+2] = [h[j-1], 2*(h[j-1]+h[j]), h[j]]
    w = times[:, None]*q*q
    rhs = np.zeros_like(w)
    rhs[:, 1:-1] = 6*np.diff(np.diff(w, axis=1)/h, axis=1)
    second = np.linalg.solve(matrix, rhs.T).T
    values = []
    for t in target_times:
        if t == 0:
            t = next(t for t in target_times if t > 0)
        for x in target_x:
            assert xs[0] <= x <= xs[-1]
            i = min(np.searchsorted(xs, x, side='right')-1, len(xs)-2)
            b = (x-xs[i])/h[i]
            a = 1-b
            v = a*w[:, i]+b*w[:, i+1]+((a**3-a)*second[:, i]+(b**3-b)*second[:, i+1])*h[i]**2/6
            dx = (w[:, i+1]-w[:, i])/h[i]+((1-3*a*a)*second[:, i]+(3*b*b-1)*second[:, i+1])*h[i]/6
            dxx = a*second[:, i]+b*second[:, i+1]
            if t < times[0] or t >= times[-1]:
                edge = 0 if t < times[0] else -1
                wt = v[edge]/times[edge]
                v, dx, dxx = [y[edge]*t/times[edge] for y in (v, dx, dxx)]
            else:
                k = np.searchsorted(times, t, side='right')-1
                wt = (v[k+1]-v[k])/(times[k+1]-times[k])
                b = (t-times[k])/(times[k+1]-times[k])
                v, dx, dxx = [(1-b)*y[k]+b*y[k+1] for y in (v, dx, dxx)]
            density = (1-x*dx/(2*v))**2-dx*dx/4*(1/v+.25)+dxx/2
            assert v > 0 and wt > 0 and density > 0
            values.append(wt/density)
    assert np.isfinite(values).all()
    return values


def shifted_quotes(source, quote_index, shift):
    quotes = np.asarray(source['implied_volatilities']).copy()
    if quote_index is None:
        quotes += shift
    else:
        quotes[quote_index] += shift
    return quotes


def panel_batches(fixture, case):
    bases, market = inputs()
    # Reset to the same PCG64 seed for every panel: batch/path identities align
    # across all quotes. Add batch means, never marginal standard errors.
    return np.asarray([
        recalibrated_bump_means(bases[case['base_case']], market | case['market'],
                                case['contract'], panel['scenarios'], **fixture['sampling'])
        for panel in case['panels']
    ])


def summarize(means):
    return dict(estimates=means.mean(axis=0).tolist(),
                standard_errors=(means.std(axis=0, ddof=1)/math.sqrt(len(means))).tolist())


def verify_fixture():
    fixture = json.loads(FIXTURE.read_text())
    source, target = fixture['quotes'], fixture['target']
    bump = fixture['sampling']['bump']
    bases, _ = inputs()
    for case in fixture['cases']:
        assert [p['quote_index'] for p in case['panels']] == [None] + fixture['quote_indices']
        for panel in case['panels']:
            assert [s['shift'] for s in panel['scenarios']] == [-bump/2, bump/2, -bump, bump, -2*bump, 2*bump]
            for scenario in panel['scenarios']:
                quotes = shifted_quotes(source, panel['quote_index'], scenario['shift'])
                expected = dupire_target(source['maturity_nodes'], source['log_moneyness_nodes'],
                                         quotes, target['time_nodes'], target['log_moneyness_nodes'])
                np.testing.assert_allclose(expected, scenario['target_variances'], rtol=0, atol=3e-15)
                assert all(target['floor'] <= v <= target['cap'] for v in expected)
                leverage = np.asarray(scenario['squared_leverage'])
                base = bases[case['base_case']]
                assert leverage.size == len(base['times'])*len(base['log_nodes'])
                assert np.isfinite(leverage).all() and (leverage > 0).all()
        panels = panel_batches(fixture, case)
        for panel, means in zip(case['panels'], panels):
            np.testing.assert_allclose(means, panel['batch_means'], rtol=0, atol=1e-9)
            result = summarize(means)
            assert max(result['standard_errors']) < fixture['acceptance']['reference_se']
            print(json.dumps(dict(case=case['id'], quote_index=panel['quote_index'], **result)), flush=True)
        summed = panels[1:].sum(axis=0)
        np.testing.assert_allclose(summed, case['sum_batch_means'], rtol=0, atol=1e-9)
        result = summarize(summed)
        assert max(result['standard_errors']) < fixture['acceptance']['reference_se']
        print(json.dumps(dict(case=case['id'], quote_indices=fixture['quote_indices'], **result)), flush=True)


def build_fixture(output, calibration_source):
    # Explicit capture mode only: calibration inputs, never expected prices/risks.
    import rust_pricing as rp
    from test_rough_dividend_continuous import FIXTURE as CONTRACTS, compile_plan, make_request

    bases, market = inputs()
    fixture = dict(
        scope='finite_quote_iv_risk_conditional_on_separately_recalibrated_inputs',
        calibration_source=calibration_source,
        source_fixture='rough-continuous-barrier-reference.json',
        interpolation='natural-cubic-w-linear-time-v1',
        units='currency_per_unit_absolute_residual_forward_iv',
        quotes=dict(maturity_nodes=[.25, 1.25], log_moneyness_nodes=[-.75, 0., .75],
                    implied_volatilities=[.22, .20, .21, .24, .22, .23]),
        target=dict(time_nodes=[0., market['fixing_time'], market['expiry_time']],
                    log_moneyness_nodes=[-.5, 0., .5], floor=1e-8, cap=4.),
        quote_indices=[4, 1, 5],
        calibration=dict(particle_count=64, seed=42, log_bandwidth=.35, minimum_effective_samples=5.),
        sampling=dict(bump=.01, seed=20261010, batches=32, pairs=8192),
        production_sampling=dict(points_per_scramble=32768, scramble_count=16),
        acceptance=dict(reference_se=.35, production_se=.35, difference_plus_4se=2.),
        columns=['vega_h_over_2', 'vega_h', 'vega_2h', 'vega_h_over_2_minus_h', 'vega_h_minus_2h'],
        cases=[],
    )
    source, target = fixture['quotes'], fixture['target']
    for original in CONTRACTS['cases']:
        case = {key: original[key] for key in ['id', 'base_case', 'market', 'contract']}
        data = json.loads(make_request(case).to_json())
        grid = data['model']['local_variance_grid']
        assert grid['time_nodes'] == target['time_nodes']
        assert grid['log_forward_moneyness_nodes'] == target['log_moneyness_nodes']
        assert (grid['floor'], grid['cap']) == (target['floor'], target['cap'])
        case['panels'] = []
        for quote_index in [None] + fixture['quote_indices']:
            panel = dict(quote_index=quote_index, scenarios=[])
            bump = fixture['sampling']['bump']
            for shift in [-bump/2, bump/2, -bump, bump, -2*bump, 2*bump]:
                quotes = shifted_quotes(source, quote_index, shift)
                variances = dupire_target(source['maturity_nodes'], source['log_moneyness_nodes'],
                                          quotes, target['time_nodes'], target['log_moneyness_nodes'])
                assert all(target['floor'] <= v <= target['cap'] for v in variances)
                grid['values'] = variances
                plan = compile_plan(case, rp.PricingRequest.from_json(json.dumps(data)))
                assert plan.time_nodes == bases[case['base_case']]['times']
                assert plan.lsv_log_moneyness_nodes == bases[case['base_case']]['log_nodes']
                panel['scenarios'].append(dict(shift=shift, target_variances=variances,
                                               squared_leverage=plan.lsv_squared_leverage))
            case['panels'].append(panel)
        panels = panel_batches(fixture, case)
        for panel, means in zip(case['panels'], panels):
            panel['batch_means'] = means.tolist()
            print(json.dumps(dict(case=case['id'], quote_index=panel['quote_index'], **summarize(means))), flush=True)
        case['sum_batch_means'] = panels[1:].sum(axis=0).tolist()
        print(json.dumps(dict(case=case['id'], quote_indices=fixture['quote_indices'],
                              **summarize(panels[1:].sum(axis=0)))), flush=True)
        fixture['cases'].append(case)
    output.write_text(json.dumps(fixture, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, help='explicitly capture calibration inputs and write new reference data')
    parser.add_argument('--calibration-source', help='provenance of the Rust wheel used only for calibration')
    args = parser.parse_args()
    if args.build:
        if not args.calibration_source:
            parser.error('--build requires --calibration-source')
        build_fixture(args.build, args.calibration_source)
    else:
        verify_fixture()
