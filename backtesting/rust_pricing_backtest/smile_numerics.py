"""P0 terminal-smile acceptance, with immutable legacy evidence and dispositions.

Quote-price finite differences validate paper equation (9). Separate paired
rebuilds of whole SSVI/eSSVI surfaces validate consistency with simulated LV
prices. Independently generated quote nodes are never treated as SSVI parameters.
"""
import argparse
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .domain import RiskBasis, fingerprint, require
from .market_shocks import MarketShockMap, solve_market_hedge
from .market_shock_numerics import run_panel as run_market_panel
from .numerics import black_scholes_call, relative_distance, threshold_check

METHOD = 'terminal_payoff_replication_iv_v1'


def smile_request(policy, smile, *, strike=100., side='call', rate=0., seed=7,
                  steps=None, reporting_nodes=None, samples=None, scale=1., risk=True):
    import rust_pricing as rp
    steps = steps or policy['base_steps']
    count = reporting_nodes or policy['base_reporting_nodes']
    require(policy['expiry_days'] % steps == 0, 'smile_policy', 'integer-day steps required')
    start = date.fromisoformat(policy['valuation_date'])
    days = policy['expiry_days']
    maturity = days/365
    times = np.arange(steps+1)*days/steps/365
    x = np.linspace(-policy['log_moneyness_extent'], policy['log_moneyness_extent'], policy['local_nodes'])
    theta_times = [maturity/2, maturity]
    theta = np.asarray(smile['theta'])*scale**2
    terminal_slope = (theta[1]-theta[0])/(theta_times[1]-theta_times[0])
    if smile['family'] == 'essvi':
        slices = [rp.EssviSlice(t, th, psi*scale, rpsi*scale)
                  for t, th, psi, rpsi in zip(theta_times, theta, smile['psi'], smile['rho_psi'])]
        model = rp.Model.local_volatility_from_essvi(slices, terminal_slope, times, x, 1e-8, 4.)
    elif smile['family'] == 'power':
        model = rp.Model.local_volatility_from_standard_ssvi_power_law(
            theta_times, theta, terminal_slope, smile['rho'], smile['eta'], smile['gamma'], times, x, 1e-8, 4.)
    else:
        require(smile['family'] == 'heston', 'smile_policy', 'unknown surface family')
        model = rp.Model.local_volatility_from_standard_ssvi_heston_like(
            theta_times, theta, terminal_slope, smile['rho'], smile['lambda'], times, x, 1e-8, 4.)
    market = rp.Market.equity(2, 1, policy['spot'],
        rp.DiscountCurve(10, [0., 1.], [1., math.exp(-rate)]), rp.DiscountCurve(11, [0., 1.], [1., 1.]))
    dates = [start+timedelta(days=days*i//steps) for i in range(1, steps+1)]
    request = rp.PricingRequest(start, rp.Product.european_vanilla(1, 2, dates[-1], strike, 1., side),
        market, model, rp.Engine.pseudo_monte_carlo(seed, samples or policy['samples'], antithetic=True),
        rp.RiskRequest(delta=risk, vega=risk, vega_kt_maturity_nodes=dates if risk else None,
                       vega_kt_log_forward_moneyness_nodes=x if risk else None,
                       vega_kt_relative_density_threshold=1e-12 if risk else None))
    p = json.loads(request.to_json())
    # Keep the calibrated local grid fixed. Reporting nodes are a nested subset
    # of its sampled surface; this changes neither paths nor Dupire derivatives.
    require((len(x)-1) % (count-1) == 0, 'smile_policy', 'nested reporting nodes required')
    stride = (len(x)-1)//(count-1)
    b = p['model']['reporting_iv_basis']
    values = np.asarray(b['implied_volatilities']).reshape(steps, len(x))[:, ::stride]
    b.update(log_forward_moneyness_nodes=list(x[::stride]), shape=[steps, count],
             implied_volatilities=values.ravel().tolist())
    if risk:
        p['risk']['vega_kt']['log_forward_moneyness_nodes'] = list(x[::stride])
    return p


def quote_price(payload, *, spot=None, shift=None):
    """Black call/put from market quotes, independent of native pricing/Gamma."""
    p, market = payload['product'], payload['market']
    start = date.fromisoformat(payload['valuation_date'])
    t = (date.fromisoformat(p['expiry'])-start).days/365
    s = market['spot'] if spot is None else spot
    # This panel explicitly uses flat continuously compounded curves.
    rate = -math.log(market['discount_curve']['discount_factors'][-1])
    forward = s*math.exp(rate*t)
    b = payload['model']['reporting_iv_basis']
    width = len(b['log_forward_moneyness_nodes'])
    ivs = np.asarray(b['implied_volatilities'][-width:])
    if shift is not None:
        ivs = ivs+np.asarray(shift)[-width:]
    sigma = float(np.interp(math.log(p['strike']/forward), b['log_forward_moneyness_nodes'], ivs))
    call = black_scholes_call(s, p['strike'], t, sigma, rate)['price']
    return call if p['side']['type'] == 'call' else call-s+p['strike']*math.exp(-rate*t)


def evaluate(payload, workers=1):
    import rust_pricing as rp
    return rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(payload)),
                                 worker_threads=workers, reduction_block_size=256).evaluate()


def dispose_legacy(market):
    """Close only the known incomparable-shock comparisons, never unknown failures."""
    out = []
    for c in market['original_acceptance']['checks']:
        if c['passed']:
            continue
        replacements = [q for q in market['checks']
            if q['check'] == 'fixed_market_hedge_grid_stability'
            and q.get('seed') == c.get('seed') and q.get('reporting_nodes') == c.get('nodes')]
        resolved = c['check'] == 'hedge_grid_stability' and bool(replacements) and all(q['passed'] for q in replacements)
        out.append({'original': c, 'status': 'superseded_comparison' if resolved else 'unresolved',
                    'reason': 'independent quote shocks changed with reporting grid' if resolved else 'no validated replacement',
                    'replacements': replacements})
    return out


def run_smile_panel(policy):
    import rust_pricing as rp
    checks, rows, paired, hedge_rows = [], [], [], []
    def check(name, observed, limit, **context):
        checks.append(threshold_check(name, float(observed), float(limit), **context))
    for smile in policy['smiles']:
        for rate in policy['rates']:
            for strike in policy['strikes']:
                for side in policy['sides']:
                    p = smile_request(policy, smile, strike=strike, side=side, rate=rate)
                    r = evaluate(p)
                    kt = r.vega_kt
                    context = dict(smile=smile['name'], rate=rate, strike=strike, side=side)
                    check('exact_replication_policy', int(kt.policy_label != METHOD), 0, **context)
                    check('sampling_moments_not_applicable', sum(e.sample_variance is not None or e.price_covariance is not None for e in kt.estimates), 0, **context)
                    b = p['model']['reporting_iv_basis']
                    n = len(b['implied_volatilities']); width = len(b['log_forward_moneyness_nodes'])
                    h = policy['quote_bump']; fd = np.zeros(n)
                    for j in range(n-width, n):
                        v = np.zeros(n); v[j] = h
                        fd[j] = (quote_price(p, shift=v)-quote_price(p, shift=-v))/(2*h)
                    risk_error = float(np.max(abs(np.asarray(kt.raw_buckets)-fd)))
                    check('equation9_quote_price_gradient', risk_error, policy['quote_gradient_absolute_tolerance'], **context)
                    check('projection_conservation', abs(math.fsum(kt.raw_buckets)-kt.projection.pre_projection), policy['algebra_absolute_tolerance'], **context)
                    check('projection_residual', abs(kt.projection.signed_residual), 0., **context)
                    market = quote_price(p)
                    check('native_price_vs_market', abs(r.value-market),
                          policy['statistical_multiplier']*r.standard_error+policy['price_absolute_tolerance'], **context)
                    delta_h = policy['spot']*1e-5
                    delta_ref = (quote_price(p, spot=policy['spot']+delta_h)-quote_price(p, spot=policy['spot']-delta_h))/(2*delta_h)
                    check('frozen_grid_delta_vs_market', abs(r.delta_raw-delta_ref),
                          policy['statistical_multiplier']*r.delta.raw.standard_error+policy['delta_absolute_tolerance'], **context)
                    rows.append(dict(**context, native_price=r.value, price_se=r.standard_error, market_price=market,
                        delta=r.delta_raw, delta_reference=delta_ref, risk_max_absolute_error=risk_error,
                        native_policy=kt.policy_label, request_hash=fingerprint(p)))

        # Time/reporting/seed refinement changes numerical grids independently.
        references = {}
        for steps in policy['time_steps']:
            for count in policy['reporting_nodes']:
                for seed in policy['seeds']:
                    p = smile_request(policy, smile, strike=100., steps=steps, reporting_nodes=count, seed=seed)
                    r = evaluate(p); kt = r.vega_kt
                    context = dict(smile=smile['name'], steps=steps, reporting_nodes=count, seed=seed)
                    risk = math.fsum(kt.raw_buckets)
                    key = count
                    references.setdefault(key, risk)
                    check('terminal_risk_time_and_seed_invariance', abs(risk-references[key]), policy['algebra_absolute_tolerance'], **context)
                    check('refined_native_price_vs_market', abs(r.value-quote_price(p)),
                        policy['statistical_multiplier']*r.standard_error+policy['price_absolute_tolerance'], **context)

        # Freeze both the baseline IV interpolant and the shocks. Merely sampling
        # a curved SSVI surface on different grids changes its linear interpolant.
        # These two effects must not be confused with numerical hedge instability.
        template = smile_request(policy, smile, reporting_nodes=policy['hedge_market_nodes'], samples=64)
        b = template['model']['reporting_iv_basis']; risk_config = template['risk']['vega_kt']
        n = policy['hedge_market_nodes']; selected = [policy['base_steps']//2-1, policy['base_steps']-1]
        b['maturity_nodes'] = [b['maturity_nodes'][i] for i in selected]
        b['implied_volatilities'] = [v for i in selected for v in b['implied_volatilities'][i*n:(i+1)*n]]
        b['shape'] = [2,n]
        risk_config['maturity_nodes'] = [risk_config['maturity_nodes'][i] for i in selected]
        market = RiskBasis(date.fromisoformat(template['valuation_date']),
            tuple(date.fromisoformat(d) for d in risk_config['maturity_nodes']),
            tuple(b['log_forward_moneyness_nodes']), method=METHOD)
        reference = None
        for count in policy['hedge_reporting_nodes']:
            x = tuple(np.linspace(-policy['log_moneyness_extent'], policy['log_moneyness_extent'], count))
            reporting = RiskBasis(market.valuation_date, market.maturity_dates, x, method=METHOD)
            mapping = MarketShockMap(reporting, market)
            payloads, risks = [], []
            for strike in policy['hedge_strikes']:
                p = json.loads(json.dumps(template)); p['product']['strike'] = strike
                p['model']['reporting_iv_basis'].update(log_forward_moneyness_nodes=list(x), shape=[2,count],
                    implied_volatilities=mapping.lift_shock(b['implied_volatilities']).tolist())
                p['risk']['vega_kt']['log_forward_moneyness_nodes'] = list(x)
                payloads.append(p); risks.append(np.asarray(evaluate(p).vega_kt.raw_buckets))
            bound = policy['hedge_position_limit']
            solution = solve_market_hedge(mapping, -risks[0], np.asarray(risks[1:]).T,
                [0.,0.], [-bound]*2, [bound]*2, [0.,0.], ridge=policy['hedge_ridge'], tolerance=policy['solver_tolerance'])
            mapped = np.array([mapping.project_risk(r) for r in risks])
            if reference is None:
                reference = (mapped, solution.solution.quantities)
            context = dict(smile=smile['name'], reporting_nodes=count)
            check('smile_fixed_market_risk_grid_stability', relative_distance(mapped,reference[0]),
                  policy['algebra_absolute_tolerance'], **context)
            check('smile_fixed_market_quantity_grid_stability', relative_distance(solution.solution.quantities,reference[1]),
                  policy['hedge_quantity_relative_tolerance'], **context)
            a = .01*mapped[1:].T; target = .01*mapped[0]
            oracle = np.linalg.solve(a.T@a+policy['hedge_ridge']*np.eye(2),a.T@target)
            require(np.all(abs(oracle)<bound), 'smile_hedge_scope', 'oracle optimum reaches a bound')
            check('smile_independent_hedge_solver', max(abs(np.asarray(solution.solution.quantities)-oracle)),
                  policy['hedge_oracle_absolute_tolerance'], **context)
            witness = np.asarray(solution.complement['unit_complement_shock'])
            h = policy['quote_bump']; quantities = [-1.,*solution.solution.quantities]
            fd = math.fsum(q*(quote_price(p,shift=h*witness)-quote_price(p,shift=-h*witness))/(2*h)
                           for p,q in zip(payloads,quantities))
            check('smile_omitted_direction_price_effect',abs(fd-solution.complement['complement_l2']),
                  policy['quote_gradient_absolute_tolerance'], **context)
            hedge_rows.append(dict(**context, quantities=list(solution.solution.quantities),
                market_risk=mapped.tolist(), complement=solution.complement, omitted_direction_price_derivative=fd))

        # Entire surfaces are rebuilt for both signs. Batch differences retain
        # the common-random-number covariance; marginal price SEs are not added.
        for bump in policy['parameter_bumps']:
            up = smile_request(policy, smile, scale=1+bump, samples=policy['paired_batch_samples'], risk=False)
            down = smile_request(policy, smile, scale=1-bump, samples=policy['paired_batch_samples'], risk=False)
            base = smile_request(policy, smile)
            risk = np.asarray(evaluate(base).vega_kt.raw_buckets)
            div = (np.asarray(up['model']['reporting_iv_basis']['implied_volatilities'])
                   - np.asarray(down['model']['reporting_iv_basis']['implied_volatilities']))/(2*bump)
            expected = float(risk@div)
            quoted_fd = (quote_price(up)-quote_price(down))/(2*bump)
            check('surface_parameter_market_chain_rule', abs(expected-quoted_fd),
                  policy['quote_gradient_absolute_tolerance'], smile=smile['name'], bump=bump)
            values = []
            for i in range(policy['paired_batches']):
                up['engine']['master_seed'] = down['engine']['master_seed'] = policy['paired_first_seed']+i
                values.append((evaluate(up).value-evaluate(down).value)/(2*bump))
            mean = float(np.mean(values)); se = float(np.std(values, ddof=1)/math.sqrt(len(values)))
            check('recalibrated_lv_parameter_crn', abs(mean-expected),
                  policy['statistical_multiplier']*se+policy['parameter_relative_tolerance']*abs(expected),
                  smile=smile['name'], bump=bump)
            paired.append(dict(smile=smile['name'], bump=bump, native_crn=mean, paired_standard_error=se,
                projected_risk=expected, market_fd=quoted_fd, batch_differences=values,
                independent_batches=len(values), antithetic_pairs_per_batch=policy['paired_batch_samples']))
    module = getattr(rp, 'rust_pricing', rp)
    return dict(status='passed' if all(c['passed'] for c in checks) else 'failed', checks=checks,
        observations=rows, paired_surface_rebuilds=paired, fixed_market_hedges=hedge_rows,
        native_binary_hash=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        policy=policy, policy_hash=fingerprint(policy))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    base = json.loads((args.policy.parent/policy['base_policy']).read_text())
    market_policy = json.loads((args.policy.parent/policy['market_policy']).read_text())
    market = run_market_panel(market_policy, base)
    smile = run_smile_panel(policy)
    legacy = dispose_legacy(market)
    passed = market['status'] == smile['status'] == 'passed' and all(x['status'] != 'unresolved' for x in legacy)
    report = dict(document_kind='vega_kt_p0_numerical_acceptance', schema_version=3,
        status='passed' if passed else 'failed', acceptance_scope=policy['scope'],
        path_dependent_gamma_recovery='not_accepted', market_gate=market, smile_gate=smile,
        legacy_dispositions=legacy, original_results_preserved=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    summary = dict(status=report['status'], scope='P0 terminal replication',
        market_checks=len(market['checks']), smile_checks=len(smile['checks']),
        failed=sum(not c['passed'] for c in market['checks']+smile['checks']),
        superseded_legacy_failures=sum(x['status']=='superseded_comparison' for x in legacy),
        unresolved_legacy_failures=sum(x['status']=='unresolved' for x in legacy))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))
    return int(not passed)


if __name__ == '__main__':
    raise SystemExit(main())
