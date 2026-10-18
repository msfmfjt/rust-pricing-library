"""Versioned flat-terminal gate with fixed market shocks and retained v1 failures.

The v1 gate is rerun unchanged and embedded verbatim. Only its hedge-grid
comparisons are superseded in this scoped v2 result. Passing this gate does
not certify general local-volatility recovery or a hedging strategy.
"""

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np

from .domain import RiskBasis, fingerprint, require
from .market_shocks import MarketShockMap, solve_market_hedge
from .numerics import black_scholes_call, flat_request, relative_distance, run_panel as run_original_panel, threshold_check


METHOD = 'flat_variance_terminal_measure_pathwise_v1'


def _basis(base, count, steps):
    start = date.fromisoformat(base['valuation_date'])
    dates = tuple(start+timedelta(days=base['expiry_days']*i//steps) for i in range(1, steps+1))
    return RiskBasis(start, dates, tuple(np.linspace(-base['log_moneyness_extent'],
                      base['log_moneyness_extent'], count)), method=METHOD)


def _exact_risk(base, basis, strike):
    # Independent interpolation of individual quote bumps, not MarketShockMap.
    x = np.log(strike/base['spot'])
    n = len(basis.log_moneyness)
    hats = np.array([np.interp(x, basis.log_moneyness, e) for e in np.eye(n)])
    risk = np.zeros(basis.size)
    risk[-n:] = hats*black_scholes_call(base['spot'], strike, base['expiry_days']/365, base['volatility'])['vega']
    return risk


def _quote_price(base, basis, strike, change):
    n = len(basis.log_moneyness)
    iv = base['volatility']+np.interp(np.log(strike/base['spot']), basis.log_moneyness, change[-n:])
    return black_scholes_call(base['spot'], strike, base['expiry_days']/365, iv)['price']


def _quote_gradient(base, basis, strike, bump):
    out = np.zeros(basis.size)
    for j in range(basis.size-len(basis.log_moneyness), basis.size):
        change = np.zeros(basis.size)
        change[j] = bump
        out[j] = (_quote_price(base, basis, strike, change)-_quote_price(base, basis, strike, -change))/(2*bump)
    return out


def _solve(mapping, risks, base):
    m = len(risks)-1
    bound = base['hedge_position_limit']
    return solve_market_hedge(mapping, -risks[0], np.asarray(risks[1:]).T,
        [0.]*m, [-bound]*m, [bound]*m, [0.]*m,
        ridge=base['hedge_ridge'], tolerance=base['solver_tolerance'])


def _exact_quantities(risks, base):
    # Independent normal-equation solution; the declared fixture has inactive
    # box constraints. Do not compare unconstrained and constrained optima.
    a, target = .01*np.asarray(risks[1:]).T, .01*np.asarray(risks[0])
    h = np.linalg.solve(a.T@a+base['hedge_ridge']*np.eye(a.shape[1]), a.T@target)
    require(np.all(abs(h) < base['hedge_position_limit']), 'market_gate_scope', 'oracle optimum reaches a bound')
    return h


def run_panel(policy, base):
    import rust_pricing as rp
    require(fingerprint(base) == policy['base_policy_hash'], 'market_gate_policy', 'original policy fingerprint differs')
    require(base['rate'] == 0., 'market_gate_scope', 'zero-carry fixture required')
    require(policy['reference_reporting_nodes'] in policy['reporting_node_counts'],
            'market_gate_policy', 'reference reporting grid is missing')
    require(bool(policy['market_node_counts']) and bool(policy['reporting_node_counts']),
            'market_gate_policy', 'nonempty market and reporting panels required')
    original = run_original_panel(base)
    replaced = [c for c in original['checks'] if c['check'] == 'hedge_grid_stability']
    checks = [dict(c, origin='unchanged_v1') for c in original['checks'] if c['check'] != 'hedge_grid_stability']
    strikes = [base['strike'], *base['hedge_strikes']]
    steps, samples = base['base_time_steps'], base['base_sampling_units']
    cache, requests, rows = {}, [], []

    def check(name, observed, limit, **context):
        checks.append(dict(threshold_check(name, float(observed), float(limit), **context), origin='fixed_market_shocks_v2'))

    def evaluate(seed, count, strike, *, count_samples=samples, time_steps=steps, covariance=False):
        request = flat_request(base, seed=seed, samples=count_samples, steps=time_steps, nodes=count, strike=strike)
        if covariance:
            payload = json.loads(request.to_json())
            payload['risk']['vega_kt']['full_bucket_covariance'] = True
            request = rp.PricingRequest.from_json(json.dumps(payload))
        plan = rp.PricingPlan.compile(request, worker_threads=1, reduction_block_size=256)
        result = plan.evaluate()
        require(result.vega_kt.policy_label == METHOD, 'market_gate_scope', 'terminal-measure method required')
        requests.append(dict(seed=seed, reporting_nodes=count, strike=strike,
            samples=count_samples, steps=time_steps, full_covariance=covariance,
            request_fingerprint=plan.request_fingerprint, plan_fingerprint=plan.plan_fingerprint))
        return result

    for seed in base['seeds']:
        for count in policy['reporting_node_counts']:
            reporting = _basis(base, count, steps)
            native, exact = [], []
            for strike in strikes:
                output = evaluate(seed, count, strike)
                r = np.asarray(output.vega_kt.raw_buckets)
                ref = _exact_risk(base, reporting, strike)
                se = np.sqrt(np.asarray([e.sample_variance for e in output.vega_kt.estimates])/samples)
                check('native_bucket_reference_six_se', max(0., float(np.max(abs(r-ref)-base['statistical_multiplier']*se))),
                      policy['quote_bump_absolute_tolerance'], seed=seed, reporting_nodes=count, strike=strike)
                native.append(r)
                exact.append(ref)
            cache[seed, count] = (reporting, np.array(native), np.array(exact))

    for seed in base['seeds']:
        for market_count in policy['market_node_counts']:
            market = _basis(base, market_count, steps)
            ref_reporting, ref_native, _ = cache[seed, policy['reference_reporting_nodes']]
            ref_map = MarketShockMap(ref_reporting, market)
            ref_risk = np.array([ref_map.project_risk(r) for r in ref_native])
            ref_h = _solve(ref_map, ref_native, base).solution.quantities
            market_exact = np.array([_exact_risk(base, market, k) for k in strikes])
            oracle_h = _exact_quantities(market_exact, base)
            for count in policy['reporting_node_counts']:
                if count < market_count:
                    continue
                reporting, native, exact = cache[seed, count]
                mapping = MarketShockMap(reporting, market)
                actual_risk = np.array([mapping.project_risk(r) for r in native])
                actual = _solve(mapping, native, base)
                actual_h = actual.solution.quantities
                exact_h = _solve(mapping, exact, base).solution.quantities
                context = dict(seed=seed, market_nodes=market_count, reporting_nodes=count)
                risk_change = relative_distance(actual_risk, ref_risk)
                quantity_change = relative_distance(actual_h, ref_h)
                check('fixed_market_risk_grid_stability', risk_change, policy['common_risk_relative_tolerance'], **context)
                check('fixed_market_hedge_grid_stability', quantity_change, policy['common_quantity_relative_tolerance'], **context)
                check('exact_market_risk_chain_rule', np.max(abs(np.array([mapping.project_risk(r) for r in exact])-market_exact)),
                      policy['algebra_absolute_tolerance'], **context)
                check('independent_exact_hedge_solution', np.max(abs(np.asarray(exact_h)-oracle_h)),
                      policy['exact_solver_quantity_absolute_tolerance'], **context)
                rows.append(dict(**context, map_id=mapping.map_id, market_basis_id=market.basis_id,
                    reporting_basis_id=reporting.basis_id, native_quantities=list(actual_h),
                    exact_solver_quantities=list(exact_h), independent_exact_quantities=oracle_h.tolist(),
                    risk_relative_change=risk_change, quantity_relative_change=quantity_change,
                    market_residual=list(actual.market_residual),
                    reporting_residual=list(actual.reporting_residual), complement_audit=actual.complement))

    covariance_rows = []
    fine_count, market_count = policy['covariance_reporting_nodes'], policy['covariance_market_nodes']
    cs, ct = policy['covariance_samples'], policy['covariance_time_steps']
    mapping = MarketShockMap(_basis(base, fine_count, ct), _basis(base, market_count, ct))
    for strike in strikes:
        fine = evaluate(base['seeds'][0], fine_count, strike, count_samples=cs, time_steps=ct, covariance=True).vega_kt
        direct = evaluate(base['seeds'][0], market_count, strike, count_samples=cs, time_steps=ct, covariance=True).vega_kt
        raw_c = np.asarray(fine.full_bucket_covariance).reshape(mapping.reporting.size, -1)
        projected_c = mapping.project_sample_covariance(raw_c)
        direct_c = np.asarray(direct.full_bucket_covariance).reshape(mapping.market.size, -1)
        covariance_gap = relative_distance(projected_c, direct_c)
        price_cov = mapping.project_risk([e.price_covariance for e in fine.estimates])
        price_cov_gap = relative_distance(price_cov, [e.price_covariance for e in direct.estimates])
        r = mapping.project_risk(fine.raw_buckets)
        fd = _quote_gradient(base, mapping.market, strike, policy['quote_bump'])
        se = np.sqrt(np.maximum(0., np.diag(projected_c))/cs)
        check('full_sample_covariance_transform', covariance_gap, policy['covariance_relative_tolerance'], strike=strike)
        check('price_covariance_transform', price_cov_gap, policy['covariance_relative_tolerance'], strike=strike)
        check('same_sample_mean_transform', relative_distance(r, direct.raw_buckets), policy['common_risk_relative_tolerance'], strike=strike)
        check('market_quote_price_difference_six_se', max(0., float(np.max(abs(r-fd)-base['statistical_multiplier']*se))),
              policy['quote_bump_absolute_tolerance'], strike=strike)
        check('exact_market_quote_price_difference', np.max(abs(_exact_risk(base, mapping.market, strike)-fd)),
              policy['quote_bump_absolute_tolerance'], strike=strike)
        covariance_rows.append(dict(strike=strike, independent_antithetic_units=cs,
            full_covariance_relative_difference=covariance_gap, price_covariance_relative_difference=price_cov_gap,
            market_standard_errors=se.tolist(), market_quote_price_gradient=fd.tolist()))

    # Show an actual residual in the shock directions omitted by a coarse
    # objective. Verify its price effect independently of the risk projection.
    count, market_count = policy['independent_shock_reporting_nodes'], policy['independent_shock_market_nodes']
    reporting, native, exact = cache[base['seeds'][0], count]
    mapping = MarketShockMap(reporting, _basis(base, market_count, steps))
    h = np.asarray(_solve(mapping, exact, base).solution.quantities)
    residual = -exact[0]+h@exact[1:]
    audit = mapping.complement_audit(residual)
    complement = np.asarray(audit['complement_buckets'])
    witness = np.asarray(audit['unit_complement_shock'])
    check('complement_orthogonal_to_market_shocks', np.max(abs(mapping.project_risk(complement))),
          policy['algebra_absolute_tolerance'])
    check('residual_pythagorean_identity', abs(audit['reporting_l2']**2-audit['covered_l2']**2-audit['complement_l2']**2),
          policy['algebra_absolute_tolerance'])
    bump = policy['quote_bump']
    def portfolio_price(sign):
        prices = np.array([_quote_price(base, reporting, strike, sign*bump*witness) for strike in strikes])
        return -prices[0]+h@prices[1:]
    fd = (portfolio_price(1)-portfolio_price(-1))/(2*bump)
    check('omitted_shock_price_difference', abs(fd-float(residual@witness)), policy['quote_bump_absolute_tolerance'])
    audit.update(market_nodes=market_count, reporting_nodes=count, exact_hedge_quantities=h.tolist(),
        exact_residual=residual.tolist(), omitted_shock_price_derivative=fd,
        native_complement=_solve(mapping, native, base).complement)

    module = getattr(rp, 'rust_pricing', rp)
    return dict(document_kind='flat_terminal_market_shock_acceptance', schema_version=2,
        status='passed' if all(c['passed'] for c in checks) else 'failed', full_acceptance='not_established',
        policy=policy, policy_hash=fingerprint(policy), base_policy_hash=fingerprint(base),
        native_binary_hash=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        runner_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_acceptance=original, superseded_original_checks=replaced,
        checks=checks, observations=rows, covariance_observations=covariance_rows,
        independent_shock_audit=audit, requests=requests,
        limitations=['Necessary spatially flat terminal-payoff gate only; the native adapter remains unvalidated.',
                     'Different market shock families can require different hedge quantities; only reporting grids vary within a comparison.',
                     'The complement is an L2 reporting-quote metric, not empirical market covariance or a hedge-effectiveness certificate.',
                     'The original v1 gate is retained with all of its failed checks.'])


def markdown_report(report):
    original = report['original_acceptance']
    failed = sum(not c['passed'] for c in report['checks'])
    old_failed = sum(not c['passed'] for c in original['checks'])
    audit = report['independent_shock_audit']
    return '\n'.join([
        '# 固定市場ショックによるVegaKT数値受入 v0.2', '',
        f"対象範囲の判定: **{report['status']}**。{len(report['checks'])}項目中{failed}件失敗。", '',
        f"元のv0.1: **{original['status']}**、{len(original['checks'])}項目中{old_failed}件失敗。元の判定は変更していない。", '',
        '市場格子9・17・33節点をそれぞれ固定し、対応する報告格子だけを細分化。一般smile・戦略の受入は未達。', '',
        f"最大リスク相対差: {max(r['risk_relative_change'] for r in report['observations']):.4g}",
        f"最大数量相対差: {max(r['quantity_relative_change'] for r in report['observations']):.4g}", '',
        '## 粗い市場ショックに含まれない残余リスク', '',
        f"33節点上の残余リスクL2: {audit['reporting_l2']:.10f}",
        f"9節点市場ショックに直交する成分L2: {audit['complement_l2']:.10f}",
        f"L2で1 vol pointの最悪方向による線形価格変動: {audit['worst_complement_one_point_pnl']:.10f}", '',
        'この成分を捨てて「完全ヘッジ」とは判定しない。方向ベクトルと独立価格差分はreport.jsonへ保存。', '',
        f"Policy SHA-256: `{report['policy_hash']}`", '',
        f"Native SHA-256: `{report['native_binary_hash']}`", ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding='utf-8'))
    base = json.loads((args.policy.parent/policy['base_policy']).read_text(encoding='utf-8'))
    report = run_panel(policy, base)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (args.output/'report.md').write_text(markdown_report(report), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], checks=len(report['checks']),
        failed=sum(not c['passed'] for c in report['checks']),
        retained_original_failures=sum(not c['passed'] for c in report['original_acceptance']['checks']),
        full_acceptance=report['full_acceptance'])))
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
