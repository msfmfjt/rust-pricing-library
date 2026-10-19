"""Joint market-IV risk and lagged P&L on an explicit local-variance extension.

All instruments use the same counter stream. Covariances describe Monte Carlo
estimation error, never market-return covariance. Calibration/discretization bias
is not included. Passing runtime gates is not universal market acceptance.
"""
from copy import deepcopy
from datetime import date, timedelta
import json
import math
import hashlib
from pathlib import Path

import numpy as np
from .domain import BacktestError, fingerprint, require
from .multi_step_extension import MultiStepExtensionCalibrator
from .multi_step_provider import gamma_boundaries
from .multi_step_smile import source_marginals
from .discrete_smile import _black


def implementation_identity():
    import rust_pricing as rp
    root = Path(__file__).parent
    return fingerprint(dict(native=hashlib.sha256(Path(getattr(rp, 'rust_pricing', rp).__file__).read_bytes()).hexdigest(),
        python={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}))


def implied_surface(payload, *, order=64):
    """Reprice the supplied fixed quote strikes under the frozen local grid."""
    g = payload['model']['local_variance_grid']
    b = payload['model']['reporting_iv_basis']
    t, x = np.asarray(g['time_nodes']), np.asarray(g['log_forward_moneyness_nodes'])
    v = np.asarray(g['values']).reshape(len(t), len(x))
    spot = payload['market']['spot']
    states = source_marginals(spot, t, x, v, order)
    strikes = spot * np.exp(b['log_forward_moneyness_nodes'])
    values = []
    for maturity in b['maturity_nodes']:
        indices = np.flatnonzero(t == maturity)
        require(len(indices) == 1 and indices[0] >= 2, 'portfolio_scope', 'quotes require at least two transitions')
        i = int(indices[0]); s, w, hats, _ = states[i-2]
        target = w @ _black(s[:, None], strikes[None, :], (hats @ v[i-1])[:, None]*(t[i]-t[i-1]))[0]
        lo = np.full(len(strikes), 1e-6); hi = np.full(len(strikes), 8.)
        require(np.all(target > np.maximum(spot-strikes, 0.)) and np.all(target < spot),
                'portfolio_quote', 'interior option prices required for IV inversion')
        for _ in range(64):
            mid = (lo+hi)/2
            below = _black(spot, strikes, maturity*mid*mid)[0] < target
            lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
        iv = (lo+hi)/2
        require(np.max(abs(_black(spot, strikes, maturity*iv*iv)[0]-target)) <= 1e-10,
                'portfolio_quote', 'IV inversion failed')
        values.append(iv)
    return np.array(values)


def frozen_spot_request(payload, spot):
    """Keep opening absolute strikes; local variance stays on log(S/S0)."""
    require(math.isfinite(spot) and spot > 0, 'portfolio_spot', 'positive finite spot required')
    p = deepcopy(payload)
    b = p['model']['reporting_iv_basis']
    b['log_forward_moneyness_nodes'] = (np.asarray(b['log_forward_moneyness_nodes'])
        + math.log(payload['market']['spot']) - math.log(spot)).tolist()
    p['market']['spot'] = spot
    p['risk'].pop('vega_kt', None)
    p['risk']['vega'] = False
    b['implied_volatilities'] = implied_surface(p).ravel().tolist()
    return p


def spot_iv_response(payload, relative_bump=1e-4):
    """d IV(K,T)/d S at frozen LV, with a paired half-bump convergence gate."""
    spot = payload['market']['spot']; derivatives = []
    require(math.isfinite(relative_bump) and 1e-6 <= relative_bump <= 1e-3,
            'portfolio_policy', 'spot IV response bump outside precision domain')
    for h in (spot*relative_bump, spot*relative_bump/2):
        pair = [np.array(frozen_spot_request(payload, spot+sign*h)['model']['reporting_iv_basis']['implied_volatilities'])
                for sign in (1, -1)]
        derivatives.append((pair[0]-pair[1])/(2*h))
    error = float(np.max(abs(derivatives[1]-derivatives[0])))
    require(error <= 1e-8, 'portfolio_delta_convention', 'frozen-grid IV spot response did not converge')
    return derivatives[1], dict(method='fixed_strike_frozen_LV_IV_spot_response_v1',
        relative_bumps=[relative_bump, relative_bump/2], refinement_error=error,
        response=derivatives[1].tolist())


def recalibrate_market_move(payload, spot, target_iv, *, extension='all_rows', variance_basis=None):
    """Independent spot and fixed-strike IV scenario, with all quotes refitted."""
    moved = frozen_spot_request(payload, spot)
    c = MultiStepExtensionCalibrator(moved, extension=extension, variance_basis=variance_basis)
    target = np.asarray(target_iv, dtype=float)
    require(target.shape == c.iv.shape and np.isfinite(target).all(), 'portfolio_shape', 'complete target IV surface required')
    return c.calibrate_all_maturities(target-c.iv)


def _shared_request(p):
    shared = deepcopy({key: p[key] for key in ('valuation_date', 'market', 'model', 'engine')})
    # RQMC scrambles are independent common replicates. Different point counts
    # and Brownian transforms are deterministic functions of the same scramble;
    # their joint means remain paired independent replicates for covariance.
    # MC path counts must still agree; no truncation or unpaired units allowed.
    if shared['engine']['type'] == 'randomized_quasi_monte_carlo':
        shared['engine'].pop('points_per_scramble', None)
        shared['engine']['variance_reduction'].pop('brownian_bridge', None)
    return shared


def sampling_covariance(samples):
    """Columns share independent-unit indices; antithetic mates are one unit."""
    a = np.asarray(samples, dtype=float)
    require(a.ndim == 2 and a.shape[0] >= 2 and a.shape[1] >= 1 and np.isfinite(a).all(),
            'portfolio_samples', 'finite paired sample matrix required')
    centered = a-a.mean(axis=0)
    with np.errstate(over='ignore', invalid='ignore'):
        covariance = centered.T @ centered/(len(a)-1)
    require(np.isfinite(covariance).all(), 'portfolio_samples', 'nonfinite joint covariance')
    return covariance


def evaluate_portfolio(requests, quantities, *, extension='all_rows', variance_basis=None,
                       subdivisions=512, relative_tolerance=1e-6,
                       gamma_relative_standard_error_limit=.25, worker_threads=1,
                       retain_samples=False, _native_cache=None):
    """Audit a common-market portfolio and contract-major joint quote covariance.

    Quantities are signed units (already multiplied by contract multipliers).
    Every nonempty instrument is audited, including zero-weight hedge candidates.
    An error never returns a partial accepted net risk.
    """
    import rust_pricing as rp
    require(isinstance(requests, dict) and 1 <= len(requests) <= 32
            and set(requests) == set(quantities), 'portfolio_shape', 'matching nonempty instrument/quantity maps required')
    ids = sorted(requests); base = requests[ids[0]]; shared = _shared_request(base)
    partitions = subdivisions if isinstance(subdivisions, dict) else {cid:subdivisions for cid in ids}
    require(set(partitions)==set(ids) and all(type(v) is int and 16<=v<=1024 for v in partitions.values()),
            'portfolio_policy', 'one valid integration subdivision count per instrument required')
    weights = np.array([quantities[cid] for cid in ids], dtype=float)
    require(np.isfinite(weights).all(), 'portfolio_quantity', 'finite signed units required')
    require(all(_shared_request(requests[cid]) == shared for cid in ids),
            'portfolio_stream_mismatch', 'valuation, complete market, model and independent randomization must match across instruments')
    c = MultiStepExtensionCalibrator(base, extension=extension, variance_basis=variance_basis)
    tangent, tangent_audit = c.all_maturity_iv_tangent(); q = c.iv.size
    engine=base['engine']
    count = engine['independent_sampling_units'] if engine['type']=='pseudo_monte_carlo' else engine['scramble_count']
    require(type(count) is int and count*len(ids)*(q+1) <= 4_000_000,
            'portfolio_budget', 'joint sample storage budget exceeded')
    audits, samples, risks = {}, [], {}
    for cid in ids:
        p = deepcopy(requests[cid]); p['risk'].pop('vega_kt', None); p['risk']['vega'] = False
        require(p['risk'].get('delta') is True and p['risk'].get('smile_dynamics') == {'type':'sticky_log_moneyness'},
                'portfolio_delta_convention', 'frozen-grid sticky-log-moneyness Delta required')
        # Private, in-memory reuse only. The complete request, tangent and all
        # execution/gating settings bind each entry; no saved audit is imported.
        cache_key = fingerprint(dict(request=requests[cid], tangent=tangent.tolist(),
            subdivisions=partitions[cid], relative_tolerance=relative_tolerance,
            gamma_relative_standard_error_limit=gamma_relative_standard_error_limit, workers=worker_threads))
        cached = None if _native_cache is None else _native_cache.get(cache_key)
        if cached is None:
            plan = rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(p, allow_nan=False)),
                                          worker_threads=worker_threads, reduction_block_size=256)
            a = plan.evaluate_vega_kt_multi_step_gamma_chain_audit(list(range(c.intervals)),
                gamma_boundaries(p, c.spot, put_knots=True),
                variance_risk_directions=tangent.T.reshape(q, c.intervals, len(c.nodes)).tolist(),
                subdivisions=partitions[cid], relative_tolerance=relative_tolerance,
                gamma_relative_standard_error_limit=gamma_relative_standard_error_limit, retain_samples=True)
        else:
            a = deepcopy(cached['audit'])
        if _native_cache is not None and cached is None and not (
                a['status'] == 'path_gamma_chain_passed' and a['estimate'] is not None
                and a['recovery_passed'] and a['validation_passed']):
            # Retain a rejected audit for diagnosis. Reuse reaches the same
            # mandatory gate below and cannot release a partial risk.
            _native_cache[cache_key] = deepcopy(dict(audit=a, risk=None))
        require(a['status'] == 'path_gamma_chain_passed' and a['estimate'] is not None
                and a['recovery_passed'] and a['validation_passed'],
                'portfolio_gamma_failure', cid+': '+a['status'])
        raw = np.asarray(a['sampling_unit_values'])
        require(raw.shape == (count, q+1) and a['time_nodes'] == c.times.tolist(),
                'portfolio_stream_mismatch', 'compiled counter grid or readout shape differs')
        covariance = sampling_covariance(raw[:, 1:])
        own = np.asarray(a['joint_sample_covariance']).reshape(3*q, 3*q)[q:2*q, q:2*q]
        means = np.array([m['mean'] for m in a['estimate']])
        require(np.allclose(raw[:, 1:].mean(axis=0), means, rtol=1e-11, atol=1e-11)
                and np.allclose(covariance, own, rtol=1e-10, atol=1e-9),
                'portfolio_sample_mismatch', 'sample readouts do not reproduce native moments')
        if cached is None:
            ordinary = plan.evaluate()
            item = dict(price=ordinary.value, frozen_grid_delta=ordinary.delta_raw, gamma=ordinary.gamma_raw,
                gamma_method=ordinary.diagnostics.gamma_method if ordinary.gamma_raw is not None else None,
                gamma_spot_bump=ordinary.diagnostics.gamma_spot_bump if ordinary.gamma_raw is not None else None,
                buckets=means.tolist(), request_hash=fingerprint(requests[cid]))
        else:
            item = deepcopy(cached['risk'])
        require(item['frozen_grid_delta'] is not None and math.isclose(item['price'], a['native_price']['mean'], rel_tol=1e-12, abs_tol=1e-12),
                'portfolio_price_mismatch', 'ordinary and Gamma prices differ')
        if _native_cache is not None and cached is None:
            _native_cache[cache_key] = deepcopy(dict(audit=a, risk=item))
        a.pop('sampling_unit_values')
        risks[cid] = item
        audits[cid] = a; samples.append(raw)
    cube = np.stack(samples, axis=1)  # independent unit, instrument, price/quote
    flat = cube[:, :, 1:].reshape(count, -1)
    joint = sampling_covariance(flat)
    net_samples = np.sum(cube * weights[None, :, None], axis=1)
    net_cov = sampling_covariance(net_samples)
    require(np.isfinite(net_samples).all(), 'portfolio_quantity', 'weighted samples overflowed')
    result = dict(document_kind='portfolio_market_risk', schema_version=1, status='passed',
        implementation_identity=implementation_identity(),
        acceptance_scope='request_specific_runtime_gates', full_market_iv_bucket_acceptance='not_established',
        instrument_order=ids, quantities={cid: float(quantities[cid]) for cid in ids},
        request_hashes={cid: fingerprint(requests[cid]) for cid in ids},
        common_market_hash=fingerprint(shared), stream_hash=fingerprint(dict(engine=shared['engine'], times=c.times.tolist(),
            valuation_date=base['valuation_date'], counter_range=[0, count], domain='Valuation')),
        instrument_engines={cid: deepcopy(requests[cid]['engine']) for cid in ids},
        sampling_unit=audits[ids[0]]['sampling_unit'], independent_sampling_units=count,
        covariance_layout='contract_then_market_quote', coordinate_order=tangent_audit['coordinate_order'],
        market_sample_covariance=joint.tolist(), market_mean_covariance=(joint/count).tolist(),
        price_and_market_sample_covariance=sampling_covariance(cube.reshape(count, -1)).tolist(),
        net_price=float(net_samples[:, 0].mean()), net_buckets=net_samples[:, 1:].mean(axis=0).tolist(),
        net_price_and_market_mean_covariance=(net_cov/count).tolist(),
        net_bucket_standard_errors=np.sqrt(np.maximum(0, np.diag(net_cov)[1:]/count)).tolist(),
        uncertainty_scope='MC_estimation_only_excludes_calibration_discretization_and_market_return_risk',
        instruments=risks, audits=audits, tangent_audit=tangent_audit,
        calibration_tangent=tangent.tolist(),
        extension=extension, variance_basis=None if variance_basis is None else np.asarray(variance_basis).tolist(),
        precision_policy=dict(subdivisions=deepcopy(subdivisions),relative_tolerance=relative_tolerance,
            gamma_relative_standard_error_limit=gamma_relative_standard_error_limit))
    if retain_samples:
        result['sampling_unit_values'] = cube.tolist()
    result['report_hash'] = fingerprint(result)
    return result


def check_report(report, requests):
    body = {k:v for k,v in report.items() if k != 'report_hash'}
    require(report.get('status') == 'passed' and report.get('report_hash') == fingerprint(body),
            'portfolio_report', 'complete unchanged opening risk report required')
    require(report.get('implementation_identity') == implementation_identity(),
            'portfolio_report', 'opening report belongs to a different implementation')
    require(report['request_hashes'] == {cid:fingerprint(p) for cid,p in requests.items()},
            'portfolio_report', 'opening risk requests differ')


def explain_pnl(report, requests, *, current_spot, current_iv, actual_asset_pnl,
                current_date=None, current_maturity_dates=None, current_log_moneyness=None,
                current_forward=None):
    """Opening quantities/risk, fixed absolute maturities and strikes, corrected Delta.

    With no current axes, current_iv is explicitly supplied at opening strikes
    and absolute maturities. With axes, interpolate in log-moneyness without
    extrapolation. Theta, curve, dividend, fixing, cross/high-order terms and
    numerical error stay in the measured asset-P&L residual.
    """
    check_report(report, requests)
    base = requests[report['instrument_order'][0]]
    b = base['model']['reporting_iv_basis']; spot = base['market']['spot']
    opening_date = date.fromisoformat(base['valuation_date'])
    maturities = [(opening_date+timedelta(days=round(t*365))).isoformat() for t in b['maturity_nodes']]
    at = current_date or base['valuation_date']
    require(date.fromisoformat(at) >= opening_date and all(m > at for m in maturities),
            'pnl_maturity', 'unexpired matched absolute maturities required')
    require(math.isfinite(current_spot) and current_spot > 0 and math.isfinite(actual_asset_pnl),
            'pnl_input', 'finite spot and actual asset P&L required')
    opening = np.asarray(b['implied_volatilities']).reshape(len(maturities), -1)
    supplied = np.asarray(current_iv, dtype=float)
    alignment = []
    if current_maturity_dates is None:
        require(current_log_moneyness is None and current_forward is None and supplied.shape == opening.shape,
                'pnl_surface', 'provide the entire surface explicitly at opening strikes and absolute maturities')
        aligned = supplied
    else:
        axis = np.asarray(current_log_moneyness, dtype=float)
        require(axis.ndim == 1 and len(axis) >= 2 and np.isfinite(axis).all() and np.all(np.diff(axis)>0)
                and len(set(current_maturity_dates)) == len(current_maturity_dates)
                and supplied.shape == (len(current_maturity_dates), len(axis)),
                'pnl_surface', 'ordered current strike axis and unique dated maturities required')
        forward = np.asarray(current_forward, dtype=float)
        require(forward.shape == (len(current_maturity_dates),) and np.isfinite(forward).all() and np.all(forward>0),
                'pnl_surface', 'one positive forward per maturity required')
        aligned = []
        for m in maturities:
            require(m in current_maturity_dates, 'pnl_maturity', 'missing absolute maturity '+m)
            i = current_maturity_dates.index(m)
            k = np.array(b['log_forward_moneyness_nodes'])+(math.log(spot)-math.log(forward[i]))
            require(np.all(k >= axis[0]) and np.all(k <= axis[-1]), 'pnl_strike', 'fixed strikes outside current surface; extrapolation refused')
            aligned.append(np.interp(k, axis, supplied[i]))
            alignment.append(dict(maturity=m, current_log_moneyness=k.tolist()))
        aligned = np.asarray(aligned)
    require(np.isfinite(aligned).all() and np.all(aligned>0), 'pnl_surface', 'finite positive current IV required')
    response, audit = spot_iv_response(base)
    ds = current_spot-spot; div = (aligned-opening).ravel()
    net = np.array(report['net_buckets'])
    frozen_delta = math.fsum(report['quantities'][cid]*report['instruments'][cid]['frozen_grid_delta']
                             for cid in report['instrument_order'])
    correction = float(net @ response)
    delta = frozen_delta-correction
    delta_pnl = delta*ds; vega_pnl = float(net @ div)
    # Joint covariance applies to the Vega/correction term; Delta sampling error
    # remains unquantified and must not be labelled total attribution uncertainty.
    effective = div-response*ds
    covariance = np.array(report['net_price_and_market_mean_covariance'])[1:, 1:]
    variance = float(effective @ covariance @ effective)
    require(variance >= -1e-10 and np.isfinite(variance), 'pnl_covariance', 'invalid projected covariance')
    explained = delta_pnl+vega_pnl
    require(all(math.isfinite(x) for x in (delta, delta_pnl, vega_pnl, explained)), 'pnl_input', 'P&L arithmetic overflowed')
    return dict(document_kind='vega_kt_pnl_explanation', schema_version=1, status='available',
        from_date=base['valuation_date'], to_date=at, opening_risk_hash=report['report_hash'],
        absolute_maturity_dates=maturities, fixed_strikes=(spot*np.exp(b['log_forward_moneyness_nodes'])).tolist(),
        alignment=alignment, spot_change=ds, iv_change=div.tolist(),
        frozen_grid_delta=frozen_delta, fixed_strike_market_delta=delta, delta_correction=correction,
        delta_convention='frozen_grid_delta_minus_VegaKT_dot_frozen_grid_IV_spot_response',
        delta_pnl=delta_pnl, vega_pnl=vega_pnl, gamma_pnl=None,
        available_greek_pnl=explained, actual_asset_pnl=actual_asset_pnl,
        unexplained_pnl=actual_asset_pnl-explained,
        vega_and_delta_correction_standard_error=math.sqrt(max(0.,variance)),
        uncertainty_scope='joint_VegaKT_estimation_only_excludes_frozen_Delta_error',
        residual_components=['Theta','Gamma','curves','dividends','fixings','cross_and_higher_order_terms','numerical_error'],
        spot_iv_response_audit=audit)


def reweight_portfolio(report, requests, quantities):
    """Reuse joint observations for another opening book, preserving covariance."""
    check_report(report, requests)
    ids = report['instrument_order']; q = len(report['coordinate_order'])
    require(set(quantities) == set(ids), 'portfolio_quantity', 'one signed unit count per instrument required')
    w = np.array([quantities[cid] for cid in ids], dtype=float)
    require(np.isfinite(w).all(), 'portfolio_quantity', 'finite signed unit counts required')
    result = deepcopy(report); result.pop('report_hash'); result.pop('sampling_unit_values', None)
    result['quantities'] = {cid:float(quantities[cid]) for cid in ids}
    result['net_price'] = math.fsum(quantities[cid]*report['instruments'][cid]['price'] for cid in ids)
    result['net_buckets'] = np.sum(np.array([report['instruments'][cid]['buckets'] for cid in ids])*w[:, None], axis=0).tolist()
    transform = np.kron(w[None, :], np.eye(q+1))
    covariance = transform@np.array(report['price_and_market_sample_covariance'])@transform.T/report['independent_sampling_units']
    require(np.isfinite(covariance).all() and np.isfinite(result['net_buckets']).all() and math.isfinite(result['net_price']),
            'portfolio_quantity', 'reweighted portfolio overflowed')
    result['net_price_and_market_mean_covariance'] = covariance.tolist()
    result['net_bucket_standard_errors'] = np.sqrt(np.maximum(0., np.diag(covariance)[1:])).tolist()
    result['report_hash'] = fingerprint(result)
    return result


def explain_backtest(result, *, extension='all_rows', variance_basis=None, worker_threads=1):
    """Produce a separate audited PNL-04 report from retained replay inputs.

    The original cash ledger and result schema are unchanged. Opening books are
    repriced under the explicitly selected extension and must match the replay's
    published Greeks. Unmatched, expired or unavailable IV remains unexplained,
    including for offsetting positions. Coverage never relies on a zero net Vega.
    """
    from .iv_alignment import _surface
    history = result.manifest['market_history']
    require(fingerprint(history) == result.manifest['full_history_hash'], 'pnl_history', 'retained market history changed')
    require(len(result.ledger) == len(result.cost_basis) == len(result.risks) == len(result.pnl_components),
            'pnl_history', 'complete committed ledger, inventory, risk and P&L rows required')
    rows = []; evidence = []; cumulative_explained = cumulative_residual = cumulative_cash = 0.
    for i, ledger in enumerate(result.ledger):
        pnl = result.pnl_components[i]
        asset_actual = math.fsum(pnl[k] for k in ('target', 'option_hedge', 'stock'))
        cash = pnl['total']-asset_actual
        row = dict(at=ledger['at'], status='no_interval' if i==0 else 'unavailable',
                   actual_asset_pnl=asset_actual, available_greek_pnl=0., unexplained_pnl=asset_actual,
                   non_asset_pnl=cash, groups={}, reason=None)
        if i:
            opening = result.risks[i-1]; inventory = result.cost_basis[i-1]['positions']
            require(opening['at'] == result.cost_basis[i-1]['at'] == history[i-1]['at'],
                    'pnl_timing', 'opening risk, inventory and history timestamps differ')
            units = {cid:0. for book in ('targets','hedges') for cid,p in inventory[book].items()
                     if p['quantity']*p['multiplier'] != 0.}
            requests = {cid:json.loads(history[i-1]['pricing_requests'][cid]) for cid in units}
            current, _ = _surface(result.risks[i])
            try:
                if units:
                    require(current is not None, 'pnl_surface', 'current accepted surface unavailable')
                    joint = evaluate_portfolio(requests, units, extension=extension, variance_basis=variance_basis,
                                               worker_threads=worker_threads)
                    # This also prevents substituting a different extension for
                    # the Vega that actually drove the saved hedge decisions.
                    for cid in units:
                        saved = opening['instruments'][cid]; new = joint['instruments'][cid]
                        require(np.allclose(saved['buckets'], new['buckets'], rtol=1e-11, atol=1e-11)
                                and math.isclose(saved['delta'], new['frozen_grid_delta'], rel_tol=1e-11, abs_tol=1e-11),
                                'pnl_risk_mismatch', 'selected extension differs from opening replay risk')
                    evidence.append(dict(at=opening['at'],joint_risk=joint))
                for book,label in (('targets','target'),('hedges','option_hedge')):
                    weights = {cid:inventory[book].get(cid,{}).get('quantity',0.)*inventory[book].get(cid,{}).get('multiplier',1.) for cid in units}
                    if not any(weights.values()):
                        row['groups'][label] = dict(status='no_opening_positions',available_greek_pnl=0.,unexplained_pnl=pnl[label])
                    else:
                        group = reweight_portfolio(joint, requests, weights)
                        basis = result.risks[i]['basis']
                        row['groups'][label] = explain_pnl(group, requests,
                            current_spot=((history[i]['spot']['bid']+history[i]['spot']['ask'])/2), current_date=history[i]['at'][:10],
                            current_iv=np.array(current['iv']).reshape(len(basis['maturity_dates']),-1),
                            current_maturity_dates=basis['maturity_dates'], current_log_moneyness=basis['log_moneyness'],
                            current_forward=current['forwards'], actual_asset_pnl=pnl[label])
                stock_delta = math.fsum(p['quantity']*p['multiplier'] for p in inventory['stock'].values())
                stock_pnl = stock_delta*(((history[i]['spot']['bid']+history[i]['spot']['ask'])/2)-((history[i-1]['spot']['bid']+history[i-1]['spot']['ask'])/2))
                row['groups']['stock'] = dict(status='available',available_greek_pnl=stock_pnl,unexplained_pnl=pnl['stock']-stock_pnl)
                row['available_greek_pnl'] = math.fsum(g['available_greek_pnl'] for g in row['groups'].values())
                row['unexplained_pnl'] = asset_actual-row['available_greek_pnl'];row['status']='available'
            except BacktestError as error:
                if error.code not in ('pnl_maturity','pnl_strike','pnl_surface'):
                    raise
                row['groups'] = {}; row['reason']=dict(code=error.code,message=str(error))
        cumulative_explained += row['available_greek_pnl']; cumulative_residual += row['unexplained_pnl']; cumulative_cash += cash
        row['total'] = row['available_greek_pnl']+row['unexplained_pnl']+cash
        row['reconciliation_error'] = row['total']-ledger['daily_pnl']
        row['cumulative_total'] = cumulative_explained+cumulative_residual+cumulative_cash
        row['cumulative_reconciliation_error'] = row['cumulative_total']-ledger['cumulative_pnl']
        require(abs(row['reconciliation_error']) <= pnl['tolerance'] and
                abs(row['cumulative_reconciliation_error']) <= pnl['cumulative_tolerance'],
                'pnl_reconciliation', 'market Greek explanation does not reconcile to saved ledger')
        rows.append(row)
    report = dict(document_kind='vega_kt_backtest_pnl',schema_version=1,source_result_hash=fingerprint(result),
        implementation_identity=implementation_identity(),rows=rows,opening_joint_risks=evidence,
        status='complete' if all(r['status'] in ('available','no_interval') for r in rows) else 'partial',
        covered_intervals=sum(r['status']=='available' for r in rows),intervals=max(0,len(rows)-1),
        method='opening_books_fixed_strike_IV_with_Delta_correction_v1',
        gamma='omitted_to_avoid_mixing_frozen_grid_Gamma_with_fixed_strike_market_Delta',
        residual='Theta_Gamma_curves_dividends_fixings_cross_higher_order_and_numerical_error',
        original_ledger_modified=False)
    report['report_hash']=fingerprint(report)
    return report
