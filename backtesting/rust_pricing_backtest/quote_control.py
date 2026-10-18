"""Fixed unit market-quote control, with a dated virtual terminal ATM call.

Only sampling error is reduced. Native Gamma/AD gates and the calibrated
quote identity are checked before releasing any adjusted market-IV risk.
"""
from copy import deepcopy
from datetime import date
import math

import numpy as np

from .domain import Risk, fingerprint, require
from .discrete_smile import _black
from .multi_step_extension import MultiStepExtensionCalibrator
from .portfolio_market_risk import check_report, evaluate_portfolio, sampling_covariance

POLICY = 'rqmc_quote_control_512x24'
CONTROL_ID = '__market_quote_control__'


def virtual_request(payload):
    """Build the control from today's available quote, never the instrument book."""
    p = deepcopy(payload)
    product, market = p['product'], p['market']
    basis = p['model']['reporting_iv_basis']
    maturity = (date.fromisoformat(product['expiry'])-date.fromisoformat(p['valuation_date'])).days/365.
    require(product['type'] == 'european_vanilla' and product['side'] == {'type': 'call'}
            and product['notional'] == 1. and maturity == basis['maturity_nodes'][-1]
            and 0. in basis['log_forward_moneyness_nodes']
            and not market.get('discrete_dividends')
            and all(market[name]['discount_factors'] and all(v == 1. for v in market[name]['discount_factors'])
                    for name in ('discount_curve', 'dividend_curve')),
            'quote_control_scope', 'terminal unit European calls, an ATM quote and zero rates/dividends required')
    product['strike'] = market['spot']
    return p


def adjust_report(native, requests):
    """Transform paired samples and remove the virtual contract from the book."""
    check_report(native, requests)
    require(CONTROL_ID in requests and native['extension'] == 'all_rows' and native['variance_basis'] is None,
            'quote_control_scope', 'explicit virtual control and all_rows calibration required')
    base = requests[CONTROL_ID]
    require(all(virtual_request(p) == base for p in requests.values()),
            'quote_control_scope', 'same dated market, engine, expiry and control request required')
    require(all(a['status'] == 'path_gamma_chain_passed' and a['recovery_passed'] and a['validation_passed']
                for a in native['audits'].values()), 'quote_control_native', 'every raw native gate remains mandatory')
    c = MultiStepExtensionCalibrator(base, extension='all_rows')
    tangent, ta = c.all_maturity_iv_tangent()
    require(np.array_equal(tangent, np.array(native['calibration_tangent'])) and ta['iv_response_residual'] <= 1e-9,
            'quote_control_calibration', 'same checked calibration tangent required')
    matches = np.flatnonzero(c.strikes == base['product']['strike'])
    require(len(matches) == 1, 'quote_control_scope', 'control must equal an exact calibrated quote strike')
    k = int(matches[0]); coordinate = c.iv.size-len(c.strikes)+k
    t, iv = c.maturities[-1], c.iv[-1, k]
    known = np.zeros(c.iv.size)
    known[coordinate] = float(_black(c.spot, c.strikes[k], t*iv*iv)[1]*2*t*iv)
    ids = native['instrument_order']; index = ids.index(CONTROL_ID)
    kept = [i for i, cid in enumerate(ids) if cid != CONTROL_ID]
    raw = np.asarray(native['sampling_unit_values'])
    count, n, width = raw.shape; q = width-1
    require(kept and (n, q, count) == (len(ids), c.iv.size, native['independent_sampling_units'])
            and count >= 2 and np.isfinite(raw).all(), 'quote_control_samples', 'complete paired samples required')
    adjusted = raw[:, kept, :].copy()
    adjusted[:, :, 1:] -= raw[:, index:index+1, 1:]
    adjusted[:, :, 1:] += known[None, None, :]
    transform = np.zeros((len(kept)*width, n*width))
    for j, i in enumerate(kept):
        for k in range(width):
            transform[j*width+k, i*width+k] = 1.
            if k: transform[j*width+k, index*width+k] -= 1.
    covariance = sampling_covariance(adjusted.reshape(count, -1))
    projection = transform @ np.array(native['price_and_market_sample_covariance']) @ transform.T
    require(np.allclose(covariance, projection, rtol=1e-9, atol=1e-8),
            'quote_control_covariance', 'paired-sample covariance must match the full linear transformation')
    result = deepcopy(native); result.pop('report_hash')
    result['instrument_order'] = [ids[i] for i in kept]
    for field in ('quantities', 'request_hashes', 'instrument_engines', 'instruments', 'audits'):
        result[field].pop(CONTROL_ID)
    if isinstance(result['precision_policy']['subdivisions'], dict):
        result['precision_policy']['subdivisions'].pop(CONTROL_ID)
    result['market_risk_estimator'] = 'fixed_market_quote_control_v1'
    result['control_variate'] = dict(policy=POLICY, coefficient=1., coefficient_fitted_from_samples=False,
        virtual=True, tradable=False, request=deepcopy(base), request_hash=fingerprint(base),
        raw_native_audit=deepcopy(native['audits'][CONTROL_ID]), expected_gradient=known.tolist(), coordinate=coordinate,
        source_native_report_hash=native['report_hash'], iv_response_residual=ta['iv_response_residual'],
        covariance_projection_error=float(np.max(abs(covariance-projection))),
        expectation_scope='declared calibration quote identity; calibration/discretization bias excluded from sampling SE',
        native_audits='unchanged raw Gamma/AD audits; controlled estimates stored separately')
    result['sampling_unit_values'] = adjusted.tolist()
    for i, cid in enumerate(result['instrument_order']):
        result['instruments'][cid]['buckets'] = adjusted[:, i, 1:].mean(axis=0).tolist()
        result['instruments'][cid]['bucket_standard_errors'] = (adjusted[:, i, 1:].std(axis=0, ddof=1)/math.sqrt(count)).tolist()
    joint = sampling_covariance(adjusted[:, :, 1:].reshape(count, -1))
    result['market_sample_covariance'] = joint.tolist()
    result['market_mean_covariance'] = (joint/count).tolist()
    result['price_and_market_sample_covariance'] = covariance.tolist()
    weights = np.array([result['quantities'][cid] for cid in result['instrument_order']])
    net = np.einsum('niq,i->nq', adjusted, weights); net_cov = sampling_covariance(net)/count
    result['net_price'] = float(net[:, 0].mean()); result['net_buckets'] = net[:, 1:].mean(axis=0).tolist()
    result['net_price_and_market_mean_covariance'] = net_cov.tolist()
    result['net_bucket_standard_errors'] = np.sqrt(np.maximum(0, np.diag(net_cov)[1:])).tolist()
    require(np.array_equal(adjusted[:, :, 0], raw[:, kept, 0]), 'quote_control_price', 'price samples must stay unchanged')
    result['report_hash'] = fingerprint(result)
    return result


def portfolio(provider, snapshot, requests):
    require(CONTROL_ID not in requests and 1 <= len(requests) <= 31,
            'quote_control_scope', 'one virtual slot and at most 31 actual contracts required')
    # This cache is bounded to a single snapshot and is never serialized or
    # populated from saved evidence. A resumed provider recomputes native risk.
    snapshot_hash = fingerprint(snapshot)
    if provider._quote_control_snapshot != snapshot_hash:
        provider._quote_control_cache.clear()
        provider._quote_control_snapshot = snapshot_hash
    augmented = deepcopy(requests)
    augmented[CONTROL_ID] = virtual_request(requests[min(requests)])
    require(all(virtual_request(p) == augmented[CONTROL_ID] for p in requests.values()),
            'quote_control_scope', 'instruments must share the same exact control')
    native = evaluate_portfolio(augmented, {cid: 0. for cid in augmented}, extension='all_rows',
        worker_threads=provider.worker_threads, retain_samples=True, _native_cache=provider._quote_control_cache)
    return adjust_report(native, augmented)


def evaluate(provider, payload, snapshot, contract):
    from .multi_step_provider import validation
    cid = contract.contract_id
    report = portfolio(provider, snapshot, {cid: payload})
    item = report['instruments'][cid]; raw = report['audits'][cid]
    expected = [dict(maturity=(d-snapshot.basis.valuation_date).days/365., log_moneyness=k)
                for d in snapshot.basis.maturity_dates for k in snapshot.basis.log_moneyness]
    require(report['coordinate_order'] == expected, 'basis_mismatch', 'control risk coordinates differ')
    provenance = dict(provider=provider.identity, snapshot=fingerprint(snapshot), contract=fingerprint(contract),
        original_request=fingerprint(payload), controlled_report=report['report_hash'])
    accepted = validation(provider.numerical_scope, provider.variance_extension, None, POLICY)
    buckets = tuple(item['buckets'])
    risk = Risk(item['price'], item['frozen_grid_delta'], buckets, snapshot.basis.basis_id,
        pre_projection=math.fsum(buckets), price_standard_error=raw['native_price']['standard_error'],
        provenance=fingerprint(provenance), bucket_standard_errors=tuple(item['bucket_standard_errors']),
        native_policy=accepted['native_policy'], payment_date=contract.payment_day,
        gamma=item['gamma'], gamma_method=item['gamma_method'], gamma_spot_bump=item['gamma_spot_bump'],
        reporting_iv=tuple(payload['model']['reporting_iv_basis']['implied_volatilities']),
        reporting_forwards=(snapshot.spot.mid,)*len(snapshot.basis.maturity_dates),
        warning_codes=('fixed_market_quote_control', 'sampling_error_excludes_calibration_discretization'))
    return risk, dict(document_kind='multi_step_provider_audit', schema_version=1, status='passed',
        provenance=provenance, risk_provenance=risk.provenance, validation=accepted,
        controlled_market_report=report, covariance_scope='paired_same_snapshot_sampling_units')
