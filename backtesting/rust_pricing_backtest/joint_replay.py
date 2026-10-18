"""Opt-in decision gates and opening-book IV attribution for daily replay.

The covariance is sampling error, not a market-risk objective. Individual risk
and provider gates remain mandatory. Every failure rolls back the snapshot.
"""
import json
import math

import numpy as np

from .domain import BacktestError, require
from .portfolio_market_risk import check_report, reweight_portfolio, explain_pnl
from .iv_alignment import _surface

POLICY = dict(method='joint_market_replay_v1', covariance='paired_sampling_error',
              gate='held_and_lot_rounded_proposed_original_iv_buckets',
              delta='existing_frozen_grid_decision_delta',
              pnl='saved_opening_risk_fixed_strike_IV_corrected_delta',
              unsupported='fail_snapshot', missing_pnl_alignment='explicit_unexplained_residual')


def requests_for(snapshot, ids):
    return {cid: json.loads(snapshot.pricing_requests[cid]) for cid in ids}


def capture(provider, snapshot, master, risks):
    if not risks:
        return None
    report = provider.joint_market_risk(snapshot, master, risks)
    requests = requests_for(snapshot, risks)
    check_report(report, requests)
    require(report['instrument_order'] == sorted(risks), 'joint_risk_mismatch', 'all evaluated contracts required')
    expected = [dict(maturity=(m-snapshot.at.date()).days/365., log_moneyness=k)
                for m in snapshot.basis.maturity_dates for k in snapshot.basis.log_moneyness]
    require(report['coordinate_order'] == expected, 'joint_risk_mismatch', 'joint coordinate order differs')
    for cid, risk in risks.items():
        item = report['instruments'][cid]
        require(risk.payment_scaling_factor == 1. and master[cid].payment_day == master[cid].expiry,
                'joint_provider_scope', 'same-day unscaled payment required')
        require(np.allclose(risk.buckets, item['buckets'], rtol=1e-11, atol=1e-11)
                and math.isclose(risk.delta, item['frozen_grid_delta'], rel_tol=1e-11, abs_tol=1e-11)
                and math.isclose(risk.price, item['price'], rel_tol=1e-11, abs_tol=1e-11),
                'joint_risk_mismatch', cid+': joint and decision risks differ')
    return report


def gate(report, snapshot, master, targets, hedges, limit):
    ids = [] if report is None else report['instrument_order']
    require(all(cid in ids for book in (targets, hedges) for cid, q in book.items() if q),
            'joint_risk_missing', 'every nonzero holding requires joint risk')
    quantities = {cid: (targets.get(cid, 0.)+hedges.get(cid, 0.))*master[cid].multiplier for cid in ids}
    if report is None:
        buckets, errors, covariance = [], [], []
    else:
        net = reweight_portfolio(report, requests_for(snapshot, ids), quantities)
        buckets, errors = net['net_buckets'], net['net_bucket_standard_errors']
        covariance = np.array(net['net_price_and_market_mean_covariance'])[1:, 1:].tolist()
    maximum = max(errors, default=0.)
    require(maximum <= limit, 'joint_risk_quality', f'net bucket standard error {maximum} exceeds {limit}')
    return dict(status='passed', opening_risk_hash=None if report is None else report['report_hash'],
                quantities=quantities, net_buckets=buckets, bucket_standard_errors=errors,
                market_mean_covariance=covariance, max_bucket_standard_error=maximum, limit=limit,
                uncertainty_scope='sampling_error_only_excludes_Delta_calibration_and_discretization')


def explain(snapshot, previous_snapshot, current, opening, inventory, pnl, ledger, previous):
    """Use only the committed opening joint report; never reprice past risk."""
    asset = math.fsum(pnl[k] for k in ('target', 'option_hedge', 'stock'))
    cash = pnl['total']-asset
    row = dict(at=ledger['at'], status='no_interval' if opening is None else 'unavailable',
               opening_risk_hash=None, actual_asset_pnl=asset, available_greek_pnl=0.,
               unexplained_pnl=asset, non_asset_pnl=cash, groups={}, reason=None)
    if opening is not None:
        require(opening['at'] == inventory['at'] == previous_snapshot.at.isoformat(),
                'pnl_timing', 'opening inventory, risk and snapshot timestamps differ')
        report = opening['joint_market']['report']
        ids = [] if report is None else report['instrument_order']
        requests = requests_for(previous_snapshot, ids)
        if report is not None:
            check_report(report, requests)
            row['opening_risk_hash'] = report['report_hash']
        positions = inventory['positions']
        require(all(cid in ids for book in ('targets', 'hedges') for cid, p in positions[book].items()
                    if p['quantity']*p['multiplier']), 'joint_risk_missing', 'opening risk lacks a held contract')
        surface, _ = _surface(current)
        try:
            for book, label in (('targets', 'target'), ('hedges', 'option_hedge')):
                weights = {cid: positions[book].get(cid, {}).get('quantity', 0.)
                           *positions[book].get(cid, {}).get('multiplier', 1.) for cid in ids}
                if not any(weights.values()):
                    row['groups'][label] = dict(status='no_opening_positions', available_greek_pnl=0., unexplained_pnl=pnl[label])
                    continue
                require(surface is not None, 'pnl_surface', 'current accepted surface unavailable')
                net = reweight_portfolio(report, requests, weights)
                row['groups'][label] = explain_pnl(net, requests, current_spot=snapshot.spot.mid,
                    current_date=snapshot.at.date().isoformat(), current_iv=np.array(surface['iv']).reshape(len(snapshot.basis.maturity_dates), -1),
                    current_maturity_dates=[m.isoformat() for m in snapshot.basis.maturity_dates],
                    current_log_moneyness=list(snapshot.basis.log_moneyness), current_forward=surface['forwards'],
                    actual_asset_pnl=pnl[label])
            stock = math.fsum(p['quantity']*p['multiplier'] for p in positions['stock'].values())
            stock_pnl = stock*(snapshot.spot.mid-previous_snapshot.spot.mid)
            row['groups']['stock'] = dict(status='available', available_greek_pnl=stock_pnl, unexplained_pnl=pnl['stock']-stock_pnl)
            row['available_greek_pnl'] = math.fsum(g['available_greek_pnl'] for g in row['groups'].values())
            row['unexplained_pnl'] = asset-row['available_greek_pnl']
            row['status'] = 'available'
        except BacktestError as error:
            if error.code not in ('pnl_maturity', 'pnl_strike', 'pnl_surface'):
                raise
            row['groups'] = {}; row['reason'] = dict(code=error.code, message=str(error))
    row['cumulative'] = {k: row[k]+(previous['cumulative'][k] if previous else 0.)
                         for k in ('available_greek_pnl', 'unexplained_pnl', 'non_asset_pnl')}
    row['total'] = row['available_greek_pnl']+row['unexplained_pnl']+cash
    row['reconciliation_error'] = row['total']-ledger['daily_pnl']
    row['cumulative_total'] = sum(row['cumulative'].values())
    row['cumulative_reconciliation_error'] = row['cumulative_total']-ledger['cumulative_pnl']
    require(abs(row['reconciliation_error']) <= pnl['tolerance']
            and abs(row['cumulative_reconciliation_error']) <= pnl['cumulative_tolerance'],
            'pnl_reconciliation', 'joint market explanation differs from ledger')
    return row


def verify_checkpoint(state, history):
    """Validate saved joint evidence before it can supply the next P&L row."""
    require(len(state['risks']) == len(state['greek_pnl']) == state['next_index'],
            'checkpoint_mismatch', 'joint risk/P&L coverage differs from committed prefix')
    for i, risk in enumerate(state['risks']):
        require(risk['at'] == history[i].at.isoformat() and 'joint_market' in risk
                and 'market_iv' in state['greek_pnl'][i], 'checkpoint_mismatch', 'missing committed joint evidence')
        report = risk['joint_market']['report']
        if report is not None:
            check_report(report, requests_for(history[i], report['instrument_order']))
    return True
