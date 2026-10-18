"""Exact discrete flat-IV quote recalibration and joint maturity-risk projection.

These utilities do not accept spatial smile bumps or promote diagnostic pricing
methods. Every maturity shock changes cumulative variance and rebuilds ALL
affected forward intervals; an isolated local-volatility bump is not substituted.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np
from .domain import BacktestError, fingerprint, require


def recalibrate_flat_market_iv(payload: dict, maturity_shifts) -> tuple[dict, dict]:
    """Return a copied request and audit for absolute IV shifts, one per maturity.

    Reporting maturities must equal the positive local-grid times. Rows must be
    flat and calibrated initially. Calendar variance, floor and cap failures are
    errors, never clipped or repaired. The last local-grid row is end padding.
    """
    try:
        model = payload['model']
        require(model['type'] == 'local_volatility', 'flat_iv_scope', 'local-vol grid required')
        grid, basis = model['local_variance_grid'], model['reporting_iv_basis']
        times = np.asarray(basis['maturity_nodes'], dtype=float)
        local_times = np.asarray(grid['time_nodes'], dtype=float)
        nx, nr = len(grid['log_forward_moneyness_nodes']), len(basis['log_forward_moneyness_nodes'])
        require(nx >= 2 and nr >= 2, 'flat_iv_shape', 'at least two local and reporting strike nodes required')
        require(times.ndim == 1 and len(times) >= 2 and np.isfinite(times).all()
                and times[0] > 0 and np.all(np.diff(times) > 0), 'flat_iv_shape', 'ordered positive maturities required')
        require(np.array_equal(local_times, np.r_[0., times]), 'flat_iv_scope', 'reporting maturities must equal positive local-grid times')
        quotes = np.asarray(basis['implied_volatilities'], dtype=float).reshape(len(times), nr)
        local = np.asarray(grid['values'], dtype=float).reshape(len(times)+1, nx)
        shift = np.asarray(maturity_shifts, dtype=float)
        require(shift.shape == times.shape and np.isfinite(shift).all(), 'flat_iv_shape', 'one finite absolute IV shift per maturity required')
        require(np.isfinite(quotes).all() and np.all(quotes > 0) and np.all(quotes == quotes[:, :1]),
                'flat_iv_scope', 'spatially flat positive reporting IV required')
        require(np.isfinite(local).all() and np.all(local == local[:, :1]), 'flat_iv_scope', 'spatially flat local variance required')
        dt = np.diff(local_times)
        current = np.diff(np.r_[0., times*quotes[:, 0]**2])/dt
        require(np.allclose(current, local[:-1, 0], rtol=1e-10, atol=0.), 'flat_iv_calibration', 'base IV does not match discrete cumulative variance')
        bumped = quotes[:, 0]+shift
        require(np.isfinite(bumped).all() and np.all(bumped > 0), 'flat_iv_quote', 'shifted IV must be positive and finite')
        cumulative = times*bumped**2
        interval = np.diff(np.r_[0., cumulative])
        variance = interval/dt
        floor, cap = float(grid['floor']), float(grid['cap'])
        require(math.isfinite(floor) and math.isfinite(cap) and 0 < floor <= cap,
                'flat_iv_bounds', 'finite positive variance bounds required')
        require(np.isfinite(variance).all() and np.all(variance >= floor) and np.all(variance <= cap),
                'flat_iv_forward_variance', 'shift violates positive calendar variance or local variance bounds')
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        if isinstance(error, BacktestError):
            raise
        raise BacktestError('flat_iv_shape', 'invalid grid or reporting quote shape') from error
    result = deepcopy(payload)
    result['model']['reporting_iv_basis']['implied_volatilities'] = np.repeat(bumped, nr).tolist()
    result['model']['local_variance_grid']['values'] = np.repeat(np.r_[variance, variance[-1]], nx).tolist()
    audit = dict(method='discrete_flat_market_iv_recalibration_v1', diagnostic_only=True,
                 source_request_hash=fingerprint(payload), recalibrated_request_hash=fingerprint(result),
                 maturity_nodes=times.tolist(), base_implied_volatilities=quotes[:, 0].tolist(),
                 absolute_iv_shifts=shift.tolist(), implied_volatilities=bumped.tolist(),
                 cumulative_variances=cumulative.tolist(), interval_variances=interval.tolist(),
                 interval_local_variances=variance.tolist(), clipping_applied=False)
    return result, audit


def maturity_risk_statistics(diagnostic: dict) -> dict:
    """Project every strike at each maturity, retaining cross-maturity covariance.

    Requires real full sample covariance. Does not infer independence from
    marginal standard errors. Output covariance is for the estimated means.
    """
    e = diagnostic.get('estimate')
    require(e is not None, 'flat_iv_recovery_failed', 'held-out Gamma recovery did not produce an estimate')
    require(e.get('full_bucket_sample_covariance') is not None,
            'flat_iv_covariance_required', 'full joint bucket covariance required')
    rows, width = len(diagnostic['reporting_maturities']), len(diagnostic['reporting_log_nodes'])
    count = e['independent_sampling_units']
    require(isinstance(count, int) and count >= 2, 'flat_iv_samples', 'at least two independent units required')
    try:
        means = np.asarray([m['mean'] for m in e['raw_buckets']], dtype=float)
        covariance = np.asarray(e['full_bucket_sample_covariance'], dtype=float).reshape(rows*width, rows*width)
    except (TypeError, ValueError, KeyError) as error:
        raise BacktestError('flat_iv_covariance', 'invalid joint statistics shape') from error
    require(means.shape == (rows*width,) and np.isfinite(means).all() and np.isfinite(covariance).all(),
            'flat_iv_covariance', 'finite joint statistics with matching shape required')
    tolerance = 1e-10*max(1., float(np.max(abs(covariance))))
    require(np.max(abs(covariance-covariance.T)) <= tolerance
            and np.linalg.eigvalsh(covariance).min() >= -tolerance,
            'flat_iv_covariance', 'symmetric positive-semidefinite sample covariance required')
    projection = np.kron(np.eye(rows), np.ones((1, width)))
    mean_covariance = projection@covariance@projection.T/count
    return dict(method='joint_maturity_iv_projection_v1', diagnostic_only=True,
                shock_definition='one_absolute_iv_unit_uniform_in_strike_at_each_maturity',
                maturity_nodes=diagnostic['reporting_maturities'],
                independent_sampling_units=count, means=(projection@means).tolist(),
                mean_covariance=mean_covariance.tolist(),
                standard_errors=np.sqrt(np.maximum(0., np.diag(mean_covariance))).tolist(),
                iv_bucket_acceptance='not_established')
