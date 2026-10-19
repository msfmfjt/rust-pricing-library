"""Finite log-Euler quote calibration with a fixed first Gaussian transition.

This diagnostic calibrator prices terminal vanilla calls by integrating the
first state and pricing the second transition analytically. It fits quoted
strikes only; interpolation between quotes and the underdetermined local
variance extension are model choices, not market observations. No exotic
payoff, recovered Gamma, or Monte Carlo sample enters calibration.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np
from .domain import BacktestError, fingerprint, require


def _cdf(x):
    x = np.asarray(x, dtype=float)
    return np.fromiter((.5*math.erfc(-v/math.sqrt(2)) for v in x.flat), float).reshape(x.shape)


def _black(spot, strike, variance):
    sd = np.sqrt(variance)
    d1 = (np.log(spot/strike)+variance/2)/sd
    price = spot*_cdf(d1)-strike*_cdf(d1-sd)
    # Derivative with respect to integrated variance, not annual volatility.
    derivative = spot*np.exp(-d1*d1/2)/(2*math.sqrt(2*math.pi)*sd)
    return price, derivative


class TwoStepSmileCalibrator:
    """Two intervals, zero carry, unit discount, first marginal held fixed.

    Newton increments use the minimum Euclidean norm in annual local-variance
    node coordinates. Rank loss, inaccurate quadrature, inadmissible variance,
    or nonconvergence fail explicitly. This is not a general Dupire calibrator.
    The request must subsequently be priced on the SAME two-interval time grid.
    """

    def __init__(self, payload, *, quadrature_order=32, price_tolerance=1e-10,
                 maximum_iterations=20, singular_value_cutoff=1e-10):
        require(isinstance(quadrature_order, int) and 16 <= quadrature_order <= 128,
                'discrete_smile_policy', 'quadrature order must be an integer in [16,128]')
        require(isinstance(maximum_iterations, int) and 1 <= maximum_iterations <= 100,
                'discrete_smile_policy', 'iteration limit must be an integer in [1,100]')
        require(math.isfinite(price_tolerance) and 1e-12 <= price_tolerance <= 1e-6
                and math.isfinite(singular_value_cutoff) and 1e-14 <= singular_value_cutoff <= 1e-3,
                'discrete_smile_policy', 'invalid numerical tolerance or rank cutoff')
        self.payload = deepcopy(payload)
        self.order, self.tolerance = quadrature_order, price_tolerance
        self.iterations, self.cutoff = maximum_iterations, singular_value_cutoff
        try:
            model, market = payload['model'], payload['market']
            require(model['type'] == 'local_volatility' and market['type'] == 'equity'
                    and not market.get('discrete_dividends'), 'discrete_smile_scope',
                    'equity local volatility without discrete dividends required')
            for name in ('discount_curve', 'dividend_curve'):
                curve = market[name]
                ct, df = np.array(curve['times']), np.array(curve['discount_factors'])
                require(ct.ndim == 1 and len(ct) >= 2 and ct.shape == df.shape
                        and np.isfinite(ct).all() and np.all(np.diff(ct) > 0) and np.all(df == 1.),
                        'discrete_smile_scope', 'unit discount and zero carry required')
            g, b = model['local_variance_grid'], model['reporting_iv_basis']
            self.times = np.asarray(g['time_nodes'], dtype=float)
            self.nodes = np.asarray(g['log_forward_moneyness_nodes'], dtype=float)
            self.quote_nodes = np.asarray(b['log_forward_moneyness_nodes'], dtype=float)
            require(self.times.shape == (3,) and self.times[0] == 0.
                    and np.isfinite(self.times).all() and np.all(np.diff(self.times) > 0)
                    and np.array_equal(b['maturity_nodes'], self.times[1:]),
                    'discrete_smile_scope', 'exactly two intervals and matching reporting maturities required')
            for nodes in (self.nodes, self.quote_nodes):
                require(nodes.ndim == 1 and 2 <= len(nodes) <= 129 and np.isfinite(nodes).all()
                        and np.all(np.diff(nodes) > 0), 'discrete_smile_shape', 'ordered finite strike nodes required')
            require(self.nodes[0] <= 0 <= self.nodes[-1]
                    and self.quote_nodes[0] >= self.nodes[0] and self.quote_nodes[-1] <= self.nodes[-1],
                    'discrete_smile_scope', 'local grid must contain spot and all quote strikes')
            self.variance = np.asarray(g['values'], dtype=float).reshape(3, len(self.nodes))
            self.iv = np.asarray(b['implied_volatilities'], dtype=float).reshape(2, len(self.quote_nodes))
            require(g.get('shape', [3, len(self.nodes)]) == [3, len(self.nodes)]
                    and b.get('shape', [2, len(self.quote_nodes)]) == [2, len(self.quote_nodes)],
                    'discrete_smile_shape', 'declared grid shape mismatch')
            self.floor, self.cap, self.spot = float(g['floor']), float(g['cap']), float(market['spot'])
            require(0 < self.floor < self.cap and math.isfinite(self.cap) and 0 < self.spot < math.inf
                    and np.isfinite(self.variance).all() and np.all(self.variance >= self.floor)
                    and np.all(self.variance <= self.cap) and np.isfinite(self.iv).all() and np.all(self.iv > 0),
                    'discrete_smile_bounds', 'finite positive quotes, spot and bounded variance required')
            require(np.all(self.variance[0] == self.variance[0, 0])
                    and np.all(self.iv[0] == self.iv[0, 0])
                    and abs(self.iv[0, 0]**2-self.variance[0, 0]) <= 1e-12,
                    'discrete_smile_first_marginal', 'first marginal must be flat and calibrated; first-maturity smile is unreachable')
            self.strikes = self.spot*np.exp(self.quote_nodes)
            require(np.isfinite(self.strikes).all() and np.all(self.strikes > 0),
                    'discrete_smile_bounds', 'finite positive quote strikes required')
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
            if isinstance(error, BacktestError):
                raise
            raise BacktestError('discrete_smile_shape', 'invalid market or grid shape') from error
        self.dt = self.times[2]-self.times[1]
        self.v0 = self.times[1]*self.variance[0, 0]
        require(math.isfinite(self.v0) and self.v0 > 0, 'discrete_smile_bounds', 'finite positive first integrated variance required')
        sd = math.sqrt(self.v0)
        self.tail_bound = self.spot*float(_cdf(-12-sd)+_cdf(-12+sd))
        require(self.tail_bound < self.tolerance/10, 'discrete_smile_quadrature', 'first-state tail bound exceeds tolerance')
        self.integration = self._integration(self.order)
        self.verification_integration = self._integration(2*self.order)
        base_prices, _ = self._prices_jacobian(self.variance[1], self.verification_integration)
        targets, _ = _black(self.spot, self.strikes, self.times[2]*self.iv[1]**2)
        require(np.max(abs(base_prices-targets))+self.tail_bound <= self.tolerance,
                'discrete_smile_base_calibration', 'base terminal quotes do not match discrete prices')

    def _integration(self, order):
        sd = math.sqrt(self.v0)
        lo, hi = -self.v0/2-12*sd, -self.v0/2+12*sd
        cuts = np.r_[lo, self.nodes[(self.nodes > lo) & (self.nodes < hi)], hi]
        z, w = np.polynomial.legendre.leggauss(order)
        x = ((cuts[:-1]+cuts[1:])[:, None]/2+np.diff(cuts)[:, None]/2*z).ravel()
        weights = (np.diff(cuts)[:, None]/2*w).ravel()*np.exp(-.5*((x+self.v0/2)/sd)**2)/(sd*math.sqrt(2*math.pi))
        hats = np.array([np.interp(x, self.nodes, unit) for unit in np.eye(len(self.nodes))]).T
        return self.spot*np.exp(x), weights, hats

    def _prices_jacobian(self, variances, integration):
        spots, weights, hats = integration
        calls, vegas = _black(spots[:, None], self.strikes[None, :], (hats@variances)[:, None]*self.dt)
        return weights@calls, (weights[:, None]*vegas).T@hats*self.dt

    def _inverse(self, jacobian, scales):
        u, s, vh = np.linalg.svd(jacobian/scales[:, None], full_matrices=False)
        rank = int(np.count_nonzero(s > s[0]*self.cutoff))
        require(rank == len(self.strikes), 'discrete_smile_rank', 'terminal quote response is rank deficient at the declared cutoff')
        return (vh.T/s)@u.T, s

    def terminal_iv_tangent(self):
        """d(local variance nodes)/d(terminal quote IV), first interval frozen."""
        _, j = self._prices_jacobian(self.variance[1], self.integration)
        _, d = _black(self.spot, self.strikes, self.times[2]*self.iv[1]**2)
        vega = d*2*self.times[2]*self.iv[1]
        require(np.isfinite(vega).all() and np.all(vega > 1e-12),
                'discrete_smile_quote', 'quote Vega is too small for stable calibration')
        inverse, singular = self._inverse(j, vega)
        return inverse, dict(singular_values=singular.tolist(), rank=len(singular),
                             iv_response_residual=float(np.max(abs(j@inverse/vega[:, None]-np.eye(len(vega))))))

    def calibrate(self, terminal_iv_shifts):
        """Return a new request and audit; never clip or return a failed fit."""
        try:
            shifts = np.asarray(terminal_iv_shifts, dtype=float)
        except (TypeError, ValueError) as error:
            raise BacktestError('discrete_smile_quote', 'invalid terminal IV shift shape') from error
        require(shifts.shape == self.iv[1].shape and np.isfinite(shifts).all(),
                'discrete_smile_quote', 'one finite absolute IV shift per terminal quote required')
        iv = self.iv[1]+shifts
        require(np.all(iv > 0), 'discrete_smile_quote', 'shifted IV must be positive')
        target, derivative = _black(self.spot, self.strikes, self.times[2]*iv**2)
        scales = derivative*2*self.times[2]*iv
        require(np.isfinite(target).all() and np.isfinite(scales).all() and np.all(scales > 1e-12),
                'discrete_smile_quote', 'quote Vega is too small for stable calibration')
        # Each conditional call price is at least intrinsic at the first state.
        first_calls, _ = _black(self.spot, self.strikes, np.full(len(iv), self.v0))
        slopes = np.diff(target)/np.diff(self.strikes)
        require(np.all(target >= first_calls) and np.all(target < self.spot)
                and np.all(slopes >= -1.) and np.all(slopes <= 0.) and np.all(np.diff(slopes) >= 0.),
                'discrete_smile_arbitrage', 'quotes violate call convexity, bounds or the fixed first marginal')
        local = self.variance[1].copy()
        history = []
        for iteration in range(self.iterations+1):
            prices, jacobian = self._prices_jacobian(local, self.integration)
            residual = prices-target
            inverse, singular = self._inverse(jacobian, scales)
            history.append(dict(iteration=iteration, maximum_price_error=float(np.max(abs(residual))),
                                singular_values=singular.tolist()))
            if np.max(abs(residual)) <= self.tolerance/2:
                break
            require(iteration < self.iterations, 'discrete_smile_nonconvergence', 'quote calibration iteration limit reached')
            step = -inverse@(residual/scales)
            accepted = False
            for backtrack in range(30):
                alpha = 2.**(-backtrack)
                trial = local+alpha*step
                if np.all(trial >= self.floor) and np.all(trial <= self.cap):
                    trial_prices, _ = self._prices_jacobian(trial, self.integration)
                    if np.linalg.norm((trial_prices-target)/scales) < np.linalg.norm(residual/scales):
                        local, accepted = trial, True
                        history[-1]['step_fraction'] = alpha
                        break
            require(accepted, 'discrete_smile_nonconvergence', 'no admissible residual-reducing calibration step')
        verified, _ = self._prices_jacobian(local, self.verification_integration)
        error = float(np.max(abs(verified-target)))+self.tail_bound
        require(error <= self.tolerance, 'discrete_smile_quadrature', 'refined quadrature did not confirm price tolerance')
        result = deepcopy(self.payload)
        result['model']['local_variance_grid']['values'] = np.vstack([self.variance[0], local, local]).ravel().tolist()
        result['model']['reporting_iv_basis']['implied_volatilities'] = np.r_[self.iv[0], iv].tolist()
        audit = dict(method='two_step_discrete_smile_quote_calibration_v1', diagnostic_only=True,
                     full_market_iv_bucket_acceptance='not_established', first_marginal='fixed_lognormal',
                     extension='minimum_euclidean_norm_annual_local_variance_newton_increments',
                     source_request_hash=fingerprint(self.payload), recalibrated_request_hash=fingerprint(result),
                     time_nodes=self.times.tolist(), quote_log_nodes=self.quote_nodes.tolist(),
                     terminal_iv_shifts=shifts.tolist(), target_prices=target.tolist(), verified_prices=verified.tolist(),
                     maximum_verified_price_error_bound=error, omitted_tail_price_bound=self.tail_bound,
                     quadrature_order=self.order, verification_quadrature_order=2*self.order,
                     price_tolerance=self.tolerance, singular_value_cutoff=self.cutoff,
                     maximum_iterations=self.iterations,
                     iteration_history=history, clipping_applied=False)
        return result, audit

    def joint_iv_tangent(self):
        """Annual-variance tangent for first parallel IV and terminal quote IVs.

        Row zero is the uniform first annual variance; remaining rows are second
        variance nodes. First-maturity individual strike bumps are unreachable.
        The first column offsets the first marginal's effect on terminal calls.
        """
        inverse, terminal_audit = self.terminal_iv_tangent()
        _, derivative = _black(self.spot, self.strikes, self.times[2]*self.iv[1]**2)
        vegas = derivative*2*self.times[2]*self.iv[1]
        sd = math.sqrt(self.v0)
        first_scale = 2*self.iv[0, 0]

        def first_response(integration):
            spots, weights, hats = integration
            z = (np.log(spots/self.spot)+self.v0/2)/sd
            calls, _ = _black(spots[:, None], self.strikes[None, :], (hats@self.variance[1])[:, None]*self.dt)
            score = self.times[1]*((z*z-1)/(2*self.v0)-z/(2*sd))
            return (weights*score)@calls

        coarse = first_response(self.integration)
        fine = first_response(self.verification_integration)
        tail_probability = float(_cdf(-12-sd)+_cdf(-12+sd))
        tail_second = (sd*sd+2)*tail_probability+(12+sd)*(math.exp(-.5*(12-sd)**2)+math.exp(-.5*(12+sd)**2))/math.sqrt(2*math.pi)
        tail_bound = self.times[1]*(1/(2*self.v0)+1/(4*sd))*self.spot*tail_second
        refinement = float(np.max((abs(coarse-fine)+tail_bound)*first_scale/vegas))
        require(np.isfinite(fine).all() and math.isfinite(refinement) and refinement <= 1e-9,
                'discrete_smile_tangent_quadrature', 'first-marginal quote response did not converge')
        weights = -first_scale*fine/vegas
        tangent = np.zeros((len(self.nodes)+1, len(self.strikes)+1))
        tangent[0, 0] = first_scale
        tangent[1:, 0] = inverse@weights
        tangent[1:, 1:] = inverse
        _, j = self._prices_jacobian(self.variance[1], self.verification_integration)
        response = (fine[:, None]*tangent[0]+j@tangent[1:])/vegas[:, None]
        target = np.column_stack([np.zeros(len(vegas)), np.eye(len(vegas))])
        residual = float(np.max(abs(response-target)))
        require(np.isfinite(tangent).all() and residual <= 1e-9,
                'discrete_smile_tangent_quadrature', 'joint quote tangent failed refined verification')
        return tangent, dict(method='two_step_joint_market_iv_tangent_v1',
            coordinate_order=['first_maturity_parallel_iv']+[f'terminal_quote_{i}' for i in range(len(vegas))],
            variance_coordinate_order='uniform_first_annual_variance_then_second_variance_nodes',
            first_iv_to_annual_variance=first_scale,
            terminal_call_first_variance_derivatives=fine.tolist(),
            first_recalibration_terminal_basis_weights=weights.tolist(),
            maximum_normalized_quadrature_difference_bound=refinement,
            first_variance_derivative_tail_bound=tail_bound,
            maximum_joint_iv_response_residual=residual,
            terminal_tangent_audit=terminal_audit, full_market_iv_bucket_acceptance='not_established')

    def calibrate_joint(self, first_iv_shift, terminal_iv_shifts):
        """Move the flat first IV, then fit terminal quotes by second variance.

        Both shifts are absolute IV units. A new working context prevents any
        mutation of this calibrator or its fixed-first methods and audit shape.
        """
        from copy import copy
        try:
            require(np.ndim(first_iv_shift) == 0, 'discrete_smile_quote', 'one finite parallel first IV shift required')
            shift = float(first_iv_shift)
        except (TypeError, ValueError) as error:
            raise BacktestError('discrete_smile_quote', 'invalid first IV shift') from error
        require(math.isfinite(shift), 'discrete_smile_quote', 'finite first IV shift required')
        first_iv = float(self.iv[0, 0])+shift
        require(math.isfinite(first_iv) and first_iv > 0 and self.floor <= first_iv*first_iv <= self.cap,
                'discrete_smile_bounds', 'shifted first variance is outside model bounds')
        working = copy(self)
        working.variance = self.variance.copy(); working.variance[0] = first_iv*first_iv
        working.iv = self.iv.copy(); working.iv[0] = first_iv
        working.v0 = self.times[1]*first_iv*first_iv
        sd = math.sqrt(working.v0)
        working.tail_bound = self.spot*float(_cdf(-12-sd)+_cdf(-12+sd))
        require(math.isfinite(working.v0) and working.tail_bound < self.tolerance/10,
                'discrete_smile_quadrature', 'shifted first-state tail exceeds tolerance')
        working.integration = working._integration(self.order)
        working.verification_integration = working._integration(2*self.order)
        result, audit = working.calibrate(terminal_iv_shifts)
        audit.update(method='two_step_joint_market_iv_calibration_v1',
            first_marginal='shifted_flat_lognormal', first_parallel_iv_shift=shift,
            first_target_iv=first_iv, first_annual_variance=first_iv*first_iv,
            first_maturity_individual_quote_shifts='unsupported')
        return result, audit
