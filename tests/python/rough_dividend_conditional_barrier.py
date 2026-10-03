"""Independent two-step rough-LSV hard Barrier integration on a frozen surface.

Condition on dividend increments and the first Volterra value, integrate the
first equity normal on split intervals, and integrate the terminal normal
analytically. No production calibration, evolution, payoff or risk calls.
"""
import math

import numpy as np

from rough_dividend_barrier_reference import cdf


def density(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def price_delta(case, market, order=20, inner_order=24, *, spot=None, tail=10.0,
                include_first_boundary=True):
    """Return price/analytic Spot Delta; reanchor the retained normalized surface.

    include_first_boundary=False is only a negative control: it deliberately
    drops that boundary's Delta contribution without changing the price.
    """
    spot = market['spot'] if spot is None else spot
    t1, t2 = market['fixing_time'], market['expiry_time']
    dt = t2 - t1
    h, eta, kappa = case['hurst'], case['eta'], case['kappa']
    alpha = case['equity_linkage']
    rho, nu = market['equity_dividend_correlation'], market['dividend_volatility']
    rho_sv = case['equity_volatility_correlation']
    rho_dv = case['dividend_volatility_correlation']
    growth = market['annual_carry'] / market['annual_discount']
    cash, dates = market['cash_means'], market['cash_times']
    funded = spot - sum(q / growth ** t for q, t in zip(cash, dates))
    if funded <= 0:
        raise ValueError('positive funded residual equity required')

    def coefficients(t):
        a, b, c = funded * growth ** t, 0.0, 0.0
        for q, ex in zip(cash, dates):
            if ex > t:
                reserve, decay = q * growth ** (t - ex), math.exp(-kappa * (ex - t))
                a += reserve * (1 - decay) * alpha
                b += reserve * decay
                c += reserve * (1 - decay) * (1 - alpha)
        return a, b, c

    a1, b1, c1 = coefficients(t1)
    a2, b2, c2 = coefficients(t2)
    nodes = np.asarray(case['log_nodes'])
    surface = np.asarray(case['squared_leverage']).reshape(3, len(nodes))
    sigma0 = math.sqrt(np.interp(0.0, nodes, surface[0]))
    # Exact covariance of the first hybrid Volterra cell with Brownian normals.
    cell = math.sqrt(2 * h) / (h + 0.5)
    hd = cell * rho_dv
    orth = (cell * rho_sv - rho * hd) / math.sqrt(1 - hd * hd)
    conditional_variance = 1 - rho * rho - orth * orth
    s1 = sigma0 * math.sqrt(t1 * conditional_variance)
    a_half1, a_half2 = math.exp(-kappa * t1 / 2), math.exp(-kappa * dt / 2)
    link1, link2 = (1 - a_half1) * alpha, (1 - a_half2) * alpha
    gh, weights = np.polynomial.hermite.hermgauss(order)
    mesh = np.meshgrid(*(3 * [math.sqrt(2) * gh]), indexing='ij')
    outer = np.stack([v.ravel() for v in mesh], axis=1)
    outer_weights = np.einsum('i,j,k->ijk', weights, weights, weights).ravel() / math.pi ** 1.5
    gl, gw = np.polynomial.legendre.leggauss(inner_order)
    total = np.zeros(2)
    for start in range(0, len(outer), 512):
        d1, v1, d2 = outer[start:start + 512].T
        m1 = sigma0 * math.sqrt(t1) * (rho * d1 + orth * v1) - sigma0 * sigma0 * t1 / 2
        volterra = t1 ** h * (hd * d1 + math.sqrt(1 - hd * hd) * v1)
        multiplier = np.exp(eta * volterra / 2 - eta * eta * t1 ** (2 * h) / 4)
        # Dividend state Y1=u1+link1*f1 after the symmetric split step.
        u1 = a_half1 * np.exp(nu * math.sqrt(t1) * d1 - nu * nu * t1 / 2) + (1 - a_half1) * (1 - alpha)
        pre_coefficient = a1 + (b1 + cash[0]) * link1
        first_upper = (market['barrier'] - (b1 + cash[0]) * u1 - c1) / pre_coefficient
        cut = np.full_like(first_upper, -np.inf)
        positive = first_upper > 0
        cut[positive] = (np.log(first_upper[positive]) - m1[positive]) / s1
        clipped = np.clip(cut, -tail, tail)

        def terminal(x, hit):
            f1 = np.exp(m1[:, None] + s1 * x)
            y1 = u1[:, None] + link1 * f1
            u2 = (a_half2 * (a_half2 * y1 + (1 - a_half2) * (alpha * f1 + 1 - alpha))
                  * np.exp(nu * math.sqrt(dt) * d2[:, None] - nu * nu * dt / 2)
                  + (1 - a_half2) * (1 - alpha))
            sigma = np.sqrt(np.interp(np.log(f1), nodes, surface[1])) * multiplier[:, None]
            s2 = sigma * math.sqrt(dt * (1 - rho * rho))
            mean = f1 * np.exp(sigma * math.sqrt(dt) * rho * d2[:, None] - sigma * sigma * dt * rho * rho / 2)
            q_post, q_pre = a2 + b2 * link2, a2 + (b2 + cash[1]) * link2
            constant = b2 * u2 + c2 - market['strike']
            exercise = -constant / q_post
            barrier = (market['barrier'] - (b2 + cash[1]) * u2 - c2) / q_pre
            use_barrier = (barrier > np.maximum(exercise, 0)) & ~np.asarray(hit)
            lower = np.maximum(exercise, 0)
            lower = np.where(use_barrier, barrier, lower)
            log_distance = np.full_like(mean, np.inf)
            positive = lower > 0
            log_distance[positive] = (np.log(mean[positive] / lower[positive]) - s2[positive] ** 2 / 2) / s2[positive]
            probability = cdf(log_distance)
            moment = mean * cdf(log_distance + s2)
            price = q_post * moment + constant * probability
            delta = growth ** t2 * moment
            # Intrinsic vanishes at the exercise boundary, but not at a barrier.
            delta += np.where(use_barrier, (q_post * lower + constant) * density(log_distance)
                              * growth ** t2 / (q_pre * s2), 0.0)
            return np.stack((price, delta), axis=-1)

        # Split at every leverage knot and at the first monitoring boundary.
        splits = [np.full_like(cut, -tail), clipped, np.full_like(cut, tail)]
        splits.extend(np.clip((node - m1) / s1, -tail, tail) for node in nodes)
        splits = np.sort(np.stack(splits, axis=1), axis=1)
        result = np.zeros((len(d1), 2))
        for left, right in zip(splits.T[:-1], splits.T[1:]):
            x = (left + right)[:, None] / 2 + (right - left)[:, None] * gl / 2
            value = terminal(x, x >= cut[:, None])
            result += (right - left)[:, None] / 2 * np.einsum('nji,nj,j->ni', value, density(x), gw)
        if include_first_boundary:
            at_cut = clipped[:, None]
            jump = terminal(at_cut, True)[:, 0, 0] - terminal(at_cut, False)[:, 0, 0]
            moving = (cut > -tail) & (cut < tail)
            result[:, 1] += np.where(moving, jump * density(clipped) * growth ** t1 / (pre_coefficient * s1), 0.0)
        total += outer_weights[start:start + len(d1)] @ result
    return tuple(total * market['annual_discount'] ** market['payment_time'])
