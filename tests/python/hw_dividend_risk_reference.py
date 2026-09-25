"""Independent one-split cash/HW oracle: Gaussian quadrature and tilted RK4.

No production model, price, covariance, payoff or risk helpers are imported.
The option target is a finite split, not a continuous-time option expectation.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import math

import numpy as np


@dataclass(frozen=True)
class Inputs:
    spot: float = 100.0
    sigma: float = 0.2
    kappa: float = 0.7
    alpha: float = 0.6
    nu: float = 0.35
    rho_fd: float = -0.25
    rho_fr: float = 0.25
    rho_dr: float = -0.2
    rate_a: float = 0.4
    rate_times: tuple = (0.0, 1.15)
    rate_vols: tuple = (0.04, 0.09)
    cash_times: tuple = (1.0, 1.4)
    cash_means: tuple = (3.0, 8.0)
    log_p: float = math.log(0.95)
    log_q: float = math.log(0.98)
    strike: float = 100.0
    expiry: float = 1.0
    payment: float = 456.0 / 365.0


def duration(a, t):
    return t if a == 0.0 else -math.expm1(-a * t) / a


def segments(p, start, end):
    for i, left in enumerate(p.rate_times):
        lo = max(start, left)
        hi = min(end, p.rate_times[i + 1] if i + 1 < len(p.rate_times) else end)
        if hi > lo:
            yield lo, hi, p.rate_vols[i]


@lru_cache(maxsize=1)
def legendre_rule():
    return np.polynomial.legendre.leggauss(32)


def covariance(p, start, end):
    """Covariance of (W_f,W_D,x innovation,integral-x innovation)."""
    rho = np.array([
        [1.0, p.rho_fd, p.rho_fr, p.rho_fr],
        [p.rho_fd, 1.0, p.rho_dr, p.rho_dr],
        [p.rho_fr, p.rho_dr, 1.0, 1.0],
        [p.rho_fr, p.rho_dr, 1.0, 1.0],
    ])
    nodes, weights = legendre_rule()
    result = np.zeros((4, 4))
    for lo, hi, eta in segments(p, start, end):
        tau = end - (lo + (nodes + 1.0) * (hi - lo) / 2.0)
        integral = tau if p.rate_a == 0.0 else -np.expm1(-p.rate_a * tau) / p.rate_a
        kernels = np.array([np.ones_like(tau), np.ones_like(tau),
                            eta * np.exp(-p.rate_a * tau), eta * integral])
        result += rho * ((kernels * (weights * (hi - lo) / 2.0)) @ kernels.T)
    return result


def bond(p, time, maturity, x):
    var_total = covariance(p, 0.0, maturity)[3, 3]
    var_past = covariance(p, 0.0, time)[3, 3]
    var_future = covariance(p, time, maturity)[3, 3]
    return np.exp((maturity - time) * p.log_p
                  - duration(p.rate_a, maturity - time) * x
                  - 0.5 * (var_total - var_past) + 0.5 * var_future)


@lru_cache(maxsize=1024)
def _cash_coefficients(time, maturity, a, times, vols, k, alpha, nu, sigma,
                       rho_fr, rho_dr, steps_per_year):
    """Forward evolution of the affine Y moment under a maturity Gaussian tilt."""
    state = (1.0, 0.0, 1.0, 0.0)  # m_f, A, B, C
    for i, left in enumerate(times):
        lo = max(time, left)
        hi = min(maturity, times[i + 1] if i + 1 < len(times) else maturity)
        if hi <= lo:
            continue
        n = max(1, math.ceil((hi - lo) * steps_per_year))
        h = (hi - lo) / n
        eta = vols[i]

        def rhs(s, v):
            loading = eta * duration(a, maturity - s)
            decay = k + nu * rho_dr * loading
            return (-sigma * rho_fr * loading * v[0],
                    k * alpha * v[0] - decay * v[1],
                    -decay * v[2], k * (1.0 - alpha) - decay * v[3])

        for step in range(n):
            s = lo + step * h
            k1 = rhs(s, state)
            k2 = rhs(s + h / 2.0, tuple(x + h / 2.0 * y for x, y in zip(state, k1)))
            k3 = rhs(s + h / 2.0, tuple(x + h / 2.0 * y for x, y in zip(state, k2)))
            k4 = rhs(s + h, tuple(x + h * y for x, y in zip(state, k3)))
            state = tuple(x + h / 6.0 * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
                          for x, v1, v2, v3, v4 in zip(state, k1, k2, k3, k4))
    return state[1:]


def cash_coefficients(p, time, maturity, steps_per_year):
    return _cash_coefficients(time, maturity, p.rate_a, p.rate_times, p.rate_vols,
                              p.kappa, p.alpha, p.nu, p.sigma, p.rho_fr, p.rho_dr,
                              steps_per_year)


@lru_cache(maxsize=4)
def gaussian_rule(order):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    normals = np.stack(np.meshgrid(*([nodes * math.sqrt(2.0)] * 3), indexing='ij'),
                       axis=-1).reshape(-1, 3)
    masses = np.prod(np.stack(np.meshgrid(*([weights / math.sqrt(math.pi)] * 3),
                                         indexing='ij'), axis=-1), axis=-1).ravel()
    return normals, masses


def cdf(x):
    return np.fromiter((0.5 * math.erfc(-v / math.sqrt(2.0)) for v in x),
                       dtype=float, count=len(x))


@lru_cache(maxsize=1024)
def price(p=Inputs(), order=40, steps_per_year=256):
    if any(t < p.expiry for t in p.cash_times):
        raise ValueError('the independent one-step reference requires cash at or after expiry')
    if any(0.0 < t < p.expiry for t in p.rate_times):
        raise ValueError('the native one-step panel requires no rate knot before expiry')
    if p.payment < p.expiry:
        raise ValueError('payment must follow fixing')
    reserve = sum(mean * math.exp((p.log_p - p.log_q) * t)
                  * sum(cash_coefficients(p, 0.0, t, steps_per_year))
                  for t, mean in zip(p.cash_times, p.cash_means))
    risky = p.spot - reserve
    if risky <= 0.0:
        raise ValueError('unfunded initial residual equity')
    cov = covariance(p, 0.0, p.expiry)
    regression = np.linalg.solve(cov[1:, 1:], cov[1:, 0])
    conditional_variance = p.sigma**2 * (p.expiry - cov[0, 1:] @ regression)
    if conditional_variance <= 0.0:
        raise ValueError('reference requires positive conditional equity variance')
    root = math.sqrt(conditional_variance)
    normals, masses = gaussian_rule(order)
    conditioned = normals @ np.linalg.cholesky(cov[1:, 1:]).T
    wd, x, ix = conditioned.T
    mean_f = np.exp(-0.5 * p.sigma**2 * p.expiry
                    + p.sigma * (conditioned @ regression) + 0.5 * conditional_variance)
    q = math.exp(-0.5 * p.kappa * p.expiry)
    # Starting f=Y=1 makes the first half drift exactly one. The last half
    # leaves Y_T=q*exp(nu W_D-nu²T/2)+(1-q)*(alpha*f_T+1-alpha).
    coefficient_f = risky * np.exp((p.log_q - p.log_p) * p.expiry + ix + 0.5 * cov[3, 3])
    remainder = np.zeros_like(coefficient_f)
    for t, mean in zip(p.cash_times, p.cash_means):
        if t == p.expiry:
            continue  # post-cash observation
        af, by, ac = cash_coefficients(p, p.expiry, t, steps_per_year)
        scale = mean * math.exp((p.expiry - t) * p.log_q) * bond(p, p.expiry, t, x)
        coefficient_f += scale * (af + by * (1.0 - q) * p.alpha)
        remainder += scale * (by * q * np.exp(-0.5 * p.nu**2 * p.expiry + p.nu * wd)
                              + by * (1.0 - q) * (1.0 - p.alpha) + ac)
    strike = p.strike - remainder
    forward = coefficient_f * mean_f
    positive = strike > 0.0
    call = forward - strike  # a negative effective strike is always in the money
    d1 = np.log(forward[positive] / strike[positive]) / root + 0.5 * root
    call[positive] = forward[positive] * cdf(d1) - strike[positive] * cdf(d1 - root)
    discount = np.exp(p.expiry * p.log_p - ix - 0.5 * cov[3, 3])
    discount *= bond(p, p.expiry, p.payment, x)
    return float(masses @ (discount * call))


# Native result label, independent-input field, optional element, first width.
PARTIALS = (
    ('spot', 'spot', None, 1e-3),
    ('initial_volatility', 'sigma', None, 1e-5),
    ('dividend_mean_reversion', 'kappa', None, 1e-5),
    ('equity_linkage', 'alpha', None, 1e-5),
    ('dividend_volatility', 'nu', None, 1e-5),
    ('cash_mean[1]', 'cash_means', 0, 1e-3),
    ('cash_mean[2]', 'cash_means', 1, 1e-3),
    ('discount_log_df[1]', 'log_p', None, 1e-5),
    ('repo_spread_log_df[1]', 'log_q', None, 1e-5),
    ('rate_mean_reversion', 'rate_a', None, 1e-5),
    ('rate_volatility[0]', 'rate_vols', 0, 1e-6),
    ('rate_volatility[1]', 'rate_vols', 1, 1e-6),
    ('equity_dividend_correlation', 'rho_fd', None, 1e-5),
    ('equity_rate_correlation', 'rho_fr', None, 1e-5),
    ('dividend_rate_correlation', 'rho_dr', None, 1e-5),
)


def shifted(p, field, index, bump):
    value = getattr(p, field)
    if index is None:
        value += bump
    else:
        value = tuple(x + (bump if i == index else 0.0) for i, x in enumerate(value))
    return replace(p, **{field: value})


def partial(p, specification, order=40, width_scale=0.5, steps_per_year=256):
    _, field, index, width = specification
    h = width * width_scale
    up = price(shifted(p, field, index, h), order, steps_per_year)
    if field == 'rate_a' and p.rate_a == 0.0:
        return (-3.0 * price(p, order, steps_per_year) + 4.0 * up
                - price(shifted(p, field, index, 2.0 * h), order, steps_per_year)) / (2.0 * h)
    return (up - price(shifted(p, field, index, -h), order, steps_per_year)) / (2.0 * h)
