"""Two-date hard Barrier oracle: Gaussian integration, no pricing-library calls.

At eta=kappa=0 and flat residual variance, f and Y are correlated GBMs.
Condition on the two dividend normals. Truncated bivariate-lognormal moments
integrate out both equity normals, including the first-monitoring boundary.
"""

import math

import numpy as np


def cdf(x):
    x = np.asarray(x, dtype=float)
    return np.fromiter((0.5 * math.erfc(-v / math.sqrt(2)) for v in x.flat),
                       dtype=float, count=x.size).reshape(x.shape)


def bivariate_cdf(a, b, rho, order=64):
    """Integrate d Phi_2(a,b;r)/dr from independent normals (r=0).

    This quadrature is distinct from the outer dividend Gauss-Hermite rule.
    Infinite cutoffs use exact marginal limits before numerical integration.
    """
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    result = np.array(cdf(a) * cdf(b), copy=True)
    finite = np.isfinite(a) & np.isfinite(b)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    r = rho * (nodes + 1) / 2
    aa, bb = a[finite, None], b[finite, None]
    density = np.exp(-(aa * aa - 2 * r * aa * bb + bb * bb) / (2 * (1 - r * r)))
    density /= 2 * math.pi * np.sqrt(1 - r * r)
    result[finite] += (rho / 2) * (density @ weights)
    return result


def bivariate_partials(a, b, rho):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    da, db = np.zeros_like(a), np.zeros_like(b)
    fa, fb = np.isfinite(a), np.isfinite(b)
    root = math.sqrt(1 - rho * rho)
    da[fa] = np.exp(-a[fa] ** 2 / 2) / math.sqrt(2 * math.pi) * cdf((b[fa] - rho * a[fa]) / root)
    db[fb] = np.exp(-b[fb] ** 2 / 2) / math.sqrt(2 * math.pi) * cdf((a[fb] - rho * b[fb]) / root)
    return da, db


def price_delta(order=96, *, spot=100.0, barrier=105.0, strike=80.0,
                cash=(5.0, 3.0, 12.0), cdf_order=64):
    t1, t2, pay = 182 / 365, 364 / 365, 456 / 365
    dates = (t1, t2, 1.4)
    sigma, nu, rho = 0.2, 0.35, -0.25
    growth = 0.98 / 0.95
    funded = spot - sum(q / growth ** t for q, t in zip(cash, dates))
    if funded <= 0:
        raise ValueError("positive funded residual equity required")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    z = math.sqrt(2) * nodes
    z1, z2 = np.meshgrid(z, z, indexing="ij")
    wy1 = math.sqrt(t1) * z1.ravel()
    wy2 = wy1 + math.sqrt(t2 - t1) * z2.ravel()
    y1, y2 = np.exp(nu * wy1 - nu * nu * t1 / 2), np.exp(nu * wy2 - nu * nu * t2 / 2)
    a1, a2 = funded * growth ** t1, funded * growth ** t2
    b1 = sum(q * growth ** (t1 - t) for q, t in zip(cash, dates) if t > t1)
    b2 = cash[2] * growth ** (t2 - dates[2])
    c = b2 * y2 - strike
    first_upper = (barrier - (b1 + cash[0]) * y1) / a1
    upper = (barrier - (b2 + cash[1]) * y2) / a2
    lower = -c / a2
    m1, m2 = sigma * rho * wy1 - sigma * sigma * t1 / 2, sigma * rho * wy2 - sigma * sigma * t2 / 2
    s1, s2 = sigma * math.sqrt((1 - rho * rho) * t1), sigma * math.sqrt((1 - rho * rho) * t2)
    correlation = math.sqrt(t1 / t2)

    def cutoff(x, mean, deviation):
        out = np.full_like(x, -np.inf)
        positive = x > 0
        out[positive] = (np.log(x[positive]) - mean[positive]) / deviation
        return out

    h1, hu, hl = cutoff(first_upper, m1, s1), cutoff(upper, m2, s2), cutoff(lower, m2, s2)
    forward = a2 * np.exp(m2 + s2 * s2 / 2)
    vanilla = forward * cdf(s2 - hl) + c * cdf(-hl)
    vanilla_delta = forward / funded * cdf(s2 - hl)

    def rectangle(a, b):
        probability = bivariate_cdf(a, b, correlation, cdf_order)
        da, db = bivariate_partials(a, b, correlation)
        # Every finite cutoff has derivative -1/(funded * marginal s).
        return probability, -da / (funded * s1) - db / (funded * s2)

    p_u, dp_u = rectangle(h1, hu)
    p_l, dp_l = rectangle(h1, hl)
    m_u, dm_u = rectangle(h1 - correlation * s2, hu - s2)
    m_l, dm_l = rectangle(h1 - correlation * s2, hl - s2)
    valid = (first_upper > 0) & (upper > np.maximum(lower, 0.0))
    knockout = np.where(valid, forward * (m_u - m_l) + c * (p_u - p_l), 0.0)
    knockout_delta = np.where(valid, forward / funded * (m_u - m_l)
                              + forward * (dm_u - dm_l) + c * (dp_u - dp_l), 0.0)
    probability_weights = np.outer(weights, weights).ravel() / math.pi
    discount = 0.95 ** pay
    return (float(discount * np.dot(probability_weights, vanilla - knockout)),
            float(discount * np.dot(probability_weights, vanilla_delta - knockout_delta)))


def price_by_terminal_conditioning(order=32, inner_order=96, *, spot=100.0):
    """Independent price check: split first equity normal at its hard barrier.

    Last equity increment is integrated with univariate truncated lognormal
    moments; the first is integrated on [-10,10] with split Gauss-Legendre.
    No bivariate CDF or analytic Delta calculation is used here.
    """
    t1, t2, pay = 182 / 365, 364 / 365, 456 / 365
    sigma, nu, rho = 0.2, 0.35, -0.25
    growth = 0.98 / 0.95
    funded = spot - 5 / growth ** t1 - 3 / growth ** t2 - 12 / growth ** 1.4
    a1, a2 = funded * growth ** t1, funded * growth ** t2
    b1, b2 = 3 * growth ** (t1 - t2) + 12 * growth ** (t1 - 1.4), 12 * growth ** (t2 - 1.4)
    n, w = np.polynomial.hermite.hermgauss(order)
    z1, z2 = np.meshgrid(n * math.sqrt(2), n * math.sqrt(2), indexing="ij")
    wy1 = math.sqrt(t1) * z1.ravel()
    dwy = math.sqrt(t2 - t1) * z2.ravel()
    y1 = np.exp(nu * wy1 - nu * nu * t1 / 2)
    y2 = np.exp(nu * (wy1 + dwy) - nu * nu * t2 / 2)
    m1, s1 = sigma * rho * wy1 - sigma * sigma * t1 / 2, sigma * math.sqrt((1 - rho * rho) * t1)
    threshold = (105 - (b1 + 5) * y1) / a1
    cut = np.full_like(threshold, -10.0)
    positive = threshold > 0
    cut[positive] = np.clip((np.log(threshold[positive]) - m1[positive]) / s1, -10, 10)
    u, v = np.polynomial.legendre.leggauss(inner_order)
    root = sigma * math.sqrt((1 - rho * rho) * (t2 - t1))
    inner = np.zeros_like(cut)
    for hit, left, right in [(False, np.full_like(cut, -10), cut),
                              (True, cut, np.full_like(cut, 10))]:
        x = (left[:, None] + right[:, None]) / 2 + (right - left)[:, None] * u / 2
        mean_f = np.exp(m1[:, None] + s1 * x + sigma * rho * dwy[:, None]
                        - (sigma * rho) ** 2 * (t2 - t1) / 2)
        c = b2 * y2 - 80
        lower = np.maximum(-c / a2, 0)
        if not hit:
            lower = np.maximum(lower, (105 - (b2 + 3) * y2) / a2)
        d2 = np.full_like(mean_f, np.inf)
        positive = lower > 0
        d2[positive] = (np.log(mean_f[positive] / lower[positive, None]) - root * root / 2) / root
        payoff = a2 * mean_f * cdf(d2 + root) + c[:, None] * cdf(d2)
        density = np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
        inner += (right - left) / 2 * ((payoff * density) @ v)
    return float(0.95 ** pay * np.dot(np.outer(w, w).ravel() / math.pi, inner))


if __name__ == "__main__":
    for n in (32, 48, 64, 96, 128):
        print(n, price_delta(n))
    print("terminal conditioning", price_by_terminal_conditioning())
