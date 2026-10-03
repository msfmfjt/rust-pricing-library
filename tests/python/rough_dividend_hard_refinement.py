"""Paired hard-Barrier refinement with one fixed leverage surface and schedule.

The raw Gaussian inputs share Brownian increments and newest Volterra-cell
integrals across each coarse/fine pair. Survival transports then act separately
on each grid; conditioned paths themselves are not a common Brownian path.
"""
from functools import lru_cache
import math

import numpy as np

from rough_dividend_barrier_reference import cdf
from rough_dividend_survival_reference import path_values


@lru_cache(None)
def cross_kernel(h, offset, order=256):
    if offset == 0:
        return 1.0
    x, w = np.polynomial.legendre.leggauss(order)
    u = (x + 1) / 2
    # x=u^(1/H) regularizes the singular kernel product; unlike the Rust
    # reference's Simpson rule this is Gaussian quadrature with another map.
    return float(w @ (u ** (0.5 / h) * (offset + u ** (1 / h)) ** (h - 0.5)))


@lru_cache(None)
def residual_coefficients(h, block, rho_dv):
    """Coarse residual in fine (dividend, volatility, near) normals plus extra."""
    if block < 1 or block & (block - 1) or not 0 < h <= 0.5:
        raise ValueError('H in (0,1/2] and a positive dyadic ratio required')
    coefficients = np.zeros((block, 3))
    if block == 1:
        coefficients[0, 2] = 1.
        return coefficients, 0.
    if h == 0.5:
        return coefficients, 1.
    p = h + 0.5
    a = math.sqrt(2 * h) / p
    r = (0.5 - h) / p
    coarse_a, coarse_r = a * block ** (h - 0.5), r * block ** h
    loadings = np.array([rho_dv, math.sqrt(1 - rho_dv * rho_dv)])
    for j in range(block):
        distance = block - j
        dw_cov = a * (distance ** p - (distance - 1) ** p)
        coefficients[j, :2] = (dw_cov - coarse_a) * loadings / coarse_r
        coefficients[j, 2] = (cross_kernel(h, distance - 1) - a * dw_cov) / (r * coarse_r)
    explained = float(np.sum(coefficients * coefficients))
    if not 0 <= explained < 1:
        raise ValueError('invalid conditional newest-cell variance')
    return coefficients, math.sqrt(1 - explained)


def coarsen(fine, extra, h, block, rho_dv):
    """Fine/coarse layout is (dividend, orthogonal volatility, near, equity)."""
    n, steps, factors = fine.shape
    if block < 1 or factors != 4 or steps % block or extra.shape != (n, steps // block):
        raise ValueError('incompatible Gaussian shapes')
    cells = fine.reshape(n, steps // block, block, 4)
    result = cells.sum(axis=2) / math.sqrt(block)
    coefficients, independent = residual_coefficients(h, block, rho_dv)
    result[:, :, 2] = np.einsum('nijq,jq->ni', cells[:, :, :, :3], coefficients) + independent * extra
    return result


def refine_case(base, steps):
    """Repeat left-constant leverage rows, retaining every original knot exactly."""
    original = np.asarray(base['times'])
    count = len(original) - 1
    if steps < count or steps % count:
        raise ValueError('grid must subdivide all retained leverage intervals')
    ratio = steps // count
    times = [original[0]]
    for left, right in zip(original[:-1], original[1:]):
        times.extend(right if j == ratio else left + (right - left) * j / ratio
                     for j in range(1, ratio + 1))
    surface = np.asarray(base['squared_leverage']).reshape(len(original), -1)
    rows = np.searchsorted(original, times, side='right') - 1
    return dict(base, times=times, squared_leverage=surface[rows].ravel().tolist(),
                monitoring_indices=[i * ratio for i in base['monitoring_indices']])


def coupled_batch_means(base, market, *, levels=(16, 32, 64, 128), seed=193,
                        batches=32, pairs=1024):
    """Return (batch, level, price/Delta) means of antithetic units."""
    if sorted(set(levels)) != list(levels) or len(levels) < 2:
        raise ValueError('at least two increasing distinct levels required')
    dt = np.diff(base['times'])
    if not np.allclose(dt, dt[0], rtol=1e-12, atol=0):
        raise ValueError('common-Brownian coupling requires a uniform base grid')
    cases = [refine_case(base, level) for level in levels]
    fine_steps = levels[-1]
    rng = np.random.Generator(np.random.PCG64(seed))
    means = []
    for _ in range(batches):
        fine = rng.standard_normal((pairs, fine_steps, 4))
        values = []
        for level, case in zip(levels, cases):
            z = fine if level == fine_steps else coarsen(
                fine, rng.standard_normal((pairs, level)), base['hurst'],
                fine_steps // level, base['dividend_volatility_correlation'])
            u = cdf(z[:, :, 3])
            plus = path_values(case, market, z[:, :, :3], u)
            minus = path_values(case, market, -z[:, :, :3], 1 - u)
            values.append(((plus + minus) / 2).mean(axis=0))
        means.append(values)
    return np.asarray(means)


def comparisons(means, levels):
    """Each row compares one level with the same final finite-grid reference."""
    mean = means.mean(axis=0)
    for i, level in enumerate(levels[:-1]):
        diff = means[:, i] - means[:, -1]
        gap, se = diff.mean(axis=0), diff.std(axis=0, ddof=1) / math.sqrt(len(means))
        unpaired = np.sqrt(means[:, i].var(axis=0, ddof=1) + means[:, -1].var(axis=0, ddof=1)) / math.sqrt(len(means))
        for j, quantity in enumerate(('price', 'delta')):
            yield dict(steps=level, fine_steps=levels[-1], quantity=quantity,
                       value=float(mean[i, j]), fine_value=float(mean[-1, j]),
                       gap=float(gap[j]), paired_se=float(se[j]), unpaired_se=float(unpaired[j]),
                       abs_gap_plus_4se=float(abs(gap[j]) + 4 * se[j]))


def deterministic_normals(steps, pattern):
    i, factor = np.indices((steps, 4))
    return ((37 * (i + 1) * (factor + 3) + 17 * (i + 1) ** 2 + 13 * pattern) % 101 - 50) / 25.


def primal_checkpoints(case, market, z):
    """Independent unconditioned full paths for cross-language state checks."""
    from rough_dividend_survival_reference import hybrid_weights
    times = np.asarray(case['times'])
    dt = np.diff(times)
    surface = np.asarray(case['squared_leverage']).reshape(len(times), -1)
    weights, residual, variance = hybrid_weights(times, case['hurst'])
    sd = market['equity_dividend_correlation']
    dv, sv = case['dividend_volatility_correlation'], case['equity_volatility_correlation']
    orth = (sv - sd * dv) / math.sqrt(1 - dv * dv)
    equity = sd * z[:, 0] + orth * z[:, 1] + math.sqrt(1 - sd * sd - orth * orth) * z[:, 3]
    increments = np.sqrt(dt) * (dv * z[:, 0] + math.sqrt(1 - dv * dv) * z[:, 1])
    drivers = weights @ increments
    drivers[1:] += residual * z[:, 2]
    f = y = 1.
    result = []
    for i in range(len(dt)):
        sigma = math.sqrt(float(np.interp(math.log(f), case['log_nodes'], surface[i])))
        sigma *= math.exp(case['eta'] * drivers[i] / 2 - case['eta'] ** 2 * variance[i] / 4)
        a = math.exp(-case['kappa'] * dt[i] / 2)
        half = a * y + (1 - a) * (case['equity_linkage'] * f + 1 - case['equity_linkage'])
        f *= math.exp(-sigma * sigma * dt[i] / 2 + sigma * math.sqrt(dt[i]) * equity[i])
        nu = market['dividend_volatility']
        y = a * half * math.exp(nu * math.sqrt(dt[i]) * z[i, 0] - nu * nu * dt[i] / 2)
        y += (1 - a) * (case['equity_linkage'] * f + 1 - case['equity_linkage'])
        if i + 1 in case['monitoring_indices']:
            result.append([float(times[i+1]), f, y])
    return result


def run_panel():
    import json
    from pathlib import Path
    directory = Path(__file__).resolve().parents[2] / 'fixtures/stochastic-dividends'
    cfg = json.loads((directory / 'rough-hard-barrier-refinement.json').read_text())
    source = json.loads((directory / cfg['source_fixture']).read_text())
    market = json.loads((directory / source['market_contract_fixture']).read_text())
    bases = {case['id']: case for case in source['cases']}
    failures = []
    for case_id in cfg['case_ids']:
        for seed in cfg['valuation_seeds']:
            means = coupled_batch_means(bases[case_id], market, levels=tuple(cfg['levels']), seed=seed,
                batches=cfg['sampling']['batches'], pairs=cfg['sampling']['antithetic_pairs_per_batch'])
            for row in comparisons(means, cfg['levels']):
                print(json.dumps(dict(scope=cfg['scope'], case=case_id, seed=seed, **row, **cfg['sampling'])), flush=True)
                if row['steps'] == cfg['acceptance']['comparison_steps']:
                    q = row['quantity']
                    if not (row['abs_gap_plus_4se'] < cfg['acceptance'][q + '_bound']
                            and row['paired_se'] < cfg['acceptance'][q + '_se']):
                        failures.append(f'{case_id} seed={seed} {q}: {row}')
    if failures:
        raise AssertionError('\n'.join(failures))


if __name__ == '__main__':
    run_panel()
