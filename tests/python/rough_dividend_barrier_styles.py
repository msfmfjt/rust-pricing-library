"""Retained independent hard Barrier Call/Put, Up/Down reference batches."""
import json
from pathlib import Path

import numpy as np

from rough_dividend_survival_reference import path_values

DIRECTORY = Path(__file__).resolve().parents[2] / 'fixtures/stochastic-dividends'


def inputs():
    cfg = json.loads((DIRECTORY / 'rough-barrier-styles-reference.json').read_text())
    source = json.loads((DIRECTORY / cfg['source_fixture']).read_text())
    market = json.loads((DIRECTORY / source['market_contract_fixture']).read_text())
    bases = {c['id']: c for c in source['cases']}
    return cfg, bases, market


def batch_means(case, base, market, sampling):
    rng = np.random.Generator(np.random.PCG64(sampling['seed']))
    pairs, steps = sampling['antithetic_pairs_per_batch'], len(base['times']) - 1
    contract = case['contract']
    market = dict(market, barrier=contract['barrier'], strike=contract['strike'])
    kwargs = {k: contract[k] for k in ('direction', 'side', 'style')}
    means = []
    for _ in range(sampling['batches']):
        z, u = rng.standard_normal((pairs, steps, 3)), rng.random((pairs, steps))
        plus = path_values(base, market, z, u, **kwargs)
        minus = path_values(base, market, -z, 1-u, **kwargs)
        means.append(((plus+minus)/2).mean(axis=0))
    return np.asarray(means)


def verify_fixture():
    cfg, bases, market = inputs()
    for case in cfg['cases']:
        means = batch_means(case, bases[case['base_case']], market, cfg['sampling'])
        np.testing.assert_allclose(means, case['batch_means'], rtol=0, atol=1e-9)
        errors = means.std(axis=0, ddof=1) / np.sqrt(len(means))
        for i, quantity in enumerate(('price', 'delta')):
            assert errors[i] < cfg['acceptance']['reference_' + quantity + '_se']
            print(json.dumps(dict(scope=cfg['scope'], case=case['id'], quantity=quantity,
                value=float(means[:, i].mean()), reference_se=float(errors[i]))), flush=True)


if __name__ == '__main__':
    verify_fixture()
