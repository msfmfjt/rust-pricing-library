"""Assemble independently reviewed domains; never derive expectations from the stub."""
from . import bass, core, dividends, heston, hull_white, multi_asset, rough, sources, vega_kt, volatility
from .module import CONTRACT


def merge_classes(*domains):
    merged = {}
    for domain in domains:
        duplicate = merged.keys() & domain.keys()
        if duplicate:
            raise RuntimeError(f'duplicate frozen class contracts: {sorted(duplicate)}')
        for name, contract in domain.items():
            if set(contract) - {'bases', 'members', 'attributes'}:
                raise RuntimeError(f'unknown frozen class contract field: {name}')
            for method, member in contract['members'].items():
                if (set(member) - {'kind', 'signature', 'return_name', 'return_annotation', 'declaration'}
                        or member['kind'] not in {'method', 'staticmethod', 'property'}):
                    raise RuntimeError(f'invalid frozen member contract: {name}.{method}')
        merged.update(domain)
    return merged


CLASSES = merge_classes(core.CLASSES, sources.CLASSES, bass.CLASSES, multi_asset.CLASSES,
                        dividends.CLASSES, heston.CLASSES, hull_white.CLASSES, rough.CLASSES,
                        volatility.CLASSES, vega_kt.CLASSES)
if set(CLASSES) - set(CONTRACT['exports']):
    raise RuntimeError('frozen classes must be declared in the module exports')
