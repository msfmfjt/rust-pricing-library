"""Inspect the installed extension, including every declared callable's signature."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
from .signatures import runtime_parameters


def verify_runtime_api(module, api: dict, version: str) -> int:
    failures = [name for name in api['symbols'] if not hasattr(module, name)]
    actual_symbols = {name for name in dir(module) if not name.startswith('_')}
    actual_symbols.discard('rust_pricing')
    if hasattr(module, '__version__'):
        actual_symbols.add('__version__')
    extra = actual_symbols - set(api['symbols'])
    if extra:
        failures.append('unexpected runtime symbols: ' + ', '.join(sorted(extra)))
    for class_name, members in api['class_members'].items():
        cls = getattr(module, class_name, None)
        if cls is None:
            continue
        failures.extend(f'{class_name}.{member}' for member in members if not hasattr(cls, member))
        declared = set(cls.__dict__)
        special = set(members) & {'__eq__', '__ne__', '__repr__'}
        failures.extend(f'missing runtime special member {class_name}.{member}' for member in sorted(special-declared))
        expected_members = {member for member in members if not member.startswith('__')}
        actual_members = {member for member in declared if not member.startswith('_')}
        extra = actual_members - expected_members
        if extra:
            failures.append(f'unexpected runtime members on {class_name}: ' + ', '.join(sorted(extra)))
    for class_name, member in api['static_methods']:
        cls = getattr(module, class_name, None)
        if cls is not None and not isinstance(cls.__dict__.get(member), staticmethod):
            failures.append(f'runtime member must be staticmethod: {class_name}.{member}')
    for class_name, member in api['properties']:
        cls = getattr(module, class_name, None)
        descriptor = cls.__dict__.get(member) if cls is not None else None
        if descriptor is not None and callable(descriptor):
            failures.append(f'runtime property member must not be callable: {class_name}.{member}')
    checked = 0
    for label, contract in api['callables'].items():
        kind = contract['kind']
        try:
            if kind == 'function':
                target = getattr(module, label)
            else:
                class_name, member = label.split('.')
                cls = getattr(module, class_name)
                if kind == 'method' and isinstance(cls.__dict__.get(member), (staticmethod, classmethod)):
                    raise ValueError('instance method has a static or class binding')
                # PyO3 exposes constructor signatures on the class, not __init__.
                target = cls if kind == 'constructor' else getattr(cls, member)
            actual = runtime_parameters(target, receiver=kind == 'method')
            if actual != contract['parameters']:
                failures.append(f'{label} signature mismatch: runtime={actual!r}, stub={contract["parameters"]!r}')
            checked += 1
        except (AttributeError, TypeError, ValueError) as error:
            failures.append(f'{label} signature unavailable or invalid: {error}')
    if not isinstance(getattr(module, '__version__', None), str) or module.__version__ != version:
        failures.append('__version__')
    actual_version = module.version() if callable(getattr(module, 'version', None)) else None
    if not isinstance(actual_version, str) or actual_version != version:
        failures.append('version()')
    for name, base in (('ValidationError', ValueError), ('PricingError', RuntimeError)):
        cls = getattr(module, name, None)
        if not isinstance(cls, type) or not issubclass(cls, base):
            failures.append(name + ' base')
    if failures:
        raise RuntimeError('wheel runtime API contract failed:\n' + '\n'.join(failures))
    return checked


def verify_runtime_symbols(python: Path, stub_api: dict[str, object], version: str) -> None:
    # Run with the target wheel's interpreter. Pin the checker import to this
    # source tree instead of depending on the caller's current import path.
    code = '''import json, sys
sys.path.insert(0, sys.argv[1])
from scripts.wheel_validation.runtime import verify_runtime_api
import rust_pricing
payload = json.load(sys.stdin)
count = verify_runtime_api(rust_pricing, payload['api'], payload['version'])
print(f'wheel-runtime verified callables={count}')
'''
    subprocess.run([str(python), '-c', code, str(Path(__file__).resolve().parents[2])],
                   input=json.dumps({'api': stub_api, 'version': version}), text=True, check=True)
