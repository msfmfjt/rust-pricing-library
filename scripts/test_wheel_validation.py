"""Exercise caller-visible signature drift without loading a native extension."""
import ast
import copy
import io
from pathlib import Path
import sys
import tarfile
from types import ModuleType
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.wheel_validation.contracts import merge_classes
from scripts.wheel_validation.runtime import verify_runtime_api
from scripts.wheel_validation.signatures import runtime_parameters, stub_parameters
from scripts.wheel_validation.stub import exported_stub_api, verify_stub_static_shape
from scripts import check_source_archive as archive


def function(source):
    namespace = {}
    exec(source, namespace)
    return namespace['f']


def parameters(source, receiver=False):
    return stub_parameters(ast.parse(source).body[0], receiver=receiver)


class SignatureTests(unittest.TestCase):
    def test_parameter_order_kinds_required_and_typed_defaults(self):
        source = 'def f(a, /, b=None, *, c, d=True, e=1, f=1.0, g="x"): pass'
        expected = [
            {'name': 'a', 'kind': 'positional_only'},
            {'name': 'b', 'kind': 'positional_or_keyword', 'default': {'type': 'NoneType', 'value': None}},
            {'name': 'c', 'kind': 'keyword_only'},
            {'name': 'd', 'kind': 'keyword_only', 'default': {'type': 'bool', 'value': True}},
            {'name': 'e', 'kind': 'keyword_only', 'default': {'type': 'int', 'value': 1}},
            {'name': 'f', 'kind': 'keyword_only', 'default': {'type': 'float', 'value': 1.0}},
            {'name': 'g', 'kind': 'keyword_only', 'default': {'type': 'str', 'value': 'x'}},
        ]
        self.assertEqual(parameters(source), expected)
        self.assertEqual(runtime_parameters(function(source), receiver=False), expected)

    def test_only_instance_receiver_binding_is_normalized(self):
        self.assertEqual(parameters('def f(self, x): pass', receiver=True),
                         runtime_parameters(function('def f(self, /, x): pass'), receiver=True))
        self.assertNotEqual(parameters('def f(self, x): pass', receiver=True),
                            runtime_parameters(function('def f(self, x, /): pass'), receiver=True))
        for source in ('def f(x): pass', 'def f(*, self): pass', 'def f(self=None): pass'):
            with self.subTest(source=source):
                with self.assertRaises(RuntimeError):
                    parameters(source, receiver=True)
                with self.assertRaises(ValueError):
                    runtime_parameters(function(source), receiver=True)

    def test_opaque_variadic_and_uninspectable_signatures_fail(self):
        for source in ('def f(*args): pass', 'def f(**kwargs): pass',
                       'def f(x=...): pass', 'def f(x=[]): pass', 'def f(x=1e999): pass'):
            with self.subTest(source=source):
                with self.assertRaises((RuntimeError, ValueError)):
                    parameters(source)
                with self.assertRaises(ValueError):
                    runtime_parameters(function(source), receiver=False)
        with self.assertRaises(TypeError):
            runtime_parameters(object(), receiver=False)
        f = function('def f(): pass')
        f.__signature__ = 'invalid'
        with self.assertRaises((TypeError, ValueError)):
            runtime_parameters(f, receiver=False)

    def test_stub_defaults_are_not_executed(self):
        with self.assertRaises(ValueError):
            parameters('def f(x=unknown()): pass')


def sample_api():
    module = ModuleType('sample_wheel')

    class Sample:
        def __init__(self, value, *, enabled=True):
            raise AssertionError('signature inspection must not construct objects')

        def __repr__(self):
            raise AssertionError('signature inspection must not call methods')

        def evaluate(self, *, count=1):
            raise AssertionError('signature inspection must not evaluate')

        @staticmethod
        def create(value=None):
            raise AssertionError('signature inspection must not call factories')

        @property
        def price(self):
            raise AssertionError('signature inspection must not read properties')

    module.Sample = Sample
    module.ValidationError = ValueError
    module.PricingError = RuntimeError
    module.__version__ = '1.0'
    module.version = lambda: '1.0'
    module.f = function('def f(value, /, other=None, *, enabled=True, count=1, scale=1.0): pass')
    api = {
        'symbols': ['Sample', 'ValidationError', 'PricingError', '__version__', 'version', 'f'],
        'class_members': {'Sample': ['__init__', '__repr__', 'evaluate', 'create', 'price']},
        'static_methods': [('Sample', 'create')],
        'properties': [('Sample', 'price')],
        'callables': {
            'version': {'kind': 'function', 'parameters': []},
            'f': {'kind': 'function', 'parameters': parameters(
                'def f(value, /, other=None, *, enabled=True, count=1, scale=1.0): pass')},
            'Sample.__init__': {'kind': 'constructor', 'parameters': parameters(
                'def f(value, *, enabled=True): pass')},
            'Sample.__repr__': {'kind': 'method', 'parameters': []},
            'Sample.evaluate': {'kind': 'method', 'parameters': parameters('def f(*, count=1): pass')},
            'Sample.create': {'kind': 'staticmethod', 'parameters': parameters('def f(value=None): pass')},
        },
    }
    return module, api


class RuntimeContractTests(unittest.TestCase):
    def test_all_callable_kinds_without_executing_native_work(self):
        module, api = sample_api()
        self.assertEqual(verify_runtime_api(module, api, '1.0'), 6)

    def test_runtime_signature_drift_is_rejected(self):
        cases = [
            'def f(renamed, /, other=None, *, enabled=True, count=1, scale=1.0): pass',
            'def f(value, other=None, *, enabled=True, count=1, scale=1.0): pass',
            'def f(value, /, other, *, enabled=True, count=1, scale=1.0): pass',
            'def f(value, /, other=None, enabled=True, *, count=1, scale=1.0): pass',
            'def f(value, /, other=None, *, enabled=1, count=1, scale=1.0): pass',
            'def f(value, /, other=None, *, enabled=True, count=1.0, scale=1.0): pass',
            'def f(value, /, other=None, *, enabled=True, count=1, scale=1): pass',
            'def f(value, /, other=None, *, count=1, enabled=True, scale=1.0): pass',
            'def f(*args, **kwargs): pass',
            'def f(value=...): pass',
        ]
        for source in cases:
            module, api = sample_api()
            module.f = function(source)
            with self.subTest(source=source), self.assertRaisesRegex(RuntimeError, 'f signature'):
                verify_runtime_api(module, api, '1.0')

    def test_constructor_static_instance_and_special_signatures_are_checked(self):
        for member, source in [
            ('__init__', 'def f(self, value, *, enabled=False): pass'),
            ('create', 'def f(value): pass'),
            ('evaluate', 'def f(self, count=1): pass'),
            ('__repr__', 'def f(self, other): pass'),
        ]:
            module, api = sample_api()
            replacement = function(source)
            setattr(module.Sample, member, staticmethod(replacement) if member == 'create' else replacement)
            with self.subTest(member=member), self.assertRaisesRegex(RuntimeError, 'Sample.' + member):
                verify_runtime_api(module, api, '1.0')

    def test_missing_opaque_members_and_descriptor_drift_are_rejected(self):
        for mutation in ('missing_function', 'opaque_function', 'extra_export', 'extra_member',
                         'missing_special', 'static_binding', 'instance_binding', 'property_binding',
                         'wrong_version', 'wrong_error_base'):
            module, api = sample_api()
            if mutation == 'missing_function':
                del module.f
            elif mutation == 'opaque_function':
                module.f = object()
            elif mutation == 'extra_export':
                module.extra = 1
            elif mutation == 'extra_member':
                module.Sample.extra = 1
            elif mutation == 'missing_special':
                del module.Sample.__repr__
            elif mutation == 'static_binding':
                module.Sample.create = function('def f(value=None): pass')
            elif mutation == 'instance_binding':
                module.Sample.evaluate = staticmethod(module.Sample.evaluate)
            elif mutation == 'property_binding':
                module.Sample.price = lambda self: 1.0
            elif mutation == 'wrong_version':
                module.__version__ = '2.0'
            else:
                module.ValidationError = RuntimeError
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                verify_runtime_api(module, api, '1.0')


class FrozenContractTests(unittest.TestCase):
    def test_compiled_operation_selection_preserves_typed_boundaries(self):
        source = (ROOT / 'rust_pricing.pyi').read_text()
        start = source.index('class CompiledPricingTask:')
        end = source.index('class PricingTaskResult:')
        for before, after in [
            ('def task(self) -> PricingTask', 'def task(self) -> object'),
            ('def with_operation(self, operation:', 'def with_operation(self, selected:'),
            ('*, gamma_relative_bump:', 'gamma_relative_bump:'),
            ('-> CompiledPricingTask:', '-> PricingTask:'),
            ('float | None = None', 'float | None = 0.01'),
        ]:
            changed = source[:start] + source[start:end].replace(before, after) + source[end:]
            self.assertNotEqual(source, changed)
            with self.subTest(after=after), self.assertRaises(RuntimeError):
                verify_stub_static_shape(ast.parse(changed))

    def test_manifest_covers_every_declared_non_property_callable(self):
        stub = (ROOT / 'rust_pricing.pyi').read_bytes()
        expected = set()
        for node in ast.parse(stub).body:
            if isinstance(node, ast.FunctionDef):
                expected.add(node.name)
            elif isinstance(node, ast.ClassDef):
                expected.update(f'{node.name}.{method.name}' for method in node.body
                    if isinstance(method, ast.FunctionDef) and
                    not any(isinstance(d, ast.Name) and d.id == 'property' for d in method.decorator_list))
        self.assertEqual(set(exported_stub_api(stub)['callables']), expected)

    def test_frozen_contracts_reject_duplicate_domains_and_unknown_fields(self):
        contract = {'Example': {'bases': [], 'members': {}}}
        with self.assertRaisesRegex(RuntimeError, 'duplicate frozen'):
            merge_classes(contract, contract)
        invalid = copy.deepcopy(contract)
        invalid['Example']['typo'] = 1
        with self.assertRaisesRegex(RuntimeError, 'unknown frozen'):
            merge_classes(invalid)
        invalid = copy.deepcopy(contract)
        invalid['Example']['members']['f'] = {'kind': 'unknown'}
        with self.assertRaisesRegex(RuntimeError, 'invalid frozen'):
            merge_classes(invalid)

    def test_comparison_declarations_preserve_positional_only_values(self):
        source = (ROOT / 'rust_pricing.pyi').read_text()
        for before, after in [('value: object, /', 'other: object, /'),
                              ('value: object, /', 'value: object')]:
            with self.subTest(after=after), self.assertRaises(RuntimeError):
                verify_stub_static_shape(ast.parse(source.replace(before, after)))


class ArchiveIntegrationTests(unittest.TestCase):
    @staticmethod
    def sources():
        paths = set(archive.REQUIRED_WHEEL_SMOKE_SNIPPETS)
        paths.update(str(path.relative_to(ROOT)) for path in (ROOT / '.github/workflows').glob('*.yml'))
        return {path: (ROOT / path).read_text() for path in paths}

    @staticmethod
    def check(sources):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode='w') as package:
            for name, source in sources.items():
                payload = source.encode()
                entry = tarfile.TarInfo(name)
                entry.size = len(payload)
                package.addfile(entry, io.BytesIO(payload))
        data.seek(0)
        with tarfile.open(fileobj=data, mode='r') as package:
            archive.check_wheel_smoke_gate(package, 'mutated-wheel-gate.tar')

    def test_archive_checks_follow_the_split_runtime_and_static_gates(self):
        source = self.sources()
        self.check(source)
        for name, before in [
            ('scripts/wheel_validation/runtime.py', "if actual != contract['parameters']:"),
            ('scripts/wheel_validation/stub.py', 'verify_stub_static_shape(tree)'),
            ('scripts/wheel_validation/package.py', 'verify_runtime_symbols(python, stub_api, metadata["Version"])'),
        ]:
            mutated = source | {name: source[name].replace(before, 'REMOVED')}
            with self.subTest(path=name), self.assertRaises(SystemExit):
                self.check(mutated)

    def test_model_ci_still_triggers_on_shared_checker_changes(self):
        source = self.sources()
        workflows = [name for name, text in source.items()
                     if name.startswith('.github/workflows/') and "      - 'scripts/smoke_test_wheel.py'" in text]
        self.assertTrue(workflows)
        for name in workflows:
            mutated = source | {name: source[name].replace("      - 'scripts/wheel_validation/**'\n", '')}
            with self.subTest(workflow=name), self.assertRaisesRegex(SystemExit, 'missing wheel gate path filter'):
                self.check(mutated)


if __name__ == '__main__':
    unittest.main()
