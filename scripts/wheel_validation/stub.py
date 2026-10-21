"""Verify the shipped type stub against independent, domain-specific contracts."""
from __future__ import annotations
import ast
from .contracts import CLASSES, CONTRACT
from .ast_helpers import (class_base_names, decorator_names, decorated_members,
                          duplicates_in, function_return_name, function_signature_shape,
                          literal_alias_values)
from .signatures import stub_parameters


def _check_method(label, method, expected):
    if 'signature' in expected and function_signature_shape(method) != expected['signature']:
        raise RuntimeError(f'wheel type stub {label} signature changed')
    if 'return_name' in expected and function_return_name(method) != expected['return_name']:
        raise RuntimeError(f'wheel type stub {label} return contract changed')
    if 'return_annotation' in expected and ast.unparse(method.returns) != expected['return_annotation']:
        raise RuntimeError(f'wheel type stub {label} return contract changed')
    if 'declaration' in expected:
        frozen = ast.parse(expected['declaration']).body[0]
        if (frozen.name != method.name or ast.dump(frozen.args) != ast.dump(method.args)
                or ast.dump(frozen.returns) != ast.dump(method.returns)
                or decorator_names(frozen) != decorator_names(method)):
            raise RuntimeError(f'wheel type stub {label} declaration contract changed')


def verify_stub_static_shape(tree: ast.Module) -> None:
    imported_names = set()
    names = []
    functions, classes, assignments = {}, {}, {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
                    assignments[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.append(node.name)
            (classes if isinstance(node, ast.ClassDef) else functions)[node.name] = node
    duplicate = sorted(duplicates_in(names))
    if duplicate:
        raise RuntimeError(f'wheel type stub has duplicate top-level definitions: {duplicate}')
    expected_names = set(CONTRACT['exports'])
    if set(names) != expected_names:
        raise RuntimeError(f'wheel type stub top-level definitions changed: missing={sorted(expected_names-set(names))}, unexpected={sorted(set(names)-expected_names)}')
    if set(classes) != set(CLASSES):
        raise RuntimeError('wheel type stub class set changed')
    for cls in classes.values():
        members = [m.name for m in cls.body if isinstance(m, ast.FunctionDef)]
        duplicate = sorted(duplicates_in(members))
        if duplicate:
            raise RuntimeError(f'wheel type stub has duplicate members in {cls.name}: {duplicate}')
    for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        if function.returns is None:
            raise RuntimeError(f'wheel type stub {function.name} is missing a return annotation')
        for argument in function.args.posonlyargs + function.args.args + function.args.kwonlyargs:
            if argument.arg not in {'self', 'cls'} and argument.annotation is None:
                raise RuntimeError(f'wheel type stub {function.name}.{argument.arg} is missing an argument annotation')
    known = {'None', 'RuntimeError', 'ValueError', 'bool', 'dict', 'float', 'int', 'list',
             'object', 'property', 'staticmethod', 'str', 'tuple'} | imported_names | set(names)
    referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    if referenced - known:
        raise RuntimeError(f'wheel type stub has unresolved names: {sorted(referenced-known)}')
    for alias, expected in CONTRACT['literals'].items():
        if literal_alias_values(assignments.get(alias)) != set(expected):
            raise RuntimeError(f'wheel type stub {alias} literal values changed')
    for name, expected in CLASSES.items():
        cls = classes[name]
        if class_base_names(cls) != expected['bases']:
            raise RuntimeError(f'wheel type stub {name} bases changed')
        if 'attributes' in expected:
            attributes = {m.target.id: ast.unparse(m.annotation) for m in cls.body
                          if isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name)}
            if attributes != expected['attributes']:
                raise RuntimeError(f'wheel type stub {name} error attributes changed')
        methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
        if set(methods) != set(expected['members']):
            raise RuntimeError(f'wheel type stub {name} members changed')
        for member, contract in expected['members'].items():
            expected_decorators = set() if contract['kind'] == 'method' else {contract['kind']}
            if decorator_names(methods[member]) != expected_decorators:
                raise RuntimeError(f'wheel type stub {name}.{member} decorator contract changed')
            _check_method(f'{name}.{member}', methods[member], contract)
    for name, expected in CONTRACT['functions'].items():
        if name not in functions:
            raise RuntimeError(f'wheel type stub is missing function {name}')
        _check_method(name, functions[name], expected)


def exported_stub_api(stub: bytes) -> dict[str, object]:
    tree = ast.parse(stub.decode('utf-8'))
    verify_stub_static_shape(tree)
    symbols, class_members, class_methods, callables = [], {}, {}, {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            symbols.append(node.name)
            if isinstance(node, ast.ClassDef):
                methods = {m.name: m for m in node.body if isinstance(m, ast.FunctionDef)}
                class_members[node.name] = list(methods)
                class_methods[node.name] = methods
                for name, method in methods.items():
                    decorators = decorator_names(method)
                    if 'property' in decorators:
                        continue
                    kind = 'constructor' if name == '__init__' else 'staticmethod' if 'staticmethod' in decorators else 'method'
                    callables[f'{node.name}.{name}'] = {
                        'kind': kind, 'parameters': stub_parameters(method, receiver=kind != 'staticmethod'),
                    }
            else:
                callables[node.name] = {'kind': 'function', 'parameters': stub_parameters(node, receiver=False)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.append(node.target.id)
    return {'symbols': symbols, 'class_members': class_members,
            'static_methods': sorted(decorated_members(class_methods, 'staticmethod')),
            'properties': sorted(decorated_members(class_methods, 'property')),
            'callables': callables}
