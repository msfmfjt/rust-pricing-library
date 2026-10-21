"""Normalize caller-visible parameters without executing a callable or evaluating code."""
from __future__ import annotations
import ast
import inspect
import math


KINDS = {
    inspect.Parameter.POSITIONAL_ONLY: 'positional_only',
    inspect.Parameter.POSITIONAL_OR_KEYWORD: 'positional_or_keyword',
    inspect.Parameter.KEYWORD_ONLY: 'keyword_only',
}


def default_value(value):
    # bool must stay distinct from int (True == 1), as must int from float.
    if type(value) not in (type(None), bool, int, float, str):
        raise ValueError(f'unsupported or opaque default: {value!r}')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite signature default')
    return {'type': type(value).__name__, 'value': value}


def stub_parameters(node: ast.FunctionDef, *, receiver: bool) -> list[dict]:
    args = node.args
    if args.vararg is not None or args.kwarg is not None:
        raise RuntimeError(f'wheel type stub {node.name} uses variadic arguments')
    positional = [(a, 'positional_only') for a in args.posonlyargs]
    positional += [(a, 'positional_or_keyword') for a in args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    parameters = []
    for (argument, kind), default in zip(positional, defaults):
        item = {'name': argument.arg, 'kind': kind}
        if default is not None:
            item['default'] = default_value(ast.literal_eval(default))
        parameters.append(item)
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        item = {'name': argument.arg, 'kind': 'keyword_only'}
        if default is not None:
            item['default'] = default_value(ast.literal_eval(default))
        parameters.append(item)
    if receiver:
        if not parameters or parameters[0]['name'] != 'self' or parameters[0]['kind'] == 'keyword_only' or 'default' in parameters[0]:
            raise RuntimeError(f'wheel type stub {node.name} must declare an instance receiver')
        parameters = parameters[1:]
    return parameters


def runtime_parameters(callable_object, *, receiver: bool) -> list[dict]:
    parameters = list(inspect.signature(callable_object).parameters.values())
    if receiver:
        if (not parameters or parameters[0].name != 'self'
                or parameters[0].kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                or parameters[0].default is not inspect.Parameter.empty):
            raise ValueError('missing or invalid instance receiver')
        parameters = parameters[1:]
    result = []
    for parameter in parameters:
        if parameter.kind not in KINDS:
            raise ValueError(f'variadic runtime parameter: {parameter.name}')
        item = {'name': parameter.name, 'kind': KINDS[parameter.kind]}
        if parameter.default is not inspect.Parameter.empty:
            item['default'] = default_value(parameter.default)
        result.append(item)
    return result
