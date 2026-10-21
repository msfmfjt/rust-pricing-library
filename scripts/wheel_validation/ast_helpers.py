"""Static contract AST helpers; independent of an installed extension."""
import ast

def duplicates_in(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def literal_alias_values(node: ast.expr | None) -> set[str]:
    if not isinstance(node, ast.Subscript):
        return set()
    if not isinstance(node.value, ast.Name) or node.value.id != "Literal":
        return set()
    slice_value = node.slice
    elements = slice_value.elts if isinstance(slice_value, ast.Tuple) else [slice_value]
    values = set()
    for element in elements:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            values.add(element.value)
    return values


def function_return_name(node: ast.FunctionDef | None) -> str | None:
    if node is None:
        return None
    annotation = node.returns
    if isinstance(annotation, ast.Name):
        return annotation.id
    return None


def function_signature_shape(node: ast.FunctionDef) -> dict[str, object]:
    arguments = node.args
    if arguments.vararg is not None or arguments.kwarg is not None:
        raise RuntimeError(f"wheel type stub {node.name} uses variadic arguments")
    positional_names = [
        argument.arg for argument in arguments.posonlyargs + arguments.args
    ]
    positional_default_names = (
        positional_names[-len(arguments.defaults) :] if arguments.defaults else []
    )
    positional_defaults = {
        name: stub_default_value(default)
        for name, default in zip(positional_default_names, arguments.defaults)
    }
    keyword_only_names = [argument.arg for argument in arguments.kwonlyargs]
    required_keyword_only = [
        name
        for name, default in zip(keyword_only_names, arguments.kw_defaults)
        if default is None
    ]
    keyword_only_defaults = {
        name: stub_default_value(default)
        for name, default in zip(keyword_only_names, arguments.kw_defaults)
        if default is not None
    }
    return {
        "positional": positional_names,
        "positional_defaults": positional_defaults,
        "keyword_only": keyword_only_names,
        "required_keyword_only": required_keyword_only,
        "keyword_only_defaults": keyword_only_defaults,
    }


def stub_default_value(node: ast.expr) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    raise RuntimeError(f"wheel type stub uses unsupported default {ast.unparse(node)}")


def decorator_names(node: ast.FunctionDef) -> set[str]:
    return {
        decorator.id
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Name)
    }


def decorated_members(
    class_methods: dict[str, dict[str, ast.FunctionDef]],
    decorator_name: str,
) -> set[tuple[str, str]]:
    return {
        (class_name, method_name)
        for class_name, methods in class_methods.items()
        for method_name, method in methods.items()
        if decorator_name in decorator_names(method)
    }


def class_base_names(node: ast.ClassDef) -> list[str]:
    bases: list[str] = []
    for base in node.bases:
        if not isinstance(base, ast.Name):
            raise RuntimeError(f"wheel type stub {node.name} uses an unsupported base")
        bases.append(base.id)
    return bases
