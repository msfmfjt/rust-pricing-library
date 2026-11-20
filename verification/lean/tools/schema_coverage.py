"""Kernel-check the published pricing and backtesting JSON schema families."""
from collections import Counter, defaultdict
from decimal import Decimal
import json
import subprocess

from mir_coverage import HERE, ROOT


def canonical(value):
    """Encode JSON equality recursively, including equality of integer and decimal numbers."""
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return str(int(value))
        return format(value, 'f').rstrip('0')
    if isinstance(value, dict):
        return '{' + ','.join(canonical(k) + ':' + canonical(value[k]) for k in sorted(value)) + '}'
    if isinstance(value, list):
        return '[' + ','.join(map(canonical, value)) + ']'
    return json.dumps(value, ensure_ascii=False)


def strings(values):
    literals = ','.join(json.dumps(x, ensure_ascii=False) for x in values)
    return f'([{literals}] : List String)'


def walk(value, path=()):
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from walk(child, path + (key,))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from walk(child, path + (str(i),))


def follow_pointer(document, tokens):
    for token in tokens:
        document = document[int(token)] if isinstance(document, list) else document[token]
    return document


def resolve(ref, current, documents):
    other, _, fragment = ref.partition('#')
    target = other or current
    if fragment and not fragment.startswith('/'):
        raise ValueError(f'Unsupported reference fragment: {ref}')
    tokens = tuple(k.replace('~1', '/').replace('~0', '~') for k in fragment.split('/')[1:])
    schema = follow_pointer(documents[target], tokens)
    if not isinstance(schema, (dict, bool)):
        raise ValueError(f'Reference is not a schema: {ref}')
    return target, tokens, schema


def discriminants(branches, current, documents):
    choices = []
    for branch in branches:
        target = current
        visited = set()
        while '$ref' in branch:
            key = (target, branch['$ref'])
            if key in visited or set(branch) != {'$ref'}:
                raise ValueError(f'Complex reference in oneOf: {key}')
            visited.add(key)
            target, _, branch = resolve(branch['$ref'], target, documents)
        choices.append(branch)
    common = set.intersection(*[
        {key for key in branch.get('required', [])
         if {'const', 'enum'} & branch.get('properties', {}).get(key, {}).keys()}
        for branch in choices
    ])
    candidates = []
    for key in sorted(common):
        tags = []
        for branch in choices:
            field = branch['properties'][key]
            values = field['enum'] if 'enum' in field else [field['const']]
            tags.extend(map(canonical, values))
        candidates.append((len(tags) - len(set(tags)), key, tags))
    if not candidates:
        raise ValueError(f'No required finite discriminator: {current}')
    _, _, tags = min(candidates)
    return tags


def run(dest):
    dest.mkdir(parents=True, exist_ok=True)
    documents, paths = {}, {}
    for directory in (ROOT/'schemas', ROOT/'backtesting/rust_pricing_backtest/schemas'):
        for path in sorted(directory.rglob('*.json')):
            obj = json.loads(path.read_text(), parse_float=Decimal)
            schema_id = obj['$id']
            if schema_id in documents:
                raise ValueError(f'Duplicate schema ID: {schema_id}')
            documents[schema_id] = obj
            paths[schema_id] = str(path.relative_to(ROOT))

    statements = set()
    counts = Counter()
    refs = defaultdict(set)
    occurrences = []

    def prove(kind, file, location, statement):
        counts[kind] += 1
        occurrences.append((kind, file, json.dumps(location, ensure_ascii=False), statement))
        statements.add(statement)

    for schema_id, document in documents.items():
        file = paths[schema_id]
        for location, obj in walk(document):
            if '$ref' in obj:
                target, pointer, _ = resolve(obj['$ref'], schema_id, documents)
                refs[target].add(pointer)
                counts['reference_occurrences'] += 1
            if 'enum' in obj:
                prove('enum_domains', file, location, f'{strings(map(canonical, obj["enum"]))}.Nodup')
            if 'required' in obj:
                names = strings(obj['required'])
                prove('required_lists', file, location, f'{names}.Nodup')
                if obj.get('additionalProperties') is False and not obj.get('patternProperties'):
                    keys = strings(sorted(obj.get('properties', {})))
                    prove('closed_object_keys', file, location, f'contained {keys} {names} = true')
            if 'oneOf' in obj:
                tags = discriminants(obj['oneOf'], schema_id, documents)
                prove('disjoint_one_of', file, location, f'{strings(tags)}.Nodup')

    # Check actual pointer segments against separately enumerated keys of each target.
    for target, pointers in refs.items():
        requested = defaultdict(set)
        for pointer in pointers:
            for i, token in enumerate(pointer):
                requested[pointer[:i]].add(token)
        for parent, children in requested.items():
            obj = follow_pointer(documents[target], parent)
            keys = list(obj) if isinstance(obj, dict) else [str(i) for i in range(len(obj))]
            prove('reference_key_groups', paths[target], parent,
                  f'contained {strings(sorted(keys))} {strings(sorted(children))} = true')
    prove('document_ids', 'all', (), f'{strings(sorted(documents))}.Nodup')
    prove('reference_document_ids', 'all', (),
          f'contained {strings(sorted(documents))} {strings(sorted(refs))} = true')

    ordered = sorted(statements)
    identities = {statement: i for i, statement in enumerate(ordered)}
    with (dest/'schema-obligations.tsv').open('w') as out:
        print('kind', 'file', 'location', 'obligation', sep='\t', file=out)
        for kind, file, location, statement in occurrences:
            print(kind, file, location, identities[statement], sep='\t', file=out)
    for start in range(0, len(ordered), 80):
        path = dest/f'SchemaCases{start:04d}.lean'
        header = ('import PricingAudit.Schema\nopen PricingAudit.Schema\n'
                  'set_option maxRecDepth 100000\nset_option maxHeartbeats 4000000\n')
        theorems = (f'theorem schemaCase{i} : {s} := by decide +kernel\n'
                    for i, s in enumerate(ordered[start:start+80], start))
        path.write_text(header + ''.join(theorems))
        subprocess.run(['lake', 'env', 'lean', str(path)], cwd=HERE, check=True)
    report = dict(documents=len(documents), unique_kernel_cases=len(ordered), **counts)
    (dest/'schema-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    print(run(HERE/'.lake/audit'), flush=True)
