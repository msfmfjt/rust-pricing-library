"""Generate strict product_graph v1 and pricing_specification v2 schemas."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from build_specification_schema import build_schema, obj, tag, ref, array

ROOT = Path(__file__).resolve().parents[1]
GRAPH_OUTPUT = ROOT / 'schemas/product-graph-v1/product_graph.schema.json'
SPEC_OUTPUT = ROOT / 'schemas/specification-v2/pricing_specification.schema.json'

def schemas():
    u32 = dict(type='integer', minimum=0, maximum=2**32-1)
    u16 = dict(type='integer', minimum=0, maximum=2**16-1)
    date = dict(type='string', format='date', pattern=r'^\d{4}-\d{2}-\d{2}$')
    number = dict(type='number')
    operations = [tag('literal', dict(value=number))]
    for name in ('terminal_spot', 'pre_dividend_spot'):
        operations.append(tag(name, dict(underlying_id=u32, observation_date=date)))
    for name, fields in [('add', 'left right'), ('subtract', 'left right'),
                         ('multiply', 'left right'), ('divide', 'numerator denominator'),
                         ('minimum', 'left right'), ('maximum', 'left right'),
                         ('indicator', 'input'), ('negate', 'input'),
                         ('barrier_hit_state', 'previous hit_weight')]:
        operations.append(tag(name, {field: u32 for field in fields.split()}))
    for name, fields in [('smooth_minimum', 'left right'), ('smooth_maximum', 'left right'),
                         ('smooth_indicator', 'input')]:
        operations.append(tag(name, {**{field: u32 for field in fields.split()},
                                    'half_width': dict(type='number', exclusiveMinimum=0)}))
    definitions = {
        'graph_operation': dict(oneOf=operations),
        'graph_node': obj(dict(id=u32, operation=ref('graph_operation'))),
        'graph_cashflow': obj(dict(output_node=u32, payment_date=date)),
        'graph_product': obj(dict(currency_id=u16,
            underlying_ids={**array(u32, 1), 'uniqueItems': True},
            nodes=array(ref('graph_node'), 1, 1_000_000),
            cashflows=array(ref('graph_cashflow'), 1, 64))),
    }
    graph = obj(dict(document_kind=dict(type='string', const='product_graph'),
                     schema_version=dict(type='integer', const=1), product=ref('graph_product')))
    graph.update({'$schema': 'https://json-schema.org/draft/2020-12/schema',
                  '$id': 'urn:rust-pricing-library:schema:product-graph-v1:product_graph',
                  'title': 'Product graph schema v1', '$defs': copy.deepcopy(definitions)})
    spec = build_schema()
    spec['$defs'].update(copy.deepcopy(definitions))
    # A PricingRequest has one market; multi-asset graphs use the existing multi-asset facade.
    graph_variant = tag('graph', dict(graph=ref('graph_product')))
    graph_variant['allOf'] = [{'properties': {'graph': {'properties': {'underlying_ids': {'maxItems': 1}}}}}]
    spec['$defs']['product']['oneOf'].append(graph_variant)
    spec['$id'] = 'urn:rust-pricing-library:schema:specification-v2:pricing_specification'
    spec['title'] = 'PricingSpecification schema v2'
    spec['properties']['schema_version']['const'] = 2
    spec['allOf'].append({'if': {'properties': {'product': {'properties': {'type': {'const': 'graph'}}}}},
                         'then': {'properties': {'risk': {'not': {'anyOf': [
                             {'required': ['payoff_smoothing']}, {'required': ['payoff_smoothing_width_ladder']}]}}}}})
    return {GRAPH_OUTPUT: graph, SPEC_OUTPUT: spec}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for path, schema in schemas().items():
        text = json.dumps(schema, indent=2, ensure_ascii=False) + '\n'
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise SystemExit(f'stale schema: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
if __name__ == '__main__':
    main()
