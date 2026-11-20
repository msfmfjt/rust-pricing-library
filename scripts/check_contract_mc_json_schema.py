"""Validate extended graph/source/task schemas and frozen-version boundaries."""
import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker
ROOT=Path(__file__).resolve().parents[1]
panels=[('product-graph-v2','product_graph','digital','product-graph-v1',1),('specification-v3','pricing_specification','contract_digital','specification-v2',2),('task-v2','pricing_task','contract_digital','task-v1',1)]
for folder,kind,fixture,old_folder,old_version in panels:
    schema=json.loads((ROOT/f'schemas/{folder}/{kind}.schema.json').read_text())
    Draft202012Validator.check_schema(schema)
    check=Draft202012Validator(schema,format_checker=FormatChecker())
    value=json.loads((ROOT/f'fixtures/{folder}/{fixture}.json').read_text());check.validate(value)
    old=json.loads((ROOT/f'schemas/{old_folder}/{kind}.schema.json').read_text())
    previous=Draft202012Validator(old,format_checker=FormatChecker())
    assert not previous.is_valid(value)
    downgraded=copy.deepcopy(value);downgraded['schema_version']=old_version
    assert not check.is_valid(downgraded),f'new opcode accepted under old {kind} version'
    for p in (ROOT/f'fixtures/{old_folder}').glob('*.golden.json'):check.validate(json.loads(p.read_text()))
print('Extended contract graph/source/task fixtures and downgrade rejections passed')
