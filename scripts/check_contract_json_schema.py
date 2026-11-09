"""Check saved contract fixtures and independent malformed source cases."""
import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker
ROOT=Path(__file__).resolve().parents[1]
schema=json.loads((ROOT/'schemas/contract-v1/contract.schema.json').read_text())
Draft202012Validator.check_schema(schema)
validator=Draft202012Validator(schema,format_checker=FormatChecker())
fixtures=list((ROOT/'fixtures/contract-v1').glob('*.json'))
for p in fixtures: validator.validate(json.loads(p.read_text()))
base=json.loads(fixtures[0].read_text())
bad=[]
for key,value in [('schema_version',2),('document_kind','product_graph'),('unknown',True)]:
    v=copy.deepcopy(base);v[key]=value;bad.append(v)
v=copy.deepcopy(base);v['observations'][0]['date']='2027-02-30';bad.append(v)
v=copy.deepcopy(base);v['root']['cashflow']['amount']={'type':'constant','value':True};bad.append(v)
v=copy.deepcopy(base);v['root']['cashflow']['direction']='net';bad.append(v)
v=copy.deepcopy(base);v['root']={'type':'exercise','id':'x','date':'2027-01-01','contract':{'type':'empty'}};bad.append(v)
for v in bad:
    if validator.is_valid(v): raise AssertionError(f'accepted invalid contract: {v}')
print(f'Contract schema: {len(fixtures)} fixtures and {len(bad)} rejection cases passed')

advanced_schema = json.loads((ROOT / "schemas/contract-v2/contract.schema.json").read_text())
Draft202012Validator.check_schema(advanced_schema)
advanced = Draft202012Validator(advanced_schema, format_checker=FormatChecker())
for fixture in sorted((ROOT / "fixtures/contract-v2").glob("*.json")):
    doc = json.loads(fixture.read_text())
    advanced.validate(doc)
    downgraded = dict(doc, schema_version=1)
    assert not validator.is_valid(downgraded), fixture.name
    wrong = copy.deepcopy(doc)
    wrong["root"]["option"]["feature"]["regression_basis"] = "quadratic"
    assert not advanced.is_valid(wrong), fixture.name
