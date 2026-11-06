"""Generate the versioned contract source schema (semantic validation stays in Rust)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def obj(properties, required=None):
    return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": list(properties) if required is None else required}
def ref(name): return {"$ref": "#/$defs/" + name}
def tagged(name, **fields): return obj({"type": {"const": name}, **fields})
def arr(items): return {"type": "array", "items": items, "maxItems": 100000}
number = {"type": "number"}
boolean = {"type": "boolean"}
identifier = {"type": "string", "minLength": 1, "maxLength": 256}
date = {"type": "string", "format": "date", "pattern": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"}
comparison = {"enum": ["less", "less_equal", "equal", "not_equal", "greater_equal", "greater"]}
numeric = [tagged("constant", value=number), tagged("observation", id=identifier)]
numeric += [tagged(n, left=ref("numeric"), right=ref("numeric")) for n in
            ("add", "subtract", "multiply", "divide", "minimum", "maximum")]
numeric += [tagged(n, input=ref("numeric")) for n in ("abs", "negate")]
numeric += [tagged("conditional", condition=ref("boolean"), then_value=ref("numeric"), else_value=ref("numeric")),
            tagged("aggregate", operation={"enum": ["arithmetic_average", "geometric_average", "running_minimum", "running_maximum"]}, values={**arr(ref("numeric")), "minItems": 1})]
bools = [tagged("constant", value=boolean), tagged("compare", operation=comparison, left=ref("numeric"), right=ref("numeric")),
         tagged("date_compare", operation=comparison, left=ref("date_expression"), right=ref("date_expression")),
         tagged("not", input=ref("boolean"))]
bools += [tagged(n, left=ref("boolean"), right=ref("boolean")) for n in ("and", "or")]
dates = [tagged("constant", value=date), tagged("conditional", condition=ref("boolean"), then_value=ref("date_expression"), else_value=ref("date_expression"))]
observation = obj({"id":identifier,"underlying":{"type":"integer","minimum":0,"maximum":4294967295},"kind":{"enum":["spot","pre_dividend_spot","rate"]},"date":date,"fixing_source":identifier,"time":{"type":"string","pattern":r"^([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$"},"timezone":identifier,"calendar":identifier},["id","underlying","kind","date","fixing_source"])
cashflow = obj({"id":identifier,"currency":{"type":"integer","minimum":0,"maximum":65535},"amount":ref("numeric"),"direction":{"enum":["receive","pay"]},"determination_date":date,"payment_date":date,"settlement_date":date})
terms = [tagged("empty"), tagged("cashflow", cashflow=ref("cashflow")), tagged("bundle", contracts=arr(ref("term"))),
         tagged("conditional", condition=ref("boolean"), then_contract=ref("term"), else_contract=ref("term")),
         tagged("reverse", contract=ref("term")), tagged("scale", factor=ref("numeric"), contract=ref("term")),
         tagged("barrier", id=identifier, observations={**arr(identifier),"minItems":1,"uniqueItems":True}, direction={"enum":["up","down"]},level=number,hit=boolean,on_hit=ref("term"),on_no_hit=ref("term"))]
schema = {"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"https://github.com/msfmfjt/rust-pricing-library/schemas/contract-v1/contract.schema.json",**obj({"document_kind":{"const":"contract"},"schema_version":{"const":1},"id":identifier,"observations":arr(ref("observation")),"root":ref("term")}),"$defs":{"numeric":{"oneOf":numeric},"boolean":{"oneOf":bools},"date_expression":{"oneOf":dates},"observation":observation,"cashflow":cashflow,"term":{"oneOf":terms}}}
if __name__ == "__main__":
    (ROOT / "schemas/contract-v1/contract.schema.json").write_text(json.dumps(schema, indent=2) + "\n")
