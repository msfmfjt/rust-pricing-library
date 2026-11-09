"""Build, price, serialize and roll a contract using the existing native engine."""
from pathlib import Path
import json
import rust_pricing as rp
ROOT = Path(__file__).resolve().parents[2]
b = rp.ContractBuilder('example-call', 2, 1, 'exchange.close')
contract = b.european('call', 100., 1., '2027-09-04', '2027-09-04')
request = rp.PricingRequest.from_json((ROOT / 'fixtures/v3/pricing_request.golden.json').read_text())
source = rp.PricingSpecification.from_request(request, worker_threads=2, reduction_block_size=32)
task = contract.pricing_task(source)
report = task.compile().evaluate().report
print('PV:', report.value, 'standard error:', report.standard_error)
assert rp.Contract.from_json(contract.to_json()).fingerprint == contract.fingerprint
observation = json.loads(contract.analysis_json())['required_observations'][0]
event = contract.fixing_event_json('published-close', observation['id'], 120.)
rolled = contract.advance('[' + event + ']', '2027-09-04')
assert json.loads(rolled.outstanding_payments_json())[0]['amount'] == 20.
checkpoint = rp.ContractLifecycle.from_json(rolled.to_json())
assert checkpoint.residual_contract.fingerprint == rolled.residual_contract.fingerprint
print('Unpaid cashflows:', checkpoint.outstanding_payments_json())
