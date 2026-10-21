"""Save complete model inputs and operations; inspect capabilities and risk units."""
from pathlib import Path
import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]
for name in ['standard_risk', 'multi_asset_bs', 'multi_asset_joint',
             'stochastic_dividend_bs', 'stochastic_dividend_lsv', 'stochastic_dividend_hw']:
    task = rp.PricingTask.from_json((ROOT / f'fixtures/task-v1/{name}.golden.json').read_text())
    saved_task = task.to_json()
    restored = rp.PricingTask.from_json(saved_task)
    plan = restored.compile()
    assert plan.capability().status == 'supported'
    result = plan.evaluate()
    report = result.report
    saved_report = report.to_json()
    assert rp.ValuationReport.from_json(saved_report).to_json() == saved_report
    print(name, report.value, report.standard_error, report.uncertainty_scope)
    for risk in report.risks[:2]:
        print(' ', risk.name, risk.unit, risk.method, risk.calibration)
    print(' ', type(result.native_result).__name__)
