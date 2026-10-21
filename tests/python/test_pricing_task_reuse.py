"""One compiled source supports independent, replayable operation selections."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import unittest

import rust_pricing as rp

ROOT = Path(__file__).resolve().parents[2]


def document(name):
    return json.loads((ROOT / f'fixtures/task-v1/{name}.golden.json').read_text())


def task(name):
    return rp.PricingTask.from_json(json.dumps(document(name)))


class CompiledTaskReuseTests(unittest.TestCase):
    def test_reused_operations_match_fresh_compilation_and_replay(self):
        for name in ('standard_risk', 'multi_asset_bs', 'multi_asset_joint',
                     'stochastic_dividend_bs', 'stochastic_dividend_hw', 'stochastic_dividend_lsv'):
            with self.subTest(source=name):
                original = task(name)
                operation = document(name)['operation']
                kwargs = {key: value for key, value in operation.items() if key != 'type'}
                price = original.with_operation('evaluate').compile()
                saved_price = price.evaluate().report.to_json()
                selected = price.with_operation(operation['type'], **kwargs)
                self.assertEqual(price.capability_for(operation['type'], **kwargs).status, 'supported')
                self.assertEqual(selected.task.to_json(), original.to_json())
                self.assertEqual(selected.task_fingerprint, original.fingerprint)
                self.assertEqual(selected.plan_fingerprint, price.plan_fingerprint)
                result = selected.evaluate()
                independent = original.compile().evaluate()
                self.assertEqual(result.report.to_json(), independent.report.to_json())
                self.assertEqual(type(result.native_result), type(independent.native_result))
                # The selected task is self-contained after releasing the original owner.
                del original
                del price
                restored = rp.PricingTask.from_json(selected.task.to_json()).compile()
                self.assertEqual(restored.evaluate().report.to_json(), result.report.to_json())
                self.assertEqual(selected.with_operation('evaluate').evaluate().report.to_json(), saved_price)
                with self.assertRaises(AttributeError):
                    selected.task = restored.task

    def test_new_gamma_task_keeps_the_same_native_plan(self):
        original = task('multi_asset_bs')
        compiled = original.compile()
        for bump in (None, .005, .02):
            with self.subTest(bump=bump):
                changed = compiled.with_operation('aad', gamma_relative_bump=bump)
                self.assertNotEqual(changed.task_fingerprint, compiled.task_fingerprint)
                self.assertEqual(changed.plan_fingerprint, compiled.plan_fingerprint)
                self.assertEqual(changed.capability().gamma_relative_bump, bump)
                fresh = original.with_operation('aad', gamma_relative_bump=bump).compile()
                self.assertEqual(changed.evaluate().report.to_json(), fresh.evaluate().report.to_json())

    def test_missing_trace_and_invalid_operation_do_not_recompile(self):
        value = document('stochastic_dividend_lsv')
        value['operation'] = {'type': 'evaluate'}
        value['source']['equity']['model']['equity']['particles']['retain_reverse_trace'] = False
        compiled = rp.PricingTask.from_json(json.dumps(value)).compile()
        before = compiled.evaluate().report.to_json()
        capability = compiled.capability_for('local_variance_risk')
        self.assertEqual((capability.status, capability.reason), ('unsupported', 'reverse_trace_required'))
        with self.assertRaises(rp.PricingError) as caught:
            compiled.with_operation('local_variance_risk')
        error = caught.exception
        self.assertEqual((error.code, error.stage, error.operation, error.reason),
                         ('unsupported_operation', 'compile', 'local_variance_risk', 'reverse_trace_required'))
        for method in (compiled.capability_for, compiled.with_operation):
            with self.assertRaises(ValueError):
                method('unknown')
            with self.assertRaises(ValueError):
                method('evaluate', gamma_relative_bump=.01)
        self.assertEqual(compiled.evaluate().report.to_json(), before)
        self.assertEqual(compiled.capability().operation, 'evaluate')

    def test_native_preflight_remains_required_for_a_new_operation(self):
        value = document('multi_asset_bs')
        value['operation'] = {'type': 'evaluate'}
        value['source']['markets'][0]['discrete_dividends'] = [
            {'event_id': 1, 'ex_time': .5, 'quote': {'type': 'fixed_cash', 'amount': 90.}}]
        original = rp.PricingTask.from_json(json.dumps(value))
        compiled = original.compile()
        self.assertEqual(original.capability('aad', gamma_relative_bump=.2).status, 'conditional')
        capability = compiled.capability_for('aad', gamma_relative_bump=.2)
        self.assertEqual((capability.status, capability.reason), ('unsupported', 'native_preflight_required'))
        with self.assertRaises(rp.PricingError) as reused:
            compiled.with_operation('aad', gamma_relative_bump=.2)
        with self.assertRaises(rp.PricingError) as independent:
            original.with_operation('aad', gamma_relative_bump=.2).compile()
        for field in ('code', 'stage', 'field', 'operation', 'reason', 'causes'):
            self.assertEqual(getattr(reused.exception, field), getattr(independent.exception, field))
        self.assertEqual(reused.exception.stage, 'compile')
        self.assertEqual(compiled.with_operation('aad').capability().status, 'supported')

    def test_changed_sources_cannot_replace_a_compiled_tasks_inputs(self):
        original = task('multi_asset_joint')
        compiled = original.compile()
        value = document('multi_asset_joint')
        changed = copy.deepcopy(value)
        changed['source']['local_correlation']['particles']['seed'] += 1
        new_task = rp.PricingTask.from_json(json.dumps(changed))
        new_plan = new_task.compile()
        self.assertNotEqual(new_plan.task_fingerprint, compiled.task_fingerprint)
        self.assertNotEqual(new_plan.plan_fingerprint, compiled.plan_fingerprint)
        self.assertEqual(compiled.task.to_json(), original.to_json())
        self.assertEqual(compiled.with_operation('evaluate').plan_fingerprint, compiled.plan_fingerprint)

    def test_concurrent_operations_use_separate_evaluation_workspaces(self):
        risk = task('multi_asset_joint').compile()
        price = risk.with_operation('evaluate')
        expected = [risk.evaluate().report.to_json(), price.evaluate().report.to_json()]
        plans = [risk, price, risk.with_operation('aad'), price.with_operation('evaluate')]
        with ThreadPoolExecutor(max_workers=4) as executor:
            actual = list(executor.map(lambda plan: plan.evaluate().report.to_json(), plans))
        self.assertEqual(actual, expected * 2)


if __name__ == '__main__':
    unittest.main()
