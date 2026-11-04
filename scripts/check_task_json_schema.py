"""Offline structure/compatibility checks for complete tasks and common reports."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run([sys.executable, str(ROOT / 'scripts/build_task_schemas.py'), '--check'], check=True)
    validators = {}
    count = 0
    for kind, directory, filename in [('task', 'task-v1', 'pricing_task'),
                                      ('report', 'report-v1', 'valuation_report'),
                                      ('report_v2', 'report-v2', 'valuation_report')]:
        schema = json.loads((ROOT / f'schemas/{directory}/{filename}.schema.json').read_text())
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        validators[kind] = validator
        for fixture in sorted((ROOT / f'fixtures/{directory}').glob('*.golden.json')):
            validator.validate(json.loads(fixture.read_text()))
            count += 1
    task = json.loads((ROOT / 'fixtures/task-v1/multi_asset_joint.golden.json').read_text())
    report = json.loads((ROOT / 'fixtures/report-v1/multi_asset_joint.golden.json').read_text())
    rejected = 0

    def reject(kind, original, path, value):
        nonlocal rejected
        changed = copy.deepcopy(original)
        parent = changed
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
        assert not validators[kind].is_valid(changed), f'{kind} accepted {path}: {value}'
        rejected += 1

    for path, value in [
        (['schema_version'], 2), (['document_kind'], 'pricing_request'),
        (['extra'], 1), (['operation', 'type'], 'unavailable'),
        (['operation', 'gamma_relative_bump'], 0),
        (['operation', 'gamma_relative_bump'], 1),
        (['source', 'driver_correlations'], None),
        (['source', 'lsv', 0], None),
        (['source', 'lsv', 0, 'value', 'particles', 'extra'], 1),
        (['source', 'lsv', 1], {'type': 'absent', 'value': {}}),
        (['source', 'numerics', 'maximum_step'], 0),
        (['source', 'hull_white', 'rate_correlations'], [1.1]),
        (['source', 'local_correlation', 'feasibility'], 'silent_projection'),
        (['source', 'local_correlation', 'second_driver_correlations'], None),
    ]:
        reject('task', task, path, value)
    for path, value in [
        (['schema_version'], 2), (['standard_error'], -1),
        (['evaluated_paths'], '0001'), (['task_fingerprint'], '1234'),
        (['uncertainty_scope'], 'total_error'), (['calibration_seeds'], [-1]),
        (['risks', 0, 'unit'], 'unitless'), (['risks', 0, 'extra'], 1),
        (['risks', 0, 'standard_errors'], None),
        (['risks', 0, 'standard_errors'], [-1]),
        (['risks', 0, 'method'], ''), (['risks', 0, 'calibration'], 'ignored'),
    ]:
        reject('report', report, path, value)
    current = json.loads((ROOT / 'fixtures/report-v2/multi_asset_joint.golden.json').read_text())
    for path, value in [
        (['schema_version'], 1), (['schema_version'], 3),
        (['calibration_diagnostics'], None),
        (['calibration_diagnostics', 0, 'kind'], 'unknown'),
        (['calibration_diagnostics', 0, 'seed'], -1),
        (['calibration_diagnostics', 0, 'underlying_ids'], [1, 1]),
        (['calibration_diagnostics', 0, 'axes', 0, 'values'], []),
        (['calibration_diagnostics', 0, 'metrics'], []),
        (['calibration_diagnostics', 0, 'metrics', 0, 'values'], []),
        (['calibration_diagnostics', 0, 'metrics', 0, 'unit'], 'unknown'),
        (['calibration_diagnostics', 0, 'metrics', 0, 'convention'], ''),
        (['calibration_diagnostics', 0, 'metrics', 0, 'extra'], 1),
    ]:
        reject('report_v2', current, path, value)
    for unit, value in [('count', -1), ('count', 0.5), ('count', 2**54), ('indicator', 2)]:
        changed = copy.deepcopy(current)
        changed['calibration_diagnostics'][0]['metrics'][0]['unit'] = unit
        reject('report_v2', changed, ['calibration_diagnostics', 0, 'metrics', 0, 'values', 0], value)
    migrated = copy.deepcopy(report)
    migrated['schema_version'] = 2
    validators['report_v2'].validate(migrated)
    assert not validators['report'].is_valid(current)
    legacy = Draft202012Validator(json.loads((ROOT / 'schemas/v3/pricing_request.schema.json').read_text()))
    assert not legacy.is_valid(task)
    print(f'Task/report schemas: {count} fixtures, {rejected} rejections and legacy separation passed')


if __name__ == '__main__':
    main()
