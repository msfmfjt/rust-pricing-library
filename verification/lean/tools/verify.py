"""Rebuild the formal audit and replay both native profiles. Exit nonzero for any discrepancy."""
import json
import re
import subprocess
import sys
from mir_coverage import HERE, cargo_environment

OUT = HERE/'.lake/audit'
MANIFEST = str(HERE/'rust-probes/Cargo.toml')


def stage(name, command, env, native=False):
    log = OUT/(name+'.log')
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
    output = log.read_text()
    data = dict(exit_code=result.returncode)
    if native and result.returncode in (0,1):
        found = re.search(r'^Disagreements with exact contracts: (\d+)$', output, re.M)
        if found and (result.returncode == 0) == (int(found[1]) == 0):
            data['counterexamples'] = int(found[1])
            data['status'] = 'counterexample' if int(found[1]) else 'passed'
    data.setdefault('status', 'passed' if not native and result.returncode == 0 else 'error')
    print(f"{name}: {data}", flush=True)
    if data['status'] == 'error':
        print(output[-3000:], file=sys.stderr)
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = cargo_environment()
    commands = [
        ('lean', ['lake','build']),
        ('mir', [sys.executable,'tools/mir_coverage.py']),
        ('mir-unit', [sys.executable,'tools/mir_coverage.py','--build','unit']),
        ('index-guards', [sys.executable,'tools/index_guards.py']),
        ('python', [sys.executable,'tools/python_coverage.py']),
        ('schema', [sys.executable,'tools/schema_coverage.py']),
        ('risk', [sys.executable,'tools/risk_evidence.py']),
        ('philox', [sys.executable,'tools/philox_vectors.py']),
        ('fields', [sys.executable,'tools/field_coverage.py']),
        ('catalog', [sys.executable,'tools/catalog.py']),
        ('numerical-oracles', [sys.executable,'tools/numerical_oracles.py']),
        ('native-debug', ['cargo','run','--locked','--manifest-path',MANIFEST,'--','--strict']),
        ('native-release', ['cargo','run','--locked','--release','--manifest-path',MANIFEST,'--','--strict']),
    ]
    stages = {}
    summaries = {}
    for name, command in commands:
        if name in ('index-guards','fields','catalog') and any(stages[k]['status'] != 'passed' for k in ('mir','mir-unit')):
            stages[name] = dict(status='blocked', reason='Rust dependency maps unavailable')
            continue
        if name == 'catalog' and stages['fields']['status'] != 'passed':
            stages[name] = dict(status='blocked', reason='Scalar getter map unavailable')
            continue
        if name in {'mir','mir-unit','python','schema','risk','philox'} and stages['lean']['status'] != 'passed':
            stages[name] = dict(status='blocked', reason='Lean build failed')
            continue
        stages[name] = stage(name, command, env, native=name.startswith('native-'))
        path = OUT/(name+'-summary.json')
        if stages[name]['status'] == 'passed' and path.is_file():
            summaries[name] = json.loads(path.read_text())
    report = dict(stages=stages, results=summaries)
    (OUT/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    if any(x['status'] in ('error','blocked') for x in stages.values()):
        return 2
    return int(any(x['status'] == 'counterexample' for x in stages.values()))


if __name__ == '__main__':
    sys.exit(main())
