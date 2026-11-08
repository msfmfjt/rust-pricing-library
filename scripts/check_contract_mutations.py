"""Run focused semantic mutants in an isolated source copy.

This complements deterministic property tests; it is not exhaustive cargo-mutants
coverage. Exit 101 only counts as killed when Rust tests actually report failure.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
ROOT = Path(__file__).resolve().parents[1]
MUTANTS = [
    ('american_window', 'crates/pricing/src/contract/advanced.rs',
     '(*start..=expiry).contains(&date)', 'date == expiry', 'contract_advanced', 'exercise_replay_uses'),
    ('bridge_touch', 'crates/pricing/src/contract/option_history.rs',
     'BarrierDirection::Down => spot <= b.level', 'BarrierDirection::Down => spot < b.level', 'contract_advanced', 'continuous_monitoring_requires'),
    ('inclusive_comparison', 'crates/pricing/src/contract/expression.rs',
     'Self::GreaterEqual => a >= b,', 'Self::GreaterEqual => a > b,', 'contract_mc', 'contract_condition_comparisons'),
    ('payment_direction', 'crates/pricing/src/contract/source.rs',
     'if self == Self::Receive { 1.0 } else { -1.0 }', 'if self == Self::Receive { 1.0 } else { 1.0 }', 'contract_builders', 'swap_fixings'),
    ('event_idempotency', 'crates/pricing/src/contract/lifecycle.rs',
     '&& previous != *e', '&& previous == *e', 'contract_lifecycle', 'replay_is_idempotent'),
    ('future_event_lookahead', 'crates/pricing/src/contract/lifecycle.rs',
     'ordered.iter().filter(|e| e.effective_date <= as_of)', 'ordered.iter()', 'contract_lifecycle', 'future_fixings'),
    ('nested_redemption_gate', 'crates/pricing/src/engine/payoff/contract.rs',
     'let b = self.select(gate, not_c, zero)?;', 'let b = self.select(not_c, gate, zero)?;', 'contract_mc', 'early_redemption'),
    ('autocall_coupon_notional', 'crates/pricing/src/contract/builders.rs',
     '(notional * p.coupon_rate).into(),', 'p.coupon_rate.into(),', 'contract_builders', 'autocall_cancels'),
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-directory', type=Path, required=True)
    parser.add_argument('--only', choices=[m[0] for m in MUTANTS])
    args = parser.parse_args()
    report = args.report_directory.resolve()
    report.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CARGO_INCREMENTAL='0', CARGO_PROFILE_DEV_DEBUG='0')
    environment['CARGO_TARGET_DIR'] = str(report / 'target')
    results = []
    with tempfile.TemporaryDirectory(prefix='contract-mutants-') as temporary:
        copy = Path(temporary) / 'source'
        tracked = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT).decode().split('\0')
        for name in sorted(set(tracked) - {''}):
            source = ROOT / name
            if source.is_file():
                target = copy / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        def run(label, arguments):
            start = time.monotonic()
            proc = subprocess.run(['cargo', 'test', '--offline', '--locked', '-p', 'pricing', *arguments], cwd=copy, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
            (report / f'{label}.log').write_text(proc.stdout)
            return proc, round(time.monotonic() - start, 3)
        base, elapsed = run('baseline', ['--test', 'contract_builders', '--test', 'contract_lifecycle', '--test', 'contract_mc', '--test', 'contract_advanced'])
        if base.returncode:
            raise RuntimeError('Mutation baseline failed; see baseline.log')
        print(f'baseline passed ({elapsed}s)', flush=True)
        for name, path, before, after, suite, test in MUTANTS:
            if args.only and name != args.only:
                continue
            source = copy / path
            original = source.read_text()
            if original.count(before) != 1:
                raise RuntimeError(f'Mutation anchor changed: {name}')
            source.write_text(original.replace(before, after))
            try:
                proc, elapsed = run(name, ['--test', suite, test])
                killed = proc.returncode == 101 and 'test result: FAILED' in proc.stdout and 'running 0 tests' not in proc.stdout
                result = dict(name=name, status='killed' if killed else 'survived_or_invalid', seconds=elapsed, test=f'{suite}::{test}')
                results.append(result)
                print(json.dumps(result), flush=True)
            finally:
                source.write_text(original)
    (report / 'results.json').write_text(json.dumps({'scope':'curated semantic mutants', 'results':results}, indent=2) + '\n')
    if any(r['status'] != 'killed' for r in results):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
