"""Recheck all stored AAD/FD sweeps that declare required bumps and both tolerances."""
from collections import Counter, defaultdict
from fractions import Fraction
import json
import math
import subprocess

from mir_coverage import HERE, ROOT

FIELDS = {'aad', 'finite_differences', 'required_bump_indices', 'absolute_tolerance',
          'relative_tolerance', 'failures'}


def records(node, path=()):
    if isinstance(node, dict):
        if FIELDS <= node.keys():
            yield path, node
        for key, child in node.items():
            yield from records(child, path + (key,))
    elif isinstance(node, list):
        for i, child in enumerate(node):
            yield from records(child, path + (i,))


def ratio(value):
    exact = Fraction(value)
    return f'(({exact.numerator} : ℚ) / {exact.denominator})'


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    declarations = {}
    occurrences = []
    groups = defaultdict(Counter)
    files = 0
    for directory in (ROOT/'fixtures', ROOT/'design/validation'):
        for source in sorted(directory.rglob('*.json')):
            files += 1
            file = str(source.relative_to(ROOT))
            for path, row in records(json.loads(source.read_text())):
                # The Rust gate first rejects a nonfinite value at any bump, including
                # a coarse bump outside the two-width tolerance comparison.
                observed = [row['aad'], *row['finite_differences']]
                if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in observed):
                    raise ValueError(f'Nonfinite derivative in retained JSON record: {file} {path}')
                a, b = row['required_bump_indices']
                values = [row['aad'], row['finite_differences'][a], row['finite_differences'][b],
                          row['absolute_tolerance'], row['relative_tolerance']]
                expected = not row['failures']
                expression = 'riskGate ' + ' '.join(map(ratio, values)) + ' = ' + str(expected).lower()
                case = declarations.setdefault(expression, len(declarations))
                groups[file + ':' + str(path[0])]['recorded_pass' if expected else 'recorded_fail'] += 1
                occurrences.append((file, json.dumps(path), row['case'], row['parameter'], case))
    if not occurrences:
        raise ValueError('No recorded sweeps were found')
    with (out/'risk-obligations.tsv').open('w') as stream:
        print('file', 'path', 'case', 'parameter', 'obligation', sep='\t', file=stream)
        for row in occurrences:
            print(*row, sep='\t', file=stream)
    sequence = list(declarations)
    for start in range(0, len(sequence), 90):
        path = out/f'RiskCases{start:04d}.lean'
        header = 'import PricingAudit.Evidence\nopen PricingAudit.Evidence\nset_option maxRecDepth 32000\n'
        theorems = (f'theorem riskCase{i} : {s} := by decide +kernel\n'
                    for i, s in enumerate(sequence[start:start+90], start))
        path.write_text(header + ''.join(theorems))
        subprocess.run(['lake', 'env', 'lean', str(path)], cwd=HERE, check=True)
    report = dict(scanned_json_files=files, recorded_sweeps=len(occurrences),
                  distinct_kernel_cases=len(declarations), groups=groups)
    (out/'risk-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    print(run(HERE/'.lake/audit'), flush=True)
