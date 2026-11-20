"""Prove failure-edge infeasibility for fixed-length, straight-line Rust MIR array guards."""
from collections import Counter
import csv
from hashlib import sha256
import json
import re
import subprocess
from mir_coverage import CRATES, HERE

BLOCK = re.compile(r'bb(\d+)(?: \(cleanup\))?: \{')
ASSERT = re.compile(
    r'assert\(move (_\d+), "index out of bounds: the length is \{\} but the index is \{\}", '
    r'const (\d+)_usize, copy (_\d+)\) -> \[success: bb\d+, unwind[^]]*\];'
)
COMPARE = re.compile(r'(_\d+) = Lt\(copy (_\d+), const (\d+)_usize\);')
ASSIGN = re.compile(r'(_\d+) = (.+);')
LITERAL = re.compile(r'const (\d+)_usize')
COPY = re.compile(r'(?:copy|move) (_\d+)')
REMAINDER = re.compile(r'Rem\((?:copy|move) _\d+, const (\d+)_usize\)')
MESSAGE = '"index out of bounds: the length is {} but the index is {}"'


def classify(assertion, statements):
    guard = ASSERT.fullmatch(assertion)
    if not guard:
        return ('other guard', None, None, 0)
    compare = COMPARE.fullmatch(statements[-1]) if statements else None
    if not compare or (compare[1], compare[3], compare[2]) != guard.groups():
        return ('other fixed-length computation', None, int(guard[2]), 0)
    index = compare[2]
    length = int(compare[3])
    copies = 0
    # Contiguous SSA-style assignments only. An intervening statement or block
    # boundary ends the match, including any possible write through an alias.
    for statement in reversed(statements[:-1]):
        assign = ASSIGN.fullmatch(statement)
        if not assign or assign[1] != index:
            break
        rhs = assign[2]
        if match := LITERAL.fullmatch(rhs):
            return ('literal', int(match[1]), length, copies)
        if match := REMAINDER.fullmatch(rhs):
            return ('remainder', int(match[1]), length, copies)
        if match := COPY.fullmatch(rhs):
            index = match[1]
            copies += 1
        else:
            break
    return ('other fixed-length computation', None, length, copies)


def guards(path):
    header = ''
    body = 0
    block = None
    statements = []
    with path.open() as source:
        for number, raw in enumerate(source, start=1):
            line = raw.strip()
            if not line:
                continue
            if not raw[0].isspace() and line.endswith('{'):
                body += 1
                header = line[:-1].rstrip()
            if match := BLOCK.fullmatch(line):
                block = int(match[1])
                statements = []
            elif line == '}':
                block = None
                statements = []
            elif line.startswith('assert(') and MESSAGE in line:
                if block is None or not header:
                    raise ValueError(f'Index guard outside MIR block in {path}:{number}')
                yield (body, header, block, number, *classify(line, statements))
            elif block is not None:
                statements.append(line)


def kernel_statement(kind, operand, length):
    # Each example quantifies over incoming registers and every bit width
    # able to represent the compiled array length, including the actual usize.
    params = '(w : Nat) (h : ' + str(length) + ' < 2^w) '
    regs = '(r : Registers w) (slot : Nat) (copies : List Nat)'
    if kind == 'literal':
        return (f'example {params}{regs} : checkIndex r slot copies '
                f'(BitVec.ofNat w {operand}) (BitVec.ofNat w {length}) = true := '
                f'literal_in_bounds {operand} {length} (by decide +kernel) h r slot copies')
    return (f'example {params}(x : BitVec w) {regs} : checkIndex r slot copies '
            f'(x % BitVec.ofNat w {operand}) (BitVec.ofNat w {length}) = true := '
            f'remainder_in_bounds {operand} {length} (by decide +kernel) (by decide +kernel) h x r slot copies')


def run(out):
    checked = Counter()
    builds = {}
    inventory = out/'index-guards.tsv'
    with inventory.open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(('build', 'body_id', 'mir_body', 'block', 'mir_line',
                         'classification', 'operand', 'length', 'adjacent_copies'))
        for suffix in ('', '-unit'):
            graph = json.loads((out/('mir'+suffix+'-summary.json')).read_text())
            for crate in CRATES:
                build = crate+suffix
                counts = Counter()
                for body, name, block, line, kind, operand, length, copies in guards(out/'mir'/(build+'.mir')):
                    writer.writerow((build, body, name, block, line, kind, operand, length, copies))
                    counts[kind] += 1
                    if kind in ('literal', 'remainder'):
                        checked[(kind, operand, length)] += 1
                # A second, independent pass in mir_coverage.py counts every
                # corresponding assert terminator, including unknown index shapes.
                total = sum(counts.values())
                expected = graph[crate]['event_sites']['index_bounds']
                if total != expected:
                    raise ValueError(f'{build}: indexed {total} guards, graph reports {expected}')
                builds[build] = dict(index_assertions=total, literal=counts['literal'],
                                     remainder=counts['remainder'],
                                     other_fixed=counts['other fixed-length computation'],
                                     other_guard=counts['other guard'])
    if not checked:
        raise ValueError('No fixed-length index obligations found')
    path = out/'IndexGuardCases.lean'
    statements = [kernel_statement(*case) for case in sorted(checked)]
    path.write_text('import PricingAudit.IndexGuards\nopen PricingAudit.IndexGuards\n' +
                    '\n'.join(statements)+'\n')
    subprocess.run(['lake', 'env', 'lean', str(path)], check=True, cwd=HERE)
    report = dict(builds=builds, distinct_kernel_cases=len(checked), checked_guard_sites=sum(checked.values()),
                  total_guard_sites=sum(x['index_assertions'] for x in builds.values()),
                  inventory_sha256=sha256(inventory.read_bytes()).hexdigest(),
                  obligations=[dict(kind=k, operand=x, length=n, occurrences=count)
                               for (k, x, n), count in sorted(checked.items())])
    (out/'index-guards-summary.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    result = run(HERE/'.lake/audit')
    print(f"{result['checked_guard_sites']} proven guard sites / {result['total_guard_sites']} "
          f"index guards across both build modes; {result['distinct_kernel_cases']} kernel cases")
