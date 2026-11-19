"""Extract Rust MIR graphs, construct backwards-reachability certificates, and check them in Lean."""
import argparse
from collections import Counter, deque
import json
import os
import sys
import sysconfig
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
CRATES = ('pricing-numerics', 'pricing', 'pricing-python')
KINDS = ('return', 'index_bounds', 'integer_zero_divisor', 'integer_overflow', 'pointer_alignment', 'null_pointer')
BLOCK = re.compile(r'^    bb(\d+)(?: \(cleanup\))?: \{$')
SINGLE = re.compile(r'bb(\d+);$')
TARGET = re.compile(r'(?:[a-zA-Z0-9_]+: |unwind )?bb(\d+)$')
UNWIND = re.compile(r'unwind (?:continue|unreachable|terminate\([a-z_]+\))$')
TERMINAL = {'return;', 'resume;', 'unreachable;', 'terminate(abi);', 'terminate(cleanup);'}


def terminator(line):
    if line in TERMINAL:
        return [], int(line == 'return;')
    if ' -> ' not in line:
        raise ValueError(f'Unexpected MIR terminator {line[:160]}')
    suffix = line.rsplit(' -> ', 1)[1]
    single = SINGLE.fullmatch(suffix)
    edges = []
    if single:
        edges.append(int(single[1]))
    elif suffix.startswith('[') and suffix.endswith('];'):
        for target in suffix[1:-2].split(', '):
            match = TARGET.fullmatch(target)
            if match:
                edges.append(int(match[1]))
            elif not UNWIND.fullmatch(target):
                raise ValueError(f'Unrecognized MIR successor {target!r}')
    elif not UNWIND.fullmatch(suffix.removesuffix(';')):
        raise ValueError(f'Unrecognized MIR edge list {suffix!r}')
    mask = 0
    if line.startswith('assert('):
        message = re.search(r', "([^"]+)"', line)
        if not message:
            raise ValueError(f'Unrecognized MIR assertion {line[:160]}')
        msg = message[1]
        if msg.startswith('index out of bounds:'):
            k = 1
        elif 'by zero' in msg or 'divisor of zero' in msg:
            k = 2
        elif 'overflow' in msg:
            k = 3
        elif msg.startswith('misaligned pointer'):
            k = 4
        elif msg.startswith('null pointer'):
            k = 5
        else:
            raise ValueError(f'Unclassified MIR assertion {msg!r}')
        mask = 1 << k
    return edges, mask


def bodies(mir):
    header = previous = ''
    code = None
    active = False
    last = None
    for raw in mir:
        line = raw.rstrip('\n')
        if line and not line[0].isspace() and line.endswith('{'):
            if code is not None:
                raise ValueError(f'Unexpected nested MIR body in {header}')
            header = line[:-1].strip()
            kind = ('ctfe_function' if previous == '// MIR FOR CTFE' else 'function') if header.startswith('fn ') else 'constant_or_static'
        elif match := BLOCK.match(line):
            if code is None:
                code = []
            if active or not header or int(match[1]) != len(code):
                raise ValueError(f'Nonconsecutive MIR block in {header}')
            active, last = True, None
        elif active and line == '    }':
            if not last:
                raise ValueError(f'Empty MIR block in {header}')
            code.append(terminator(last))
            active = False
        elif active and line.strip():
            last = line.strip()
        elif line == '}' and code is not None:
            yield header, kind, code
            header, code = '', None
        if line.strip():
            previous = line.strip()
    if active or code is not None:
        raise ValueError('Truncated MIR input')


def certify(rows):
    incoming = [[] for _ in rows]
    for i, (successors, _) in enumerate(rows):
        for target in successors:
            if target >= len(rows):
                raise ValueError(f'Edge to missing block {target}')
            incoming[target].append(i)
    ranks = [[0] * len(KINDS) for _ in rows]
    for k in range(len(KINDS)):
        todo = deque(i for i, (_, mask) in enumerate(rows) if (mask >> k) & 1)
        for i in todo:
            ranks[i][k] = 1
        while todo:
            i = todo.popleft()
            for parent in incoming[i]:
                if ranks[parent][k] == 0:
                    ranks[parent][k] = 1 + ranks[i][k]
                    todo.append(parent)
    return [dict(edges=e, mask=m, ranks=r) for (e, m), r in zip(rows, ranks)]


def kernel_counts(line):
    counts = re.search(r'kernel_bounds=(\d+) kernel_ranks=(\d+)\s*$', line)
    if not counts:
        raise ValueError('Missing Lean kernel verification totals')
    return dict(bounds=int(counts[1]), ranks=int(counts[2]))


def cargo_environment():
    env = os.environ.copy()
    # Some relocatable CPython builds retain their original sysconfig prefix.
    # The native-link unit tests need the library belonging to the active interpreter.
    name = sysconfig.get_config_var('LDLIBRARY') or ''
    candidates = [Path(sys.base_prefix)/'lib', Path(sysconfig.get_config_var('LIBDIR') or '.')]
    for directory in candidates:
        if (directory/name).is_file():
            env['LIBRARY_PATH'] = os.pathsep.join(filter(None, (str(directory), env.get('LIBRARY_PATH'))))
            break
    return env


def verify(mir_dir, output, compile_mir, build):
    output.mkdir(parents=True, exist_ok=True)
    mir_dir.mkdir(parents=True, exist_ok=True)
    checked = subprocess.run(['rustc', '--version'], check=True, text=True, capture_output=True).stdout.strip()
    if not checked.startswith('rustc 1.98.1 '):
        raise ValueError(f'MIR extraction is pinned to Rust 1.98.1; got {checked}')
    with (ROOT/'Cargo.toml').open('rb') as manifest:
        members = tomllib.load(manifest)['workspace']['members']
    if {str(Path('crates')/c) for c in CRATES} != set(members):
        raise ValueError('Workspace membership changed; update the MIR crate list')
    summary = {}
    for crate in CRATES:
        label = crate if build == 'production' else crate+'-unit'
        mir = mir_dir / (label + '.mir')
        if compile_mir:
            # Cargo fingerprints the external output path. A fresh staging path also
            # regenerates artifacts deleted after an earlier successful build.
            with tempfile.TemporaryDirectory(prefix=crate+'-', dir=mir_dir) as stage:
                dump_mir = Path(stage)/'body.mir'
                deps = Path(stage)/'sources.d'
                emit = f'--emit=mir={dump_mir},dep-info={deps}'
                flags = ['--all-features'] if build == 'production' else ['--profile', 'test']
                subprocess.run(['cargo', 'rustc', '--locked', *flags, '-p', crate, '--lib', '--', emit],
                               cwd=ROOT, env=cargo_environment(), check=True)
                dump_mir.replace(mir)
                deps.replace(mir.with_suffix('.d'))
        dump = output / (label + '.ndjson')
        kinds, sites, can_reach = Counter(), Counter(), Counter()
        blocks = edges = 0
        with mir.open() as input_file, dump.open('w') as stream:
            for name, kind, rows in bodies(input_file):
                cert = certify(rows)
                kinds[kind] += 1
                blocks += len(rows)
                edges += sum(len(e) for e, _ in rows)
                for k, label in enumerate(KINDS):
                    sites[label] += sum((mask >> k) & 1 for _, mask in rows)
                    can_reach[label] += int(cert[0]['ranks'][k] > 0)
                stream.write(json.dumps(dict(name=name, blocks=cert), separators=(',', ':')) + '\n')
        result = subprocess.run(['lake', 'env', str(HERE/'.lake/build/bin/mircheck'), str(dump)], cwd=HERE, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        print(result.stdout.rstrip(), flush=True)
        entry_counts = list(can_reach.values())
        expected = f'bodies={sum(kinds.values())} blocks={blocks} entry_reach=#' + str(entry_counts)
        if expected not in result.stdout:
            raise ValueError(f'Independent Lean counts differ: {expected}')
        summary[crate] = dict(body_kinds=kinds, blocks=blocks, edges=edges, event_sites=sites, entry_reaches=can_reach, kernel_obligations=kernel_counts(result.stdout))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', choices=['production', 'unit'], default='production')
    parser.add_argument('--mir-dir', type=Path, help='reuse existing dumps from the pinned compiler')
    parser.add_argument('--output', type=Path, default=HERE/'.lake/audit')
    args = parser.parse_args()
    out = args.output.resolve()
    stats = verify((args.mir_dir or out/'mir').resolve(), out, args.mir_dir is None, args.build)
    name = 'mir-summary.json' if args.build == 'production' else 'mir-unit-summary.json'
    (out/name).write_text(json.dumps(stats, indent=2) + '\n')
