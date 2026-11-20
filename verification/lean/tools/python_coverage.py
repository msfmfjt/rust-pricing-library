"""Check the bytecode graphs of every module in the separately shipped backtesting package."""
import dis
import json
import sys
import types
import subprocess
from pathlib import Path
from mir_coverage import HERE, ROOT, certify, kernel_counts

KINDS = ('return', 'explicit_raise', 'conditional_jump', 'back_edge', 'call', 'exception_handler')


def nested(code):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from nested(const)


def graph(code):
    instructions = list(dis.get_instructions(code, adaptive=False, show_caches=False))
    indices = {op.offset: i for i, op in enumerate(instructions)}
    handlers = dis.Bytecode(code).exception_entries
    destinations = {h.target for h in handlers}
    rows = []
    for i, op in enumerate(instructions):
        successors = set()
        jump = op.opcode in dis.hasjrel or op.opcode in dis.hasjabs
        unconditional = op.opname in {'JUMP_FORWARD', 'JUMP_BACKWARD', 'JUMP_BACKWARD_NO_INTERRUPT'}
        raises = op.opname in {'RAISE_VARARGS', 'RERAISE'}
        returns = op.opname in {'RETURN_VALUE', 'RETURN_CONST', 'RETURN_GENERATOR'}
        if jump:
            successors.add(indices[op.argval])
        # RETURN_GENERATOR also resumes at the next instruction when iterated.
        if i + 1 < len(instructions) and not (unconditional or raises or (returns and op.opname != 'RETURN_GENERATOR')):
            successors.add(i+1)
        for h in handlers:
            if h.start <= op.offset < h.end:
                successors.add(indices[h.target])
        mask = sum(1 << k for k, yes in enumerate((returns, raises, jump and not unconditional,
                   any(j <= i for j in successors), op.opname in {'CALL', 'CALL_FUNCTION_EX', 'CALL_INTRINSIC_1', 'CALL_INTRINSIC_2'},
                   op.offset in destinations)) if yes)
        rows.append((sorted(successors), mask))
    return rows


def run(output):
    if sys.version_info[:2] != (3,12):
        raise RuntimeError('Bytecode topology is pinned to CPython 3.12')
    output.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT/'backtesting/rust_pricing_backtest').rglob('*.py'))
    dump = output/'backtesting.ndjson'
    bodies = instructions = 0
    entry = [0]*len(KINDS)
    with dump.open('w') as out:
        for p in files:
            module = compile(p.read_bytes(), str(p.relative_to(ROOT)), 'exec', dont_inherit=True, optimize=0)
            for code in nested(module):
                rows = graph(code)
                cert = certify(rows)
                name = f'{p.relative_to(ROOT)}:{code.co_firstlineno}:{code.co_qualname}'
                out.write(json.dumps(dict(name=name, blocks=cert), separators=(',', ':'))+'\n')
                bodies += 1
                instructions += len(rows)
                for k in range(len(KINDS)):
                    entry[k] += int(cert[0]['ranks'][k] > 0)
    proc = subprocess.run(['lake', 'env', str(HERE/'.lake/build/bin/mircheck'), str(dump)], cwd=HERE,
                          text=True, capture_output=True, check=True)
    print(proc.stdout.rstrip(), flush=True)
    if f'bodies={bodies} blocks={instructions} entry_reach=#{entry}' not in proc.stdout:
        raise ValueError('Independent Lean bytecode counts differ')
    return dict(files=len(files), code_objects=bodies, instructions=instructions,
                entry_reaches=dict(zip(KINDS, entry)), kernel_obligations=kernel_counts(proc.stdout))


if __name__ == '__main__':
    dest = HERE/'.lake/audit'
    stats = run(dest)
    (dest/'python-summary.json').write_text(json.dumps(stats, indent=2)+'\n')
