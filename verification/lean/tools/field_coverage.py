"""Cross-check every same-file direct scalar getter against its emitted Rust MIR projection."""
import csv
import json
import re
import subprocess
from mir_coverage import CRATES, HERE, ROOT

IMPL = re.compile(r'<impl at (crates/[^:]+\.rs):(\d+):(\d+): [^>]+>::(\w+)\(')
READ = re.compile(r'_0 = (?:copy|move) \((?:_1|\(\*_1\))\.(\d+): (\w+)\);')


def matching_bodies(mir, wanted):
    key = None
    inside = False
    instructions = []
    for line in mir:
        line = line.rstrip('\n')
        if line.startswith('fn '):
            match = IMPL.search(line)
            key = match.groups() if match and match.groups() in wanted else None
            instructions = []
        elif key is not None:
            if line.startswith('    bb') and line.endswith(': {'):
                inside = True
            elif line == '    }':
                inside = False
            elif line == '}':
                yield key, instructions
                key, inside = None, False
            elif inside:
                instructions.append(line.strip())


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p.relative_to(ROOT)) for c in CRATES for p in (ROOT/'crates'/c/'src').rglob('*.rs'))
    declarations = out/'field-declarations.tsv'
    with declarations.open('w') as stream:
        subprocess.run(['cargo','run','--locked','--quiet','--manifest-path',str(HERE/'rust-probes/Cargo.toml'),
                        '--bin','fields','--',str(ROOT)], cwd=HERE, input='\n'.join(files)+'\n',
                       text=True, stdout=stream, check=True)
    with declarations.open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    wanted = {(r['file'], r['impl_line'], r['impl_col'], r['method']): r for r in rows}
    if len(rows) != len(wanted) or not wanted:
        raise ValueError('Duplicate or empty list of getters')
    seen = {}
    for build in ('', '-unit'):
        for crate in CRATES:
            with (out/'mir'/(crate+build+'.mir')).open() as stream:
                for key, ins in matching_bodies(stream, wanted):
                    read = READ.fullmatch(ins[0]) if len(ins) == 2 and ins[1] == 'return;' else None
                    if not read or read[2] != wanted[key]['type']:
                        raise ValueError(f'Unexpected lowering for {key}: {ins[:6]}')
                    indices, builds = seen.setdefault(key, (set(), set()))
                    indices.add(int(read[1]))
                    builds.add(build or 'production')
    missing = wanted.keys()-seen.keys()
    if missing:
        raise ValueError(f'{len(missing)} scalar getters lack MIR: {sorted(missing)[:5]}')
    header = 'import PricingAudit.Projection\nopen PricingAudit.Projection\n'
    with (out/'field-obligations.tsv').open('w') as mapping:
        print('file','impl_line','method','field','source_index','mir_index','builds','theorem',sep='\t',file=mapping)
        cases = []
        for key in sorted(wanted):
            row = wanted[key]
            indices, builds = seen[key]
            for mir_index in sorted(indices):
                name = f'fieldCase{len(cases)}'
                n, index = row['width'], row['source_index']
                # Universally quantified storage forces distinct field indices to remain distinguishable.
                cases.append(f'theorem {name} (α : Type) (s : Fin {n} → α) : '
                    f'execute s [.readField ⟨{mir_index}, by decide⟩, .ret] none = '
                    f'some (s ⟨{index}, by decide⟩) := rfl\n')
                print(key[0], key[1], key[3], row['field'], index, mir_index, ','.join(sorted(builds)), name,
                      sep='\t',file=mapping)
    for start in range(0, len(cases), 100):
        path = out/f'FieldCases{start:04}.lean'
        path.write_text(header + ''.join(cases[start:start+100]))
        subprocess.run(['lake','env','lean',str(path)], cwd=HERE, check=True)
    report = dict(eligible_getters=len(wanted), kernel_cases=len(cases), source_files=len({k[0] for k in wanted}))
    (out/'fields-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    print(run(HERE/'.lake/audit'))
