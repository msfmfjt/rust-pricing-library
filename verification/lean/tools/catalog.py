"""Index every shipped Rust/Python source and the source locations linked to mathematical contracts."""
import ast
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from mir_coverage import CRATES, HERE, ROOT


def contract_modules():
    root = (HERE/'PricingAudit.lean').read_text()
    modules, all_names = {}, set()
    for path in sorted((HERE/'PricingAudit').glob('*.lean')):
        src = path.read_text()
        namespace = re.search(r'^namespace (PricingAudit\S*)', src, re.M)[1]
        names = set(re.findall(r'^theorem (\w+)', src, re.M))
        modules[path.stem] = names
        all_names.update(namespace+'.'+name for name in names)
        if 'import PricingAudit.'+path.stem+'\n' not in root:
            raise ValueError(f'Module missing from the default build: {path.stem}')
    checked = set(re.findall(r'``(PricingAudit[.\w]+)', root))
    if checked != all_names:
        raise ValueError(f'Axiom audit differs from contract declarations: {checked ^ all_names}')
    return modules, len(all_names)


def deps(directory, suffix, sources):
    result = set()
    for crate in CRATES:
        path = directory/(crate+suffix+'.d')
        content = path.read_text()
        result.update(sources.intersection(re.findall(r'crates/[^\s:]*?\.rs(?=[\s:])', content)))
    return result


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    rust = sorted(str(p.relative_to(ROOT)) for c in CRATES for p in (ROOT/'crates'/c/'src').rglob('*.rs'))
    python = sorted(str(p.relative_to(ROOT)) for p in (ROOT/'backtesting/rust_pricing_backtest').rglob('*.py'))
    modules, theorem_count = contract_modules()
    linked = defaultdict(list)
    variants = defaultdict(list)
    with (HERE/'contracts.tsv').open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            path, module = row['source_path'], row['lean_module']
            name = module+'.'+row['representative_theorem']
            if row['representative_theorem'] not in modules[module]:
                raise ValueError(f'Unknown Lean theorem: {name}')
            present = row['source_anchor'] in (ROOT/path).read_text()
            variant = row['variant']
            if variant != '-':
                matches = variants[variant]
                if not present:
                    continue
                matches.append(name)
            elif not present:
                raise ValueError(f'Contract source anchor missing: {row}')
            linked[path].append(name)
    for variant, matches in variants.items():
        if len(matches) != 1:
            raise ValueError(f'Expected one source contract for {variant}, found {matches}')
    with (out/'rust-declarations.tsv').open('w') as stream:
        subprocess.run(['cargo','run','--locked','--quiet','--manifest-path',str(HERE/'rust-probes/Cargo.toml'),
                        '--bin','declarations','--',str(ROOT)], input='\n'.join(rust)+'\n',
                       stdout=stream, cwd=HERE, text=True, check=True)
    with (out/'rust-declarations.tsv').open() as stream:
        rust_counts = Counter(row['file'] for row in csv.DictReader(stream, delimiter='\t'))
    py_counts = Counter()
    with (out/'python-declarations.tsv').open('w') as stream:
        print('file','line','name',sep='\t',file=stream)
        for name in python:
            tree = ast.parse((ROOT/name).read_bytes(), filename=name)
            for declaration in ast.walk(tree):
                if isinstance(declaration,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    py_counts[name] += 1
                    print(name, declaration.lineno, declaration.name, sep='\t', file=stream)
    with (out/'field-obligations.tsv').open() as stream:
        getters = Counter(row['file'] for row in csv.DictReader(stream, delimiter='\t'))
    production = deps(out/'mir','',set(rust))
    unit = deps(out/'mir','-unit',set(rust))
    with (out/'source-coverage.tsv').open('w') as stream:
        print('source','sha256','declared_functions','rust_production','rust_unit','python_bytecode',
              'scalar_getter_proofs','specific_contracts',sep='\t',file=stream)
        for path in sorted(rust+python):
            py = path.endswith('.py')
            print(path, hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
                  (py_counts if py else rust_counts)[path], int(path in production), int(path in unit),
                  int(py), getters[path], ','.join(linked.get(path, ())) or '-',sep='\t',file=stream)
    report = dict(named_theorems=theorem_count, rust_files=len(rust), python_files=len(python),
                  rust_source_declarations=rust_counts.total(), python_source_declarations=py_counts.total(),
                  production_rust_files=len(production), unit_rust_files=len(unit),
                  compiled_rust_files=len(production|unit), uncompiled_rust_files=sorted(set(rust)-production-unit),
                  files_with_mapped_contracts=sum(path in linked for path in rust+python),
                  files_with_getter_contracts=len(getters),
                  files_with_direct_or_mapped_contracts=sum(path in linked or path in getters for path in rust+python),
                  source_contract_links=sum(map(len, linked.values())),
                  matched_source_variants={k: names[0] for k, names in variants.items()})
    (out/'catalog-summary.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    print(run(HERE/'.lake/audit'))
