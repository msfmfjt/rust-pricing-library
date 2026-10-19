"""Provenance and safe persistence for the current per-matrix diagnostic CLI."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backtesting'))
from rust_pricing_backtest.domain import BacktestError, fingerprint, require

ACCEPTANCE = 'design/validation/vega-kt-quote-control-provider-v0.1.json'
RUNNER = 'scripts/check_raw_variance_basis.py'
PROTOCOL = 'current_variance_basis_validation_v1'
INTEGRITY = 'report_content_sha256'
AUXILIARY = (
    RUNNER, 'scripts/raw_variance_basis.py', 'scripts/test_raw_variance_basis.py',
    'scripts/check_custom_variance_basis.py', 'scripts/custom_basis_cases.py',
    'scripts/extension_reference.py', 'scripts/extension_put_reference.py',
    'scripts/multi_step_cases.py', 'scripts/multi_step_reference.py',
    'scripts/check_multi_step_gamma_chain.py', 'fixtures/local-vol/multi-step-gamma-v0.1.json',
    'fixtures/backtesting/raw-variance-basis-policy-v0.1.json',
)
RUNTIME = ('scripts/variance_basis_runtime.py', 'scripts/check_current_variance_basis.py',
           'scripts/test_variance_basis_runtime.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'basis_runtime_json', 'duplicate JSON key')
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                          parse_constant=lambda _: require(False, 'basis_runtime_json', 'nonfinite JSON value'))
    except (ValueError, UnicodeError) as error:
        if isinstance(error, BacktestError): raise
        raise BacktestError('basis_runtime_json', 'invalid JSON document') from error


def native_binary_hash():
    import rust_pricing as rp
    return sha(Path(getattr(rp, 'rust_pricing', rp).__file__))


def current_provenance(binary_hash=None):
    """Verify the current accepted production code and the unchanged numerical runner.

    Report-only documents and earlier CLI boundary versions are not numerical
    dependencies. Every native/Python package file, oracle and policy is checked.
    New orchestration files are captured separately in the run's source manifest.
    """
    prior = read_json(ROOT/ACCEPTANCE)
    require(prior['status'] == 'passed' and prior['failed'] == 0, 'basis_reference',
            'passing current provider acceptance required')
    binary = native_binary_hash() if binary_hash is None else binary_hash
    require(binary == prior['native_binary_sha256'], 'basis_reference', 'same accepted native binary required')
    names = {n for n in prior['source_sha256'] if n.startswith(('crates/', 'backtesting/rust_pricing_backtest/'))}
    names.update(AUXILIARY)
    require(names <= prior['source_sha256'].keys(), 'basis_reference', 'missing accepted source dependency')
    for name in sorted(names):
        require(sha(ROOT/name) == prior['source_sha256'][name], 'basis_reference', 'accepted source changed: '+name)
    names.update((*RUNTIME, ACCEPTANCE))
    return {n: sha(ROOT/n) for n in sorted(names)}, binary, sha(ROOT/ACCEPTANCE)


def isolated_runner(provenance):
    """Load an isolated instance of the byte-unchanged numerical implementation.

    Only its provenance provider and report writer are replaced. Its calculation,
    price oracle, checks, thresholds and continuation loop remain unchanged.
    Other imports of the historical checker keep their historical provenance.
    """
    require(sha(ROOT/RUNNER) == provenance[0][RUNNER], 'basis_reference', 'numerical runner changed')
    spec = importlib.util.spec_from_file_location('_current_variance_basis_runner', ROOT/RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.provenance = lambda: provenance
    module.write = write_report
    return module


def sealed(report):
    body = {k: v for k, v in report.items() if k != INTEGRITY}
    body['runtime_protocol'] = PROTOCOL
    return dict(body, **{INTEGRITY: fingerprint(body)})


def write_report(path, report):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(sealed(report), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify_resume(path):
    report = read_json(path)
    require(isinstance(report, dict) and report.get('runtime_protocol') == PROTOCOL
            and report.get(INTEGRITY) == fingerprint({k: v for k, v in report.items() if k != INTEGRITY}),
            'basis_resume_integrity', 'current sealed report required; changed or historical reports must restart')
    require(report.get('document_kind') == 'raw_variance_basis_validation'
            and report.get('phase') in ('native', 'complete') and report.get('status') in ('in_progress', 'passed', 'failed'),
            'basis_resume_integrity', 'independent prices and a resumable native phase required')
    return report


def guard_destinations(output, basis, resume=None):
    """Protect inputs, source/evidence records and their symlink/hardlink aliases."""
    record = read_json(ROOT/ACCEPTANCE)
    protected = [ROOT/n for n in (*record['source_sha256'], ACCEPTANCE, *RUNTIME)]
    protected += [Path(basis)] + ([Path(resume)] if resume else [])
    partial = Path(output).with_suffix('.partial.json')
    outputs = [Path(output), partial]
    outputs += [p.with_suffix(p.suffix+'.tmp') for p in outputs]
    require(len({p.resolve() for p in outputs}) == len(outputs), 'basis_output_collision',
            'final, partial and temporary output paths must be distinct')
    for path in outputs:
        for other in protected:
            require(path.resolve() != other.resolve() and not (path.exists() and other.exists() and path.samefile(other)),
                    'basis_output_collision', 'output must not replace an input, saved result or accepted source: '+str(path))
