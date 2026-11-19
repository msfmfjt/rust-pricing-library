"""Check complete blocks emitted by the public Rust API against the Lean Philox model."""
import json
import subprocess
from mir_coverage import HERE

def run(out):
    out.mkdir(parents=True, exist_ok=True)
    cmd = ['cargo', 'run', '--locked', '--quiet', '--manifest-path',
           str(HERE/'rust-probes/Cargo.toml'), '--bin', 'philox']
    text = subprocess.check_output(cmd, cwd=HERE, text=True)
    (out/'philox-vectors.txt').write_text(text)
    lines = text.splitlines()
    if not lines:
        raise ValueError('Rust returned no Philox blocks')
    for start in range(0, len(lines), 30):
        path = out/f'PhiloxCases{start:04}.lean'
        statements = ['import PricingAudit.Sampling', 'open PricingAudit.Sampling',
                      'set_option maxRecDepth 20000', 'set_option maxHeartbeats 1000000']
        for i, line in enumerate(lines[start:start+30], start):
            seed, path_id, base, domain, a, b, c, d = map(int, line.split())
            statements.append(f'theorem philoxCase{i} : rounds 10 (BitVec.ofNat 32 ({seed} % 2^32)) '
                f'(BitVec.ofNat 32 ({seed} / 2^32)) ⟨BitVec.ofNat 32 ({path_id} % 2^32), '
                f'BitVec.ofNat 32 ({path_id} / 2^32), BitVec.ofNat 32 ({base} / 4), {domain}⟩ = '
                f'⟨{a}, {b}, {c}, {d}⟩ := by decide +kernel')
        path.write_text('\n'.join(statements)+'\n')
        subprocess.run(['lake','env','lean',str(path)], cwd=HERE, check=True)
    report = dict(kernel_blocks=len(lines), native_words=4*len(lines), native_midpoints=4*len(lines))
    (out/'philox-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__ == '__main__':
    print(run(HERE/'.lake/audit'))
