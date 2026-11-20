"""Compare native numerical repairs with exact rationals and independent Black fixtures."""
import csv
from fractions import Fraction
import json
import math
import random
import struct
import subprocess
import sys
from mir_coverage import HERE, cargo_environment

SIGN = 1 << 63


def bits(value):
    return struct.unpack('>Q', struct.pack('>d', value))[0]


def value(word):
    return struct.unpack('>d', struct.pack('>Q', word))[0]


def ulps(a, b):
    # Opposite signed zeroes represent the same real number.
    def ordered(x):
        return SIGN - (x ^ SIGN) if x & SIGN else SIGN + x
    return abs(ordered(a) - ordered(b))


def rounded(x):
    try:
        return bits(float(x))
    except OverflowError:
        return bits(-math.inf if x < 0 else math.inf)


def division_cases():
    rng = random.Random(137762)
    exponents = [0, 1, 2, 54, 300, 822, 823, 824, 1022, 1023, 1024, 1222,
                 1223, 1224, 1534, 2000, 2045, 2046]

    def finite():
        exponent = rng.choice(exponents) if rng.randrange(2) else rng.randrange(2047)
        return (rng.getrandbits(1) << 63) | (exponent << 52) | rng.getrandbits(52)

    cases = []
    for _ in range(4096):
        cases.append(tuple(finite() for _ in range(4)))
    # Include near-parallel and anti-parallel vectors, exact cancellations, axes,
    # and powers of two. Purely random words almost never exercise these.
    for _ in range(512):
        a, b = finite(), finite()
        c = max(1, a & (SIGN - 1))
        d = max(1, b & (SIGN - 1))
        for row in ((c, d, c, d), (c, d ^ SIGN, d, c), (c, d, c, d - 1),
                    (a, b, c, 0), (a, b, 0, d)):
            cases.append(row)
    powers = [1, bits(sys.float_info.min), bits(0.5), bits(1.), bits(2.),
              bits(2.**1000), bits(sys.float_info.max)]
    cases.extend((a, a, d, d) for a in powers for d in powers)
    for _ in range(2048):
        c, d = rng.uniform(.25, .95), rng.uniform(.25, .95)
        a, b = c * sys.float_info.max, d * sys.float_info.max
        for x, y in ((a, b), (math.nextafter(a, math.inf), b), (a, math.nextafter(b, math.inf))):
            cases.append(tuple(map(bits, (x, y, c, d))))
    for words in cases:
        a, b, c, d = (Fraction(value(w)) for w in words)
        denom = c*c + d*d
        yield ('div', words, (rounded((a*c+b*d)/denom), rounded((b*c-a*d)/denom)))


def black_cases():
    with (HERE/'tools/black-reference.tsv').open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            words = [int(row[k], 16) for k in ('forward', 'strike', 'sigma', 'call', 'put')]
            yield 'black', words[:3], words[3:]


def generate_black_fixtures():
    """Optional regeneration with mpmath 1.3.0. The audit itself uses the checked-in bits."""
    import mpmath as mp
    mp.mp.dps = 180
    rows = set()

    def record(f, k, s):
        f, k, s = map(mp.mpf, (f, k, s))
        if f == k:
            time_value = f * mp.erf(s / (2 * mp.sqrt(2)))
        else:
            low, high = sorted((f, k))
            d1 = mp.log(low / high) / s + s/2
            def cdf(x):
                return mp.erfc(-x / mp.sqrt(2)) / 2
            time_value = low * cdf(d1) - high * cdf(d1 - s)
        prices = time_value + max(f-k, 0), time_value + max(k-f, 0)
        rows.add(tuple(f'{bits(float(x)):016x}' for x in (f, k, s, *prices)))

    cutoff = 2.**-8
    for s in (2.**-1000, 2.**-60, 2.**-51, 2.**-40, 2.**-23, 2.**-15,
              math.nextafter(cutoff, 0), cutoff, math.nextafter(cutoff, math.inf), .2, 2.):
        for f in ([1.] if s < 2.**-250 else [2.**-700, 1., 2.**700]):
            record(f, f, s)
    for s, moneyness in [
        (2.**-51, [.5, 2, 8, 30, 37]),
        (2.**-15, [.9375, 1.9999, 2, 2.0001, 4, 8, 20, 37, 38, 39, 54, 56]),
        (cutoff, [.5, 1.99999, 2.00001, 8, 37, 39, 55]),
        (math.nextafter(cutoff, math.inf), [8, 37, 39]),
        (.2, [8, 12, 30, 37.1, 38, 39, 54, 56]),
        (2., [10, 30, 38, 39]),
    ]:
        for q in moneyness:
            for f in ([1., 2.**700] if q >= 37 and s > cutoff else
                      [1., 2.**990] if q >= 37 else [1.]):
                k = float(mp.mpf(f) * mp.exp(mp.mpf(s)*q))
                record(f, k, s)
                record(k, f, s)
    print('forward', 'strike', 'sigma', 'call', 'put', sep='\t')
    for row in sorted(rows):
        print(*row, sep='\t')


def main():
    cases = [*division_cases(), *black_cases()]
    inputs = ''.join(f"{kind} " + ' '.join(f'{x:016x}' for x in words) + '\n'
                     for kind, words, _ in cases)
    command = ['cargo', 'run', '--quiet', '--locked', '--manifest-path',
               str(HERE/'rust-probes/Cargo.toml'), '--bin', 'numerical_oracles']
    native = subprocess.run(command, cwd=HERE, env=cargo_environment(), input=inputs,
                            text=True, capture_output=True, check=False)
    if native.returncode:
        raise RuntimeError(native.stderr)
    lines = native.stdout.splitlines()
    if len(lines) != len(cases):
        raise ValueError('Native result count differs from input count')
    max_ulps = 0
    max_black_relative = 0.
    failures = []
    for (kind, words, expected), line in zip(cases, lines, strict=True):
        actual = [int(x, 16) for x in line.split()]
        for got, want in zip(actual, expected, strict=True):
            distance = ulps(got, want)
            if kind == 'div':
                finite = math.isfinite(value(want))
                passed = math.isfinite(value(got)) and distance <= 3 if finite else got == want
                if finite:
                    max_ulps = max(max_ulps, distance)
            else:
                s = value(words[2])
                f, k = map(value, words[:2])
                relative_strike = (f-k)/k
                log_fk = math.log1p(relative_strike) if abs(relative_strike) < .5 else math.log(f/k)
                q = abs(log_fk)/s
                relative = abs(value(got)/value(want)-1.) if want else 0.
                if value(want) >= sys.float_info.min:
                    max_black_relative = max(max_black_relative, relative)
                # The Gaussian tail's relative sensitivity to one rounding of q grows as q^2.
                tolerance = max(3e-13, 8*sys.float_info.epsilon*(1+q*q))
                passed = (got == 0 if want == 0 else
                          value(got) > 0. and (distance <= 2 or relative <= tolerance))
            if not passed:
                failures.append((kind, [hex(x) for x in words], hex(got), hex(want), distance))
    result = dict(complex_quotients=sum(k == 'div' for k, _, _ in cases),
                  complex_max_component_ulp=max_ulps, black_pairs=sum(k == 'black' for k, _, _ in cases),
                  black_max_normal_relative_error=max_black_relative, failures=len(failures))
    print(json.dumps(result))
    if failures:
        raise ValueError(f'First disagreements: {failures[:8]}')
    (HERE/'.lake/audit').mkdir(exist_ok=True)
    (HERE/'.lake/audit/numerical-oracles-summary.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    if sys.argv[1:] == ['--generate-black']:
        generate_black_fixtures()
    else:
        main()
