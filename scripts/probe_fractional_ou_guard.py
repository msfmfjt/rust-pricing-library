"""Compare the fractional-OU range guard with its single logical mutation.

This bounded, deterministic diagnostic does not prove unreachability or change
mutation scores. It copies the actual Rust source into a fresh output directory;
the repository and its mutation configuration are never edited. Requires rustc
from the repository's pinned toolchain; no Python packages are needed.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "crates/pricing-numerics/src/fractional.rs"
REDUCTION = "crates/pricing-numerics/src/reduction.rs"
GUARD = "result.is_finite() && result.abs() <= 1.0 + 1e-11"
MUTATED_GUARD = GUARD.replace("&&", "||")
DRIVER = r'''
#![allow(dead_code)]
#![forbid(unsafe_code)]
#[path = "reduction.rs"] mod reduction;
pub use reduction::NeumaierSum;
#[path = "original.rs"] mod original;
#[path = "mutant.rs"] mod mutant;

fn bits(value: Option<f64>) -> String {
    value.map_or_else(|| "none".to_owned(), |v| format!("{:016x}", v.to_bits()))
}

fn main() {
    let mut hursts = vec![f64::from_bits(1), f64::MIN_POSITIVE, 1e-300, 1e-100,
        1e-50, 1e-16, 1e-10, 1e-6, 1e-3, 0.005, 0.01, 0.025, 0.05, 0.1,
        0.2, 0.25, 0.3, 0.4, 0.45, 0.49, 0.4999, 0.5_f64.next_down(), 0.5];
    let mut lags = vec![0.0, f64::from_bits(1), f64::MIN_POSITIVE, 1e-300, 1e-100,
        1e-20, 1e-12, 1e-6, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0,
        64.0_f64.next_down(), 64.0, 64.0_f64.next_up(), 100.0, 1e3, 1e5,
        1e10, 1e30, 1e100, 1e308, f64::MAX];
    hursts.extend((1..=100).map(|i| f64::from(i) / 200.0));
    hursts.extend((1..=53).map(|i| 2.0_f64.powi(-i)));
    // Parsing avoids intermediate overflow in powi for subnormal powers of ten.
    lags.extend((-323..=308).step_by(4).map(|i| format!("1e{i}").parse::<f64>().unwrap()));
    lags.extend((-80..=80).map(|i| 2.0_f64.powf(f64::from(i) / 4.0)));
    hursts.sort_by(f64::total_cmp);
    hursts.dedup_by(|a, b| a.to_bits() == b.to_bits());
    lags.sort_by(f64::total_cmp);
    lags.dedup_by(|a, b| a.to_bits() == b.to_bits());
    for h in hursts {
        for &lag in &lags {
            assert!(h.is_finite() && h > 0.0 && h <= 0.5);
            assert!(lag.is_finite() && lag >= 0.0);
            let before = original::fractional_ou_correlation(h, lag);
            let after = mutant::fractional_ou_correlation(h, lag);
            println!("{:016x},{:016x},{},{}", h.to_bits(), lag.to_bits(),
                bits(before), bits(after));
        }
    }
}
'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def number(bits: str) -> float:
    return struct.unpack(">d", bytes.fromhex(bits))[0]


def summarize(path: Path) -> dict:
    counts = Counter()
    hursts, lags, pairs = set(), set(), set()
    differing = []
    maximum = None
    for line in path.read_text(encoding="utf-8").splitlines():
        h, lag, before, after = line.split(",")
        assert (h, lag) not in pairs, "Duplicate input pair"
        pairs.add((h, lag))
        hursts.add(h)
        lags.add(lag)
        if before == after:
            counts["both_none" if before == "none" else "same_finite_result"] += 1
        else:
            counts["different_result"] += 1
            differing.append({"hurst_bits": h, "lag_bits": lag,
                              "original_bits": before, "mutant_bits": after})
        if after != "none":
            value = number(after)
            assert abs(value) < float("inf"), "Unexpected nonfinite mutant success"
            if maximum is None or abs(value) > abs(maximum["value"]):
                maximum = {"value": value, "value_bits": after,
                           "hurst_bits": h, "lag_bits": lag}
            if abs(value) > 1.0 + 1e-11:
                assert before == "none", "Original accepted an out-of-range result"
                counts["range_guard_witness"] += 1
    assert pairs and len(pairs) == len(hursts) * len(lags), "Incomplete Cartesian grid"
    return {
        "case_count": len(pairs),
        "hurst_bits": sorted(hursts), "lag_bits": sorted(lags),
        "counts": {key: counts[key] for key in
                   ("same_finite_result", "both_none", "different_result", "range_guard_witness")},
        "maximum_absolute_result": maximum, "differing_cases": differing,
        "raw_rows": {"file": path.name, "bytes": path.stat().st_size,
                     "sha256": sha256(path.read_bytes())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing directories are rejected")
    args = parser.parse_args()
    output = args.output.resolve()
    source = (ROOT / SOURCE).read_bytes()
    reduction = (ROOT / REDUCTION).read_bytes()
    assert source.count(GUARD.encode()) == 1, "Expected exactly one current range guard"
    mutated = source.replace(GUARD.encode(), MUTATED_GUARD.encode())
    output.mkdir(parents=True, exist_ok=False)
    for name, data in [("original.rs", source), ("mutant.rs", mutated),
                       ("reduction.rs", reduction), ("probe.rs", DRIVER.encode())]:
        (output / name).write_bytes(data)
    record = {
        "source": SOURCE, "source_sha256": sha256(source),
        "reduction_sha256": sha256(reduction), "mutant_sha256": sha256(mutated),
        "driver_sha256": sha256(DRIVER.encode()),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "rustc": subprocess.check_output(["rustc", "--version"], cwd=ROOT, text=True).strip(),
        "guard_change": [GUARD, MUTATED_GUARD], "profiles": {},
        "interpretation": "A bounded input search, not an independent accuracy oracle, a proof of equivalence/unreachability, or a replacement for cargo-mutants outcomes.",
    }
    results = {}
    for profile, flags in [("unoptimized", []), ("optimized", ["-O"])]:
        binary = output / ("probe-" + profile)
        command = ["rustc", "--edition=2024", *flags, str(output / "probe.rs"), "-o", str(binary)]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
        rows = output / (profile + ".csv")
        with rows.open("w", encoding="utf-8", newline="\n") as log:
            subprocess.run([str(binary)], cwd=ROOT, check=True, stdout=log)
        record["profiles"][profile] = {"command": command, "exit_code": 0, **summarize(rows)}
        results[profile] = rows.read_bytes()
    record["profile_rows_identical"] = results["unoptimized"] == results["optimized"]
    (output / "summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "profiles": {
        name: {"case_count": value["case_count"], "counts": value["counts"]}
        for name, value in record["profiles"].items()},
        "profile_rows_identical": record["profile_rows_identical"]}, indent=2))


if __name__ == "__main__":
    main()
