"""Run the complete ignored extended acceptance inventory in balanced CI jobs.

The same release binary is reused by all shards on a platform. Timing weights
only schedule whole tests: numerical parameters and test bodies are unchanged.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TIMINGS = Path(__file__).with_name("extended_acceptance_timings.json")
SHARD_COUNT = 4
TARGETS = {"accuracy": "extended_model_acceptance", "risk": "extended_risk_acceptance"}
SELECTIONS = ("risk", *(f"accuracy-{index}" for index in range(SHARD_COUNT)))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint():
    # Include dirty and new sources, runtime fixture inputs, and the selection
    # policy. HEAD alone does not identify a locally modified acceptance build.
    paths = subprocess.check_output([
        "git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
        "crates", "fixtures", ".cargo", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml",
        "scripts/extended_acceptance_shards.py", "scripts/extended_acceptance_timings.json",
    ], cwd=ROOT).decode("utf-8").split("\0")
    records = {name: sha256(ROOT / name) for name in sorted(set(paths)) if name}
    if not records:
        raise ValueError("acceptance source inventory is empty")
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()


def weights():
    result = json.loads(TIMINGS.read_text(encoding="utf-8"))["seconds"]
    if not result or any(not isinstance(v, (int, float)) or not 0 < v < float("inf")
                         for v in result.values()):
        raise ValueError("acceptance timings must be finite positive seconds")
    return result


def partition(cases, durations=None, count=SHARD_COUNT):
    durations = weights() if durations is None else durations
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("acceptance inventory is empty or duplicated")
    if set(cases) != set(durations):
        raise ValueError("acceptance timing inventory differs from compiled ignored tests: "
                         f"missing={sorted(set(cases) - set(durations))}; "
                         f"stale={sorted(set(durations) - set(cases))}")
    if not 1 <= count <= len(cases):
        raise ValueError("shard count would create empty shards")
    groups, totals = [[] for _ in range(count)], [0.0] * count
    for case in sorted(cases, key=lambda name: (-durations[name], name)):
        index = min(range(count), key=lambda i: (totals[i], i))
        groups[index].append(case)
        totals[index] += durations[case]
    groups = [sorted(group) for group in groups]
    if Counter(case for group in groups for case in group) != Counter(cases):
        raise ValueError("acceptance partition loses or repeats tests")
    return groups


def inventory(binary, selected=()):
    result = subprocess.run([str(binary), "--ignored", "--list", "--format=terse",
                             "--exact", *selected], check=True, capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines or any(not re.fullmatch(r"[\w:]+: test", line) for line in lines):
        raise ValueError("unexpected or empty libtest inventory")
    cases = [line.removesuffix(": test") for line in lines]
    if len(cases) != len(set(cases)):
        raise ValueError("duplicate compiled acceptance tests")
    return cases


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    before = source_fingerprint()
    command = ["cargo", "test", "--locked", "--release", "-p", "pricing",
               "--test", TARGETS["accuracy"], "--test", TARGETS["risk"],
               "--no-run", "--message-format=json"]
    result = subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True)
    candidates = {}
    for line in result.stdout.splitlines():
        event = json.loads(line)
        target = event.get("target", {}).get("name")
        if event.get("reason") == "compiler-artifact" and event.get("executable"):
            if target in candidates:
                raise ValueError(f"duplicate compiled target: {target}")
            candidates[target] = Path(event["executable"])
    if set(candidates) != set(TARGETS.values()):
        raise ValueError("Cargo did not produce exactly the two acceptance binaries")
    manifest = {"source_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_fingerprint": before,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "compiler": subprocess.check_output(["rustc", "-vV"], cwd=ROOT, text=True),
        "build_environment": {key: os.environ.get(key) for key in
                              ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET")},
        "build_command": command, "binaries": {}}
    for suite, target in TARGETS.items():
        binary = candidates[target]
        destination = directory / binary.name
        shutil.copy2(binary, destination)
        cases = inventory(destination)
        if suite == "accuracy":
            partition(cases)
        manifest["binaries"][suite] = {
            "file": binary.name, "sha256": sha256(destination), "cases": cases}
    if source_fingerprint() != before:
        raise ValueError("acceptance source changed while compiling; rebuild the bundle")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def load_bundle(directory, suite):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if manifest["source_head"] != current_head or set(manifest["binaries"]) != set(TARGETS):
        raise ValueError("acceptance bundle does not match this checkout")
    if manifest["source_fingerprint"] != source_fingerprint():
        raise ValueError("acceptance source fingerprint differs from compiled bundle")
    if manifest["platform"] != {"system": platform.system(), "machine": platform.machine()}:
        raise ValueError("acceptance binary platform differs from this runner")
    record = manifest["binaries"][suite]
    if Path(record["file"]).name != record["file"]:
        raise ValueError("acceptance binary must be a bundle-local filename")
    binary = (directory / record["file"]).resolve()
    if sha256(binary) != record["sha256"]:
        raise ValueError("acceptance binary checksum differs from producer")
    # Artifact downloads do not preserve Unix executable permissions.
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    cases = inventory(binary)
    if cases != record["cases"]:
        raise ValueError("acceptance binary inventory differs from producer")
    return binary, manifest, cases


def run(selection, directory, reports):
    suite = "risk" if selection == "risk" else "accuracy"
    binary, manifest, cases = load_bundle(directory, suite)
    selected = cases if suite == "risk" else partition(cases)[int(selection.rsplit("-", 1)[1])]
    if Counter(inventory(binary, selected)) != Counter(selected):
        raise ValueError("libtest exact filter does not match selected acceptance tests")
    reports.mkdir(parents=True, exist_ok=True)
    log = reports / f"extended-model-{selection}.log"
    report_path = reports / f"extended-model-{selection}.json"
    command = [str(binary), "--ignored", "--exact", "--nocapture", "--test-threads=1", *selected]
    report = {"selection": selection, "source_head": manifest["source_head"],
              "source_fingerprint": manifest["source_fingerprint"],
              "binary_sha256": manifest["binaries"][suite]["sha256"],
              "platform": manifest["platform"], "compiler": manifest["compiler"],
              "inventory": cases, "selected": selected, "command": command,
              "status": "running", "log": log.name}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    summaries = []
    with log.open("w", encoding="utf-8") as output:
        print(f"acceptance selection={selection} cases={json.dumps(selected)}", flush=True)
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8") as process:
            for line in process.stdout:
                output.write(line)
                output.flush()
                print(line, end="", flush=True)
                if line.startswith("test result:"):
                    summaries.append(line.strip())
        returncode = process.returncode
    expected = f"test result: ok. {len(selected)} passed; 0 failed; 0 ignored; 0 measured; "
    passed = returncode == 0 and len(summaries) == 1 and summaries[0].startswith(expected)
    report.update(returncode=returncode, elapsed_seconds=time.monotonic() - started,
                  status="passed" if passed else "failed", summaries=summaries, log_sha256=sha256(log))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"acceptance {selection} failed or did not execute every selected test")
    return report


def combine(directory, output):
    paths = sorted(directory.glob("extended-model-accuracy-*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected_selections = {f"accuracy-{i}" for i in range(SHARD_COUNT)}
    if Counter(r["selection"] for r in records) != Counter(expected_selections):
        raise ValueError("missing or duplicate acceptance shard reports")
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    current_fingerprint = source_fingerprint()
    if any(r["status"] != "passed" or r["returncode"] != 0 or r["source_head"] != current_head
           or r["source_fingerprint"] != current_fingerprint
           for r in records):
        raise ValueError("acceptance shard failed, incomplete or from a different source")
    if len({(r["binary_sha256"], json.dumps(r["platform"], sort_keys=True), r["compiler"])
            for r in records}) != 1:
        raise ValueError("acceptance shards did not reuse the same binary")
    cases = records[0]["inventory"]
    groups = partition(cases)
    for record in records:
        index = int(record["selection"].rsplit("-", 1)[1])
        if record["inventory"] != cases or record["selected"] != groups[index]:
            raise ValueError("acceptance report inventory differs from complete partition")
        log = directory / f"extended-model-{record['selection']}.log"
        if record["log"] != log.name or sha256(log) != record["log_sha256"]:
            raise ValueError("acceptance evidence log differs from executed shard")
        summaries = [line for line in log.read_text(encoding="utf-8").splitlines()
                     if line.startswith("test result:")]
        expected = f"test result: ok. {len(record['selected'])} passed; 0 failed; 0 ignored; 0 measured; "
        if len(summaries) != 1 or not summaries[0].startswith(expected):
            raise ValueError("acceptance evidence does not show complete successful execution")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(f"=== {record['selection']} / {record['binary_sha256']} ===\n")
            destination.write((directory / record["log"]).read_text(encoding="utf-8"))
    summary = {"source_head": current_head, "status": "passed", "test_count": len(cases),
               "source_fingerprint": current_fingerprint,
               "binary_sha256": records[0]["binary_sha256"], "shards": records,
               "total_test_seconds": sum(r["elapsed_seconds"] for r in records),
               "longest_shard_seconds": max(r["elapsed_seconds"] for r in records)}
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--build", action="store_true")
    choice.add_argument("--run", choices=SELECTIONS)
    choice.add_argument("--combine", action="store_true")
    choice.add_argument("--plan", action="store_true")
    parser.add_argument("--binary-directory", type=Path, default=Path("target/extended-acceptance-binaries"))
    parser.add_argument("--reports", type=Path, default=Path("extended-acceptance-reports"))
    parser.add_argument("--output", type=Path, default=Path("extended-model-accuracy.log"))
    args = parser.parse_args()
    if args.build:
        build(args.binary_directory)
    elif args.run:
        run(args.run, args.binary_directory, args.reports)
    elif args.combine:
        combine(args.reports, args.output)
    else:
        timings = weights()
        print(json.dumps([{ "selection": f"accuracy-{i}", "estimated_seconds": sum(timings[n] for n in group),
                           "cases": group} for i, group in enumerate(partition(list(timings)))], indent=2))


if __name__ == "__main__":
    main()
