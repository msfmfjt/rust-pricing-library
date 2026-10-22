"""Run reproducible Rust mutation checks and retain complete, scoped evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CARGO_MUTANTS_VERSION = "27.1.0"
PACKAGES = ("pricing", "pricing-numerics")
SOURCE_PATHS = tuple(f"crates/{package}/src" for package in PACKAGES)


def capture(argv: list[str], root: Path) -> str:
    return subprocess.run(
        argv, cwd=root, check=True, stdout=subprocess.PIPE, encoding="utf-8"
    ).stdout


def selection_args(args: argparse.Namespace, root: Path, output: Path) -> list[str]:
    """Select mutations separately from the packages whose tests observe them."""
    command = ["--baseline=run", "--no-shuffle", "--build-timeout=600", f"--jobs={args.jobs}"]
    if args.scope == "smoke":
        command += [
            "--package=pricing",
            "--test-package=pricing",
            "--file=crates/pricing/src/core/number.rs",
            "--cargo-arg=--lib",
            "--cargo-test-arg=core::number::tests",
            "--timeout=60",
        ]
    else:
        for package in args.package or PACKAGES:
            command.append(f"--package={package}")
        # Numerical utility mutations must also face their downstream consumers.
        command += [f"--test-package={package}" for package in PACKAGES]
        # In 27.1.0 --test-package applies to mutants but NOT to the baseline:
        # lab::run_baseline selects only packages with generated mutants. Pass
        # the same packages through Cargo too, so an already-broken consumer
        # cannot falsely kill every mutant, or set an unrealistically short timer.
        command += [f"--cargo-arg=--package={package}" for package in PACKAGES]
        if args.scope == "changed":
            untracked = capture(
                ["git", "ls-files", "--others", "--exclude-standard", "--", *SOURCE_PATHS], root
            ).splitlines()
            if any(path.endswith(".rs") for path in untracked):
                raise ValueError("Stage new Rust source files with git add before changed-line testing")
            merge_base = capture(["git", "merge-base", args.base, "HEAD"], root).strip()
            # Compare against the actual checkout (including tracked local edits).
            # Fixed diff formatting prevents local Git config from changing the
            # interpretation; renames become deletion/addition and are inspected.
            diff = capture(
                ["git", "diff", "--no-ext-diff", "--no-textconv", "--no-color",
                 "--no-renames", "--src-prefix=a/", "--dst-prefix=b/", "--unified=0",
                 merge_base, "--", *SOURCE_PATHS], root
            )
            diff_path = output / "changes.diff"
            diff_path.write_text(diff, encoding="utf-8", newline="\n")
            (output / "merge-base.txt").write_text(merge_base + "\n", encoding="utf-8")
            command.append(f"--in-diff={diff_path}")
        command += [f"--file={pattern}" for pattern in args.file]
        if args.shard:
            command += [f"--shard={args.shard}", "--sharding=round-robin"]
    return command


def mutation_names(inventory: list[dict]) -> set[str]:
    """Validate the inventory before trusting its size or matching outcomes."""
    if not isinstance(inventory, list):
        raise ValueError("Expected a JSON list of selected mutants")
    names = set()
    for mutant in inventory:
        name = mutant.get("name") if isinstance(mutant, dict) else None
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Missing or duplicate selected mutation identity")
        names.add(name)
    return names


def check_phases(entry: dict, expected_packages: tuple[str, ...]) -> None:
    """Reconcile the summary with actual build/test commands and exit statuses."""
    phases = entry.get("phase_results")
    if not isinstance(phases, list) or any(not isinstance(p, dict) for p in phases):
        raise ValueError("Invalid build/test phase records")
    if [p.get("phase") for p in phases] not in (["Build"], ["Build", "Test"]):
        raise ValueError("Expected ordered build/test phases")
    for phase in phases:
        argv = phase.get("argv")
        if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("Invalid build/test command")
        cargo_args = argv[:argv.index("--")] if "--" in argv else argv
        if len(cargo_args) < 2 or cargo_args[1] != "test":
            raise ValueError("Build/test command did not use cargo test")
        if ("--no-run" in cargo_args) != (phase["phase"] == "Build"):
            raise ValueError("Build/test command has inconsistent --no-run mode")
        tested = {arg.split("=", 1)[1].split("@", 1)[0]
                  for arg in cargo_args if arg.startswith("--package=")}
        if not set(expected_packages).issubset(tested):
            raise ValueError("Build/test command omitted required test packages")

    def status(phase):
        value = phase.get("process_status")
        if value in ("Success", "Timeout"):
            return value
        if (isinstance(value, dict) and set(value) == {"Failure"}
                and type(value["Failure"]) is int and value["Failure"] != 0):
            return "Failure"
        raise ValueError("Invalid or unsupported build/test exit status")

    build = status(phases[0])
    if build != "Success":
        if len(phases) != 1:
            raise ValueError("Tests reported after an unsuccessful build")
        inferred = "Unviable" if build == "Failure" else "Timeout"
    else:
        if len(phases) != 2:
            raise ValueError("Successful build did not run tests")
        inferred = {"Success": "MissedMutant", "Failure": "CaughtMutant", "Timeout": "Timeout"}[
            status(phases[1])]
    if entry.get("scenario") == "Baseline":
        if inferred != "MissedMutant" or entry.get("summary") != "Success":
            raise ValueError("Unmutated baseline did not pass")
    elif entry.get("summary") != inferred:
        raise ValueError("Mutation summary disagrees with build/test exit statuses")


def check_outcomes(report: dict, selected: list[dict], expected_packages: tuple[str, ...] = PACKAGES) -> dict:
    """Fail closed on partial/check-only output, not just on a mutation score."""
    if not isinstance(report, dict):
        raise ValueError("Invalid mutation report")
    selected_names = mutation_names(selected)
    names = ("total_mutants", "caught", "missed", "timeout", "unviable", "success")
    counts = {name: report.get(name) for name in names}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("Invalid or missing mutation counts")
    if report.get("cargo_mutants_version") != CARGO_MUTANTS_VERSION:
        raise ValueError("Unexpected cargo-mutants report version")
    if not report.get("end_time") or counts["total_mutants"] != len(selected_names):
        raise ValueError("Mutation run is incomplete")
    if sum(counts[name] for name in names[1:]) != len(selected_names):
        raise ValueError("Mutation outcome counts do not reconcile")
    outcomes = report.get("outcomes", [])
    summaries = {"CaughtMutant": "caught", "MissedMutant": "missed", "Timeout": "timeout",
                 "Unviable": "unviable", "Success": "success"}
    observed = dict.fromkeys(names[1:], 0)
    identities = set()
    if not isinstance(outcomes, list):
        raise ValueError("Invalid mutation outcome records")
    for entry in outcomes:
        if not isinstance(entry, dict):
            raise ValueError("Invalid mutation outcome record")
        check_phases(entry, expected_packages)
        scenario = entry.get("scenario")
        if scenario == "Baseline":
            continue
        mutant = scenario.get("Mutant") if isinstance(scenario, dict) else None
        identity = mutant.get("name") if isinstance(mutant, dict) else None
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("Missing or duplicate mutation identity")
        category = summaries.get(entry.get("summary"))
        if category is None:
            raise ValueError("Unknown mutation outcome")
        identities.add(identity)
        observed[category] += 1
    if identities != selected_names:
        raise ValueError("Mutation outcomes do not match the selected inventory")
    if any(observed[name] != counts[name] for name in observed):
        raise ValueError("Mutation records do not match aggregate counts")
    baselines = [entry for entry in outcomes if entry.get("scenario") == "Baseline"]
    if len(baselines) != 1 or baselines[0].get("summary") != "Success":
        raise ValueError("Unmutated baseline did not pass")
    if counts["success"] or counts["caught"] + counts["missed"] + counts["timeout"] == 0:
        raise ValueError("No test outcomes (or check-only output)")
    return counts


def write_summary(output: Path, record: dict) -> None:
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    lines = ["## Rust mutation testing", "", f"Scope: `{record['scope']}`",
             f"Status: **{record['status']}**", f"Selected mutants: {record.get('selected', 'unknown')}"]
    if "counts" in record:
        lines += ["", "| Outcome | Count |", "| --- | ---: |"]
        lines += [f"| {key} | {value} |" for key, value in record["counts"].items()]
    if "error" in record:
        lines += ["", record["error"]]
    if record["status"] == "no_mutants":
        lines += ["", "No applicable changed-line mutants; this is not whole-library coverage."]
    elif record["status"] == "empty_shard":
        lines += ["", "This shard is empty; the unsharded selection is nonempty. Review all shards."]
    lines += ["", "Timeouts and unviable builds are not counted as caught mutants.", ""]
    summary = "\n".join(lines)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(summary)


def run(args: argparse.Namespace, root: Path = ROOT) -> int:
    parent = (root / args.output).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    # Never let an interrupted or failed invocation reuse old successful evidence.
    output = Path(tempfile.mkdtemp(prefix=f"{args.scope}-", dir=parent))
    record = {"scope": args.scope, "status": "error", "tool_version": CARGO_MUTANTS_VERSION}
    print(f"Mutation evidence: {output}", flush=True)
    code = 1
    try:
        actual = capture(["cargo", "mutants", "--version"], root).strip()
        if actual != f"cargo-mutants {CARGO_MUTANTS_VERSION}":
            raise ValueError(f"Install cargo-mutants {CARGO_MUTANTS_VERSION}; found {actual!r}")
        record["head"] = capture(["git", "rev-parse", "HEAD"], root).strip()
        record["worktree_status"] = capture(["git", "status", "--porcelain"], root)
        options = selection_args(args, root, output)
        command = ["cargo", "mutants", *options, f"--output={output}"]
        record["command"] = command
        selection = capture([*command, "--list", "--json"], root)
        # 27.1.0 prints no JSON at all when --in-diff removes every mutant.
        # capture(check=True) has already distinguished tool failure from this.
        selected = json.loads(selection) if selection.strip() else []
        mutation_names(selected)
        (output / "selected.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
        record["selected"] = len(selected)
        if args.list:
            record["status"] = "listed_only"
            code = 0
        elif not selected:
            if args.scope == "full" and args.shard:
                unsharded_command = [item for item in command
                                     if not item.startswith(("--shard=", "--sharding="))]
                unsharded = capture([*unsharded_command, "--list", "--json"], root)
                (output / "unsharded.json").write_text(unsharded, encoding="utf-8")
                if not mutation_names(json.loads(unsharded)):
                    raise ValueError("Empty full selection before sharding: check filters")
                record["status"] = "empty_shard"
            elif args.scope != "changed":
                raise ValueError("Empty smoke/full selection: check filters or shard selection")
            else:
                record["status"] = "no_mutants"
            code = 0
        else:
            code = subprocess.run(command, cwd=root, check=False).returncode
            record["cargo_mutants_exit_code"] = code
            report_path = output / "mutants.out" / "outcomes.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            expected_packages = ("pricing",) if args.scope == "smoke" else PACKAGES
            record["counts"] = check_outcomes(report, selected, expected_packages)
            if code == 0 and (report["missed"] or report["timeout"]):
                code = 1
            record["status"] = "passed" if code == 0 else "failed"
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        record["error"] = str(error)
        code = code or 1
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        code = 130
    finally:
        write_summary(output, record)
    return code if code >= 0 else 128 - code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("smoke", "changed", "full"), nargs="?", default="smoke")
    parser.add_argument("--tool-version", action="version", version=CARGO_MUTANTS_VERSION)
    parser.add_argument("--base", help="PR base ref/SHA; required for changed scope")
    parser.add_argument("--package", choices=PACKAGES, action="append")
    parser.add_argument("--file", action="append", default=[], help="Production source glob (repeatable)")
    parser.add_argument("--shard", help="Zero-based shard, e.g. 0/4; use every shard for full coverage")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output", default="target/mutation", help="Parent for fresh evidence directories")
    parser.add_argument("--list", action="store_true", help="List selection without claiming test coverage")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.scope == "changed" and not args.base:
        parser.error("changed requires --base (use the actual PR base, including stacked PRs)")
    if args.scope != "changed" and args.base:
        parser.error("--base is only valid for changed")
    if args.scope == "smoke" and (args.package or args.file or args.shard):
        parser.error("smoke has a fixed selection; use full for custom filters")
    return args


if __name__ == "__main__":
    sys.exit(run(parse_args()))
