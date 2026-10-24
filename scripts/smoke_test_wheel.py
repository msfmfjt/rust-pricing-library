"""Install a built wheel, validate its contracts and run isolated Python suites."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

# Also support direct execution and existing importlib-based gate consumers.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from scripts.wheel_validation.archive import selected_wheel
from scripts.wheel_validation.package import prepare_wheel
from scripts.wheel_validation.provider_shards import SHARD_NAMES, SUITES
from scripts.wheel_validation.stub import exported_stub_api, verify_stub_static_shape

TEST_DIRECTORY = Path("tests/python")
NAMED_TEST_GROUPS = {
    "custom-basis": frozenset(suite.module + ".py" for suite in SUITES if suite.name == "custom-basis"),
    "providers": frozenset(suite.module + ".py" for suite in SUITES if suite.name != "custom-basis"),
}
TEST_GROUPS = ("all", *NAMED_TEST_GROUPS, "remaining")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="?", help="wheel path (default: the single wheel in dist)")
    parser.add_argument("--phase", choices=("full", "package", "validation"), default="full",
                        help="full: package and all validation; package: install and verify only")
    parser.add_argument("--group", choices=TEST_GROUPS, default="all",
                        help="validation group; examples run with all or remaining")
    parser.add_argument("--shard", choices=SHARD_NAMES,
                        help="provider cases for one policy and interval count, or its contracts")
    parser.add_argument("--report-directory", type=Path,
                        help="write discovery inventory and per-process completion records")
    args = parser.parse_args(argv)
    if args.phase != "validation" and args.group != "all":
        parser.error("--group requires --phase validation")
    if args.shard and (args.phase != "validation" or args.group != "all"):
        parser.error("--shard requires --phase validation and cannot be combined with --group")
    if args.report_directory and args.phase == "package":
        parser.error("--report-directory requires validation")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    wheel = selected_wheel(args.wheel)
    python = prepare_wheel(wheel)
    if args.phase != "package":
        run_validation(python, args.group, args.shard, args.report_directory)


def run_examples(python: Path) -> None:
    subprocess.run([str(python), "examples/python/european_bs.py"], check=True)
    subprocess.run([str(python), "examples/python/pricing_specification.py"], check=True)
    subprocess.run([str(python), "examples/python/product_graph.py"], check=True)
    subprocess.run([str(python), "examples/python/pricing_task.py"], check=True)
    subprocess.run([str(python), "examples/python/typed_pricing_sources.py"], check=True)
    subprocess.run([str(python), "examples/python/american_lsm.py"], check=True)
    subprocess.run([str(python), "examples/python/american_dual.py"], check=True)
    subprocess.run([str(python), "examples/python/local_vol_vegakt.py"], check=True)
    subprocess.run([str(python), "examples/python/bergomi_lsv.py"], check=True)
    subprocess.run([str(python), "examples/python/hull_white_lsv.py"], check=True)
    subprocess.run([str(python), "examples/python/rough_bergomi.py"], check=True)
    subprocess.run([str(python), "examples/python/pure_bergomi.py"], check=True)
    subprocess.run([str(python), "examples/python/rough_volatility_families.py"], check=True)
    subprocess.run([str(python), "examples/python/rough_family_aad_lsv.py"], check=True)
    subprocess.run([str(python), "examples/python/rough_family_lsv_spot_delta.py"], check=True)
    subprocess.run([str(python), "examples/python/rough_family_market_iv.py"], check=True)
    subprocess.run([str(python), "examples/python/heston_mc_parameter_risk.py"], check=True)
    subprocess.run([str(python), "examples/python/heston_mc_hurst_risk.py"], check=True)
    subprocess.run([str(python), "examples/python/heston_lsv_parameter_risk.py"], check=True)
    subprocess.run([str(python), "examples/python/mixed_bergomi_parameter_risk.py"], check=True)
    subprocess.run([str(python), "examples/python/mixed_bergomi_hurst_risk.py"], check=True)
    subprocess.run([str(python), "examples/python/mixed_bergomi_shape_risk.py"], check=True)
    subprocess.run([str(python), "examples/python/quadratic_heston_parameter_risk.py"], check=True)
    subprocess.run([str(python), "examples/python/rough_family_gamma_bump.py"], check=True)
    subprocess.run([str(python), "examples/python/heston_fourier.py"], check=True)
    subprocess.run([str(python), "examples/python/heston_fourier_greeks.py"], check=True)
    subprocess.run([str(python), "examples/python/heston_parameter_risk.py"], check=True)
    subprocess.run([str(python), "examples/python/heston_hurst_risk.py"], check=True)
    subprocess.run([str(python), "examples/python/heston_calibration.py"], check=True)
    subprocess.run([str(python), "examples/python/heston_iv_calibration.py"], check=True)
    subprocess.run([str(python), "examples/python/heston_iv_refinement.py"], check=True)
    subprocess.run([str(python), "examples/python/heston_iv_holdout.py"], check=True)
    subprocess.run([str(python), "examples/python/multi_asset_lsv.py"], check=True)
    subprocess.run([str(python), "examples/python/multi_asset_bergomi_two_factor.py"], check=True)
    subprocess.run([str(python), "examples/python/bergomi_hull_white.py"], check=True)
    subprocess.run([str(python), "examples/python/multi_asset_rough_bergomi.py"], check=True)
    subprocess.run([str(python), "examples/python/local_correlation.py"], check=True)
    subprocess.run([str(python), "examples/python/local_correlation_lsv_hw.py"], check=True)
    subprocess.run([str(python), "examples/python/path_dependence.py"], check=True)
    subprocess.run([str(python), "examples/python/rough_dividend_hard_barrier.py"], check=True)
    subprocess.run([str(python), "examples/python/rough_dividend_continuous_barrier.py"], check=True)
    subprocess.run([str(python), "examples/python/rough_dividend_market_iv.py"], check=True)


def discover_test_modules(test_directory: Path = TEST_DIRECTORY) -> list[str]:
    modules = sorted(path.name for path in test_directory.glob("test*.py") if path.is_file())
    if not modules:
        raise RuntimeError(f"no test modules found in {test_directory}")
    if any(not Path(name).stem.isidentifier() for name in modules):
        raise RuntimeError("test module filenames must be valid Python identifiers")
    return modules


def select_test_modules(modules: list[str], group: str) -> list[str]:
    if group not in TEST_GROUPS:
        raise ValueError(f"unknown test group: {group}")
    if len(set(modules)) != len(modules):
        raise RuntimeError("duplicate test modules")
    named = set().union(*NAMED_TEST_GROUPS.values())
    missing = named.difference(modules)
    if missing:
        raise RuntimeError(f"named test modules are missing: {sorted(missing)}")
    if group == "all":
        selected = modules
    elif group == "remaining":
        selected = [name for name in modules if name not in named]
    else:
        selected = [name for name in modules if name in NAMED_TEST_GROUPS[group]]
    if not selected:
        raise RuntimeError(f"test group is empty: {group}")
    return selected


def verify_test_inventory(test_directory: Path, modules: list[str]) -> dict[str, int]:
    """Compare multisets, including duplicate IDs, without running any test bodies."""
    def test_ids(suite: unittest.TestSuite):
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                yield from test_ids(test)
            else:
                yield test.id()

    def inventory(pattern: str) -> Counter[str]:
        loader = unittest.TestLoader()
        suite = loader.discover(str(test_directory), pattern=pattern)
        if loader.errors:
            raise RuntimeError("test discovery failed:\n" + "\n".join(loader.errors))
        return Counter(test_ids(suite))

    original = inventory("test*.py")
    combined: Counter[str] = Counter()
    counts = {}
    for module in modules:
        ids = inventory(module)
        if not ids:
            raise RuntimeError(f"test module is empty: {module}")
        counts[module] = ids.total()
        combined.update(ids)
    if not original or original != combined:
        raise RuntimeError(
            "module test inventory differs from full discovery: "
            f"missing={dict(original - combined)}, extra={dict(combined - original)}"
        )
    print(f"wheel-smoke inventory: {len(modules)} modules / {original.total()} tests; "
          "test-ID Counter matches full discovery", flush=True)
    return counts


def run_validation(python: Path, group: str, shard: str | None = None,
                   report_directory: Path | None = None) -> None:
    if shard is not None and (shard not in SHARD_NAMES or group != "all"):
        raise ValueError("invalid provider shard selection")
    modules = discover_test_modules()
    selected = select_test_modules(modules, group)
    if report_directory is not None:
        report_directory.mkdir(parents=True, exist_ok=True)
    # Import-only discovery has its own process so it cannot retain module caches
    # or class state in any of the actual test processes.
    code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from scripts.smoke_test_wheel import verify_test_inventory\n"
        "from scripts.wheel_validation.provider_shards import verify_partition\n"
        "counts = verify_test_inventory(Path(sys.argv[1]), json.loads(sys.argv[2]))\n"
        "shards = verify_partition(Path(sys.argv[1]))\n"
        "if sys.argv[3]:\n"
        "    path = Path(sys.argv[3]) / 'inventory.json'\n"
        "    path.write_text(json.dumps({'modules': counts, 'provider_shards': shards}, "
        "indent=2) + '\\n', encoding='utf-8')\n"
    )
    subprocess.run(
        [str(python), "-X", "faulthandler", "-c", code,
         str(TEST_DIRECTORY), json.dumps(modules), str(report_directory or "")],
        check=True,
    )
    if shard is not None:
        run_test_shard(python, shard, report_directory)
        return
    print(f"wheel-smoke group={group}: {len(selected)} modules", flush=True)
    if group in ("all", "remaining"):
        run_examples(python)
    run_test_modules(python, selected, report_directory=report_directory)


def run_test_process(command: list[str], kind: str, name: str,
                     report_directory: Path | None = None) -> str | None:
    record = {"kind": kind, "name": name, "command": command, "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat()}

    def save():
        if report_directory is not None:
            report_directory.mkdir(parents=True, exist_ok=True)
            path = report_directory / f"{name}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)

    save()
    print(f"wheel-smoke start {kind}={name}", flush=True)
    started = time.monotonic()
    try:
        returncode = subprocess.run(command, check=False).returncode
    except OSError as error:
        returncode = 1
        record["launch_error"] = str(error)
        print(f"wheel-smoke launch error {kind}={name}: {error}", flush=True)
    elapsed = time.monotonic() - started
    detail = f"exit={returncode}"
    if returncode < 0:
        try:
            signal_name = signal.Signals(-returncode).name
        except ValueError:
            signal_name = str(-returncode)
        detail += f" signal={signal_name}"
        record["signal"] = signal_name
    record.update(status="failed" if returncode else "passed", returncode=returncode,
                  elapsed_seconds=elapsed, finished_at=datetime.now(timezone.utc).isoformat())
    save()
    print(f"wheel-smoke finish {kind}={name} elapsed={elapsed:.3f}s {detail}", flush=True)
    return f"{name} ({detail})" if returncode else None


def run_test_shard(python: Path, shard: str, report_directory: Path | None = None) -> None:
    command = [str(python), "-X", "faulthandler", "-m",
               "scripts.wheel_validation.provider_shards", "--run", shard]
    failure = run_test_process(command, "shard", shard, report_directory)
    if failure:
        raise RuntimeError("wheel provider shard failed: " + failure)


def run_test_modules(python: Path, modules: list[str],
                     test_directory: Path = TEST_DIRECTORY,
                     report_directory: Path | None = None) -> None:
    if not modules:
        raise RuntimeError("no test modules selected")
    failures = []
    for module in modules:
        command = [str(python), "-X", "faulthandler", "-m", "unittest", "discover",
                   "-s", str(test_directory), "-p", module, "-v"]
        failure = run_test_process(command, "module", module, report_directory)
        if failure:
            failures.append(failure)
    if failures:
        raise RuntimeError("wheel test modules failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
