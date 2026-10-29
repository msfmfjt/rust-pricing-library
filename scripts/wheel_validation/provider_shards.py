"""Partition expensive provider suites without changing their numerical cases.

The plan is importable without a built extension. Coverage verification imports
the installed-wheel tests, but never constructs providers or runs test bodies.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
import faulthandler
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import unittest

try:
    from .remaining_shards import REMAINING_GROUPS
except ImportError:
    # Direct CLI invocation from the workflow's matrix-generation step.
    from remaining_shards import REMAINING_GROUPS


@dataclass(frozen=True)
class ProviderSuite:
    name: str
    module: str
    class_name: str
    axis: str
    policies: tuple[str, ...]


SUITES = (
    ProviderSuite("custom-basis", "test_vega_kt_custom_basis_provider",
                  "CustomBasisProviderTests", "policies",
                  ("row_hats", "weighted_time_hats", "coupled_row_hats")),
    ProviderSuite("coupled-basis", "test_vega_kt_coupled_basis_provider",
                  "CoupledBasisProviderTests", "policies", ("coupled_row_hats",)),
    ProviderSuite("extension", "test_vega_kt_extension_provider",
                  "ExtensionProviderTests", "extensions", ("all_rows", "linear_time")),
    ProviderSuite("put-extension", "test_vega_kt_extension_put_provider",
                  "PutExtensionProviderTests", "extensions", ("all_rows", "linear_time")),
)


@dataclass(frozen=True)
class ProviderShard:
    name: str
    suite: ProviderSuite
    policy: str | None = None
    intervals: int | None = None
    method: str | None = None
    parameters: tuple[str | int, ...] | None = None


UNSPLIT_SHARDS = tuple(
    shard
    for suite in SUITES
    for shard in (
        ProviderShard(f"{suite.name}-contracts", suite),
        *(ProviderShard(f"{suite.name}-{policy}-{n}", suite, policy, n)
          for policy in suite.policies for n in (4, 6)),
    )
)


def split_shard(shard: ProviderShard) -> tuple[ProviderShard, ...]:
    # Four dated ledgers plus the panels exceeded 330 minutes on Windows.
    # Keep each ledger's strategies, checkpoint replays and repairs together.
    if shard.suite.axis != "policies" or shard.intervals != 4:
        return (shard,)
    return (
        replace(shard, name=f"{shard.name}-panels", method="test_a_native_panels"),
        *(replace(shard, name=f"{shard.name}-{side}-{kind}",
                  method="test_b_dated_ledgers", parameters=(shard.policy, 4, side, kind))
          for side in ("call", "put") for kind in ("arithmetic_asian", "fixed_lookback")),
    )


SHARDS = tuple(part for shard in UNSPLIT_SHARDS for part in split_shard(shard))
# Retain the original process grouping for reproducing historical native faults.
# These aliases are selectable locally, but must not duplicate cases in CI.
COMBINED_SHARDS = tuple(shard for shard in UNSPLIT_SHARDS if shard not in SHARDS)
SHARD_NAMES = tuple(shard.name for shard in (*SHARDS, *COMBINED_SHARDS))


def iter_cases(test: unittest.TestCase):
    """Use the same case list for execution and the pre-run coverage check."""
    cases = type(test).wheel_cases()[test._testMethodName]
    if not cases:
        raise RuntimeError(f"no provider cases selected for {test.id()}")
    for parameters in cases:
        print(f"wheel-provider case={test.id()} parameters={json.dumps(parameters)}", flush=True)
        yield parameters


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


def load_suite(spec: ProviderSuite, test_directory: Path):
    directory = str(test_directory.resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    module = importlib.import_module(spec.module)
    if Path(module.__file__).resolve().parent != test_directory.resolve():
        raise RuntimeError(f"provider module imported from wrong directory: {spec.module}")
    cls = getattr(module, spec.class_name)
    return module, cls


@contextmanager
def configured(cls, shard: ProviderShard):
    """Restore even inherited attributes; never leave inventory discovery mutated."""
    overrides = {} if shard.policy is None else {
        shard.suite.axis: (shard.policy,), "intervals": (shard.intervals,),
    }
    if shard.parameters is not None and shard.method is None:
        raise RuntimeError("provider case selection requires a method")
    if shard.method is not None:
        original_cases = cls.wheel_cases

        def selected_cases(_cls):
            cases = original_cases()
            if shard.method not in cases:
                raise RuntimeError(f"unknown provider case method: {shard.method}")
            return {
                method: [values for values in values_list
                         if method == shard.method
                         and (shard.parameters is None or tuple(values) == shard.parameters)]
                for method, values_list in cases.items()
            }

        overrides["wheel_cases"] = classmethod(selected_cases)
    missing = object()
    previous = {name: cls.__dict__.get(name, missing) for name in overrides}
    try:
        for name, value in overrides.items():
            if not hasattr(cls, name):
                raise RuntimeError(f"provider shard has unknown axis: {name}")
            setattr(cls, name, value)
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                if name in cls.__dict__:
                    delattr(cls, name)
            else:
                setattr(cls, name, value)


def selected_methods(cls, shard: ProviderShard) -> list[str]:
    methods = unittest.TestLoader().getTestCaseNames(cls)
    cases = cls.wheel_cases()
    if set(cases).difference(methods):
        raise RuntimeError("provider cases refer to missing test methods")
    selected = [method for method in methods
                if (method not in cases if shard.policy is None else bool(cases.get(method)))]
    if not selected:
        raise RuntimeError(f"empty provider shard: {shard.name}")
    return selected


def inventory(tests) -> Counter:
    counts = Counter()
    for test in tests:
        cases = type(test).wheel_cases() if hasattr(type(test), "wheel_cases") else {}
        parameters = cases.get(test._testMethodName, [None])
        if not parameters:
            raise RuntimeError(f"empty provider method: {test.id()}")
        for values in parameters:
            counts[(test.id(), json.dumps(values, separators=(",", ":")))] += 1
    return counts


def verify_partition(test_directory: Path, shards=SHARDS) -> dict:
    """Check actual test methods and loop cases, including duplicate test IDs."""
    names = [shard.name for shard in shards]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate provider shard names")
    if any(shard.suite not in SUITES for shard in shards):
        raise RuntimeError("unknown provider suite in shard plan")
    report = {}
    for spec in SUITES:
        module, cls = load_suite(spec, test_directory)
        loader = unittest.TestLoader()
        original = inventory(flatten(loader.loadTestsFromModule(module)))
        if loader.errors or not original:
            raise RuntimeError(f"provider discovery failed: {spec.module}: {loader.errors}")
        combined = Counter()
        for shard in shards:
            if shard.suite != spec:
                continue
            with configured(cls, shard):
                methods = selected_methods(cls, shard)
                counts = inventory(cls(method) for method in methods)
                combined.update(counts)
                report[shard.name] = {
                    "module": spec.module, "methods": methods,
                    "cases": [{"test": key[0], "parameters": json.loads(key[1]), "count": count}
                              for key, count in sorted(counts.items())],
                }
        if original != combined:
            raise RuntimeError(
                f"provider case inventory differs for {spec.module}: "
                f"missing={dict(original - combined)}, extra={dict(combined - original)}"
            )
    print(f"wheel-provider inventory: {len(report)} shards / "
          f"{sum(c['count'] for shard in report.values() for c in shard['cases'])} cases; "
          "case-ID Counter matches full discovery", flush=True)
    return report


def run_shard(name: str, test_directory: Path = Path("tests/python")) -> bool:
    shard = next(shard for shard in (*SHARDS, *COMBINED_SHARDS) if shard.name == name)
    _, cls = load_suite(shard.suite, test_directory)
    original = os.environ.get("PRICING_TEST_ORIGINAL_EVIDENCE")
    if shard in COMBINED_SHARDS:
        os.environ["PRICING_TEST_ORIGINAL_EVIDENCE"] = "1"
    try:
        with configured(cls, shard):
            methods = selected_methods(cls, shard)
            tests = unittest.TestSuite(cls(method) for method in methods)
            return unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful()
    finally:
        if original is None:
            os.environ.pop("PRICING_TEST_ORIGINAL_EVIDENCE", None)
        else:
            os.environ["PRICING_TEST_ORIGINAL_EVIDENCE"] = original


def ci_matrix() -> dict:
    return {"include": [
        {"os": os_name, "target": target, "selection": selection,
         "option": "--group" if selection in REMAINING_GROUPS else "--shard"}
        for os_name, target in (
            ("ubuntu-24.04", "x86_64-unknown-linux-gnu"),
            ("macos-15", "aarch64-apple-darwin"),
            ("windows-2025", "x86_64-pc-windows-msvc"),
        )
        for selection in (*(shard.name for shard in SHARDS), *REMAINING_GROUPS)
    ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run", choices=SHARD_NAMES)
    choice.add_argument("--github-matrix", action="store_true")
    args = parser.parse_args()
    if args.github_matrix:
        matrix = json.dumps(ci_matrix(), separators=(",", ":"))
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
            output.write(f"matrix={matrix}\n")
    else:
        # -X faulthandler covers ABRT/SEGV/BUS/ILL/FPE, but not the TRAP
        # observed in macOS provider validation. Keep the default fatal action
        # after dumping Python stacks; never turn a native crash into a pass.
        if sys.platform != "win32" and hasattr(signal, "SIGTRAP"):
            faulthandler.register(signal.SIGTRAP, all_threads=True, chain=True)
        if not run_shard(args.run):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
