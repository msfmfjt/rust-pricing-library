"""Diagnose the two retained macOS failures; never replace production wheel gates."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.smoke_test_wheel import run_test_process, run_validation

SHARDS = ("custom-basis-row_hats-4-call-fixed_lookback", "extension-all_rows-4")
ASAN_OPTIONS = "detect_leaks=0:abort_on_error=1"
# This standalone diagnostic deliberately contains a fault. It is never linked
# into the pricing library; both its normal and faulting exits must be verified.
PROBE_SOURCE = """
fn main() {
    let allocation = std::hint::black_box(Box::new([7_u8; 64]));
    let pointer = std::hint::black_box(allocation.as_ptr());
    if std::env::args().any(|arg| arg == "--fault") {
        drop(allocation);
        println!("{}", unsafe { std::ptr::read_volatile(pointer) });
    } else {
        println!("{}", unsafe { std::ptr::read_volatile(pointer) });
        drop(allocation);
    }
}
"""
PYTHON_ALLOCATOR_MARKER = "ASan Python allocator probe ready"
PYTHON_ALLOCATOR_SOURCE = """
import ctypes
import resource
import sys
import rust_pricing

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
process = ctypes.CDLL(None)
assert hasattr(process, '__asan_init')
process.malloc.argtypes = [ctypes.c_size_t]
process.malloc.restype = ctypes.c_void_p
process.free.argtypes = [ctypes.c_void_p]
process.free.restype = None
allocation = process.malloc(64)
assert allocation
process.free(allocation)
print('ASan Python allocator probe ready', rust_pricing.__file__, flush=True)
if '--fault' in sys.argv:
    # Deliberate fault in this isolated control, never in the pricing workload.
    process.free(allocation)
"""


def save(directory: Path, name: str, value: object) -> None:
    (directory / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def interpreter() -> Path:
    """Bypass framework launchers whose startup exec loses ASan interposition."""
    framework = sysconfig.get_config_var("PYTHONFRAMEWORK")
    if framework:
        executable = (Path(sysconfig.get_config_var("prefix")) / "Resources"
                      / f"{framework}.app" / "Contents/MacOS" / framework)
    else:
        executable = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    if not executable.is_file():
        raise RuntimeError(f"Python interpreter is unavailable: {executable}")
    return executable


def sanitizer_runtime() -> Path:
    directory = Path(subprocess.check_output(
        ["rustc", "--print", "target-libdir"], text=True).strip())
    runtimes = list(directory.glob("librustc*_rt.asan.dylib"))
    if len(runtimes) != 1:
        raise RuntimeError(f"expected one Rust ASan runtime, found {runtimes}")
    return runtimes[0].resolve()


def instrumented_extension() -> dict:
    distribution = metadata.distribution("rust-pricing")
    extensions = [Path(distribution.locate_file(name)) for name in distribution.files or ()
                  if Path(name).suffix == ".so"]
    if len(extensions) != 1:
        raise RuntimeError("expected one installed pricing extension")
    extension = extensions[0]
    symbols = subprocess.check_output(["nm", "-u", str(extension)], text=True)
    if "___asan_report_load" not in symbols or "___asan_report_store" not in symbols:
        raise RuntimeError("the installed pricing extension is not ASan-instrumented")
    return {"path": str(extension), "version": distribution.version,
            "sha256": hashlib.sha256(extension.read_bytes()).hexdigest()}


def verify_detector(directory: Path) -> None:
    source, executable = directory / "asan-probe.rs", directory / "asan-probe"
    source.write_text(PROBE_SOURCE, encoding="utf-8")
    env = dict(os.environ, RUSTC_BOOTSTRAP="1", ASAN_OPTIONS=ASAN_OPTIONS)
    subprocess.run(["rustc", "-Zsanitizer=address", "-Cdebuginfo=1", str(source),
                    "-o", str(executable)], env=env, check=True)
    results = []
    for label, arguments in (("normal", []), ("fault", ["--fault"])):
        result = subprocess.run([str(executable), *arguments], env=env,
                                capture_output=True, text=True)
        (directory / f"asan-probe-{label}.log").write_text(
            result.stdout + result.stderr, encoding="utf-8")
        results.append({"case": label, "exit_code": result.returncode,
                        "heap_use_after_free": "AddressSanitizer: heap-use-after-free" in result.stderr})
    save(directory, "asan-probe.json", results)
    if (results[0]["exit_code"] != 0 or results[0]["heap_use_after_free"]
            or results[1]["exit_code"] == 0 or not results[1]["heap_use_after_free"]):
        raise RuntimeError("ASan did not distinguish the normal and deliberate faulty probe")


def verify_python_allocator(directory: Path, executable: Path) -> None:
    """Prove free interception in the actual Python process after wheel import."""
    source = directory / "asan-python-allocator.py"
    source.write_text(PYTHON_ALLOCATOR_SOURCE, encoding="utf-8")
    results = []
    for label, arguments in (("normal", []), ("fault", ["--fault"])):
        command = [str(executable), "-X", "faulthandler", str(source), *arguments]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired as error:
            output = (error.stdout or b""), (error.stderr or b"")
            log = "".join(part.decode("utf-8", errors="replace")
                          if isinstance(part, bytes) else part for part in output)
            (directory / f"asan-python-allocator-{label}.log").write_text(log, encoding="utf-8")
            results.append({"case": label, "status": "timed_out", "exit_code": None})
            save(directory, "asan-python-allocator.json", results)
            raise RuntimeError("ASan Python allocator control timed out") from error
        log = result.stdout + result.stderr
        (directory / f"asan-python-allocator-{label}.log").write_text(log, encoding="utf-8")
        results.append({
            "case": label, "status": "completed", "exit_code": result.returncode,
            "pricing_imported_and_first_free_completed": PYTHON_ALLOCATOR_MARKER in result.stdout,
            "double_free": "AddressSanitizer: attempting double-free" in result.stderr,
            "allocation_stack": "previously allocated by thread" in result.stderr,
            "free_stack": "freed by thread" in result.stderr,
        })
        save(directory, "asan-python-allocator.json", results)
    normal, fault = results
    if (not all(item["pricing_imported_and_first_free_completed"] for item in results)
            or normal["exit_code"] != 0 or normal["double_free"]
            or fault["exit_code"] != -signal.SIGABRT or not fault["double_free"]
            or not fault["allocation_stack"] or not fault["free_stack"]):
        raise RuntimeError("ASan Python allocator control did not prove double-free detection")


def prepare_sanitizer(directory: Path) -> Path:
    if platform.system() != "Darwin":
        raise RuntimeError("this diagnostic targets the retained macOS failures")
    extension = instrumented_extension()
    runtime = sanitizer_runtime()
    executable = interpreter()
    verify_detector(directory)
    settings = {"DYLD_INSERT_LIBRARIES": str(runtime),
                "__PYVENV_LAUNCHER__": sys.executable,
                "ASAN_OPTIONS": ASAN_OPTIONS,
                "PYTHONPATH": os.pathsep.join(str(ROOT / name)
                                              for name in ("tests/python", "backtesting", "scripts"))}
    save(directory, "asan-environment.json", {
        "source_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "host": platform.platform(), "python": sys.version, "extension": extension,
        "runtime": str(runtime), "interpreter": str(executable), "settings": settings,
        "compiler": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "limitations": "Diagnostic only. CPython, NumPy, SciPy, the standard library and OS libraries are not rebuilt with instrumentation. Leak checking is disabled; invalid-access/free checks remain enabled.",
    })
    os.environ.update(settings)
    probe = ("import ctypes, rust_pricing; "
             "assert hasattr(ctypes.CDLL(None), '__asan_init'); "
             "print('ASan pricing import passed', rust_pricing.__file__, flush=True)")
    failure = run_test_process([str(executable), "-X", "faulthandler", "-c", probe],
                               "diagnostic", "asan-import", directory)
    if failure:
        raise RuntimeError(failure)
    verify_python_allocator(directory, executable)
    return executable


def calibration_only(directory: Path) -> None:
    sys.path.insert(0, str(ROOT / "backtesting"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import numpy as np
    from multi_step_cases import model
    from rust_pricing_backtest.multi_step_extension import MultiStepExtensionCalibrator
    from rust_pricing_backtest.named_variance_basis import named_variance_basis

    if "rust_pricing" in sys.modules:
        raise RuntimeError("calibration isolation must not import native pricing")
    report = {"host": platform.platform(), "python": sys.version,
              "numpy": np.__version__, "status": "running", "evaluations": []}
    save(directory, "calibration-only.json", report)
    np.show_config()
    for cycle in range(8):
        for intervals in (4, 6):
            payload = model(intervals)
            for policy in ("row_hats", "weighted_time_hats", "coupled_row_hats"):
                basis, _ = named_variance_basis(payload, policy)
                calibrator = MultiStepExtensionCalibrator(payload, extension="custom", variance_basis=basis)
                tangent, audit = calibrator.all_maturity_iv_tangent()
                if not np.isfinite(tangent).all() or "rust_pricing" in sys.modules:
                    raise RuntimeError("calibration isolation invariants failed")
                observation = {"cycle": cycle, "intervals": intervals, "policy": policy,
                               "iv_response_residual": audit["iv_response_residual"]}
                report["evaluations"].append(observation)
                print(json.dumps(observation), flush=True)
        time.sleep(5)  # Let the BLAS workqueue become idle between batches.
        save(directory, "calibration-only.json", report)
    report.update(status="completed", native_imported=False)
    save(directory, "calibration-only.json", report)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--calibration-only", action="store_true")
    selection.add_argument("--shard", choices=SHARDS)
    parser.add_argument("--report-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    directory = args.report_directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.calibration_only:
        calibration_only(directory)
    else:
        python = prepare_sanitizer(directory)
        run_validation(python, "all", args.shard, directory)


if __name__ == "__main__":
    main()
