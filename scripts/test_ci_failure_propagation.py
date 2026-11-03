"""Exercise native failure propagation through the actual CI command blocks."""
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import textwrap
import unittest

from check_source_archive import check_ci_workflow, ci_job_text


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
REUSABLE_WORKFLOW = (ROOT / ".github/workflows/extended-acceptance.yml").read_text(encoding="utf-8")
BLOCKS = {
    "rust": (
        "Show pinned toolchain",
        "Validate pricing source compositions against JSON Schema",
        "Check default and minimal feature configurations",
        "Run tests",
        "Run statistical acceptance",
    ),
    "supported-wheels": ("Build private wheel", "Check replay fixtures"),
}


def validate_archive(workflow, reusable=REUSABLE_WORKFLOW):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, source in (("ci.yml", workflow), ("extended-acceptance.yml", reusable)):
            content = source.encode("utf-8")
            member = tarfile.TarInfo(".github/workflows/" + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode="r") as archive:
        check_ci_workflow(archive, "failure-probe")


def command_block(job, name):
    match = re.search(
        rf"^      - name: {re.escape(name)}\n        run: \|\n((?:          [^\n]*\n)+)",
        ci_job_text(WORKFLOW, job), re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"missing command block: {job}/{name}")
    return textwrap.dedent(match.group(1))


def bash_executable():
    # On Windows use Git's bash, as Actions does, rather than a WSL launcher.
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            directory = Path(git).resolve().parent
            # PATH may expose Git's cmd/, bin/, or mingw64/bin/ entry point.
            for root in (directory.parent, directory.parent.parent):
                candidate = root / "bin/bash.exe"
                if candidate.is_file():
                    return str(candidate)
        raise AssertionError("Git for Windows bash is required for CI failure probes")
    bash = shutil.which("bash")
    if bash is None:
        raise AssertionError("bash is required for CI failure probes")
    return bash


def run_block(source):
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "probe.sh"
        script.write_text(source, encoding="utf-8", newline="\n")
        # This is the documented Actions `shell: bash` invocation. The tested
        # commands are shell functions; only the deliberate native exit is real.
        return subprocess.run(
            [bash_executable(), "--noprofile", "--norc", "-eo", "pipefail", script.as_posix()],
            capture_output=True, text=True, check=False, timeout=30,
        )


class CiFailurePropagationTests(unittest.TestCase):
    def test_archive_rejects_missing_failure_guards(self):
        validate_archive(WORKFLOW)
        for job_name in BLOCKS:
            job = ci_job_text(WORKFLOW, job_name)
            for index, changed in enumerate((
                job.replace("    defaults:\n      run:\n        shell: bash\n", ""),
                job.replace("        shell: bash", "        shell: pwsh"),
                job.replace("        run:", "        shell: pwsh\n        run:", 1),
                job.replace("        run:", "        continue-on-error: true\n        run:", 1),
            )):
                with self.subTest(job=job_name, mutation=index), self.assertRaises(SystemExit):
                    validate_archive(WORKFLOW.replace(job, changed))

    def test_every_command_failure_stops_its_actual_ci_block(self):
        for job, names in BLOCKS.items():
            for name in names:
                source = command_block(job, name)
                commands = source.strip().splitlines()
                for fail_at in range(len(commands) + 1):
                    with self.subTest(job=job, step=name, fail_at=fail_at):
                        prelude = f"probe_calls=0\nprobe_fail_at={fail_at}\n" + r'''
probe() {
    probe_calls=$((probe_calls + 1))
    printf 'command-%s\n' "$probe_calls"
    if [ "$probe_calls" -eq "$probe_fail_at" ]; then
        command python -c 'import sys; sys.exit(23)'
    fi
}
cargo() { probe; }
python() { probe; }
rustc() { probe; }
rustup() { probe; }
git() { probe; }
'''
                        result = run_block(prelude + source)
                        count = fail_at or len(commands)
                        self.assertEqual(result.returncode, 23 if fail_at else 0, result.stderr)
                        self.assertEqual(result.stdout.splitlines(),
                                         [f"command-{i}" for i in range(1, count + 1)])

    def test_native_failure_in_pipeline_is_not_masked_by_successful_consumer(self):
        # The dependency-direction command is a real CI pipeline. Both ends
        # fail independently; successful JSON/checker execution cannot hide cargo.
        pipeline = "cargo metadata --locked --format-version 1 --no-deps | python scripts/check_dependency_direction.py"
        self.assertIn("        run: " + pipeline, ci_job_text(WORKFLOW, "rust"))
        for producer, consumer in ((0, 0), (23, 0), (0, 29)):
            with self.subTest(producer=producer, consumer=consumer):
                source = (
                    f"cargo() {{ command python -c 'import sys; sys.exit({producer})'; }}\n"
                    f"python() {{ command python -c 'import sys; sys.stdin.read(); sys.exit({consumer})'; }}\n"
                    + pipeline + "\nprintf 'continued\\n'\n"
                )
                result = run_block(source)
                self.assertEqual(result.returncode, consumer or producer, result.stderr)
                self.assertEqual(result.stdout, "" if producer or consumer else "continued\n")


if __name__ == "__main__":
    unittest.main()
