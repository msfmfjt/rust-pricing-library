#!/bin/bash
set -euo pipefail
work="$GITHUB_WORKSPACE"
kit="$work/diagnostics/tools/pr206"
out="$RUNNER_TEMP/individual-history"
cd "$work/pr"
test "$(git rev-parse HEAD)" = d35cf226e2989482f8a2fec568ae85e02a6b973a
test "$(sw_vers -buildVersion)" = 24H232
test "$(sw_vers -productVersion)" = 15.8.2
test "$(uname -m)" = arm64
sw_vers
lldb --version
wheel=dist/rust_pricing-0.1.0-cp312-cp312-macosx_11_0_arm64.whl
python - "$wheel" <<'PY'
import hashlib,platform,sys,zipfile
from pathlib import Path
p=Path(sys.argv[1])
assert platform.python_version()=='3.12.10'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='67aefb5dfcef28a88f60ddeff4d52a38654d1702516ef13bdc729b97f0089c8a'
with zipfile.ZipFile(p) as z:
    lib,=(n for n in z.namelist() if n.endswith('.so'))
    assert hashlib.sha256(z.read(lib)).hexdigest()=='86fd537769196736e298a7c4a029ddcbe66e57a43aa2b94ea5ca09e59bebb5e4'
PY
export PIP_CONSTRAINT="$kit/requirements.txt"
python -m pip install -q -r "$PIP_CONSTRAINT" "$wheel"
python -m unittest discover -s scripts -p test_native_debugger.py -v
python "$kit/run.py" "$SELECTION" "$out"
