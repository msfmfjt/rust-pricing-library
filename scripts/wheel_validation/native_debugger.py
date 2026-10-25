"""Capture native wheel failures with LLDB; a debugger exit is not a test exit.

Imported both by the wheel runner and LLDB's own embedded Python interpreter.
The latter imports lldb only inside capture(), so normal package checks do not
require debugger Python bindings in the wheel's environment.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import uuid


def capture(debugger, destination: str, request_id: str) -> None:
    """Called after normal termination or a fatal stop, without continuing it."""
    import lldb

    path = Path(destination)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("request_id") == request_id:
            return  # Keep the first terminal observation, including a crash.
    process = debugger.GetSelectedTarget().GetProcess()
    state = process.GetState()
    state_name = {
        lldb.eStateExited: "exited", lldb.eStateStopped: "stopped",
        lldb.eStateCrashed: "crashed",
    }.get(state, "incomplete")
    record = {
        "schema_version": 1, "request_id": request_id,
        "valid_process": bool(process.IsValid()), "pid": process.GetProcessID(),
        "state": state_name, "lldb_state": state,
        "exit_code": process.GetExitStatus() if state_name == "exited" else None,
        "exit_description": process.GetExitDescription(),
        "threads": [],
    }
    if state_name != "exited":
        for thread in process:
            record["threads"].append({
                "id": thread.GetThreadID(), "name": thread.GetName(),
                "stop_reason": thread.GetStopReason(),
                "stop_description": thread.GetStopDescription(4096),
                "stop_data": [thread.GetStopReasonDataAtIndex(i)
                              for i in range(thread.GetStopReasonDataCount())],
            })
        record["diagnostics"] = {}
        for command in ("process status", "thread backtrace all", "register read", "image list"):
            result = lldb.SBCommandReturnObject()
            debugger.GetCommandInterpreter().HandleCommand(command, result)
            print(f"wheel-native {command}\n{result.GetOutput() or ''}{result.GetError() or ''}",
                  flush=True)
            record["diagnostics"][command] = bool(result.Succeeded())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def outcome(record: object, debugger_returncode: int, request_id: str) -> int:
    """Only a current, valid, normally exited child can pass the gate."""
    if not isinstance(record, dict) or (
        type(record.get("schema_version")) is not int or record["schema_version"] != 1
        or record.get("request_id") != request_id
        or record.get("valid_process") is not True
        or type(record.get("pid")) is not int or record["pid"] <= 0
        or record.get("state") != "exited"
        or type(record.get("exit_code")) is not int
        or not 0 <= record["exit_code"] <= 255
    ):
        return 1
    return record["exit_code"] or (1 if debugger_returncode else 0)


def run(command: list[str], report_directory: Path, name: str) -> tuple[int, dict]:
    executable = shutil.which("lldb")
    if executable is None:
        raise OSError("LLDB is required for --native-debugger lldb")
    if not name or Path(name).name != name:
        raise ValueError("native diagnostic name must be a filename")
    directory = report_directory.resolve() / "native-debugger"
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / f"{name}.json"
    log_path = directory / f"{name}.log"
    request_id = uuid.uuid4().hex
    module_path = str(Path(__file__).resolve())
    # Python literals protect paths inside LLDB's embedded interpreter. Launch
    # the original argv without a shell or LLDB's shell-expanding `run` alias.
    initialize = (
        "script import importlib.util; "
        f"s = importlib.util.spec_from_file_location('wheel_native_debugger', {module_path!r}); "
        "wheel_native_debugger = importlib.util.module_from_spec(s); "
        "s.loader.exec_module(wheel_native_debugger)"
    )
    finish = f"script wheel_native_debugger.capture(lldb.debugger, {str(result_path)!r}, {request_id!r})"
    arguments = [
        executable, "--batch", "--no-lldbinit", "-o", initialize,
        "-o", "settings set auto-confirm true",
        "-o", "settings set target.detach-on-error false",
        "-o", "settings set target.arg0 " + json.dumps(command[0], ensure_ascii=False),
        "-o", "process launch --disable-aslr false --shell-expand-args false",
        "-o", finish, "-o", "quit", "-k", finish, "-k", "quit", "--", *command,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace") as child:
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            debugger_returncode = child.wait()
    metadata = {"debugger_returncode": debugger_returncode,
                "log": str(log_path), "result": str(result_path)}
    try:
        record = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        metadata["diagnostic_error"] = str(error)
        return 1, metadata
    metadata["process"] = record
    return outcome(record, debugger_returncode, request_id), metadata
