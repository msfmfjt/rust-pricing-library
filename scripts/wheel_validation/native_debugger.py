"""Capture native wheel failures with LLDB; a debugger exit is not a test exit.

Imported both by the wheel runner and LLDB's own embedded Python interpreter.
The latter imports lldb only inside debugger callbacks, so package checks do not
require debugger Python bindings in the wheel's environment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid


def file_path(spec) -> str | None:
    filename = spec.GetFilename()
    return str(Path(spec.GetDirectory() or "") / filename) if filename else None


def stack_snapshot(thread) -> dict:
    """Read bounded frame metadata without evaluating code in the stopped child."""
    import lldb

    count = thread.GetNumFrames()
    frames = []
    for index in range(min(count, 128)):
        frame = thread.GetFrameAtIndex(index)
        entry = {"index": index, "valid": bool(frame.IsValid())}
        if entry["valid"]:
            module = frame.GetModule()
            line = frame.GetLineEntry()
            pc = frame.GetPC()
            address = frame.GetPCAddress().GetFileAddress()
            entry.update({
                "pc": None if pc == lldb.LLDB_INVALID_ADDRESS else hex(pc),
                "file_address": None if address == lldb.LLDB_INVALID_ADDRESS else hex(address),
                "function": frame.GetFunctionName() or frame.GetSymbol().GetName(),
                "module": module.GetFileSpec().GetFilename() if module.IsValid() else None,
                "module_path": file_path(module.GetFileSpec()) if module.IsValid() else None,
                "module_uuid": module.GetUUIDString() if module.IsValid() else None,
                "source": {"path": file_path(line.GetFileSpec()), "line": line.GetLine(),
                           "column": line.GetColumn()} if line.IsValid() else None,
            })
        frames.append(entry)
    return {"frame_count": count, "frames_truncated": count > len(frames), "frames": frames}


def launch(debugger, command: list[str], destination: str, request_id: str) -> None:
    """Launch literal argv once, then observe exit or a genuine native stop."""
    import lldb

    target = debugger.CreateTarget(command[0])
    if not target.IsValid():
        raise RuntimeError("LLDB could not create the wheel process target")
    debugger.SetSelectedTarget(target)
    debugger.SetAsync(False)
    info = lldb.SBLaunchInfo(command[1:])
    # CreateTarget resolves symlinks. Retain the venv executable path and argv0
    # explicitly so Python discovers the same environment as a direct launch.
    info.SetExecutableFile(lldb.SBFileSpec(command[0]), True)
    info.SetWorkingDirectory(os.getcwd())
    info.SetEnvironmentEntries([f"{key}={value}" for key, value in os.environ.items()], False)
    info.SetLaunchFlags(lldb.eLaunchFlagDebug)
    info.SetShellExpandArguments(False)
    info.SetDetachOnError(False)
    error = lldb.SBError()
    process = target.Launch(info, error)
    # SBTarget.Launch waits synchronously. Retain any output not already emitted
    # by LLDB's IO handler, including Python's fault-handler diagnostics.
    for read in (process.GetSTDOUT, process.GetSTDERR):
        while output := read(4096):
            print(output, end="", flush=True)
    if error.Fail():
        message = f"LLDB could not launch the wheel process: {error.GetCString()}"
        print(message, flush=True)
        raise RuntimeError(message)
    capture(debugger, destination, request_id)


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
            observation = {
                "id": thread.GetThreadID(), "name": thread.GetName(),
                "stop_reason": thread.GetStopReason(),
                "stop_description": thread.GetStopDescription(4096),
                "stop_data": [thread.GetStopReasonDataAtIndex(i)
                              for i in range(thread.GetStopReasonDataCount())],
            }
            try:
                observation.update(stack_snapshot(thread))
            except Exception as error:
                # A damaged/unavailable stack must not discard the original
                # stop reason or prevent the existing text diagnostics.
                observation["frames_error"] = f"{type(error).__name__}: {error}"
            record["threads"].append(observation)
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
    # LLDB's CLI turns trailing argv into commands and evaluates backticks even
    # when shell expansion is disabled. Transport only hex into that parser;
    # the embedded Python API receives the original strings as data.
    initialize = (
        "import importlib.util; "
        f"s = importlib.util.spec_from_file_location('wheel_native_debugger', {module_path!r}); "
        "wheel_native_debugger = importlib.util.module_from_spec(s); "
        "s.loader.exec_module(wheel_native_debugger); "
        f"wheel_native_debugger.launch(lldb.debugger, {command!r}, "
        f"{str(result_path)!r}, {request_id!r})"
    )
    initialize = f"script exec(bytes.fromhex('{initialize.encode('utf-8').hex()}').decode('utf-8'))"
    arguments = [
        executable, "--batch", "--no-lldbinit",
        "-o", "settings set auto-confirm true",
        "-o", "settings set target.detach-on-error false",
        # macOS framework Python execs its actual interpreter during startup.
        # Do not stop on that handoff; fatal signal/exception stops still fail.
        "-o", "settings set target.process.stop-on-exec false",
        "-o", initialize, "-o", "quit",
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
