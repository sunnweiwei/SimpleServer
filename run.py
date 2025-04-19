from __future__ import annotations

"""Flask server to execute SWE‑Bench tool calls with clean error traces.

Key points
==========
* **Working directory** – all actions run relative to `/testbed`.
* **Mixed cell execution** – Jupyter‑style `!shell` + Python.
* **apply_patch** – V4A patches applied with relative paths.
* **Multi‑call support** – executes every `function_call`.
* **Clean errors** – stack traces stripped of server internals; only user‑relevant
  frames (those in `<string>` or under `/testbed`) are returned.
"""

import io
import json
import os
import subprocess
import textwrap
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request

import apply_patch

###############################################################################
# Environment & constants
###############################################################################

ROOT = Path("/testbed").resolve()
os.chdir(ROOT)
SERVER_FILE = Path(__file__).resolve().as_posix()

###############################################################################
# Flask setup
###############################################################################

app = Flask(__name__)

###############################################################################
# Shared Python globals
###############################################################################

_EXEC_GLOBALS: Dict[str, Any] = {
    "__name__": "__main__",
    "__file__": "<agent>",
}

###############################################################################
# Error‑trace sanitiser
###############################################################################

def _clean_trace(tb: str) -> str:
    """Remove frames originating from the server implementation."""
    cleaned: List[str] = []
    for line in tb.splitlines():
        if SERVER_FILE in line or "/flask/" in line or "/runpy.py" in line:
            continue  # strip server & framework frames
        cleaned.append(line)
    return "\n".join(cleaned)

###############################################################################
# Low‑level executors
###############################################################################

def _exec_python(src: str) -> Tuple[str, str]:
    os.chdir(ROOT)
    out_io, err_io = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out_io), redirect_stderr(err_io):
            exec(src, _EXEC_GLOBALS)  # nosec – trusted agent code
    except Exception:  # pylint: disable=broad-except
        err_io.write(_clean_trace(traceback.format_exc()))
    return out_io.getvalue(), err_io.getvalue()


def _exec_shell(cmd: str) -> Tuple[str, str]:
    proc = subprocess.Popen(cmd, shell=True, text=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        stderr = _clean_trace(stderr or f"Command exited with {proc.returncode}")
    return stdout, stderr


def _exec_apply_patch(block: str) -> Tuple[str, str]:
    os.chdir(ROOT)
    try:
        res = apply_patch.process_patch(block, apply_patch.open_file, apply_patch.write_file, apply_patch.remove_file)
        return res + "\n", ""
    except Exception:  # pylint: disable=broad-except
        return "", _clean_trace(traceback.format_exc())

###############################################################################
# Timeout wrapper
###############################################################################

def _run_with_timeout(fn, timeout: int, *args, **kwargs) -> Tuple[str, str, bool]:
    out, err = {}, {}

    def _target():
        try:
            o, e = fn(*args, **kwargs)
            out[0], err[0] = o, e
        except Exception:  # pylint: disable=broad-except
            err[0] = _clean_trace(traceback.format_exc())

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return "", "Timed out", True
    return out.get(0, ""), err.get(0, ""), False

###############################################################################
# Cell dispatcher
###############################################################################

def _dispatch(cell: str) -> Tuple[str, str]:
    cell = textwrap.dedent(cell)

    # Handle apply_patch cell
    if cell.lstrip().startswith("%%bash") and "apply_patch" in cell:
        lines = cell.splitlines()
        patch_lines: List[str] = []
        cap = False
        for ln in lines:
            if "apply_patch" in ln:
                cap = False
            if "<<\"EOF\"" in ln or ln.strip().endswith("<<EOF"):
                cap = True
                continue
            if ln.strip() == "EOF" and cap:
                break
            if cap:
                patch_lines.append(ln)
        return _exec_apply_patch("\n".join(patch_lines))

    py_buf: List[str] = []
    outs, errs = [], []

    def flush_py():
        if py_buf:
            o, e = _exec_python("\n".join(py_buf))
            outs.append(o)
            errs.append(e)
            py_buf.clear()

    for ln in cell.splitlines():
        if ln.lstrip().startswith("!"):
            flush_py()
            o, e = _exec_shell(ln.lstrip()[1:].lstrip())
            outs.append(o)
            errs.append(e)
        else:
            py_buf.append(ln)
    flush_py()
    return "".join(outs), "".join(errs)

###############################################################################
# Flask routes
###############################################################################

@app.route("/alive", methods=["GET"])
def alive():
    return "ok", 200


@app.route("/execute", methods=["POST"])
def execute():
    overall_start = time.time()
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    payload = request.get_json()
    if not isinstance(payload, list) or not payload:
        return jsonify({"error": "Expected a non‑empty list of messages"}), 400

    results = []
    for idx, msg in enumerate(payload):
        if msg.get("type") != "function_call" or msg.get("name") != "python":
            continue
        try:
            args = json.loads(msg.get("arguments", "{}"))
        except json.JSONDecodeError as exc:
            results.append({"index": idx, "output": "", "error": str(exc), "timed_out": False, "duration": 0.0})
            continue
        code_inp = args.get("input")
        if code_inp is None:
            results.append({"index": idx, "output": "", "error": "'input' missing", "timed_out": False, "duration": 0.0})
            continue

        start = time.time()
        out, err, timed = _run_with_timeout(_dispatch, 60, code_inp)
        results.append({"index": idx, "output": out, "error": err, "timed_out": timed, "duration": round(time.time() - start, 3)})

    if not results:
        return jsonify({"error": "No python function_call messages found"}), 400

    return jsonify({"results": results, "overall_duration": round(time.time() - overall_start, 3)})

###############################################################################
# Entrypoint
###############################################################################

if __name__ == "__main__":
    port = 4444
    app.run(host="0.0.0.0", port=port, threaded=True)
