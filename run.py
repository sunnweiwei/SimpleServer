from __future__ import annotations

"""Flask server for executing **all** SWE‑Bench tool calls with /testbed root.

Highlights
==========
* **Default working dir** → `/testbed` for shell, Python, and `apply_patch`.
* **Python execution** – stateful globals across requests.
* **Shell commands**   – run each `!cmd` line via subprocess in `/testbed`.
* **Mixed cells**      – Jupyter‑style interleaving of `!` and Python.
* **apply_patch**      – V4A patches applied relative to `/testbed`.
* **Multi‑call support** – sequentially executes every `function_call`.
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

import apply_patch  # SWE‑Bench helper

###############################################################################
# Constants & environment
###############################################################################

ROOT = Path("/testbed").resolve()
os.chdir(ROOT)  # Ensure server starts in /testbed

###############################################################################
# Flask setup
###############################################################################

app = Flask(__name__)

###############################################################################
# Global Python execution context (stateful across calls & cells)
###############################################################################

_EXEC_GLOBALS: Dict[str, Any] = {
    "__name__": "__main__",
    "__file__": "<agent>",
}

###############################################################################
# Utility helpers
###############################################################################

def _run_with_timeout(fn, timeout: int, *args, **kwargs) -> Tuple[str, str, bool]:
    out, err = {}, {}

    def _target():
        try:
            o, e = fn(*args, **kwargs)
            out[0], err[0] = o, e
        except Exception:  # pylint: disable=broad-except
            err[0] = traceback.format_exc()

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return "", f"Timed out after {timeout}s", True
    return out.get(0, ""), err.get(0, ""), False


def _exec_python(src: str) -> Tuple[str, str]:
    # Guarantee we are inside /testbed for any relative paths
    os.chdir(ROOT)
    out_io, err_io = io.StringIO(), io.StringIO()
    with redirect_stdout(out_io), redirect_stderr(err_io):
        exec(src, _EXEC_GLOBALS)  # nosec – trusted agent code
    return out_io.getvalue(), err_io.getvalue()


def _exec_shell(cmd: str) -> Tuple[str, str]:
    proc = subprocess.Popen(
        cmd,
        shell=True,
        text=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        stderr = f"[exit {proc.returncode}] {stderr}"
    return stdout, stderr


def _exec_apply_patch(block: str) -> Tuple[str, str]:
    # Ensure cwd for file operations
    os.chdir(ROOT)
    try:
        res = apply_patch.process_patch(
            block,
            apply_patch.open_file,
            apply_patch.write_file,
            apply_patch.remove_file,
        )
        return res + "\n", ""
    except Exception:  # pylint: disable=broad-except
        return "", traceback.format_exc()

###############################################################################
# Dispatcher – handles !shell, python, and apply_patch in a single cell
###############################################################################

def _dispatch(cell: str) -> Tuple[str, str]:
    cell = textwrap.dedent(cell)

    # ---- apply_patch block ----
    if cell.lstrip().startswith("%%bash") and "apply_patch" in cell:
        patch_lines: List[str] = []
        capture = False
        for ln in cell.splitlines():
            if "apply_patch" in ln:
                capture = False
            if "<<\"EOF\"" in ln or ln.strip().endswith("<<EOF"):
                capture = True
                continue
            if ln.strip() == "EOF" and capture:
                break
            if capture:
                patch_lines.append(ln)
        return _exec_apply_patch("\n".join(patch_lines))

    # ---- Mixed Jupyter‑style cell ----
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
# Routes
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
            results.append({"index": idx, "output": "", "error": f"Bad JSON: {exc}", "timed_out": False, "duration": 0.0})
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
