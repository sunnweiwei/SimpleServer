from __future__ import annotations

"""Flask server to execute SWE‑Bench tool calls **and** evaluate patches.

Key features
============
* Default workdir `/testbed` (override via POST body).
* Jupyter‑style mixed execution (`!shell` + Python).
* `apply_patch` support.
* Multi‑call execution for every `function_call`.
* Clean stack‑traces (server frames stripped).
* `/diff` endpoint – returns cleaned patch against a base commit.
* **NEW** `/evaluate` endpoint – apply a model patch, run an eval script, and
  return a structured evaluation report similar to SWE‑Bench’s.
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
# Patch helpers
###############################################################################

def _remove_binary_diffs(patch_text: str) -> str:
    lines, cleaned, block, binary = patch_text.splitlines(), [], [], False
    for ln in lines:
        if ln.startswith('diff --git '):
            if block and not binary:
                cleaned.extend(block)
            block, binary = [ln], False
        elif 'Binary files' in ln:
            binary = True
            block.append(ln)
        else:
            block.append(ln)
    if block and not binary:
        cleaned.extend(block)
    return '\n'.join(cleaned)


def get_git_patch(base_commit: str, workdir: Path = ROOT) -> str:
    """Return cleaned git diff (no binary) relative to *base_commit*."""
    cwd = workdir.resolve()

    # Remove nested .git dirs
    nested = subprocess.run('find . -type d -name .git -not -path "./.git"', shell=True, cwd=cwd, text=True, stdout=subprocess.PIPE)
    for git_dir in filter(None, nested.stdout.strip().split('\n')):
        subprocess.run(f'rm -rf "{git_dir}"', shell=True, cwd=cwd)

    # Stage all changes
    subprocess.run('git add -A', shell=True, cwd=cwd)

    # Remove obvious binaries from staging
    remove_bin_cmd = r'''
        for f in $(git status --porcelain | grep -E "^(M| M|\?\?|A| A)" | cut -c4-); do
            if [ -f "$f" ] && (file -b "$f" | grep -q "executable" || git check-attr binary "$f" | grep -q "binary: set"); then
                git rm -f "$f" 2>/dev/null || rm -f "$f"
            fi
        done
    '''.strip()
    subprocess.run(remove_bin_cmd, shell=True, cwd=cwd)

    diff = subprocess.run(f'git diff --no-color --cached {base_commit}', shell=True, cwd=cwd, text=True, stdout=subprocess.PIPE)
    return _remove_binary_diffs(diff.stdout)


###############################################################################
# Patch‑evaluation helper
###############################################################################

def _evaluate_patch(model_patch: str, eval_script: str, workdir: Path, timeout: int = 1800) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        'empty_generation': False,
        'failed_apply_patch': False,
        'error_eval': False,
        'test_timeout': False,
        'resolved': False,
        'apply_output': '',
        'eval_output': '',
    }

    if not model_patch.strip():
        report['empty_generation'] = True
        return report

    # Write patch & script to tmp
    patch_path = Path('/tmp/patch.diff')
    patch_path.write_text(model_patch)
    script_path = Path('/tmp/eval.sh')
    script_path.write_text(eval_script)
    script_path.chmod(0o755)

    cwd = workdir.resolve()

    # Apply patch (git apply then patch fallback)
    apply_cmd = (
        "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS') || "
        "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS') || "
        "echo 'APPLY_PATCH_FAIL'"
    )
    out, err, _ = _exec_shell(apply_cmd, cwd)
    apply_out = (out + err).strip()
    report['apply_output'] = apply_out

    if 'APPLY_PATCH_FAIL' in apply_out:
        report['failed_apply_patch'] = True
        return report

    # Run evaluation script
    eval_out, eval_err, exit_code = _exec_shell('/tmp/eval.sh', cwd, timeout=timeout)
    report['eval_output'] = eval_out + eval_err

    if exit_code == -9:
        report['test_timeout'] = True
        return report
    if exit_code != 0:
        report['error_eval'] = True
        return report

    # Very simple heuristic: script should print "RESOLVED" when bug fixed
    report['resolved'] = 'RESOLVED' in report['eval_output'].upper()
    return report

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
            results.append({"index": idx, "call_id": msg['call_id'], "output": "", "error": str(exc), "timed_out": False, "duration": 0.0})
            continue
        code_inp = args.get("input")
        if code_inp is None:
            results.append({"index": idx, "call_id": msg['call_id'], "output": "", "error": "'input' missing", "timed_out": False, "duration": 0.0})
            continue

        start = time.time()
        out, err, timed = _run_with_timeout(_dispatch, 60, code_inp)
        results.append({"index": idx, "call_id": msg['call_id'], "output": out, "error": err, "timed_out": timed, "duration": round(time.time() - start, 3)})

    if not results:
        return jsonify({"error": "No python function_call messages found"}), 400

    return jsonify({"results": results, "overall_duration": round(time.time() - overall_start, 3)})

@app.route('/diff', methods=['POST'])
def diff_endpoint():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json()
    base_commit = data.get('base_commit')
    if not base_commit:
        return jsonify({'error': 'base_commit missing'}), 400
    workdir = Path(data.get('dir', ROOT)).resolve()
    try:
        patch = get_git_patch(base_commit, workdir)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    return jsonify({'patch': patch})


@app.route('/evaluate', methods=['POST'])
def evaluate_endpoint():
    """Apply *model_patch*, run *eval_script*, return SWE‑Bench‑style report."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json()
    model_patch = data.get('model_patch', '')
    eval_script = data.get('eval_script', '')
    workdir = Path(data.get('dir', ROOT)).resolve()
    timeout = int(data.get('timeout', 1800))
    report = _evaluate_patch(model_patch, eval_script, workdir, timeout)
    return jsonify({'report': report})


###############################################################################
# Entrypoint
###############################################################################

if __name__ == "__main__":
    port = 4444
    app.run(host="0.0.0.0", port=port, threaded=True)
