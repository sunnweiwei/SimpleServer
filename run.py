from __future__ import annotations

"""Flask server for SWE‑Bench automation.

Highlights
==========
* Workdir defaults to `/testbed` (override per‑request).
* Execute mixed Python / `!shell` cells (`/execute`).
* Generate cleaned git diff (`/diff`).
* Apply patch & run evaluation script (`/evaluate`).
* **NEW** Run arbitrary shell command (`/command`).
* All stack traces cleaned of server internals.
"""

import io
import json
import os
import subprocess
import tempfile
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
# Environment / constants
###############################################################################

DEFAULT_ROOT = Path("/testbed").resolve()
SERVER_FILE = Path(__file__).resolve().as_posix()

###############################################################################
# Flask setup
###############################################################################

app = Flask(__name__)

###############################################################################
# Shared Python execution context
###############################################################################

_EXEC_GLOBALS: Dict[str, Any] = {
    "__name__": "__main__",
    "__file__": "<agent>",
}

###############################################################################
# Utility – error‑trace cleaner
###############################################################################

def _clean_trace(tb: str) -> str:
    return "\n".join(
        ln for ln in tb.splitlines()
        if SERVER_FILE not in ln and "/flask/" not in ln and "/runpy.py" not in ln
    )

###############################################################################
# Low‑level executors
###############################################################################

def _exec_python(src: str, cwd: Path) -> Tuple[str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            os.chdir(cwd)
            exec(src, _EXEC_GLOBALS)  # nosec B102
    except Exception:
        stderr.write(_clean_trace(traceback.format_exc()))
    return stdout.getvalue(), stderr.getvalue()


def _exec_shell(cmd: str, cwd: Path, timeout: int | None = None) -> Tuple[str, str, int]:
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return "", "Timed out", -9
    if proc.returncode != 0:
        err = _clean_trace(err or f"Command exited with {proc.returncode}")
    return out, err, proc.returncode


def _exec_apply_patch(block: str, cwd: Path) -> Tuple[str, str]:
    os.chdir(cwd)
    try:
        res = apply_patch.process_patch(block, apply_patch.open_file, apply_patch.write_file, apply_patch.remove_file)
        return res + "\n", ""
    except Exception:
        return "", _clean_trace(traceback.format_exc())

###############################################################################
# Timeout wrapper for dispatch cells
###############################################################################

def _run_with_timeout(fn, timeout: int, *args, **kwargs) -> Tuple[str, str, bool]:
    result, err = {}, {}
    def _target():
        try:
            o, e = fn(*args, **kwargs)
            result[0], err[0] = o, e
        except Exception:
            err[0] = _clean_trace(traceback.format_exc())
    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return "", "Timed out", True
    return result.get(0, ""), err.get(0, ""), False

###############################################################################
# Mixed‑cell dispatcher
###############################################################################

def _dispatch(cell: str, cwd: Path) -> Tuple[str, str]:
    cell = textwrap.dedent(cell)

    # apply_patch block
    if cell.lstrip().startswith("%%bash") and "apply_patch" in cell:
        patch_lines, cap = [], False
        for ln in cell.splitlines():
            if "apply_patch" in ln:
                cap = False
            if "<<\"EOF\"" in ln or ln.strip().endswith("<<EOF"):
                cap = True
                continue
            if ln.strip() == "EOF" and cap:
                break
            if cap:
                patch_lines.append(ln)
        return _exec_apply_patch("\n".join(patch_lines), cwd)

    py_buf, outs, errs = [], [], []

    def flush_py():
        if py_buf:
            o, e = _exec_python("\n".join(py_buf), cwd)
            outs.append(o)
            errs.append(e)
            py_buf.clear()

    for ln in cell.splitlines():
        if ln.lstrip().startswith("!"):
            flush_py()
            o, e, _ = _exec_shell(ln.lstrip()[1:].lstrip(), cwd)
            outs.append(o)
            errs.append(e)
        else:
            py_buf.append(ln)
    flush_py()
    return "".join(outs), "".join(errs)

###############################################################################
# Git‑patch helpers (unchanged)
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


def get_git_patch(base_commit: str, workdir: Path = DEFAULT_ROOT) -> str:
    cwd = workdir.resolve()
    subprocess.run('find . -type d -name .git -not -path "./.git" -exec rm -rf {} +', shell=True, cwd=cwd)
    subprocess.run('git add -A', shell=True, cwd=cwd)
    remove_bin_cmd = r'''for f in $(git status --porcelain | grep -E "^(M| M|\?\?|A| A)" | cut -c4-); do if [ -f "$f" ] && (file -b "$f" | grep -q "executable" || git check-attr binary "$f" | grep -q "binary: set"); then git rm -f "$f" 2>/dev/null || rm -f "$f"; fi; done'''
    subprocess.run(remove_bin_cmd, shell=True, cwd=cwd)
    diff = subprocess.run(f'git diff --no-color --cached {base_commit}', shell=True, cwd=cwd, text=True, stdout=subprocess.PIPE)
    return _remove_binary_diffs(diff.stdout)

###############################################################################
# Patch‑evaluation helper (unchanged)
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
    patch_path = Path('/tmp/patch.diff'); patch_path.write_text(model_patch)
    script_path = Path('/tmp/eval.sh'); script_path.write_text(eval_script); script_path.chmod(0o755)
    cwd = workdir.resolve()
    apply_cmd = "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS') || (patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS') || echo 'APPLY_PATCH_FAIL'"
    out, err, _ = _exec_shell(apply_cmd, cwd)
    apply_out = (out + err).strip(); report['apply_output'] = apply_out
    if 'APPLY_PATCH_FAIL' in apply_out:
        report['failed_apply_patch'] = True
        return report
    eval_out, eval_err, rc = _exec_shell('/tmp/eval.sh', cwd, timeout=timeout)
    report['eval_output'] = eval_out + eval_err
    if rc == -9:
        report['test_timeout'] = True
        return report
    if rc != 0:
        report['error_eval'] = True
        return report
    report['resolved'] = 'RESOLVED' in report['eval_output'].upper()
    return report

###############################################################################
# Flask routes
###############################################################################

@app.route('/alive', methods=['GET'])
def alive():
    return 'ok', 200


@app.route('/execute', methods=['POST'])
def execute_endpoint():
    start = time.time()
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    payload = request.get_json()
    if not isinstance(payload, list) or not payload:
        return jsonify({'error': 'Expected non‑empty list'}), 400
    results = []
    for idx, msg in enumerate(payload):
        if msg.get('type') != 'function_call' or msg.get('name') != 'python':
            continue
        try:
            args = json.loads(msg.get('arguments', '{}'))
        except json.JSONDecodeError as exc:
            results.append({'index': idx, 'call_id': msg['call_id'], 'output': '', 'error': str(exc), 'timed_out': False, 'duration': 0.0})
            continue
        code = args.get('input');
        if code is None:
            results.append({'index': idx, 'call_id': msg['call_id'], 'output': '', 'error': "'input' missing", 'timed_out': False, 'duration': 0.0}); continue
        o_start = time.time()
        out, err, timed = _run_with_timeout(_dispatch, 60, code, DEFAULT_ROOT)
        results.append({'index': idx, 'call_id': msg['call_id'], 'output': out, 'error': err, 'timed_out': timed, 'duration': round(time.time() - o_start, 3)})
    if not results:
        return jsonify({'error': 'No python function_call messages found'}), 400
    return jsonify({'results': results, 'overall_duration': round(time.time() - start, 3)})


@app.route('/diff', methods=['POST'])
def diff_endpoint():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json(); base_commit = data.get('base_commit')
    if not base_commit:
        return jsonify({'error': 'base_commit missing'}), 400
    workdir = Path(data.get('dir', DEFAULT_ROOT)).resolve()
    try:
        patch = get_git_patch(base_commit, workdir)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    return jsonify({'patch': patch})


@app.route('/evaluate', methods=['POST'])
def evaluate_endpoint():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json()
    model_patch = data.get('model_patch', '')
    eval_script = data.get('eval_script', '')
    workdir = Path(data.get('dir', DEFAULT_ROOT)).resolve()
    timeout = int(data.get('timeout', 1800))
    report = _evaluate_patch(model_patch, eval_script, workdir, timeout)
    return jsonify({'report': report})


# ────────────────────────────────────────────────────────────────────────────
# NEW: /command – run an arbitrary shell command
# --------------------------------------------------------------------------

@app.route('/command', methods=['POST'])
def command_endpoint():
    """Run a single shell command and return stdout/stderr/rc."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json()
    cmd = data.get('cmd') or data.get('command')
    if not cmd:
        return jsonify({'error': 'cmd missing'}), 400
    workdir = Path(data.get('dir', DEFAULT_ROOT)).resolve()
    timeout = int(data.get('timeout', 120))
    start = time.time()
    out, err, rc = _exec_shell(cmd, workdir, timeout=timeout)
    return jsonify({
        'stdout': out,
        'stderr': err,
        'returncode': rc,
        'duration': round(time.time() - start, 3)
    })

###############################################################################
# Entrypoint
###############################################################################

if __name__ == '__main__':
    port = 4444
    app.run(host='0.0.0.0', port=port, threaded=True)
