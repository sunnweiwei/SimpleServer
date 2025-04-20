from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import threading
import time
import traceback
import textwrap
import uuid
from code import InteractiveConsole
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request
from jupyter_client import KernelManager          # ← NEW: Jupyter kernel manager

import apply_patch

# ─────────────────────────────────────────────────────────────────────────────
#  Environment setup
# ─────────────────────────────────────────────────────────────────────────────

SANDBOX_PREFIX = Path("/opt/miniconda3/envs/testbed").resolve()
if not (SANDBOX_PREFIX / "bin").exists():
    raise RuntimeError(f"Conda env not found at {SANDBOX_PREFIX}")

proc_env = subprocess.run(
    ["/bin/bash", "-lc",
     f"source /opt/miniconda3/etc/profile.d/conda.sh && "
     f"conda activate {SANDBOX_PREFIX} && env"],
    stdout=subprocess.PIPE,
    text=True,
    check=True
)

_SANDBOX_ENV: Dict[str, str] = {}
for line in proc_env.stdout.splitlines():
    k, _, v = line.partition("=")
    _SANDBOX_ENV[k] = v
_SANDBOX_ENV["PATH"] = f"{SANDBOX_PREFIX / 'bin'}:{_SANDBOX_ENV.get('PATH', '')}"

DEFAULT_ROOT = Path("/testbed").resolve()
DEFAULT_ROOT.mkdir(parents=True, exist_ok=True)

SERVER_FILE = Path(__file__).resolve().as_posix()

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Stateful bash shell (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _start_shell() -> subprocess.Popen[str]:
    return subprocess.Popen(
        ["/bin/bash"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=DEFAULT_ROOT,
        env=_SANDBOX_ENV,
        text=True,
        bufsize=1,
    )

SHELL_PROC = _start_shell()
SHELL_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
#  Lightweight internal Python REPL (kept for helpers)
# ─────────────────────────────────────────────────────────────────────────────

_REPL_PATH = Path(tempfile.gettempdir()) / "python_repl.py"
_REPL_PATH.write_text(textwrap.dedent("""
import sys, json, io, traceback
from code import InteractiveConsole
from contextlib import redirect_stdout, redirect_stderr
console = InteractiveConsole(globals())
for raw in sys.stdin:
    try:
        msg = json.loads(raw)
        src = msg.get('code', '')
        outbuf, errbuf = io.StringIO(), io.StringIO()
        with redirect_stdout(outbuf), redirect_stderr(errbuf):
            console.runsource(src)
    except Exception:
        errbuf.write(traceback.format_exc())
    sys.stdout.write(json.dumps({'out': outbuf.getvalue(), 'err': errbuf.getvalue()}) + '\\n')
    sys.stdout.flush()
"""))

def _start_repl() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [str(SANDBOX_PREFIX / "bin/python"), "-u", str(_REPL_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=DEFAULT_ROOT,
        env=_SANDBOX_ENV,
        text=True,
        bufsize=1,
    )

_PY_LOCK = threading.Lock()
PY_REPL = _start_repl()

# ─────────────────────────────────────────────────────────────────────────────
#  Jupyter kernel (new, stateful, full notebook semantics)
# ─────────────────────────────────────────────────────────────────────────────

def _start_kernel() -> Tuple[KernelManager, "BlockingKernelClient"]:
    km = KernelManager(kernel_name="python3")
    km.start_kernel(cwd=str(DEFAULT_ROOT), env=_SANDBOX_ENV)
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready()
    return km, kc

_KM, _KC = _start_kernel()
_KERNEL_LOCK = threading.Lock()

def _exec_notebook(code: str, cwd: Path) -> Tuple[str, str]:
    """
    Execute *code* in the persistent Jupyter kernel and capture stdout / stderr.
    """
    global _KM, _KC
    stdout_chunks, stderr_chunks = [], []

    with _KERNEL_LOCK:
        # Revive the kernel if it has died.
        if _KC is None or not _KC.is_alive():
            try:
                _KM.shutdown_kernel(now=True, restart=False)
            except Exception:
                pass
            _KM, _KC = _start_kernel()

        # Ensure correct working directory inside the kernel.
        exec_id = _KC.execute(f"%cd {cwd}\n{code}", allow_stdin=False)

        # Drain the IOPub channel.
        while True:
            msg = _KC.get_iopub_msg(timeout=60)
            if msg["parent_header"].get("msg_id") != exec_id:
                continue

            mtype, content = msg["msg_type"], msg["content"]

            if mtype == "stream":  # regular prints
                target = stdout_chunks if content["name"] == "stdout" else stderr_chunks
                target.append(content["text"])
            elif mtype in ("execute_result", "display_data"):
                txt = content.get("data", {}).get("text/plain")
                if txt:
                    stdout_chunks.append(f"{txt}\n")
            elif mtype == "error":
                stderr_chunks.append("\n".join(content["traceback"]) + "\n")
            elif mtype == "status" and content["execution_state"] == "idle":
                break

    return "".join(stdout_chunks), "".join(stderr_chunks)

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_trace(tb: str) -> str:
    return "\n".join(
        ln for ln in tb.splitlines()
        if SERVER_FILE not in ln and "/flask/" not in ln
    )


def _run_with_timeout(fn, timeout: int, *args) -> Tuple[str, str, bool]:
    res, err = {}, {}

    def _target():
        try:
            o, e = fn(*args)
            res[0], err[0] = o, e
        except Exception:
            err[0] = _clean_trace(traceback.format_exc())

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return "", "Timed out", True
    return res.get(0, ""), err.get(0, ""), False


# ─────────────────────────────────────────────────────────────────────────────
#  Executors
# ─────────────────────────────────────────────────────────────────────────────

def _exec_shell(cmd: str, cwd: Path) -> tuple[str, str]:
    global SHELL_PROC
    with SHELL_LOCK:
        if SHELL_PROC.poll() is not None:            # bash died → restart
            SHELL_PROC = _start_shell()
        marker = uuid.uuid4().hex
        try:
            SHELL_PROC.stdin.write(f"cd {cwd}\n{cmd}\necho {marker}$?\n")
            SHELL_PROC.stdin.flush()
        except (BrokenPipeError, IOError):
            SHELL_PROC = _start_shell()
            SHELL_PROC.stdin.write(f"cd {cwd}\n{cmd}\necho {marker}$?\n")
            SHELL_PROC.stdin.flush()

        out_lines: list[str] = []
        while True:
            line = SHELL_PROC.stdout.readline()
            if not line:
                SHELL_PROC = _start_shell()
                return "", "shell crashed", -1
            if line.startswith(marker):
                rc = int(line[len(marker):].strip())
                break
            out_lines.append(line)
        return "".join(out_lines), "" if rc == 0 else f"exit {rc}"


def _exec_python(src: str, cwd: Path) -> tuple[str, str]:
    """Private helper (still used internally)."""
    global PY_REPL

    def _send(code: str) -> tuple[str, str]:
        msg = json.dumps({'code': code})
        PY_REPL.stdin.write(msg + "\n")
        PY_REPL.stdin.flush()
        resp = PY_REPL.stdout.readline()
        data = json.loads(resp)
        return data['out'], data['err']

    with _PY_LOCK:
        if PY_REPL.poll() is not None:
            PY_REPL = _start_repl()
        try:
            return _send(src)
        except (BrokenPipeError, OSError):
            PY_REPL = _start_repl()
            return _send(src)


def _exec_apply_patch(block: str, cwd: Path) -> Tuple[str, str]:
    os.chdir(cwd)
    try:
        res = apply_patch.process_patch(
            block, apply_patch.open_file, apply_patch.write_file, apply_patch.remove_file
        )
        return res + '\n', ''
    except Exception:
        return '', _clean_trace(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  Cell dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch(cell: str, cwd: Path) -> Tuple[str, str]:
    """
    • If the cell is a `%%bash … apply_patch …` block → run through
      `_exec_apply_patch` (unchanged behaviour).
    • Otherwise forward the entire cell verbatim to the notebook kernel.
    """
    cell = textwrap.dedent(cell)

    # Special‑case: git patch application
    if cell.lstrip().startswith('%%bash') and 'apply_patch' in cell:
        lines, capturing, patch_lines = cell.splitlines(), False, []
        for ln in lines:
            if '<<"EOF"' in ln or ln.strip().endswith('<<EOF'):
                capturing = True
                continue
            if ln.strip() == 'EOF' and capturing:
                break
            if capturing:
                patch_lines.append(ln)
        return _exec_apply_patch("\n".join(patch_lines), cwd)

    # General execution: delegate to Jupyter kernel
    return _exec_notebook(cell, cwd)

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
    diff = subprocess.run(f'git diff --no-color --cached {base_commit}', shell=True, cwd=cwd, text=True,
                          stdout=subprocess.PIPE)
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
    patch_path = Path('/tmp/patch.diff');
    patch_path.write_text(model_patch)
    script_path = Path('/tmp/eval.sh');
    script_path.write_text(eval_script);
    script_path.chmod(0o755)
    cwd = workdir.resolve()
    # apply_cmd = "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS') || (patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS') || echo 'APPLY_PATCH_FAIL'"
    apply_cmd = (
        'cd /testbed && '
        "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
        "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "echo 'APPLY_PATCH_FAIL')))"
    )
    out, err, _ = _exec_shell(apply_cmd, cwd)
    apply_out = (out + err).strip();
    report['apply_output'] = apply_out
    if 'APPLY_PATCH_FAIL' in apply_out:
        report['failed_apply_patch'] = True
        return report

    eval_out, eval_err, rc = _exec_shell('/tmp/eval.sh', cwd, timeout=timeout, merge=True)
    report['eval_output'] = eval_out + eval_err
    print(eval_out)
    print('=' * 30)
    print(eval_err)
    report['eval_output'] = eval_out
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
            results.append(
                {'index': idx, 'call_id': msg['call_id'], 'output': '', 'error': str(exc), 'timed_out': False,
                 'duration': 0.0})
            continue
        code = args.get('input')
        if code is None:
            results.append(
                {'index': idx, 'call_id': msg['call_id'], 'output': '', 'error': "'input' missing", 'timed_out': False,
                 'duration': 0.0});
            continue
        o_start = time.time()
        out, err, timed = _run_with_timeout(_dispatch, 60, code, DEFAULT_ROOT)
        results.append({'index': idx, 'call_id': msg['call_id'], 'output': out, 'error': err, 'timed_out': timed,
                        'duration': round(time.time() - o_start, 3)})
    if not results:
        return jsonify({'error': 'No python function_call messages found'}), 400
    return jsonify({'results': results, 'overall_duration': round(time.time() - start, 3)})


@app.route('/diff', methods=['POST'])
def diff_endpoint():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json();
    base_commit = data.get('base_commit')
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


@app.route('/upload_file', methods=['POST'])
def upload_file_endpoint():
    """Upload a file (or zip archive) to the server.

    Query params:
      destination – absolute/relative path where the file/dir should land
      recursive   – 'true' if the upload is a zip of a directory to be extracted
    """
    dest = request.args.get('destination')
    if not dest:
        return jsonify({'error': 'destination param missing'}), 400
    recursive = request.args.get('recursive', 'false').lower() == 'true'

    if 'file' not in request.files:
        return jsonify({'error': 'file field missing'}), 400
    file_storage = request.files['file']

    try:
        target_path = (Path(dest) if Path(dest).is_absolute() else DEFAULT_ROOT / dest).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if recursive:
            # Expect a zip file – extract
            with tempfile.TemporaryDirectory() as tmpdir:
                archive_path = Path(tmpdir) / 'upload.zip'
                file_storage.save(archive_path)
                import zipfile
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(target_path)
        else:
            file_storage.save(target_path)
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({'error': str(exc)}), 500
    return jsonify({'status': 'ok', 'path': str(target_path), 'recursive': recursive})


###############################################################################
# Entrypoint
###############################################################################

if __name__ == '__main__':
    port = 4444
    app.run(host='0.0.0.0', port=port, threaded=True)
