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
import signal
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request

import apply_patch

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────────────
SANDBOX_PREFIX = Path("/opt/miniconda3/envs/testbed").resolve()
if not (SANDBOX_PREFIX / "bin").exists():
    raise RuntimeError(f"Conda env not found at {SANDBOX_PREFIX}")

proc_env = subprocess.run(
    ["/bin/bash", "-lc",
     f"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate {SANDBOX_PREFIX} && env"],
    stdout=subprocess.PIPE,
    text=True,
    check=True
)
_SANDBOX_ENV: Dict[str, str] = {
    **dict(line.split("=", 1) for line in proc_env.stdout.splitlines() if "=" in line)
}
_SANDBOX_ENV["PATH"] = f"{SANDBOX_PREFIX/'bin'}:{_SANDBOX_ENV.get('PATH','')}"

DEFAULT_ROOT = Path("/testbed").resolve()
DEFAULT_ROOT.mkdir(parents=True, exist_ok=True)
SERVER_FILE = Path(__file__).resolve().as_posix()

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1) Stateful shell for legacy commands
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
        bufsize=1
    )

SHELL_PROC = _start_shell()
SHELL_LOCK = threading.Lock()

def _exec_shell(cmd: str, cwd: Path, timeout: int = None) -> Tuple[str, str, int]:
    """
    Execute a shell command within a persistent bash process.
    On timeout, only interrupt the running command, not the shell itself.
    """
    with SHELL_LOCK:
        if SHELL_PROC.poll() is not None:
            # Restart shell if it exited unexpectedly
            globals()['SHELL_PROC'] = _start_shell()

        marker = uuid.uuid4().hex
        SHELL_PROC.stdin.write(f"cd {cwd}\n{cmd}\necho {marker}$?\n")
        SHELL_PROC.stdin.flush()

        out_lines: List[str] = []
        rc = -1
        start = time.time()
        while True:
            if timeout and (time.time() - start) > timeout:
                # On timeout, interrupt the running command without killing the shell
                try:
                    SHELL_PROC.send_signal(signal.SIGINT)
                except Exception:
                    pass
                return "", "Timed out", -9
            line = SHELL_PROC.stdout.readline()
            if not line:
                # Shell died; restart it
                globals()['SHELL_PROC'] = _start_shell()
                return "", "shell crashed", -1
            if line.startswith(marker):
                rc = int(line[len(marker):].strip())
                break
            out_lines.append(line)
        return "".join(out_lines), "" if rc == 0 else f"exit {rc}", rc


# ─────────────────────────────────────────────────────────────────────────────
# 2) IPython‑based REPL
# ─────────────────────────────────────────────────────────────────────────────
_REPL_PATH = Path(tempfile.gettempdir()) / "ipython_repl.py"
_REPL_PATH.write_text(textwrap.dedent("""
import sys, json, io, traceback, os
from contextlib import redirect_stdout, redirect_stderr
from IPython.core.interactiveshell import InteractiveShell

shell = InteractiveShell.instance()
shell.separate_in = ''
shell.separate_out = ''
shell.separate_out2 = ''

for raw in sys.stdin:
    try:
        msg = json.loads(raw)
        code = msg.get('code','')
        cwd = msg.get('cwd')
        if cwd:
            os.chdir(cwd)
        outbuf, errbuf = io.StringIO(), io.StringIO()
        with redirect_stdout(outbuf), redirect_stderr(errbuf):
            shell.run_cell(code)
        out, err = outbuf.getvalue(), errbuf.getvalue()
    except Exception:
        out, err = '', traceback.format_exc()
    sys.stdout.write(json.dumps({'out':out,'err':err}) + '\\n')
    sys.stdout.flush()
"""))

def _start_repl() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [str(SANDBOX_PREFIX/"bin/python"), "-u", str(_REPL_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=DEFAULT_ROOT,
        env=_SANDBOX_ENV,
        text=True,
        bufsize=1
    )

PY_REPL = _start_repl()
_PY_LOCK = threading.Lock()

def _exec_apply_patch(block: str, cwd: Path) -> Tuple[str, str]:
    os.chdir(cwd)
    try:
        out = apply_patch.process_patch(
            block, apply_patch.open_file, apply_patch.write_file, apply_patch.remove_file
        )
        return out + "\n", ""
    except Exception:
        tb = traceback.format_exc().splitlines()
        clean = [ln for ln in tb if SERVER_FILE not in ln]
        return "", "\n".join(clean)

def _dispatch(cell: str, cwd: Path) -> Tuple[str, str]:
    global PY_REPL
    cell = textwrap.dedent(cell)

    lines = cell.splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        i += 1
    remainder = "\n".join(lines[i:])

    # Special: %%bash + apply_patch after comments
    if remainder.lstrip().startswith("%%bash") and "apply_patch" in cell:
        patch_lines: List[str] = []
        cap = False
        for ln in remainder.splitlines():
            if '<<"EOF"' in ln or ln.strip().endswith("<<EOF"):
                cap = True
                continue
            if cap and ln.strip() == "EOF":
                break
            if cap:
                patch_lines.append(ln)
        return _exec_apply_patch("\n".join(patch_lines), cwd)

    # Otherwise, send raw cell to IPython REPL
    msg = json.dumps({"code": cell, "cwd": str(cwd)})
    with _PY_LOCK:
        if PY_REPL.poll() is not None:
            globals()['PY_REPL'] = _start_repl()

        PY_REPL.stdin.write(msg + "\n")
        PY_REPL.stdin.flush()

        while True:
            resp = PY_REPL.stdout.readline()
            if resp:
                try:
                    data = json.loads(resp)
                    return data["out"], data["err"]
                except json.JSONDecodeError:
                    continue

            # If stdout closed unexpectedly, restart REPL
            err_text = PY_REPL.stderr.read() or "(no stderr output)"
            globals()['PY_REPL'] = _start_repl()
            return "", f"REPL crashed:\n{err_text}"

def _run_with_timeout(fn, timeout: int, *args) -> Tuple[str, str, bool]:
    """
    Run fn(*args) with a timeout. On timeout, interrupt only the running command or cell.
    """
    global SHELL_PROC, PY_REPL
    res: Dict[int, str] = {}
    err: Dict[int, str] = {}

    def target():
        try:
            o, e = fn(*args)
            res[0], err[0] = o, e
        except Exception:
            err[0] = traceback.format_exc()

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        # On timeout, interrupt the running command/cell without killing the env
        try:
            SHELL_PROC.send_signal(signal.SIGINT)
        except Exception:
            pass
        try:
            PY_REPL.send_signal(signal.SIGINT)
        except Exception:
            pass
        return "", "Timed out", True
    return res.get(0, ""), err.get(0, ""), False


# ─── Git‑patch & evaluation helpers (unchanged) ───────────────────────────────
def _remove_binary_diffs(patch_text: str) -> str:
    lines, cleaned, block, binary = patch_text.splitlines(), [], [], False
    for ln in lines:
        if ln.startswith("diff --git "):
            if block and not binary:
                cleaned.extend(block)
            block, binary = [ln], False
        elif "Binary files" in ln:
            binary = True
            block.append(ln)
        else:
            block.append(ln)
    if block and not binary:
        cleaned.extend(block)
    return "\n".join(cleaned)

def get_git_patch(base_commit: str, workdir: Path = DEFAULT_ROOT) -> str:
    cwd = workdir.resolve()
    subprocess.run(
        'find . -type d -name .git -not -path "./.git" -exec rm -rf {} +',
        shell=True, cwd=cwd
    )
    subprocess.run('git add -A', shell=True, cwd=cwd)
    remove_bin_cmd = r'''
for f in $(git status --porcelain | grep -E "^(M| M|\?\?|A| A)" | cut -c4-);
do
  if [ -f "$f" ] && (
        file -b "$f" | grep -q "executable" ||
        git check-attr binary "$f" | grep -q "binary: set"
     ); then
    git rm -f "$f" 2>/dev/null || rm -f "$f"
  fi
done
'''
    subprocess.run(remove_bin_cmd, shell=True, cwd=cwd)
    diff = subprocess.run(
        f'git diff --no-color --cached {base_commit}',
        shell=True, cwd=cwd, text=True, stdout=subprocess.PIPE
    )
    return _remove_binary_diffs(diff.stdout)

def _evaluate_patch(model_patch: str, eval_script: str, workdir: Path, timeout: int = 1800) -> Dict[str, Any]:
    report = {
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

    Path('/tmp/patch.diff').write_text(model_patch)
    script_path = Path('/tmp/eval.sh')
    script_path.write_text(eval_script)
    script_path.chmod(0o755)
    cwd = workdir.resolve()

    apply_cmd = (
        'cd /testbed && '
        "(git apply -v /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "(echo 'Failed to apply patch with git apply, trying with patch command...' && "
        "(patch --batch --fuzz=5 -p1 -i /tmp/patch.diff && echo 'APPLY_PATCH_PASS' || "
        "echo 'APPLY_PATCH_FAIL')))"
    )
    out, err, rc = _exec_shell(apply_cmd, cwd, timeout=timeout)
    apply_out = (out + err).strip()
    report['apply_output'] = apply_out
    if 'APPLY_PATCH_FAIL' in apply_out:
        report['failed_apply_patch'] = True
        return report

    eval_out, eval_err, rc = _exec_shell('/tmp/eval.sh', cwd, timeout=timeout)
    report['eval_output'] = eval_out + eval_err
    if rc == -9:
        report['test_timeout'] = True
    elif rc != 0:
        report['error_eval'] = True
    else:
        report['resolved'] = 'RESOLVED' in report['eval_output'].upper()
    return report

# ─── Flask routes ─────────────────────────────────────────────────────────────
@app.route('/alive', methods=['GET'])
def alive():
    return 'ok', 200

@app.route('/execute', methods=['POST'])
def execute_endpoint():
    start = time.time()
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    payload = request.get_json()
    results = []
    for idx, msg in enumerate(payload):
        if msg.get('type') != 'function_call' or msg.get('name') != 'python':
            continue
        try:
            args = json.loads(msg.get('arguments', '{}'))
        except json.JSONDecodeError as exc:
            results.append({'index': idx, 'call_id': msg.get('call_id'), 'output': '', 'error': str(exc), 'timed_out': False, 'duration': 0.0})
            continue
        code = args.get('input')
        if code is None:
            results.append({'index': idx, 'call_id': msg.get('call_id'), 'output': '', 'error': "'input' missing", 'timed_out': False, 'duration': 0.0})
            continue
        o_start = time.time()
        out, err, timed = _run_with_timeout(_dispatch, 60, code, DEFAULT_ROOT)
        print(err)
        results.append({'index': idx, 'call_id': msg.get('call_id'), 'output': out, 'error': err, 'timed_out': timed, 'duration': round(time.time() - o_start, 3)})
    if not results:
        return jsonify({'error': 'No python function_call messages found'}), 400
    return jsonify({'results': results, 'overall_duration': round(time.time() - start, 3)})

@app.route('/diff', methods=['POST'])
def diff_endpoint():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json()
    base_commit = data.get('base_commit')
    if not base_commit:
        return jsonify({'error': 'base_commit missing'}), 400
    try:
        patch = get_git_patch(base_commit, Path(data.get('dir', DEFAULT_ROOT)))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    return jsonify({'patch': patch})

@app.route('/evaluate', methods=['POST'])
def evaluate_endpoint():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json()
    report = _evaluate_patch(
        data.get('model_patch', ''),
        data.get('eval_script', ''),
        Path(data.get('dir', DEFAULT_ROOT)),
        int(data.get('timeout', 1800))
    )
    return jsonify({'report': report})

@app.route('/command', methods=['POST'])
def command_endpoint():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json()
    cmd = data.get('cmd') or data.get('command')
    if not cmd:
        return jsonify({'error': 'cmd missing'}), 400
    cwd = Path(data.get('dir', DEFAULT_ROOT)).resolve()
    timeout = int(data.get('timeout', 120))
    out, err, rc = _exec_shell(cmd, cwd, timeout=timeout)
    return jsonify({'stdout': out, 'stderr': err, 'returncode': rc, 'duration': timeout})

@app.route('/upload_file', methods=['POST'])
def upload_file_endpoint():
    dest = request.args.get('destination')
    if not dest:
        return jsonify({'error': 'destination param missing'}), 400
    recursive = request.args.get('recursive', 'false').lower() == 'true'
    if 'file' not in request.files:
        return jsonify({'error': 'file field missing'}), 400
    file_storage = request.files['file']
    try:
        target = (Path(dest) if Path(dest).is_absolute() else DEFAULT_ROOT / dest).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if recursive:
            import zipfile
            with tempfile.TemporaryDirectory() as td:
                archive = Path(td) / "upload.zip"
                file_storage.save(archive)
                with zipfile.ZipFile(archive, "r") as zf:
                    zf.extractall(target)
        else:
            file_storage.save(target)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    return jsonify({'status': 'ok', 'path': str(target), 'recursive': recursive})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4444, threaded=True)
