from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, List

from flask import Flask, jsonify, request

import apply_patch

# ---------------------------------------------------------------------------
#  Minimal implementation of the V4A *apply_patch* processor
#     (adapted from the reference code bundled with SWE‑Bench).
# ---------------------------------------------------------------------------

class DiffError(ValueError):
    """Raised when the patch text is malformed."""


def _open_file(path: str) -> str:
    with open(path, "rt", encoding="utf-8") as fh:
        return fh.read()


def _write_file(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _remove_file(path: str) -> None:
    Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
#  Action executors
# ---------------------------------------------------------------------------

_PATCH_RE = re.compile(r"\*\*\* Begin Patch[\s\S]*?\*\*\* End Patch", re.MULTILINE)
_BASH_RE = re.compile(r"^(?:!|%%bash)(.*)$", re.MULTILINE)


def _run_shell(cmd: str) -> tuple[str, str, int]:
    """Executes *cmd* in a subprocess, returns (stdout, stderr, returncode)."""
    completed = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return completed.stdout, completed.stderr, completed.returncode


def _run_python(code: str) -> tuple[str, str, int]:
    """Executes Python *code* in an isolated namespace and captures streams."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exit_code = 0
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, {"__name__": "__main__"})
    except Exception as exc:  # pylint: disable=broad-except
        print(exc, file=stderr_buf)
        exit_code = 1
    return stdout_buf.getvalue(), stderr_buf.getvalue(), exit_code


def _apply_patch(patch_text: str) -> tuple[str, str, int]:
    """Applies a V4A patch and returns a status tuple."""
    try:
        # patch_mod = _ensure_patch_mod()
        result = apply_patch.process_patch(
            patch_text, _open_file, _write_file, _remove_file
        )
        return result + "\n", "", 0
    except DiffError as exc:
        return "", str(exc) + "\n", 1


# ---------------------------------------------------------------------------
#  Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/healthz", methods=["GET"])
def healthz():  # noqa: D401 – simple health endpoint
    return "ok", 200


@app.route("/run", methods=["POST"])
def run_action():
    """Main endpoint – executes the action(s) contained in *content*."""
    data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    content: str | None = data.get("content") or data.get("llm_output")
    if not content:
        return jsonify(error="Missing 'content' field"), 400

    # 1) Apply **all** patch blocks first (order matters for SWE‑Bench).
    stdout_parts: List[str] = []
    stderr_parts: List[str] = []
    status_code = 0

    for patch_block in _PATCH_RE.findall(content):
        out, err, code = _apply_patch(patch_block)
        stdout_parts.append(out)
        stderr_parts.append(err)
        status_code = status_code or code  # propagate first non‑zero

    # Remove patch blocks so they are not re‑processed as code.
    remaining = _PATCH_RE.sub("", content).strip()

    # 2) Shell command – we only run the **first** detected command for safety.
    bash_match = _BASH_RE.search(remaining)
    if bash_match:
        cmd_block = remaining[bash_match.start() : bash_match.end()]
        # Handle %%bash header
        cmd_text = cmd_block
        if cmd_block.lstrip().startswith("%%bash"):
            cmd_text = cmd_block.split("\n", 1)[1]
        # Handle leading '!'
        cmd_text = cmd_text.lstrip()[1:] if cmd_text.lstrip().startswith("!") else cmd_text
        out, err, code = _run_shell(cmd_text)
        stdout_parts.append(out)
        stderr_parts.append(err)
        status_code = status_code or code
        remaining = remaining.replace(cmd_block, "", 1).strip()

    # 3) Whatever is left is treated as Python.
    if remaining:
        out, err, code = _run_python(remaining)
        stdout_parts.append(out)
        stderr_parts.append(err)
        status_code = status_code or code

    return (
        jsonify(
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            returncode=status_code,
        ),
        200,
    )


# ---------------------------------------------------------------------------
#  CLI helper
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="LLM Action Execution Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    app.run(host=args.host, port=args.port, threaded=True)



if __name__ == "__main__":
    main()
