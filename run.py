# server.py
import sys
import io
import json
import subprocess
import os
import pathlib
import traceback
import threading
from flask import Flask, request, jsonify
from queue import Queue
from contextlib import redirect_stdout

# --- Configuration ---
WORKSPACE_DIR = pathlib.Path("/testbed")
EXECUTION_TIMEOUT = 300.0  # seconds
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 4444

# --- Global State for Python Execution Context ---
# Caution: This simple dictionary is not sandboxed.
# For security, consider using a more robust execution environment.
python_globals = {"__builtins__": __builtins__}
python_locals = {} # Separate locals can sometimes be useful

# --- Create Workspace ---
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# --- apply_patch.py Logic (Embedded) ---
# Note: Copied directly from the provided code.
# Modifications: File operations are now relative to WORKSPACE_DIR.
from __future__ import annotations

# import pathlib # Already imported
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)


# --------------------------------------------------------------------------- #
#  Domain objects
# --------------------------------------------------------------------------- #
class ActionType(str, Enum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


@dataclass
class FileChange:
    type: ActionType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    move_path: Optional[str] = None


@dataclass
class Commit:
    changes: Dict[str, FileChange] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Exceptions
# --------------------------------------------------------------------------- #
class DiffError(ValueError):
    """Any problem detected while parsing or applying a patch."""


# --------------------------------------------------------------------------- #
#  Helper dataclasses used while parsing patches
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    orig_index: int = -1
    del_lines: List[str] = field(default_factory=list)
    ins_lines: List[str] = field(default_factory=list)


@dataclass
class PatchAction:
    type: ActionType
    new_file: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)
    move_path: Optional[str] = None


@dataclass
class Patch:
    actions: Dict[str, PatchAction] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Patch text parser
# --------------------------------------------------------------------------- #
@dataclass
class Parser:
    current_files: Dict[str, str]
    lines: List[str]
    index: int = 0
    patch: Patch = field(default_factory=Patch)
    fuzz: int = 0

    # ------------- low-level helpers -------------------------------------- #
    def _cur_line(self) -> str:
        if self.index >= len(self.lines):
            raise DiffError("Unexpected end of input while parsing patch")
        return self.lines[self.index]

    @staticmethod
    def _norm(line: str) -> str:
        """Strip CR so comparisons work for both LF and CRLF input."""
        return line.rstrip("\r")

    # ------------- scanning convenience ----------------------------------- #
    def is_done(self, prefixes: Optional[Tuple[str, ...]] = None) -> bool:
        if self.index >= len(self.lines):
            return True
        if (
            prefixes
            and len(prefixes) > 0
            and self._norm(self._cur_line()).startswith(prefixes)
        ):
            return True
        return False

    def startswith(self, prefix: Union[str, Tuple[str, ...]]) -> bool:
        return self._norm(self._cur_line()).startswith(prefix)

    def read_str(self, prefix: str) -> str:
        """
        Consume the current line if it starts with *prefix* and return the text
        **after** the prefix.  Raises if prefix is empty.
        """
        if prefix == "":
            raise ValueError("read_str() requires a non-empty prefix")
        if self._norm(self._cur_line()).startswith(prefix):
            text = self._cur_line()[len(prefix) :]
            self.index += 1
            return text
        return ""

    def read_line(self) -> str:
        """Return the current raw line and advance."""
        line = self._cur_line()
        self.index += 1
        return line

    # ------------- public entry point -------------------------------------- #
    def parse(self) -> None:
        while not self.is_done(("*** End Patch",)):
            # ---------- UPDATE ---------- #
            path = self.read_str("*** Update File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Duplicate update for file: {path}")
                move_to = self.read_str("*** Move to: ")
                if path not in self.current_files:
                    raise DiffError(f"Update File Error - missing file: {path}")
                text = self.current_files[path]
                action = self._parse_update_file(text)
                action.move_path = move_to or None
                self.patch.actions[path] = action
                continue

            # ---------- DELETE ---------- #
            path = self.read_str("*** Delete File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Duplicate delete for file: {path}")
                if path not in self.current_files:
                    raise DiffError(f"Delete File Error - missing file: {path}")
                self.patch.actions[path] = PatchAction(type=ActionType.DELETE)
                continue

            # ---------- ADD ---------- #
            path = self.read_str("*** Add File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Duplicate add for file: {path}")
                if path in self.current_files:
                    raise DiffError(f"Add File Error - file already exists: {path}")
                self.patch.actions[path] = self._parse_add_file()
                continue

            raise DiffError(f"Unknown line while parsing: {self._cur_line()}")

        if not self.startswith("*** End Patch"):
            raise DiffError("Missing *** End Patch sentinel")
        self.index += 1  # consume sentinel

    # ------------- section parsers ---------------------------------------- #
    def _parse_update_file(self, text: str) -> PatchAction:
        action = PatchAction(type=ActionType.UPDATE)
        lines = text.split("\n")
        index = 0
        while not self.is_done(
            (
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            )
        ):
            def_str = self.read_str("@@ ")
            section_str = ""
            if not def_str and self._norm(self._cur_line()) == "@@":
                section_str = self.read_line()

            if not (def_str or section_str or index == 0):
                raise DiffError(f"Invalid line in update section:\n{self._cur_line()}")

            if def_str.strip():
                found = False
                if def_str not in lines[:index]:
                    for i, s in enumerate(lines[index:], index):
                        if s == def_str:
                            index = i + 1
                            found = True
                            break
                if not found and def_str.strip() not in [
                    s.strip() for s in lines[:index]
                ]:
                    for i, s in enumerate(lines[index:], index):
                        if s.strip() == def_str.strip():
                            index = i + 1
                            self.fuzz += 1
                            found = True
                            break

            next_ctx, chunks, end_idx, eof = peek_next_section(self.lines, self.index)
            new_index, fuzz = find_context(lines, next_ctx, index, eof)
            if new_index == -1:
                ctx_txt = "\n".join(next_ctx)
                raise DiffError(
                    f"Invalid {'EOF ' if eof else ''}context at {index}:\n{ctx_txt}"
                )
            self.fuzz += fuzz
            for ch in chunks:
                ch.orig_index += new_index
                action.chunks.append(ch)
            index = new_index + len(next_ctx)
            self.index = end_idx
        return action

    def _parse_add_file(self) -> PatchAction:
        lines: List[str] = []
        while not self.is_done(
            ("*** End Patch", "*** Update File:", "*** Delete File:", "*** Add File:")
        ):
            s = self.read_line()
            if not s.startswith("+"):
                raise DiffError(f"Invalid Add File line (missing '+'): {s}")
            lines.append(s[1:])  # strip leading '+'
        return PatchAction(type=ActionType.ADD, new_file="\n".join(lines))


# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #
def find_context_core(
    lines: List[str], context: List[str], start: int
) -> Tuple[int, int]:
    if not context:
        return start, 0

    for i in range(start, len(lines)):
        if lines[i : i + len(context)] == context:
            return i, 0
    for i in range(start, len(lines)):
        if [s.rstrip() for s in lines[i : i + len(context)]] == [
            s.rstrip() for s in context
        ]:
            return i, 1
    for i in range(start, len(lines)):
        if [s.strip() for s in lines[i : i + len(context)]] == [
            s.strip() for s in context
        ]:
            return i, 100
    return -1, 0


def find_context(
    lines: List[str], context: List[str], start: int, eof: bool
) -> Tuple[int, int]:
    if eof:
        new_index, fuzz = find_context_core(lines, context, len(lines) - len(context))
        if new_index != -1:
            return new_index, fuzz
        new_index, fuzz = find_context_core(lines, context, start)
        return new_index, fuzz + 10_000
    return find_context_core(lines, context, start)


def peek_next_section(
    lines: List[str], index: int
) -> Tuple[List[str], List[Chunk], int, bool]:
    old: List[str] = []
    del_lines: List[str] = []
    ins_lines: List[str] = []
    chunks: List[Chunk] = []
    mode = "keep"
    orig_index = index

    while index < len(lines):
        s = lines[index]
        if s.startswith(
            (
                "@@",
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            )
        ):
            break
        if s == "***":
            break
        if s.startswith("***"):
            raise DiffError(f"Invalid Line: {s}")
        index += 1

        last_mode = mode
        if s == "":
            s = " "
        if s[0] == "+":
            mode = "add"
        elif s[0] == "-":
            mode = "delete"
        elif s[0] == " ":
            mode = "keep"
        else:
            raise DiffError(f"Invalid Line: {s}")
        s = s[1:]

        if mode == "keep" and last_mode != mode:
            if ins_lines or del_lines:
                chunks.append(
                    Chunk(
                        orig_index=len(old) - len(del_lines),
                        del_lines=del_lines,
                        ins_lines=ins_lines,
                    )
                )
            del_lines, ins_lines = [], []

        if mode == "delete":
            del_lines.append(s)
            old.append(s)
        elif mode == "add":
            ins_lines.append(s)
        elif mode == "keep":
            old.append(s)

    if ins_lines or del_lines:
        chunks.append(
            Chunk(
                orig_index=len(old) - len(del_lines),
                del_lines=del_lines,
                ins_lines=ins_lines,
            )
        )

    if index < len(lines) and lines[index] == "*** End of File":
        index += 1
        return old, chunks, index, True

    if index == orig_index:
        raise DiffError("Nothing in this section")
    return old, chunks, index, False


# --------------------------------------------------------------------------- #
#  Patch → Commit and Commit application
# --------------------------------------------------------------------------- #
def _get_updated_file(text: str, action: PatchAction, path: str) -> str:
    if action.type is not ActionType.UPDATE:
        raise DiffError("_get_updated_file called with non-update action")
    orig_lines = text.split("\n")
    dest_lines: List[str] = []
    orig_index = 0

    for chunk in action.chunks:
        if chunk.orig_index > len(orig_lines):
            raise DiffError(
                f"{path}: chunk.orig_index {chunk.orig_index} exceeds file length"
            )
        if orig_index > chunk.orig_index:
            raise DiffError(
                f"{path}: overlapping chunks at {orig_index} > {chunk.orig_index}"
            )

        dest_lines.extend(orig_lines[orig_index : chunk.orig_index])
        orig_index = chunk.orig_index

        dest_lines.extend(chunk.ins_lines)
        orig_index += len(chunk.del_lines)

    dest_lines.extend(orig_lines[orig_index:])
    return "\n".join(dest_lines)


def patch_to_commit(patch: Patch, orig: Dict[str, str]) -> Commit:
    commit = Commit()
    for path, action in patch.actions.items():
        if action.type is ActionType.DELETE:
            commit.changes[path] = FileChange(
                type=ActionType.DELETE, old_content=orig[path]
            )
        elif action.type is ActionType.ADD:
            if action.new_file is None:
                raise DiffError("ADD action without file content")
            commit.changes[path] = FileChange(
                type=ActionType.ADD, new_content=action.new_file
            )
        elif action.type is ActionType.UPDATE:
            new_content = _get_updated_file(orig[path], action, path)
            commit.changes[path] = FileChange(
                type=ActionType.UPDATE,
                old_content=orig[path],
                new_content=new_content,
                move_path=action.move_path,
            )
    return commit


# --------------------------------------------------------------------------- #
#  User-facing helpers
# --------------------------------------------------------------------------- #
def text_to_patch(text: str, orig: Dict[str, str]) -> Tuple[Patch, int]:
    lines = text.splitlines()  # preserves blank lines, no strip()
    # Adjust index check for potential surrounding bash commands
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if Parser._norm(line).startswith("*** Begin Patch"):
            start_idx = i
            break
    for i in range(len(lines) - 1, -1, -1):
        if Parser._norm(lines[i]) == "*** End Patch":
            end_idx = i
            break

    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
         raise DiffError("Invalid patch text - missing or misplaced sentinels")

    # Only parse the content between the sentinels
    parser = Parser(current_files=orig, lines=lines[start_idx+1:end_idx], index=0)
    parser.parse()
    return parser.patch, parser.fuzz


def identify_files_needed(text: str) -> List[str]:
    lines = text.splitlines()
    # Only look between sentinels
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if Parser._norm(line).startswith("*** Begin Patch"):
            start_idx = i
            break
    for i in range(len(lines) - 1, -1, -1):
        if Parser._norm(lines[i]) == "*** End Patch":
            end_idx = i
            break
    if start_idx == -1 or end_idx == -1:
        return [] # Or raise error? Assume empty if no sentinels

    patch_lines = lines[start_idx+1:end_idx]

    return [
        line[len("*** Update File: ") :]
        for line in patch_lines
        if line.startswith("*** Update File: ")
    ] + [
        line[len("*** Delete File: ") :]
        for line in patch_lines
        if line.startswith("*** Delete File: ")
    ]


def identify_files_added(text: str) -> List[str]:
    lines = text.splitlines()
    # Only look between sentinels
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if Parser._norm(line).startswith("*** Begin Patch"):
            start_idx = i
            break
    for i in range(len(lines) - 1, -1, -1):
        if Parser._norm(lines[i]) == "*** End Patch":
            end_idx = i
            break
    if start_idx == -1 or end_idx == -1:
         return [] # Or raise error? Assume empty if no sentinels

    patch_lines = lines[start_idx+1:end_idx]

    return [
        line[len("*** Add File: ") :]
        for line in patch_lines
        if line.startswith("*** Add File: ")
    ]


# --------------------------------------------------------------------------- #
#  File-system helpers (MODIFIED FOR SERVER CONTEXT)
# --------------------------------------------------------------------------- #
def _resolve_path(rel_path: str) -> pathlib.Path:
    """Resolves a relative path against the WORKSPACE_DIR, ensuring it stays within."""
    abs_path = (WORKSPACE_DIR / rel_path).resolve()
    # Security check: Ensure the resolved path is still within the workspace
    if WORKSPACE_DIR.resolve() not in abs_path.parents and abs_path != WORKSPACE_DIR.resolve():
        raise DiffError(f"Attempted file access outside workspace: {rel_path}")
    # Prevent accessing the workspace root directly if rel_path is empty or '.'
    if not rel_path or rel_path == '.':
         raise DiffError(f"Invalid relative path specified: '{rel_path}'")
    return abs_path

def load_files(paths: List[str], open_fn: Callable[[str], str]) -> Dict[str, str]:
    # Path validation happens inside open_fn now
    return {path: open_fn(path) for path in paths}


def apply_commit(
    commit: Commit,
    write_fn: Callable[[str, str], None],
    remove_fn: Callable[[str], None],
) -> None:
    for path, change in commit.changes.items():
        # Path validation happens inside write/remove_fn now
        if change.type is ActionType.DELETE:
            remove_fn(path)
        elif change.type is ActionType.ADD:
            if change.new_content is None:
                raise DiffError(f"ADD change for {path} has no content")
            write_fn(path, change.new_content)
        elif change.type is ActionType.UPDATE:
            if change.new_content is None:
                raise DiffError(f"UPDATE change for {path} has no new content")
            target = change.move_path or path
            write_fn(target, change.new_content)
            if change.move_path and path != target: # Ensure we don't delete if move is to same path
                remove_fn(path)


def process_patch(
    text: str,
    open_fn: Callable[[str], str],
    write_fn: Callable[[str, str], None],
    remove_fn: Callable[[str], None],
) -> str:
    # Basic check for start sentinel presence
    if "*** Begin Patch" not in text:
         raise DiffError("Patch text must contain *** Begin Patch")
    paths = identify_files_needed(text) # Identifies paths mentioned between sentinels
    # load_files now uses the workspace-aware open_fn
    orig = load_files(paths, open_fn)
    patch, _fuzz = text_to_patch(text, orig) # Uses content between sentinels
    commit = patch_to_commit(patch, orig)
    # apply_commit now uses the workspace-aware write/remove_fn
    apply_commit(commit, write_fn, remove_fn)
    return "Done!"


# --------------------------------------------------------------------------- #
#  Default FS helpers (MODIFIED FOR SERVER CONTEXT)
# --------------------------------------------------------------------------- #
def open_file(path: str) -> str:
    """Reads a file relative to the workspace directory."""
    target_path = _resolve_path(path)
    if not target_path.is_file():
        raise DiffError(f"File not found in workspace: {path}")
    with open(target_path, "rt", encoding="utf-8") as fh:
        return fh.read()

def write_file(path: str, content: str) -> None:
    """Writes a file relative to the workspace directory."""
    target_path = _resolve_path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wt", encoding="utf-8") as fh:
        fh.write(content)

def remove_file(path: str) -> None:
    """Removes a file relative to the workspace directory."""
    try:
        target_path = _resolve_path(path)
        target_path.unlink(missing_ok=True) # Allow removing non-existent files silently
    except DiffError:
         print(f"Warning: Attempt to remove file outside workspace ignored: {path}", file=sys.stderr)
    except IsADirectoryError:
         print(f"Warning: Attempt to remove directory with remove_file: {path}", file=sys.stderr)
         # Optionally, implement rmdir or shutil.rmtree if needed
    except OSError as e:
        print(f"Warning: Error removing file {path}: {e}", file=sys.stderr)


# --- END of apply_patch.py Logic ---


# --- Execution Functions ---

def execute_shell_command(command: str, result_queue: Queue):
    """Executes a shell command in the workspace directory with timeout."""
    full_command = command[1:] # Remove the leading '!'
    output = ""
    error_output = ""
    return_code = -1
    timed_out = False
    try:
        # Security Note: shell=True can be dangerous if the command is constructed
        # from untrusted input. Here we assume the agent provides the command.
        # Consider using shell=False and splitting the command if possible.
        process = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=WORKSPACE_DIR,
            timeout=EXECUTION_TIMEOUT,
            check=False, # Don't raise exception on non-zero exit
        )
        output = process.stdout
        error_output = process.stderr
        return_code = process.returncode
    except subprocess.TimeoutExpired:
        output = f"Command timed out after {EXECUTION_TIMEOUT} seconds."
        timed_out = True
    except Exception as e:
        error_output = f"Error executing shell command: {e}\n{traceback.format_exc()}"

    result_queue.put({
        "output": output,
        "error": error_output,
        "return_code": return_code,
        "timed_out": timed_out
    })

def execute_python_code(code: str, result_queue: Queue):
    """Executes Python code in the persistent context with timeout."""
    output_buffer = io.StringIO()
    error_output = ""
    timed_out = False

    def target():
        nonlocal error_output
        try:
            # Change working directory for the execution duration
            original_cwd = os.getcwd()
            os.chdir(WORKSPACE_DIR)
            try:
                with redirect_stdout(output_buffer):
                    exec(code, python_globals, python_locals)
            finally:
                # Ensure CWD is restored even if exec fails
                os.chdir(original_cwd)
        except Exception:
            error_output = traceback.format_exc()
        finally:
             # Signal completion (or timeout interrupt)
             if not timed_out:
                 result_queue.put({
                     "output": output_buffer.getvalue(),
                     "error": error_output,
                     "timed_out": False
                 })

    thread = threading.Thread(target=target)
    thread.daemon = True # Allow program to exit even if this thread is running (after timeout)
    thread.start()
    thread.join(timeout=EXECUTION_TIMEOUT)

    if thread.is_alive():
        timed_out = True
        # Note: Reliably stopping a thread executing arbitrary Python code is hard.
        # This timeout mechanism primarily prevents the server from hanging indefinitely.
        # The thread might still run in the background until the exec finishes or errors.
        # For true sandboxing/interruption, multiprocessing or more complex setups are needed.
        result_queue.put({
            "output": output_buffer.getvalue(), # Get whatever output occurred before timeout
            "error": f"Execution timed out after {EXECUTION_TIMEOUT} seconds.",
            "timed_out": True
        })
    # If thread finished normally, the result is already in the queue from target()

def execute_apply_patch(patch_command: str, result_queue: Queue):
    """Applies a patch using the embedded logic."""
    output = ""
    error_output = ""
    timed_out = False # Patch application itself doesn't have explicit timeout here

    # Basic check for structure
    if not patch_command.strip().startswith("%%bash"):
         result_queue.put({
            "output": "",
            "error": "Invalid apply_patch command: Does not start with %%bash.",
            "timed_out": False
         })
         return
    if "<<\"EOF\"" not in patch_command:
         result_queue.put({
            "output": "",
            "error": "Invalid apply_patch command: Missing <<\"EOF\".",
            "timed_out": False
         })
         return

    # Extract the actual patch text between the sentinels
    try:
        # Find patch content between "*** Begin Patch" and "*** End Patch"
        # This assumes the surrounding apply_patch command structure is present
        patch_text_full = patch_command # The whole input is needed for context identification
        # --- Call the embedded patch processor ---
        result = process_patch(patch_text_full, open_file, write_file, remove_file)
        output = result # Should be "Done!" on success
    except DiffError as e:
        error_output = f"Patch Error: {e}"
    except Exception as e:
        error_output = f"Error processing patch command: {e}\n{traceback.format_exc()}"

    result_queue.put({
        "output": output,
        "error": error_output,
        "timed_out": timed_out
    })


# --- Flask App ---
app = Flask(__name__)

@app.route('/execute', methods=['POST'])
def execute():
    """
    Expects JSON like:
    [
        {... previous messages ...},
        {'type': 'function_call', 'name': 'python', 'arguments': '{"input": "..."}'}
    ]
    Executes the input and returns JSON like:
    {'output': '...', 'error': '...', 'timed_out': False}
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if not isinstance(data, list) or not data:
        return jsonify({"error": "Expected a non-empty list of messages"}), 400

    # Find the function call message
    function_call_msg = None
    for msg in reversed(data): # Check recent messages first
        if isinstance(msg, dict) and msg.get('type') == 'function_call':
            function_call_msg = msg
            break

    if not function_call_msg:
        return jsonify({"error": "No message with type 'function_call' found"}), 400

    if function_call_msg.get('name') != 'python':
         return jsonify({"error": f"Expected function name 'python', got '{function_call_msg.get('name')}'"}), 400

    try:
        args_str = function_call_msg.get('arguments', '{}')
        args = json.loads(args_str)
        input_command = args.get('input')
        if input_command is None:
            return jsonify({"error": "Missing 'input' key in function call arguments"}), 400
        if not isinstance(input_command, str):
             return jsonify({"error": "'input' must be a string"}), 400

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to decode JSON arguments"}), 400
    except Exception as e:
        return jsonify({"error": f"Error parsing arguments: {e}"}), 400

    result_queue = Queue(maxsize=1)
    final_result = {"output": "", "error": "Execution failed to produce result.", "timed_out": False}

    try:
        if input_command.startswith('!'):
            print(f"--- Executing Shell: {input_command[:100]}... ---")
            execute_shell_command(input_command, result_queue)

        elif input_command.strip().startswith("%%bash") and "apply_patch" in input_command:
             print(f"--- Executing Patch: {input_command[:100]}... ---")
             execute_apply_patch(input_command, result_queue)

        else:
            print(f"--- Executing Python: {input_command[:100]}... ---")
            execute_python_code(input_command, result_queue)

        # Wait for the result from the execution function (with a small buffer over internal timeout)
        try:
             final_result = result_queue.get(timeout=EXECUTION_TIMEOUT + 2.0)
        except result_queue.Empty:
             final_result = {"output": "", "error": "Execution timed out or failed to return.", "timed_out": True}

    except Exception as e:
         print(f"--- Execution Error --- \n{traceback.format_exc()}\n--------------------")
         final_result = {"output": "", "error": f"Server error during execution: {e}", "timed_out": False}


    # Combine stdout and stderr for the final output, similar to notebook behavior
    combined_output = final_result.get("output", "")
    error_msg = final_result.get("error", "")
    if error_msg:
         # Prepend error to output or just return error? Let's combine.
         # The agent description says output *or* timeout. Let's put errors in the output stream.
        combined_output += f"\n--- STDERR/ERROR ---\n{error_msg}"


    print(f"--- Result --- \nOutput:\n{final_result.get('output', '')}\nError:\n{final_result.get('error', '')}\nTimed Out: {final_result.get('timed_out', False)}\n--------------")

    # Return structure mimicking a successful function execution result for the agent
    # The 'content' field would typically hold the stdout/stderr result.
    # You might need to adjust this structure based on how the calling LLM framework
    # expects function results.
    return jsonify({
         "tool_code": "python", # Or function_call_msg.get('name')
         "tool_name": "python",
         "is_error": bool(final_result.get("error")) or final_result.get("timed_out", False),
         "stdout": final_result.get("output", ""),
         "stderr": final_result.get("error", ""),
         "status": "error" if bool(final_result.get("error")) or final_result.get("timed_out", False) else "success",
         "exit_code": final_result.get("return_code", None) # Only relevant for shell commands
         # Legacy/Alternative formatting:
         # "content": combined_output,
         # "timed_out": final_result.get('timed_out', False)
    })


if __name__ == '__main__':
    print(f"--- Starting SWE Agent Execution Server ---")
    print(f"Workspace directory: {WORKSPACE_DIR.resolve()}")
    print(f"Listening on {LISTEN_HOST}:{LISTEN_PORT}")
    print(f"Execution timeout: {EXECUTION_TIMEOUT} seconds")
    # Use waitress or gunicorn for production instead of Flask's dev server
    # from waitress import serve
    # serve(app, host=LISTEN_HOST, port=LISTEN_PORT)
    # Or for development:
    app.run(host=LISTEN_HOST, port=LISTEN_PORT, debug=False) # debug=True causes issues with threads/state sometimes
