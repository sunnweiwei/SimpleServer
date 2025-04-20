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
# from code import InteractiveConsole # No longer needed for dispatch
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import queue # For kernel message handling

from flask import Flask, jsonify, request
import jupyter_client # Added for kernel interaction

import apply_patch


SANDBOX_PREFIX = Path("/opt/miniconda3/envs/testbed").resolve()
if not (SANDBOX_PREFIX / "bin").exists():
    raise RuntimeError(f"Conda env not found at {SANDBOX_PREFIX}")

# --- Environment Setup ---
# Run command once to get the environment variables from the activated conda env
proc_env = subprocess.run(
    ["/bin/bash", "-lc",
     f"source /opt/miniconda3/etc/profile.d/conda.sh && conda activate {SANDBOX_PREFIX} && env"],
    stdout=subprocess.PIPE,
    text=True,
    check=True
)
_SANDBOX_ENV: Dict[str, str] = {}
for line in proc_env.stdout.splitlines():
    k, _, v = line.partition("=")
    _SANDBOX_ENV[k] = v
# Ensure PATH includes the conda env bin directory explicitly
_SANDBOX_ENV["PATH"] = f"{SANDBOX_PREFIX / 'bin'}:{_SANDBOX_ENV.get('PATH', '')}"
# Ensure the correct PYTHONEXECUTABLE is hinted for subprocesses if needed
_SANDBOX_ENV["PYTHONEXECUTABLE"] = str(SANDBOX_PREFIX / "bin/python")

# Default working directory
DEFAULT_ROOT = Path("/testbed").resolve()
DEFAULT_ROOT.mkdir(parents=True, exist_ok=True)

SERVER_FILE = Path(__file__).resolve().as_posix()

# Flask app
app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stateful shell process (kept for /command and /evaluate)
# ─────────────────────────────────────────────────────────────────────────────
def _start_shell() -> subprocess.Popen[str]:
    # Start bash with the specific conda environment activated
    # Using '-lc' ensures profile scripts are loaded, including conda activation logic.
    # We still pass _SANDBOX_ENV for robustness, though activation should set most things.
    return subprocess.Popen(
        ["/bin/bash", "-lc", f"conda activate {SANDBOX_PREFIX} && exec /bin/bash"], # Activate env within the interactive shell
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=DEFAULT_ROOT, env=_SANDBOX_ENV, text=True, bufsize=1, # Pass the sandbox env
        # The 'exec' replaces the initial shell, preventing an extra shell layer.
    )

SHELL_PROC = _start_shell()              # Initial launch
SHELL_LOCK = threading.Lock()

# --- Shell Execution Function (modified slightly for timeout/merge flexibility) ---
def _exec_shell(cmd: str, cwd: Path, timeout: Optional[int] = None, merge: bool = False) -> Tuple[str, str, int]:
    """
    Executes a shell command in the persistent, sandboxed bash process.

    Args:
        cmd: The command string to execute.
        cwd: The working directory for the command.
        timeout: Optional timeout in seconds.
        merge: If True, merge stdout and stderr into the first string result.

    Returns:
        A tuple (stdout, stderr, returncode). stderr might be empty if merge=True.
        returncode is -1 on shell crash, -9 on timeout, or the command's exit code.
    """
    global SHELL_PROC
    output_buffer = io.StringIO()
    error_buffer = io.StringIO() # Keep separate unless merging
    rc = -1 # Default error code

    def _target():
        nonlocal rc
        global SHELL_PROC # Allow modification in case of crash
        with SHELL_LOCK:
            if SHELL_PROC.poll() is not None:  # bash died -> restart
                print("Shell process died. Restarting.")
                SHELL_PROC = _start_shell()

            marker = uuid.uuid4().hex
            full_command = f"cd {cwd.as_posix()!r} && ({cmd}) && echo {marker} RC=$? || echo {marker} RC=$?\n"
            # print(f"Shell sending: {full_command!r}") # Debug

            try:
                SHELL_PROC.stdin.write(full_command)
                SHELL_PROC.stdin.flush()
            except (BrokenPipeError, IOError) as e:
                print(f"Shell pipe error on write: {e}. Attempting restart.")
                SHELL_PROC = _start_shell() # Restart for next call
                try:
                    SHELL_PROC.stdin.write(full_command)
                    SHELL_PROC.stdin.flush()
                except (BrokenPipeError, IOError) as e2:
                     print(f"Shell pipe error even after restart: {e2}.")
                     error_buffer.write(f"Shell communication error: {e2}\n")
                     rc = -1 # Indicate shell communication failure
                     return # Exit thread

            # --- Read output ---
            while True:
                try:
                    line = SHELL_PROC.stdout.readline()
                    # print(f"Shell recv: {line!r}") # Debug
                except IOError as e:
                    print(f"Shell pipe error on read: {e}. Assuming crash.")
                    error_buffer.write(f"Shell read error: {e}\n")
                    SHELL_PROC = _start_shell() # Restart for next call
                    rc = -1
                    break # Exit reading loop

                if not line: # EOF, bash exited unexpectedly
                    print("Shell process EOF reached unexpectedly. Restarting.")
                    error_buffer.write("Shell crashed or exited unexpectedly.\n")
                    SHELL_PROC = _start_shell() # Restart for next call
                    rc = -1
                    break # Exit reading loop

                if line.startswith(marker):
                    try:
                        rc = int(line[len(marker):].split('=')[-1].strip())
                    except (ValueError, IndexError):
                        print(f"Failed to parse return code from: {line!r}")
                        error_buffer.write(f"Failed to parse return code line: {line}\n")
                        rc = -1 # Parsing failed
                    break # Command finished

                # Append to buffer (conditionally merge later)
                output_buffer.write(line)

    # --- Timeout Handling ---
    if timeout is not None:
        th = threading.Thread(target=_target, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
             # Thread still running, means timeout
             print(f"Shell command timed out after {timeout}s.")
             # Try to interrupt the process group (more effective than killing Popen object alone)
             try:
                 pgid = os.getpgid(SHELL_PROC.pid)
                 os.killpg(pgid, signal.SIGTERM) # Send SIGTERM to the process group
                 time.sleep(0.5) # Give it a moment to terminate
                 if SHELL_PROC.poll() is None: # Still alive?
                     os.killpg(pgid, signal.SIGKILL) # Force kill
                 print("Sent kill signal due to timeout.")
             except ProcessLookupError:
                 print("Shell process already gone during timeout handling.")
             except Exception as e:
                 print(f"Error trying to kill shell process group: {e}")

             # Ensure shell is restarted for subsequent calls after a timeout kill
             SHELL_PROC = _start_shell()
             return "", "Command timed out", -9 # Use -9 convention for timeout
    else:
        # No timeout, run directly in the current thread
        _target()

    stdout_val = output_buffer.getvalue()
    stderr_val = error_buffer.getvalue()

    # Assign stderr based on exit code only if buffer is empty (shell errors)
    if rc != 0 and not stderr_val and not merge :
         stderr_val = f"Command exited with non-zero status: {rc}"
    elif rc != 0 and not stderr_val and merge:
         # Add exit code info to stdout if merging and stderr buffer is empty
         stdout_val += f"\nCommand exited with non-zero status: {rc}"


    if merge:
        return stdout_val + stderr_val, "", rc
    else:
        return stdout_val, stderr_val, rc


# ─────────────────────────────────────────────────────────────────────────────
# Stateful Jupyter Kernel process (New)
# ─────────────────────────────────────────────────────────────────────────────
KERNEL_MANAGER: Optional[jupyter_client.KernelManager] = None
KERNEL_CLIENT: Optional[jupyter_client.BlockingKernelClient] = None
KERNEL_LOCK = threading.Lock()

def _start_jupyter_kernel() -> Tuple[jupyter_client.KernelManager, jupyter_client.BlockingKernelClient]:
    """Starts a Jupyter kernel in the specified Conda environment."""
    global KERNEL_MANAGER, KERNEL_CLIENT
    print("Starting Jupyter kernel...")

    # Ensure previous kernel is shutdown if necessary
    _stop_jupyter_kernel()

    # Use MultiKernelManager if managing multiple kernel types, but KernelManager is fine for one.
    # We pass the environment variables, including the crucial PATH and PYTHONEXECUTABLE
    # pointing to the sandboxed environment.
    km = jupyter_client.KernelManager(
        # Specify the python executable directly for robustness
         kernel_cmd=[str(SANDBOX_PREFIX / 'bin/python'),
                     '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
         # Set the CWD for the kernel
         cwd=DEFAULT_ROOT.as_posix(),
    )
    # We don't pass env= explicitly here, relying on kernel_cmd and PATH in the Popen env.
    # If direct env passing is needed, it often involves lower-level process management.
    km.start_kernel()
    print(f"Kernel process started (PID: {km.kernel.pid}). Waiting for connection file...")

    kc = km.client()
    kc.start_channels() # Starts ZMQ channels

    try:
        # Wait for the kernel to be ready, polling is_alive()
        # This replaces wait_for_ready() which can sometimes hang
        # Use kc.hb_channel.is_beating() for a more direct heartbeat check
        for _ in range(300): # Wait up to 30 seconds (300 * 0.1s)
             if kc.is_alive() and kc.hb_channel.is_beating():
                 print("Kernel client connected and heartbeat received.")
                 break
             time.sleep(0.1)
        else:
             raise TimeoutError("Timeout waiting for kernel client to connect or heartbeat.")

        # Send an initial command to set the working directory within the kernel
        # This is more reliable than relying solely on the CWD of the launch process
        init_code = f"import os\nos.chdir({str(DEFAULT_ROOT)!r})"
        msg_id = kc.execute(init_code, store_history=False, silent=False)
        # Wait for this initial command to complete
        kc.get_shell_msg(timeout=10) # Wait for execute_reply
        # Consume potential IOPub messages from init
        while True:
            try:
                kc.get_iopub_msg(timeout=0.2)
            except queue.Empty:
                break
        print(f"Kernel initialized in CWD: {DEFAULT_ROOT}")


    except Exception as e:
        print(f"Error during kernel startup/connection: {e}")
        try:
            if km.is_alive():
                km.shutdown_kernel(now=True)
        except Exception as shutdown_e:
             print(f"Error shutting down kernel after startup failure: {shutdown_e}")
        raise RuntimeError("Failed to start or connect to Jupyter kernel.") from e

    KERNEL_MANAGER = km
    KERNEL_CLIENT = kc
    print("Jupyter kernel started successfully.")
    return km, kc

def _stop_jupyter_kernel():
    """Stops the currently running Jupyter kernel and client."""
    global KERNEL_MANAGER, KERNEL_CLIENT
    with KERNEL_LOCK: # Ensure exclusive access during shutdown
        if KERNEL_CLIENT:
            try:
                if KERNEL_CLIENT.is_alive():
                    KERNEL_CLIENT.stop_channels()
                KERNEL_CLIENT = None
                print("Kernel client channels stopped.")
            except Exception as e:
                print(f"Error stopping kernel client channels: {e}")

        if KERNEL_MANAGER:
            try:
                if KERNEL_MANAGER.is_alive():
                    KERNEL_MANAGER.shutdown_kernel(now=True) # Force shutdown
                KERNEL_MANAGER = None
                print("Kernel manager shutdown signal sent.")
            except Exception as e:
                print(f"Error shutting down kernel manager: {e}")

# --- Initialize Kernel on Startup ---
try:
    _start_jupyter_kernel()
except Exception as e:
    print(f"FATAL: Could not start Jupyter kernel on server startup: {e}")
    # Depending on requirements, you might exit here or allow the server
    # to run without kernel functionality for other endpoints.
    # sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_trace(tb: str) -> str:
    # Keep filtering specific paths if desired
    return "\n".join(
        ln for ln in tb.splitlines()
        if SERVER_FILE not in ln and "/flask/" not in ln and "jupyter_client" not in ln # Added jupyter_client filter
    )


def _run_with_timeout(fn, timeout: int, *args) -> Tuple[str, str, bool]:
    # This helper remains largely the same, but the function 'fn' it calls
    # will now be the Jupyter kernel execution wrapper.
    res, err = {}, {}
    timed_out_flag = [False] # Use list to allow modification inside thread

    def _target():
        try:
            o, e = fn(*args)
            res[0], err[0] = o, e
        except Exception:
            # Capture exceptions from the execution function itself
            err[0] = _clean_trace(traceback.format_exc())

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        timed_out_flag[0] = True
        # Timeout occurred. The underlying function (`_dispatch_jupyter`)
        # should ideally handle kernel interruption, but we signal timeout here.
        # Note: Truly interrupting the kernel execution robustly from here is hard.
        # jupyter_client allows sending interrupts, which the kernel might honor.
        # We'll primarily rely on the timeout within get_iopub_msg inside _dispatch_jupyter.
        print(f"Execution timed out after {timeout} seconds at timeout wrapper level.")
        return "", f"Timed out after {timeout} seconds", True

    # Return results captured by the thread, or empty strings if something went wrong
    return res.get(0, ""), err.get(0, ""), timed_out_flag[0]


# ─────────────────────────────────────────────────────────────────────────────
# Dispatchers
# ─────────────────────────────────────────────────────────────────────────────

# _exec_python is removed, replaced by _dispatch_jupyter

def _exec_apply_patch(block: str, cwd: Path) -> Tuple[str, str]:
    # This function remains the same, handling the patch application directly.
    original_cwd = Path.cwd()
    try:
        os.chdir(cwd) # Change CWD for apply_patch logic
        res = apply_patch.process_patch(
            block, apply_patch.open_file, apply_patch.write_file, apply_patch.remove_file
        )
        return res + '\n', ''
    except Exception:
        return '', _clean_trace(traceback.format_exc())
    finally:
        os.chdir(original_cwd) # Change back CWD


def _dispatch_jupyter(cell: str, cwd: Path) -> Tuple[str, str]:
    """
    Executes a code cell using the persistent Jupyter kernel.
    Handles the '%%bash apply_patch' case separately.

    Args:
        cell: The code string to execute (can be multiline, mixed Python/shell).
        cwd: The desired working directory (Note: kernel CWD is set at start,
             use %cd or !cd within the cell for dynamic changes).

    Returns:
        Tuple[str, str]: Combined stdout and stderr.
    """
    global KERNEL_CLIENT, KERNEL_MANAGER # Allow access for potential restart

    cell = textwrap.dedent(cell) # Keep dedent

    # --- Special Case: Apply Patch ---
    # Keep this logic exactly as before
    if cell.lstrip().startswith('%%bash') and 'apply_patch' in cell:
        lines, cap, patch = cell.splitlines(), False, []
        for ln in lines:
            # Simplified EOF detection slightly
            if '<<EOF' in ln: # Handles '<<EOF"' and '<<EOF'
                cap = True
                # Skip the line with <<EOF itself
                continue
            if cap and ln.strip() == 'EOF':
                cap = False # Stop capturing
                break
            if cap:
                patch.append(ln)
        if patch: # Only run if patch content was found
             print("Dispatching to _exec_apply_patch")
             # Use DEFAULT_ROOT or decide if cwd needs to be passed differently
             return _exec_apply_patch("\n".join(patch), DEFAULT_ROOT)
        else:
             print("Warning: 'apply_patch' cell found but no patch content captured.")
             # Fall through to Jupyter execution or return error? Let's fall through for now.


    # --- Jupyter Kernel Execution ---
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    traceback_out: Optional[str] = None # Store error traceback if kernel sends one

    with KERNEL_LOCK:
        # --- Check Kernel Status and Restart if Necessary ---
        if KERNEL_CLIENT is None or not KERNEL_CLIENT.is_alive() or not KERNEL_CLIENT.hb_channel.is_beating():
            print("Kernel client is not alive or not beating. Attempting restart...")
            try:
                _start_jupyter_kernel() # This will reassign KERNEL_CLIENT and KERNEL_MANAGER
                if KERNEL_CLIENT is None: # Check if restart failed
                     raise RuntimeError("Kernel restart failed.")
                print("Kernel restarted successfully.")
            except Exception as e:
                print(f"Failed to restart kernel: {e}")
                return "", f"Kernel unavailable: Failed to restart kernel: {e}"

        # --- Execute Code ---
        try:
            # Ensure CWD is correct *within the kernel* if needed per-cell.
            # The most reliable way is via magic commands in the cell itself.
            # Example: prepend f"%cd {cwd.as_posix()!r}\n" + cell if needed,
            # but the user should ideally control CWD via the cell content.
            # We set the initial CWD when starting the kernel.

            msg_id = KERNEL_CLIENT.execute(cell, store_history=False, silent=False)
            print(f"Sent code to kernel, msg_id: {msg_id}")

            # --- Process Execution Replies ---
            # Wait for the 'execute_reply' message on the shell channel
            shell_reply = KERNEL_CLIENT.get_shell_msg(timeout=60) # Timeout for shell reply
            status = shell_reply['content']['status']
            print(f"Shell reply status: {status}")

            if status == 'error':
                # Error occurred during execution compilation/setup
                err_content = shell_reply['content']
                traceback_lines = err_content.get('traceback', [])
                traceback_out = "\n".join(traceback_lines)
                stderr_buf.write(f"Execution Error: {err_content.get('ename', 'Unknown Error')}\n")
                stderr_buf.write(f"Error Value: {err_content.get('evalue', 'N/A')}\n")
                if traceback_out:
                     stderr_buf.write("Traceback:\n")
                     stderr_buf.write(_clean_trace(traceback_out) + "\n") # Use cleaned traceback

            # Process messages published on the iopub channel (stdout, stderr, etc.)
            # Loop until execution state is 'idle'
            # Use a longer timeout here as this covers actual code execution time
            iopub_timeout = 60 # Adjust as needed (total time for cell execution)
            start_time = time.monotonic()
            while True:
                try:
                    # Check elapsed time against timeout
                    elapsed = time.monotonic() - start_time
                    remaining_timeout = max(0.1, iopub_timeout - elapsed) # Ensure non-negative timeout

                    msg = KERNEL_CLIENT.get_iopub_msg(timeout=remaining_timeout)
                    msg_type = msg['header']['msg_type']
                    content = msg['content']

                    # print(f"IOPub msg type: {msg_type}") # Debug

                    if msg_type == 'status':
                        if content['execution_state'] == 'idle':
                            print("Kernel status is idle, execution finished.")
                            break # Idle means execution finished for this request
                    elif msg_type == 'stream':
                        stream_name = content['name'] # 'stdout' or 'stderr'
                        if stream_name == 'stdout':
                            stdout_buf.write(content['text'])
                        elif stream_name == 'stderr':
                            stderr_buf.write(content['text'])
                    elif msg_type == 'execute_result' or msg_type == 'display_data':
                         # Output from display() or last expression (if not ending with ';')
                         data = content.get('data', {})
                         text_plain = data.get('text/plain')
                         if text_plain:
                             stdout_buf.write(text_plain + "\n") # Append result/display to stdout
                    elif msg_type == 'error':
                        # Runtime errors during execution are published here
                        traceback_lines = content.get('traceback', [])
                        traceback_out = "\n".join(traceback_lines) # Capture for later
                        stderr_buf.write(f"Runtime Error: {content.get('ename', 'Unknown Error')}\n")
                        stderr_buf.write(f"Error Value: {content.get('evalue', 'N/A')}\n")
                        if traceback_out:
                             stderr_buf.write("Traceback:\n")
                             stderr_buf.write(_clean_trace(traceback_out) + "\n") # Use cleaned traceback

                except queue.Empty:
                    # Timeout waiting for IOPub messages
                    # This might mean the code is finished, or it genuinely timed out
                    # Check kernel status again to be sure
                    current_status_msg_id = KERNEL_CLIENT.kernel_info()
                    current_status_reply = KERNEL_CLIENT.get_shell_msg(timeout=5)
                    if current_status_reply['content']['status'] == 'ok':
                        # If kernel info request succeeds, assume previous command finished
                         print("IOPub timeout, but kernel is responsive. Assuming execution finished.")
                         break
                    else:
                        stderr_buf.write(f"\nTimeout waiting for kernel execution response (IOPub) after {iopub_timeout}s.\n")
                        # Attempt kernel interrupt?
                        # KERNEL_MANAGER.interrupt_kernel() # Requires KERNEL_MANAGER to be accessible
                        print(f"Timeout waiting for IOPub message after {iopub_timeout} seconds.")
                        return stdout_buf.getvalue(), stderr_buf.getvalue() # Return whatever we got

                except Exception as e:
                    stderr_buf.write(f"\nError processing kernel message: {e}\n")
                    traceback.print_exc() # Print full trace for server logs
                    break # Stop processing on unexpected error

        except Exception as e:
            # Catch errors during the execute call or message handling setup
            print(f"Error during kernel communication: {e}")
            stderr_buf.write(f"Server Error during kernel communication: {_clean_trace(traceback.format_exc())}\n")
            # Attempt to restart the kernel in case it's wedged
            try:
                 print("Attempting kernel restart after communication error...")
                 _start_jupyter_kernel()
            except Exception as restart_e:
                 print(f"Failed to restart kernel after error: {restart_e}")
                 stderr_buf.write(f"Kernel restart also failed: {restart_e}\n")


    # Return combined outputs
    # If a traceback was received via 'error' message type, ensure it's in stderr
    final_stderr = stderr_buf.getvalue()
    # This check avoids duplicating tracebacks if they were already written during stream/error handling
    if traceback_out and _clean_trace(traceback_out) not in final_stderr:
         final_stderr += "\n--- Traceback ---\n" + _clean_trace(traceback_out)

    return stdout_buf.getvalue(), final_stderr


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
        elif ln.startswith('GIT binary patch') or 'Binary files' in ln: # Added GIT binary patch case
            binary = True
            block.append(ln) # Keep the diff line for context if needed later
            # Skip subsequent binary diff lines until next file
        elif binary and (ln.startswith('literal ') or ln.startswith('delta ')):
             continue # Skip git binary patch data lines
        else:
            block.append(ln)
    if block and not binary:
        cleaned.extend(block)
    return '\n'.join(cleaned)


def get_git_patch(base_commit: str, workdir: Path = DEFAULT_ROOT) -> str:
    cwd = workdir.resolve()
    # Ensure CWD exists
    cwd.mkdir(parents=True, exist_ok=True)

    print(f"Generating git patch against {base_commit} in {cwd}")

    # Use _exec_shell for consistency and environment handling
    def run_shell_cmd(cmd):
        print(f"Running: {cmd}")
        out, err, rc = _exec_shell(cmd, cwd=cwd, timeout=60, merge=True)
        print(f"Output:\n{out}")
        if rc != 0:
            # Log error but don't necessarily raise immediately, git diff might still work
            print(f"Warning: Shell command failed with rc={rc}:\n{err}")
        return out+err # Return combined output

    # Clean nested .git dirs more robustly
    run_shell_cmd('find . -mindepth 1 -type d -name .git -exec echo "Removing nested git dir: {}" \; -exec rm -rf {} \;')

    # Add all changes, including untracked files
    run_shell_cmd('git add -A')

    # Remove binary files - using the existing logic within a shell command
    # This command is complex; ensure quoting works. Using ''' might be safer.
    remove_bin_cmd = r'''
    git status --porcelain | while IFS= read -r line; do
        # Extract filename, handling spaces and potential quotes
        fname=$(echo "$line" | sed -e 's/^[ MA D??RMCU][ MA D??RMCU] "//' -e 's/"$//')
        # Check if it exists and is a file
        if [ -f "$fname" ]; then
            # Check if git thinks it's binary OR file command identifies binary types
            # Using check-attr might be more reliable if .gitattributes is used
            is_binary=false
            if git check-attr -a -- "$fname" | grep -q "binary: set"; then
                is_binary=true
            elif file -b --mime-encoding "$fname" | grep -q "binary"; then
                is_binary=true
            # Add more checks if needed (e.g., executables by permission)
            # elif [ -x "$fname" ]; then is_binary=true; fi
            fi

            if $is_binary; then
                echo "Removing binary file detected: $fname"
                git rm --cached -f "$fname" >/dev/null 2>&1 || echo "Failed git rm binary" # Remove from index
                rm -f "$fname" >/dev/null 2>&1 || echo "Failed rm binary" # Remove from working dir
            fi
        fi
    done
    '''
    run_shell_cmd(remove_bin_cmd)

    # Regenerate the diff after removing binaries
    diff_cmd = f'git diff --no-color --cached {base_commit}'
    diff_out, _, rc = _exec_shell(diff_cmd, cwd=cwd, timeout=60)
    if rc != 0:
        raise RuntimeError(f"git diff command failed with exit code {rc}. Output:\n{diff_out}")

    print(f"Raw git diff length: {len(diff_out)}")
    cleaned_patch = _remove_binary_diffs(diff_out)
    print(f"Cleaned git patch length: {len(cleaned_patch)}")
    return cleaned_patch


###############################################################################
# Patch‑evaluation helper (unchanged structure, uses updated _exec_shell)
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
        # Added fields for more detail
        'eval_stdout': '',
        'eval_stderr': '',
        'eval_returncode': -1,
    }
    if not model_patch.strip():
        report['empty_generation'] = True
        print("Evaluation skipped: Empty patch received.")
        return report

    # Use temporary directory within container for clarity
    tmp_dir = Path(tempfile.gettempdir())
    patch_path = tmp_dir / 'patch.diff'
    script_path = tmp_dir / 'eval.sh'

    try:
        patch_path.write_text(model_patch)
        script_path.write_text(eval_script)
        script_path.chmod(0o755)
    except Exception as e:
        report['error_eval'] = True
        report['eval_stderr'] = f"Error writing temp files: {e}"
        print(f"Evaluation error: Failed to write patch/script files: {e}")
        return report

    cwd = workdir.resolve()
    print(f"Evaluating patch in {cwd}")

    # Apply patch using git apply first, then fallback to patch command
    # Run within the persistent shell for environment consistency
    apply_cmd = (
        f'cd {cwd.as_posix()!r} && ' # Ensure correct directory
        f'(git apply -v --whitespace=fix --recount {patch_path.as_posix()!r} && echo APPLY_PATCH_PASS) || '
        f'(echo "Git apply failed, trying patch command..." && patch --batch --fuzz=5 -p1 -i {patch_path.as_posix()!r} && echo APPLY_PATCH_PASS || '
        f' echo APPLY_PATCH_FAIL)'
    )

    print(f"Running apply command: {apply_cmd}")
    # Use merge=True as we just need the combined output and success marker
    apply_out, _, apply_rc = _exec_shell(apply_cmd, DEFAULT_ROOT, timeout=300, merge=True) # Use DEFAULT_ROOT as CWD is handled in cmd
    report['apply_output'] = apply_out.strip()
    print(f"Apply patch output:\n{report['apply_output']}")

    if 'APPLY_PATCH_FAIL' in report['apply_output'] or apply_rc != 0:
        report['failed_apply_patch'] = True
        print("Evaluation failed: Patch application failed.")
        # Clean up temporary files
        patch_path.unlink(missing_ok=True)
        script_path.unlink(missing_ok=True)
        return report

    print("Patch applied successfully. Running evaluation script...")
    # Run evaluation script using the persistent shell
    eval_cmd = script_path.as_posix()
    eval_out, eval_err, eval_rc = _exec_shell(eval_cmd, cwd, timeout=timeout, merge=False) # Keep stdout/stderr separate

    report['eval_stdout'] = eval_out
    report['eval_stderr'] = eval_err
    report['eval_returncode'] = eval_rc
    report['eval_output'] = eval_out + ("\n--- STDERR ---\n" + eval_err if eval_err else "") # Combine for compatibility

    print(f"Eval script stdout:\n{eval_out}")
    if eval_err:
        print(f"Eval script stderr:\n{eval_err}")
    print(f"Eval script return code: {eval_rc}")


    if eval_rc == -9: # Specific timeout code from _exec_shell
        report['test_timeout'] = True
        print("Evaluation timed out.")
    elif eval_rc != 0:
        report['error_eval'] = True
        print(f"Evaluation script failed with return code {eval_rc}.")

    # Check for resolution marker *only* if the script succeeded (rc=0)
    if eval_rc == 0:
        # Check both stdout and stderr for the resolved marker, case-insensitive
        resolved_marker = 'RESOLVED'
        if resolved_marker in eval_out.upper() or resolved_marker in eval_err.upper():
             report['resolved'] = True
             print("Resolution marker found.")
        else:
             print("Evaluation script succeeded, but resolution marker not found.")
    else:
         # If script failed or timed out, it cannot be resolved
         report['resolved'] = False


    # Clean up temporary files
    patch_path.unlink(missing_ok=True)
    script_path.unlink(missing_ok=True)

    return report


###############################################################################
# Flask routes
###############################################################################

@app.route('/alive', methods=['GET'])
def alive():
    # Check kernel status too?
    kernel_ok = False
    if KERNEL_CLIENT and KERNEL_CLIENT.is_alive() and KERNEL_CLIENT.hb_channel.is_beating():
         kernel_ok = True
    shell_ok = False
    if SHELL_PROC and SHELL_PROC.poll() is None:
         shell_ok = True
    return jsonify({'status': 'ok', 'kernel_running': kernel_ok, 'shell_running': shell_ok}), 200


@app.route('/execute', methods=['POST'])
def execute_endpoint():
    start = time.time()
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    payload = request.get_json()
    if not isinstance(payload, list) or not payload:
        return jsonify({'error': 'Expected non-empty list'}), 400

    results = []
    for idx, msg in enumerate(payload):
        # Simplified check: process any message with 'arguments' assumed to contain 'input' code
        # Adapt this if the structure is more complex (e.g., different function names)
        if 'arguments' not in msg or 'call_id' not in msg :
            print(f"Skipping message at index {idx}: Missing 'arguments' or 'call_id'. Keys: {list(msg.keys())}")
            continue

        try:
            args = json.loads(msg.get('arguments', '{}'))
            code = args.get('input')
            call_id = msg['call_id'] # Use the provided call_id

            if code is None:
                 print(f"Skipping message at index {idx}, call_id {call_id}: 'input' missing in arguments.")
                 results.append({
                     'index': idx,
                     'call_id': call_id,
                     'output': '',
                     'error': "'input' code missing in arguments",
                     'timed_out': False,
                     'duration': 0.0
                 })
                 continue

            print(f"Executing code for index {idx}, call_id {call_id}...")
            o_start = time.time()
            # Use _dispatch_jupyter via _run_with_timeout
            # Pass DEFAULT_ROOT as cwd - kernel runs there initially.
            # User code needs %cd or !cd to change directory if needed.
            out, err, timed = _run_with_timeout(_dispatch_jupyter, 600, code, DEFAULT_ROOT) # Increased timeout to 10 mins
            o_duration = round(time.time() - o_start, 3)

            results.append({
                'index': idx,
                'call_id': call_id,
                'output': out,
                'error': err,
                'timed_out': timed,
                'duration': o_duration
            })
            print(f"Finished execution for index {idx}, call_id {call_id}. Duration: {o_duration}s, Timed Out: {timed}")

        except json.JSONDecodeError as exc:
            # Error decoding the 'arguments' string
            err_msg = f"JSONDecodeError in arguments: {exc}"
            print(f"Error processing message at index {idx}: {err_msg}")
            results.append({
                'index': idx,
                'call_id': msg.get('call_id', f'error_{idx}'), # Attempt to get call_id
                'output': '',
                'error': err_msg,
                'timed_out': False,
                'duration': 0.0
            })
            continue
        except Exception as e:
             # Catch unexpected errors during processing loop
             err_msg = f"Unexpected error processing message: {_clean_trace(traceback.format_exc())}"
             print(f"Error processing message at index {idx}: {err_msg}")
             results.append({
                 'index': idx,
                 'call_id': msg.get('call_id', f'error_{idx}'),
                 'output': '',
                 'error': err_msg,
                 'timed_out': False,
                 'duration': round(time.time() - start, 3) # Duration up to the error
             })
             continue # Continue to next message if possible


    if not results:
         # This might happen if the input list had items, but none matched the expected format
         print("No executable messages found in the payload.")
         return jsonify({'error': 'No messages with executable code found in payload'}), 400

    overall_duration = round(time.time() - start, 3)
    print(f"Overall execution finished. Duration: {overall_duration}s")
    return jsonify({'results': results, 'overall_duration': overall_duration})


@app.route('/diff', methods=['POST'])
def diff_endpoint():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json()
    base_commit = data.get('base_commit')
    if not base_commit:
        return jsonify({'error': 'base_commit missing'}), 400

    # Allow specifying directory, default to DEFAULT_ROOT
    workdir_str = data.get('dir', DEFAULT_ROOT.as_posix())
    try:
        workdir = Path(workdir_str).resolve()
        # Basic check to prevent escaping the intended workspace significantly
        # You might want stricter checks depending on security needs
        if not str(workdir).startswith(str(DEFAULT_ROOT.parent)):
             return jsonify({'error': f'Workdir {workdir_str} is outside allowed paths'}), 400
        workdir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
    except Exception as e:
        return jsonify({'error': f'Invalid workdir path: {e}'}), 400


    try:
        print(f"Diff endpoint: Requesting patch against {base_commit} in {workdir}")
        patch = get_git_patch(base_commit, workdir)
        print(f"Diff endpoint: Generated patch of length {len(patch)}")
        return jsonify({'patch': patch})
    except Exception as exc:
        print(f"Error generating git patch: {exc}")
        # traceback.print_exc() # Log full traceback for debugging
        return jsonify({'error': f'Error generating patch: {exc}'}), 500


@app.route('/evaluate', methods=['POST'])
def evaluate_endpoint():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json()
    model_patch = data.get('model_patch', '')
    eval_script = data.get('eval_script', '')

    # Allow specifying directory, default to DEFAULT_ROOT
    workdir_str = data.get('dir', DEFAULT_ROOT.as_posix())
    try:
        workdir = Path(workdir_str).resolve()
        if not str(workdir).startswith(str(DEFAULT_ROOT.parent)):
             return jsonify({'error': f'Workdir {workdir_str} is outside allowed paths'}), 400
        workdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return jsonify({'error': f'Invalid workdir path: {e}'}), 400

    timeout = int(data.get('timeout', 1800)) # Default 30 minutes

    try:
        print(f"Evaluate endpoint: Evaluating patch in {workdir} with timeout {timeout}s")
        report = _evaluate_patch(model_patch, eval_script, workdir, timeout)
        print(f"Evaluate endpoint: Evaluation complete. Resolved: {report.get('resolved')}")
        return jsonify({'report': report})
    except Exception as e:
        print(f"Error during patch evaluation: {e}")
        # traceback.print_exc()
        return jsonify({'error': f'Error during evaluation: {e}'}), 500


@app.route('/command', methods=['POST'])
def command_endpoint():
    """Run a single shell command and return stdout/stderr/rc."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    data = request.get_json()
    cmd = data.get('cmd') or data.get('command')
    if not cmd:
        return jsonify({'error': 'cmd missing'}), 400

    workdir_str = data.get('dir', DEFAULT_ROOT.as_posix())
    try:
        workdir = Path(workdir_str).resolve()
        if not str(workdir).startswith(str(DEFAULT_ROOT.parent)):
             return jsonify({'error': f'Workdir {workdir_str} is outside allowed paths'}), 400
        workdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return jsonify({'error': f'Invalid workdir path: {e}'}), 400

    timeout = int(data.get('timeout', 120)) # Default 2 minutes
    merge_output = data.get('merge_output', False) # Option to merge stdout/stderr

    start = time.time()
    print(f"Command endpoint: Running cmd='{cmd}' in '{workdir}' with timeout={timeout} merge={merge_output}")
    try:
        # Use the persistent shell (_exec_shell handles CWD and env)
        out, err, rc = _exec_shell(cmd, workdir, timeout=timeout, merge=merge_output)
        duration = round(time.time() - start, 3)
        print(f"Command endpoint: Finished. rc={rc}, duration={duration}s")
        return jsonify({
            'stdout': out,
            'stderr': err,
            'returncode': rc,
            'duration': duration
        })
    except Exception as e:
         print(f"Error executing command via _exec_shell: {e}")
         # traceback.print_exc()
         return jsonify({
             'stdout': '',
             'stderr': f'Server error executing command: {e}',
             'returncode': -1, # Indicate server-side failure
             'duration': round(time.time() - start, 3)
         }), 500


@app.route('/upload_file', methods=['POST'])
def upload_file_endpoint():
    """Upload a file (or zip archive) to the server.

    Query params:
      destination – absolute/relative path where the file/dir should land inside /testbed
      recursive   – 'true' if the upload is a zip of a directory to be extracted
    """
    dest_param = request.args.get('destination')
    if not dest_param:
        return jsonify({'error': 'destination query parameter missing'}), 400

    recursive = request.args.get('recursive', 'false').lower() == 'true'

    if 'file' not in request.files:
        return jsonify({'error': 'Multipart file field named "file" missing'}), 400
    file_storage = request.files['file']
    if not file_storage.filename:
         return jsonify({'error': 'Uploaded file has no filename'}), 400


    try:
        # Sanitize destination path and ensure it's within DEFAULT_ROOT
        # Normalize to prevent '..' escapes etc.
        target_path = (DEFAULT_ROOT / Path(dest_param).name).resolve() # Basic: use filename under DEFAULT_ROOT
        # More flexible: allow subdirs but ensure it stays within DEFAULT_ROOT
        if Path(dest_param).is_absolute():
             # If abs path provided, ensure it's within DEFAULT_ROOT
             candidate_path = Path(dest_param).resolve()
             if not str(candidate_path).startswith(str(DEFAULT_ROOT)):
                  return jsonify({'error': 'Absolute destination path is outside allowed directory'}), 400
             target_path = candidate_path
        else:
             # Relative path, resolve it safely within DEFAULT_ROOT
             target_path = (DEFAULT_ROOT / dest_param).resolve()
             if not str(target_path).startswith(str(DEFAULT_ROOT)):
                  return jsonify({'error': 'Relative destination path resolves outside allowed directory'}), 400


        print(f"Upload endpoint: Destination '{dest_param}', Recursive: {recursive}, Target Path: {target_path}")

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if recursive:
            # Expect a zip file – extract it to the target *directory*
            # Ensure target_path refers to a directory name, not a file name for extraction
            if target_path.exists() and not target_path.is_dir():
                 return jsonify({'error': f'Cannot extract archive: target path {target_path} exists and is not a directory'}), 400
            target_path.mkdir(parents=True, exist_ok=True) # Ensure target dir exists

            with tempfile.TemporaryDirectory(prefix="upload_zip_") as tmpdir:
                archive_path = Path(tmpdir) / 'upload.zip'
                file_storage.save(archive_path)
                print(f"Saved uploaded zip to {archive_path}")

                import zipfile
                try:
                    with zipfile.ZipFile(archive_path, 'r') as zf:
                        # Basic security: Check for potentially malicious filenames? (Optional)
                        # for member in zf.infolist():
                        #    if member.filename.startswith('/') or '..' in member.filename:
                        #        raise ValueError(f"Zip contains potentially unsafe path: {member.filename}")
                        zf.extractall(target_path)
                    print(f"Extracted archive to {target_path}")
                except zipfile.BadZipFile:
                    return jsonify({'error': 'Uploaded file is not a valid zip archive'}), 400
                except Exception as zip_e:
                    return jsonify({'error': f'Error extracting zip file: {zip_e}'}), 500
        else:
            # Save single file - ensure target_path refers to the *file* name
            # If dest_param was 'mydir/', target_path might be '.../mydir'. We need '.../mydir/filename'.
            if target_path.is_dir():
                 final_file_path = (target_path / file_storage.filename).resolve()
                 # Re-check bounds after adding filename
                 if not str(final_file_path).startswith(str(DEFAULT_ROOT)):
                     return jsonify({'error': 'Final file path resolves outside allowed directory'}), 400
                 target_path = final_file_path

            elif not str(target_path.parent).startswith(str(DEFAULT_ROOT)):
                 # Check parent again in case the resolved path was tricky
                 return jsonify({'error': 'Final file path parent resolves outside allowed directory'}), 400


            file_storage.save(target_path)
            print(f"Saved single file to {target_path}")

    except Exception as exc:  # pylint: disable=broad-except
        error_message = f"Error during file upload/processing: {exc}"
        print(error_message)
        # traceback.print_exc() # Log full trace for debugging
        return jsonify({'error': error_message}), 500

    return jsonify({'status': 'ok', 'path': str(target_path), 'recursive': recursive}), 200


###############################################################################
# Entrypoint & Cleanup
###############################################################################
import atexit
import signal

def cleanup():
    print("Server shutting down. Cleaning up resources...")
    _stop_jupyter_kernel()
    # Stop shell process
    global SHELL_PROC
    if SHELL_PROC and SHELL_PROC.poll() is None:
        print("Terminating shell process...")
        try:
             # Try terminating process group first
             pgid = os.getpgid(SHELL_PROC.pid)
             os.killpg(pgid, signal.SIGTERM)
             SHELL_PROC.wait(timeout=2)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
             if SHELL_PROC.poll() is None: # Still alive?
                 print("Shell process did not terminate gracefully, sending SIGKILL...")
                 try:
                      os.killpg(pgid, signal.SIGKILL)
                 except Exception as e:
                     print(f"Failed to kill shell process group: {e}")
                     # Fallback to killing just the Popen object
                     try:
                          SHELL_PROC.kill()
                     except Exception as kill_e:
                           print(f"Fallback shell kill failed: {kill_e}")

        SHELL_PROC = None
        print("Shell process cleanup attempted.")

# Register cleanup function to be called on exit
atexit.register(cleanup)
# Handle common termination signals gracefully
signal.signal(signal.SIGTERM, lambda signum, frame: exit(0))
signal.signal(signal.SIGINT, lambda signum, frame: exit(0))


if __name__ == '__main__':
    port = 4444
    print(f"Starting Flask server on 0.0.0.0:{port}")
    print(f"Default root directory: {DEFAULT_ROOT}")
    print(f"Using Conda environment: {SANDBOX_PREFIX}")
    # Use threaded=True for handling concurrent requests if needed by timeout logic, etc.
    # Use debug=False in production/deployment
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)
    # Note: app.run() is blocking. Cleanup happens via atexit/signals.
