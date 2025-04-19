#!/usr/bin/env python3

import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import the apply_patch functionality
from apply_patch import process_patch, open_file, write_file, remove_file

# For Jupyter notebook execution
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError
from queue import Queue

# Maximum time to wait for command execution (in seconds)
EXECUTION_TIMEOUT = 120.0
LINE_LIMIT = 2000  # Maximum number of output lines to return

# Global kernel session for stateful execution
notebook_kernel = None


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time."""

    def signal_handler(signum, frame):
        raise TimeoutException("Execution timed out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)


def capture_output(cmd: str) -> Tuple[int, str]:
    """
    Execute a command and capture its output.

    Args:
        cmd: Command to execute

    Returns:
        Tuple of (return_code, output)
    """
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    output_lines = []
    line_count = 0
    truncated = False

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break

        output_lines.append(line)
        line_count += 1

        if line_count >= LINE_LIMIT:
            truncated = True
            process.terminate()
            break

    return_code = process.wait()
    output = ''.join(output_lines)

    if truncated:
        output += f"\n[Output truncated after {LINE_LIMIT} lines]"

    return return_code, output


def run_command(cmd: str) -> str:
    """
    Run a shell command with timeout protection.

    Args:
        cmd: Command to execute

    Returns:
        Command output
    """
    try:
        with time_limit(EXECUTION_TIMEOUT):
            return_code, output = capture_output(cmd)
            return output
    except TimeoutException:
        return f"Execution timed out after {EXECUTION_TIMEOUT} seconds."


def get_or_create_notebook_kernel():
    """
    Get or create a Jupyter notebook kernel for stateful execution.

    Returns:
        NotebookClient: A notebook client with a running kernel
    """
    global notebook_kernel

    if notebook_kernel is None:
        # Create a new notebook with a single cell
        notebook = nbformat.v4.new_notebook()

        # Create a client to execute the notebook
        notebook_kernel = NotebookClient(
            notebook,
            timeout=EXECUTION_TIMEOUT,
            kernel_name="python3",
            resources={}
        )

        # Start the kernel
        notebook_kernel.start_new_kernel()

        # Initialize the kernel with necessary imports
        init_cell = nbformat.v4.new_code_cell("""
import sys
import os
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
        """)

        try:
            notebook_kernel.execute_cell(init_cell)
        except Exception as e:
            print(f"Error initializing kernel: {str(e)}")

    return notebook_kernel


def execute_python_notebook(code: str) -> str:
    """
    Execute Python code in a Jupyter notebook environment.

    Args:
        code: Python code to execute

    Returns:
        Execution output
    """
    kernel = get_or_create_notebook_kernel()

    # Create a new cell with the code
    cell = nbformat.v4.new_code_cell(code)

    try:
        # Execute the cell
        kernel.execute_cell(cell)

        # Collect outputs
        outputs = []
        for output in cell.outputs:
            if output.output_type == 'stream':
                outputs.append(output.text)
            elif output.output_type == 'display_data' and 'text/plain' in output.data:
                outputs.append(output.data['text/plain'])
            elif output.output_type == 'execute_result' and 'text/plain' in output.data:
                outputs.append(output.data['text/plain'])
            elif output.output_type == 'error':
                # Format error output similar to a traceback
                err_name = output.ename
                err_value = output.evalue
                err_traceback = '\n'.join(output.traceback)
                outputs.append(f"{err_name}: {err_value}\n{err_traceback}")

        # Join all outputs
        return '\n'.join(outputs)

    except CellTimeoutError:
        return f"Execution timed out after {EXECUTION_TIMEOUT} seconds."
    except CellExecutionError as e:
        return f"Execution error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"


def handle_apply_patch(patch_text: str) -> str:
    """
    Apply a patch to the codebase.

    Args:
        patch_text: Patch text in the specified format

    Returns:
        Result of applying the patch
    """
    try:
        result = process_patch(patch_text, open_file, write_file, remove_file)
        return result
    except Exception as e:
        return f"Error applying patch: {str(e)}\n{traceback.format_exc()}"


def process_input(input_text: str) -> str:
    """
    Process the input based on its type (Python code, shell command, or patch).

    Args:
        input_text: Input text to process

    Returns:
        Execution output
    """
    # Check if it's a bash apply_patch command
    if input_text.startswith("%%bash\napply_patch"):
        # Extract the patch text
        pattern = r'apply_patch <<"EOF"\n(.*?)\nEOF'
        match = re.search(pattern, input_text, re.DOTALL)
        if match:
            patch_text = match.group(1)
            return handle_apply_patch(patch_text)
        else:
            return "Error: Invalid apply_patch format"

    # Check if it's a shell command (either with ! or %%bash)
    if input_text.startswith("!"):
        cmd = input_text[1:]
        return run_command(cmd)
    elif input_text.startswith("%%bash"):
        cmd = input_text[len("%%bash\n"):]
        return run_command(cmd)

    # Check if it's a special cell magic
    if input_text.startswith("%%"):
        magic_type = input_text.split("\n")[0]
        return f"Magic command {magic_type} is not supported."

    # Otherwise, assume it's Python code
    return execute_python_notebook(input_text)


def parse_agent_message(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse an agent message to extract function calls.

    Args:
        message: Agent message dictionary

    Returns:
        Extracted function call if present, None otherwise
    """
    if message.get('type') == 'function_call':
        return {
            'name': message.get('name'),
            'arguments': message.get('arguments'),
            'call_id': message.get('call_id')
        }
    return None


def handle_agent_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle an agent request by executing the requested action.

    Args:
        request_data: Request data containing the agent message

    Returns:
        Result of the execution
    """
    messages = request_data.get('messages', [])

    # Find the last function call in the messages
    function_call = None
    for message in reversed(messages):
        call = parse_agent_message(message)
        if call:
            function_call = call
            break

    if not function_call:
        return {
            'error': 'No function call found in messages'
        }

    # Currently, we only support the 'python' function
    if function_call['name'] == 'python':
        try:
            arguments = json.loads(function_call['arguments'])
            input_text = arguments.get('input', '')
            output = process_input(input_text)

            return {
                'call_id': function_call['call_id'],
                'content': output,
                'status': 'success'
            }
        except Exception as e:
            return {
                'call_id': function_call['call_id'],
                'error': str(e),
                'status': 'error'
            }
    else:
        return {
            'call_id': function_call['call_id'],
            'error': f"Unsupported function: {function_call['name']}",
            'status': 'error'
        }


def cleanup_notebook_kernel():
    """Clean up the notebook kernel when the server shuts down."""
    global notebook_kernel
    if notebook_kernel:
        try:
            notebook_kernel.kc.stop_channels()
            notebook_kernel.km.shutdown_kernel()
        except:
            pass


def main():
    """
    Main entry point for the server.
    """
    import argparse
    import atexit

    # Register cleanup function to handle kernel shutdown
    atexit.register(cleanup_notebook_kernel)

    parser = argparse.ArgumentParser(
        description='Execute Python code or terminal commands in a Jupyter notebook environment')
    parser.add_argument('--port', type=int, default=4444, help='Port to listen on')
    parser.add_argument('--host', type=str, default='localhost', help='Host to bind to')
    args = parser.parse_args()

    # For simplicity, we'll use Flask for the web server
    try:
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        @app.route('/execute', methods=['POST'])
        def execute():
            request_data = request.json
            result = handle_agent_request(request_data)
            return jsonify(result)

        # Initialize the notebook kernel on startup
        get_or_create_notebook_kernel()
        print(f"Jupyter kernel initialized")
        print(f"Server running on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port)

    except ImportError as e:
        required_packages = ["flask", "nbformat", "nbclient"]
        missing_package = next((pkg for pkg in required_packages if pkg in str(e)), None)
        if missing_package:
            print(f"{missing_package} is not installed. Please install it with 'pip install {missing_package}'")
        else:
            print(f"Missing required packages. Please install with: pip install flask nbformat nbclient")
        sys.exit(1)


if __name__ == "__main__":
    main()
