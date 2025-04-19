#!/usr/bin/env python3

import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import code
import threading

# Import the apply_patch functionality
from apply_patch import process_patch, open_file, write_file, remove_file

# Maximum time to wait for command execution (in seconds)
EXECUTION_TIMEOUT = 60.0
LINE_LIMIT = 2000  # Maximum number of output lines to return

# Global interactive interpreter for stateful execution
interpreter = None
interpreter_lock = threading.Lock()


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


@contextmanager
def time_limit(seconds):
    """Context manager to limit execution time."""

    def signal_handler(signum, frame):
        raise TimeoutException("Execution timed out")

    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


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


class CaptureOutput:
    """Capture stdout and stderr."""

    def __init__(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

    def get_output(self):
        return self.stdout.getvalue() + self.stderr.getvalue()


def get_or_create_interpreter():
    """
    Get or create a Python interpreter for stateful execution.

    Returns:
        code.InteractiveInterpreter: An interactive interpreter
    """
    global interpreter

    if interpreter is None:
        # Create a new interpreter with a fresh namespace
        interpreter = code.InteractiveInterpreter()

        # Initialize with some common imports
        interpreter.runsource("""
import sys
import os
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
        """)

    return interpreter


def execute_python_interpreter(code_str: str) -> str:
    """
    Execute Python code in an interactive interpreter.

    Args:
        code_str: Python code to execute

    Returns:
        Execution output
    """
    global interpreter_lock

    with interpreter_lock:  # Ensure thread safety
        interpreter = get_or_create_interpreter()

        with CaptureOutput() as output:
            try:
                with time_limit(EXECUTION_TIMEOUT):
                    # Execute the code
                    more = interpreter.runsource(code_str)

                    # If more is True, the input is incomplete
                    if more:
                        return "Incomplete input. Please provide complete Python statements."
            except TimeoutException:
                return f"Execution timed out after {EXECUTION_TIMEOUT} seconds."
            except Exception as e:
                print(f"Error: {str(e)}")
                traceback.print_exc()

        return output.get_output()


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
    return execute_python_interpreter(input_text)


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


def main():
    """
    Main entry point for the server.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Execute Python code or terminal commands in a stateful environment')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--host', type=str, default='localhost', help='Host to bind to')
    args = parser.parse_args()

    # Initialize the interpreter on startup
    get_or_create_interpreter()
    print(f"Python interpreter initialized")

    # For simplicity, we'll use Flask for the web server
    try:
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        @app.route('/execute', methods=['POST'])
        def execute():
            request_data = request.json
            result = handle_agent_request(request_data)
            return jsonify(result)

        print(f"Server running on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port)

    except ImportError as e:
        print(f"Flask is not installed. Please install it with 'pip install flask'")
        sys.exit(1)


if __name__ == "__main__":
    main()
