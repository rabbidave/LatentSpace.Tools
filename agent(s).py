#!/usr/bin/env python3
import sys
import os
import venv
import subprocess
import json
import logging
import re
import ast
from io import StringIO
from pathlib import Path
import time
import threading
import queue
import gradio as gr
import pandas as pd
import duckdb

# --- Configuration ---
VENV_DIR = Path(".agent_venv_auto_v2") # Using unique name for this version
DEPENDENCIES = ["psutil", "openai", "gradio", "duckdb", "pandas"]

# --- Primary LLM Configuration ---
# IMPORTANT: Configure these for your PRIMARY agent
API_BASE = "https://openrouter.ai/api/v1"  # OpenRouter API endpoint
API_KEY = "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" # OpenRouter API key
MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free" # Replace with your PRIMARY model

# --- Confirmation LLM Configuration ---
# IMPORTANT: Configure these for your secondary CONFIRMATION agent
CONFIRMATION_API_BASE = "https://openrouter.ai/api/v1" # OpenRouter API endpoint
CONFIRMATION_API_KEY = "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" # OpenRouter API key
CONFIRMATION_MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free" # Replace with your CONFIRMATION model (can be the same, but ideally tuned for safety checks)

# --- Tool Configuration ---
WORKSPACE_DIR = Path("tool_workspace").resolve()
MAX_OUTPUT_LINES = 100 # Limit captured stdout/stderr lines from execute_code


# --- Database Configuration ---
DB_FILE = Path("agent_actions.db").resolve()
# --- Safety Regex ---
# Allows: kill_process(pid=NUMBER) OR execute_code(code="STRING_LITERAL" | 'STRING_LITERAL')
# This regex attempts to correctly match string literals enclosed in single or double quotes,
# handling basic escaped quotes within the string. More complex nesting might still fail.
# Allows: kill_process(pid=NUMBER) OR execute_code(code="...") OR execute_code(code='...')
# Handles basic escaped quotes inside the string literal.
SAFETY_REGEX = r'(^kill_process\(pid=\d+\)$)|(^execute_code\(code=(["\'])((?:\\.|(?!\3).)*)\3\)$)'

# --- Tool Call Parsing Regex ---
# Matches the overall structure: tool_name(arguments) anchored start/end
TOOL_CALL_REGEX = re.compile(r'^\s*(\w+)\((.*)\)\s*$')
# Extracts pid=NUMBER from arguments, allowing whitespace
PID_ARG_REGEX = re.compile(r'^\s*pid=(\d+)\s*$')
# Extracts code="STRING" or code='STRING', capturing the string content (group 2)
# Handles potential whitespace around the equals sign and the string literal.
CODE_ARG_REGEX = re.compile(r'^\s*code\s*=\s*(["\'])(.*?)\1\s*$', re.DOTALL)

# --- System Prompt for Primary Agent ---
# Instructs the primary agent on its role, constraints, tools, and STRICT output format.
SYSTEM_PROMPT = """
You are an AI assistant. Your ONLY task is to analyze system process data and propose actions using specific tools: `kill_process` and `execute_code`.
You MUST follow the output format EXACTLY. DO NOT add any explanations, commentary, greetings, or any text other than the specified tool calls or "NO_ACTIONS_NEEDED".

**Output Format Specification (MANDATORY):**
*   Your response MUST contain ONLY tool calls or the exact text "NO_ACTIONS_NEEDED".
*   Each tool call MUST be on its own line. NO other text is allowed.
*   `kill_process` format: `kill_process(pid=NUMBER)` (e.g., `kill_process(pid=1234)`)
*   `execute_code` format: `execute_code(code="YOUR_PYTHON_CODE_STRING")` or `execute_code(code='YOUR_PYTHON_CODE_STRING')`.
*   If NO actions are needed, output ONLY the exact text: `NO_ACTIONS_NEEDED`

**Example of BAD Output (Do NOT do this):**
```text
Okay, I see [application] running with PID 5678. I will kill it.
kill_process(pid=5678)
```

**Example of GOOD Output:**
```text
kill_process(pid=5678)
```

**Few-Shot Examples:**

*Example 1: [application] Found*
Input Process List Snippet:
```json
[
  {{ "pid": 123, "name": "Finder", "username": "user", "cpu_percent": 0.1, "memory_mb": 50.5 }},
  {{ "pid": 456, "name": "[application]", "username": "user", "cpu_percent": 5.2, "memory_mb": 250.8 }},
  {{ "pid": 789, "name": "kernel_task", "username": "root", "cpu_percent": 1.0, "memory_mb": 1024.0 }}
]
```
Expected Output:
```text
kill_process(pid=456)
```

*Example 2: No [application] Found, No Action Needed*
Input Process List Snippet:
```json
[
  {{ "pid": 123, "name": "Finder", "username": "user", "cpu_percent": 0.1, "memory_mb": 50.5 }},
  {{ "pid": 789, "name": "kernel_task", "username": "root", "cpu_percent": 1.0, "memory_mb": 1024.0 }}
]
```
Expected Output:
```text
NO_ACTIONS_NEEDED
```

*Example 3: Execute Code Requested (Hypothetical)*
Input Process List Snippet:
```json
[
  {{ "pid": 111, "name": "data_processor", "username": "user", "cpu_percent": 0.0, "memory_mb": 15.0 }},
  {{ "pid": 222, "name": "monitor_script", "username": "user", "cpu_percent": 0.0, "memory_mb": 5.0 }}
]
```
(Assume internal logic decided to run code based on this)
Expected Output:
```text
execute_code(code="print('Checking status...')")
```

**Instructions:**
1.  Analyze the COMPLETE process list data provided below.
2.  Identify the process named "WhatsApp". If found, generate a `kill_process(pid=...)` call for its PID.
3.  Identify the process named "komorebi-bar.exe". If NOT found, return Python which runs the following script as a background process: C:/Users/mcnug/start-elevated-komorebi.ps1
4.  Output ONLY the required tool calls (one per line) or ONLY "NO_ACTIONS_NEEDED" if "WhatsApp" is not found or no other actions are deemed necessary based on your analysis and the examples provided. Adhere strictly to the output format.

**Process List Data:**
{process_list_json}
"""

# --- Prompt for Confirmation Agent ---
# Instructs the safety agent on its role and expected output (APPROVE/DENY).
CONFIRMATION_PROMPT = """
You are a safety confirmation agent. Review the following proposed tool calls.
Your goal is to prevent clearly harmful or nonsensical actions.
Approve actions unless they seem obviously dangerous (e.g., killing critical system processes like 'System', 'csrss.exe', 'wininit.exe', executing malicious-looking code like deleting files outside the workspace).
Killing a standard user application like 'WhatsApp' or 'TikTok' based on a request is generally acceptable. Executing code to start expected helper scripts (like 'start-elevated-komorebi.ps1') is also acceptable.

Proposed Tool Calls:
{proposed_calls_str}

Do you approve these actions? Respond ONLY with "APPROVE" or "DENY". If denying, add a short reason after DENY (e.g., "DENY - Attempting to kill critical process 'csrss.exe'").
"""


# --- Logging Setup ---
# Configures basic logging to the console.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# --- Tool Implementation: get_processes ---
try:
    # Try importing psutil early, but handle failure gracefully until venv setup
    import psutil
except ImportError:
    psutil = None # Placeholder until venv setup ensures it's imported

def get_current_processes():
    """Retrieves current process list as JSON string. Uses log_queue."""
    processes = []
    # Use log_queue instead of logging
    # log_queue.put("Attempting to retrieve process list...") # Already logged in worker thread
    try:
        # Ensure psutil is available after venv setup
        global psutil
        if psutil is None and 'psutil' not in sys.modules:
             import psutil # Import within function if not globally available yet
        elif psutil is None:
             raise NameError("psutil could not be imported.") # Raise error if import failed previously

        # Iterate and gather process info
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_info']):
            try:
                mem_info = proc.info['memory_info']
                mem_mb = round(mem_info.rss / (1024 * 1024), 2) if mem_info else 0
                cpu_percent = round(proc.info['cpu_percent'], 2) if proc.info['cpu_percent'] is not None else 0.0
                process_info = {
                    "pid": proc.info['pid'],
                    "name": proc.info['name'] if proc.info['name'] else 'N/A',
                    "username": proc.info['username'] if proc.info['username'] is not None else 'N/A',
                    "cpu_percent": cpu_percent,
                    "memory_mb": mem_mb,
                }
                processes.append(process_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue # Ignore processes that disappeared or are inaccessible
            except Exception as e:
                # Log warning but continue iterating other processes
                log_queue.put(f"Warning: Could not retrieve info for PID {proc.pid if proc else 'N/A'}: {e}")
                continue
        # log_queue.put(f"Successfully retrieved info for {len(processes)} processes.") # Already logged in worker
        return json.dumps(processes, indent=2) # Return data as formatted JSON string
    except NameError:
         # Handle case where psutil is still not importable
         log_queue.put("ERROR: psutil not imported. Dependencies might be missing or venv setup failed.")
         return json.dumps({"error": "psutil not available."})
    except Exception as e:
        # Catch broader errors during process iteration
        log_queue.put(f"ERROR: Failed to retrieve process list: {e}")
        return json.dumps({"error": f"Failed to retrieve process list: {str(e)}"})

# --- Tool Implementation: kill_process ---
def terminate_process_by_pid(pid: int):
    """Terminates a process by its PID, returning JSON status. Uses log_queue."""
    # log_queue.put(f"Attempting to terminate process with PID: {pid}") # Logged in worker thread
    try:
        # Ensure psutil is available
        global psutil
        if psutil is None and 'psutil' not in sys.modules:
             import psutil
        elif psutil is None:
             raise NameError("psutil could not be imported.")

        # Validate PID is integer
        if not isinstance(pid, int): raise ValueError("PID must be an integer.")
        # Get process and name (useful for logs/messages)
        process = psutil.Process(pid)
        process_name = process.name()
        log_queue.put(f"Terminating process: PID={pid}, Name='{process_name}'") # Use queue
        # Terminate gracefully first
        process.terminate()
        try:
            # Wait briefly for termination, then report success
            process.wait(timeout=2)
            log_queue.put(f"Process PID={pid}, Name='{process_name}' terminated successfully.") # Use queue
            return json.dumps({"status": "success", "pid": pid, "message": f"Process PID {pid} ('{process_name}') terminated."})
        except psutil.TimeoutExpired:
            # Force kill if graceful termination failed
            log_queue.put(f"Process PID={pid}, Name='{process_name}' did not terminate gracefully. Forcing kill.") # Use queue
            process.kill()
            process.wait(timeout=1) # Wait briefly for kill signal to take effect
            log_queue.put(f"Process PID={pid}, Name='{process_name}' killed forcefully.") # Use queue
            return json.dumps({"status": "success", "pid": pid, "message": f"Process PID {pid} ('{process_name}') forcefully killed."})
    # --- Error Handling for kill_process ---
    except psutil.NoSuchProcess:
        log_queue.put(f"ERROR: Process with PID {pid} not found.") # Use queue
        return json.dumps({"status": "error", "pid": pid, "message": f"Process with PID {pid} not found."})
    except psutil.AccessDenied:
        log_queue.put(f"ERROR: Permission denied to terminate process PID {pid}.") # Use queue
        return json.dumps({"status": "error", "pid": pid, "message": f"Permission denied to terminate process PID {pid}. Try running with higher privileges."})
    except ValueError as ve:
        log_queue.put(f"ERROR: Invalid PID provided: {ve}") # Use queue
        return json.dumps({"status": "error", "pid": pid, "message": f"Invalid PID provided: {str(ve)}"})
    except NameError:
         log_queue.put("ERROR: psutil not imported. Dependencies might be missing or venv setup failed.") # Use queue
         return json.dumps({"status": "error", "pid": pid, "message": "psutil not available."})
    except Exception as e:
        log_queue.put(f"ERROR: An unexpected error occurred while trying to terminate PID {pid}: {e}") # Use queue
        return json.dumps({"status": "error", "pid": pid, "message": f"An unexpected error occurred: {str(e)}"})


# --- Tool Implementation: execute_code_arbitrary (Includes AST Validation) ---
class CodeValidationError(Exception):
    """Custom exception for code validation failures."""
    pass

# Define builtins allowed within the sandboxed code execution
SAFE_BUILTINS = {
    'print': print, 'str': str, 'int': int, 'float': float, 'bool': bool, 'bytes': bytes,
    'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
    'len': len, 'repr': repr, 'range': range, 'abs': abs, 'round': round,
    'min': min, 'max': max, 'sum': sum, 'sorted': sorted, 'enumerate': enumerate, 'zip': zip,
    'all': all, 'any': any, 'isinstance': isinstance, 'issubclass': issubclass, 'type': type,
    'True': True, 'False': False, 'None': None,
    # Allow common exceptions to be caught/raised within the snippet
    'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError, 'KeyError': KeyError, 'IndexError': IndexError,
    'AssertionError': AssertionError,
    # Allow 'open' but its usage is validated by validate_code_ast
    'open': open,
}

# Define allowed Abstract Syntax Tree (AST) node types for basic operations and control flow
ALLOWED_NODE_TYPES = {
    ast.Module, ast.Expr, ast.Constant, ast.Call, ast.Name, ast.Attribute,
    ast.Load, ast.Store, ast.Assign, ast.AugAssign, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
    ast.If, ast.For, ast.While, ast.Break, ast.Continue, # Basic control flow
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Index, ast.Slice, # Data structures
    ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, # Comprehensions
    ast.Pass, ast.Assert, # Simple statements
    ast.FormattedValue, ast.JoinedStr, # F-strings
    ast.withitem, ast.With, # Context managers (like 'with open(...)')
    ast.Return, # Allow returning values if code was structured in functions
    ast.Constant, # Allow True, False, None literals (NameConstant deprecated in 3.14)
    ast.keyword, # For keyword arguments in function calls
}

# Explicitly define disallowed AST node types for critical security risks
# Corrected: Removed ast.Exec as it's Python 2 specific. exec() in Py3 is a function call (ast.Call).
DISALLOWED_NODE_TYPES = {
    ast.Import, ast.ImportFrom, # Prevent importing arbitrary modules
    ast.Delete, # Prevent deleting variables/attributes
    ast.Global, ast.Nonlocal, # Prevent manipulation of outer scopes
    # ast.Try, ast.Raise could potentially be allowed if carefully considered
}

def validate_path(path_str, current_workspace_dir: Path):
    """Checks if a path is safe (within current_workspace_dir). Raises CodeValidationError if not."""
    try:
        # Resolve path relative to the *passed* workspace directory
        resolved_path = (current_workspace_dir / path_str).resolve()
        # Security check: Ensure the resolved path is actually inside the current_workspace_dir
        resolved_path.relative_to(current_workspace_dir)
        return resolved_path # Return the validated, absolute path
    except ValueError: # This exception is raised by relative_to if the path is outside
        raise CodeValidationError(f"Path traversal detected or path outside designated workspace '{current_workspace_dir}': '{path_str}'")
    except Exception as e: # Catch other filesystem or resolution errors
        raise CodeValidationError(f"Invalid or unresolvable path '{path_str}' relative to '{current_workspace_dir}': {e}")

def validate_code_ast(code_string, current_workspace_dir: Path):
    """Validates Python code using AST analysis. Raises CodeValidationError on violations."""
    try:
        tree = ast.parse(code_string) # Parse code into Abstract Syntax Tree
    except SyntaxError as e:
        # Handle code that isn't valid Python syntax
        raise CodeValidationError(f"Syntax error in code: {e}")

    allowed_call_names = set(SAFE_BUILTINS.keys()) # Functions allowed to be called directly

    # Walk through every node in the parsed AST
    for node in ast.walk(tree):
        node_type = type(node)

        # Check against explicitly disallowed node types
        if node_type in DISALLOWED_NODE_TYPES:
             raise CodeValidationError(f"Disallowed AST node type used: {node_type.__name__}")

        # Optional strictness: Check if the node type is within the generally allowed set
        # if node_type not in ALLOWED_NODE_TYPES:
        #     log_queue.put(f"Warning: AST node type {node_type.__name__} not explicitly allowed, review needed.")
        #     # Depending on security posture, could raise CodeValidationError here too

        # --- Specific Node Validations for Added Security ---

        # Validate 'open' calls: check path, mode, arguments
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'open':
            if not node.args: raise CodeValidationError("Invalid 'open' call: Requires at least a filename.")
            # Ensure filename argument is a simple string literal
            filename_arg = node.args[0]
            if not isinstance(filename_arg, ast.Constant) or not isinstance(filename_arg.value, str):
                raise CodeValidationError("Filename argument for 'open' must be a literal string.")
            # Validate the path is within the allowed workspace (pass the current workspace)
            validated_path = validate_path(filename_arg.value, current_workspace_dir) # Raises error if invalid
            # log_queue.put(f"Debug: Validated safe path for open call: {validated_path}") # Log success via queue if needed
            # Validate mode argument if provided
            if len(node.args) > 1:
                mode_arg = node.args[1]
                if not isinstance(mode_arg, ast.Constant) or not isinstance(mode_arg.value, str):
                     raise CodeValidationError("Mode argument for 'open' must be a literal string.")
                mode = mode_arg.value
                # Basic check on allowed characters in the mode string
                if not set(mode) <= set('rwxabt+'):
                     raise CodeValidationError(f"Invalid characters in file mode '{mode}'.")

        # Validate other function calls (are they in SAFE_BUILTINS?)
        elif isinstance(node, ast.Call):
             func = node.func
             called_name = None
             if isinstance(func, ast.Name): # Direct function call like print()
                 called_name = func.id
             # Could add further checks for method calls (ast.Attribute) if needed
             # e.g., ensure only safe methods like list.append, str.split are called.
             if called_name and called_name not in allowed_call_names:
                 raise CodeValidationError(f"Disallowed function call: {called_name}")

        # Prevent access to potentially dangerous 'dunder' attributes
        # Allows common safe ones (__init__, __repr__, __str__) and context managers (__enter__, __exit__)
        if isinstance(node, ast.Attribute):
            allowed_dunders = {'__init__', '__repr__', '__str__', '__enter__', '__exit__'}
            if node.attr.startswith('__') and node.attr.endswith('__') and node.attr not in allowed_dunders:
                 raise CodeValidationError(f"Access to potentially unsafe attribute '{node.attr}' disallowed.")

def run_arbitrary_code_safely(code: str, workspace_dir: Path, max_output_lines: int):
    """Executes AST-validated code in a restricted env, capturing output. Returns JSON status. Uses log_queue."""
    log_queue.put(f"Attempting to execute arbitrary code snippet (first 100 chars): {code[:100]}...")
    # Redirect stdout/stderr to capture output from the executed code
    old_stdout, old_stderr = sys.stdout, sys.stderr
    redirected_output, redirected_error = StringIO(), StringIO()
    sys.stdout, sys.stderr = redirected_output, redirected_error
    exit_code, error_message = 0, "" # Track execution status

    try:
        # 1. Validate code via AST *before* attempting execution, passing workspace dir
        validate_code_ast(code, workspace_dir)
        log_queue.put("AST validation passed.")

        # 2. Prepare restricted environment globals (only safe builtins allowed)
        safe_globals = {'__builtins__': SAFE_BUILTINS}
        # NOTE: Add other safe modules (e.g., math, json, datetime) here ONLY IF
        # ABSOLUTELY NECESSARY and their usage is understood and validated.

        # 3. Execute the validated code using exec in the restricted environment
        exec(code, safe_globals, {}) # Use empty locals dictionary for isolation
        log_queue.put("Code execution completed.")

    except CodeValidationError as e:
        # Handle validation errors (code structure/content is disallowed)
        log_queue.put(f"ERROR: Code validation failed: {e}")
        error_message, exit_code = f"Code validation failed: {str(e)}", 1
    except Exception as e:
        # Handle runtime errors *during* the execution of the sandboxed code
        log_queue.put(f"ERROR: Error DURING code snippet execution: {e}")
        # Format runtime error message concisely for reporting back
        import traceback
        tb_lines = traceback.format_exception_only(type(e), e)
        error_message, exit_code = f"Runtime error during execution: {''.join(tb_lines).strip()}", 1
    finally:
        # 4. Restore original stdout/stderr streams
        sys.stdout, sys.stderr = old_stdout, old_stderr

        # 5. Get captured output, limiting lines using the passed parameter
        stdout_val = redirected_output.getvalue().splitlines()
        stderr_val = redirected_error.getvalue().splitlines()
        limited_stdout = "\n".join(stdout_val[:max_output_lines]) # Use parameter
        limited_stderr = "\n".join(stderr_val[:max_output_lines]) # Use parameter
        # Add truncation notice if output exceeded the limit
        if len(stdout_val) > max_output_lines: limited_stdout += f"\n... (truncated - {len(stdout_val)} total lines)"
        if len(stderr_val) > max_output_lines: limited_stderr += f"\n... (truncated - {len(stderr_val)} total lines)"

        # 6. Format result into JSON structure for consistent return value
        result_data = {
            "status": "success" if exit_code == 0 else "error", # Overall status
            "exit_code": exit_code, # 0 for success, 1 for error
            "stdout": limited_stdout, # Captured standard output
            "stderr": limited_stderr, # Captured standard error
            "error_message": error_message # Summary of validation or runtime error
        }

# --- Database Logging Helper ---
def log_action_to_db(conn, tool_name: str, arguments: dict, status: str, result_summary: str):
    """Logs the details of an executed action to the DuckDB database."""
    if not conn:
        log_queue.put("Warning: Database connection is not available. Cannot log action.")
        return
    try:
        # Convert arguments dict to string for storage
        args_str = json.dumps(arguments)
        # Truncate result_summary if it's too long for the DB column (optional, adjust size if needed)
        max_summary_len = 500 # Example limit
        summary = (result_summary[:max_summary_len] + '...') if len(result_summary) > max_summary_len else result_summary
        
        conn.execute("""
            INSERT INTO action_log (tool_name, arguments, status, result_summary)
            VALUES (?, ?, ?, ?);
        """, (tool_name, args_str, status, summary))
        # log_queue.put(f"Action logged to DB: {tool_name}, Status: {status}") # Optional: Log success via queue
    except Exception as e:
        log_queue.put(f"ERROR: Failed to log action to database: {e}")

        log_queue.put(f"Execution Result: Status={result_data['status']}, ExitCode={result_data['exit_code']}")
        return json.dumps(result_data, indent=2) # Return result as JSON string

# --- Tool Dispatcher ---
# Maps tool names (as used in LLM prompts/parsing) to their implementation functions
TOOLS = {
    "kill_process": terminate_process_by_pid,
    "execute_code": run_arbitrary_code_safely,
}

# --- Agent Interaction Logic ---
try:
    # Try importing OpenAI library, handle failure until venv setup
    from openai import OpenAI
except ImportError:
    OpenAI = None # Placeholder

def get_llm_response(client: OpenAI, conversation_history: list, model_name: str):
    """Generic function to query an LLM endpoint, returning the response content or error. Uses log_queue."""
    if client is None: # Check if client initialization failed earlier
        log_queue.put("ERROR: LLM client is None in get_llm_response.")
        return "ERROR_LLM_COMMUNICATION: OpenAI client not initialized."
    # log_queue.put(f"Querying model {model_name} at {client.base_url}...") # Logged in worker thread
    try:
        # Make the API call to the chat completions endpoint
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=conversation_history,
            temperature=0.1, # Use lower temperature for more deterministic/predictable responses
            max_tokens=500, # Adjust based on expected length of tool calls or confirmation
            stream=False, # Don't stream responses for this use case
        )
        # Extract the response content
        response_content = chat_completion.choices[0].message.content.strip()
        # log_queue.put(f"LLM raw response ({model_name}): {response_content[:200]}...") # Logged in worker thread
        return response_content
    except Exception as e:
        # Handle errors during the API call
        log_queue.put(f"ERROR: Error querying LLM {model_name} at {client.base_url}: {e}")
        return f"ERROR_LLM_COMMUNICATION: {e}" # Return error indicator

def parse_tool_calls(response_content: str):
    """Parses tool calls from LLM response based on strict line-by-line format."""
    # Handle the specific "no actions needed" response
    if response_content == "NO_ACTIONS_NEEDED":
        logging.info("LLM indicated no actions needed.") # Keep logging for this info? Or queue? Let's keep logging for now.
        return [] # Return empty list, signifying no tools to call

    parsed_calls = []
    potential_calls = response_content.strip().split('\n') # Split response into lines
    logging.info(f"Potential tool call lines: {potential_calls}") # Keep logging

    # Process each line to see if it matches the expected tool call format
    for line in potential_calls:
        line = line.strip()
        if not line: continue # Skip empty lines

        # Match the overall tool_name(arguments) structure
        match = TOOL_CALL_REGEX.match(line)
        if not match:
            # Warn if a line doesn't conform to the expected basic structure
            logging.warning(f"Line does not match expected tool call format: '{line}'") # Keep logging
            continue

        tool_name = match.group(1) # Extract tool name
        args_str = match.group(2).strip() # Extract arguments string (inside parentheses)
        args_dict = {} # Dictionary to hold parsed arguments

        try:
            # Parse arguments specifically based on the detected tool name
            if tool_name == "kill_process":
                pid_match = PID_ARG_REGEX.match(args_str) # Expect 'pid=NUMBER'
                if pid_match:
                    args_dict['pid'] = int(pid_match.group(1)) # Store integer PID
                else:
                    # Argument format mismatch for kill_process
                    logging.warning(f"Could not parse pid=NUMBER from kill_process args: '{args_str}' in line '{line}'") # Keep logging
                    continue # Skip this invalidly formatted call
            elif tool_name == "execute_code":
                code_match = CODE_ARG_REGEX.match(args_str) # Expect 'code="STRING"' or 'code=\'STRING\''
                if code_match:
                    args_dict['code'] = code_match.group(2) # Store the code string content
                else:
                    # Argument format mismatch for execute_code
                    logging.warning(f"Could not parse code=\"...\" from execute_code args: '{args_str}' in line '{line}'") # Keep logging
                    continue # Skip this invalidly formatted call
            else:
                # Handle cases where the LLM hallucinates a tool name not defined
                logging.warning(f"Unsupported tool name encountered during parsing: '{tool_name}' in line '{line}'") # Keep logging
                continue

            # If parsing succeeded, store the structured call information
            parsed_calls.append({
                "raw_call": line, # Keep original string for validation/confirmation prompt
                "tool_name": tool_name,
                "arguments": args_dict
            })
            logging.info(f"Successfully parsed tool call: {line}") # Keep logging

        except Exception as e:
            # Catch errors during argument parsing/conversion (e.g., int conversion)
            logging.warning(f"Error processing arguments for tool call '{line}': {e}") # Keep logging

    # Return the list of successfully parsed tool calls
    return parsed_calls

# --- Gradio UI and Application Logic ---

# Queue for thread-safe communication between agent thread and Gradio UI
log_queue = queue.Queue()
# Global reference to the agent thread
agent_thread = None

# Function to run the agent loop in a separate thread
def agent_worker_thread(config_state, agent_running_state, stop_requested_state, user_decision_state, proposed_actions_state):
    """The main worker loop for the agent, running in a separate thread."""
    import duckdb
    db_conn = None # Initialize db connection variable
    log_queue.put("Agent thread started.")

    primary_client = None
    confirmation_client = None
    primary_client_initialized = False
    confirmation_client_initialized = False

    # --- Initialize LLM Clients ---
    # This needs to be done *inside* the thread after venv is confirmed
    try:
        global OpenAI
        if OpenAI is None and 'openai' not in sys.modules: from openai import OpenAI
        elif OpenAI is None: raise NameError("OpenAI library could not be imported.")

        # Read config from state
        current_config = config_state.value # Access the dictionary within the state

        primary_client = OpenAI(base_url=current_config.get("api_base"), api_key=current_config.get("api_key"))
        log_queue.put(f"Primary OpenAI client initialized for: {current_config.get('api_base')}")
        primary_client_initialized = True

        confirmation_client = OpenAI(base_url=current_config.get("confirmation_api_base"), api_key=current_config.get("confirmation_api_key"))
        log_queue.put(f"Confirmation OpenAI client initialized for: {current_config.get('confirmation_api_base')}")
        confirmation_client_initialized = True

    except NameError:
         log_queue.put("ERROR: OpenAI library not found. Check venv setup.")
    except Exception as e:
        log_queue.put(f"ERROR: Failed to initialize OpenAI client(s): {e}")
        # If clients fail, the loop below will handle it

    # --- Initialize Database ---
    try:
        db_path = str(DB_FILE) # Use the defined path
        log_queue.put(f"Connecting to database: {db_path}")
        db_conn = duckdb.connect(database=db_path, read_only=False)
        # Create table if it doesn't exist
        db_conn.execute("""
            CREATE TABLE IF NOT EXISTS action_log (
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tool_name VARCHAR,
                arguments VARCHAR, 
                status VARCHAR, 
                result_summary VARCHAR
            );
        """)
        log_queue.put("Database connection established and table ensured.")
    except Exception as e:
        log_queue.put(f"ERROR: Failed to connect to or initialize database {DB_FILE}: {e}")
        db_conn = None # Ensure connection is None if setup failed
        # Optionally, decide if the agent should stop if DB fails
        # agent_running_state.value = False 
        # log_queue.put("Agent stopping due to database initialization failure.")

    # --- Main Agent Loop ---
    while agent_running_state.value: # Check the running state flag
        log_queue.put("\n--- Starting New Agent Cycle ---")

        # Check if clients are initialized
        if not primary_client_initialized or not confirmation_client_initialized:
            log_queue.put("ERROR: LLM clients not initialized. Skipping cycle.")
            time.sleep(5)
            continue

        # Read current config for this cycle (in case it changed)
        current_config = config_state.value
        model_name = current_config.get("model_name")
        confirmation_model_name = current_config.get("confirmation_model_name")
        workspace_dir = Path(current_config.get("workspace_dir", "tool_workspace")).resolve() # Use default if not set
        max_output_lines = int(current_config.get("max_output_lines", 100)) # Use default if not set
        system_prompt_template = current_config.get("system_prompt", SYSTEM_PROMPT) # Get system prompt from config
        confirmation_prompt_template = current_config.get("confirmation_prompt", CONFIRMATION_PROMPT) # Get confirmation prompt from config

        # Ensure workspace exists (using the potentially updated path)
        try:
            workspace_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log_queue.put(f"Warning: Failed to ensure workspace directory {workspace_dir}: {e}")


        # 1. Get Current Process List Data
        log_queue.put("Fetching current process list...")
        process_list_json = get_current_processes() # Assumes psutil is imported
        if '"error":' in process_list_json:
            log_queue.put(f"ERROR: Failed to get process list. Skipping cycle. Error: {process_list_json}")
            time.sleep(10) # Longer sleep on process error
            continue
        log_queue.put("Process list fetched.")

        # 2. Format Prompt for Primary Agent
        current_prompt = system_prompt_template.format(process_list_json=process_list_json) # Use template from config
        conversation = [
            # The system role message might be redundant if the template includes it, but keep for now
            {"role": "system", "content": "You are an AI assistant following instructions precisely and adhering strictly to output format."},
            {"role": "user", "content": current_prompt}
        ]

        # 3. Get Primary Agent's Decision
        log_queue.put("Querying primary agent...")
        agent_response = get_llm_response(primary_client, conversation, model_name)
        if agent_response.startswith("ERROR_LLM_COMMUNICATION"):
            log_queue.put(f"ERROR: Primary LLM communication error: {agent_response}. Skipping cycle.")
            time.sleep(5)
            continue
        log_queue.put("Primary agent responded.")

        # 4. Parse Tool Calls
        log_queue.put("Parsing primary agent response...")
        parsed_calls = parse_tool_calls(agent_response)

        if not parsed_calls:
            log_queue.put("Primary agent proposed no tool calls or 'NO_ACTIONS_NEEDED'.")
            if agent_response != "NO_ACTIONS_NEEDED":
                 log_queue.put(f"(Agent raw response: '{agent_response[:100]}...')")
            time.sleep(5) # Sleep before next cycle
            continue

        # 5. Validate Proposed Calls (Regex)
        validated_calls = []
        proposed_calls_str_list = []
        log_queue.put("\n--- Proposed Actions (Pending Validation) ---")
        for call_info in parsed_calls:
            raw_call = call_info["raw_call"]
            proposed_calls_str_list.append(raw_call)
            log_queue.put(f"Proposed: {raw_call}")
            if re.match(SAFETY_REGEX, raw_call):
                validated_calls.append(call_info)
            else:
                log_queue.put(f"Action REJECTED (Failed System Regex): {raw_call}")

        if not validated_calls:
            log_queue.put("\nNo valid actions proposed after regex check.")
            time.sleep(5)
            continue

        # Store proposed actions for potential UI display
        proposed_actions_state.value = validated_calls
        proposed_calls_display = "\n".join([call['raw_call'] for call in validated_calls])


        # --- Human-in-the-Loop Check ---
        if stop_requested_state.value:
            log_queue.put("\n--- Stop Requested: Waiting for User Input ---")
            log_queue.put("Proposed Actions:\n" + proposed_calls_display)
            user_decision_state.value = None # Clear previous decision
            # The UI should now show the Approve/Deny buttons

            while user_decision_state.value is None and agent_running_state.value:
                # Wait for the user to click Approve or Deny in the UI
                time.sleep(0.5)

            if not agent_running_state.value: # Check if user stopped agent while waiting
                 log_queue.put("Agent stopped by user while waiting for input.")
                 break # Exit the main while loop

            decision = user_decision_state.value
            log_queue.put(f"User decision: {decision}")
            stop_requested_state.value = False # Reset the flag
            user_decision_state.value = None # Reset decision state

            if decision == "deny":
                log_queue.put("Execution skipped due to user denial.")
                time.sleep(5)
                continue # Skip to the next cycle

            # If approved, fall through to execution
            log_queue.put("User approved actions. Proceeding with execution.")
            # Optional: Could skip confirmation agent if user approved, or run it anyway
            # For now, let's assume user approval bypasses confirmation agent

        else:
             # Optional: Run confirmation agent if stop wasn't requested
             log_queue.put("\n--- Seeking Confirmation from Safety Agent ---")
             # Pass the confirmation prompt template from config
             is_approved_by_safety = get_confirmation_decision(
                 confirmation_client,
                 validated_calls,
                 confirmation_model_name,
                 confirmation_prompt_template # Pass the template
             )
             if not is_approved_by_safety:
                 log_queue.put("\nExecution halted: Actions not approved by confirmation agent.")
                 time.sleep(5)
                 continue


        # 7. Execute Approved Calls
        log_queue.put("\n--- Executing Actions ---")
        # Use validated_calls as they passed regex and either user approval or safety agent
        confirmed_calls = validated_calls
        for call_info in confirmed_calls:
            tool_name = call_info["tool_name"]
            args = call_info["arguments"]
            raw_call = call_info["raw_call"]
            log_queue.put(f"\nExecuting: {raw_call}")

            if tool_name in TOOLS:
                tool_function = TOOLS[tool_name]
                action_status = "error" # Default status
                result_summary = "" # Default summary
                try:
                    # Pass workspace and max_lines to execute_code
                    if tool_name == "execute_code":
                        # Add workspace_dir and max_output_lines from current config to the arguments
                        exec_args = {
                            **args,
                            "workspace_dir": workspace_dir,
                            "max_output_lines": max_output_lines
                        }
                        result_json = tool_function(**exec_args)
                    else:
                         result_json = tool_function(**args)

                    # Parse result to get status and summary for DB logging
                    try:
                        result_data = json.loads(result_json)
                        action_status = result_data.get("status", "unknown")
                        # Create a concise summary from the result JSON
                        if action_status == "success":
                            result_summary = result_data.get("message", "Success")
                        else: # Error status
                            result_summary = result_data.get("message", result_data.get("error_message", "Error occurred"))
                            if tool_name == "execute_code": # Add stderr for code errors
                                stderr = result_data.get("stderr")
                                if stderr: result_summary += f" | Stderr: {stderr[:100]}..." # Limit length

                    except json.JSONDecodeError:
                        action_status = "error"
                        result_summary = f"Tool returned non-JSON result: {result_json[:200]}..." # Log raw result snippet
                    except Exception as parse_e:
                         action_status = "error"
                         result_summary = f"Error parsing tool result JSON: {parse_e}"

                    log_queue.put(f"Result:\n{result_json}") # Still log full result to console queue

                except Exception as e:
                    # Capture execution error for DB logging
                    action_status = "error"
                    result_summary = f"Exception during tool execution: {str(e)}"
                    log_queue.put(f"ERROR executing tool '{tool_name}' with args {args}: {e}")

                # Log the action to the database using the helper function
                log_action_to_db(db_conn, tool_name, args, action_status, result_summary)
            else:
                log_queue.put(f"ERROR: Unknown tool '{tool_name}'")

        log_queue.put("\n--- Cycle Complete ---")
        time.sleep(5) # Wait before starting the next cycle

    log_queue.put("Agent thread finished.")

    # --- Cleanup --- 
    if db_conn:
        try:
            db_conn.close()
            log_queue.put("Database connection closed.")
        except Exception as e:
            log_queue.put(f"Warning: Error closing database connection: {e}")


# --- Database Log Fetching for UI ---
def get_db_logs_as_dataframe():
    """Connects to DuckDB, fetches action logs, returns as Pandas DataFrame."""
    logs_df = pd.DataFrame() # Default empty DataFrame
    db_conn = None
    try:
        db_path = str(DB_FILE)
        if not DB_FILE.exists():
            # Log to console queue as UI might not be fully loaded yet
            log_queue.put(f"Database file {db_path} not found. Cannot fetch logs.")
            return logs_df # Return empty DataFrame

        db_conn = duckdb.connect(database=db_path, read_only=True) # Read-only connection
        logs_df = db_conn.execute("SELECT timestamp, tool_name, arguments, status, result_summary FROM action_log ORDER BY timestamp DESC").fetchdf()
        # Optional: Format timestamp for better readability in UI
        if not logs_df.empty and 'timestamp' in logs_df.columns:
             logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    except Exception as e:
        log_queue.put(f"ERROR: Failed to fetch logs from database {DB_FILE}: {e}")
        # Optionally return a DataFrame with an error message
        logs_df = pd.DataFrame({"Error": [f"Failed to fetch logs: {e}"]})
    finally:
        if db_conn:
            try:
                db_conn.close()
            except Exception as e:
                 log_queue.put(f"Warning: Error closing read-only DB connection: {e}")
    return logs_df


# --- Gradio UI Definition ---
def create_gradio_ui():
    # Default configuration values
    default_config = {
        "api_base": API_BASE,
        "api_key": API_KEY,
        "model_name": MODEL_NAME,
        "confirmation_api_base": CONFIRMATION_API_BASE,
        "confirmation_api_key": CONFIRMATION_API_KEY,
        "confirmation_model_name": CONFIRMATION_MODEL_NAME,
        "workspace_dir": str(WORKSPACE_DIR),
        "max_output_lines": MAX_OUTPUT_LINES,
        "system_prompt": SYSTEM_PROMPT, # Add default system prompt
        "confirmation_prompt": CONFIRMATION_PROMPT, # Add default confirmation prompt
    }

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # --- State Variables ---
        # Holds the configuration dictionary
        config_state = gr.State(default_config)
        # Boolean flag indicating if the agent thread should be running
        agent_running_state = gr.State(False)
        # Boolean flag indicating user requested a stop for review
        stop_requested_state = gr.State(False)
        # Stores user decision ("approve" or "deny")
        user_decision_state = gr.State(None)
        # Stores the list of proposed actions when waiting for user
        proposed_actions_state = gr.State([])

        # --- UI Layout ---
        gr.Markdown("# Agent Control Panel")
        gr.Markdown("""
```ascii
+-------------------------+      +---------------------+      +--------------------+      +---------------------+
| 1. Get Process List     | ---> | 2. Primary Agent    | ---> | 3. Parse Tool Calls| ---> | 4. Regex Validation |
|  (get_current_processes)|      |   (Propose Actions) |      |  (parse_tool_calls)|      |   (SAFETY_REGEX)    |
+-------------------------+      +---------------------+      +--------------------+      +----------+----------+
                                                                                                      |
                                                                                                      | (Valid Calls)
                                                                                                      V
                                +---------------------------------------------------------------------+-------+
                                | 5. Confirmation Stage                                               |       |
                                |                                                                     |       |
                                |   +-----------------------------+     +-------------------------+   |       |
                                |   | [IF Stop Requested]         | OR  | [ELSE]                  |   |       |
                                |   | Human Review (UI Buttons)   | --> | Confirmation Agent      |   |       |
                                |   |   - Approve? -> Execute     |     | (get_confirmation_decision) |       |       
                                |   |   - Deny? ----> Skip        |     |   - Approve? -> Execute |   |       |
                                |   +-----------------------------+     |   - Deny? ----> Skip    |   |       |
                                |                                       +-------------------------+   |       |
                                +---------------------------------------------------------------------+-------+
                                                                                                      |
                                                                                                      |
                                                                                                      V
+-------------------------+      +-------------------------+      +-------------------------+      +--------------------+
| 8. Log Action to DB     | <--- | 7. Parse Result         | <--- | 6. Execute Actions      | <--- |  (Approved Calls)  |
|   (log_action_to_db)    |      |   (Status/Summary)      |      |   (TOOLS[tool_name])    |      |                    |
+-------------------------+      +-------------------------+      +-------------------------+      +--------------------+
                                                                                                           |
                                                                                                           V
                                                                                                   (Start Next Cycle)
```
""")

        with gr.Tabs():
            with gr.TabItem("Agent Control"):
                with gr.Row():
                    start_button = gr.Button("Start Agent")
                    stop_button = gr.Button("Stop Agent")
                    request_stop_button = gr.Button("Request Stop for Review")
                agent_status = gr.Textbox(label="Agent Status", value="Stopped", interactive=False)
                # output_log removed as it's static without 'every=1'

                with gr.Column(visible=False) as hitl_controls: # Hidden by default
                    gr.Markdown("## Human Review Requested")
                    proposed_actions_display = gr.Textbox(label="Proposed Actions", lines=5, interactive=False)
                    with gr.Row():
                        approve_button = gr.Button("Approve & Continue")
                        deny_button = gr.Button("Deny & Continue")

            with gr.TabItem("Configuration"):
                gr.Markdown("Configure LLM endpoints and other settings.")
                api_base_input = gr.Textbox(label="Primary API Base URL", value=default_config["api_base"])
                api_key_input = gr.Textbox(label="Primary API Key", type="password", value=default_config["api_key"])
                model_name_input = gr.Textbox(label="Primary Model Name", value=default_config["model_name"])
                conf_api_base_input = gr.Textbox(label="Confirmation API Base URL", value=default_config["confirmation_api_base"])
                conf_api_key_input = gr.Textbox(label="Confirmation API Key", type="password", value=default_config["confirmation_api_key"])
                conf_model_name_input = gr.Textbox(label="Confirmation Model Name", value=default_config["confirmation_model_name"])
                workspace_dir_input = gr.Textbox(label="Workspace Directory", value=default_config["workspace_dir"])
                max_output_lines_input = gr.Number(label="Max Output Lines (Execute Code)", value=default_config["max_output_lines"], precision=0)
                primary_prompt_input = gr.TextArea(label="Primary Agent System Prompt", value=default_config["system_prompt"], lines=10)
                confirmation_prompt_input = gr.TextArea(label="Confirmation Agent Prompt", value=default_config["confirmation_prompt"], lines=5)
                save_config_button = gr.Button("Save Configuration")
                config_save_status = gr.Textbox(label="Status", value="", interactive=False)

            with gr.TabItem("Action Log"):
                gr.Markdown("View recorded agent actions.")
                refresh_logs_button = gr.Button("Refresh Logs")
                action_log_display = gr.DataFrame(label="Action History", interactive=False)


        # --- UI Update Logic ---
        def update_log():
            """Periodically checks the queue and updates the log display."""
            logs = []
            while not log_queue.empty():
                logs.append(log_queue.get())

            # Determine current status based on state and logs
            current_status = "Unknown"
            if not agent_running_state.value:
                current_status = "Stopped"
            elif stop_requested_state.value and user_decision_state.value is None:
                 current_status = "Waiting for User Input"
            elif any("Executing Actions" in log for log in logs):
                 current_status = "Executing Actions"
            elif any("Fetching current process list" in log for log in logs):
                 current_status = "Fetching Processes"
            elif any("Querying primary agent" in log for log in logs):
                 current_status = "Querying Primary Agent"
            elif any("Seeking Confirmation" in log for log in logs):
                 current_status = "Seeking Confirmation"
            elif agent_running_state.value:
                 current_status = "Running" # Default if running but no specific phase detected

            # Update HITL controls visibility
            show_hitl = stop_requested_state.value and user_decision_state.value is None

            # Format proposed actions for display
            proposed_text = ""
            if show_hitl:
                actions = proposed_actions_state.value
                if isinstance(actions, list):
                     proposed_text = "\n".join([call.get('raw_call', str(call)) for call in actions])


            # Join new logs with existing log content (optional: limit total log size)
            new_log_content = "\n".join(logs) if logs else ""

            # Fetch latest logs from DB for the Action Log tab
            latest_logs_df = get_db_logs_as_dataframe()

            # Return updates for multiple components
            # output_log update removed
            return {
                agent_status: current_status,
                hitl_controls: gr.update(visible=show_hitl),
                proposed_actions_display: proposed_text,
                action_log_display: latest_logs_df # Add DataFrame update
            }

        # Schedule the log update function to run periodically
        # Now updates status, HITL controls, proposed actions, AND the action log DataFrame
        demo.load(
            update_log,
            None,
            [agent_status, hitl_controls, proposed_actions_display, action_log_display], # Add action_log_display here
            # Removed 'every=2' as it causes TypeError in this Gradio version
        )


        # --- Callback Functions ---
        def save_config(api_base, api_key, model, conf_base, conf_key, conf_model, ws_dir, max_lines, system_prompt, confirmation_prompt):
            new_config = {
                "api_base": api_base, "api_key": api_key, "model_name": model,
                "confirmation_api_base": conf_base, "confirmation_api_key": conf_key, "confirmation_model_name": conf_model,
                "workspace_dir": ws_dir, "max_output_lines": max_lines,
                "system_prompt": system_prompt, # Save system prompt
                "confirmation_prompt": confirmation_prompt, # Save confirmation prompt
            }
            config_state.value = new_config # Update the state object's value
            log_queue.put("Configuration saved.")
            return "Configuration saved successfully."

        def start_agent_callback():
            global agent_thread
            if agent_thread is None or not agent_thread.is_alive():
                agent_running_state.value = True
                stop_requested_state.value = False
                user_decision_state.value = None
                # Pass state objects to the thread function
                agent_thread = threading.Thread(target=agent_worker_thread, args=(
                    config_state, agent_running_state, stop_requested_state,
                    user_decision_state, proposed_actions_state
                ), daemon=True)
                agent_thread.start()
                log_queue.put("Agent start requested.")
                return "Agent starting..."
            else:
                return "Agent is already running."

        def stop_agent_callback():
            if agent_thread and agent_thread.is_alive():
                agent_running_state.value = False # Signal thread to stop
                # The thread will check this flag and exit its loop
                log_queue.put("Agent stop requested. Finishing current cycle...")
                return "Agent stopping..."
            else:
                 return "Agent is not running."

        def request_stop_callback():
            if agent_running_state.value:
                stop_requested_state.value = True
                log_queue.put("Stop for review requested. Will pause after current cycle analysis.")
                return "Stop requested. Agent will pause for review."
            else:
                return "Agent is not running."

        def approve_actions_callback():
            if stop_requested_state.value:
                user_decision_state.value = "approve"
                # Return updates for status and hide the HITL controls
                return {
                    agent_status: "Approval registered. Resuming...",
                    hitl_controls: gr.update(visible=False)
                }
            return {agent_status: "", hitl_controls: gr.update()} # No change if not waiting

        def deny_actions_callback():
             if stop_requested_state.value:
                user_decision_state.value = "deny"
                # Return updates for status and hide the HITL controls
                return {
                    agent_status: "Denial registered. Skipping actions...",
                    hitl_controls: gr.update(visible=False)
                }
             return {agent_status: "", hitl_controls: gr.update()} # No change if not waiting


        # --- Connect Callbacks to UI Components ---
        save_config_button.click(
            save_config,
            inputs=[
                api_base_input, api_key_input, model_name_input,
                conf_api_base_input, conf_api_key_input, conf_model_name_input,
                workspace_dir_input, max_output_lines_input,
                primary_prompt_input, confirmation_prompt_input # Add new inputs
            ],
            outputs=[config_save_status]
        )
        start_button.click(start_agent_callback, outputs=[agent_status])
        stop_button.click(stop_agent_callback, outputs=[agent_status])
        request_stop_button.click(request_stop_callback, outputs=[agent_status])
        approve_button.click(approve_actions_callback, outputs=[agent_status, hitl_controls])
        deny_button.click(deny_actions_callback, outputs=[agent_status, hitl_controls])

        # Modify Refresh Logs button to trigger full UI update, including status and HITL controls
        refresh_logs_button.click(
            update_log, # Call the main UI update function
            inputs=None,
            # Ensure outputs match what update_log returns
            outputs=[agent_status, hitl_controls, proposed_actions_display, action_log_display]
        )

    return demo


# --- Modified Tool Implementations (Placeholder for potential changes) ---
# Need to review/modify terminate_process_by_pid, run_arbitrary_code_safely, get_confirmation_decision
# to ensure they work correctly with config from state and potentially logging via queue.

# Example modification needed for get_confirmation_decision:
# Added confirmation_prompt_template parameter
def get_confirmation_decision(confirmation_client: OpenAI, validated_calls: list, confirmation_model_name: str, confirmation_prompt_template: str):
    """Gets approval ('APPROVE'/'DENY') from the confirmation LLM endpoint."""
    if not validated_calls:
        log_queue.put("No validated calls to seek confirmation for.") # Use queue
        return False

    proposed_calls_str = "\n".join([call['raw_call'] for call in validated_calls])
    log_queue.put(f"Seeking confirmation for:\n{proposed_calls_str}") # Use queue

    confirmation_conversation = [
        # System message might be part of the template now, adjust if needed
        {"role": "system", "content": "You are a safety confirmation agent. Respond ONLY with APPROVE or DENY, optionally followed by a short reason for denial."},
        # Use the passed template
        {"role": "user", "content": confirmation_prompt_template.format(proposed_calls_str=proposed_calls_str)}
    ]

    # Pass the specific model name for confirmation
    response = get_llm_response(confirmation_client, confirmation_conversation, confirmation_model_name)

    if response.startswith("ERROR_LLM_COMMUNICATION"):
        log_queue.put(f"Failed to get confirmation due to communication error: {response}") # Use queue
        log_queue.put("ERROR: Could not reach confirmation agent. Denying action by default.") # Use queue
        return False
    elif response.startswith("APPROVE"):
        log_queue.put(f"Confirmation agent approved actions.") # Use queue
        return True
    else:
        log_queue.put(f"Confirmation agent denied actions. Response: {response}") # Use queue
        return False

# --- Venv Setup and Relaunch Logic ---
def setup_venv():
    """Checks venv, creates/installs if needed, relaunches script within venv."""
    # Define path to python executable within the target venv directory
    venv_py = VENV_DIR / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')

    # Check if already running within the desired venv by comparing executable paths
    if Path(sys.executable).resolve() == venv_py.resolve():
        logging.info("Already running in the correct venv.")
        return True # Proceed with main logic

    # Check if this process was relaunched (indicated by --in-venv flag)
    if "--in-venv" in sys.argv:
        logging.info("Running after venv setup attempt (relaunched).")
        return True # Proceed with main logic

    # --- Setup Required if neither condition above is met ---
    print(f"Setting up virtual environment in {VENV_DIR}...")
    try:
        # Create venv directory if it doesn't exist
        if not VENV_DIR.exists():
            logging.info(f"Creating venv directory: {VENV_DIR}")
            # Create the virtual environment, ensuring pip is included
            venv.create(VENV_DIR, with_pip=True)
        else:
             logging.info("Venv directory already exists.")

        # Install dependencies using the venv's pip executable
        pip_path = str(VENV_DIR / ('Scripts/pip.exe' if sys.platform == 'win32' else 'bin/pip'))
        logging.info(f"Using pip at: {pip_path}")
        print(f"Installing dependencies: {', '.join(DEPENDENCIES)}")
        # Use subprocess to call pip install, suppressing stdout for cleaner console
        # Stderr is piped to catch potential installation errors
        subprocess.check_call([pip_path, "install"] + DEPENDENCIES, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print("Dependencies installed.")

    except Exception as e:
        # Handle errors during venv creation or dependency installation
        logging.error(f"Failed to set up virtual environment: {e}", exc_info=True)
        print(f"\nERROR: Failed to set up venv. Please check permissions and Python installation.")
        sys.exit(1) # Exit if setup fails

    # --- Relaunch the script using the venv's python interpreter ---
    print(f"Relaunching script inside virtual environment ({venv_py})...")
    # Log the command being used for relaunching
    logging.info(f"Relaunching with: {venv_py} {' '.join(sys.argv)} --in-venv")
    try:
        # Use os.execv to replace the current process with the new one running in the venv
        # Pass original command-line arguments and add '--in-venv' flag for detection
        os.execv(str(venv_py), [str(venv_py)] + sys.argv + ["--in-venv"])
    except Exception as e:
         # Handle errors during the relaunch attempt (e.g., venv_py not found)
         logging.error(f"Failed to relaunch script in venv: {e}", exc_info=True)
         print(f"\nERROR: Failed to relaunch script in venv.")
         sys.exit(1) # Exit if relaunch fails

    # This return statement should theoretically not be reached if os.execv is successful,
    # as the current process is replaced.
    return False


# TODO: Modify run_arbitrary_code_safely to accept workspace_dir and max_output_lines
#       and potentially use log_queue instead of logging directly.
# TODO: Modify terminate_process_by_pid to potentially use log_queue.
# TODO: Modify get_current_processes to potentially use log_queue.
# TODO: Modify get_llm_response to potentially use log_queue.


# --- Script Entry Point ---
if __name__ == "__main__":
    # First, ensure the virtual environment is set up correctly
    if setup_venv():
        # If setup_venv returns True, we are in the correct venv
        # Create and launch the Gradio interface
        logging.info("Starting Gradio application...")
        print("Setting up Gradio interface...")
        app = create_gradio_ui()
        print("Launching Gradio interface... Access it in your browser.")
        # Launch Gradio app (blocking call)
        app.launch() # Add share=True if you need external access
        logging.info("Gradio application stopped.")
    else:
        # This path should only be reached if setup_venv failed to relaunch correctly
        logging.error("Venv setup failed to complete or relaunch.")
        print("ERROR: Could not activate virtual environment properly.")
        sys.exit(1) # Exit with an error code
