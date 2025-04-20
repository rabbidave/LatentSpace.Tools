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
import time # Added for potential delays

# --- Configuration ---
VENV_DIR = Path(".agent_venv_auto_v2") # Using unique name for this version
DEPENDENCIES = ["psutil", "openai"]

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
Okay, I see WhatsApp running with PID 5678. I will kill it.
kill_process(pid=5678)
```

**Example of GOOD Output:**
```text
kill_process(pid=5678)
```

**Few-Shot Examples:**

*Example 1: WhatsApp Found*
Input Process List Snippet:
```json
[
  {{ "pid": 123, "name": "Finder", "username": "user", "cpu_percent": 0.1, "memory_mb": 50.5 }},
  {{ "pid": 456, "name": "WhatsApp", "username": "user", "cpu_percent": 5.2, "memory_mb": 250.8 }},
  {{ "pid": 789, "name": "kernel_task", "username": "root", "cpu_percent": 1.0, "memory_mb": 1024.0 }}
]
```
Expected Output:
```text
kill_process(pid=456)
```

*Example 2: No WhatsApp Found, No Action Needed*
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
2.  Identify any process named "WhatsApp". If found, generate a `kill_process(pid=...)` call for its PID.
3.  Output ONLY the required tool calls (one per line) or ONLY "NO_ACTIONS_NEEDED" if WhatsApp is not found or no other actions are deemed necessary based on your analysis and the examples provided. Adhere strictly to the output format.

**Process List Data:**
{process_list_json}
"""

# --- Prompt for Confirmation Agent ---
# Instructs the safety agent on its role and expected output (APPROVE/DENY).
CONFIRMATION_PROMPT = """
You are a safety confirmation agent. Review the following proposed tool calls.
Your goal is to prevent clearly harmful or nonsensical actions.
Approve actions unless they seem obviously dangerous (e.g., killing critical system processes, executing malicious-looking code).
Killing a standard user application like 'WhatsApp' based on a request is generally acceptable.

Proposed Tool Calls:
{proposed_calls_str}

Do you approve these actions? Respond ONLY with "APPROVE" or "DENY". If denying, add a short reason after DENY (e.g., "DENY - Attempting to kill critical process 'kernel_task'").
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
    """Retrieves current process list as JSON string."""
    processes = []
    logging.info("Attempting to retrieve process list...")
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
                logging.warning(f"Could not retrieve info for PID {proc.pid if proc else 'N/A'}: {e}")
                continue
        logging.info(f"Successfully retrieved info for {len(processes)} processes.")
        return json.dumps(processes, indent=2) # Return data as formatted JSON string
    except NameError:
         # Handle case where psutil is still not importable
         logging.error("psutil not imported. Dependencies might be missing or venv setup failed.")
         return json.dumps({"error": "psutil not available."})
    except Exception as e:
        # Catch broader errors during process iteration
        logging.error(f"Failed to retrieve process list: {e}", exc_info=True)
        return json.dumps({"error": f"Failed to retrieve process list: {str(e)}"})

# --- Tool Implementation: kill_process ---
def terminate_process_by_pid(pid: int):
    """Terminates a process by its PID, returning JSON status."""
    logging.info(f"Attempting to terminate process with PID: {pid}")
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
        logging.warning(f"Terminating process: PID={pid}, Name='{process_name}'")
        # Terminate gracefully first
        process.terminate()
        try:
            # Wait briefly for termination, then report success
            process.wait(timeout=2)
            logging.info(f"Process PID={pid}, Name='{process_name}' terminated successfully.")
            return json.dumps({"status": "success", "pid": pid, "message": f"Process PID {pid} ('{process_name}') terminated."})
        except psutil.TimeoutExpired:
            # Force kill if graceful termination failed
            logging.warning(f"Process PID={pid}, Name='{process_name}' did not terminate gracefully. Forcing kill.")
            process.kill()
            process.wait(timeout=1) # Wait briefly for kill signal to take effect
            logging.info(f"Process PID={pid}, Name='{process_name}' killed forcefully.")
            return json.dumps({"status": "success", "pid": pid, "message": f"Process PID {pid} ('{process_name}') forcefully killed."})
    # --- Error Handling for kill_process ---
    except psutil.NoSuchProcess:
        logging.error(f"Process with PID {pid} not found.")
        return json.dumps({"status": "error", "pid": pid, "message": f"Process with PID {pid} not found."})
    except psutil.AccessDenied:
        logging.error(f"Permission denied to terminate process PID {pid}.")
        return json.dumps({"status": "error", "pid": pid, "message": f"Permission denied to terminate process PID {pid}. Try running with higher privileges."})
    except ValueError as ve:
        logging.error(f"Invalid PID provided: {ve}")
        return json.dumps({"status": "error", "pid": pid, "message": f"Invalid PID provided: {str(ve)}"})
    except NameError:
         logging.error("psutil not imported. Dependencies might be missing or venv setup failed.")
         return json.dumps({"status": "error", "pid": pid, "message": "psutil not available."})
    except Exception as e:
        logging.error(f"An unexpected error occurred while trying to terminate PID {pid}: {e}", exc_info=True)
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
    ast.NameConstant, # Allow True, False, None literals
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

def validate_path(path_str):
    """Checks if a path is safe (within WORKSPACE_DIR). Raises CodeValidationError if not."""
    try:
        # Resolve path relative to the defined workspace directory
        resolved_path = (WORKSPACE_DIR / path_str).resolve()
        # Security check: Ensure the resolved path is actually inside the WORKSPACE_DIR
        # This prevents path traversal attacks (e.g., using '../')
        resolved_path.relative_to(WORKSPACE_DIR)
        return resolved_path # Return the validated, absolute path
    except ValueError: # This exception is raised by relative_to if the path is outside
        raise CodeValidationError(f"Path traversal detected or path outside designated workspace: '{path_str}'")
    except Exception as e: # Catch other filesystem or resolution errors
        raise CodeValidationError(f"Invalid or unresolvable path '{path_str}': {e}")

def validate_code_ast(code_string):
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
        #     logging.warning(f"AST node type {node_type.__name__} not explicitly allowed, review needed.")
        #     # Depending on security posture, could raise CodeValidationError here too

        # --- Specific Node Validations for Added Security ---

        # Validate 'open' calls: check path, mode, arguments
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'open':
            if not node.args: raise CodeValidationError("Invalid 'open' call: Requires at least a filename.")
            # Ensure filename argument is a simple string literal
            filename_arg = node.args[0]
            if not isinstance(filename_arg, ast.Constant) or not isinstance(filename_arg.value, str):
                raise CodeValidationError("Filename argument for 'open' must be a literal string.")
            # Validate the path is within the allowed workspace
            validated_path = validate_path(filename_arg.value) # Raises error if invalid
            logging.debug(f"Validated safe path for open call: {validated_path}") # Log success
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

def run_arbitrary_code_safely(code: str):
    """Executes AST-validated code in a restricted env, capturing output. Returns JSON status."""
    logging.info(f"Attempting to execute arbitrary code snippet (first 100 chars): {code[:100]}...")
    # Redirect stdout/stderr to capture output from the executed code
    old_stdout, old_stderr = sys.stdout, sys.stderr
    redirected_output, redirected_error = StringIO(), StringIO()
    sys.stdout, sys.stderr = redirected_output, redirected_error
    exit_code, error_message = 0, "" # Track execution status

    try:
        # 1. Validate code via AST *before* attempting execution
        validate_code_ast(code)
        logging.info("AST validation passed.")

        # 2. Prepare restricted environment globals (only safe builtins allowed)
        safe_globals = {'__builtins__': SAFE_BUILTINS}
        # NOTE: Add other safe modules (e.g., math, json, datetime) here ONLY IF
        # ABSOLUTELY NECESSARY and their usage is understood and validated.

        # 3. Execute the validated code using exec in the restricted environment
        exec(code, safe_globals, {}) # Use empty locals dictionary for isolation
        logging.info("Code execution completed.")

    except CodeValidationError as e:
        # Handle validation errors (code structure/content is disallowed)
        logging.error(f"Code validation failed: {e}")
        error_message, exit_code = f"Code validation failed: {str(e)}", 1
    except Exception as e:
        # Handle runtime errors *during* the execution of the sandboxed code
        logging.error(f"Error DURING code snippet execution: {e}", exc_info=False) # Log full trace only internally
        # Format runtime error message concisely for reporting back
        import traceback
        tb_lines = traceback.format_exception_only(type(e), e)
        error_message, exit_code = f"Runtime error during execution: {''.join(tb_lines).strip()}", 1
    finally:
        # 4. Restore original stdout/stderr streams
        sys.stdout, sys.stderr = old_stdout, old_stderr

        # 5. Get captured output, limiting lines to prevent excessive data
        stdout_val = redirected_output.getvalue().splitlines()
        stderr_val = redirected_error.getvalue().splitlines()
        limited_stdout = "\n".join(stdout_val[:MAX_OUTPUT_LINES])
        limited_stderr = "\n".join(stderr_val[:MAX_OUTPUT_LINES])
        # Add truncation notice if output exceeded the limit
        if len(stdout_val) > MAX_OUTPUT_LINES: limited_stdout += f"\n... (truncated - {len(stdout_val)} total lines)"
        if len(stderr_val) > MAX_OUTPUT_LINES: limited_stderr += f"\n... (truncated - {len(stderr_val)} total lines)"

        # 6. Format result into JSON structure for consistent return value
        result_data = {
            "status": "success" if exit_code == 0 else "error", # Overall status
            "exit_code": exit_code, # 0 for success, 1 for error
            "stdout": limited_stdout, # Captured standard output
            "stderr": limited_stderr, # Captured standard error
            "error_message": error_message # Summary of validation or runtime error
        }
        logging.info(f"Execution Result: Status={result_data['status']}, ExitCode={result_data['exit_code']}")
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
    """Generic function to query an LLM endpoint, returning the response content or error."""
    if client is None: # Check if client initialization failed earlier
        return "ERROR_LLM_COMMUNICATION: OpenAI client not initialized."
    logging.info(f"Querying model {model_name} at {client.base_url}...")
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
        logging.info(f"LLM raw response ({model_name}): {response_content[:200]}...") # Log truncated response
        return response_content
    except Exception as e:
        # Handle errors during the API call
        logging.error(f"Error querying LLM {model_name} at {client.base_url}: {e}", exc_info=True)
        return f"ERROR_LLM_COMMUNICATION: {e}" # Return error indicator

def parse_tool_calls(response_content: str):
    """Parses tool calls from LLM response based on strict line-by-line format."""
    # Handle the specific "no actions needed" response
    if response_content == "NO_ACTIONS_NEEDED":
        logging.info("LLM indicated no actions needed.")
        return [] # Return empty list, signifying no tools to call

    parsed_calls = []
    potential_calls = response_content.strip().split('\n') # Split response into lines
    logging.info(f"Potential tool call lines: {potential_calls}")

    # Process each line to see if it matches the expected tool call format
    for line in potential_calls:
        line = line.strip()
        if not line: continue # Skip empty lines

        # Match the overall tool_name(arguments) structure
        match = TOOL_CALL_REGEX.match(line)
        if not match:
            # Warn if a line doesn't conform to the expected basic structure
            logging.warning(f"Line does not match expected tool call format: '{line}'")
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
                    logging.warning(f"Could not parse pid=NUMBER from kill_process args: '{args_str}' in line '{line}'")
                    continue # Skip this invalidly formatted call
            elif tool_name == "execute_code":
                code_match = CODE_ARG_REGEX.match(args_str) # Expect 'code="STRING"' or 'code=\'STRING\''
                if code_match:
                    args_dict['code'] = code_match.group(2) # Store the code string content
                else:
                    # Argument format mismatch for execute_code
                    logging.warning(f"Could not parse code=\"...\" from execute_code args: '{args_str}' in line '{line}'")
                    continue # Skip this invalidly formatted call
            else:
                # Handle cases where the LLM hallucinates a tool name not defined
                logging.warning(f"Unsupported tool name encountered during parsing: '{tool_name}' in line '{line}'")
                continue

            # If parsing succeeded, store the structured call information
            parsed_calls.append({
                "raw_call": line, # Keep original string for validation/confirmation prompt
                "tool_name": tool_name,
                "arguments": args_dict
            })
            logging.info(f"Successfully parsed tool call: {line}")

        except Exception as e:
            # Catch errors during argument parsing/conversion (e.g., int conversion)
            logging.warning(f"Error processing arguments for tool call '{line}': {e}")

    # Return the list of successfully parsed tool calls
    return parsed_calls

def get_confirmation_decision(confirmation_client: OpenAI, validated_calls: list):
    """Gets approval ('APPROVE'/'DENY') from the confirmation LLM endpoint."""
    if not validated_calls:
        # If no calls passed the initial validation, can't seek confirmation
        logging.info("No validated calls to seek confirmation for.")
        return False

    # Format the list of validated calls for the confirmation agent's prompt
    proposed_calls_str = "\n".join([call['raw_call'] for call in validated_calls])
    logging.info(f"Seeking confirmation for:\n{proposed_calls_str}")

    # Prepare the conversation history for the confirmation agent
    confirmation_conversation = [
        {"role": "system", "content": "You are a safety confirmation agent. Respond ONLY with APPROVE or DENY, optionally followed by a short reason for denial."},
        {"role": "user", "content": CONFIRMATION_PROMPT.format(proposed_calls_str=proposed_calls_str)}
    ]

    # Query the confirmation agent
    response = get_llm_response(confirmation_client, confirmation_conversation, CONFIRMATION_MODEL_NAME)

    # Process the confirmation response
    if response.startswith("ERROR_LLM_COMMUNICATION"):
        # Handle inability to reach the confirmation agent
        logging.error(f"Failed to get confirmation due to communication error: {response}")
        print("ERROR: Could not reach confirmation agent. Denying action by default.")
        return False # Fail safely if confirmation cannot be obtained
    elif response.startswith("APPROVE"):
        # Explicit approval received
        logging.info(f"Confirmation agent approved actions.")
        print("Confirmation agent approved.")
        return True
    else:
        # Any other response (including "DENY" or unexpected formats) is treated as denial
        logging.warning(f"Confirmation agent denied actions. Response: {response}")
        print(f"Confirmation agent DENIED actions. Reason (if provided): {response}")
        return False

# --- Main Execution Logic ---
def main():
    """Main script logic: setup, prompt agent, validate, confirm, execute."""
    logging.info("Script started in virtual environment.")
    # Ensure workspace directory exists for execute_code tool's file operations
    try:
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"Workspace directory ensured: {WORKSPACE_DIR}")
    except Exception as e:
        logging.error(f"Failed to create workspace directory {WORKSPACE_DIR}: {e}. 'execute_code' file operations might fail.")
        # Allow script to continue, but warn the user or handle more gracefully if needed

    # Initialize LLM Clients (Primary and Confirmation) outside the loop
    primary_client = None
    confirmation_client = None
    primary_client_initialized = False
    confirmation_client_initialized = False

    try:
        # Ensure openai library is available after venv setup
        global OpenAI
        if OpenAI is None and 'openai' not in sys.modules: from openai import OpenAI
        elif OpenAI is None: raise NameError("OpenAI library could not be imported.")

        # Initialize client for the primary agent
        primary_client = OpenAI(base_url=API_BASE, api_key=API_KEY)
        logging.info(f"Primary OpenAI client initialized for: {API_BASE}")
        primary_client_initialized = True
        # Initialize client for the confirmation agent
        confirmation_client = OpenAI(base_url=CONFIRMATION_API_BASE, api_key=CONFIRMATION_API_KEY)
        logging.info(f"Confirmation OpenAI client initialized for: {CONFIRMATION_API_BASE}")
        confirmation_client_initialized = True
    except NameError:
         logging.error("OpenAI library not found. Check venv setup.")
         # Don't exit here, allow loop to potentially retry later if needed
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client(s): {e}")
        # Don't exit here, allow loop to potentially retry later if needed

    # --- Main Agent Loop ---
    while True:
        print("\n--- Starting New Agent Cycle ---")
        # Check if clients are initialized before proceeding
        if not primary_client_initialized or not confirmation_client_initialized:
            logging.error("LLM clients not initialized. Skipping cycle.")
            print("ERROR: LLM clients failed to initialize. Check API keys/endpoints. Retrying in 5 seconds...")
            time.sleep(5)
            continue # Attempt re-initialization or skip cycle

        # 1. Get Current Process List Data
        logging.info("Fetching current process list...")
        process_list_json = get_current_processes()
        if '"error":' in process_list_json:
            logging.error(f"Failed to get process list. Skipping cycle. Error: {process_list_json}")
            print("ERROR: Failed to get process list. Retrying in 30 seconds...")
            time.sleep(30)
            continue
        logging.info("Process list fetched successfully.")

        # 2. Format Prompt for Primary Agent
        current_prompt = SYSTEM_PROMPT.format(process_list_json=process_list_json)
        # Create conversation history for this cycle
        conversation = [
            {"role": "system", "content": "You are an AI assistant following instructions precisely and adhering strictly to output format."},
            {"role": "user", "content": current_prompt}
        ]

        # 3. Get Primary Agent's Decision (Proposed Tool Calls)
        logging.info("Querying primary agent...")
        agent_response = get_llm_response(primary_client, conversation, MODEL_NAME)
        if agent_response.startswith("ERROR_LLM_COMMUNICATION"):
            logging.error(f"Primary LLM communication error: {agent_response}. Skipping cycle.")
            print("ERROR: Failed to communicate with primary LLM. Retrying in 5 seconds...")
            time.sleep(5)
            continue
        logging.info("Primary agent responded.")

        # 4. Parse Tool Calls from the Primary Agent's Response
        logging.info("Parsing primary agent response...")
        parsed_calls = parse_tool_calls(agent_response)

        # Handle "NO_ACTIONS_NEEDED" or no valid calls
        if not parsed_calls:
            logging.info("Primary agent proposed no tool calls or indicated 'NO_ACTIONS_NEEDED'.")
            print("\nPrimary agent proposed no actions for this cycle.")
            if agent_response != "NO_ACTIONS_NEEDED":
                print(f"(Agent's raw response differed from expected 'NO_ACTIONS_NEEDED': '{agent_response[:100]}...')")
            # Go to sleep before next cycle
            print("\n--- Cycle Complete. Sleeping for 5 seconds... ---")
            time.sleep(5)
            continue # Start next cycle

        # 5. Validate Proposed Calls against Safety Regex (Structural Check)
        validated_calls = []
        print("\n--- Primary Agent Proposed Actions (Pending Validation & Confirmation) ---")
        for call_info in parsed_calls:
            raw_call = call_info["raw_call"]
            print(f"Proposed: {raw_call}")
            # Check if the raw call string matches the predefined safety regex
            if re.match(SAFETY_REGEX, raw_call):
                logging.info(f"Call PASSED safety regex validation: {raw_call}")
                validated_calls.append(call_info) # Add to list for confirmation step
            else:
                # Log and report calls failing the regex structure check
                logging.error(f"Call FAILED safety regex validation: {raw_call}")
                print(f"Action REJECTED (Failed System Regex): {raw_call}")

        # Handle case where no calls passed validation
        if not validated_calls:
            logging.info("No proposed calls passed safety regex validation.")
            print("\nNo valid actions proposed by the agent after regex check.")
            # Go to sleep before next cycle
            print("\n--- Cycle Complete. Sleeping for 5 seconds... ---")
            time.sleep(5)
            continue # Start next cycle

        # 6. Get Confirmation from Second (Confirmation) Agent
        print("\n--- Seeking Confirmation from Safety Agent ---")
        is_approved = get_confirmation_decision(confirmation_client, validated_calls)

        # 7. Execute Approved Calls (only if confirmation received)
        if not is_approved:
            logging.warning("Execution halted for this cycle due to lack of confirmation from safety agent.")
            print("\nExecution halted: Actions not approved by confirmation agent.")
            # Go to sleep before next cycle
            print("\n--- Cycle Complete. Sleeping for 5 seconds... ---")
            time.sleep(5)
            continue # Start next cycle

        # Proceed only if confirmation was received (is_approved is True)
        print("\n--- Executing Confirmed Actions ---")
        confirmed_calls = validated_calls # Use the list that passed validation
        for call_info in confirmed_calls:
            tool_name = call_info["tool_name"]
            args = call_info["arguments"] # Arguments are already parsed into dict
            raw_call = call_info["raw_call"] # For logging/reporting context
            print(f"\nExecuting: {raw_call}")

            # Dispatch call to the appropriate tool function from the TOOLS dictionary
            if tool_name in TOOLS:
                tool_function = TOOLS[tool_name]
                try:
                    # Execute tool function with unpacked arguments (**args)
                    result_json = tool_function(**args)
                    print(f"Result:\n{result_json}") # Print the JSON result from the tool
                except Exception as e:
                    # Catch errors during the execution of the tool function itself
                    logging.error(f"Error executing tool '{tool_name}' with args {args}: {e}", exc_info=True)
                    print(f"Error executing tool '{tool_name}': {e}")
            else:
                # Handle case where tool name is unknown at execution time
                logging.error(f"Unknown tool name found during execution: {tool_name}")
                print(f"Error: Unknown tool '{tool_name}'")

        logging.info("Execution actions for this cycle finished.")
        print("\n--- Cycle Complete. Sleeping for 5 seconds... ---")
        time.sleep(5) # Wait before starting the next cycle

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

# --- Script Entry Point ---
if __name__ == "__main__":
    # First, ensure the virtual environment is set up correctly
    if setup_venv():
        # If setup_venv returns True, we are in the correct venv (or were relaunched into it)
        # Proceed to run the main application logic
        main()
    else:
        # This path should only be reached if setup_venv failed to relaunch correctly
        logging.error("Venv setup failed to complete or relaunch.")
        print("ERROR: Could not activate virtual environment properly.")
        sys.exit(1) # Exit with an error code