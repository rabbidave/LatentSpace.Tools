# LLM Interface with Code Execution and Testing

An interactive interface for running multiple local LLMs with Python code execution and test assertion capabilities.

## Setup

### Prerequisites
- Python 3.7+
- LMStudio or similar local LLM server
- Required packages: `gradio`, `openai`

### Installation
1. Clone repository
2. Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install gradio openai
```

### Model Configuration
1. Start LMStudio
   - Launch LMStudio
   - Load your models
   - Start server (default: http://localhost:1234)
2. Edit model settings in `LLMManager.__init__`:
```python
self.model_a_id = "your-model-name@q4_k_m"  # First model
self.model_b_id = "your-second-model"       # Second model
```

## Code Execution and Testing

The interface supports Python code execution and test assertions:

### Code Blocks

Use the RUN-CODE marker for executable code:
```python
RUN-CODE
```python
def add_numbers(a, b):
    return a + b
result = add_numbers(5, 7)
print(f'Result: {result}')
```
```

### Test Assertions

Use the TEST-ASSERT marker for test blocks:
```python
TEST-ASSERT
```python
assert result == 12, "Addition should work"
assert add_numbers(-1, 1) == 0, "Should handle negatives"
```
```

### Test-Driven Features

- Tests have access to variables from previous code execution
- Generation stops after 2 successful test passes
- Failed tests don't count toward pass total
- Test results appear in status display

### Allowed Operations
- File operations (os, Path)
- Basic Python functions
- Console output
- File reading/writing
- Test assertions

### Blocked Operations
- Network requests
- System commands
- Input operations
- Unsafe imports
- Code evaluation (eval/exec)

## Usage

1. Start the interface:
```bash
python gradio-llm-interface.py
```

2. Access web UI at `http://localhost:1337`

3. Features:
- Real-time streaming responses
- Stop generation button
- Code execution with safety controls
- Test assertion verification
- Pass count-based stopping
- Dual model interaction

## Example Workflow

1. Send a request:
```
Write a function to calculate the factorial of a number and test it with both positive and negative inputs
```

2. The LLM might respond with:
```python
RUN-CODE
```python
def factorial(n):
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0:
        return 1
    return n * factorial(n - 1)

# Try some examples
print(f"factorial(5) = {factorial(5)}")
```
```

TEST-ASSERT
```python
assert factorial(0) == 1, "Factorial of 0 should be 1"
assert factorial(5) == 120, "Factorial of 5 should be 120"
try:
    factorial(-1)
    assert False, "Should raise error for negative numbers"
except ValueError:
    pass
```
```

The system will:
1. Execute the code block
2. Run the tests
3. Track test passes
4. Stop after required passes
5. Show all outputs in the interface

## Logging

Logs are stored in `logs/` directory. Set logging level in script:
```python
logging.basicConfig(level=logging.DEBUG)  # More verbose
# or
logging.basicConfig(level=logging.INFO)   # Less verbose
```

## Error Handling

- Failed tests show error messages
- Unsafe code patterns are blocked
- Execution errors are logged
- Test errors don't crash the system

## Custom Configuration

Edit `system_message` in `LLMManager.__init__` to customize:
- Test pass requirements
- Code execution rules
- Test assertion format
- Generation behavior