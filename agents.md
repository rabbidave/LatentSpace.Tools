# LLM Interface with Code Execution

An interactive interface for running multiple local LLMs with Python code execution capabilities.

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

## Customizing System Prompt

Edit `system_message` in `LLMManager.__init__`:
```python
self.system_message = {
    "role": "system",
    "content": """Your custom instructions here.
    
    Code execution format:
    RUN-CODE
    ```python
    your_code_here
    ```
    """
}
```

## Code Execution

The interface supports Python code execution with safety restrictions:

### Allowed Operations
- File operations (os, Path)
- Basic Python functions
- Console output
- File reading/writing

### Blocked Operations
- Network requests
- System commands
- Input operations
- Unsafe imports

### Example Code Block
```python
RUN-CODE
```python
with open('test.txt', 'w') as f:
    f.write('Hello World')
print('File created!')
```
```

## Usage

1. Start the interface:
```bash
python gradio-llm-interface.py
```

2. Access web UI at `http://localhost:7860`

3. Features:
- Real-time streaming responses
- Stop generation button
- Code execution with safety controls
- Dual model interaction

## Logging

Logs are stored in `logs/` directory. Set logging level in script:
```python
logging.basicConfig(level=logging.DEBUG)  # More verbose
# or
logging.basicConfig(level=logging.INFO)   # Less verbose
```