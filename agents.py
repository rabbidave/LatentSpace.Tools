#!/usr/bin/env python3
import os
import sys
import re
import venv
import logging
import subprocess
import traceback
import time
from io import StringIO
from pathlib import Path
from datetime import datetime

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"llm_interface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more verbose logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("LLMInterface")

def setup_venv():
    """Create and activate a virtual environment, installing required packages."""
    logger.info("Setting up virtual environment...")
    
    venv_dir = Path(".venv")
    if not venv_dir.exists():
        logger.info("Creating new virtual environment...")
        venv.create(venv_dir, with_pip=True)
    
    # Get the path to the Python executable in the virtual environment
    if sys.platform == "win32":
        python_path = venv_dir / "Scripts" / "python.exe"
        pip_path = venv_dir / "Scripts" / "pip.exe"
    else:
        python_path = venv_dir / "bin" / "python"
        pip_path = venv_dir / "bin" / "pip"

    # Install required packages
    requirements = ["gradio", "openai"]
    logger.info("Installing required packages...")
    try:
        subprocess.check_call([str(pip_path), "install"] + requirements)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        raise

    return str(python_path)

def restart_in_venv():
    """Restart the script in the virtual environment if not already running in it."""
    if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
        logger.info("Not running in venv, setting up and restarting...")
        python_path = setup_venv()
        
        logger.info("Restarting script in virtual environment...")
        try:
            os.execv(python_path, [python_path] + sys.argv)
        except Exception as e:
            logger.error(f"Failed to restart in virtual environment: {e}")
            raise
    else:
        logger.info("Already running in virtual environment")

# Only import Gradio and OpenAI after ensuring we're in the venv
try:
    import gradio as gr
    from openai import OpenAI
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    raise

class LLMManager:
    def __init__(self):
        logger.info("Initializing LLMManager...")
        try:
            self.llama_api = OpenAI(
                api_key="api_key",
                base_url="http://127.0.0.1:1234/v1/"
            )
            
            # Enhanced system message with clearer code execution instructions
            self.system_message = {
                "role": "system",
                "content": """You are an AI assistant with Python code execution capabilities. To run code:

1. ALWAYS use this exact format:
RUN-CODE
```python
your_code_here
```

2. Important rules for code generation:
- Each code block must start with 'RUN-CODE' on its own line
- Code must be within triple backticks with 'python' specified
- Keep code blocks focused and self-contained
- Avoid using input() or network requests
- You can use os library for file operations
- Always explain what the code will do before running it

Example:
I'll create a simple file with some text.

RUN-CODE
```python
with open('test.txt', 'w') as f:
    f.write('Hello from the AI assistant!')
print('File created successfully!')
```

Remember: The RUN-CODE marker is essential for code execution."""
            }
            
            # Test API connection with code generation prompt
            logger.info("Testing API connection with code generation prompt...")
            test_prompt = "Generate a simple Python code to print numbers 1 to 5."
            try:
                test_response = self.llama_api.chat.completions.create(
                    model="exaone-3.5-32b-instruct@q5_k_m",
                    messages=[
                        self.system_message,
                        {"role": "user", "content": test_prompt}
                    ],
                    max_tokens=500
                )
                logger.info(f"API test response: {test_response}")
                
                # Check if response contains code block
                test_content = test_response.choices[0].message.content
                if "RUN-CODE" in test_content:
                    logger.info("API test successful - code generation confirmed")
                else:
                    logger.warning("API test response doesn't contain code block")
                
            except Exception as e:
                logger.error(f"API test failed: {str(e)}")
                raise
            
            self.model_a_id = "exaone-3.5-32b-instruct@q4_k_m"
            self.model_b_id = "qwq-32b-preview"
            
            self.conversation = [self.system_message]
            logger.info("LLMManager initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLMManager: {e}")
            raise

    def query_llama(self, model, messages, stream=False):
        """Query the LLM model with streaming and cancellation support."""
        logger.info(f"Querying model: {model}")
        logger.debug("Messages being sent:")
        for msg in messages:
            logger.debug(f"- {msg['role']}: {msg['content'][:200]}...")  # Log first 200 chars of each message
        
        try:
            chat_completion = self.llama_api.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=stream,
                top_p=0.95,
                presence_penalty=0,
                frequency_penalty=0
            )
            
            if stream:
                response = ""
                try:
                    for chunk in chat_completion:
                        logger.debug(f"Raw chunk: {chunk}")
                        content = None
                        
                        # Enhanced chunk handling
                        if hasattr(chunk.choices[0], 'delta'):
                            content = getattr(chunk.choices[0].delta, 'content', None)
                        elif hasattr(chunk.choices[0], 'text'):
                            content = chunk.choices[0].text
                        elif hasattr(chunk.choices[0], 'message'):
                            content = chunk.choices[0].message.content
                            
                        if content:
                            logger.debug(f"Extracted content: {content}")
                            response += content
                            # Check for partial code blocks
                            if "RUN-CODE" in response:
                                logger.info("Detected potential code block forming")
                            yield content
                    
                    logger.info(f"Stream completed. Final response length: {len(response)}")
                    if "RUN-CODE" in response:
                        logger.info("Final response contains code block(s)")
                        code_blocks = re.findall(r'RUN-CODE\n```(?:python)?\n(.*?)\n```', response, re.DOTALL)
                        logger.info(f"Found {len(code_blocks)} code block(s)")
                    
                except Exception as e:
                    error_msg = f"Error processing stream: {str(e)}"
                    logger.error(error_msg)
                    yield error_msg
            else:
                response = chat_completion.choices[0].message.content
                logger.info(f"Received response of length: {len(response)}")
                if "RUN-CODE" in response:
                    logger.info("Response contains code block(s)")
                return response.strip()
                    
        except Exception as e:
            error_msg = f"Error querying model {model}: {str(e)}"
            logger.error(error_msg)
            yield error_msg if stream else error_msg

    def run_code(self, code):
        """Execute code with enhanced safety checks and logging."""
        logger.info("Preparing to execute code block")
        logger.debug(f"Code to execute:\n{code}")
        
        # Basic safety checks
        dangerous_patterns = [
            "rm -rf",  # Dangerous file operations
            "system(",  # System command execution
            "eval(",   # Code evaluation
            "exec(",   # Code execution
            "input(",  # User input
            "requests.",  # Network requests
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                error_msg = f"Potentially unsafe code detected: {pattern}"
                logger.warning(error_msg)
                return error_msg
        
        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            # Set up safe execution environment
            safe_globals = {
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'set': set,
                'os': os,
                'Path': Path
            }
            
            # Execute code in restricted environment
            exec(code, safe_globals, {})
            output = captured_output.getvalue()
            logger.info("Code execution successful")
            logger.debug(f"Code output:\n{output}")
            return output
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Code execution failed: {str(e)}\n{error_trace}")
            return f"Error executing code:\n{error_trace}"
        finally:
            sys.stdout = old_stdout

    def process_message(self, message):
        """Process a user message with enhanced code detection and execution."""
        logger.info("Processing new user message")
        
        if not message.strip():
            logger.warning("Empty message received")
            yield "Please enter a message", "Empty message received"
            return

        try:
            self.conversation.append({"role": "user", "content": message})
            
            # Enhanced Model A processing with code detection
            logger.info("Getting Model A response")
            response_a = ""
            
            try:
                async_response = self.query_llama(self.model_a_id, self.conversation, stream=True)
                for chunk in async_response:
                    if isinstance(chunk, str) and chunk.startswith("Error"):
                        yield chunk, "Error occurred with Model A"
                        return
                    response_a += chunk
                    yield f"Model A Response:\n{response_a}\n\nModel B Response: Waiting...", "Processing Model A response..."
                
                # Log complete Model A response
                logger.debug(f"Complete Model A response:\n{response_a}")
                
            except Exception as e:
                error_msg = f"Error getting Model A response: {str(e)}"
                logger.error(error_msg)
                yield error_msg, "Error with Model A"
                return

            self.conversation.append({"role": "assistant", "name": self.model_a_id, "content": response_a})
            
            # Enhanced code detection and execution for Model A
            code_blocks = re.findall(r'RUN-CODE\n```(?:python)?\n(.*?)\n```', response_a, re.DOTALL)
            if code_blocks:
                logger.info(f"Found {len(code_blocks)} code block(s) in Model A response")
                for i, code in enumerate(code_blocks, 1):
                    logger.info(f"Executing code block {i}")
                    output = self.run_code(code.strip())
                    code_response = f"Code block {i} output:\n{output}"
                    self.conversation.append({"role": "assistant", "name": self.model_a_id, "content": code_response})
                    yield f"Model A Response:\n{response_a}\n\nCode Output:\n{output}\n\nModel B Response: Waiting...", f"Executed code block {i} from Model A"
            
            # Similar enhancement for Model B
            logger.info("Getting Model B response")
            response_b = ""
            
            try:
                async_response = self.query_llama(self.model_b_id, self.conversation, stream=True)
                for chunk in async_response:
                    if isinstance(chunk, str) and chunk.startswith("Error"):
                        yield f"Model A Response:\n{response_a}\n\nModel B Response: Error occurred", "Error occurred with Model B"
                        return
                    response_b += chunk
                    yield f"Model A Response:\n{response_a}\n\nModel B Response:\n{response_b}", "Processing Model B response..."
                
                logger.debug(f"Complete Model B response:\n{response_b}")
                
            except Exception as e:
                error_msg = f"Error getting Model B response: {str(e)}"
                logger.error(error_msg)
                yield f"Model A Response:\n{response_a}\n\nModel B Response: Error: {error_msg}", "Error with Model B"
                return

            self.conversation.append({"role": "assistant", "name": self.model_b_id, "content": response_b})
            
            # Enhanced code detection and execution for Model B
            code_blocks = re.findall(r'RUN-CODE\n```(?:python)?\n(.*?)\n```', response_b, re.DOTALL)
            if code_blocks:
                logger.info(f"Found {len(code_blocks)} code block(s) in Model B response")
                for i, code in enumerate(code_blocks, 1):
                    logger.info(f"Executing code block {i}")
                    output = self.run_code(code.strip())
                    code_response = f"Code block {i} output:\n{output}"
                    self.conversation.append({"role": "assistant", "name": self.model_b_id, "content": code_response})
                    yield f"Model A Response:\n{response_a}\n\nModel B Response:\n{response_b}\n\nCode Output:\n{output}", f"Executed code block {i} from Model B"
            else:
                yield f"Model A Response:\n{response_a}\n\nModel B Response:\n{response_b}", "Completed"
                
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            yield error_msg

    def get_conversation_history(self):
        """Get formatted conversation history."""
        try:
            history = ""
            for msg in self.conversation[1:]:  # Skip system message
                role = msg.get("name", msg["role"])
                content = msg["content"]
                history += f"\n{role}: {content}\n"
            return history
        except Exception as e:
            error_msg = f"Error getting conversation history: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def clear_conversation(self):
        """Clear conversation history while preserving system message."""
        try:
            system_message = self.conversation[0]  # Save system message
            self.conversation = [system_message]  # Reset with only system message
            logger.info("Conversation cleared")
            return "Conversation cleared."
        except Exception as e:
            error_msg = f"Error clearing conversation: {str(e)}"
            logger.error(error_msg)
            return error_msg

def create_ui():
    """Create and configure the Gradio interface with streaming and stop functionality."""
    logger.info("Creating Gradio interface")
    
    try:
        manager = LLMManager()
        # Add cancellation flag
        manager.should_stop = False
        
        with gr.Blocks(title="LLM Interaction Manager") as interface:
            gr.Markdown("# LLM Interaction Manager")
            gr.Markdown("Enter your message below to interact with the AI models. They can generate and execute Python code.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    input_message = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Input Message",
                        lines=3
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("Submit", variant="primary")
                        stop_btn = gr.Button("Stop Generation", variant="secondary")
                        clear_btn = gr.Button("Clear Conversation")
                        
                with gr.Column(scale=3):
                    conversation_display = gr.Textbox(
                        label="Conversation History",
                        lines=20,
                        interactive=False
                    )
                    
            status_display = gr.Textbox(
                label="Status/Errors",
                lines=2,
                interactive=False,
                visible=True
            )
            
            def handle_submit(message):
                """Handle message submission with streaming updates."""
                if not message:
                    return "", "Please enter a message"
                
                try:
                    logger.info(f"Handling new message submission: {message[:50]}...")
                    manager.should_stop = False
                    
                    # Create generator for streaming updates
                    message_generator = manager.process_message(message)
                    
                    while True:
                        try:
                            if manager.should_stop:
                                logger.info("Generation stopped by user")
                                yield "Generation stopped by user.", "Stopped"
                                break
                                
                            result = next(message_generator)
                            if isinstance(result, tuple):
                                yield result[0], result[1]  # conversation, status
                            else:
                                yield result, "Processing..."
                                
                        except StopIteration:
                            break
                        except Exception as e:
                            error_msg = f"Error in message stream: {str(e)}"
                            logger.error(error_msg)
                            yield "", error_msg
                            break
                            
                except Exception as e:
                    error_msg = f"Error processing message: {str(e)}"
                    logger.error(error_msg)
                    yield "", error_msg
                try:
                    logger.info(f"Handling new message submission: {message[:50]}...")
                    manager.should_stop = False
                    
                    # Create generator for streaming updates
                    message_generator = manager.process_message(message)
                    
                    while True:
                        try:
                            if manager.should_stop:
                                logger.info("Generation stopped by user")
                                yield "Generation stopped by user.", "Stopped"
                                break
                                
                            result = next(message_generator)
                            if isinstance(result, tuple):
                                yield result[0], result[1]  # conversation, status
                            else:
                                yield result, "Processing..."
                                
                        except StopIteration:
                            break
                        except Exception as e:
                            error_msg = f"Error in message stream: {str(e)}"
                            logger.error(error_msg)
                            yield "", error_msg
                            break
                            
                except Exception as e:
                    error_msg = f"Error processing message: {str(e)}"
                    logger.error(error_msg)
                    yield "", error_msg
                    
            def handle_stop():
                """Handle stop button click."""
                manager.should_stop = True
                return "Stopping generation...", "Stopping..."
            
            def handle_clear():
                """Handle conversation clearing with error tracking."""
                try:
                    result = manager.clear_conversation()
                    return "", result
                except Exception as e:
                    error_msg = f"Error clearing conversation: {str(e)}"
                    logger.error(error_msg)
                    return "", error_msg
            
            # Wire up the interface events with streaming
            submit_event = submit_btn.click(
                fn=handle_submit,
                inputs=input_message,
                outputs=[conversation_display, status_display],
                show_progress=True
            )
            
            # Add stop button handler
            stop_btn.click(
                fn=handle_stop,
                inputs=None,
                outputs=[conversation_display, status_display]
            )
            
            clear_btn.click(
                fn=handle_clear,
                inputs=None,
                outputs=[conversation_display, status_display]
            )
            
            # Show conversation history on load
            interface.load(
                fn=manager.get_conversation_history,
                inputs=None,
                outputs=conversation_display
            )
        
        # Enable queueing for streaming
        interface.queue()
        return interface
    
    except Exception as e:
        error_msg = f"Error creating UI: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise

def main():
    """Main entry point with enhanced error handling."""
    logger.info("Starting LLM Interface application")
    
    try:
        # Ensure we're running in a virtual environment
        restart_in_venv()
        
        # Create and launch the interface
        interface = create_ui()
        logger.info("Launching Gradio interface")
        interface.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
            debug=True
        )
        
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()