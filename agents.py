#!/usr/bin/env python3
import os
import sys
import re
import venv
import logging
import subprocess
import traceback
import time
from itertools import zip_longest
from io import StringIO
from pathlib import Path
from datetime import datetime
from git import Repo, InvalidGitRepositoryError
import html
import json

try:
    from openai import OpenAI
    import gradio as gr
except ImportError as e:
    logger.error(f"Required package not found: {e}. Will be installed in venv setup.")

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"llm_interface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("LLMInterface")

def setup_venv():
    """Create and activate virtual environment."""
    logger.info("Setting up virtual environment...")

    venv_dir = Path(".venv")
    if not venv_dir.exists():
        logger.info("Creating new virtual environment...")
        venv.create(venv_dir, with_pip=True)

    if sys.platform == "win32":
        python_path = venv_dir / "Scripts" / "python.exe"
        pip_path = venv_dir / "Scripts" / "pip.exe"
    else:
        python_path = venv_dir / "bin" / "python"
        pip_path = venv_dir / "bin" / "pip"

    # Install required packages
    try:
        logger.info("Installing required packages...")
        subprocess.check_call([str(pip_path), "install", "gradio", "openai", "GitPython"])
        logger.info("Package installation successful")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install packages: {e}")
        raise

    return str(python_path)

def restart_in_venv():
    """Ensure running in virtual environment."""
    if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
        logger.info("Not running in venv, setting up and restarting...")
        python_path = setup_venv()
        os.execv(python_path, [python_path] + sys.argv)

class ExecutionManager:
    """Manages code execution and diffs."""

    def __init__(self):
        self.repo = self.get_git_repo()
        self.last_code = None  # Store the last executed code
        self.last_output = None  # Store the last execution output

    def get_git_repo(self):
        """Get the Git repository or None if not a Git repo."""
        try:
            repo = Repo(search_parent_directories=True)
            return repo
        except InvalidGitRepositoryError:
            logger.info("Not a Git repository.")
            return None

    def capture_file_state(self):
        """Capture the current state of .py files."""
        return {}

    def generate_diff(self, old_state, new_state):
       """Generate diff between two file states."""
       return ""

    def update_last_code_and_output(self, code, output):
        """Updates the last executed code and output."""
        self.last_code = code
        self.last_output = output

    def get_last_code_html(self):
        """Returns the last executed code as HTML."""
        if self.last_code:
            escaped_code = html.escape(self.last_code)
            return f"<pre><code>{escaped_code}</code></pre>"
        else:
            return "<p>No code executed yet.</p>"

    def get_last_output_html(self):
        """Returns the last output as HTML."""
        if self.last_output:
            escaped_output = html.escape(self.last_output)
            return f"<pre>{escaped_output}</pre>"
        else:
            return "<p>No output yet.</p>"

class LLMManager:
    def __init__(self, execution_manager):
        logger.info("Initializing LLMManager...")
        self.execution_manager = execution_manager
        try:
            self.llama_api = OpenAI(
                api_key="api_key",
                base_url="http://127.0.0.1:1234/v1/"  # Your local/remote LLM base URL
            )

            # Track execution context and test passes
            self.last_execution_locals = {}
            self.passed_tests_count = 0
            self.max_passed_tests = 4  # Increase if needed

            # Enhanced system message
            self.system_message = {
                "role": "system",
                "content": """You are an AI assistant with Python code execution capabilities.

1. For code execution, use:
RUN-CODE
```python
your_code_here


For tests, use:
TEST-ASSERT
```python
assert condition, "Test message"

Important rules:

Each block must start with its marker on its own line (e.g. \n)

Run-Code Code must be within triple backticks with 'python' specified

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

Test-Assert Tests have access to variables from code execution

Generation stops after 2 successful test passes

Example:
I'll create a function and test it.

RUN-CODE

def add(a, b):
    return a + b
result = add(5, 7)
print(f'Result: {result}')

TEST-ASSERT

assert result == 12, "Addition should work"
assert add(-1, 1) == 0, "Should handle negatives"
```"""
            }

            self.model_a_id = "qwen2.5-coder-7b-instruct"
            self.model_b_id = "qwen2.5-coder-14b-instruct"
            self.conversation = [self.system_message]

        except Exception as e:
            logger.error(f"Failed to initialize LLMManager: {e}")
            raise

    def run_code(self, code):
        """Execute code with safety checks and diff tracking."""
        logger.info("Preparing to execute code block")
        logger.debug(f"Code to execute:\n{code}")

        # Basic safety checks
        dangerous_patterns = [
            "rm -rf",
            "system(",
            "eval(",
            "exec(",
            "input(",
            "requests.",
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

            # Execute code and save locals for test access
            self.last_execution_locals = {}
            exec(code, safe_globals, self.last_execution_locals)
            output = captured_output.getvalue()
            logger.info("Code execution successful")

            # Append the copy/paste footer to the output
            output += "\n\n---\nHave fun y'all! 🤠🪄🤖\n"

            # Update last executed code and output
            self.execution_manager.update_last_code_and_output(code, output)

            return output
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Code execution failed: {str(e)}\n{error_trace}")
            return f"Error executing code:\n{error_trace}"
        finally:
            sys.stdout = old_stdout

    def run_tests(self, test_code):
        """Execute test assertions with access to previous code context."""
        logger.info("Running test assertions")
        logger.debug(f"Test code:\n{test_code}")

        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Include previous execution context in test environment
            test_globals = {
                'print': print,
                'assert': assert_,
                **self.last_execution_locals
            }

            exec(test_code, test_globals, {})
            output = captured_output.getvalue()
            self.passed_tests_count += 1
            logger.info(f"Tests passed. Count: {self.passed_tests_count}")

            # Update last executed code and output
            self.execution_manager.update_last_code_and_output(test_code, output)

            return f"Unit tests passed: {output}"  # Include captured output
        except AssertionError as e:
            logger.info(f"Test failed: {str(e)}")
            return f"Test failed: {str(e)}"
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            return f"Error running tests: {str(e)}"
        finally:
            sys.stdout = old_stdout

    def should_stop_generation(self):
        """Check if enough tests have passed to stop generation."""
        # More flexible stopping condition:
        # Stop if we've processed all blocks at least once AND met the passed test count
        return self.passed_tests_count >= self.max_passed_tests or self.all_blocks_processed_once

    def query_llama(self, model, messages, stream=False):
        """Query the LLM model with streaming support."""
        logger.info(f"Querying model: {model}")

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
                for chunk in chat_completion:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta and hasattr(chunk.choices[0].delta, 'content'):
                            content = chunk.choices[0].delta.content
                        elif hasattr(chunk.choices[0], 'text'):
                            content = chunk.choices[0].text
                        else:
                            # If there's no content, yield an empty string to maintain streaming
                            content = ""
                    else:
                        # If there's no choices, yield an empty string to maintain streaming
                        content = ""

                    if content is not None:
                        yield content
            else:
                return chat_completion.choices[0].message.content.strip()

        except Exception as e:
            error_msg = f"Error querying model {model}: {str(e)}"
            logger.error(error_msg)
            if stream:
                yield f"Error: {error_msg}"
            else:
                return f"Error: {error_msg}"

    def process_message(self, message):
        """Process a user message with code execution and testing."""
        logger.info("Processing new user message")
        self.passed_tests_count = 0  # Reset test counter
        self.all_blocks_processed_once = False # Reset block processing flag

        if not message.strip():
            logger.warning("Empty message received")
            yield "Please enter a message", "Empty message received", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
            return

        try:
            self.conversation.append({"role": "user", "content": message})

            # --- Process Model A ---
            logger.info("Getting Model A response")
            response_a = ""

            try:
                for chunk in self.query_llama(self.model_a_id, self.conversation, stream=True):
                    response_a += chunk
                    temp_conversation = self.get_conversation_history() + f"\n{self.model_a_id}: {response_a}\n\n"
                    yield temp_conversation, "Processing Model A response...", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                    
            except Exception as e:
                error_msg = f"Error getting Model A response: {str(e)}"
                logger.error(error_msg)
                yield self.get_conversation_history(), "Error with Model A", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                return

            self.conversation.append({"role": "assistant", "name": self.model_a_id, "content": response_a})

            # --- Process code and test blocks from Model A ---
            code_blocks = re.findall(r'RUN-CODE\n\s*```(?:python)?\n(.*?)\n\s*```', response_a, re.DOTALL)
            test_blocks = re.findall(r'TEST-ASSERT\n\s*```(?:python)?\n(.*?)\n\s*```', response_a, re.DOTALL)

            logger.debug(f"Model A Code Blocks: {code_blocks}")
            logger.debug(f"Model A Test Blocks: {test_blocks}")


            # If there are code blocks or test blocks
            if code_blocks or test_blocks:
                # Iterate over code and test blocks
                for i, code in enumerate(code_blocks):
                     # Execute the code block
                    if code:
                        logger.info(f"Executing code block {i+1} from Model A")
                        logger.debug(f"Code to execute (Model A block {i+1}):\n{code.strip()}")
                        output = self.run_code(code.strip())
                        logger.debug(f"Output from code block {i+1}:\n{output}")
                        print(f"Output from code block {i+1}:\n{output}")  # Debugging

                        # Append the output to the conversation
                        code_response = f"Code block {i+1} output:\n{output}"
                        self.conversation.append({"role": "assistant", "name": self.model_a_id, "content": code_response})
                        yield self.get_conversation_history(), f"Executed code block {i+1} from Model A", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                        time.sleep(0.05)

                        # Run associated tests if they exist
                        if i < len(test_blocks):
                            test = test_blocks[i]
                            logger.info(f"Executing test block {i+1} from Model A")
                            logger.debug(f"Test to execute (Model A block {i+1}):\n{test.strip()}")
                            test_result = self.run_tests(test.strip())
                            logger.debug(f"Result from test block {i+1}:\n{test_result}")
                            print(f"Result from test block {i+1}:\n{test_result}")  # Debugging
                            
                            yield self.get_conversation_history(), f"Executed test block {i+1} from Model A", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                            time.sleep(0.05)

                        if self.should_stop_generation():
                            logger.info("Stopping generation - required test passes achieved")
                            yield self.get_conversation_history(), "Complete", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                            return
                
                self.all_blocks_processed_once = True

            # --- Handoff to Model B ---
            print("\n--- Handoff to Model B ---")
            print(f"Conversation so far:\n{self.get_conversation_history()}")

            # --- Process Model B if needed ---
            if not self.should_stop_generation():
                logger.info("Getting Model B response")
                response_b = ""
                try:
                    for chunk in self.query_llama(self.model_b_id, self.conversation, stream=True):
                        response_b += chunk
                        temp_conversation = self.get_conversation_history() + f"\n{self.model_b_id}: {response_b}\n\n"
                        yield temp_conversation, "Processing Model B response...", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()

                except Exception as e:
                    error_msg = f"Error getting Model B response: {str(e)}"
                    logger.error(error_msg)
                    yield self.get_conversation_history(), "Error with Model B", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                    return

                self.conversation.append({"role": "assistant", "name": self.model_b_id, "content": response_b})

                # --- Process code and test blocks from Model B ---
                code_blocks = re.findall(r'RUN-CODE\n\s*```(?:python)?\n(.*?)\n\s*```', response_b, re.DOTALL)
                test_blocks = re.findall(r'TEST-ASSERT\n\s*```(?:python)?\n(.*?)\n\s*```', response_b, re.DOTALL)

                logger.debug(f"Model B Code Blocks: {code_blocks}")
                logger.debug(f"Model B Test Blocks: {test_blocks}")


                # If there are code blocks or test blocks
                if code_blocks or test_blocks:
                    # Iterate over code and test blocks
                    for i, code in enumerate(code_blocks):
                        # Execute the code block
                        if code:
                            logger.info(f"Executing code block {i+1} from Model B")
                            logger.debug(f"Code to execute (Model B block {i+1}):\n{code.strip()}")
                            output = self.run_code(code.strip())
                            logger.debug(f"Output from code block {i+1}:\n{output}")
                            print(f"Output from code block {i+1}:\n{output}") # Debugging

                            # Append the output to the conversation
                            code_response = f"Code block {i+1} output:\n{output}"
                            self.conversation.append({"role": "assistant", "name": self.model_b_id, "content": code_response})
                            yield self.get_conversation_history(), f"Executed code block {i+1} from Model B", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                            time.sleep(0.05)

                            # Run associated tests if they exist
                            if i < len(test_blocks):
                                test = test_blocks[i]
                                logger.info(f"Executing test block {i+1} from Model B")
                                logger.debug(f"Test to execute (Model B block {i+1}):\n{test.strip()}")
                                test_result = self.run_tests(test.strip())
                                logger.debug(f"Result from test block {i+1}:\n{test_result}")
                                print(f"Result from test block {i+1}:\n{test_result}")  # Debugging
                                
                                yield self.get_conversation_history(), f"Executed test block {i+1} from Model B", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                                time.sleep(0.05)

                            if self.should_stop_generation():
                                logger.info("Stopping generation - required test passes achieved")
                                yield self.get_conversation_history(), "Complete", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                                return

                yield self.get_conversation_history(), "Completed", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            yield self.get_conversation_history(), "Error occurred", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()

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
            self.passed_tests_count = 0  # Reset test counter
            logger.info("Conversation cleared")
            return "Conversation cleared."
        except Exception as e:
            error_msg = f"Error clearing conversation: {str(e)}"
            logger.error(error_msg)
            return error_msg

def create_ui():
    """Create and configure the Gradio interface."""
    logger.info("Creating Gradio interface")

    try:
        execution_manager = ExecutionManager()
        manager = LLMManager(execution_manager)

        with gr.Blocks(title="🚂🤖🪄 Conductor") as interface:
            gr.Markdown("# 🚂🤖🪄 Conductor")
            gr.Markdown("Enter your message to interact with the AI models. Code will be executed and tested until pass criteria are met.")

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
                        label="Conversation & Results",
                        lines=20,
                        interactive=False
                    )

            last_code_display = gr.HTML(
                label="Last Executed Code"
            )

            last_output_display = gr.HTML(
                label="Last Output"
            )

            with gr.Row():
                show_last_code_btn = gr.Button("Show Last Code")
                show_last_output_btn = gr.Button("Show Last Output")

            status_display = gr.Textbox(
                label="Status/Tests",
                lines=2,
                interactive=False,
                visible=True
            )

            def handle_submit(message):
                """Handle message submission with streaming."""
                if not message:
                    return "", "Please enter a message", "", ""

                try:
                    logger.info(f"Handling new message: {message[:50]}...")

                    result = manager.process_message(message)

                    for response in result:
                        yield response # No need for time.sleep here, it's handled in process_message

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    yield "", f"Error: {e}", execution_manager.get_last_code_html(), execution_manager.get_last_output_html()

            def handle_stop():
                """Handle stop button click."""
                # The stopping mechanism is handled in should_stop_generation
                return "Stopping generation...", "Stopping..."

            def handle_clear():
                """Handle conversation clearing."""
                try:
                    result = manager.clear_conversation()
                    return "", result, "<p>No code executed yet.</p>", "<p>No output yet.</p>"
                except Exception as e:
                    error_msg = f"Error clearing conversation: {str(e)}"
                    logger.error(error_msg)
                    return "", error_msg, execution_manager.get_last_code_html(), execution_manager.get_last_output_html()

            def handle_show_last_code():
                """Handle show last code button click."""
                return execution_manager.get_last_code_html()

            def handle_show_last_output():
                """Handle show last output button click."""
                return execution_manager.get_last_output_html()

            # Wire up the interface events
            submit_btn.click(
                fn=handle_submit,
                inputs=input_message,
                outputs=[conversation_display, status_display, last_code_display, last_output_display],
                show_progress=True
            )

            stop_btn.click(
                fn=handle_stop,
                inputs=None,
                outputs=[conversation_display, status_display]
            )

            clear_btn.click(
                fn=handle_clear,
                inputs=None,
                outputs=[conversation_display, status_display, last_code_display, last_output_display]
            )

            # Wire up the last code and output-related events
            show_last_code_btn.click(
                fn=handle_show_last_code,
                inputs=None,
                outputs=last_code_display
            )

            show_last_output_btn.click(
                fn=handle_show_last_output,
                inputs=None,
                outputs=last_output_display
            )

            # Show conversation history on load
            interface.load(
                fn=manager.get_conversation_history,
                inputs=None,
                outputs=conversation_display
            )

        interface.queue()
        return interface

    except Exception as e:
        error_msg = f"Error creating UI: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise

def main():
    """Main entry point."""
    logger.info("🚂🤖🪄 Initializing Conductor ")

    try:
        # Ensure we're running in a virtual environment
        restart_in_venv()

        # Create and launch the interface
        interface = create_ui()
        logger.info("Launching Gradio interface")
        interface.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=31337,
            debug=True
        )

    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
