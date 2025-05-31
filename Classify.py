import os
os.environ["HF_TOKEN"] = "hf_<insert_token_here>"

#!/usr/bin/env python
"""
Enhanced Standard Transformer-Based Classification and Reranking Service 
with VLM-Powered Markdown Processing and Complete Policy Validation

This enhanced version includes:
1. VLM-based Documentation Processing
2. Complete Policy Validation
3. Auto-Localization of Dependencies
4. Comprehensive Testing Suite
"""

import os
import sys
import venv
import json
import time
import shutil
import argparse
import logging
import subprocess
import signal
import copy
import tempfile
import requests
import re
import ast
import urllib.request
import urllib.parse
from unittest import mock # For mocking in tests
import base64 # Added for minimal PNG
from huggingface_hub import hf_hub_download

try:
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    from huggingface_hub.utils._errors import HfHubHTTPError # Older path

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from PIL import Image  # Add this import for type checking

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) # Logger instance will be created after venv check

# Create global logger instance
logger = logging.getLogger("classifier_service_tool")

# Venv setup constants
VENV_DIR = Path(__file__).parent.resolve() / ".venv_classifier_service_tool"
REQUIRED_PACKAGES = [
    "transformers[torch]>=4.48.0",
    "numpy",
    "scikit-learn",
    "tqdm",
    "flask",
    "waitress",
    "packaging",
    "flask-cors",
    "protobuf",
    "tiktoken",
    "sentencepiece",
    "opencv-python",
    "Pillow",
    "sentence-transformers>=2.2.0",
    "ranx>=0.3.10",
    "requests>=2.25.0",
    "llama-cpp-python",  # Added for VLM markdown processing
    "huggingface-hub>=0.20.0", # For downloading models from the Hub
]

# Global Variables
stop_signal_received = False
DEFAULT_DOCS_URL = "https://github.com/rabbidave/LatentSpace.Tools/blob/main/classify.md"
RAG_COMPONENT_CACHE = {}

# Global flag, will be updated in __main__ after venv confirmation and llama_cpp import attempt
LLAMA_CPP_AVAILABLE = False

# Placeholder for global availability, will be properly imported in __main__ if venv is OK
np = None
SentenceTransformer = None
Flask = None
request = None
jsonify = None
CORS = None
serve = None
torch = None

def setup_signal_handling():
    """Set up signal handling for graceful shutdown."""
    def signal_handler(sig, frame):
        global stop_signal_received
        if not stop_signal_received:
            logger.info(f"Received signal {sig}. Initiating graceful shutdown...")
            stop_signal_received = True
        else:
            logger.warning(f"Received second signal {sig}. Forcing exit.")
            sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

def ensure_venv():
    """Checks for venv, creates/installs if needed, and re-executes if not active."""
    # USER_PROVIDED_TOKEN_CHECK_START
    print(f"ensure_venv: Python sees HF_TOKEN: {os.environ.get('HF_TOKEN')}")
    # USER_PROVIDED_TOKEN_CHECK_END
    venv_path = os.path.abspath(VENV_DIR)
    
    # Determine the expected Python executable path within the venv
    if sys.platform == "win32":
        expected_python_executable = os.path.join(venv_path, "Scripts", "python.exe")
        pip_executable_in_venv = os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        expected_python_executable = os.path.join(venv_path, "bin", "python")
        pip_executable_in_venv = os.path.join(venv_path, "bin", "pip")

    # Normalize paths for reliable comparison
    is_running_in_target_venv_executable = os.path.normcase(os.path.abspath(sys.executable)) == os.path.normcase(os.path.abspath(expected_python_executable))
    
    # Also check sys.prefix, as sys.executable might be a shim in some cases
    is_sys_prefix_in_target_venv = os.path.normcase(sys.prefix).startswith(os.path.normcase(venv_path))

    actually_in_target_venv = is_running_in_target_venv_executable or is_sys_prefix_in_target_venv
    logger.debug(f"Venv check: sys.executable match: {is_running_in_target_venv_executable}, sys.prefix match: {is_sys_prefix_in_target_venv}. Effective in venv: {actually_in_target_venv}")


    if actually_in_target_venv:
        logger.info(f"Running with Python interpreter from '{VENV_DIR}'. Venv considered active.")
        # Optional: Check if a core dependency is importable. If not, attempt reinstall ONCE without re-exec.
        # This handles cases where the venv exists but might be corrupted.
        try: # Quick check for a key dependency
            import sentence_transformers
            logger.debug("Key dependency (sentence_transformers) seems available in venv.")
        except ImportError:
            logger.warning("Key dependency (sentence_transformers) not importable despite being in venv. Attempting re-install of all packages...")
            try:
                subprocess.run([pip_executable_in_venv, "install", "--disable-pip-version-check"] + REQUIRED_PACKAGES, check=True, capture_output=True, text=True)
                logger.info("Re-installation of packages in existing venv completed.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error during re-installation of packages in existing venv: {e.stderr}")
                # Proceeding, but imports might fail later. Script will not loop on this.
        return True

    logger.info(f"Setting up virtual environment at '{venv_path}'...")
    
    if not os.path.isdir(venv_path):
        try:
            venv.create(venv_path, with_pip=True, system_site_packages=False)
            logger.info("Virtual environment created successfully.")
        except Exception as e:
            logger.error(f"Error creating virtual environment: {e}", exc_info=True)
            sys.exit(1)

    if sys.platform == "win32":
        python_executable = os.path.join(venv_path, "Scripts", "python.exe")
        pip_executable = os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        python_executable = os.path.join(venv_path, "bin", "python")
        pip_executable = os.path.join(venv_path, "bin", "pip")

    if not os.path.exists(python_executable) or not os.path.exists(pip_executable):
        logger.error(f"Virtual environment setup failed. Executables not found.")
        sys.exit(1)

    try:
        subprocess.run([pip_executable, "install", "--upgrade", "pip", "--disable-pip-version-check"],
                      check=True, capture_output=True, text=True)
        logger.info("Pip upgraded successfully.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to upgrade pip: {e.stderr}")

    current_packages_to_install = list(REQUIRED_PACKAGES)
    
    logger.info(f"Installing required packages: {', '.join(current_packages_to_install)}")
    install_command = [pip_executable, "install", "--disable-pip-version-check"] + current_packages_to_install
    try:
        result = subprocess.run(install_command, check=True, capture_output=True, text=True)
        logger.info("Required packages installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing packages: {e.stderr}")
        sys.exit(1)

    logger.info(f"Restarting script using Python from '{venv_path}'...")
    script_path = os.path.abspath(__file__)
    
    # Prepare arguments for re-execution with quoting for Windows
    original_args = sys.argv[1:]
    processed_args_for_exec = []

    if sys.platform == "win32":
        for arg in original_args:
            # If argument contains a space and is not already "simply" quoted.
            # This handles basic cases. Complex cases with internal quotes might need more.
            if " " in arg and not (arg.startswith('"') and arg.endswith('"')):
                processed_args_for_exec.append(f'"{arg}"')
            else:
                processed_args_for_exec.append(arg)
    else:
        # For non-Windows, pass arguments as they are.
        processed_args_for_exec = original_args
            
    exec_args = [python_executable, script_path] + processed_args_for_exec
    
    logger.debug(f"ensure_venv: Re-executing with args: {exec_args}") # Log the exact args for os.execv

    try:
        # Preserve environment variables including HF_TOKEN and log them for debugging
        current_env = os.environ.copy()
        logger.debug(f"ensure_venv: Environment before execve - HF_TOKEN: {'SET' if 'HF_TOKEN' in current_env else 'NOT SET'}")
        if 'HF_TOKEN' in current_env:
            logger.debug(f"ensure_venv: HF_TOKEN value before execve: {current_env['HF_TOKEN'][:5]}...")
        os.execve(python_executable, exec_args, current_env)
    except OSError as e:
        logger.error(f"os.execv failed: {e}", exc_info=True)
        sys.exit(1)
    
    return False # Should not be reached if os.execv is successful


# --- VLM-Based Markdown Processing ---

class MarkdownReformatterVLM:
    """VLM-powered markdown reformatter using GGUF models via llama-cpp-python."""
    
    # Default model changed to the one known to work in tests
    DEFAULT_HF_REPO_ID_FALLBACK = "bartowski/google_gemma-3-4b-it-qat-GGUF"
    DEFAULT_GGUF_FILENAME_FALLBACK = "google_gemma-3-4b-it-qat-IQ4_NL.gguf"

    # Default for testing, as per user request
    DEFAULT_TEST_HF_REPO_ID = "bartowski/google_gemma-3-4b-it-qat-GGUF"
    DEFAULT_TEST_GGUF_FILENAME = "google_gemma-3-4b-it-qat-IQ4_NL.gguf"

    def __init__(self, model_path_or_repo_id: str, gguf_filename_in_repo: Optional[str] = None, **kwargs):
        self.model: Optional[llama_cpp.Llama] = None
        self.is_loaded = False

        self.model_load_path: Optional[str] = None # For existing local paths
        self.hf_repo_id: Optional[str] = None
        self.gguf_filename_to_download: Optional[str] = None

        if os.path.exists(model_path_or_repo_id) and os.path.isfile(model_path_or_repo_id):
            self.model_load_path = model_path_or_repo_id
            logger.info(f"MarkdownReformatterVLM configured to use local model path: {self.model_load_path}")
        else:
            self.hf_repo_id = model_path_or_repo_id
            self.gguf_filename_to_download = gguf_filename_in_repo or self.DEFAULT_GGUF_FILENAME_FALLBACK # Fallback if only repo_id given
            # For the specific test case, DEFAULT_TEST_GGUF_FILENAME will be passed as gguf_filename_in_repo
            logger.info(f"MarkdownReformatterVLM configured to use Hugging Face model: repo_id='{self.hf_repo_id}', filename='{self.gguf_filename_to_download}'.")

        # Model parameters
        self.n_ctx = kwargs.get('n_ctx', 4096)
        self.n_gpu_layers = kwargs.get('n_gpu_layers', 0)
        self.verbose = kwargs.get('verbose', False)

    def load_model(self) -> bool:
        """Load the GGUF model."""
        if not LLAMA_CPP_AVAILABLE:
            logger.error("llama-cpp-python not available. Cannot load VLM model.")
            return False
            
        try:
            model_path_to_load = self.model_load_path

            if not model_path_to_load and self.hf_repo_id and self.gguf_filename_to_download:
                logger.info(f"Attempting to download GGUF model '{self.gguf_filename_to_download}' from repo '{self.hf_repo_id}'...")
                current_hf_token = os.environ.get("HF_TOKEN") # Use env token
                logger.debug(f"HF_TOKEN value passed to hf_hub_download: {'SET' if current_hf_token else 'NOT SET'}")
                if current_hf_token:
                    logger.debug(f"HF_TOKEN (first 5 chars for check): {current_hf_token[:5]}..." if len(current_hf_token) > 5 else "Token too short for prefix.")
                try:
                    model_path_to_load = hf_hub_download(
                        repo_id=self.hf_repo_id,
                        filename=self.gguf_filename_to_download,
                        token=os.environ.get("HF_TOKEN") # Explicitly pass the env token
                    )
                    logger.info(f"Successfully downloaded model to: {model_path_to_load}")
                except HfHubHTTPError as e_http:
                    logger.error(f"HTTP error downloading model {self.hf_repo_id}/{self.gguf_filename_to_download}: {e_http}", exc_info=True)
                    return False
                except Exception as e_download: # Catch other potential download errors
                    logger.error(f"Failed to download model {self.hf_repo_id}/{self.gguf_filename_to_download}: {e_download}", exc_info=True)
                    return False
            
            if not model_path_to_load or not os.path.exists(model_path_to_load):
                logger.error(f"Final model path for VLM not found or not resolved: {model_path_to_load if model_path_to_load else 'None'}")
                if not self.model_load_path and self.hf_repo_id : # If it was an HF attempt
                    logger.info(f"Ensure the Hugging Face repo '{self.hf_repo_id}' and filename '{self.gguf_filename_to_download}' are correct, and HF_TOKEN is valid if required.")
                return False

            logger.info(f"Loading GGUF model from path: {model_path_to_load}")
            self.model = llama_cpp.Llama(
                model_path=str(model_path_to_load), # Ensure it's a string
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose
            )
            
            self.is_loaded = True
            logger.info(f"Successfully loaded VLM model from {model_path_to_load}")
            return True
            
        except Exception as e:
            logger.error(f"Error during VLM loading: {e}", exc_info=True) # Corrected: Was 'VLM inference'
            return False
    
    def reformat(self, markdown_text: str, prompt_template: str) -> str:
        """Reformat markdown using the VLM."""
        if not self.is_loaded:
            if not self.load_model(): # Attempt to load if not already
                raise RuntimeError("VLM model not loaded and failed to load.")
        
        if not self.model: # Should be caught by above, but as a safeguard
             raise RuntimeError("VLM model is None after load attempt.")

        try:
            # Construct the full prompt
            full_prompt = prompt_template.format(raw_markdown_content=markdown_text)
            
            logger.debug(f"VLM prompt length: {len(full_prompt)} characters")
            
            # Generate response
            response = self.model(
                full_prompt,
                # max_tokens=8192, # Old: Potentially problematic if > n_ctx - prompt_tokens
                max_tokens=None, # New: Let llama-cpp-python derive from n_ctx and prompt length for safety.
                                 # Llama.__call__ defaults max_tokens to n_ctx - prompt_tokens if None.
                temperature=0.1,
                top_p=0.9,
                stop=["```\n\n", "\n\n---", "END_OUTPUT"], # Stop sequences
                echo=False # Don't echo the prompt in the output
            )
            
            if response and "choices" in response and response["choices"] and "text" in response["choices"]: # Corrected: response["choices"] is a list
                output_text = response["choices"]["text"].strip() 
                logger.debug(f"VLM output length: {len(output_text)} characters")
                return output_text
            else:
                logger.error(f"No valid response or choices from VLM. Response: {response}")
                return ""
                
        except Exception as e:
            logger.error(f"Error during VLM inference: {e}", exc_info=True)
            return ""

def create_vlm_markdown_prompt_template() -> str:
    """Create the prompt template for VLM markdown processing."""
    return """You are an expert document processor specialized in preparing markdown documents for RAG (Retrieval Augmented Generation) indexing.

Your task is to analyze the following markdown document and output a structured JSON array where each object represents a semantic chunk optimized for search and retrieval.

Requirements:
1. Create semantic chunks that preserve context and meaning.
2. Maintain code blocks as complete units when possible.
3. Identify and preserve document structure (headers, sections).
4. Generate unique, descriptive IDs for each chunk (e.g., based on section and type).
5. Extract relevant metadata for each chunk.

Output a valid JSON array with this exact structure, enclosed in a JSON code block:
```json
[
  {{
    "id": "unique_chunk_identifier_from_content_and_type",
    "text": "The complete text content of this chunk. Preserve markdown formatting within the text, especially for code blocks.",
    "metadata": {{
      "h1_section": "Primary section title or null if not applicable",
      "h2_section": "Secondary section title or null", 
      "h3_section": "Tertiary section title or null",
      "h4_section": "Quaternary section title or null",
      "is_code_block": true_if_this_entire_chunk_is_a_single_code_block_else_false,
      "contains_code_elements": true_if_chunk_contains_any_inline_code_or_code_blocks_else_false,
      "contains_commands": true_if_chunk_contains_shell_commands_or_cli_examples_else_false,
      "chunk_type": "header|content|code|example|reference|list_item|table_fragment",
      "topics": ["extracted_topic1", "extracted_topic2"]
    }}
  }}
]
```

Markdown Document to Process:
---
{raw_markdown_content}
---

JSON Output:"""

def reformat_markdown_with_vlm(
    markdown_content: str, 
    vlm_instance: MarkdownReformatterVLM, 
    source_url: str,
    target_chunk_size_hint: Optional[int] = 1000
) -> List[Dict[str, Any]]:
    """
    Reformat markdown content using VLM into structured chunks.
    """
    logger.info(f"Processing markdown with VLM ({len(markdown_content)} chars from {source_url})")
    
    try:
        # Get the prompt template
        prompt_template = create_vlm_markdown_prompt_template()
        
        # Process with VLM
        vlm_output = vlm_instance.reformat(markdown_content, prompt_template)
        
        if not vlm_output:
            logger.error(f"VLM returned empty output for {source_url}")
            # Fallback for this specific document
            return fallback_markdown_processing(markdown_content, source_url, target_chunk_size_hint)
        
        # Parse VLM output
        chunks = parse_vlm_output(vlm_output, source_url)
        
        if not chunks: # If VLM output parsing failed
            logger.warning(f"Failed to parse VLM output for {source_url}, using fallback.")
            return fallback_markdown_processing(markdown_content, source_url, target_chunk_size_hint)

        # Apply secondary chunking if needed
        if target_chunk_size_hint:
            chunks = apply_secondary_chunking(chunks, target_chunk_size_hint)
        
        logger.info(f"Successfully processed markdown from {source_url} into {len(chunks)} chunks via VLM.")
        return chunks
        
    except Exception as e:
        logger.error(f"Error in VLM markdown processing for {source_url}: {e}", exc_info=True)
        # Fallback to simple chunking for this document
        return fallback_markdown_processing(markdown_content, source_url, target_chunk_size_hint)

def parse_vlm_output(vlm_output: str, source_url: str) -> List[Dict[str, Any]]:
    """Parse VLM output into chunk structure."""
    try:
        # Extract JSON from VLM output (handles ```json ... ``` or just [...] )
        json_str = None
        # Try to find JSON within a fenced code block first
        json_match_fenced = re.search(r'```json\s*(\[.*?\])\s*```', vlm_output, re.DOTALL)
        if json_match_fenced:
            json_str = json_match_fenced.group(1)
        else:
            # If not found, try to find a raw JSON array (more lenient)
            json_match_raw = re.search(r'(\[.*?\])', vlm_output, re.DOTALL)
            if json_match_raw:
                json_str = json_match_raw.group(1)
        
        if not json_str:
            logger.error(f"Could not extract JSON array from VLM output for {source_url}. Raw output sample: {vlm_output[:500]}...")
            return []
        
        # Parse JSON
        chunks_data = json.loads(json_str)
        
        if not isinstance(chunks_data, list):
            logger.error(f"VLM output for {source_url} is not a JSON array. Parsed data type: {type(chunks_data)}")
            return []
        
        # Validate and enhance chunks
        validated_chunks = []
        for i, chunk_data in enumerate(chunks_data):
            try:
                validated_chunk = validate_and_enhance_chunk(chunk_data, source_url, i)
                if validated_chunk:
                    validated_chunks.append(validated_chunk)
            except Exception as e: # Catch errors during individual chunk validation
                logger.warning(f"Error validating chunk {i} from {source_url}: {e}. Chunk data: {str(chunk_data)[:200]}")
                continue # Skip this problematic chunk
        
        return validated_chunks
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error for {source_url}: {e}. Problematic JSON string sample: {json_str[:500] if json_str else 'N/A'}...")
        logger.debug(f"Full VLM output for {source_url} that failed to parse: {vlm_output[:1000]}...")
        return []
    except Exception as e:
        logger.error(f"Error parsing VLM output for {source_url}: {e}", exc_info=True)
        return []

def validate_and_enhance_chunk(chunk_data: Dict, source_url: str, index: int) -> Optional[Dict[str, Any]]:
    """Validate and enhance a chunk from VLM output."""
    try:
        # Ensure required fields
        if not isinstance(chunk_data, dict):
            logger.warning(f"Chunk {index} from {source_url} is not a dictionary.")
            return None
        
        text = chunk_data.get("text", "").strip()
        if not text:
            logger.warning(f"Chunk {index} from {source_url} has empty text.")
            return None
        
        # Generate ID if missing or ensure it's a string
        chunk_id = str(chunk_data.get("id", f"vlm_chunk_{source_url}_{index}"))
        
        # Ensure metadata structure
        metadata = chunk_data.get("metadata", {})
        if not isinstance(metadata, dict): # Ensure metadata is a dict
            logger.warning(f"Chunk {index} from {source_url} has invalid metadata type: {type(metadata)}. Using empty metadata.")
            metadata = {}

        enhanced_metadata = {
            "h1_section": metadata.get("h1_section"),
            "h2_section": metadata.get("h2_section"),
            "h3_section": metadata.get("h3_section"), 
            "h4_section": metadata.get("h4_section"),
            "is_code_block": bool(metadata.get("is_code_block", False)),
            "contains_code_elements": bool(metadata.get("contains_code_elements", '`' in text or "```" in text)), # Default if not provided
            "contains_commands": bool(metadata.get("contains_commands", False)),
            "chunk_type": metadata.get("chunk_type", "content"),
            "topics": metadata.get("topics", []),
            "source_url": source_url,
            "vlm_processed": True,
            "char_count": len(text),
            "chunk_index": index
        }
        
        return {
            "id": chunk_id,
            "text": text,
            "metadata": enhanced_metadata
        }
        
    except Exception as e: # Catch any unexpected error during validation
        logger.error(f"Unexpected error validating chunk {index} from {source_url}: {e}")
        return None

def apply_secondary_chunking(chunks: List[Dict[str, Any]], target_size: int) -> List[Dict[str, Any]]:
    """Apply secondary chunking to oversized chunks."""
    result_chunks = []
    
    for chunk_idx, chunk in enumerate(chunks):
        text = chunk["text"]
        if len(text) <= target_size:
            result_chunks.append(chunk)
        else:
            # Split large chunk
            logger.info(f"Applying secondary chunking to chunk {chunk.get('id', chunk_idx)} (length {len(text)}) from {chunk.get('metadata',{}).get('source_url')}")
            sub_chunks_text = split_text_with_overlap(text, target_size, target_size // 4)
            for i, sub_text in enumerate(sub_chunks_text):
                if not sub_text.strip(): # Skip empty sub-chunks
                    continue
                sub_chunk = copy.deepcopy(chunk)
                sub_chunk["text"] = sub_text
                sub_chunk["id"] = f"{chunk.get('id', f'chunk_{chunk_idx}')}_part_{i+1}"
                sub_chunk["metadata"]["char_count"] = len(sub_text)
                sub_chunk["metadata"]["is_sub_chunk"] = True
                sub_chunk["metadata"]["parent_chunk_id"] = chunk.get('id', f'chunk_{chunk_idx}')
                result_chunks.append(sub_chunk)
    
    return result_chunks

def split_text_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks, respecting sentence boundaries and newlines."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start_index = 0
    
    while start_index < len(text):
        end_index = min(start_index + chunk_size, len(text))
        
        # If we are not at the end of the text, try to find a good split point
        if end_index < len(text):
            # Prefer double newlines, then single newlines, then sentence-ending punctuation
            split_delimiters = ['\n\n', '\n', '. ', '! ', '? ']
            best_split_pos = -1

            for delim in split_delimiters:
                # Search backwards from end_index for the delimiter
                pos = text.rfind(delim, start_index, end_index)
                if pos != -1 and pos > start_index : # Ensure it's a valid split point
                    # Check if this split point is better (closer to desired chunk_size, but not too small)
                    if pos + len(delim) > start_index + (chunk_size // 4): # Avoid tiny first chunks
                         best_split_pos = pos + len(delim)
                         break # Found a good delimiter
            
            if best_split_pos != -1:
                end_index = best_split_pos
        
        current_chunk_text = text[start_index:end_index].strip()
        if current_chunk_text: # Add non-empty chunks
            chunks.append(current_chunk_text)
        
        if end_index >= len(text): # Reached the end
            break
            
        # Move start_index for the next chunk, considering overlap
        # Ensure overlap doesn't cause infinite loops if chunk_size is too small relative to overlap
        start_index = max(start_index + 1, end_index - overlap) 
        if start_index >= end_index : # Safety break if start_index doesn't advance
            logger.warning(f"Chunking might be stuck. Start: {start_index}, End: {end_index}. Advancing past end.")
            start_index = end_index 


    return chunks

def fallback_markdown_processing(markdown_content: str, source_url: str, target_chunk_size: Optional[int]) -> List[Dict[str, Any]]:
    """Fallback to simple Python-based markdown processing."""
    logger.warning(f"Using fallback markdown processing for {source_url}")
    
    # First, split by major headers (H1, H2) to create large semantic blocks
    # This regex splits by H1 or H2, keeping the delimiter (header line)
    # (?=^#{1,2} ) looks ahead for H1 or H2 at the start of a line
    blocks = re.split(r'(?=^#{1,2} )', markdown_content, flags=re.MULTILINE)
    
    all_chunks = []
    chunk_id_counter = 0
    current_h1 = None
    current_h2 = None

    for block_content in blocks:
        if not block_content.strip():
            continue

        block_lines = block_content.strip().split('\n')
        first_line = block_lines[0]

        if first_line.startswith('# ') and not first_line.startswith('##'):
            current_h1 = first_line[2:].strip()
            current_h2 = None # Reset H2 when a new H1 is encountered
            # If the block is just the header, might skip or handle as header-only chunk
            # For now, the content after header will be processed
        elif first_line.startswith('## '):
            current_h2 = first_line[3:].strip()

        # Further split the block if it's too large, or process as is
        # We can use a simple text splitter for these blocks or iterate by paragraphs/code blocks
        
        # Simplistic approach: treat the whole block as one, then apply secondary if too big
        # More advanced: iterate through paragraphs and code blocks within this block
        
        # For this fallback, let's use the split_text_with_overlap on the block_content
        # if target_chunk_size is provided.
        
        if target_chunk_size and len(block_content) > target_chunk_size:
            sub_chunks_text = split_text_with_overlap(block_content, target_chunk_size, target_chunk_size // 4)
        else:
            sub_chunks_text = [block_content]

        for sub_text in sub_chunks_text:
            if not sub_text.strip():
                continue

            # Basic metadata extraction for this sub_chunk
            # For H3/H4, one might parse sub_text, but this fallback is simpler
            current_h3 = None # Simplified: not extracting H3/H4 in this basic fallback
            current_h4 = None
            
            # Re-check headers for the sub_text, in case split_text_with_overlap split them
            sub_text_lines = sub_text.strip().split('\n')
            sub_first_line = sub_text_lines[0]
            final_h1, final_h2 = current_h1, current_h2

            if sub_first_line.startswith('# ') and not sub_first_line.startswith('##'):
                final_h1 = sub_first_line[2:].strip()
                final_h2 = None
            elif sub_first_line.startswith('## '):
                final_h2 = sub_first_line[3:].strip()


            is_code_block_chunk = sub_text.strip().startswith('```') and sub_text.strip().endswith('```')
            contains_code_elements = '`' in sub_text or "```" in sub_text
            contains_commands = bool(re.search(r'^\s*(curl|python|pip|docker|git)', sub_text, re.MULTILINE | re.IGNORECASE))


            chunk_data = {
                "id": f"fallback_chunk_{source_url}_{chunk_id_counter}",
                "text": sub_text.strip(),
                "metadata": {
                    "h1_section": final_h1,
                    "h2_section": final_h2,
                    "h3_section": current_h3, # Simplified
                    "h4_section": current_h4, # Simplified
                    "is_code_block": is_code_block_chunk,
                    "contains_code_elements": contains_code_elements,
                    "contains_commands": contains_commands,
                    "chunk_type": "code" if is_code_block_chunk else "content",
                    "topics": [], # Fallback doesn't extract topics
                    "source_url": source_url,
                    "vlm_processed": False,
                    "char_count": len(sub_text.strip()),
                    "chunk_index": chunk_id_counter
                }
            }
            all_chunks.append(chunk_data)
            chunk_id_counter += 1
            
    return all_chunks


# --- Enhanced Documentation RAG Functions ---

def fetch_documentation(docs_url_or_path_list: Union[str, List[str]]) -> str: # Corrected parameter name
    """Fetch documentation content from URL(s) or local file(s).
    If a list is provided, it concatenates content from all sources.
    If a single URL/path is provided, it fetches that one.
    """
    
    sources_to_fetch = [docs_url_or_path_list] if isinstance(docs_url_or_path_list, str) else docs_url_or_path_list
    if sources_to_fetch is None: # Handle case where docs_url is None (e.g. from argparse nargs='*')
        sources_to_fetch = []

    combined_content = ""
    
    for source_item in sources_to_fetch:
        logger.info(f"Fetching documentation from: {source_item}")
        content_for_item = ""
        try:
            if source_item.startswith(('http://', 'https://')):
                with urllib.request.urlopen(source_item, timeout=30) as response:
                    content_for_item = response.read().decode('utf-8')
            else: # Assume local file path
                file_path = Path(source_item)
                if file_path.is_file():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content_for_item = f.read()
                else:
                    logger.error(f"Local file not found: {source_item}")
                    continue # Skip this source
            
            # Add a source marker if combining multiple, or for clarity
            combined_content += f"\n\n# Source Document: {source_item}\n\n{content_for_item}"
            logger.debug(f"Successfully fetched {len(content_for_item)} characters from {source_item}")
            
        except Exception as e:
            logger.error(f"Failed to fetch documentation from {source_item}: {e}")
            continue # Skip this source
    
    if not combined_content.strip() and sources_to_fetch:
        # Raise error only if all sources failed or were empty
        raise RuntimeError(f"Could not fetch any documentation content from {sources_to_fetch}")
    
    logger.info(f"Successfully fetched {len(combined_content)} total characters of documentation.")
    return combined_content

def save_chunks_as_jsonl(chunks: List[Dict[str, Any]], output_path: Path) -> None:
    """Save chunks to a JSONL file."""
    logger.info(f"Saving {len(chunks)} chunks to {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    logger.info(f"Successfully saved chunks to {output_path}")

def build_documentation_rag_index(
    docs_url_or_path_list: Union[str, List[str]], 
    index_path: Path, 
    embedding_model_id: str, 
    chunk_size: int, 
    chunk_overlap: int, # Note: chunk_overlap is not directly used by VLM, but by secondary/fallback
    vlm_model_path: Optional[str] = None,
    processing_strategy: str = "vlm"
) -> bool:
    """Build a RAG index from documentation using VLM or fallback processing.
       Processes each document source individually for better metadata.
    """
    try:
        logger.info(f"Building documentation RAG index at {index_path} using strategy '{processing_strategy}'")
        
        all_processed_chunks: List[Dict[str, Any]] = []
        
        # Ensure docs_url_or_path_list is a list, even if None or single string was passed
        if docs_url_or_path_list is None:
            sources_to_process = []
        elif isinstance(docs_url_or_path_list, str):
            sources_to_process = [docs_url_or_path_list]
        else: # It's already a list
            sources_to_process = docs_url_or_path_list


        vlm_processor_instance: Optional[MarkdownReformatterVLM] = None
        if processing_strategy == "vlm" and vlm_model_path and LLAMA_CPP_AVAILABLE:
            # Correctly instantiate using model_path_or_repo_id for local paths or repo IDs
            # If vlm_model_path contains a GGUF filename and is not a directory, it's likely a direct path
            # If it's a repo ID, gguf_filename_in_repo might be needed, or defaults will apply
            # For this function, assume vlm_model_path is a direct file path or a repo_id that MarkdownReformatterVLM can handle.
            # Or, the user should provide both repo_id and filename_in_repo if needed for MarkdownReformatterVLM
            # Simplification: Assume vlm_model_path IS the model path or a repo_id that MarkdownReformatterVLM can handle.
            vlm_processor_instance = MarkdownReformatterVLM(model_path_or_repo_id=vlm_model_path)
            if not vlm_processor_instance.load_model():
                logger.warning("VLM model loading failed. Will fallback to Python processing for all documents.")
                processing_strategy = "python" 
                vlm_processor_instance = None 

        if not sources_to_process:
            logger.warning("No documentation sources provided to build_documentation_rag_index. Index not built.")
            return False

        for doc_source_identifier in sources_to_process:
            logger.info(f"Fetching and processing documentation from: {doc_source_identifier}")
            try:
                markdown_content_for_source = fetch_documentation(doc_source_identifier) # fetch_documentation handles single item string
                
                source_header = f"# Source Document: {doc_source_identifier}" # fetch_documentation adds this prefix
                actual_content_test = markdown_content_for_source.replace(source_header, "").strip()
                if not actual_content_test :
                    logger.warning(f"No actual content fetched from {doc_source_identifier}, skipping.")
                    continue

                source_url_for_chunks = str(doc_source_identifier) 
                doc_chunks: List[Dict[str, Any]] = []

                if processing_strategy == "vlm" and vlm_processor_instance and vlm_processor_instance.is_loaded:
                    logger.info(f"Using VLM-based markdown processing for {source_url_for_chunks}")
                    doc_chunks = reformat_markdown_with_vlm(
                        markdown_content_for_source, 
                        vlm_processor_instance, 
                        source_url_for_chunks, 
                        chunk_size 
                    )
                else:
                    if processing_strategy == "vlm": 
                        logger.info(f"VLM processing requested but conditions not met for {source_url_for_chunks}. Using fallback Python processing.")
                    else: 
                        logger.info(f"Using fallback Python-based markdown processing for {source_url_for_chunks}")
                    doc_chunks = fallback_markdown_processing(markdown_content_for_source, source_url_for_chunks, chunk_size) 
                
                if doc_chunks:
                    all_processed_chunks.extend(doc_chunks)
                else:
                    logger.warning(f"No chunks created from {source_url_for_chunks}")

            except RuntimeError as e: 
                logger.error(f"Could not fetch or process content from {doc_source_identifier}: {e}")
            except Exception as e:
                logger.error(f"General error processing document {doc_source_identifier}: {e}", exc_info=True)
        
        if not all_processed_chunks:
            logger.error("No chunks created from any documentation source(s). Index not built.")
            return False
        
        temp_jsonl = index_path.parent / f"{index_path.name}_temp_chunks.jsonl"
        save_chunks_as_jsonl(all_processed_chunks, temp_jsonl)
        
        retriever = RAGRetriever(index_path)
        retriever.index_corpus(
            corpus_path=temp_jsonl,
            embedding_model_id=embedding_model_id,
            doc_id_field="id",
            text_field="text",
            metadata_fields=[ 
                "h1_section", "h2_section", "h3_section", "h4_section",
                "is_code_block", "contains_code_elements", "contains_commands", 
                "chunk_type", "topics", "source_url", "vlm_processed",
                "char_count", "chunk_index", "is_sub_chunk", "parent_chunk_id" 
            ]
        )
        
        temp_jsonl.unlink(missing_ok=True)
        
        logger.info(f"Successfully built documentation RAG index with {len(all_processed_chunks)} chunks from {len(sources_to_process)} source(s).")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build documentation RAG index: {e}", exc_info=True)
        return False

# --- RAGRetriever Class (Preserved) ---
class RAGRetriever:
    INDEX_CONFIG_FILE = "rag_index_config.json"
    DOCUMENTS_FILE = "documents.jsonl"
    EMBEDDINGS_FILE = "embeddings.npy"

    def __init__(self, index_path: Union[str, Path]):
        self.index_path = Path(index_path)
        self.config: Optional[Dict[str, Any]] = None
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model: Optional[SentenceTransformer] = None
        self._is_loaded = False

    def _load_model(self, model_id: str):
        if self.model and self.config and self.config.get("embedding_model_id") == model_id:
            logger.debug(f"SentenceTransformer model '{model_id}' already loaded.")
            return
        try:
            logger.debug(f"RAGRetriever._load_model: Attempting to load model_id='{model_id}'.")
            current_hf_token = os.environ.get("HF_TOKEN")
            logger.debug(f"RAGRetriever._load_model: HF_TOKEN from env: {'SET' if current_hf_token else 'NOT SET'}")
            logger.debug(f"RAGRetriever._load_model: Explicit token arg being passed: {current_hf_token}")
            self.model = SentenceTransformer(model_id, token=os.environ.get("HF_TOKEN"))
            logger.info(f"SentenceTransformer model '{model_id}' loaded.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_id}': {e}", exc_info=True) # Added exc_info
            raise

    def index_corpus(self, corpus_path: Union[str, Path],
                     embedding_model_id: str = "all-MiniLM-L6-v2",
                     doc_id_field: str = "id",
                     text_field: str = "text",
                     metadata_fields: Optional[List[str]] = None,
                     batch_size: int = 32):
        logger.info(f"Starting RAG corpus indexing from '{corpus_path}'")
        corpus_path = Path(corpus_path)
        
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        self._load_model(embedding_model_id)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        parsed_documents = []
        texts_to_embed = []
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    doc_id = item.get(doc_id_field)
                    text_content = item.get(text_field)

                    if doc_id is None or text_content is None:
                        logger.warning(f"Skipping line {i+1}: missing ID or text")
                        continue
                    
                    # Ensure text_content is a string
                    text_content_str = str(text_content)

                    metadata = {}
                    if metadata_fields: # Check if metadata_fields list is provided
                        item_metadata_obj = item.get("metadata", {}) # Get metadata dict once, defaults to {}
                        for m_field in metadata_fields: # Iterate through desired metadata fields
                            if m_field in item_metadata_obj: # Check if the field exists in the item's metadata
                                metadata[m_field] = item_metadata_obj[m_field]
                            elif m_field in item: # Fallback: check if field exists at the top level of the item
                                 metadata[m_field] = item[m_field]

                    
                    parsed_documents.append({
                        "id": str(doc_id),
                        "text": text_content_str,
                        "metadata": metadata
                    })
                    texts_to_embed.append(text_content_str)

                except json.JSONDecodeError:
                    logger.warning(f"Skipping line {i+1}: JSON decode error")
                except Exception as e:
                    logger.warning(f"Skipping line {i+1} due to unexpected error: {e}")
        
        if not texts_to_embed:
            logger.error("No valid documents found in corpus to embed.")
            return

        logger.info(f"Embedding {len(texts_to_embed)} documents...")
        if not self.model:
            logger.error("Embedding model not loaded. Cannot proceed with indexing.")
            return

        embeddings_array = self.model.encode(
            texts_to_embed, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            normalize_embeddings=True # Normalizing is good for cosine similarity
        )
        
        # Save documents
        with open(self.index_path / self.DOCUMENTS_FILE, 'w', encoding='utf-8') as f_docs:
            for doc in parsed_documents:
                f_docs.write(json.dumps(doc) + '\n')
        
        # Save embeddings
        np.save(self.index_path / self.EMBEDDINGS_FILE, embeddings_array)
        
        # Save config
        self.config = {
            "embedding_model_id": embedding_model_id,
            "doc_id_field": doc_id_field,
            "text_field": text_field,
            "metadata_fields": metadata_fields or [],
            "num_documents": len(parsed_documents),
            "embedding_dimension": embeddings_array.shape if embeddings_array is not None and embeddings_array.ndim == 2 and embeddings_array.shape[0] > 0 else None, # Corrected to get dim
            "index_timestamp": time.time()
        }
        
        with open(self.index_path / self.INDEX_CONFIG_FILE, 'w', encoding='utf-8') as f_cfg:
            json.dump(self.config, f_cfg, indent=2)
        
        logger.info(f"Successfully indexed {len(parsed_documents)} documents to {self.index_path}")
        self.documents = parsed_documents
        self.embeddings = embeddings_array
        self._is_loaded = True

    def load_index(self) -> bool:
        if self._is_loaded:
            return True
            
        logger.info(f"Loading RAG index from '{self.index_path}'")
        
        if not self.index_path.is_dir():
            logger.error(f"Index directory not found: {self.index_path}")
            return False
        
        config_file = self.index_path / self.INDEX_CONFIG_FILE
        docs_file = self.index_path / self.DOCUMENTS_FILE
        embed_file = self.index_path / self.EMBEDDINGS_FILE

        if not all([config_file.exists(), docs_file.exists(), embed_file.exists()]):
            logger.error(f"Index files missing in '{self.index_path}'. Cannot load.")
            return False
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            self._load_model(self.config["embedding_model_id"])

            self.documents = []
            with open(docs_file, 'r', encoding='utf-8') as f_docs:
                for line in f_docs:
                    self.documents.append(json.loads(line))
            
            self.embeddings = np.load(embed_file)

            # CORRECTED_RAG_LOAD_INDEX_ASSERTION
            if self.embeddings is None or len(self.documents) != self.embeddings.shape[0]:
                actual_doc_count = len(self.documents)
                emb_shape = self.embeddings.shape if self.embeddings is not None else "None"
                logger.error(f"Document count ({actual_doc_count}) and embedding count's first dimension ({emb_shape}) mismatch. Index corrupt.")
                return False
            
            self._is_loaded = True
            logger.info(f"Successfully loaded RAG index with {len(self.documents)} documents from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading RAG index from {self.index_path}: {e}", exc_info=True)
            self._is_loaded = False # Ensure consistent state on failure
            self.config = None
            self.documents = []
            self.embeddings = None
            return False

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._is_loaded and not self.load_index(): # Attempt to load if not loaded
            logger.error("Cannot retrieve: RAG index not loaded and failed to load.")
            return []
        
        if not self.model or self.embeddings is None or not self.documents:
            logger.error("Cannot retrieve: model, embeddings, or documents unavailable even after load attempt.")
            return []
            
        try:
            query_embedding = self.model.encode(query, normalize_embeddings=True)
            # Ensure embeddings are 2D for dot product, query_embedding is 1D
            # Similarities: (num_docs, embed_dim) dot (embed_dim,) -> (num_docs,)
            similarities = np.dot(self.embeddings, query_embedding).flatten() 
            
            # Get indices of top_k largest similarities
            # Using argpartition for efficiency if only top_k are needed, then sort only those.
            # For typical top_k values, argsort is fine.
            if top_k >= len(similarities): # Handle case where top_k is more than available docs
                 top_k_indices = np.argsort(-similarities) # Sort all
            else:
                 top_k_indices = np.argpartition(-similarities, top_k)[:top_k]
                 # Sort only the selected top_k partition by their scores
                 top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]


            results = []
            for idx in top_k_indices:
                doc = self.documents[idx]
                score = float(similarities[idx]) # Make sure score is standard float
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "score": score
                })
            return results # Added missing return
            
        except Exception as e:
            logger.error(f"Error during RAG retrieval: {e}", exc_info=True)
            return []

    def get_status(self) -> Dict[str, Any]:
        if not self._is_loaded:
            exists = self.index_path.is_dir() and \
                     (self.index_path / self.INDEX_CONFIG_FILE).exists() and \
                     (self.index_path / self.DOCUMENTS_FILE).exists() and \
                     (self.index_path / self.EMBEDDINGS_FILE).exists()
            if exists:
                try:
                    with open(self.index_path / self.INDEX_CONFIG_FILE, 'r') as f:
                        cfg = json.load(f)
                    return {
                        "status": "exists_not_loaded",
                        "index_path": str(self.index_path),
                        "num_documents": cfg.get("num_documents"),
                        "embedding_model_id": cfg.get("embedding_model_id")
                    }
                except Exception as e:
                    logger.warning(f"Could not read config for existing but unloaded index: {e}")
                    return {"status": "exists_corrupt_config", "index_path": str(self.index_path)}
            else:
                return {"status": "does_not_exist", "index_path": str(self.index_path)}
        
        # If loaded
        return {
            "status": "loaded",
            "index_path": str(self.index_path),
            "num_documents": len(self.documents) if self.documents else self.config.get("num_documents",0),
            "embedding_model_id": self.config.get("embedding_model_id", "N/A") if self.config else "N/A",
            "embedding_dimension": self.embeddings.shape if self.embeddings is not None else (self.config.get("embedding_dimension", "N/A") if self.config else "N/A"),
        }

# --- Placeholder Model Classes (for completeness) ---

class ModernBERTClassifier:
    """I/O Validator using transformer models."""
    DEFAULT_MODEL_ID = "microsoft/deberta-v3-base"
    
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, **kwargs):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.is_setup = False
        logger.info(f"ModernBERTClassifier initialized with model: {model_id}")
    
    def setup(self):
        """Load model and tokenizer from Hugging Face."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        try:
            logger.info(f"Loading ModernBERT model: {self.model_id}")
            current_hf_token = os.environ.get("HF_TOKEN")
            logger.debug(f"ModernBERTClassifier.setup: Attempting to load tokenizer for model_id='{self.model_id}'.")
            logger.debug(f"ModernBERTClassifier.setup: HF_TOKEN from env: {'SET' if current_hf_token else 'NOT SET'}")
            logger.debug(f"ModernBERTClassifier.setup: Explicit token arg being passed: {current_hf_token}")
            logger.debug(f"ModernBERTClassifier.setup: use_fast=False")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False, token=os.environ.get("HF_TOKEN"))
            logger.debug(f"ModernBERTClassifier.setup: Attempting to load model for model_id='{self.model_id}'.")
            logger.debug(f"ModernBERTClassifier.setup: HF_TOKEN from env (for model): {'SET' if current_hf_token else 'NOT SET'}")
            logger.debug(f"ModernBERTClassifier.setup: Explicit token arg for model (if passed): {current_hf_token}")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, token=os.environ.get("HF_TOKEN"))
            self.is_setup = True
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True) # Added exc_info
            raise
        return self
    
    def classify_input_output_pair(self, input_text: str, output_text: str) -> Dict[str, Any]:
        """Classify input-output pair using transformer model."""
        if not self.is_setup:
            self.setup()

        try:
            # Tokenize input-output pair
            inputs = self.tokenizer(
                input_text,
                output_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length"
            )
            
            # Get model predictions
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = logits.softmax(dim=-1).detach().numpy()
            
            # Get the index of the predicted class
            pred_idx_val = probabilities.argmax()
            if isinstance(pred_idx_val, np.ndarray): # Ensure we have a scalar value before conversion
                prediction = int(pred_idx_val.item()) if pred_idx_val.size == 1 else int(pred_idx_val.reshape(-1).item())
            else:
                prediction = int(pred_idx_val)

            # Get the confidence score (max probability)
            conf_score_val = probabilities.max()
            if isinstance(conf_score_val, np.ndarray): # Ensure we have a scalar value
                confidence = float(conf_score_val.item()) if conf_score_val.size == 1 else float(conf_score_val.reshape(-1).item())
            else:
                confidence = float(conf_score_val)
            
            prob_positive_value = 0.0
            # CORRECTED_MODERNBERT_PROBABILITIES_HANDLING
            if probabilities.ndim == 2 and probabilities.shape == 1:
                if probabilities.shape == 2: # Standard binary classification (logits for class 0 and class 1)
                    prob_positive_scalar = probabilities 
                elif probabilities.shape == 1: # Single score output (often regression or already a probability)
                    prob_positive_scalar = probabilities
                elif probabilities.shape > 2: # Multi-class (more than 2 classes)
                    logger.warning(f"ModernBERT: Multi-class output ({probabilities.shape} classes). 'probability_positive' taken from class 1 score if available, else 0.")
                    prob_positive_scalar = probabilities if probabilities.shape > 1 else 0.0
                else: # Should not happen if shape >= 1
                    logger.warning(f"ModernBERT: Unexpected probabilities shape {probabilities.shape}. Setting 'probability_positive' to 0.0.")
                    prob_positive_scalar = 0.0
            else:
                logger.warning(f"ModernBERT: Unexpected probabilities shape {probabilities.shape}. Cannot determine 'probability_positive'. Setting to 0.0.")
                prob_positive_scalar = 0.0 # Fallback

            if isinstance(prob_positive_scalar, np.ndarray):
                prob_positive_value = float(prob_positive_scalar.item()) if prob_positive_scalar.size == 1 else float(prob_positive_scalar.reshape(-1).item())
            else:
                prob_positive_value = float(prob_positive_scalar)

            return {
                "prediction": prediction,
                "probability_positive": prob_positive_value,
                "confidence": confidence,
                "details": f"Model: {self.model_id} | Input length: {len(input_text)} | Output length: {len(output_text)}"
            }
        except Exception as e:
            logger.error(f"Classification error: {e}", exc_info=True) # Added exc_info
            return {
                "prediction": 0,
                "probability_positive": 0.0,
                "confidence": 0.0,
                "details": f"Error during classification: {str(e)}"
            }
    
    @classmethod
    def load(cls, model_dir: str, **kwargs):
        """Load a fine-tuned model from directory."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        try:
            logger.info(f"Loading fine-tuned model from: {model_dir}")
            instance = cls(model_dir, **kwargs)
            current_hf_token = os.environ.get("HF_TOKEN") # Use env token
            instance.tokenizer = AutoTokenizer.from_pretrained(model_dir, token=os.environ.get("HF_TOKEN"))
            instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir, token=os.environ.get("HF_TOKEN"))
            instance.is_setup = True
            logger.info("Fine-tuned model loaded successfully")
            return instance
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}", exc_info=True) # Added exc_info
            raise

class ColBERTReranker:
    """ColBERT-based sensitivity classifier using MaxSim technique."""
    DEFAULT_MODEL_ID = "distilbert-base-uncased"
    
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, reference_examples: Optional[Dict[str, List[str]]] = None):
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        logger.debug(f"ColBERTReranker.__init__: Attempting to load tokenizer for model_id='{model_id}'.")
        current_hf_token = os.environ.get("HF_TOKEN") # Use env token
        logger.debug(f"ColBERTReranker.__init__: HF_TOKEN from env: {'SET' if current_hf_token else 'NOT SET'}")
        logger.debug(f"ColBERTReranker.__init__: Explicit token arg being passed: {current_hf_token}")
        logger.debug(f"ColBERTReranker.__init__: use_fast=True")
        
        # Simplified model loading for the new default (DistilBERT)
        try:
            self.model_id = model_id # Will be "distilbert-base-uncased" if default is used
            logger.info(f"Attempting to load model: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, use_fast=True, token=os.environ.get("HF_TOKEN")
            ) # trust_remote_code=False or omitted for standard models like DistilBERT
            self.model = AutoModel.from_pretrained(
                self.model_id, token=os.environ.get("HF_TOKEN")
            )
            logger.info(f"Successfully loaded model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to load ColBERT model '{self.model_id}': {e}", exc_info=True) # Added exc_info
            # If even the standard DistilBERT fails, the class instance will be unusable.
            raise RuntimeError(f"ColBERTReranker failed to load model ('{self.model_id}').") from e

        self.reference_embeddings = {}
        self.reference_examples = reference_examples or {
            "Class 1: PII": ["SSN: 123-45-6789", "Credit card: 4111-1111-1111-1111"],
            "Class 2: Confidential": ["Project codename: Phoenix", "Internal memo"],
            "Class 3: Internal": ["Meeting notes", "Draft document"],
            "Class 4: Public": ["Press release", "Public blog post"]
        }
        
        for class_name, examples in self.reference_examples.items():
            self.reference_embeddings[class_name] = [
                self._get_text_embedding(example) for example in examples
            ]
        
        logger.info(f"ColBERTReranker initialized with model: {self.model_id}")
    
    def _get_text_embedding(self, text: str) -> "torch.Tensor":
        """Get token-level embeddings for a text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0)
    
    def _maxsim_score(self, query_embed: "torch.Tensor", doc_embed: "torch.Tensor") -> float:
        """Compute MaxSim score between query and document embeddings."""
        similarities = torch.nn.functional.cosine_similarity(
            query_embed.unsqueeze(1),  # [Q_tokens, 1, EmbDim]
            doc_embed.unsqueeze(0),    # [1, D_tokens, EmbDim] 
            dim=-1                     # Result: [Q_tokens, D_tokens]
        )
        max_sims = similarities.max(dim=-1).values  # Max similarity for each query token: [Q_tokens]
        return max_sims.mean().item() # Average of these max similarities
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """Classify text sensitivity using ColBERT-style MaxSim technique."""
        try:
            text_embed = self._get_text_embedding(text)
            class_scores = {}
            
            for class_name, ref_embeds in self.reference_embeddings.items():
                scores = [self._maxsim_score(text_embed, ref_embed) for ref_embed in ref_embeds]
                class_scores[class_name] = max(scores) if scores else 0.0 # Handle empty scores
            
            predicted_class = max(class_scores, key=class_scores.get) if class_scores else "Class 5: Unknown"
            max_score = class_scores.get(predicted_class, 0.0)
            
            return {
                "predicted_class": predicted_class,
                "class_scores": class_scores,
                "confidence": max_score,
                "details": f"ColBERT classification using {self.model_id}"
            }
        except Exception as e:
            logger.error(f"Error in sensitivity classification: {e}", exc_info=True) # Added exc_info
            return {
                "predicted_class": "Class 5: Unknown",
                "class_scores": {},
                "confidence": 0.0,
                "details": f"Classification error: {str(e)}"
            }
    
    @classmethod
    def load(cls, model_dir: str, reference_examples_path: Optional[str] = None):
        """Load a ColBERT model with optional reference examples."""
        reference_examples = None
        if reference_examples_path:
            try:
                with open(reference_examples_path, "r") as f:
                    reference_examples = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load reference examples: {e}")
        
        return cls(model_dir, reference_examples)

class VisionLanguageProcessor:
    """Vision-Language Model processor for image and video analysis."""
    DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf" # Example, ensure this or chosen model works with pipeline
    
    def __init__(self, model_id_or_dir: str = DEFAULT_MODEL_ID, **kwargs): # Renamed model_id to model_id_or_dir
        from transformers import pipeline, AutoProcessor # Keep here for init-time check if needed
         
        self.model_id = model_id_or_dir # Use the new name
        self.processor = None # Initialize to None
        self.model = None # Initialize to None
        self.device = None 
        self.is_setup = False
        logger.info(f"VisionLanguageProcessor initialized with model: {self.model_id}")

    def setup(self):
        """Load the VLM model and processor."""
        from transformers import pipeline, AutoProcessor
        import torch
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        current_hf_token = os.environ.get("HF_TOKEN") # Use env token
        
        try:
            # For many VLM pipelines, processor is loaded automatically if model_id is a repo string.
            # However, explicitly loading it can give more control or handle local models better.
            # If pipeline can infer processor from model_id, this might be redundant but harmless.
            try:
                logger.debug(f"VisionLanguageProcessor.setup: Loading AutoProcessor for {self.model_id}")
                # Rely on HF_TOKEN in environment rather than passing token explicitly
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            except Exception as e_proc:
                logger.warning(f"Could not explicitly load AutoProcessor for {self.model_id}, pipeline might load it internally. Error: {e_proc}")
                # Processor might still be loaded by pipeline if it's a standard HF model ID.

            # Diagnostic step: try to download the preprocessor config explicitly for LLaVA
            if self.model_id and "llava" in self.model_id.lower():
                try:
                    logger.debug(f"Attempting to manually download preprocessor_config.json for {self.model_id}")
                    # Rely on HF_TOKEN in environment rather than passing token explicitly
                    config_path = hf_hub_download(
                        repo_id=self.model_id,
                        filename="preprocessor_config.json"
                    )
                    logger.info(f"Successfully downloaded preprocessor_config.json for LLaVA to {config_path}")
                except Exception as e_cfg_download:
                    logger.error(f"Manual download of LLaVA preprocessor_config.json failed: {e_cfg_download}", exc_info=True)
            
            logger.debug(f"VisionLanguageProcessor.setup: Attempting to load pipeline for model='{self.model_id}'")
            logger.debug(f"VisionLanguageProcessor.setup: HF_TOKEN from env: {'SET' if current_hf_token else 'NOT SET'}")
            
            # For LLaVA style models, the pipeline often needs model and tokenizer/processor specified if not standard HF.
            # The "image-to-text" task is generic; specific models might need different task names or parameters.
            # Rely on HF_TOKEN in environment rather than passing token explicitly
            self.model = pipeline(
                "image-to-text", # Check if this task is appropriate for the model
                model=self.model_id,
                # processor=self.processor, # Often not needed if model_id is a repo name; pipeline infers. Only pass if required.
                                           # For local models, model and tokenizer/processor might need to be passed.
                device=0 if self.device == "cuda" else -1, # pipeline device mapping
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True # Often needed for VLMs
            )
            self.is_setup = True
            logger.info(f"VLM model for API item processing loaded successfully using pipeline for {self.model_id}")
        except Exception as e:
            logger.error(f"Error loading VLM model via pipeline: {e}", exc_info=True)
            # Try to load processor and model manually if pipeline fails (more complex)
            # This part is model-specific and harder to generalize in a placeholder.
            # For a real implementation, one would add specific loading logic for chosen models.
            logger.warning("Pipeline loading failed. Manual VLM loading not implemented in this placeholder.")
            raise # Re-raise to indicate setup failure
        return self
    
    def _load_image(self, image_source: Union[str, bytes, Path, "Image.Image"]) -> "Image.Image": # Extended types
        """Load image from file path, URL, bytes, or PIL Image."""
        from PIL import Image
        import requests
        from io import BytesIO
        
        if isinstance(image_source, Image.Image): # Already a PIL Image
            return image_source
        elif isinstance(image_source, bytes):
            return Image.open(BytesIO(image_source))
        elif isinstance(image_source, Path): # Handle Path object
            return Image.open(image_source)
        elif isinstance(image_source, str) and image_source.startswith(("http://", "https://")):
            response = requests.get(image_source)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        elif isinstance(image_source, str): # Assume local file path string
             return Image.open(image_source)
        else:
            raise TypeError(f"Unsupported image_source type: {type(image_source)}")
    
    def describe_image(self, image_source: Union[str, bytes, Path, "Image.Image"], prompt: Optional[str] = None) -> Dict[str, Any]:
        """Analyze an image using VLM with optional custom prompt."""
        from PIL import Image # Keep for type hint if Image.Image not used directly
        import torch
        
        if not self.is_setup:
            self.setup()
        if not self.model: # Check if model was loaded by setup
            raise RuntimeError("VLM model (pipeline) not available for image description.")

        
        try:
            image = self._load_image(image_source)
            
            final_prompt = prompt or "Describe this image in detail, focusing on any text, objects, and activities."
            
            if image.mode == 'RGBA': # Ensure RGB format
                image = image.convert('RGB')

            # Image-to-text pipeline typically takes image and optional prompt kwargs
            # Example for LLaVA-like models often involves constructing a specific prompt format
            # For a generic pipeline, it might be simpler:
            if self.model.tokenizer and hasattr(self.model.tokenizer, "chat_template") and self.model.tokenizer.chat_template:
                 # More complex models might need specific chat templating
                 # This is a placeholder for such logic. LLaVA might need USER: <image>\n{prompt}\nASSISTANT:
                 # For simplicity, we'll try with the generic prompt first.
                 logger.debug("VLM has chat template, advanced prompting might be needed for optimal results.")

            # The pipeline call itself
            # Max_new_tokens might be passed inside generate_kwargs for some pipelines
            outputs = self.model(image, prompt=final_prompt, generate_kwargs={"max_new_tokens": 250})
            
            description = ""
            # CORRECTED_VLP_OUTPUT_PARSING
            if outputs and isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs, dict) and "generated_text" in outputs:
                description = outputs["generated_text"]
                # Sometimes the prompt itself is part of generated_text, needs stripping
                if final_prompt in description:
                     description = description.split(final_prompt,1)[-1].strip() # Get text after prompt
                # LLaVA specific: often includes "ASSISTANT: " prefix
                if description.startswith("ASSISTANT: "):
                    description = description.replace("ASSISTANT: ", "", 1).strip()
            else: 
                logger.warning(f"Unexpected VLM output structure for image description: {str(outputs)[:200]}")
                description = str(outputs) # Fallback to string representation if parsing fails

            return {
                "description": description.strip(), 
                "analysis": self._analyze_description(description),
                "prompt_used": final_prompt,
                "model": self.model_id,
                "details": f"Processed image of size {image.size} with {self.model_id}"
            }
        except Exception as e:
            logger.error(f"Image processing error: {e}", exc_info=True)
            return {
                "description": "Error processing image",
                "analysis": {},
                "details": f"Error: {str(e)}"
            }
    
    def _analyze_description(self, description: str) -> Dict[str, Any]:
        """Perform basic analysis on the generated description."""
        analysis = {
            "contains_text_guess": any(c.isalpha() for c in description), 
            "word_count": len(description.split()),
        }
        return analysis
    
    def describe_video_frames(self, video_source: Union[str, Path], prompt: Optional[str] = None,
                            frame_interval: int = 5) -> Dict[str, Any]:
        """Analyze video by sampling frames at given interval (seconds)."""
        import cv2
        from PIL import Image
        import tempfile
        
        if not self.is_setup:
            self.setup()
        
        video_path_str = ""
        temp_file_created = False
        try:
            if isinstance(video_source, Path):
                video_path_str = str(video_source)
            elif isinstance(video_source, str) and video_source.startswith(("http://", "https://")):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file: 
                    response = requests.get(video_source, stream=True) 
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    video_path_str = temp_file.name
                    temp_file_created = True
            elif isinstance(video_source, str): 
                video_path_str = video_source
            else:
                raise TypeError(f"Unsupported video_source type: {type(video_source)}")

            if not Path(video_path_str).exists():
                 raise FileNotFoundError(f"Video file not found at {video_path_str}")

            cap = cv2.VideoCapture(video_path_str)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path_str}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0: fps = 30 
            frame_step = int(fps * frame_interval)
            if frame_step <= 0 : frame_step = int(fps) 
            
            frame_descriptions = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_step == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    frame_desc_result = self.describe_image(pil_image, prompt) 
                    frame_desc_result["timestamp_seconds"] = round(frame_count / fps, 2)
                    frame_descriptions.append(frame_desc_result)
                
                frame_count += 1
            
            cap.release()
            
            summary = self._summarize_frame_descriptions(frame_descriptions)
            
            return {
                "frame_descriptions": frame_descriptions,
                "summary": summary,
                "total_frames_read": frame_count,
                "frames_analyzed": len(frame_descriptions),
                "details": f"Analyzed {len(frame_descriptions)} frames from {frame_count} total frames read."
            }
        except Exception as e:
            logger.error(f"Video processing error: {e}", exc_info=True)
            return {
                "frame_descriptions": [],
                "summary": "Error processing video",
                "details": f"Error: {str(e)}"
            }
        finally:
            if temp_file_created and video_path_str and Path(video_path_str).exists():
                try:
                    Path(video_path_str).unlink()
                    logger.debug(f"Temporary video file {video_path_str} deleted.")
                except Exception as e_del:
                    logger.warning(f"Could not delete temporary video file {video_path_str}: {e_del}")

    
    def _summarize_frame_descriptions(self, frame_descs: List[Dict]) -> str:
        """Generate a summary from multiple frame descriptions."""
        if not frame_descs: return "No frames analyzed."
        
        unique_descriptions_texts = set()
        for desc_item in frame_descs:
            if isinstance(desc_item, dict) and isinstance(desc_item.get("description"), str):
                unique_descriptions_texts.add(desc_item["description"])
        
        if not unique_descriptions_texts: return "No valid descriptions found in analyzed frames."
        
        return "Video content highlights: " + "; ".join(list(unique_descriptions_texts)[:5]) 

    @classmethod
    def load(cls, model_dir: str, **kwargs):
        """Load a VLM from a local directory."""
        return cls(model_id_or_dir=model_dir, **kwargs) 


# --- Enhanced Classification API with Complete Policy Logic ---

class ClassificationAPI:
    """Enhanced Classification API with complete policy validation."""
    
    def __init__(self, modernbert_model_dir: Optional[str],
                 host: str, port: int,
                 policy_config_path: Optional[str] = "policy_config.json",
                 vlm_model_id_or_dir: Optional[str] = None, 
                 global_rag_retriever_index_path: Optional[str] = None):
        
        self.modernbert_model_dir = modernbert_model_dir
        self.host = host
        self.port = port
        self.policy_config_path = policy_config_path
        self.vlm_for_item_processing_id_or_dir = vlm_model_id_or_dir 
        
        self.api_policy_config: Dict[str, Any] = {}
        self.modernbert_classifier: Optional[ModernBERTClassifier] = None
        self.colbert_reranker: Optional[ColBERTReranker] = None
        self.vision_language_processor: Optional[VisionLanguageProcessor] = None 
        
        self.global_rag_retriever_index_path = global_rag_retriever_index_path
        self.documentation_rag_retriever: Optional[RAGRetriever] = None 
        
        self.app = Flask(__name__)
        CORS(self.app) 
        self.request_count = 0

    def _handle_request_policy(self, payload: Dict, policy_rules: Dict, files: Optional[Any], response_data: Dict) -> List[str]:
        """
        Complete implementation of policy validation logic.
        Returns a list of violation reason strings. Populates response_data.
        """
        violations = []
        
        try:
            violations.extend(self._handle_legacy_policy_validation(payload, policy_rules, response_data))
            
            if "input_items" in payload: 
                violations.extend(self._handle_multimodal_policy_validation(payload, policy_rules, files, response_data))
            
            violations.extend(self._handle_additional_policy_checks(payload, policy_rules, response_data))
            
        except Exception as e:
            logger.error(f"Critical error in policy processing: {e}", exc_info=True)
            violations.append(f"Policy processing error: {str(e)}")
        
        return violations

    def _handle_legacy_policy_validation(self, payload: Dict, policy: Dict, response_data: Dict) -> List[str]:
        """Handle legacy format policy validation (ModernBERT, ColBERT on input/output text)."""
        violations = []
        
        input_text = payload.get("input_text", "")
        output_text = payload.get("output_text", "")
        
        if policy.get("modernbert_io_validation", False) and self.modernbert_classifier:
            if input_text or output_text: 
                try:
                    io_result = self.modernbert_classifier.classify_input_output_pair(input_text, output_text)
                    response_data["modernbert_io_validation"] = io_result
                    
                    if io_result.get("prediction") == 0: 
                        prob_positive = io_result.get("probability_positive", 0.0)
                        violations.append(f"ModernBERT I/O validation failed (validity score: {prob_positive:.2f}).")
                        
                except Exception as e:
                    logger.error(f"Error in ModernBERT I/O validation: {e}", exc_info=True)
                    violations.append("I/O validation processing error (ModernBERT).")
                    response_data["modernbert_io_validation"] = {"error": str(e)}
        
        if policy.get("colbert_input_sensitivity", False) and self.colbert_reranker and input_text:
            try:
                input_sensitivity = self.colbert_reranker.classify_text(input_text)
                response_data["colbert_input_sensitivity"] = input_sensitivity
                predicted_class = input_sensitivity.get("predicted_class")
                
                disallowed_classes = policy.get("disallowed_colbert_input_classes", [])
                if predicted_class in disallowed_classes:
                    violations.append(f"Input text classified as disallowed sensitivity: '{predicted_class}'.")
                
                allowed_classes = policy.get("allowed_colbert_input_classes", [])
                if allowed_classes and predicted_class not in allowed_classes:
                     violations.append(f"Input text sensitivity class '{predicted_class}' not in allowed list.")
                    
            except Exception as e:
                logger.error(f"Error in ColBERT input sensitivity check: {e}", exc_info=True)
                violations.append("Input sensitivity check processing error (ColBERT).")
                response_data["colbert_input_sensitivity"] = {"error": str(e)}

        if policy.get("colbert_output_sensitivity", False) and self.colbert_reranker and output_text:
            try:
                output_sensitivity = self.colbert_reranker.classify_text(output_text)
                response_data["colbert_output_sensitivity"] = output_sensitivity
                predicted_class = output_sensitivity.get("predicted_class")

                disallowed_classes = policy.get("disallowed_colbert_output_classes", [])
                if predicted_class in disallowed_classes:
                    violations.append(f"Output text classified as disallowed sensitivity: '{predicted_class}'.")

                allowed_classes = policy.get("allowed_colbert_output_classes", [])
                if allowed_classes and predicted_class not in allowed_classes:
                    violations.append(f"Output text sensitivity class '{predicted_class}' not in allowed list.")

            except Exception as e:
                logger.error(f"Error in ColBERT output sensitivity check: {e}", exc_info=True)
                violations.append("Output sensitivity check processing error (ColBERT).")
                response_data["colbert_output_sensitivity"] = {"error": str(e)}
        
        return violations

    def _handle_multimodal_policy_validation(self, payload: Dict, policy: Dict, files: Optional[Any], response_data: Dict) -> List[str]:
        """Handle multimodal format policy validation (VLM on items, derived text checks)."""
        violations = []
        
        input_items = payload.get("input_items", [])
        if not isinstance(input_items, list):
            violations.append("Invalid 'input_items' format: expected a list.")
            return violations

        item_processing_rules = policy.get("item_processing_rules", [])
        if not item_processing_rules: 
            return violations

        response_data["item_processing_results"] = {} 
        
        for item_idx, item in enumerate(input_items):
            if not isinstance(item, dict):
                violations.append(f"Item at index {item_idx} is not a valid object.")
                continue

            item_id = item.get("id", f"item_{item_idx}")
            item_type = item.get("type", "unknown_type")
            filename_in_form = item.get("filename_in_form") 
            
            item_result_data = {"item_id": item_id, "item_type": item_type, "violations": []} 

            try:
                matching_rule = next((rule for rule in item_processing_rules if rule.get("item_type") == item_type), None)
                
                if not matching_rule:
                    if policy.get("strict_item_type_matching", False): 
                        item_result_data["violations"].append(f"No processing rule defined for item type: '{item_type}'.")
                    continue 

                vlm_config = matching_rule.get("vlm_processing", {})
                if vlm_config.get("required", False) and self.vision_language_processor:
                    file_obj_for_vlm = None
                    if not filename_in_form:
                        item_result_data["violations"].append(f"VLM processing required for item '{item_id}', but 'filename_in_form' is missing.")
                    elif not files or filename_in_form not in files:
                        item_result_data["violations"].append(f"Required file '{filename_in_form}' for item '{item_id}' not found in form data.")
                    else:
                        file_obj_for_vlm = files[filename_in_form] # This is a FileStorage object
                    
                    vlm_output_text = ""
                    if file_obj_for_vlm: # Proceed only if file object is available
                        try:
                            image_bytes = file_obj_for_vlm.read() 
                            file_obj_for_vlm.seek(0) 
                            
                            if item_type in ["image", "screenshot"]:
                                vlm_result = self.vision_language_processor.describe_image(
                                    image_source=image_bytes, 
                                    prompt=vlm_config.get("prompt")
                                )
                            else:
                                item_result_data["violations"].append(f"VLM processing not supported for item type: '{item_type}'.")
                                vlm_result = None

                            if vlm_result:
                                item_result_data["vlm_analysis"] = vlm_result
                                vlm_output_text = vlm_result.get("description", "")
                        except Exception as e_vlm:
                            logger.error(f"Error during VLM processing for item '{item_id}': {e_vlm}", exc_info=True)
                            item_result_data["violations"].append(f"VLM processing failed for item '{item_id}'.")
                            item_result_data["vlm_analysis"] = {"error": str(e_vlm)}
                        
                        derived_checks_config = matching_rule.get("derived_text_checks", {})
                        if derived_checks_config and vlm_output_text:
                            if derived_checks_config.get("colbert_sensitivity", False) and self.colbert_reranker:
                                try:
                                    derived_sensitivity = self.colbert_reranker.classify_text(vlm_output_text)
                                    item_result_data["derived_text_sensitivity"] = derived_sensitivity
                                    derived_class = derived_sensitivity.get("predicted_class")
                                    disallowed_derived = derived_checks_config.get("disallowed_classes", [])
                                    if derived_class in disallowed_derived:
                                        item_result_data["violations"].append(f"VLM-derived text for item '{item_id}' classified as disallowed sensitivity: '{derived_class}'.")
                                except Exception as e_colbert_derived:
                                    logger.error(f"ColBERT on derived text for '{item_id}' failed: {e_colbert_derived}", exc_info=True)
                                    item_result_data["violations"].append(f"Sensitivity check on VLM output for '{item_id}' failed.")
                                    item_result_data["derived_text_sensitivity"] = {"error": str(e_colbert_derived)}
                            
                            blocked_keywords = derived_checks_config.get("blocked_keywords", [])
                            if blocked_keywords:
                                found_kws = [kw for kw in blocked_keywords if kw.lower() in vlm_output_text.lower()]
                                if found_kws:
                                    item_result_data["violations"].append(f"VLM-derived text for item '{item_id}' contains blocked keywords: {', '.join(found_kws)}.")
                                    item_result_data["blocked_keywords_found"] = found_kws
                
            except Exception as e_item:
                logger.error(f"Error processing item '{item_id}': {e_item}", exc_info=True)
                item_result_data["violations"].append(f"General error processing item '{item_id}'.")
            
            response_data["item_processing_results"][item_id] = item_result_data
            violations.extend(item_result_data["violations"]) 
        
        return violations

    def _handle_additional_policy_checks(self, payload: Dict, policy: Dict, response_data: Dict) -> List[str]:
        """Handle additional policy-specific checks (custom rules, rate limiting placeholder)."""
        violations = []
        
        try:
            custom_rules = policy.get("custom_validation_rules", [])
            if not isinstance(custom_rules, list):
                violations.append("Invalid 'custom_validation_rules' format in policy.")
                return violations 

            for rule_idx, rule in enumerate(custom_rules):
                if not isinstance(rule, dict):
                    violations.append(f"Custom rule at index {rule_idx} is not a valid object.")
                    continue

                rule_type = rule.get("type")
                
                if rule_type == "text_length_limit":
                    max_length = rule.get("max_length")
                    text_fields = rule.get("text_fields", []) 
                    if not isinstance(max_length, int) or not isinstance(text_fields, list):
                         violations.append(f"Invalid 'text_length_limit' rule config at index {rule_idx}.")
                         continue
                    for field in text_fields:
                        text_value = payload.get(field, "")
                        if isinstance(text_value, str) and len(text_value) > max_length:
                            violations.append(f"Field '{field}' (length {len(text_value)}) exceeds maximum length of {max_length} characters.")
                
                elif rule_type == "required_fields":
                    required_field_names = rule.get("fields", [])
                    if not isinstance(required_field_names, list):
                        violations.append(f"Invalid 'required_fields' rule config at index {rule_idx}.")
                        continue
                    for field_name in required_field_names:
                        if field_name not in payload or not payload[field_name]: 
                            violations.append(f"Required field '{field_name}' is missing or empty.")
                
                elif rule_type == "format_validation":
                    field_to_validate = rule.get("field")
                    regex_pattern = rule.get("pattern")
                    if not field_to_validate or not regex_pattern:
                        violations.append(f"Invalid 'format_validation' rule config at index {rule_idx}.")
                        continue
                    if field_to_validate in payload:
                        try:
                            if not re.match(regex_pattern, str(payload[field_to_validate])):
                                violations.append(f"Field '{field_to_validate}' does not match required format pattern.")
                        except re.error as re_e:
                             violations.append(f"Invalid regex pattern in 'format_validation' for field '{field_to_validate}': {re_e}")
            
            if policy.get("rate_limiting", {}).get("enabled", False):
                logger.debug(f"Rate limiting check (conceptual) for policy with max_requests: {policy['rate_limiting'].get('max_requests_per_minute')}")
            
        except Exception as e:
            logger.error(f"Error during additional policy checks: {e}", exc_info=True)
            violations.append(f"Additional policy validation error: {str(e)}")
        
        return violations

    def _generate_help_queries(self, violation_reasons: List[str], policy: Dict[str, Any]) -> List[str]:
        """Generate help queries based on violation reasons for RAG."""
        help_queries: Set[str] = set() 
        
        common_terms_map = {
            "pii": ["handling PII", "sensitive data policy", "data privacy"],
            "sensitive": ["data sensitivity levels", "confidential information rules"],
            "modernbert i/o": ["input output validation", "API request format", "model validation"],
            "colbert": ["content sensitivity classification", "text analysis policy"],
            "vlm": ["image content policy", "video analysis rules", "multimodal validation"],
            "length limit": ["text length restrictions", "payload size limits"],
            "required field": ["mandatory fields", "API data requirements"],
            "format validation": ["data format rules", "pattern matching requirements"]
        }

        for violation in violation_reasons:
            violation_lower = violation.lower()
            found_term_query = False
            for term, queries in common_terms_map.items():
                if term in violation_lower:
                    help_queries.update(queries)
                    found_term_query = True
            if not found_term_query: 
                keywords = re.findall(r'\b[a-zA-Z]{4,}\b', violation_lower) 
                if keywords:
                    help_queries.add("how to fix " + " ".join(list(set(keywords))[:3])) 
                else:
                    help_queries.add(f"guidelines for '{violation_lower[:30]}...'")


        if policy.get("description"):
            help_queries.add(f"understanding policy: {policy['description'][:50]}")
        
        return list(help_queries)

    def _format_doc_suggestions(self, retrieved_docs: List[Dict[str, Any]], 
                               violation_reasons: List[str]) -> List[Dict[str, Any]]:
        """Format retrieved documentation as helpful suggestions."""
        suggestions = []
        if not retrieved_docs: return suggestions

        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            
            suggestion_title = self._generate_suggestion_title(doc, metadata)
            
            suggestion = {
                "title": suggestion_title,
                "content_preview": doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"],
                "full_content_available": True, 
                "document_id": doc.get("id"),
                "relevance_score": doc.get("score", 0.0),
                "source_url": metadata.get("source_url", "N/A"),
                "primary_section": metadata.get("h1_section", "Documentation"),
                "secondary_section": metadata.get("h2_section", ""),
                "chunk_type": metadata.get("chunk_type", "content"),
                "tags": {
                    "is_code_example": metadata.get("is_code_block", False) or metadata.get("chunk_type") == "code",
                    "contains_commands": metadata.get("contains_commands", False),
                    "vlm_processed_doc": metadata.get("vlm_processed", False)
                }
            }
            suggestions.append(suggestion)
        
        return suggestions

    def _generate_suggestion_title(self, doc: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Generate a helpful title for a documentation suggestion."""
        if metadata.get("h2_section"):
            return f"{metadata.get('h1_section', 'Help')}: {metadata['h2_section']}"
        elif metadata.get("h1_section"):
            return metadata["h1_section"]
        
        if metadata.get("is_code_block", False) or metadata.get("chunk_type") == "code":
            return "Relevant Code Example/Reference"
        if metadata.get("contains_commands", False):
            return "Relevant Command Reference"
        if doc.get("id"):
            return f"Documentation Snippet: {doc['id']}"
        
        return "General Documentation"

    def _add_documentation_assistance(self, response_data: Dict[str, Any], 
                                    policy: Dict[str, Any], violation_reasons: List[str]):
        """Add documentation assistance to the response if violations occurred and configured."""
        response_data["documentation_assistance_attempted"] = False
        try:
            docs_assist_config = policy.get("documentation_assistance", {})
            if not docs_assist_config.get("enabled", False):
                return 
            
            response_data["documentation_assistance_attempted"] = True
            docs_index_path_str = docs_assist_config.get("index_path")
            if not docs_index_path_str:
                logger.warning("Documentation assistance enabled but no RAG index_path configured in policy.")
                response_data["documentation_suggestions"] = {"error": "Documentation RAG index path not configured."}
                return
            
            policy_docs_retriever = self._get_cached_rag_retriever(docs_index_path_str)
            
            if not policy_docs_retriever or not policy_docs_retriever._is_loaded:
                logger.warning(f"Could not load documentation RAG index from {docs_index_path_str} for policy assistance.")
                response_data["documentation_suggestions"] = {"error": f"Documentation RAG index at '{docs_index_path_str}' not available or failed to load."}
                return
            
            help_queries = self._generate_help_queries(violation_reasons, policy)
            if not help_queries:
                response_data["documentation_suggestions"] = {"message": "No specific help queries generated for violations."}
                return
            
            all_retrieved_docs_map: Dict[str, Dict[str, Any]] = {} 
            max_suggestions_per_query = docs_assist_config.get("max_suggestions_per_query", 2)
            min_threshold = docs_assist_config.get("min_similarity_threshold", 0.1)
            
            for query_idx, query_text in enumerate(help_queries[:docs_assist_config.get("max_help_queries_to_use", 3)]):
                retrieved_for_query = policy_docs_retriever.retrieve(query_text, top_k=max_suggestions_per_query)
                for doc in retrieved_for_query:
                    if doc.get("score", 0.0) >= min_threshold:
                        doc_id = str(doc.get("id", ""))
                        if doc_id not in all_retrieved_docs_map or doc["score"] > all_retrieved_docs_map[doc_id]["score"]:
                            all_retrieved_docs_map[doc_id] = doc
            
            if not all_retrieved_docs_map:
                response_data["documentation_suggestions"] = {
                    "message": "No relevant documentation found for these violations.",
                    "queries_used": help_queries[:docs_assist_config.get("max_help_queries_to_use", 3)]
                }
                return
            
            sorted_unique_docs = sorted(all_retrieved_docs_map.values(), key=lambda x: x["score"], reverse=True)
            
            final_suggestions_count = docs_assist_config.get("max_total_suggestions", 5)
            top_docs_for_response = sorted_unique_docs[:final_suggestions_count]
            
            formatted_suggestions = self._format_doc_suggestions(top_docs_for_response, violation_reasons)
            
            response_data["documentation_suggestions"] = {
                "suggestions": formatted_suggestions,
                "help_queries_used": help_queries[:docs_assist_config.get("max_help_queries_to_use", 3)],
                "total_relevant_chunks_found": len(sorted_unique_docs),
                "showing_top_n": len(top_docs_for_response)
            }
            
        except Exception as e:
            logger.error(f"Error adding documentation assistance: {e}", exc_info=True)
            response_data["documentation_suggestions"] = {"error": f"Failed to provide documentation assistance: {str(e)}"}

    def _get_cached_rag_retriever(self, index_path_str: str) -> Optional[RAGRetriever]:
        """Get a cached RAG retriever or load a new one. Caches per index_path_str."""
        global RAG_COMPONENT_CACHE
        normalized_path_key = str(Path(index_path_str).resolve())

        if normalized_path_key in RAG_COMPONENT_CACHE:
            logger.debug(f"Using cached RAGRetriever for {normalized_path_key}")
            return RAG_COMPONENT_CACHE[normalized_path_key]
        
        logger.info(f"Attempting to load RAGRetriever for {normalized_path_key} (not found in cache).")
        try:
            retriever = RAGRetriever(normalized_path_key) 
            if retriever.load_index():
                RAG_COMPONENT_CACHE[normalized_path_key] = retriever
                logger.info(f"Successfully loaded and cached RAGRetriever for {normalized_path_key}.")
                return retriever
            else:
                logger.error(f"Failed to load RAGRetriever for {normalized_path_key}.")
                return None
        except Exception as e:
            logger.error(f"Exception initializing or loading RAGRetriever for {normalized_path_key}: {e}", exc_info=True)
            return None

    def setup(self, **kwargs):
        """Setup the classification API with all components."""
        logger.info("Setting up ClassificationAPI...")
        
        if self.policy_config_path:
            policy_file = Path(self.policy_config_path)
            if policy_file.exists() and policy_file.is_file():
                try:
                    with open(policy_file, 'r', encoding='utf-8') as f:
                        self.api_policy_config = json.load(f)
                    logger.info(f"Loaded API policy configuration from {self.policy_config_path} with {len(self.api_policy_config)} policies.")
                except json.JSONDecodeError as e_json:
                    logger.error(f"Failed to parse policy config JSON from {self.policy_config_path}: {e_json}", exc_info=True)
                except Exception as e:
                    logger.error(f"Failed to load policy config from {self.policy_config_path}: {e}", exc_info=True)
            else:
                 logger.warning(f"Policy config path {self.policy_config_path} does not exist or is not a file. No policies loaded.")
        else:
            logger.warning("No policy_config_path provided. API will operate without defined policies.")

        try:
            if self.modernbert_model_dir: 
                self.modernbert_classifier = ModernBERTClassifier.load(self.modernbert_model_dir)
            else: 
                self.modernbert_classifier = ModernBERTClassifier().setup()
        except Exception as e_mb:
            logger.error(f"Failed to initialize ModernBERT classifier: {e_mb}", exc_info=True)
        
        try:
            self.colbert_reranker = ColBERTReranker() 
        except Exception as e_colbert:
            logger.error(f"Failed to initialize ColBERT reranker: {e_colbert}", exc_info=True)

        try:
            if self.vlm_for_item_processing_id_or_dir: 
                 self.vision_language_processor = VisionLanguageProcessor(model_id_or_dir=self.vlm_for_item_processing_id_or_dir).setup()
            else: 
                 self.vision_language_processor = VisionLanguageProcessor().setup() 
        except Exception as e_vlp:
            logger.error(f"Failed to initialize VisionLanguageProcessor for items: {e_vlp}", exc_info=True)
        
        if self.global_rag_retriever_index_path:
            logger.info(f"Attempting to load global RAG retriever from: {self.global_rag_retriever_index_path}")
            self.documentation_rag_retriever = self._get_cached_rag_retriever(self.global_rag_retriever_index_path)
            if self.documentation_rag_retriever:
                logger.info("Global RAG retriever loaded successfully.")
            else:
                logger.warning("Failed to load global RAG retriever.")
        
        @self.app.route('/service/validate', methods=['POST'])
        def service_validate():
            self.request_count += 1
            request_timestamp = time.time()
            
            is_multipart = request.content_type and 'multipart/form-data' in request.content_type.lower()
            
            payload: Optional[Dict] = None
            files: Optional[Any] = None

            if is_multipart:
                if 'json_payload' not in request.form:
                    return jsonify({"overall_status": "ERROR_BAD_REQUEST", "error_message": "Multipart request missing 'json_payload' form field."}), 400
                try:
                    payload = json.loads(request.form['json_payload'])
                except json.JSONDecodeError as e_json_form:
                    return jsonify({"overall_status": "ERROR_BAD_REQUEST", "error_message": f"Invalid JSON in 'json_payload': {e_json_form}"}), 400
                files = request.files
            else: 
                try:
                    payload = request.get_json()
                    if payload is None: 
                         return jsonify({"overall_status": "ERROR_BAD_REQUEST", "error_message": "Request body must be valid JSON and Content-Type 'application/json'."}), 400
                except Exception as e_json_body: 
                     return jsonify({"overall_status": "ERROR_BAD_REQUEST", "error_message": f"Failed to parse JSON request body: {e_json_body}"}), 400
            
            api_class_name = payload.get("api_class")
            if not api_class_name:
                return jsonify({"overall_status": "ERROR_BAD_REQUEST", "error_message": "'api_class' field is required in JSON payload."}), 400
            
            response_data = {
                "request_id": payload.get("request_id", f"req_{int(request_timestamp)}_{self.request_count}"),
                "api_class_requested": api_class_name,
                "timestamp_utc": request_timestamp,
                "processing_details": {} 
            }
            
            active_policy_rules = self.api_policy_config.get(api_class_name)
            if not active_policy_rules:
                response_data["overall_status"] = "REJECT_INVALID_POLICY"
                response_data["error_message"] = f"Policy definition for API class '{api_class_name}' not found or not loaded."
                return jsonify(response_data), 400 
            
            violation_reasons = self._handle_request_policy(payload, active_policy_rules, files, response_data["processing_details"])
            
            if violation_reasons:
                response_data["violation_reasons"] = violation_reasons
                response_data["overall_status"] = "REJECT_POLICY_VIOLATION"
                self._add_documentation_assistance(response_data, active_policy_rules, violation_reasons)
            else:
                response_data["overall_status"] = "PASS"
            
            response_data["processing_time_seconds"] = round(time.time() - request_timestamp, 3)
            return jsonify(response_data)

        @self.app.route('/modernbert/classify', methods=['POST'])
        def modernbert_classify_endpoint():
            if not self.modernbert_classifier:
                return jsonify({"error": "ModernBERT classifier not available or not setup."}), 503
            
            payload = request.get_json()
            if not payload: return jsonify({"error": "Request body must be JSON."}), 400
            
            input_text = payload.get("input_text", "")
            output_text = payload.get("output_text", "")
            
            try:
                result = self.modernbert_classifier.classify_input_output_pair(input_text, output_text)
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error in /modernbert/classify endpoint: {e}", exc_info=True)
                return jsonify({"error": f"ModernBERT classification failed: {str(e)}"}), 500

        @self.app.route('/colbert/classify_sensitivity', methods=['POST'])
        def colbert_classify_sensitivity_endpoint():
            if not self.colbert_reranker:
                return jsonify({"error": "ColBERT reranker not available."}), 503
            
            payload = request.get_json()
            if not payload: return jsonify({"error": "Request body must be JSON."}), 400
            
            text_to_classify = payload.get("text", "")
            if not text_to_classify: return jsonify({"error": "'text' field required."}), 400

            try:
                result = self.colbert_reranker.classify_text(text_to_classify)
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error in /colbert/classify_sensitivity endpoint: {e}", exc_info=True)
                return jsonify({"error": f"ColBERT sensitivity classification failed: {str(e)}"}), 500

        @self.app.route('/rag/query', methods=['POST']) 
        def rag_query_endpoint():
            if not self.documentation_rag_retriever or not self.documentation_rag_retriever._is_loaded:
                return jsonify({"error": "Global RAG retriever not available or not loaded."}), 503
            
            payload = request.get_json()
            if not payload: return jsonify({"error": "Request body must be JSON."}), 400
            
            query_text = payload.get("query", "")
            top_k_results = payload.get("top_k", 5)
            if not query_text: return jsonify({"error": "'query' field required."}), 400

            try:
                results = self.documentation_rag_retriever.retrieve(query_text, top_k=top_k_results)
                return jsonify({"query": query_text, "results": results})
            except Exception as e:
                logger.error(f"Error in /rag/query endpoint: {e}", exc_info=True)
                return jsonify({"error": f"RAG query failed: {str(e)}"}), 500
        
        @self.app.route('/status', methods=['GET'])
        def status_endpoint():
            policy_status = "Not loaded"
            if self.api_policy_config:
                policy_status = f"{len(self.api_policy_config)} policies loaded"
            elif self.policy_config_path and not Path(self.policy_config_path).exists():
                policy_status = f"Config file not found at {self.policy_config_path}"
            elif self.policy_config_path:
                 policy_status = f"Config file found at {self.policy_config_path} but failed to load policies (check logs)"


            return jsonify({
                "service_status": "running",
                "total_requests_processed": self.request_count,
                "policy_configuration_status": policy_status,
                "component_status": {
                    "modernbert_classifier_available": self.modernbert_classifier is not None and self.modernbert_classifier.is_setup,
                    "colbert_reranker_available": self.colbert_reranker is not None,
                    "vision_language_processor_available": self.vision_language_processor is not None and self.vision_language_processor.is_setup,
                    "global_documentation_rag_retriever_loaded": self.documentation_rag_retriever is not None and self.documentation_rag_retriever._is_loaded,
                    "global_documentation_rag_details": self.documentation_rag_retriever.get_status() if self.documentation_rag_retriever else {"status": "not_initialized"}
                },
                "cached_rag_retrievers": {path: retriever.get_status() for path, retriever in RAG_COMPONENT_CACHE.items()}
            })

    def run(self, **kwargs):
        """Run the API server."""
        self.setup(**kwargs) 
        logger.info(f"Starting Enhanced Classification API server on http://{self.host}:{self.port}")
        logger.info(f"CORS enabled for all origins on this server.")
        logger.info(f"Number of policies loaded: {len(self.api_policy_config)}")
        
        logger.info(f"ModernBERT Classifier: {'Available' if self.modernbert_classifier and self.modernbert_classifier.is_setup else 'Not Available'}")
        logger.info(f"ColBERT Reranker: {'Available' if self.colbert_reranker else 'Not Available'}")
        logger.info(f"Vision Language Processor (for items): {'Available' if self.vision_language_processor and self.vision_language_processor.is_setup else 'Not Available'}")
        if self.documentation_rag_retriever and self.documentation_rag_retriever._is_loaded:
             logger.info(f"Global Documentation RAG Retriever: Loaded with {self.documentation_rag_retriever.get_status().get('num_documents')} documents.")
        else:
            logger.info(f"Global Documentation RAG Retriever: Not loaded or not configured ({self.global_rag_retriever_index_path}).")


        serve(self.app, host=self.host, port=self.port, threads=8) 

# --- Codebase Indexing (Preserved) ---

def parse_and_chunk_python_code(code_content: str, strategy: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """Parse Python code and create chunks based on the specified strategy."""
    logger.info(f"Parsing Python code ({len(code_content)} chars) with strategy '{strategy}'")
    
    chunks = []
    
    tree = None
    try:
        tree = ast.parse(code_content)
    except SyntaxError as e:
        logger.error(f"Syntax error parsing Python code: {e}. Falling back to 'lines' strategy if not already.")
        if strategy != "lines": 
            strategy = "lines" 
            logger.info("Switched to 'lines' strategy due to syntax error.")
    
    lines = code_content.split('\n')
    
    if strategy == "functions" and tree:
        try:
            for node in tree.body: 
                if isinstance(node, ast.FunctionDef): 
                    start_line = node.lineno -1
                    end_line = getattr(node, 'end_lineno', start_line + len(ast.unparse(node).splitlines())) -1
                    
                    func_lines = lines[start_line : end_line + 1]
                    func_text = '\n'.join(func_lines)
                    signature = lines[start_line].strip() 
                    docstring = ast.get_docstring(node)
                    
                    chunks.append({
                        'id': f"function_{node.name}",
                        'text': func_text,
                        'metadata': {
                            'type': 'function', 'name': node.name, 'signature': signature,
                            'start_line': start_line + 1, 'end_line': end_line + 1,
                            'docstring': docstring or "", 'line_count': end_line - start_line + 1,
                            'class_context': None 
                        }
                    })
                elif isinstance(node, ast.ClassDef): 
                    for class_item_node in node.body:
                        if isinstance(class_item_node, ast.FunctionDef): 
                            method_node = class_item_node
                            start_line = method_node.lineno - 1
                            end_line = getattr(method_node, 'end_lineno', start_line + len(ast.unparse(method_node).splitlines())) -1

                            method_lines = lines[start_line : end_line + 1]
                            method_text = '\n'.join(method_lines)
                            signature = lines[start_line].strip()
                            docstring = ast.get_docstring(method_node)
                            
                            chunks.append({
                                'id': f"method_{node.name}_{method_node.name}", 
                                'text': method_text,
                                'metadata': {
                                    'type': 'method', 'name': method_node.name, 'signature': signature,
                                    'start_line': start_line + 1, 'end_line': end_line + 1,
                                    'docstring': docstring or "", 'line_count': end_line - start_line + 1,
                                    'class_context': node.name 
                                }
                            })
        except Exception as e_func:
            logger.error(f"Error processing functions/methods: {e_func}. AST parsing might have issues.", exc_info=True)
            if not chunks: strategy = "lines" 

    elif strategy == "classes" and tree:
        try:
            for node in tree.body: 
                if isinstance(node, ast.ClassDef): 
                    start_line = node.lineno - 1
                    end_line = getattr(node, 'end_lineno', start_line + len(ast.unparse(node).splitlines())) -1
                    
                    class_lines = lines[start_line : end_line + 1]
                    class_text = '\n'.join(class_lines)
                    
                    methods_signatures = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef): 
                            method_sig_line = lines[item.lineno - 1].strip()
                            methods_signatures.append(method_sig_line)
                    
                    docstring = ast.get_docstring(node)
                    
                    chunks.append({
                        'id': f"class_{node.name}",
                        'text': class_text, 
                        'metadata': {
                            'type': 'class', 'name': node.name,
                            'start_line': start_line + 1, 'end_line': end_line + 1,
                            'methods_signatures': methods_signatures, 'docstring': docstring or "",
                            'method_count': len(methods_signatures), 'line_count': end_line - start_line + 1
                        }
                    })
        except Exception as e_class:
            logger.error(f"Error processing classes: {e_class}. AST parsing might have issues.", exc_info=True)
            if not chunks: strategy = "lines" 
    
    if strategy == "lines" or not chunks: 
        if strategy != "lines" and not chunks: 
            logger.info(f"No chunks from '{strategy}' strategy, falling back to 'lines' strategy.")
        
        line_chunks_text = split_text_with_overlap(code_content, chunk_size, chunk_overlap)
        
        current_char_offset = 0
        for i, chunk_text in enumerate(line_chunks_text):
            if not chunk_text.strip(): continue

            start_line_num = code_content.count('\n', 0, current_char_offset) + 1
            chunk_line_count = chunk_text.count('\n') + 1
            end_line_num = start_line_num + chunk_line_count -1
            
            chunks.append({
                'id': f"code_segment_{i+1}",
                'text': chunk_text,
                'metadata': {
                    'type': 'code_segment',
                    'estimated_start_line': start_line_num,
                    'estimated_end_line': end_line_num,
                    'line_count': chunk_line_count,
                    'char_count': len(chunk_text)
                }
            })
            current_char_offset += len(chunk_text) + 1 

    logger.info(f"Created {len(chunks)} code chunks using '{strategy}' strategy.")
    return chunks

def build_codebase_rag_index(code_file_path: Path, index_path: Path, embedding_model_id: str, 
                           strategy: str, chunk_size: int, chunk_overlap: int):
    """Build a RAG index from Python source code."""
    try:
        logger.info(f"Building codebase RAG index from {code_file_path} with strategy '{strategy}'")
        
        if not code_file_path.exists() or not code_file_path.is_file():
            raise FileNotFoundError(f"Code file not found or is not a file: {code_file_path}")
        
        with open(code_file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        chunks = parse_and_chunk_python_code(code_content, strategy, chunk_size, chunk_overlap)
        
        if not chunks:
            logger.error("No chunks created from source code. RAG index for codebase cannot be built.")
            return False
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temp_jsonl = index_path.parent / f"{index_path.name}_temp_code_chunks.jsonl"
        
        save_chunks_as_jsonl(chunks, temp_jsonl)
        
        retriever = RAGRetriever(index_path)
        
        if strategy == "functions":
            metadata_fields = ["type", "name", "signature", "start_line", "end_line", "docstring", "line_count", "class_context"]
        elif strategy == "classes":
            metadata_fields = ["type", "name", "start_line", "end_line", "methods_signatures", "docstring", "method_count", "line_count"]
        else: 
            metadata_fields = ["type", "estimated_start_line", "estimated_end_line", "line_count", "char_count"]
        
        retriever.index_corpus(
            corpus_path=temp_jsonl,
            embedding_model_id=embedding_model_id,
            doc_id_field="id", 
            text_field="text",
            metadata_fields=metadata_fields
        )
        
        temp_jsonl.unlink(missing_ok=True)
        
        logger.info(f"Successfully built codebase RAG index at {index_path} with {len(chunks)} chunks.")
        return True
        
    except Exception as e:
        logger.error(f"Failed to build codebase RAG index for {code_file_path}: {e}", exc_info=True)
        return False

# --- Example File Creation with Enhanced Features ---

def create_example_files(base_path: Path,
                         docs_url: Optional[Union[str, List[str]]] = None, # Now correctly takes list or str or None
                         auto_build_docs_rag: bool = False,
                         docs_rag_index_name: str = "tool_documentation",
                         chunk_size: int = 500, 
                         chunk_overlap: int = 50, 
                         vlm_model_path: Optional[str] = None, 
                         processing_strategy: str = "vlm"): 
    """Create enhanced example files with VLM support."""
    logger.info(f"Creating example files in {base_path}...")
    base_path.mkdir(parents=True, exist_ok=True)

    sample_docs_content = """
# Enhanced Classifier Service Tool Documentation

## Overview

The Enhanced Classifier Service Tool is a comprehensive AI-powered classification and validation system featuring:

1. **VLM-Powered Markdown Processing**: Advanced document processing using GGUF models.
2. **Complete Policy Validation**: Full implementation of all policy checks.
3. **Input/Output Validation**: Using transformer models for content validation.
4. **Content Sensitivity Classification**: Using ColBERT-like models for sensitivity.
5. **Vision-Language Processing**: For images and videos in API requests.
6. **Retrieval Augmented Generation**: Enhanced RAG capabilities with documentation assistance.

## VLM-Enhanced Features

### Markdown Processing with VLM

The tool can use Vision-Language Models to intelligently process and chunk markdown documentation for RAG indexing:

```bash
python your_script_name.py create-example \\
    --auto-build-docs-rag \\
    --docs-url ./sample_doc.md \\
    --docs-vlm-model-path /path/to/your_model.gguf \\
    --processing-strategy vlm
```

Benefits:
- **Semantic Chunking**: Better understanding of document structure.
- **Context Preservation**: Maintains logical flow.
- **Code Block Handling**: Intelligent processing of code.
- **Metadata Enrichment**: Automatic extraction of topics (future).

### Complete Policy Implementation

The API `/service/validate` endpoint now includes comprehensive policy validation:

- Legacy Format Support (input_text, output_text)
- Multimodal Processing (input_items, file uploads)
- Custom Validation Rules (length, required fields, format)
- Documentation Assistance on violations

## API Endpoints

### Enhanced Validation Endpoint `/service/validate`

Supports complex policy configurations and multipart/form-data for files.

Example Payload (JSON part of multipart):
```json
{
  "api_class": "EnhancedMultimodalPolicy",
  "input_text": "Sample input text for general validation.",
  "output_text": "Sample output text for general validation.", 
  "input_items": [
    {
      "id": "image_item_1",
      "type": "image",
      "filename_in_form": "uploaded_image.jpg"
    }
  ]
}
```

### Direct Model Access (Placeholders)

#### ModernBERT I/O Classification `/modernbert/classify`
```bash
curl -X POST -H "Content-Type: application/json" \\
     -d '{"input_text": "Is this good?", "output_text": "Yes, it is."}' \\
     http://localhost:8080/modernbert/classify
```

#### ColBERT Sensitivity Analysis `/colbert/classify_sensitivity`
```bash
curl -X POST -H "Content-Type: application/json" \\
     -d '{"text": "Analyze this for PII like SSN 123-45-6789."}' \\
     http://localhost:8080/colbert/classify_sensitivity
```

#### Global RAG Query `/rag/query` (if global index configured)
```bash
curl -X POST -H "Content-Type: application/json" \\
     -d '{"query": "How to handle PII?", "top_k": 3}' \\
     http://localhost:8080/rag/query
```

## Enhanced Policy Configuration Example (`enhanced_policy_config.json`)

```json
{
  "SimpleValidation": {
    "description": "Basic I/O validation only.",
    "modernbert_io_validation": true
  },
  "EnhancedMultimodalPolicy": {
    "description": "Complete validation with VLM for items and documentation assistance.",
    "modernbert_io_validation": true,
    "colbert_input_sensitivity": true,
    "colbert_output_sensitivity": true,
    "disallowed_colbert_input_classes": ["Class 1: PII"],
    "disallowed_colbert_output_classes": ["Class 1: PII", "Class 2: Confidential"],
    "item_processing_rules": [
      {
        "item_type": "image",
        "vlm_processing": {
          "required": true,
          "prompt": "Describe this image, focusing on any sensitive content, text, or PII."
        },
        "derived_text_checks": {
          "colbert_sensitivity": true,
          "disallowed_classes": ["Class 1: PII"],
          "blocked_keywords": ["confidential_token", "private_info"]
        }
      }
    ],
    "custom_validation_rules": [
      {"type": "text_length_limit", "max_length": 2000, "text_fields": ["input_text", "output_text"]},
      {"type": "required_fields", "fields": ["input_text"]}
    ],
    "documentation_assistance": {
      "enabled": true,
      "index_path": "./tool_examples/tool_documentation", 
      "max_suggestions_per_query": 2,
      "max_total_suggestions": 5,
      "min_similarity_threshold": 0.15
    }
  }
}
```
## Codebase Analysis

### Index Your Codebase
```bash
python your_script_name.py index-codebase \\
    --code-file-path ./your_script_name.py \\
    --index-path ./my_code_index \\
    --code-chunk-strategy functions 
```

### Query Code Index
```bash
python your_script_name.py rag retrieve \\
    --index-path ./my_code_index \\
    --query "how to parse python code" \\
    --top-k 3
```
"""

    docs_file_path = base_path / "sample_tool_documentation.md"
    with open(docs_file_path, 'w', encoding='utf-8') as f:
        f.write(sample_docs_content)
    logger.info(f"Sample documentation markdown saved to {docs_file_path}")

    # Handle docs_url being None, single string, or list
    effective_docs_sources: List[str] = []
    if docs_url is None: # No specific docs_url provided, use the generated sample
        effective_docs_sources = [str(docs_file_path)]
    elif isinstance(docs_url, str): # Single URL/path string
        effective_docs_sources = [docs_url]
    else: # Already a list
        effective_docs_sources = docs_url

    
    if auto_build_docs_rag:
        docs_rag_index_full_path = base_path / docs_rag_index_name
        logger.info(f"Attempting to build documentation RAG index at {docs_rag_index_full_path} from sources: {effective_docs_sources}")
        
        if not effective_docs_sources:
            logger.warning("No documentation sources to build RAG index from, even after checking defaults. Skipping RAG build.")
        else:
            success = build_documentation_rag_index(
                docs_url_or_path_list=effective_docs_sources, 
                index_path=docs_rag_index_full_path,
                embedding_model_id="all-MiniLM-L6-v2", 
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                vlm_model_path=vlm_model_path, 
                processing_strategy=processing_strategy
            )
            
            if success:
                logger.info(f"Documentation RAG index built successfully at {docs_rag_index_full_path}")
            else:
                logger.error(f"Failed to build documentation RAG index. Check logs.")
    else:
        logger.info("Skipping documentation RAG index build as --auto-build-docs-rag was not specified.")


    policy_docs_rag_path_str = (Path(docs_rag_index_name)).as_posix() # Relative to base_path

    enhanced_policy_config_content = {
        "SimpleValidation": {
            "description": "Basic I/O validation only using ModernBERT.",
            "modernbert_io_validation": True
        },
        "StrictPIICheck": {
            "description": "Comprehensive PII detection for input and output using ColBERT.",
            "modernbert_io_validation": True, 
            "colbert_input_sensitivity": True,
            "colbert_output_sensitivity": True,
            "disallowed_colbert_input_classes": ["Class 1: PII"],
            "disallowed_colbert_output_classes": ["Class 1: PII"]
        },
        "EnhancedMultimodalPolicy": {
            "description": "Complete validation with VLM for items and documentation assistance.",
            "modernbert_io_validation": True,
            "colbert_input_sensitivity": True,
            "colbert_output_sensitivity": True,
            "disallowed_colbert_input_classes": ["Class 1: PII"],
            "disallowed_colbert_output_classes": ["Class 1: PII", "Class 2: Confidential"],
            "item_processing_rules": [
                {
                    "item_type": "image", 
                    "vlm_processing": { 
                        "required": True,
                        "prompt": "Analyze this image for sensitive content, PII, or policy violations."
                    },
                    "derived_text_checks": { 
                        "colbert_sensitivity": True,
                        "disallowed_classes": ["Class 1: PII", "Class 2: Confidential"], 
                        "blocked_keywords": ["highly_confidential_internal_use_only", "top_secret_project_alpha"]
                    }
                }
            ],
            "custom_validation_rules": [
                {"type": "text_length_limit", "max_length": 2048, "text_fields": ["input_text", "output_text"]},
                {"type": "required_fields", "fields": ["input_text", "user_id"]}, 
                {"type": "format_validation", "field": "user_email", "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
            ],
            "rate_limiting": {"enabled": True, "max_requests_per_minute": 100}, 
            "documentation_assistance": {
                "enabled": True,
                "index_path": policy_docs_rag_path_str, 
                "max_suggestions_per_query": 3,
                "max_total_suggestions": 5,
                "min_similarity_threshold": 0.1,
                "max_help_queries_to_use": 3
            }
        }
    }
    
    policy_config_file_path = base_path / "enhanced_policy_config.json"
    with open(policy_config_file_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_policy_config_content, f, indent=2)
    logger.info(f"Enhanced policy configuration saved to {policy_config_file_path}")


    sample_rag_corpus_content = [
        {"id": "doc_rag_1", "text": "VLM processing uses advanced models for document understanding.", "metadata": {"category": "vlm", "difficulty": "advanced"}},
        {"id": "doc_rag_2", "text": "Policy validation ensures API requests meet criteria.", "metadata": {"category": "api_policy", "difficulty": "intermediate"}},
        {"id": "doc_rag_3", "text": "For PII, use ColBERT sensitivity checks and disallow specific classes.", "metadata": {"category": "pii_handling", "difficulty": "intermediate"}},
        {"id": "doc_rag_4", "text": "Example command: curl -X POST ... /service/validate", "metadata": {"category": "api_usage", "is_command": True}}
    ]
    sample_rag_corpus_file_path = base_path / "sample_generic_rag_corpus.jsonl"
    with open(sample_rag_corpus_file_path, 'w', encoding='utf-8') as f:
        for item in sample_rag_corpus_content:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Sample generic RAG corpus saved to {sample_rag_corpus_file_path}")

    logger.info(f"Enhanced example files created successfully in {base_path}")


# --- Main CLI Function ---

def _initialize_and_run():
    """
    Runs main_cli. Assumes all necessary global imports have been made
    in the `if __name__ == "__main__":` block after venv setup.
    """
    logger.debug("Global imports should be complete. Calling main_cli.")
    return main_cli()
def main_cli():
    """Main CLI function that runs after dependencies are imported."""
    setup_signal_handling()
    
    logger.info(f"DEBUG: main_cli received sys.argv: {sys.argv}")
    
    parser = argparse.ArgumentParser(
        description="Enhanced Transformer-based Classification Service with VLM Processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    parser_index_codebase = subparsers.add_parser(
        "index-codebase",
        help="Create a RAG index from Python source code."
    )
    parser_index_codebase.add_argument("--code-file-path", type=str, default=__file__, help="Path to the Python code file to index.")
    parser_index_codebase.add_argument("--index-path", type=str, required=True, help="Directory path to save the RAG index for code.")
    parser_index_codebase.add_argument("--embedding-model-id", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model for embeddings.")
    parser_index_codebase.add_argument(
        "--code-chunk-strategy",
        type=str,
        choices=['functions', 'classes', 'lines'],
        default='functions',
        help="Strategy for chunking Python code (functions/methods, classes, or lines)."
    )
    parser_index_codebase.add_argument("--code-chunk-size", type=int, default=1000, help="Target chunk size in characters (for 'lines' strategy or very long functions/classes).")
    parser_index_codebase.add_argument("--code-chunk-overlap", type=int, default=100, help="Overlap size in characters (for 'lines' strategy).")

    parser_create_example = subparsers.add_parser(
        "create-example",
        help="Create example files (documentation, policy config, RAG corpus)."
    )
    parser_create_example.add_argument("--output-dir", type=str, default="enhanced_tool_examples", help="Directory to save example files.")    
    parser_create_example.add_argument(
        "--docs-url", 
        nargs='*',  
        default=None, 
        help="URL(s) or local path(s) to markdown documentation file(s) to process (space-separated). Defaults to internal sample doc if not provided."
    )
    parser_create_example.add_argument("--auto-build-docs-rag", action="store_true", help="Automatically build RAG index from the documentation.")
    parser_create_example.add_argument("--docs-rag-index-name", type=str, default="tool_documentation", help="Name for the documentation RAG index directory (created within --output-dir).")
    parser_create_example.add_argument("--chunk-size", type=int, default=800, help="Target chunk size for VLM/fallback document processing.")
    parser_create_example.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap for secondary/fallback document processing.")
    parser_create_example.add_argument(
        "--docs-vlm-model-path",
        type=str,
        help="Path to local GGUF model file OR HuggingFace Repo ID for VLM-based documentation processing." # Updated help
    )
    parser_create_example.add_argument(
        "--processing-strategy",
        type=str,
        choices=['vlm', 'python'],
        default='vlm',
        help="Documentation processing strategy ('vlm' or 'python' fallback)."
    )

    parser_rag = subparsers.add_parser("rag", help="RAG (Retrieval Augmented Generation) utilities.")
    rag_subparsers = parser_rag.add_subparsers(dest="rag_command", required=True)

    parser_rag_index_cmd = rag_subparsers.add_parser("index", help="Create a RAG index from a JSONL corpus.")
    parser_rag_index_cmd.add_argument("--corpus-path", type=str, required=True, help="Path to corpus file (JSONL format).")
    parser_rag_index_cmd.add_argument("--index-path", type=str, required=True, help="Directory path to save the RAG index.")
    parser_rag_index_cmd.add_argument("--embedding-model-id", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model for embeddings.")
    parser_rag_index_cmd.add_argument("--doc-id-field", type=str, default="id", help="Field name for document ID in JSONL.")
    parser_rag_index_cmd.add_argument("--text-field", type=str, default="text", help="Field name for text content in JSONL.")
    parser_rag_index_cmd.add_argument("--metadata-fields", nargs='*', help="List of metadata field names to include from JSONL (e.g., category, source). Assumes they are under a 'metadata' key in each JSONL object or top-level.")
    parser_rag_index_cmd.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation.")

    parser_rag_retrieve_cmd = rag_subparsers.add_parser("retrieve", help="Retrieve documents from a RAG index (CLI query tool).")
    parser_rag_retrieve_cmd.add_argument("--index-path", type=str, required=True, help="Path to the RAG index directory.")
    parser_rag_retrieve_cmd.add_argument("--query", type=str, required=True, help="Query string to search for.")
    parser_rag_retrieve_cmd.add_argument("--top-k", type=int, default=5, help="Number of top documents to retrieve.")

    parser_serve = subparsers.add_parser("serve", help="Start the enhanced classification API server.")
    parser_serve.add_argument("--modernbert-model-dir", type=str, default=None, help="Path to a directory containing a trained ModernBERT model (for real model usage, otherwise uses placeholder).")
    parser_serve.add_argument("--policy-config-path", type=str, default="enhanced_tool_examples/enhanced_policy_config.json", help="Path to the API policy configuration JSON file.")
    parser_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host address for the API server.")
    parser_serve.add_argument("--port", type=int, default=8080, help="Port for the API server.")
    parser_serve.add_argument("--global-rag-retriever-index-path", type=str, default=None, help="Path to a global RAG index for the /rag/query endpoint and potentially default documentation assistance if not specified in policy.")
    parser_serve.add_argument("--vlm-model-path", type=str, help="Path to VLM GGUF model OR HF Repo ID for item processing in API policies (e.g., LLaVA). Placeholder if not provided.") # Updated help

    parser_test = subparsers.add_parser("test", help="Run comprehensive internal tests for the system.")
    parser_test.epilog = ("For detailed instructions on localizing, executing, and "
                          "troubleshooting the test suite, especially for IDE-based development or "
                          "automated agents, please refer to the 'ide-agent.md' guide in the project documentation.")
    # MODIFIED_LINE_CHOICES
    parser_test.add_argument("--test-type", choices=['all', 'vlm', 'policy', 'rag', 'codebase', 'model_logic'], default='all', help="Specific category of tests to run.")
    parser_test.add_argument("--verbose", action="store_true", help="Enable verbose logging during tests.")

    parser_finetune_io_validator = subparsers.add_parser(
        "finetune-io-validator",
        help="Fine-tune the ModernBERT I/O Validator on labeled input/output pairs."
    )
    parser_finetune_io_validator.add_argument("--data-path", type=str, required=True, help="Path to JSONL file with input_text, output_text, and label fields.")
    parser_finetune_io_validator.add_argument("--base-model-id", type=str, default=ModernBERTClassifier.DEFAULT_MODEL_ID, help="Base HuggingFace model ID to fine-tune.")
    parser_finetune_io_validator.add_argument("--output-model-dir", type=str, required=True, help="Directory to save the fine-tuned model and tokenizer.")
    parser_finetune_io_validator.add_argument("--num-train-epochs", type=int, default=3, help="Number of training epochs.")
    parser_finetune_io_validator.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate.")
    parser_finetune_io_validator.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser_finetune_io_validator.add_argument("--test-size", type=float, default=0.1, help="Proportion of data to use for evaluation.")

    args = parser.parse_args()

    if args.command == "index-codebase":
        success = build_codebase_rag_index(
            code_file_path=Path(args.code_file_path),
            index_path=Path(args.index_path),
            embedding_model_id=args.embedding_model_id,
            strategy=args.code_chunk_strategy,
            chunk_size=args.code_chunk_size,
            chunk_overlap=args.code_chunk_overlap
        )
        return 0 if success else 1

    elif args.command == "create-example":
        create_example_files(
            base_path=Path(args.output_dir),
            docs_url=args.docs_url, # args.docs_url is now None or a list of strings
            auto_build_docs_rag=args.auto_build_docs_rag,
            docs_rag_index_name=args.docs_rag_index_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            vlm_model_path=args.docs_vlm_model_path, 
            processing_strategy=args.processing_strategy
        )
        return 0

    elif args.command == "rag":
        if args.rag_command == "index":
            retriever = RAGRetriever(args.index_path)
            retriever.index_corpus(
                corpus_path=args.corpus_path,
                embedding_model_id=args.embedding_model_id,
                doc_id_field=args.doc_id_field,
                text_field=args.text_field,
                metadata_fields=args.metadata_fields,
                batch_size=args.batch_size
            )
            return 0 
        elif args.rag_command == "retrieve":
            retriever = RAGRetriever(args.index_path)
            if not retriever.load_index():
                logger.error(f"Failed to load RAG index from {args.index_path} for retrieval.")
                return 1
            results = retriever.retrieve(args.query, top_k=args.top_k)
            print(json.dumps(results, indent=2)) 
            return 0

    elif args.command == "serve":
        api = ClassificationAPI(
            modernbert_model_dir=args.modernbert_model_dir,
            host=args.host,
            port=args.port,
            policy_config_path=args.policy_config_path,
            vlm_model_id_or_dir=args.vlm_model_path, 
            global_rag_retriever_index_path=args.global_rag_retriever_index_path
        )
        api.run() 
        return 0 

    elif args.command == "test":
        passed, total = run_comprehensive_tests(args.test_type, args.verbose)
        return 0 if passed == total and total > 0 else 1 

    elif args.command == "finetune-io-validator":
        try:
            # This function would need to be defined, it's a placeholder for now.
            # metrics = _finetune_io_validator_model(
            #     data_path=args.data_path,
            #     base_model_id=args.base_model_id,
            #     output_model_dir=args.output_model_dir,
            #     num_train_epochs=args.num_train_epochs,
            #     learning_rate=args.learning_rate,
            #     batch_size=args.batch_size,
            #     test_size=args.test_size,
            # )
            logger.info(f"Fine-tuning requested. Placeholder: Fine-tuning logic needs to be implemented in _finetune_io_validator_model.")
            # logger.info(f"Fine-tuning completed. Metrics: {metrics}") # If it were implemented
            return 0 # Placeholder return
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}", exc_info=True)
            return 1

    else:
        parser.print_help()
        return 1
    return 0 

# --- Comprehensive Testing Framework ---

def run_comprehensive_tests(test_type: str = "all", verbose: bool = False) -> Tuple[int, int]:
    """Run comprehensive tests for the enhanced system. Returns (passed_tests, total_tests)."""
    logger.info(f"Running '{test_type}' tests...")
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG) 
        logger.info("Verbose logging enabled for tests.")
    else:
        logging.getLogger().setLevel(logging.INFO) 

    test_results_summary: Dict[str, Dict[str, Any]] = {}
    
    with tempfile.TemporaryDirectory(prefix="cls_service_tests_") as temp_dir_str:
        temp_dir_path = Path(temp_dir_str)
        logger.info(f"Using temporary directory for test artifacts: {temp_dir_path}")

        if test_type in ["all", "vlm"]:
            test_results_summary["VLM Processing"] = test_vlm_processing(temp_dir_path)
        
        # CORRECTED_POLICY_VALIDATION_CALL
        if test_type in ["all", "policy"] and "Policy Validation" not in test_results_summary : # Prevent double call if 'all' and 'policy' are both somehow active
            test_results_summary["Policy Validation"] = test_policy_validation(temp_dir_path)
        
        if test_type in ["all", "rag"]:
            test_results_summary["RAG Functionality"] = test_rag_functionality(temp_dir_path)
        
        if test_type in ["all", "codebase"]:
            test_results_summary["Codebase Indexing"] = test_codebase_indexing(temp_dir_path)
        
        # MODIFIED_SECTION_MODEL_LOGIC_TESTS_CALL (Corrected Structure)
        if test_type in ["all", "model_logic"]: 
            # The individual test functions return a dict {test_name: result}, so spread them.
            model_logic_results = {}
            model_logic_results.update(test_colbert_reranker_logic(temp_dir_path))
            model_logic_results.update(test_modernbert_classifier_default_inference(temp_dir_path))
            test_results_summary["Model Logic Tests"] = model_logic_results
            
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests_run = 0
    total_tests_passed = 0
    
    for category, tests_in_category in test_results_summary.items():
        # CORRECTED_TEST_CATEGORY_PRINTING (remove extra "TESTS")
        print(f"\n{category.upper()}:") # Removed extra "TESTS"
        cat_total = len(tests_in_category)
        cat_passed = 0
        for test_name, result_details in tests_in_category.items():
            status_icon = "[PASS]" if result_details.get("passed", False) else "[FAIL]"
            print(f"  - {test_name}: {status_icon}")
            if not result_details.get("passed", False):
                print(f"    Reason: {result_details.get('error', 'Unknown error')}")
                if "details" in result_details: print(f"    Details: {result_details['details']}")
            total_tests_run += 1
            if result_details.get("passed", False):
                total_tests_passed += 1
                cat_passed +=1
        print(f"  Category Summary: {cat_passed}/{cat_total} passed.")
    
    print("\n" + "="*60)
    if total_tests_run == 0:
        print("No tests were run for the selected type.")
    elif total_tests_passed == total_tests_run:
        print(f"🎉 All {total_tests_passed}/{total_tests_run} tests PASSED! 🎉")
    else:
        print(f"⚠️  {total_tests_passed}/{total_tests_run} tests passed. {total_tests_run - total_tests_passed} tests FAILED! ⚠️")
    print("="*60)
    
    return total_tests_passed, total_tests_run


def test_vlm_processing(temp_test_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Test VLM-based markdown processing capabilities."""
    results: Dict[str, Dict[str, Any]] = {}
    
    test_name = "vlm_model_loading" 
    try:
        logger.info(f"Testing: {test_name} - This may download '{MarkdownReformatterVLM.DEFAULT_TEST_GGUF_FILENAME}' from '{MarkdownReformatterVLM.DEFAULT_TEST_HF_REPO_ID}' if not cached.")
        formatter = MarkdownReformatterVLM(
            model_path_or_repo_id=MarkdownReformatterVLM.DEFAULT_TEST_HF_REPO_ID,
            gguf_filename_in_repo=MarkdownReformatterVLM.DEFAULT_TEST_GGUF_FILENAME
        )
        can_load = formatter.load_model()
        
        if LLAMA_CPP_AVAILABLE:
            results[test_name] = {
                "passed": can_load,
                "details": f"Attempted to download and load Gemma GGUF. Load success: {can_load}. Ensure HF_TOKEN is valid if this fails."
            }
        else: 
            results[test_name] = {
                "passed": not can_load, "details": "load_model correctly returned False as llama-cpp is unavailable."
            }
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e), "details": "Ensure HF_TOKEN is valid and model accessible."} # Added details
    
    test_name = "prompt_template_creation"
    try:
        logger.info(f"Testing: {test_name}")
        prompt = create_vlm_markdown_prompt_template()
        assert "{raw_markdown_content}" in prompt, "Prompt missing content placeholder."
        assert "JSON Output:" in prompt, "Prompt missing JSON output instruction."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}
    
    test_name = "vlm_output_parsing_valid_json"
    try:
        logger.info(f"Testing: {test_name}")
        mock_vlm_output_valid = '''Some preamble text...
    ```json
    [
      {
        "id": "test_chunk_1_valid",
        "text": "This is a valid test chunk.",
        "metadata": {
          "h1_section": "Test Section Valid", "is_code_block": false,
          "chunk_type": "content", "topics": ["testing_valid"]
        }
      }
    ]
```
Some postamble text...'''
        parsed_chunks = parse_vlm_output(mock_vlm_output_valid, "test_url_valid")
        assert len(parsed_chunks) == 1, f"Expected 1 chunk, got {len(parsed_chunks)}. Chunks: {parsed_chunks}"
        chunk = parsed_chunks[0]
        assert isinstance(chunk, dict), "Parsed chunk is not a dictionary."
        assert chunk.get("id") == "test_chunk_1_valid", f"Chunk ID mismatch. Got {chunk.get('id')}"
        assert chunk.get("metadata", {}).get("chunk_type") == "content", "Incorrect chunk type"
        assert chunk.get("metadata", {}).get("source_url") == "test_url_valid", "Source URL not added."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    test_name = "vlm_output_parsing_malformed_json"
    try:
        logger.info(f"Testing: {test_name}")
        mock_vlm_output_malformed = '```json\n[\n  {"id": "malformed", "text": "incomplete json", \n]\n```' # Note: intentionally malformed
        parsed_chunks_malformed = parse_vlm_output(mock_vlm_output_malformed, "test_url_malformed")
        assert len(parsed_chunks_malformed) == 0, "Expected 0 chunks for malformed JSON."
        results[test_name] = {"passed": True, "details": "Correctly handled malformed JSON by returning empty list."}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    test_name = "fallback_markdown_processing_basic"
    try:
        logger.info(f"Testing: {test_name}")
        sample_markdown_fallback = """# Header One
Content for section one.

## Sub Header Two
More content.
```python
print("Hello from fallback")
```
"""
        fallback_chunks = fallback_markdown_processing(sample_markdown_fallback, "test_url_fallback", target_chunk_size=500)
        assert len(fallback_chunks) > 0, "Fallback processing should produce chunks."
        h1_found = any(c['metadata'].get('h1_section') == "Header One" for c in fallback_chunks)
        h2_found = any(c['metadata'].get('h2_section') == "Sub Header Two" for c in fallback_chunks)
        code_element_found_in_metadata = any(c['metadata'].get('contains_code_elements') for c in fallback_chunks if "print(" in c['text'])
        assert h1_found or h2_found, "Fallback did not capture H1/H2 headers in metadata."
        assert code_element_found_in_metadata, "Fallback did not mark chunk with code as containing code elements."

        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    test_name = "secondary_chunking_large_text"
    try:
        logger.info(f"Testing: {test_name}")
        large_text_content = "This is a very long string. " * 100
        initial_chunk = {
            "id": "large_test_chunk",
            "text": large_text_content,
            "metadata": {"h1_section": "Large Text Section", "source_url": "test_secondary_chunk"}
        }
        target_size = 500
        secondary_chunks = apply_secondary_chunking([initial_chunk], target_size)
        assert len(secondary_chunks) > 1, "Secondary chunking should split large text."
        for sc in secondary_chunks:
            assert len(sc["text"]) <= target_size * 1.2, f"Sub-chunk too large: {len(sc['text'])} vs target {target_size}" 
            assert sc["metadata"]["parent_chunk_id"] == "large_test_chunk", "Parent ID not set in sub-chunk."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}
        
    return results


def test_policy_validation(temp_test_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Test complete policy validation implementation using placeholder models."""
    results: Dict[str, Dict[str, Any]] = {}
    
    api_for_test = ClassificationAPI(
        modernbert_model_dir=None,
        host="localhost", port=0,
        policy_config_path=None
    )
    api_for_test.setup() # This is where self.colbert_reranker might become None due to its own init issues

    test_name = "legacy_policy_pass_case"
    try:
        logger.info(f"Testing: {test_name}")
        payload = {"input_text": "This is a benign question.", "output_text": "This is a benign answer."}
        policy_rules = {
            "modernbert_io_validation": True,
            "colbert_input_sensitivity": True, "disallowed_colbert_input_classes": ["Class 1: PII"],
            "colbert_output_sensitivity": True, "disallowed_colbert_output_classes": ["Class 1: PII"]
        }
        response_data_store = {}
        
        # Mock ModernBERTClassifier response for "Good" classification
        with mock.patch.object(api_for_test.modernbert_classifier, 'classify_input_output_pair') as mock_bert_classify:
            mock_bert_classify.return_value = {
                "prediction": 1,
                "probability_positive": 0.98,
                "confidence": 0.98,
                "details": "Mocked successful validation"
            }

            original_colbert_reranker = api_for_test.colbert_reranker
            temp_colbert_mock_assigned_for_pass_case = False

            colbert_input_active = policy_rules.get("colbert_input_sensitivity", False)
            colbert_output_active = policy_rules.get("colbert_output_sensitivity", False)

            if colbert_input_active or colbert_output_active:
                if api_for_test.colbert_reranker is None: # If ColBERTReranker failed its own init
                    logger.warning(f"{test_name}: api_for_test.colbert_reranker is None due to init failure. Assigning temporary mock for policy test.")
                    api_for_test.colbert_reranker = mock.Mock(spec=ColBERTReranker)
                    temp_colbert_mock_assigned_for_pass_case = True
                
                # api_for_test.colbert_reranker is now either the real one (if it loaded) or a Mock
                with mock.patch.object(api_for_test.colbert_reranker, 'classify_text') as mock_colbert_classify_pass:
                    mock_colbert_classify_pass.return_value = {"predicted_class": "Class 4: Public", "confidence": 0.9, "class_scores": {"Class 4: Public": 0.9}}
                    violations = api_for_test._handle_legacy_policy_validation(payload, policy_rules, response_data_store)
            else:
                violations = api_for_test._handle_legacy_policy_validation(payload, policy_rules, response_data_store)
            
            if temp_colbert_mock_assigned_for_pass_case:
                api_for_test.colbert_reranker = original_colbert_reranker

        assert len(violations) == 0, f"Expected 0 violations, got {len(violations)}: {violations}"
        assert response_data_store["modernbert_io_validation"]["prediction"] == 1, \
            f"Expected mocked prediction 1, got {response_data_store['modernbert_io_validation']['prediction']}"
        if colbert_input_active:
            assert "colbert_input_sensitivity" in response_data_store, "ColBERT input sensitivity data missing."
        if colbert_output_active:
            assert "colbert_output_sensitivity" in response_data_store, "ColBERT output sensitivity data missing."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    test_name = "legacy_policy_modernbert_fail_case"
    try:
        logger.info(f"Testing: {test_name}")
        payload = {"input_text": "This is a problematic input.", "output_text": "This is a problematic output."}
        policy_rules = {"modernbert_io_validation": True}
        response_data_store = {}
        with mock.patch.object(api_for_test.modernbert_classifier, 'classify_input_output_pair') as mock_bert_fail:
            mock_bert_fail.return_value = {
                "prediction": 0, "probability_positive": 0.1, "confidence": 0.9, "details": "Mocked failed validation"
            }
            violations = api_for_test._handle_legacy_policy_validation(payload, policy_rules, response_data_store)
        
        assert len(violations) == 1, f"Expected 1 ModernBERT violation, got {len(violations)}: {violations}"
        # CORRECTED_LEGACY_POLICY_MODERNBERT_FAIL_ASSERTION
        if not any("ModernBERT I/O validation failed" in v for v in violations):
            print(f"Actual violation messages: {violations}")  # Temporary for debugging
        assert any("ModernBERT I/O validation failed" in v for v in violations), "Violation message mismatch for ModernBERT fail."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    test_name = "legacy_policy_input_pii_fail"
    try:
        logger.info(f"Testing: {test_name}")
        payload_pii = {"input_text": "My SSN is 123-45-6789.", "output_text": "Okay."}
        policy_rules_pii = {"colbert_input_sensitivity": True, "disallowed_colbert_input_classes": ["Class 1: PII"]}
        response_data_store_pii = {}
        
        original_colbert_reranker = api_for_test.colbert_reranker
        temp_colbert_mock_assigned = False
        if api_for_test.colbert_reranker is None: # If ColBERTReranker failed its own init
            logger.warning(f"{test_name}: api_for_test.colbert_reranker is None due to init failure. Assigning temporary mock for policy test.")
            api_for_test.colbert_reranker = mock.Mock(spec=ColBERTReranker)
            temp_colbert_mock_assigned = True

        # api_for_test.colbert_reranker is now either the real one (if it loaded and not None) or a Mock
        with mock.patch.object(api_for_test.colbert_reranker, 'classify_text') as mock_colbert_classify:
            mock_colbert_classify.return_value = {
                "predicted_class": "Class 1: PII", "confidence": 0.9, "class_scores": {"Class 1: PII": 0.9}
            }
            violations_pii = api_for_test._handle_legacy_policy_validation(payload_pii, policy_rules_pii, response_data_store_pii)
        
        if temp_colbert_mock_assigned:
            api_for_test.colbert_reranker = original_colbert_reranker
        assert len(violations_pii) > 0, f"Expected PII violation, got {len(violations_pii)}. Violations: {violations_pii}. ColBERT mock called: {mock_colbert_classify.called if hasattr(mock_colbert_classify, 'called') else 'Mock not effective'}"
        assert any("Input text classified as disallowed sensitivity" in v for v in violations_pii), "PII violation message mismatch."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    test_name = "multimodal_vlm_item_processing"
    try:
        logger.info(f"Testing: {test_name}")
        payload_mm = {
            "input_items": [{"id": "img_1", "type": "image", "filename_in_form": "test_image.jpg"}]
        }
        two_by_two_png_bytes = base64.b64decode(
            b'iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4XmP8z8AARFDw/z/DeiAmeMIAFkQPt6gUX3EAAAAASUVORK5CYII='
        )
        
        class MockFileStorage:
            def __init__(self, filename, content=two_by_two_png_bytes):
                self.filename = filename; self.content = content
            def read(self): return self.content
            def seek(self, offset): pass
        
        mock_files = {"test_image.jpg": MockFileStorage("test_image.jpg", content=two_by_two_png_bytes)}

        policy_rules_mm = {
            "item_processing_rules": [{
                "item_type": "image",
                "vlm_processing": {"required": True, "prompt": "Check for sensitive content."},
                "derived_text_checks": {"colbert_sensitivity": True, "disallowed_classes": ["Class 1: PII"]}
            }]
        }
        response_data_store_mm = {}
        if not api_for_test.vision_language_processor or not api_for_test.vision_language_processor.is_setup :
            logger.info("Test forcing VLP setup for policy test.")
            api_for_test.vision_language_processor = VisionLanguageProcessor().setup()

        with mock.patch.object(api_for_test.vision_language_processor, 'describe_image') as mock_describe:
            mock_describe.return_value = {
                "description": "Mocked VLM: Benign image content",
                "analysis": {"contains_text_guess": False, "word_count": 5}
            }
            violations_mm = api_for_test._handle_multimodal_policy_validation(payload_mm, policy_rules_mm, mock_files, response_data_store_mm)
        
        assert "item_processing_results" in response_data_store_mm, "Item processing results missing."
        item_result = response_data_store_mm["item_processing_results"].get("img_1", {})
        assert item_result, "Results for 'img_1' missing"
        assert "vlm_analysis" in item_result, f"VLM analysis missing from item_result: {item_result}"
        assert len(violations_mm) == 0, f"Unexpected violations: {violations_mm}"
        assert item_result.get("vlm_analysis", {}).get("description") == "Mocked VLM: Benign image content", "VLM description mismatch in results."
        results[test_name] = {"passed": True}

    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e), "details": "Check VLP setup or placeholder VLM logic."}


    test_name = "custom_rule_text_length_fail"
    try:
        logger.info(f"Testing: {test_name}")
        payload_len = {"input_text": "x" * 1001}
        policy_rules_len = {"custom_validation_rules": [
            {"type": "text_length_limit", "max_length": 1000, "text_fields": ["input_text"]}
        ]}
        response_data_store_len = {}
        violations_len = api_for_test._handle_additional_policy_checks(payload_len, policy_rules_len, response_data_store_len)
        assert len(violations_len) == 1, "Expected 1 length violation."
        assert any("exceeds maximum length" in v for v in violations_len), "Length violation message mismatch."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    test_name = "doc_assistance_query_generation"
    try:
        logger.info(f"Testing: {test_name}")
        violations_for_query = ["Input text classified as disallowed sensitivity: 'Class 1: PII'."]
        policy_for_query = {"description": "Test policy for PII"}
        queries = api_for_test._generate_help_queries(violations_for_query, policy_for_query)
        assert len(queries) > 0, "Should generate help queries."
        assert any("pii" in q.lower() for q in queries), "PII related query expected."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    return results


def test_rag_functionality(temp_test_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Test RAG indexing and retrieval functionality."""
    results: Dict[str, Dict[str, Any]] = {}
    
    test_name = "rag_index_create_load_retrieve"
    try:
        logger.info(f"Testing: {test_name}")
        corpus_file = temp_test_dir / "test_rag_corpus.jsonl"
        index_dir = temp_test_dir / "my_test_rag_index"
        
        test_corpus_data = [
            {"id": "rag_doc_1", "text": "Artificial intelligence is revolutionizing industries.", "metadata": {"topic": "AI"}},
            {"id": "rag_doc_2", "text": "Python is a versatile programming language for AI.", "metadata": {"topic": "Python"}},
            {"id": "rag_doc_3", "text": "Machine learning is a subset of AI.", "metadata": {"topic": "ML"}}
        ]
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for item in test_corpus_data:
                f.write(json.dumps(item) + '\n')

        retriever = RAGRetriever(index_dir)
        retriever.index_corpus(corpus_path=corpus_file, embedding_model_id="all-MiniLM-L6-v2", metadata_fields=["topic"])
        
        assert index_dir.exists(), "RAG index directory not created."
        assert (index_dir / RAGRetriever.INDEX_CONFIG_FILE).exists(), "RAG config file missing."

        loaded_retriever = RAGRetriever(index_dir)
        assert loaded_retriever.load_index(), "Failed to load created RAG index."
        assert loaded_retriever.get_status()["num_documents"] == len(test_corpus_data), "Document count mismatch after load."

        retrieved_items = loaded_retriever.retrieve("AI programming", top_k=1)
        assert len(retrieved_items) == 1, f"Retrieved {len(retrieved_items)} items, expected 1"
        assert retrieved_items[0]["id"] in ["rag_doc_1", "rag_doc_2"], f"Unexpected ID: {retrieved_items[0]['id']}"
        assert "topic" in retrieved_items[0]["metadata"], "Missing metadata field in retrieved item."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e), "details":"Check SentenceTransformer loading and HF Token for all-MiniLM-L6-v2"} # Added details

    test_name = "documentation_rag_build_fallback"
    try:
        logger.info(f"Testing: {test_name}")
        sample_doc_md = temp_test_dir / "sample_doc_for_rag.md"
        sample_doc_md.write_text("# Test Doc\n\nThis is content about API validation.\n\n## Section Two\nMore details.")
        
        doc_index_dir = temp_test_dir / "my_doc_rag_index"
        success = build_documentation_rag_index(
            docs_url_or_path_list=str(sample_doc_md), 
            index_path=doc_index_dir,
            embedding_model_id="all-MiniLM-L6-v2",
            chunk_size=100, 
            chunk_overlap=10,
            processing_strategy="python" 
        )
        assert success, "build_documentation_rag_index (fallback) failed."
        assert doc_index_dir.exists(), "Documentation RAG index (fallback) dir not created."

        doc_retriever = RAGRetriever(doc_index_dir)
        assert doc_retriever.load_index(), "Failed to load doc RAG index (fallback)."
        ret_docs = doc_retriever.retrieve("API validation", top_k=1)
        assert len(ret_docs) > 0, f"No results from doc RAG index (fallback) for 'API validation', got {len(ret_docs)}"
        assert "API validation" in ret_docs[0]["text"], f"Retrieved doc text mismatch. Got: {ret_docs[0]['text'] if ret_docs else 'N/A'}"
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e), "details":"Check SentenceTransformer loading for RAG."} # Added details
        
    return results

def test_codebase_indexing(temp_test_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Test codebase indexing functionality."""
    results: Dict[str, Dict[str, Any]] = {}
    
    sample_python_code = """
import json

def top_level_function(param1, param2):
    \"\"\"This is a top-level function docstring.\"\"\"
    x = param1 + param2
    return x

class MyClass:
    \"\"\"Docstring for MyClass.\"\"\"
    class_var = 100

    def __init__(self, name):
        self.name = name

    def method_one(self):
        \"\"\"Method one docstring.\"\"\"
        return f"Hello from {self.name}"

    def _private_method(self): # Methods starting with _ are often included by AST walkers
        pass 
"""
    code_file = temp_test_dir / "sample_test_code.py"
    code_file.write_text(sample_python_code)

    test_name = "code_parsing_strategy_functions"
    try:
        logger.info(f"Testing: {test_name}")
        chunks_func = parse_and_chunk_python_code(sample_python_code, "functions", 1000, 100)
        
        func_names_found = {c["metadata"]["name"] for c in chunks_func if c["metadata"]["type"] == "function"}
        method_names_found = {c["metadata"]["name"] for c in chunks_func if c["metadata"]["type"] == "method"}
        
        assert "top_level_function" in func_names_found, "Top-level function not chunked."
        assert "__init__" in method_names_found, "__init__ method not chunked."
        assert "method_one" in method_names_found, "method_one not chunked."
        assert "_private_method" in method_names_found, "_private_method not chunked by 'functions' strategy."
        assert len(chunks_func) == 4, f"Expected 4 chunks for 'functions' strategy, got {len(chunks_func)}."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    test_name = "code_parsing_strategy_classes"
    try:
        logger.info(f"Testing: {test_name}")
        chunks_class = parse_and_chunk_python_code(sample_python_code, "classes", 1000, 100)
        assert len(chunks_class) == 1, "Expected 1 class chunk."
        chunk = chunks_class[0]
        assert isinstance(chunk, dict), "Class chunk is not a dictionary."
        assert chunk.get("metadata", {}).get("name") == "MyClass", "Class name mismatch."
        assert chunk.get("metadata", {}).get("type") == "class", "Chunk type not 'class'."
        assert len(chunk.get("metadata", {}).get("methods_signatures", [])) == 3, "Incorrect number of method signatures for MyClass."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    test_name = "codebase_rag_build_and_query"
    try:
        logger.info(f"Testing: {test_name}")
        code_index_dir = temp_test_dir / "my_codebase_index"
        success_build = build_codebase_rag_index(
            code_file_path=code_file,
            index_path=code_index_dir,
            embedding_model_id="all-MiniLM-L6-v2",
            strategy="functions", 
            chunk_size=1000, chunk_overlap=100
        )
        assert success_build, "Codebase RAG index building failed."
        
        code_retriever = RAGRetriever(code_index_dir)
        assert code_retriever.load_index(), "Failed to load codebase RAG index."
        
        ret_code_items = code_retriever.retrieve("method one docstring", top_k=1)
        assert len(ret_code_items) > 0, f"Retrieval returned no items for 'method one docstring'. Got: {ret_code_items}"
        assert len(ret_code_items) == 1, f"Retrieval from code index did not return 1 item for 'method one docstring', got {len(ret_code_items)}."
        assert ret_code_items[0]["metadata"]["name"] == "method_one", f"Retrieved wrong code chunk. Got: {ret_code_items[0]['metadata']['name']}"
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e), "details": "Check SentenceTransformer loading and RAG logic."} # Added details
        
    return results

# NEW_MODEL_LOGIC_TEST_DEFINITIONS_START
def test_colbert_reranker_logic(temp_test_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Tests the ColBERTReranker's internal MaxSim logic using its loaded model."""
    results: Dict[str, Dict[str, Any]] = {}
    test_name = "colbert_reranker_direct_classification"
    try:
        logger.info(f"Testing: {test_name} - Directly testing ColBERTReranker's MaxSim logic.")
        reranker = ColBERTReranker(
            reference_examples={ 
                "Positive": ["good example content"],
                "Negative": ["bad example content"]
            }
        )
        assert reranker.model is not None, "ColBERTReranker model not loaded."
        assert reranker.tokenizer is not None, "ColBERTReranker tokenizer not loaded."

        classification_good = reranker.classify_text("This is good example content, very positive.")
        assert "predicted_class" in classification_good, "Missing 'predicted_class' in good classification."
        assert "class_scores" in classification_good, "Missing 'class_scores'."
        
        results[test_name] = {"passed": True, "details": f"Tested with model: {reranker.model_id}"}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e), "details": "Ensure ColBERTReranker can load its default model (e.g., distilbert-base-uncased) and HF Hub access is working."}
    return {test_name: results.get(test_name, {"passed": False, "error": "Initialization failed before assertions."})}

def test_modernbert_classifier_default_inference(temp_test_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Tests the ModernBERTClassifier's default model loading and inference call."""
    results: Dict[str, Dict[str, Any]] = {}
    test_name = "modernbert_classifier_direct_inference"
    try:
        logger.info(f"Testing: {test_name} - Directly testing ModernBERTClassifier's inference.")
        classifier = ModernBERTClassifier() 
        classifier.setup() 
        assert classifier.is_setup, "ModernBERTClassifier did not complete setup."
        result = classifier.classify_input_output_pair("Sample input", "Sample output")
        assert isinstance(result, dict) and "prediction" in result and "probability_positive" in result, "Unexpected result structure from ModernBERT."
        results[test_name] = {"passed": True, "details": f"Tested with model: {classifier.model_id}"}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e), "details": "Ensure ModernBERTClassifier can load its default model and HF Hub access is working."}
    return {test_name: results.get(test_name, {"passed": False, "error": "Initialization failed before assertions."})}
# NEW_MODEL_LOGIC_TEST_DEFINITIONS_END


if __name__ == "__main__":
    logger.info(f"Script Start/Restart: HF_TOKEN from env: {os.environ.get('HF_TOKEN')}")
    is_venv_ok = ensure_venv()
    logger.info(f"Global Scope: HF_TOKEN from env before _initialize_and_run: {os.environ.get('HF_TOKEN')}")
    exit_code = 1 

    if is_venv_ok:
        logger.info("Virtual environment confirmed. Loading core dependencies globally...")
        try:
            import numpy
            np = numpy

            from tqdm import tqdm
            import ranx
            from packaging import version 

            from sentence_transformers import SentenceTransformer as GlobalSentenceTransformer, util as st_util
            SentenceTransformer = GlobalSentenceTransformer 

            from flask import Flask as GlobalFlask, request as GlobalRequest, jsonify as GlobalJsonify
            from flask_cors import CORS as GlobalCORS
            from waitress import serve as GlobalServe
            Flask = GlobalFlask
            request = GlobalRequest
            jsonify = GlobalJsonify
            CORS = GlobalCORS
            serve = GlobalServe
            
            logger.info("Successfully imported core non-ML/LLM dependencies globally.")

            try:
                import llama_cpp
                LLAMA_CPP_AVAILABLE = True 
                logger.info("llama-cpp-python is available globally for VLM processing.")
            except ImportError:
                logger.warning("llama-cpp-python not available globally. VLM processing features will be limited.")

            import torch as global_torch 
            torch = global_torch
            import transformers 
            
            logger.info("Successfully imported core ML/LLM dependencies (PyTorch, Transformers) globally.")

            exit_code = _initialize_and_run()
        except SystemExit as e: 
            exit_code = e.code if isinstance(e.code, int) else 1
        except KeyboardInterrupt:
            logger.info("Process interrupted by user (KeyboardInterrupt). Exiting.")
            exit_code = 130 
        except ImportError as e_import:
            logger.critical(f"A required dependency could not be imported globally even after venv setup: {e_import}", exc_info=True)
            exit_code = 1
        except Exception as e_main:
            logger.critical(f"An unhandled exception occurred in _initialize_and_run: {e_main}", exc_info=True)
            exit_code = 1 
    else:
        logger.error("Failed to activate or confirm virtual environment. Exiting.")
        exit_code = 1
    
    logger.info(f"Script finished with exit code {exit_code}.")
    # USER_PROVIDED_TOKEN_CHECK_START
    print(f"__main__: Python sees HF_TOKEN: {os.environ.get('HF_TOKEN')}")
    # USER_PROVIDED_TOKEN_CHECK_END
    sys.exit(exit_code)

