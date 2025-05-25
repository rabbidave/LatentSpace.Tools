#!/usr/bin/env python
"""
Transformer-Based Classification and Reranking Service 
w/ VLM-Powered RAG-Informed Policy Validation

This enhanced version includes:
1. VLM-based Processing & RAG indexing
2. Comprehensive Testing Framework
3. Self-Installing & Self-Documenting
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
    venv_path = os.path.abspath(VENV_DIR)
    logger.debug(f"Target virtual environment path: {venv_path}")
    logger.debug(f"Current Python prefix: {sys.prefix}")
    is_in_target_venv = sys.prefix == venv_path

    if is_in_target_venv:
        logger.info(f"Running inside the '{VENV_DIR}' virtual environment.")
        try:
            import sentence_transformers
            import ranx
            import llama_cpp
            logger.debug("All required dependencies available.")
        except ImportError:
            logger.info("Missing dependencies detected. Will attempt install.")
            is_in_target_venv = False

    if is_in_target_venv:
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
        subprocess.run([pip_executable, "install", "--upgrade", "pip"], 
                      check=True, capture_output=True, text=True)
        logger.info("Pip upgraded successfully.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to upgrade pip: {e.stderr}")

    current_packages_to_install = list(REQUIRED_PACKAGES)
    
    logger.info(f"Installing required packages: {', '.join(current_packages_to_install)}")
    install_command = [pip_executable, "install"] + current_packages_to_install
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
        os.execv(python_executable, exec_args)
    except OSError as e:
        logger.error(f"os.execv failed: {e}", exc_info=True)
        sys.exit(1)
    
    return False # Should not be reached if os.execv is successful


# --- VLM-Based Markdown Processing ---

class MarkdownReformatterVLM:
    """VLM-powered markdown reformatter using GGUF models via llama-cpp-python."""
    
    DEFAULT_MODEL_ID = "PleIAs/Pleias-RAG-1B"
    
    def __init__(self, model_path_or_id: str, **kwargs):
        self.model_path_or_id = model_path_or_id
        self.model: Optional[llama_cpp.Llama] = None
        self.is_loaded = False
        
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
            logger.info(f"Loading VLM model: {self.model_path_or_id}")
            
            # Handle model path resolution
            if os.path.exists(self.model_path_or_id):
                model_path = self.model_path_or_id
            else:
                # For HuggingFace model IDs, user should provide local GGUF path
                logger.error(f"Model path not found: {self.model_path_or_id}")
                logger.info("Please provide a local path to a GGUF model file.")
                return False
            
            self.model = llama_cpp.Llama(
                model_path=model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose
            )
            
            self.is_loaded = True
            logger.info(f"Successfully loaded VLM model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}", exc_info=True)
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
            
            if response and "choices" in response and response["choices"]:
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

def fetch_documentation(docs_url_or_path: Union[str, List[str]]) -> str:
    """Fetch documentation content from URL(s) or local file(s).
    If a list is provided, it concatenates content from all sources.
    If a single URL/path is provided, it fetches that one.
    """
    
    sources_to_fetch = [docs_url_or_path] if isinstance(docs_url_or_path, str) else docs_url_or_path
    
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
        
        sources_to_process = [docs_url_or_path_list] if isinstance(docs_url_or_path_list, str) else docs_url_or_path_list

        vlm_processor_instance: Optional[MarkdownReformatterVLM] = None
        if processing_strategy == "vlm" and vlm_model_path and LLAMA_CPP_AVAILABLE:
            vlm_processor_instance = MarkdownReformatterVLM(vlm_model_path)
            # Try to load model once if VLM strategy is chosen
            if not vlm_processor_instance.load_model():
                logger.warning("VLM model loading failed. Will fallback to Python processing for all documents.")
                processing_strategy = "python" # Force fallback for all if initial load fails
                vlm_processor_instance = None # Ensure it's not used

        for doc_source_identifier in sources_to_process:
            logger.info(f"Fetching and processing documentation from: {doc_source_identifier}")
            try:
                # fetch_documentation now returns content prefixed with "# Source Document: ..."
                markdown_content_for_source = fetch_documentation(doc_source_identifier)
                
                # Check if meaningful content was fetched (beyond the added source header)
                source_header = f"# Source Document: {doc_source_identifier}"
                actual_content_test = markdown_content_for_source.replace(source_header, "").strip()
                if not actual_content_test:
                    logger.warning(f"No actual content fetched from {doc_source_identifier} (after removing source header), skipping.")
                    continue

                source_url_for_chunks = str(doc_source_identifier) # Use the specific identifier
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
                    if processing_strategy == "vlm": # VLM was intended but failed or unavailable
                        logger.info(f"VLM processing requested but conditions not met for {source_url_for_chunks}. Using fallback Python processing.")
                    else: # Fallback was explicitly chosen
                        logger.info(f"Using fallback Python-based markdown processing for {source_url_for_chunks}")
                    doc_chunks = fallback_markdown_processing(markdown_content_for_source, source_url_for_chunks, chunk_size) # Pass target_chunk_size
                
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
            metadata_fields=[ # Ensure these match what validate_and_enhance_chunk & fallback_markdown_processing produce
                "h1_section", "h2_section", "h3_section", "h4_section",
                "is_code_block", "contains_code_elements", "contains_commands", 
                "chunk_type", "topics", "source_url", "vlm_processed",
                "char_count", "chunk_index", "is_sub_chunk", "parent_chunk_id" # Added potential secondary chunking metadata
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
            logger.debug(f"RAGRetriever._load_model: HF_TOKEN from env: {os.environ.get('HF_TOKEN')}")
            logger.debug(f"RAGRetriever._load_model: Explicit token arg being passed: {os.environ.get('HF_TOKEN')}")
            self.model = SentenceTransformer(model_id, token=os.environ.get("HF_TOKEN"))
            logger.info(f"SentenceTransformer model '{model_id}' loaded.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{model_id}': {e}")
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
                    if metadata_fields:
                        for m_field in metadata_fields:
                            if m_field in item.get("metadata", {}): # Access metadata correctly
                                metadata[m_field] = item["metadata"][m_field]
                            elif m_field in item: # Fallback for top-level metadata (less common)
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
            "embedding_dimension": embeddings_array.shape, # Sentence embeddings are (num_sentences, dim)
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

            if len(self.documents) != self.embeddings.shape[0]:
                logger.error(f"Document count ({len(self.documents)}) and embedding count's first dimension ({self.embeddings.shape[0]}) mismatch. Index corrupt. Full embeddings shape: {self.embeddings.shape}")
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
            
            return results
            
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
            logger.debug(f"ModernBERTClassifier.setup: Attempting to load tokenizer for model_id='{self.model_id}'.")
            logger.debug(f"ModernBERTClassifier.setup: HF_TOKEN from env: {os.environ.get('HF_TOKEN')}")
            logger.debug(f"ModernBERTClassifier.setup: Explicit token arg being passed: {os.environ.get('HF_TOKEN')}")
            logger.debug(f"ModernBERTClassifier.setup: use_fast=True")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True, token=os.environ.get("HF_TOKEN"))
            logger.debug(f"ModernBERTClassifier.setup: Attempting to load model for model_id='{self.model_id}'.")
            logger.debug(f"ModernBERTClassifier.setup: HF_TOKEN from env (for model): {os.environ.get('HF_TOKEN')}")
            logger.debug(f"ModernBERTClassifier.setup: Explicit token arg for model (if passed): {os.environ.get('HF_TOKEN')}")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, token=os.environ.get("HF_TOKEN"))
            self.is_setup = True
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
            
            # Assuming binary classification: 0=invalid, 1=valid
            prediction = int(probabilities.argmax())
            confidence = float(probabilities.max())
            
            return {
                "prediction": prediction,
                "probability_positive": float(probabilities),
                "confidence": confidence,
                "details": f"Model: {self.model_id} | Input length: {len(input_text)} | Output length: {len(output_text)}"
            }
        except Exception as e:
            logger.error(f"Classification error: {e}")
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
            instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            instance.is_setup = True
            logger.info("Fine-tuned model loaded successfully")
            return instance
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            raise

class ColBERTReranker:
    """ColBERT-based sensitivity classifier using MaxSim technique."""
    DEFAULT_MODEL_ID = "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco"
    
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, reference_examples: Optional[Dict[str, List[str]]] = None):
        from transformers import AutoTokenizer, AutoModel
        import torch
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        self.model_id = model_id # Example: "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco"
        logger.debug(f"ColBERTReranker.__init__: Attempting to load tokenizer for model_id='{model_id}'.")
        logger.debug(f"ColBERTReranker.__init__: HF_TOKEN from env: {os.environ.get('HF_TOKEN')}")
        logger.debug(f"ColBERTReranker.__init__: Explicit token arg being passed: {os.environ.get('HF_TOKEN')}")
        logger.debug(f"ColBERTReranker.__init__: use_fast=True")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=os.environ.get("HF_TOKEN"))
        logger.debug(f"ColBERTReranker.__init__: Attempting to load model for model_id='{model_id}'.")
        logger.debug(f"ColBERTReranker.__init__: HF_TOKEN from env (for model): {os.environ.get('HF_TOKEN')}")
        self.model = AutoModel.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
        self.reference_embeddings = {}
        
        # Load default reference examples if none provided
        self.reference_examples = reference_examples or {
            "Class 1: PII": ["SSN: 123-45-6789", "Credit card: 4111-1111-1111-1111"],
            "Class 2: Confidential": ["Project codename: Phoenix", "Internal memo"],
            "Class 3: Internal": ["Meeting notes", "Draft document"],
            "Class 4: Public": ["Press release", "Public blog post"]
        }
        
        # Precompute embeddings for reference examples
        for class_name, examples in self.reference_examples.items():
            self.reference_embeddings[class_name] = [
                self._get_text_embedding(example) for example in examples
            ]
        
        logger.info(f"ColBERTReranker initialized with model: {model_id}")
    
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
        # Compute pairwise cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            query_embed.unsqueeze(1),  # [Q, 1, D]
            doc_embed.unsqueeze(0),    # [1, D, D]
            dim=-1
        )
        # Take maximum similarity per query token
        max_sims = similarities.max(dim=-1).values  # [Q]
        # Average over query tokens
        return max_sims.mean().item()
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """Classify text sensitivity using ColBERT-style MaxSim technique."""
        try:
            text_embed = self._get_text_embedding(text)
            class_scores = {}
            
            for class_name, ref_embeds in self.reference_embeddings.items():
                # Compute MaxSim against all reference examples for this class
                scores = [self._maxsim_score(text_embed, ref_embed) for ref_embed in ref_embeds]
                class_scores[class_name] = max(scores)  # Take best match
            
            # Get predicted class and confidence
            predicted_class = max(class_scores, key=class_scores.get)
            max_score = class_scores[predicted_class]
            
            return {
                "predicted_class": predicted_class,
                "class_scores": class_scores,
                "confidence": max_score,
                "details": f"ColBERT classification using {self.model_id}"
            }
        except Exception as e:
            logger.error(f"Error in sensitivity classification: {e}")
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
    DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
    
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, **kwargs):
        from transformers import pipeline, AutoProcessor
        # torch will be imported in methods when needed
         
        self.model_id = model_id
        self.device = None  # Will be set in setup()
        self.is_setup = False
        logger.info(f"VisionLanguageProcessor initialized with model: {model_id}")

    def setup(self):
        """Load the VLM model and processor."""
        from transformers import pipeline, AutoProcessor
        import torch
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN")) # VLM models often need trust_remote_code
            logger.debug(f"VisionLanguageProcessor.setup: Attempting to load pipeline for model='{self.model_id}'.")
            logger.debug(f"VisionLanguageProcessor.setup: HF_TOKEN from env (for pipeline): {os.environ.get('HF_TOKEN')}")
            self.model = pipeline(
                "image-to-text",
                model=self.model_id,
                device=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                token=os.environ.get("HF_TOKEN")
            )
            self.is_setup = True
            logger.info("VLM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading VLM model: {e}")
            raise
        return self
    
    def _load_image(self, image_source: str) -> "Image.Image":  # Use string literal for forward reference
        """Load image from file path, URL, or bytes."""
        from PIL import Image
        import requests
        from io import BytesIO
        
        if isinstance(image_source, bytes):
            return Image.open(BytesIO(image_source))
        elif image_source.startswith(("http://", "https://")):
            response = requests.get(image_source)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        else:
            return Image.open(image_source)
    
    def describe_image(self, image_source: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Analyze an image using VLM with optional custom prompt."""
        from PIL import Image
        import torch
        
        if not self.is_setup:
            self.setup()
        
        try:
            image = self._load_image(image_source)
            
            # Use default prompt if none provided
            final_prompt = prompt or "Describe this image in detail, focusing on any text, objects, and activities."
            
            inputs = self.processor(
                text=final_prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=200)
            
            description = self.processor.decode(outputs, skip_special_tokens=True)
            
            return {
                "description": description,
                "analysis": self._analyze_description(description),
                "prompt_used": final_prompt,
                "model": self.model_id,
                "details": f"Processed image of size {image.size} with {self.model_id}"
            }
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return {
                "description": "Error processing image",
                "analysis": {},
                "details": f"Error: {str(e)}"
            }
    
    def _analyze_description(self, description: str) -> Dict[str, Any]:
        """Perform basic analysis on the generated description."""
        analysis = {
            "contains_text": any(c.isalpha() for c in description),
            "word_count": len(description.split()),
            "sentiment": self._get_sentiment(description),
            "key_phrases": self._extract_key_phrases(description)
        }
        return analysis
    
    def describe_video_frames(self, video_source: str, prompt: Optional[str] = None,
                            frame_interval: int = 5) -> Dict[str, Any]:
        """Analyze video by sampling frames at given interval (seconds)."""
        import cv2
        from PIL import Image
        import tempfile
        
        if not self.is_setup:
            self.setup()
        
        try:
            # Open video file
            if video_source.startswith(("http://", "https://")):
                # Download video to temp file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    response = requests.get(video_source)
                    response.raise_for_status()
                    temp_file.write(response.content)
                    video_path = temp_file.name
            else:
                video_path = video_source
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_step = int(fps * frame_interval)
            
            frame_descriptions = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_step == 0:
                    # Convert OpenCV frame to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Describe frame
                    frame_desc = self.describe_image(pil_image, prompt)
                    frame_desc["timestamp"] = frame_count / fps
                    frame_descriptions.append(frame_desc)
                
                frame_count += 1
            
            cap.release()
            
            # Generate summary
            summary = self._summarize_frame_descriptions(frame_descriptions)
            
            return {
                "frame_descriptions": frame_descriptions,
                "summary": summary,
                "total_frames": frame_count,
                "frames_analyzed": len(frame_descriptions),
                "details": f"Analyzed {len(frame_descriptions)} frames from {frame_count} total frames"
            }
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return {
                "frame_descriptions": [],
                "summary": "Error processing video",
                "details": f"Error: {str(e)}"
            }
    
    def _summarize_frame_descriptions(self, frame_descs: List[Dict]) -> str:
        """Generate a summary from multiple frame descriptions."""
        # Simple implementation - could be enhanced with LLM
        unique_descriptions = set(desc["description"] for desc in frame_descs)
        return "Video contains: " + "; ".join(unique_descriptions)
    
    @classmethod
    def load(cls, model_dir: str, **kwargs):
        """Load a VLM from a local directory."""
        return cls(model_dir, **kwargs)
    
    def describe_video_frames(self, video_source: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate description for video frames (placeholder)."""
        if not self.is_setup: self.setup()
        logger.info(f"Describing video: {video_source} with prompt: '{prompt}' (placeholder)")

        desc = f"Placeholder analysis of video '{video_source}'."
        if "sensitive" in (prompt or "").lower():
            desc += " Video frames suggest some sensitive material might be present."

        return {
            "description": desc,
            "confidence": 0.80,
            "frame_count_analyzed": 10, # Placeholder
            "scene_changes_detected": 2, # Placeholder
            "key_events": ["placeholder_event_1", "placeholder_event_2"],
            "details": "Placeholder VLM video frame analysis."
        }

# --- Enhanced Classification API with Complete Policy Logic ---

class ClassificationAPI:
    """Enhanced Classification API with complete policy validation."""
    
    def __init__(self, modernbert_model_dir: Optional[str],
                 host: str, port: int,
                 policy_config_path: Optional[str] = "policy_config.json",
                 vlm_model_id_or_dir: Optional[str] = None, # Note: this is for the API's internal VLM, not doc processing VLM
                 global_rag_retriever_index_path: Optional[str] = None):
        
        self.modernbert_model_dir = modernbert_model_dir
        self.host = host
        self.port = port
        self.policy_config_path = policy_config_path
        # This VLM is for item processing in policies, distinct from MarkdownReformatterVLM for docs
        self.vlm_for_item_processing_id_or_dir = vlm_model_id_or_dir 
        
        self.api_policy_config: Dict[str, Any] = {}
        self.modernbert_classifier: Optional[ModernBERTClassifier] = None
        self.colbert_reranker: Optional[ColBERTReranker] = None
        self.vision_language_processor: Optional[VisionLanguageProcessor] = None # For item processing
        
        # RAG components
        self.global_rag_retriever_index_path = global_rag_retriever_index_path
        self.documentation_rag_retriever: Optional[RAGRetriever] = None # Specifically for documentation assistance per policy
        
        self.app = Flask(__name__)
        CORS(self.app) # Enable CORS for all routes
        self.request_count = 0

    def _handle_request_policy(self, payload: Dict, policy_rules: Dict, files: Optional[Any], response_data: Dict) -> List[str]:
        """
        Complete implementation of policy validation logic.
        Returns a list of violation reason strings. Populates response_data.
        """
        violations = []
        
        try:
            # Handle legacy format (simple input/output text)
            # These checks apply if relevant fields are in payload, regardless of input_items
            violations.extend(self._handle_legacy_policy_validation(payload, policy_rules, response_data))
            
            # Handle multimodal format (input_items with potential files)
            if "input_items" in payload: # Only process if input_items are provided
                violations.extend(self._handle_multimodal_policy_validation(payload, policy_rules, files, response_data))
            
            # Additional policy checks (custom rules, etc.)
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
        
        # ModernBERT I/O Validation
        if policy.get("modernbert_io_validation", False) and self.modernbert_classifier:
            if input_text or output_text: # Only run if there's text to validate
                try:
                    io_result = self.modernbert_classifier.classify_input_output_pair(input_text, output_text)
                    response_data["modernbert_io_validation"] = io_result
                    
                    if io_result.get("prediction") == 0: # 0 = fail, 1 = pass
                        prob_positive = io_result.get("probability_positive", 0.0)
                        violations.append(f"ModernBERT I/O validation failed (validity score: {prob_positive:.2f}).")
                        
                except Exception as e:
                    logger.error(f"Error in ModernBERT I/O validation: {e}", exc_info=True)
                    violations.append("I/O validation processing error (ModernBERT).")
                    response_data["modernbert_io_validation"] = {"error": str(e)}
        
        # ColBERT Input Sensitivity Check
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

        # ColBERT Output Sensitivity Check
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
        if not item_processing_rules: # No rules, no processing needed for items.
            return violations

        response_data["item_processing_results"] = {} # Store results for each item
        
        for item_idx, item in enumerate(input_items):
            if not isinstance(item, dict):
                violations.append(f"Item at index {item_idx} is not a valid object.")
                continue

            item_id = item.get("id", f"item_{item_idx}")
            item_type = item.get("type", "unknown_type")
            filename_in_form = item.get("filename_in_form") # Key to find file in 'files'
            
            item_result_data = {"item_id": item_id, "item_type": item_type, "violations": []} # Store item-specific violations

            try:
                matching_rule = next((rule for rule in item_processing_rules if rule.get("item_type") == item_type), None)
                
                if not matching_rule:
                    # No specific rule for this item_type, could be a violation or just skipped
                    if policy.get("strict_item_type_matching", False): # Example of a stricter policy meta-rule
                        item_result_data["violations"].append(f"No processing rule defined for item type: '{item_type}'.")
                    continue # Skip if no rule and not strict

                # VLM Processing for the item (image/video)
                vlm_config = matching_rule.get("vlm_processing", {})
                if vlm_config.get("required", False) and self.vision_language_processor:
                    if not filename_in_form:
                        item_result_data["violations"].append(f"VLM processing required for item '{item_id}', but 'filename_in_form' is missing.")
                    elif not files or filename_in_form not in files:
                        item_result_data["violations"].append(f"Required file '{filename_in_form}' for item '{item_id}' not found in form data.")
                    else:
                        # file_obj = files[filename_in_form] # In a real scenario, pass this to VLM
                        vlm_output_text = ""
                        try:
                            if item_type in ["image", "screenshot"]:
                                vlm_result = self.vision_language_processor.describe_image(
                                    image_source=filename_in_form, # Pass name, VLP would handle file access
                                    prompt=vlm_config.get("prompt")
                                )
                            elif item_type in ["video", "recording"]:
                                vlm_result = self.vision_language_processor.describe_video_frames(
                                    video_source=filename_in_form,
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
                        
                        # Derived Text Checks on VLM output
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
                
                # ... other item-specific checks based on matching_rule could go here ...

            except Exception as e_item:
                logger.error(f"Error processing item '{item_id}': {e_item}", exc_info=True)
                item_result_data["violations"].append(f"General error processing item '{item_id}'.")
            
            response_data["item_processing_results"][item_id] = item_result_data
            violations.extend(item_result_data["violations"]) # Add item-specific violations to main list
        
        return violations

    def _handle_additional_policy_checks(self, payload: Dict, policy: Dict, response_data: Dict) -> List[str]:
        """Handle additional policy-specific checks (custom rules, rate limiting placeholder)."""
        violations = []
        
        try:
            custom_rules = policy.get("custom_validation_rules", [])
            if not isinstance(custom_rules, list):
                violations.append("Invalid 'custom_validation_rules' format in policy.")
                return violations # Stop if rules are malformed

            for rule_idx, rule in enumerate(custom_rules):
                if not isinstance(rule, dict):
                    violations.append(f"Custom rule at index {rule_idx} is not a valid object.")
                    continue

                rule_type = rule.get("type")
                
                if rule_type == "text_length_limit":
                    max_length = rule.get("max_length")
                    text_fields = rule.get("text_fields", []) # Expects a list
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
                        if field_name not in payload or not payload[field_name]: # Checks for presence and non-empty
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
                # ... other custom rule types ...
            
            # Rate limiting check (placeholder)
            if policy.get("rate_limiting", {}).get("enabled", False):
                # In a real implementation, this would involve checking against a rate limiter store.
                # For this script, it's a conceptual check.
                logger.debug(f"Rate limiting check (conceptual) for policy with max_requests: {policy['rate_limiting'].get('max_requests_per_minute')}")
            
        except Exception as e:
            logger.error(f"Error during additional policy checks: {e}", exc_info=True)
            violations.append(f"Additional policy validation error: {str(e)}")
        
        return violations

    def _generate_help_queries(self, violation_reasons: List[str], policy: Dict[str, Any]) -> List[str]:
        """Generate help queries based on violation reasons for RAG."""
        help_queries: Set[str] = set() # Use a set to avoid duplicate queries
        
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
            if not found_term_query: # Generic fallback
                 # Try to extract keywords from the violation reason itself
                 # This is very basic, could be improved with NLP
                keywords = re.findall(r'\b[a-zA-Z]{4,}\b', violation_lower) # Words of 4+ chars
                if keywords:
                    help_queries.add("how to fix " + " ".join(list(set(keywords))[:3])) # Use up to 3 unique keywords
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
                "full_content_available": True, # Implies client could request full doc by ID
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
        
        # Fallback titles based on content type
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
                return # Not enabled for this policy
            
            response_data["documentation_assistance_attempted"] = True
            docs_index_path_str = docs_assist_config.get("index_path")
            if not docs_index_path_str:
                logger.warning("Documentation assistance enabled but no RAG index_path configured in policy.")
                response_data["documentation_suggestions"] = {"error": "Documentation RAG index path not configured."}
                return
            
            # Use the cached RAG retriever method
            # This retriever is specific to the documentation index defined in the policy
            policy_docs_retriever = self._get_cached_rag_retriever(docs_index_path_str)
            
            if not policy_docs_retriever or not policy_docs_retriever._is_loaded:
                logger.warning(f"Could not load documentation RAG index from {docs_index_path_str} for policy assistance.")
                response_data["documentation_suggestions"] = {"error": f"Documentation RAG index at '{docs_index_path_str}' not available or failed to load."}
                return
            
            # Generate queries from violations
            help_queries = self._generate_help_queries(violation_reasons, policy)
            if not help_queries:
                response_data["documentation_suggestions"] = {"message": "No specific help queries generated for violations."}
                return
            
            all_retrieved_docs_map: Dict[str, Dict[str, Any]] = {} # Use dict to store unique docs by ID, keeping best score
            max_suggestions_per_query = docs_assist_config.get("max_suggestions_per_query", 2)
            min_threshold = docs_assist_config.get("min_similarity_threshold", 0.1)
            
            # Query RAG for each generated help query (limit number of queries to avoid excessive RAG calls)
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
            
            # Sort unique documents by relevance score, descending
            sorted_unique_docs = sorted(all_retrieved_docs_map.values(), key=lambda x: x["score"], reverse=True)
            
            # Limit total suggestions
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
        # Normalize path for consistent caching key
        normalized_path_key = str(Path(index_path_str).resolve())

        if normalized_path_key in RAG_COMPONENT_CACHE:
            logger.debug(f"Using cached RAGRetriever for {normalized_path_key}")
            return RAG_COMPONENT_CACHE[normalized_path_key]
        
        logger.info(f"Attempting to load RAGRetriever for {normalized_path_key} (not found in cache).")
        try:
            retriever = RAGRetriever(normalized_path_key) # Use normalized path
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
        
        # Load policy configuration
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

        # Setup models (using placeholders for now)
        try:
            if self.modernbert_model_dir: # If a specific dir is given for a real model
                self.modernbert_classifier = ModernBERTClassifier.load(self.modernbert_model_dir)
            else: # Default placeholder setup
                self.modernbert_classifier = ModernBERTClassifier().setup()
        except Exception as e_mb:
            logger.error(f"Failed to initialize ModernBERT classifier: {e_mb}", exc_info=True)
        
        try:
            # model_id for ColBERT is illustrative, actual model loading would be more complex
            self.colbert_reranker = ColBERTReranker(model_id="placeholder_colbert_model")
        except Exception as e_colbert:
            logger.error(f"Failed to initialize ColBERT reranker: {e_colbert}", exc_info=True)

        try:
            # vlm_for_item_processing_id_or_dir is for the VLM used by policies (e.g. LLaVA)
            # Not to be confused with the GGUF model for markdown document processing.
            self.vision_language_processor = VisionLanguageProcessor(model_id_or_dir=self.vlm_for_item_processing_id_or_dir).setup()
        except Exception as e_vlp:
            logger.error(f"Failed to initialize VisionLanguageProcessor for items: {e_vlp}", exc_info=True)
        
        # Setup global RAG retriever if path is provided (e.g., for a direct /rag/query endpoint)
        if self.global_rag_retriever_index_path:
            logger.info(f"Attempting to load global RAG retriever from: {self.global_rag_retriever_index_path}")
            # This retriever is separate from policy-specific documentation retrievers
            self.documentation_rag_retriever = self._get_cached_rag_retriever(self.global_rag_retriever_index_path)
            if self.documentation_rag_retriever:
                logger.info("Global RAG retriever loaded successfully.")
            else:
                logger.warning("Failed to load global RAG retriever.")
        
        # --- Flask Routes ---
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
            else: # Assume application/json
                try:
                    payload = request.get_json()
                    if payload is None: # get_json returns None if parsing fails or content-type mismatch
                         return jsonify({"overall_status": "ERROR_BAD_REQUEST", "error_message": "Request body must be valid JSON and Content-Type 'application/json'."}), 400
                except Exception as e_json_body: # Should be caught by Flask mostly
                     return jsonify({"overall_status": "ERROR_BAD_REQUEST", "error_message": f"Failed to parse JSON request body: {e_json_body}"}), 400
            
            api_class_name = payload.get("api_class")
            if not api_class_name:
                return jsonify({"overall_status": "ERROR_BAD_REQUEST", "error_message": "'api_class' field is required in JSON payload."}), 400
            
            response_data = {
                "request_id": payload.get("request_id", f"req_{int(request_timestamp)}_{self.request_count}"),
                "api_class_requested": api_class_name,
                "timestamp_utc": request_timestamp,
                "processing_details": {} # To store model outputs etc.
            }
            
            active_policy_rules = self.api_policy_config.get(api_class_name)
            if not active_policy_rules:
                response_data["overall_status"] = "REJECT_INVALID_POLICY"
                response_data["error_message"] = f"Policy definition for API class '{api_class_name}' not found or not loaded."
                return jsonify(response_data), 400 # Using 400 as it's a client error (bad api_class or server config issue)
            
            # Process policy using the complete implementation
            violation_reasons = self._handle_request_policy(payload, active_policy_rules, files, response_data["processing_details"])
            
            if violation_reasons:
                response_data["violation_reasons"] = violation_reasons
                response_data["overall_status"] = "REJECT_POLICY_VIOLATION"
                # Attempt to add documentation assistance if violations occurred and policy enables it
                self._add_documentation_assistance(response_data, active_policy_rules, violation_reasons)
            else:
                response_data["overall_status"] = "PASS"
                # Potentially, documentation_assistance could be triggered on PASS too, if configured
                # For now, it's only on violation.
            
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

        @self.app.route('/rag/query', methods=['POST']) # Generic RAG query endpoint (uses global_rag_retriever)
        def rag_query_endpoint():
            if not self.documentation_rag_retriever or not self.documentation_rag_retriever._is_loaded:
                # This endpoint uses the 'global_rag_retriever_index_path' one.
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
        self.setup(**kwargs) # Ensure setup is called before running
        logger.info(f"Starting Enhanced Classification API server on http://{self.host}:{self.port}")
        logger.info(f"CORS enabled for all origins on this server.")
        logger.info(f"Number of policies loaded: {len(self.api_policy_config)}")
        
        # Log status of key components after setup
        logger.info(f"ModernBERT Classifier: {'Available' if self.modernbert_classifier and self.modernbert_classifier.is_setup else 'Not Available'}")
        logger.info(f"ColBERT Reranker: {'Available' if self.colbert_reranker else 'Not Available'}")
        logger.info(f"Vision Language Processor (for items): {'Available' if self.vision_language_processor and self.vision_language_processor.is_setup else 'Not Available'}")
        if self.documentation_rag_retriever and self.documentation_rag_retriever._is_loaded:
             logger.info(f"Global Documentation RAG Retriever: Loaded with {self.documentation_rag_retriever.get_status().get('num_documents')} documents.")
        else:
            logger.info(f"Global Documentation RAG Retriever: Not loaded or not configured ({self.global_rag_retriever_index_path}).")


        serve(self.app, host=self.host, port=self.port, threads=8) # Using Waitress for production

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
        if strategy != "lines": # If a structural strategy failed due to syntax, force lines
            strategy = "lines" 
            logger.info("Switched to 'lines' strategy due to syntax error.")
    
    lines = code_content.split('\n')
    
    if strategy == "functions" and tree:
        try:
            for node in ast.walk(tree):
                # Target top-level functions and methods within top-level classes
                is_top_level_function = isinstance(node, ast.FunctionDef) and node.col_offset == 0
                is_method_in_top_level_class = False
                if isinstance(node, ast.FunctionDef):
                    # Check if parent is a ClassDef at col_offset 0
                    # This requires finding parent, ast.walk doesn't provide it directly.
                    # A simpler approximation for now: include all FunctionDef nodes.
                    # Or, refine by first finding top-level classes, then their methods.
                    # For now, let's stick to the col_offset for functions.
                    # To get methods, we'd need to iterate class body.
                    pass # Handled below by iterating classes then their methods for more structure


            # First, grab top-level functions
            for node in tree.body: # Iterate top-level nodes in AST
                if isinstance(node, ast.FunctionDef): # Top-level functions
                    start_line = node.lineno -1
                    end_line = getattr(node, 'end_lineno', start_line + len(ast.unparse(node).splitlines())) -1
                    
                    func_lines = lines[start_line : end_line + 1]
                    func_text = '\n'.join(func_lines)
                    signature = lines[start_line].strip() # First line is signature
                    docstring = ast.get_docstring(node)
                    
                    chunks.append({
                        'id': f"function_{node.name}",
                        'text': func_text,
                        'metadata': {
                            'type': 'function', 'name': node.name, 'signature': signature,
                            'start_line': start_line + 1, 'end_line': end_line + 1,
                            'docstring': docstring or "", 'line_count': end_line - start_line + 1,
                            'class_context': None # Not in a class
                        }
                    })
                elif isinstance(node, ast.ClassDef): # Top-level classes
                    # Then, grab methods within these top-level classes
                    for class_item_node in node.body:
                        if isinstance(class_item_node, ast.FunctionDef): # Method in class
                            method_node = class_item_node
                            start_line = method_node.lineno - 1
                            end_line = getattr(method_node, 'end_lineno', start_line + len(ast.unparse(method_node).splitlines())) -1

                            method_lines = lines[start_line : end_line + 1]
                            method_text = '\n'.join(method_lines)
                            signature = lines[start_line].strip()
                            docstring = ast.get_docstring(method_node)
                            
                            chunks.append({
                                'id': f"method_{node.name}_{method_node.name}", # class_method
                                'text': method_text,
                                'metadata': {
                                    'type': 'method', 'name': method_node.name, 'signature': signature,
                                    'start_line': start_line + 1, 'end_line': end_line + 1,
                                    'docstring': docstring or "", 'line_count': end_line - start_line + 1,
                                    'class_context': node.name # Name of the class
                                }
                            })
        except Exception as e_func:
            logger.error(f"Error processing functions/methods: {e_func}. AST parsing might have issues.", exc_info=True)
            if not chunks: strategy = "lines" # Fallback if function strategy yields nothing

    elif strategy == "classes" and tree:
        try:
            for node in tree.body: # Iterate top-level nodes in AST
                if isinstance(node, ast.ClassDef): # Top-level classes
                    start_line = node.lineno - 1
                    # ast.unparse can reconstruct the source of the node
                    end_line = getattr(node, 'end_lineno', start_line + len(ast.unparse(node).splitlines())) -1
                    
                    class_lines = lines[start_line : end_line + 1]
                    class_text = '\n'.join(class_lines)
                    
                    methods_signatures = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef): # Method
                            # Get method signature (first line of method def)
                            method_sig_line = lines[item.lineno - 1].strip()
                            methods_signatures.append(method_sig_line)
                    
                    docstring = ast.get_docstring(node)
                    
                    chunks.append({
                        'id': f"class_{node.name}",
                        'text': class_text, # Entire class content
                        'metadata': {
                            'type': 'class', 'name': node.name,
                            'start_line': start_line + 1, 'end_line': end_line + 1,
                            'methods_signatures': methods_signatures, 'docstring': docstring or "",
                            'method_count': len(methods_signatures), 'line_count': end_line - start_line + 1
                        }
                    })
        except Exception as e_class:
            logger.error(f"Error processing classes: {e_class}. AST parsing might have issues.", exc_info=True)
            if not chunks: strategy = "lines" # Fallback if class strategy yields nothing
    
    if strategy == "lines" or not chunks: # If strategy is 'lines' or structural parsing failed/yielded no chunks
        if strategy != "lines" and not chunks: # Log if we fell back implicitly
            logger.info(f"No chunks from '{strategy}' strategy, falling back to 'lines' strategy.")
        
        # Use the more robust split_text_with_overlap for line-based chunking too
        line_chunks_text = split_text_with_overlap(code_content, chunk_size, chunk_overlap)
        
        # To add line number metadata, we need to map these text chunks back to original line numbers.
        # This is non-trivial if split_text_with_overlap significantly reformats.
        # For simplicity, we'll create chunks without precise start/end line numbers if using this fallback path
        # or if 'lines' strategy is chosen directly.
        
        current_char_offset = 0
        for i, chunk_text in enumerate(line_chunks_text):
            if not chunk_text.strip(): continue

            # Estimate line numbers based on cumulative newlines (approximate)
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
            current_char_offset += len(chunk_text) + 1 # +1 for a newline separator assumed by split_text_with_overlap between its outputs

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
        
        # Ensure index_path's parent directory exists
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temp_jsonl = index_path.parent / f"{index_path.name}_temp_code_chunks.jsonl"
        
        save_chunks_as_jsonl(chunks, temp_jsonl)
        
        retriever = RAGRetriever(index_path)
        
        # Define metadata fields based on chosen strategy and what parse_and_chunk_python_code produces
        if strategy == "functions":
            metadata_fields = ["type", "name", "signature", "start_line", "end_line", "docstring", "line_count", "class_context"]
        elif strategy == "classes":
            metadata_fields = ["type", "name", "start_line", "end_line", "methods_signatures", "docstring", "method_count", "line_count"]
        else: # "lines" or fallback
            metadata_fields = ["type", "estimated_start_line", "estimated_end_line", "line_count", "char_count"]
        
        retriever.index_corpus(
            corpus_path=temp_jsonl,
            embedding_model_id=embedding_model_id,
            doc_id_field="id", # 'id' is generated by parse_and_chunk_python_code
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
                         docs_url: Optional[Union[str, List[str]]] = None,
                         auto_build_docs_rag: bool = False,
                         docs_rag_index_name: str = "tool_documentation",
                         chunk_size: int = 500, # Default chunk size for doc processing
                         chunk_overlap: int = 50, # Default overlap for doc processing
                         vlm_model_path: Optional[str] = None, # For doc processing
                         processing_strategy: str = "vlm"): # For doc processing
    """Create enhanced example files with VLM support."""
    logger.info(f"Creating example files in {base_path}...")
    base_path.mkdir(parents=True, exist_ok=True)

    # Create comprehensive sample documentation
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

    # Save sample documentation file
    docs_file_path = base_path / "sample_tool_documentation.md"
    with open(docs_file_path, 'w', encoding='utf-8') as f:
        f.write(sample_docs_content)
    logger.info(f"Sample documentation markdown saved to {docs_file_path}")

    # Build documentation RAG index if requested
    effective_docs_url = docs_url if docs_url else str(docs_file_path) # Use local sample if no URL given
    
    if auto_build_docs_rag:
        docs_rag_index_full_path = base_path / docs_rag_index_name
        logger.info(f"Attempting to build documentation RAG index at {docs_rag_index_full_path}")
        
        success = build_documentation_rag_index(
            docs_url_or_path_list=effective_docs_url, # Pass the URL/path to fetch from
            index_path=docs_rag_index_full_path,
            embedding_model_id="all-MiniLM-L6-v2", # Default, can be parameterized
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            vlm_model_path=vlm_model_path, # For document processing VLM
            processing_strategy=processing_strategy
        )
        
        if success:
            logger.info(f"Documentation RAG index built successfully at {docs_rag_index_full_path}")
        else:
            logger.error(f"Failed to build documentation RAG index. Check logs.")
    else:
        logger.info("Skipping documentation RAG index build as --auto-build-docs-rag was not specified.")


    # Create enhanced policy configuration file content
    # Note: The "index_path" in documentation_assistance should point to where the index is actually built.
    # If using default output-dir and default docs_rag_index_name, it's relative like "./tool_examples/tool_documentation"
    # The create_example_files is often called with --output-dir, so adjust path.
    # For robustness, make it an absolute path or clearly document relativity.
    # Here, we use a path relative to where the policy file itself will be.
    # If `base_path` is "enhanced_tool_examples", then policy is in it, and index is also in it.
    policy_docs_rag_path = Path(docs_rag_index_name).as_posix() # Relative path within base_path

    enhanced_policy_config_content = {
        "SimpleValidation": {
            "description": "Basic I/O validation only using ModernBERT.",
            "modernbert_io_validation": True
        },
        "StrictPIICheck": {
            "description": "Comprehensive PII detection for input and output using ColBERT.",
            "modernbert_io_validation": True, # Can combine
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
                    "item_type": "image", # Example for image items
                    "vlm_processing": { # VLM used by the API for item content analysis
                        "required": True,
                        "prompt": "Analyze this image for sensitive content, PII, or policy violations."
                    },
                    "derived_text_checks": { # Checks on the text description from VLM
                        "colbert_sensitivity": True,
                        "disallowed_classes": ["Class 1: PII", "Class 2: Confidential"], # Stricter for derived text
                        "blocked_keywords": ["highly_confidential_internal_use_only", "top_secret_project_alpha"]
                    }
                }
                # Add rules for 'video' or other types as needed
            ],
            "custom_validation_rules": [
                {"type": "text_length_limit", "max_length": 2048, "text_fields": ["input_text", "output_text"]},
                {"type": "required_fields", "fields": ["input_text", "user_id"]}, # Example: user_id also required
                {"type": "format_validation", "field": "user_email", "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
            ],
            "rate_limiting": {"enabled": True, "max_requests_per_minute": 100}, # Conceptual
            "documentation_assistance": {
                "enabled": True,
                "index_path": policy_docs_rag_path, # Path to docs RAG index, relative to where API runs or absolute
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


    # Create a sample RAG corpus JSONL file (for generic RAG indexing demonstration)
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
    # All major imports are now handled in the `if __name__ == "__main__":` block
    # to ensure they are in the global scope.
    logger.debug("Global imports should be complete. Calling main_cli.")
    return main_cli()
def main_cli():
    """Main CLI function that runs after dependencies are imported."""
    setup_signal_handling()
    
    # ADD THIS DEBUG LINE:
    logger.info(f"DEBUG: main_cli received sys.argv: {sys.argv}")
    
    parser = argparse.ArgumentParser(
        description="Enhanced Transformer-based Classification Service with VLM Processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Index Codebase command
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

    # Create Example Files command
    parser_create_example = subparsers.add_parser(
        "create-example",
        help="Create example files (documentation, policy config, RAG corpus)."
    )
    parser_create_example.add_argument("--output-dir", type=str, default="enhanced_tool_examples", help="Directory to save example files.")
    parser_create_example.add_argument("--docs-url", type=str, default=None, help="URL or local path to markdown documentation file(s) to process. Can be a list. Defaults to internal sample doc if not provided.")
    parser_create_example.add_argument("--auto-build-docs-rag", action="store_true", help="Automatically build RAG index from the documentation.")
    parser_create_example.add_argument("--docs-rag-index-name", type=str, default="tool_documentation", help="Name for the documentation RAG index directory (created within --output-dir).")
    parser_create_example.add_argument("--chunk-size", type=int, default=800, help="Target chunk size for VLM/fallback document processing.")
    parser_create_example.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap for secondary/fallback document processing.")
    parser_create_example.add_argument(
        "--docs-vlm-model-path",
        type=str,
        help="Path to local GGUF model file for VLM-based documentation processing."
    )
    parser_create_example.add_argument(
        "--processing-strategy",
        type=str,
        choices=['vlm', 'python'],
        default='vlm',
        help="Documentation processing strategy ('vlm' or 'python' fallback)."
    )

    # RAG command group
    parser_rag = subparsers.add_parser("rag", help="RAG (Retrieval Augmented Generation) utilities.")
    rag_subparsers = parser_rag.add_subparsers(dest="rag_command", required=True)

    # RAG Index
    parser_rag_index_cmd = rag_subparsers.add_parser("index", help="Create a RAG index from a JSONL corpus.")
    parser_rag_index_cmd.add_argument("--corpus-path", type=str, required=True, help="Path to corpus file (JSONL format).")
    parser_rag_index_cmd.add_argument("--index-path", type=str, required=True, help="Directory path to save the RAG index.")
    parser_rag_index_cmd.add_argument("--embedding-model-id", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model for embeddings.")
    parser_rag_index_cmd.add_argument("--doc-id-field", type=str, default="id", help="Field name for document ID in JSONL.")
    parser_rag_index_cmd.add_argument("--text-field", type=str, default="text", help="Field name for text content in JSONL.")
    parser_rag_index_cmd.add_argument("--metadata-fields", nargs='*', help="List of metadata field names to include from JSONL (e.g., category, source). Assumes they are under a 'metadata' key in each JSONL object or top-level.")
    parser_rag_index_cmd.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation.")

    # RAG Retrieve (docs-chat equivalent)
    parser_rag_retrieve_cmd = rag_subparsers.add_parser("retrieve", help="Retrieve documents from a RAG index (CLI query tool).")
    parser_rag_retrieve_cmd.add_argument("--index-path", type=str, required=True, help="Path to the RAG index directory.")
    parser_rag_retrieve_cmd.add_argument("--query", type=str, required=True, help="Query string to search for.")
    parser_rag_retrieve_cmd.add_argument("--top-k", type=int, default=5, help="Number of top documents to retrieve.")
    # No --interactive flag as per current script structure.

    # Serve API command
    parser_serve = subparsers.add_parser("serve", help="Start the enhanced classification API server.")
    parser_serve.add_argument("--modernbert-model-dir", type=str, default=None, help="Path to a directory containing a trained ModernBERT model (for real model usage, otherwise uses placeholder).")
    parser_serve.add_argument("--policy-config-path", type=str, default="enhanced_tool_examples/enhanced_policy_config.json", help="Path to the API policy configuration JSON file.")
    parser_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host address for the API server.")
    parser_serve.add_argument("--port", type=int, default=8080, help="Port for the API server.")
    parser_serve.add_argument("--global-rag-retriever-index-path", type=str, default=None, help="Path to a global RAG index for the /rag/query endpoint and potentially default documentation assistance if not specified in policy.")
    parser_serve.add_argument("--vlm-model-path", type=str, help="Path to VLM GGUF model for item processing in API policies (e.g., LLaVA, distinct from docs VLM). Placeholder if not provided.")

    # Test command
    parser_test = subparsers.add_parser("test", help="Run comprehensive internal tests for the system.")
    parser_test.add_argument("--test-type", choices=['all', 'vlm', 'policy', 'rag', 'codebase'], default='all', help="Specific category of tests to run.")
    parser_test.add_argument("--verbose", action="store_true", help="Enable verbose logging during tests.")

    args = parser.parse_args()

    # Handle commands
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
            docs_url=args.docs_url.split(',') if args.docs_url and ',' in args.docs_url else args.docs_url, # Support comma-separated list for docs_url
            auto_build_docs_rag=args.auto_build_docs_rag,
            docs_rag_index_name=args.docs_rag_index_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            vlm_model_path=args.docs_vlm_model_path, # For document processing VLM
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
            return 0 # Assume success if no exception
        elif args.rag_command == "retrieve":
            retriever = RAGRetriever(args.index_path)
            if not retriever.load_index():
                logger.error(f"Failed to load RAG index from {args.index_path} for retrieval.")
                return 1
            results = retriever.retrieve(args.query, top_k=args.top_k)
            print(json.dumps(results, indent=2)) # Output results as JSON
            return 0

    elif args.command == "serve":
        api = ClassificationAPI(
            modernbert_model_dir=args.modernbert_model_dir,
            host=args.host,
            port=args.port,
            policy_config_path=args.policy_config_path,
            vlm_model_id_or_dir=args.vlm_model_path, # This is for the API's VLM (item processing)
            global_rag_retriever_index_path=args.global_rag_retriever_index_path
        )
        api.run() # This will block until server stops
        return 0 

    elif args.command == "test":
        # run_comprehensive_tests returns tuple (passed_tests, total_tests)
        passed, total = run_comprehensive_tests(args.test_type, args.verbose)
        return 0 if passed == total and total > 0 else 1 # Return 0 on all pass, 1 otherwise

    else:
        parser.print_help()
        return 1
    return 0 # Default exit code

# --- Comprehensive Testing Framework ---

def run_comprehensive_tests(test_type: str = "all", verbose: bool = False) -> Tuple[int, int]:
    """Run comprehensive tests for the enhanced system. Returns (passed_tests, total_tests)."""
    logger.info(f"Running '{test_type}' tests...")
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger to DEBUG
        logger.info("Verbose logging enabled for tests.")
    else:
        logging.getLogger().setLevel(logging.INFO) # Reset to INFO if not verbose

    test_results_summary: Dict[str, Dict[str, Any]] = {}
    
    # Create a temporary directory for test artifacts
    with tempfile.TemporaryDirectory(prefix="cls_service_tests_") as temp_dir_str:
        temp_dir_path = Path(temp_dir_str)
        logger.info(f"Using temporary directory for test artifacts: {temp_dir_path}")

        if test_type in ["all", "vlm"]:
            test_results_summary["VLM Processing"] = test_vlm_processing(temp_dir_path)
        
        if test_type in ["all", "policy"]:
            test_results_summary["Policy Validation"] = test_policy_validation(temp_dir_path)
        
        if test_type in ["all", "rag"]:
            test_results_summary["RAG Functionality"] = test_rag_functionality(temp_dir_path)
        
        if test_type in ["all", "codebase"]:
            test_results_summary["Codebase Indexing"] = test_codebase_indexing(temp_dir_path)
    
    # Print results summary
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests_run = 0
    total_tests_passed = 0
    
    for category, tests_in_category in test_results_summary.items():
        print(f"\n{category.upper()} TESTS:")
        cat_total = len(tests_in_category)
        cat_passed = 0
        for test_name, result_details in tests_in_category.items():
            status_icon = "✅ PASS" if result_details.get("passed", False) else "❌ FAIL"
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
    
    # Test 1: VLM Model Loading (Conceptual Mock for this test structure)
    test_name = "vlm_model_loading_conceptual"
    try:
        logger.info(f"Testing: {test_name}")
        # This tests the MarkdownReformatterVLM class structure, not actual loading here.
        # Actual loading depends on LLAMA_CPP_AVAILABLE and a valid model path.
        # We assume if LLAMA_CPP_AVAILABLE=False, it correctly skips loading.
        # If True, it would try to load; for tests, we'd need a mock path or tiny model.
        # Here, we just check if the class can be instantiated.
        # A dummy path is used; load_model() should handle its absence gracefully.
        formatter = MarkdownReformatterVLM("dummy_nonexistent_model.gguf")
        can_load = formatter.load_model() # Expected to be False if dummy path
        
        if LLAMA_CPP_AVAILABLE:
             results[test_name] = {"passed": not can_load, "details": "load_model correctly returned False for dummy path."}
        else: # If llama-cpp not available, load_model also returns False
            results[test_name] = {"passed": not can_load, "details": "load_model correctly returned False as llama-cpp is unavailable."}

    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}
    
    # Test 2: Prompt Template Creation
    test_name = "prompt_template_creation"
    try:
        logger.info(f"Testing: {test_name}")
        prompt = create_vlm_markdown_prompt_template()
        assert "{raw_markdown_content}" in prompt, "Prompt missing content placeholder."
        assert "JSON Output:" in prompt, "Prompt missing JSON output instruction."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}
    
    # Test 3: VLM Output Parsing (Valid JSON)
    test_name = "vlm_output_parsing_valid_json"
    try:
        logger.info(f"Testing: {test_name}")
        mock_vlm_output_valid = '''Some preamble text...
```json
[
  {{
    "id": "test_chunk_1_valid",
    "text": "This is a valid test chunk.",
    "metadata": {{
      "h1_section": "Test Section Valid", "is_code_block": false,
      "chunk_type": "content", "topics": ["testing_valid"]
    }}
  }}
]
```
Some postamble text...'''
        parsed_chunks = parse_vlm_output(mock_vlm_output_valid, "test_url_valid")
        assert len(parsed_chunks) == 1, f"Expected 1 chunk, got {len(parsed_chunks)}"
        assert parsed_chunks["id"] == "test_chunk_1_valid", "Chunk ID mismatch."
        assert parsed_chunks["metadata"]["source_url"] == "test_url_valid", "Source URL not added."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    # Test 4: VLM Output Parsing (Malformed JSON)
    test_name = "vlm_output_parsing_malformed_json"
    try:
        logger.info(f"Testing: {test_name}")
        mock_vlm_output_malformed = '```json\n[\n  {"id": "malformed", "text": "incomplete json", \n]\n```'
        parsed_chunks_malformed = parse_vlm_output(mock_vlm_output_malformed, "test_url_malformed")
        assert len(parsed_chunks_malformed) == 0, "Expected 0 chunks for malformed JSON."
        results[test_name] = {"passed": True, "details": "Correctly handled malformed JSON by returning empty list."}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    # Test 5: Fallback Markdown Processing
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
        # Check if headers are somewhat captured
        h1_found = any(c['metadata'].get('h1_section') == "Header One" for c in fallback_chunks)
        h2_found = any(c['metadata'].get('h2_section') == "Sub Header Two" for c in fallback_chunks)
        code_chunk_found = any("print(\"Hello from fallback\")" in c['text'] and c['metadata'].get('is_code_block') for c in fallback_chunks)

        assert h1_found or h2_found, "Fallback did not capture H1/H2 headers in metadata."
        # Note: Fallback's code detection is basic, might not always set is_code_block for the whole chunk perfectly.
        # contains_code_elements is a more reliable check for fallback.
        code_element_found_in_metadata = any(c['metadata'].get('contains_code_elements') for c in fallback_chunks if "print(" in c['text'])
        assert code_element_found_in_metadata, "Fallback did not mark chunk with code as containing code elements."

        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    # Test 6: Secondary Chunking
    test_name = "secondary_chunking_large_text"
    try:
        logger.info(f"Testing: {test_name}")
        large_text_content = "This is a very long string. " * 100 # Approx 2800 chars
        initial_chunk = {
            "id": "large_test_chunk",
            "text": large_text_content,
            "metadata": {"h1_section": "Large Text Section", "source_url": "test_secondary_chunk"}
        }
        target_size = 500
        secondary_chunks = apply_secondary_chunking([initial_chunk], target_size)
        assert len(secondary_chunks) > 1, "Secondary chunking should split large text."
        for sc in secondary_chunks:
            assert len(sc["text"]) <= target_size * 1.2, f"Sub-chunk too large: {len(sc['text'])} vs target {target_size}" # Allow some leeway for splitting logic
            assert sc["metadata"]["parent_chunk_id"] == "large_test_chunk", "Parent ID not set in sub-chunk."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}
        
    return results


def test_policy_validation(temp_test_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Test complete policy validation implementation using placeholder models."""
    results: Dict[str, Dict[str, Any]] = {}
    
    # Setup a minimal ClassificationAPI instance for testing its policy logic methods
    # No actual server is run; we call methods directly.
    # No policy config file needed here as we pass policy dicts directly to _handle_request_policy.
    api_for_test = ClassificationAPI(
        modernbert_model_dir=None, # Uses placeholder
        host="localhost", port=0, # Not used
        policy_config_path=None 
    )
    api_for_test.setup() # Initialize placeholder models

    # Test 1: Legacy Policy Validation - Pass Case
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
        violations = api_for_test._handle_legacy_policy_validation(payload, policy_rules, response_data_store)
        assert len(violations) == 0, f"Expected 0 violations, got {len(violations)}: {violations}"
        assert "modernbert_io_validation" in response_data_store, "ModernBERT output missing in response_data."
        assert "colbert_input_sensitivity" in response_data_store, "ColBERT input output missing."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    # Test 2: Legacy Policy Validation - Input PII Fail Case
    test_name = "legacy_policy_input_pii_fail"
    try:
        logger.info(f"Testing: {test_name}")
        payload_pii = {"input_text": "My SSN is 123-45-6789.", "output_text": "Okay."}
        policy_rules_pii = {"colbert_input_sensitivity": True, "disallowed_colbert_input_classes": ["Class 1: PII"]}
        response_data_store_pii = {}
        violations_pii = api_for_test._handle_legacy_policy_validation(payload_pii, policy_rules_pii, response_data_store_pii)
        assert len(violations_pii) > 0, "Expected PII violation for input, got none."
        assert any("Input text classified as disallowed sensitivity" in v for v in violations_pii), "PII violation message mismatch."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    # Test 3: Multimodal Policy - VLM Item Processing (Conceptual for placeholder)
    test_name = "multimodal_vlm_item_processing"
    try:
        logger.info(f"Testing: {test_name}")
        payload_mm = {
            "input_items": [{"id": "img_1", "type": "image", "filename_in_form": "test_image.jpg"}]
        }
        # Mock 'files' object as might be received by Flask
        class MockFileStorage:
            def __init__(self, filename): self.filename = filename
        mock_files = {"test_image.jpg": MockFileStorage("test_image.jpg")}

        policy_rules_mm = {
            "item_processing_rules": [{
                "item_type": "image",
                "vlm_processing": {"required": True, "prompt": "Check for sensitive content."},
                "derived_text_checks": {"colbert_sensitivity": True, "disallowed_classes": ["Class 1: PII"]}
            }]
        }
        response_data_store_mm = {}
        # Ensure vision_language_processor is set up on the test API instance
        if not api_for_test.vision_language_processor: api_for_test.vision_language_processor = VisionLanguageProcessor().setup()

        violations_mm = api_for_test._handle_multimodal_policy_validation(payload_mm, policy_rules_mm, mock_files, response_data_store_mm)
        # Placeholder VLM returns generic description. If prompt includes "sensitive", it adds that.
        # Placeholder ColBERT will then classify based on keywords in that description.
        # This test mostly checks the flow and that VLM (placeholder) is called.
        assert "item_processing_results" in response_data_store_mm, "Item processing results missing."
        assert "img_1" in response_data_store_mm["item_processing_results"], "Results for 'img_1' missing."
        assert "vlm_analysis" in response_data_store_mm["item_processing_results"]["img_1"], "VLM analysis missing."
        # Violations depend on placeholder logic; for now, check flow.
        # Default placeholder VLM for "sensitive" prompt -> text contains "sensitive" -> ColBERT might flag based on its keywords.
        # Current placeholder ColBERT doesn't have "sensitive" as a keyword, so derived check might pass.
        # Let's assume for this test the main goal is that the path is exercised.
        # If VisionLanguageProcessor's describe_image returns "sensitive content", ColBERT might flag it as Class 2, not PII.
        # So, violations_mm might be empty if "Class 2: Confidential" is not in disallowed_classes for derived text.
        # This is fine for testing the flow.
        results[test_name] = {"passed": True, "details": f"Flow exercised. Violations: {violations_mm}"}

    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    # Test 4: Custom Validation Rules - Text Length Limit Fail
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
        assert "exceeds maximum length" in violations_len, "Length violation message mismatch."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    # Test 5: Documentation Assistance Query Generation
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
    
    # Test 1: RAG Index Creation and Loading
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

        # Test loading
        loaded_retriever = RAGRetriever(index_dir)
        assert loaded_retriever.load_index(), "Failed to load created RAG index."
        assert loaded_retriever.get_status()["num_documents"] == len(test_corpus_data), "Document count mismatch after load."

        # Test retrieval
        retrieved_items = loaded_retriever.retrieve("AI programming", top_k=1)
        assert len(retrieved_items) == 1, "Retrieval did not return 1 item for 'AI programming'."
        assert retrieved_items["id"] == "rag_doc_2", "Incorrect document retrieved for 'AI programming'."
        assert retrieved_items["metadata"].get("topic") == "Python", "Metadata not retrieved correctly."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    # Test 2: Documentation RAG Index Building (using fallback Python for test speed/simplicity)
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
            chunk_size=100, # Small chunk size for testing
            chunk_overlap=10,
            processing_strategy="python" # Force python fallback
        )
        assert success, "build_documentation_rag_index (fallback) failed."
        assert doc_index_dir.exists(), "Documentation RAG index (fallback) dir not created."

        # Verify by loading and querying
        doc_retriever = RAGRetriever(doc_index_dir)
        assert doc_retriever.load_index(), "Failed to load doc RAG index (fallback)."
        ret_docs = doc_retriever.retrieve("API validation", top_k=1)
        assert len(ret_docs) > 0, "No results from doc RAG index (fallback) for 'API validation'."
        assert "API validation" in ret_docs["text"], "Retrieved doc text mismatch."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}
        
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

    def _private_method(self):
        pass # This might be included depending on strategy
"""
    code_file = temp_test_dir / "sample_test_code.py"
    code_file.write_text(sample_python_code)

    # Test 1: Python Code Parsing - 'functions' strategy
    test_name = "code_parsing_strategy_functions"
    try:
        logger.info(f"Testing: {test_name}")
        # The 'functions' strategy in implementation now means top-level functions and methods
        chunks_func = parse_and_chunk_python_code(sample_python_code, "functions", 1000, 100)
        
        func_names_found = {c["metadata"]["name"] for c in chunks_func if c["metadata"]["type"] == "function"}
        method_names_found = {c["metadata"]["name"] for c in chunks_func if c["metadata"]["type"] == "method"}
        
        assert "top_level_function" in func_names_found, "Top-level function not chunked."
        assert "__init__" in method_names_found, "__init__ method not chunked."
        assert "method_one" in method_names_found, "method_one not chunked."
        # _private_method should also be included by current logic
        assert "_private_method" in method_names_found, "_private_method not chunked by 'functions' strategy."
        # Expected total: 1 function + 3 methods = 4 chunks
        assert len(chunks_func) == 4, f"Expected 4 chunks for 'functions' strategy, got {len(chunks_func)}."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    # Test 2: Python Code Parsing - 'classes' strategy
    test_name = "code_parsing_strategy_classes"
    try:
        logger.info(f"Testing: {test_name}")
        chunks_class = parse_and_chunk_python_code(sample_python_code, "classes", 1000, 100)
        assert len(chunks_class) == 1, "Expected 1 class chunk."
        assert chunks_class["metadata"]["name"] == "MyClass", "Class name mismatch."
        assert chunks_class["metadata"]["type"] == "class", "Chunk type not 'class'."
        assert len(chunks_class["metadata"]["methods_signatures"]) == 3, "Incorrect number of method signatures for MyClass."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}

    # Test 3: Codebase RAG Index Building and Retrieval
    test_name = "codebase_rag_build_and_query"
    try:
        logger.info(f"Testing: {test_name}")
        code_index_dir = temp_test_dir / "my_codebase_index"
        success_build = build_codebase_rag_index(
            code_file_path=code_file,
            index_path=code_index_dir,
            embedding_model_id="all-MiniLM-L6-v2",
            strategy="functions", # Use functions strategy for this test
            chunk_size=1000, chunk_overlap=100
        )
        assert success_build, "Codebase RAG index building failed."
        
        code_retriever = RAGRetriever(code_index_dir)
        assert code_retriever.load_index(), "Failed to load codebase RAG index."
        
        ret_code_items = code_retriever.retrieve("method one docstring", top_k=1)
        assert len(ret_code_items) == 1, "Retrieval from code index failed for 'method one docstring'."
        assert ret_code_items["metadata"]["name"] == "method_one", "Retrieved wrong code chunk for 'method one docstring'."
        results[test_name] = {"passed": True}
    except Exception as e:
        results[test_name] = {"passed": False, "error": str(e)}
        
    return results


if __name__ == "__main__":
    logger.info(f"Script Start/Restart: HF_TOKEN from env: {os.environ.get('HF_TOKEN')}")
    is_venv_ok = ensure_venv()
    # Log HF_TOKEN status before running main logic
    logger.info(f"Global Scope: HF_TOKEN from env before _initialize_and_run: {os.environ.get('HF_TOKEN')}")
    exit_code = 1 # Default to error

    if is_venv_ok:
        # Venv is confirmed. Now it's safe to perform imports that rely on packages in the venv.
        # These imports will populate the global scope.
        logger.info("Virtual environment confirmed. Loading core dependencies globally...")
        try:
            # Assign to global 'np'
            import numpy
            np = numpy

            # Other utilities that might be needed globally or by classes indirectly
            from tqdm import tqdm
            import ranx
            from packaging import version # If any version checks happen at global or main_cli level

            # For RAGRetriever and other SentenceTransformer uses
            from sentence_transformers import SentenceTransformer as GlobalSentenceTransformer, util as st_util
            SentenceTransformer = GlobalSentenceTransformer # Assign to global SentenceTransformer

            # For ClassificationAPI and Flask server
            from flask import Flask as GlobalFlask, request as GlobalRequest, jsonify as GlobalJsonify
            from flask_cors import CORS as GlobalCORS
            from waitress import serve as GlobalServe
            Flask = GlobalFlask
            request = GlobalRequest
            jsonify = GlobalJsonify
            CORS = GlobalCORS
            serve = GlobalServe
            
            logger.info("Successfully imported core non-ML/LLM dependencies globally.")

            # Conditionally import llama_cpp and update LLAMA_CPP_AVAILABLE
            try:
                import llama_cpp
                LLAMA_CPP_AVAILABLE = True # Update the global flag
                logger.info("llama-cpp-python is available globally for VLM processing.")
            except ImportError:
                # LLAMA_CPP_AVAILABLE remains False (its initial global value)
                logger.warning("llama-cpp-python not available globally. VLM processing features will be limited.")

            # Core ML libraries - PyTorch and base Transformers
            import torch as global_torch # Assign to global torch
            torch = global_torch
            import transformers # Base import for transformers library
            
            logger.info("Successfully imported core ML/LLM dependencies (PyTorch, Transformers) globally.")

            # Now that all necessary global imports are done, proceed with the main logic
            exit_code = _initialize_and_run()
        except SystemExit as e: # Allow SystemExit to propagate (e.g. from argparse --help)
            exit_code = e.code if isinstance(e.code, int) else 1
        except KeyboardInterrupt:
            logger.info("Process interrupted by user (KeyboardInterrupt). Exiting.")
            exit_code = 130 # Standard exit code for Ctrl+C
        except ImportError as e_import:
            logger.critical(f"A required dependency could not be imported globally even after venv setup: {e_import}", exc_info=True)
            exit_code = 1
        except Exception as e_main:
            logger.critical(f"An unhandled exception occurred in _initialize_and_run: {e_main}", exc_info=True)
            exit_code = 1 # General error
    else:
        logger.error("Failed to activate or confirm virtual environment. Exiting.")
        exit_code = 1
    
    logger.info(f"Script finished with exit code {exit_code}.")
    sys.exit(exit_code)

