#!/usr/bin/env python
"""
Standard Transformer-Based Classification and Reranking Service

A self-installing CLI/Python tool that:
1. Fine-tunes a standard Hugging Face Transformer model (e.g., DeBERTa-v3) for binary
   classification of input-output text pairs (I/O Validation).
2. Fine-tunes a ColBERT-style model (e.g., from sentence-transformers like msmarco-distilbert-base-tas-b)
   for data sensitivity classification by comparing input against reference examples via MaxSim.
3. Provides CLI-based direct classification using the fine-tuned or base models for both I/O
   validation and sensitivity classification.
4. Deploys a RESTful API server featuring:
    a. Direct access to the fine-tuned I/O validation model.
    b. Direct access to the (base or fine-tuned) ColBERT sensitivity classification model.
    c. A policy-driven '/service/validate' endpoint that integrates both models to enforce
       custom rules on input and output text based on their content and sensitivity.
5. Manages its own Python virtual environment and dependencies.
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
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("classifier_service_tool")

# Attempt to import PyTorch and Transformers early
try:
    import torch
    from transformers.optimization import get_linear_schedule_with_warmup
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    from torch.optim import AdamW
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AutoConfig # Keep this
    from packaging import version
    
    # Custom modern_bert.py imports and registrations are removed.
    # We will use standard Hugging Face models.
    logger.info("Standard Hugging Face models will be used. Custom 'modern_bert.py' and its registrations are no longer needed.")

except ImportError as e:
    logger.error("CRITICAL IMPORT ERROR: Failed to import PyTorch or Transformers. The script cannot function.", exc_info=True)
    sys.exit(1) # Exit if essential libraries are missing
except Exception as e:
    logger.error(f"An unexpected error occurred during initial imports: {e}", exc_info=True)
    sys.exit(1) # Exit on other unexpected import errors

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
]

# Global Variables
stop_signal_received = False

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
        return True

    logger.info(f"Not running inside the target '{VENV_DIR}' virtual environment.")
    venv_exists = os.path.isdir(venv_path)
    logger.debug(f"Virtual environment exists at '{venv_path}': {venv_exists}")

    if not venv_exists:
        logger.info(f"Creating virtual environment in '{venv_path}'...")
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
    logger.debug(f"Python executable in venv: {python_executable}")
    logger.debug(f"Pip executable in venv: {pip_executable}")


    if not os.path.exists(python_executable):
        logger.error(f"Python executable not found at '{python_executable}'. Venv creation might have failed.")
        sys.exit(1)
    if not os.path.exists(pip_executable):
        logger.error(f"Pip executable not found at '{pip_executable}'. Venv creation might have failed.")
        sys.exit(1)

    try:
        logger.info("Attempting to upgrade pip in the virtual environment...")
        pip_upgrade_cmd = [pip_executable, "install", "--upgrade", "pip"]
        logger.debug(f"Running pip upgrade command: {' '.join(pip_upgrade_cmd)}")
        subprocess.run(pip_upgrade_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        logger.info("Pip upgraded successfully in the virtual environment.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to upgrade pip: {e.stderr}. Continuing with existing pip version.")
    except Exception as e:
        logger.warning(f"An unexpected error occurred while upgrading pip: {e}. Continuing.")

    current_packages_to_install = list(REQUIRED_PACKAGES)
    if sys.platform != "win32":
        try:
            logger.info("Temporarily installing torch to check CUDA for flash-attn decision...")
            torch_install_cmd = [pip_executable, "install", "torch"]
            logger.debug(f"Running torch install command for CUDA check: {' '.join(torch_install_cmd)}")
            subprocess.run(torch_install_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')

            cuda_check_script = "import torch; print(torch.cuda.is_available())"
            cuda_check_cmd = [python_executable, "-c", cuda_check_script]
            logger.debug(f"Running CUDA check command: {' '.join(cuda_check_cmd)}")
            result = subprocess.run(cuda_check_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
            cuda_available = result.stdout.strip().lower() == "true"
            logger.debug(f"CUDA available check result: {cuda_available}")

            if cuda_available:
                logger.info("CUDA is available. Adding flash-attn to requirements for non-Windows OS.")
                current_packages_to_install.append("flash-attn>=2.0.0")
            else:
                logger.info("CUDA not available (or check failed). Skipping flash-attn installation.")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed during CUDA check/torch install for flash-attn: {e.stderr}. Skipping flash-attn consideration.")
        except Exception as e:
            logger.warning(f"An unexpected error occurred during CUDA check for flash-attn: {e}. Skipping flash-attn.")

    logger.info(f"Installing/checking required packages in '{venv_path}': {', '.join(current_packages_to_install)}")
    install_command = [pip_executable, "install"] + current_packages_to_install
    logger.debug(f"Running package install command: {' '.join(install_command)}")
    try:
        result = subprocess.run(install_command, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        logger.info("Required packages installed/verified successfully.")
        if result.stdout: logger.debug(f"pip install stdout:\n{result.stdout}")
        if result.stderr: logger.debug(f"pip install stderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing required packages using command: {' '.join(e.cmd)}")
        logger.error(f"Pip stdout:\n{e.stdout if e.stdout else 'N/A'}")
        logger.error(f"Pip stderr:\n{e.stderr if e.stderr else 'N/A'}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during package installation: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"\nRestarting script using Python from '{venv_path}'...\n{'='*20}\n")
    script_path = os.path.abspath(__file__)
    exec_args = [python_executable, script_path] + sys.argv[1:]
    logger.debug(f"Executing os.execv with: {exec_args}")
    try:
        os.execv(python_executable, exec_args)
    except OSError as e:
        logger.error(f"os.execv failed: {e}. Executable='{python_executable}', Script='{script_path}'", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during script restart attempt: {e}", exc_info=True)
        sys.exit(1)
    return False # Should not be reached if os.execv is successful

# Import other necessary modules after venv check
import numpy as np
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve


class DataProcessor:
    """Handles JSONL data loading for I/O Validator (ModernBERTClassifier) fine-tuning."""
    def __init__(self, jsonl_paths: Union[str, List[str]]):
        if isinstance(jsonl_paths, str): self.jsonl_paths = [Path(jsonl_paths)]
        else: self.jsonl_paths = [Path(p) for p in jsonl_paths]
        self.data_entries: List[Dict[str, Any]] = []
        self.stats = { "total_files_processed": 0, "total_lines_read": 0, "valid_entries": 0,
                       "invalid_entries_missing_fields": 0, "invalid_entries_wrong_type": 0,
                       "invalid_entries_json_decode_error": 0, "files_with_errors": set() }
        logger.debug(f"DataProcessor initialized with paths: {self.jsonl_paths}")

    def load_and_validate(self) -> bool:
        logger.info(f"Loading I/O Validator training data from {len(self.jsonl_paths)} path(s)...")
        for file_path_obj in self.jsonl_paths:
            file_path_str = str(file_path_obj)
            logger.debug(f"Processing file: {file_path_str}")
            if not file_path_obj.exists() or not file_path_obj.is_file():
                logger.warning(f"File not found: {file_path_str}. Skipping."); self.stats["files_with_errors"].add(file_path_str); continue
            self.stats["total_files_processed"] += 1
            try:
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        self.stats["total_lines_read"] += 1; line = line.strip()
                        if not line: logger.debug(f"Skipping empty line {i+1} in {file_path_str}"); continue
                        try:
                            item = json.loads(line)
                            if 'input' in item and 'output_good_sample' in item and 'output_bad_sample' in item:
                                if not all(isinstance(item[k], str) for k in ['input', 'output_good_sample', 'output_bad_sample']):
                                    logger.debug(f"Invalid type in line {i+1} (triplet format) in {file_path_str}: {item}")
                                    self.stats["invalid_entries_wrong_type"] += 1; continue
                                self.data_entries.append(item); self.stats["valid_entries"] += 1
                            elif 'input' in item and 'label' in item:
                                if not isinstance(item['input'], str) or not isinstance(item['label'], int) or item['label'] not in (0,1):
                                    logger.debug(f"Invalid type/value in line {i+1} (labeled format) in {file_path_str}: {item}")
                                    self.stats["invalid_entries_wrong_type"] += 1; continue
                                self.data_entries.append({'input': item['input'], 'label': item['label']}); self.stats["valid_entries"] += 1
                            else:
                                logger.debug(f"Missing fields in line {i+1} in {file_path_str}: {item}")
                                self.stats["invalid_entries_missing_fields"] += 1
                        except json.JSONDecodeError:
                            logger.debug(f"JSON decode error in line {i+1} in {file_path_str}: '{line[:100]}...'")
                            self.stats["invalid_entries_json_decode_error"] += 1; self.stats["files_with_errors"].add(file_path_str)
                        except Exception as e_item:
                            logger.debug(f"Unexpected error processing item from line {i+1} in {file_path_str}: {e_item}", exc_info=True)
                            self.stats["invalid_entries_json_decode_error"] += 1; self.stats["files_with_errors"].add(file_path_str) # Classify as decode error for simplicity
            except Exception as e_file: logger.error(f"Error processing file {file_path_str}: {e_file}", exc_info=True); self.stats["files_with_errors"].add(file_path_str)
        logger.info(f"Data loading stats: {self.stats}")
        return bool(self.data_entries)

    def prepare_classification_data(self, separator: str = " [SEP] ", balance_classes: bool = True) -> Tuple[List[str], List[int]]:
        texts, labels = [], []
        logger.debug(f"Preparing classification data with separator: '{separator}', balance_classes: {balance_classes}")
        for item_idx, item in enumerate(self.data_entries):
            if 'output_good_sample' in item:
                texts.append(f"{item['input'].strip()}{separator}{item['output_good_sample'].strip()}"); labels.append(1)
                texts.append(f"{item['input'].strip()}{separator}{item['output_bad_sample'].strip()}"); labels.append(0)
            elif 'label' in item: texts.append(item['input'].strip()); labels.append(item['label'])
            if item_idx < 5: logger.debug(f"Sample item {item_idx} being processed: {item} -> texts/labels added")


        if balance_classes and labels:
            pos_indices = [i for i, lbl in enumerate(labels) if lbl == 1]
            neg_indices = [i for i, lbl in enumerate(labels) if lbl == 0]
            logger.debug(f"Balancing classes: {len(pos_indices)} positive, {len(neg_indices)} negative samples initially.")
            if not pos_indices or not neg_indices: logger.warning("Cannot balance: one class has no samples.")
            elif len(pos_indices) != len(neg_indices):
                import random
                min_samples = min(len(pos_indices), len(neg_indices))
                logger.debug(f"Balancing to {min_samples} samples per class.")
                chosen_pos = random.sample(pos_indices, min_samples)
                chosen_neg = random.sample(neg_indices, min_samples)
                balanced_texts, balanced_labels = [], []
                for i in chosen_pos + chosen_neg: balanced_texts.append(texts[i]); balanced_labels.append(labels[i])
                combined = list(zip(balanced_texts, balanced_labels)); random.shuffle(combined)
                texts, labels = [list(t) for t in zip(*combined)] if combined else ([], [])
                logger.info(f"Classes balanced. New count: {len(texts)}")
        logger.info(f"Prepared {len(texts)} samples for I/O Validator classification.")
        if texts: logger.debug(f"First few prepared samples: Texts: {texts[:2]}, Labels: {labels[:2]}")
        return texts, labels

    def perform_train_test_split(self, texts: List[str], labels: List[int], test_size: float = 0.2, random_state: int = 42) -> Dict[str, List[Any]]:
        logger.debug(f"Performing train/test split: test_size={test_size}, random_state={random_state}, num_samples={len(texts)}")
        if not texts or not labels:
            logger.warning("Empty texts or labels for train/test split.")
            return {'train_texts': [], 'train_labels': [], 'test_texts': [], 'test_labels': []}
        stratify_param = labels if len(set(labels)) >= 2 else None
        logger.debug(f"Stratify parameter for split: {'labels' if stratify_param else 'None'}")
        if test_size <= 0 or int(len(texts) * test_size) < 1 :
            logger.info(f"Test size {test_size} too small for {len(texts)} samples, using all data for training.")
            return {'train_texts': texts, 'train_labels': labels, 'test_texts': [], 'test_labels': []}
        X_train, X_test, y_train, y_test = sklearn_train_test_split(texts, labels, test_size=test_size, stratify=stratify_param, random_state=random_state)
        logger.info(f"Train/test split: {len(X_train)} train, {len(X_test)} test samples.")
        return {'train_texts': X_train, 'train_labels': y_train, 'test_texts': X_test, 'test_labels': y_test}

class ModernBERTClassifier: # Name kept for consistency, but it's a standard Transformer now
    DEFAULT_MAX_LENGTH = 256; DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_MODEL_A_ID = "microsoft/deberta-v3-base" 

    def __init__(self, model_dir: str = "model_files", use_mlflow: bool = False, base_model_id_for_training: Optional[str] = None):
        self.model_dir = Path(model_dir).resolve()
        self.model_id: Optional[str] = None 
        self.args_base_model_id = base_model_id_for_training
        self.model: Optional[AutoModelForSequenceClassification] = None; self.tokenizer: Optional[AutoTokenizer] = None
        self.device: Optional[torch.device] = None; self.separator: str = " [SEP] "
        self.use_mlflow = use_mlflow
        logger.debug(f"I/O Validator (ModernBERTClassifier) initialized. model_dir='{self.model_dir}', base_for_train='{base_model_id_for_training}', use_mlflow={self.use_mlflow}")
        if self.use_mlflow:
            try: import mlflow; self.mlflow_client = mlflow; logger.debug("MLflow imported successfully.")
            except ImportError: logger.warning("MLflow not found, disabling."); self.use_mlflow = False

    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"I/O Validator using device: {self.device}")
        has_flash_attn = False
        if self.device.type == 'cuda':
            try: import flash_attn; has_flash_attn = True; logger.info("Flash Attention 2 available for I/O Validator.")
            except ImportError: logger.info("Flash Attention 2 not found for I/O Validator."); logger.debug("flash_attn import failed", exc_info=logger.level==logging.DEBUG)

        model_load_path = self.model_id if self.model_id and Path(self.model_id).is_dir() else self.args_base_model_id if hasattr(self, 'args_base_model_id') and self.args_base_model_id else self.DEFAULT_MODEL_A_ID
        if not self.model_id:
            self.model_id = model_load_path

        logger.info(f"I/O Validator setup: Loading tokenizer and model from '{model_load_path}'")
        logger.debug(f"HF_TOKEN env var present: {bool(os.getenv('HF_TOKEN'))}")
        logger.debug(f"HUGGING_FACE_HUB_TOKEN env var present: {bool(os.getenv('HUGGING_FACE_HUB_TOKEN'))}")
        
        # Add detailed hf_hub logging
        os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "debug"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Faster downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Faster downloads
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"   # Increase timeout
        os.environ["HF_HUB_OFFLINE"] = "0"              # Ensure online mode

        # Explicitly pass token to ensure it's used
        # Prioritize HF_TOKEN, then HUGGING_FACE_HUB_TOKEN
        auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not auth_token:
            logger.warning("HF_TOKEN or HUGGING_FACE_HUB_TOKEN not found in environment for model loading!")
        else:
            logger.debug(f"Using auth_token for from_pretrained: {'*' * (len(auth_token) - 4) + auth_token[-4:] if auth_token else 'None'}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_load_path, token=auth_token) # Pass token
        except Exception as e_tokenizer:
            logger.error(f"DIRECT HUGGINGFACE ERROR during AutoTokenizer.from_pretrained for '{model_load_path}': {e_tokenizer}", exc_info=True)
            raise # Re-raise to be caught by the main handler

        if self.tokenizer.pad_token is None:
            logger.warning(f"Tokenizer for I/O Validator '{model_load_path}' missing pad_token, adding default '{self.DEFAULT_PAD_TOKEN}'.")
            self.tokenizer.add_special_tokens({'pad_token': self.DEFAULT_PAD_TOKEN})
        
        num_labels_for_model = 2
        model_kwargs = {"num_labels": num_labels_for_model, "token": auth_token} # Pass token
        if has_flash_attn and version.parse(torch.__version__) >= version.parse("2.0"): model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.debug(f"Model kwargs for I/O Validator from_pretrained: {model_kwargs}")

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_load_path, **model_kwargs)
        except Exception as e_hf_model_load: # Changed variable name to avoid conflict
            logger.error(f"DIRECT HUGGINGFACE ERROR during AutoModelForSequenceClassification.from_pretrained for '{model_load_path}': {e_hf_model_load}", exc_info=True)
            # Check if the issue is with snapshot_download specifically
            from huggingface_hub.hf_api import HfApi
            from huggingface_hub.utils import EntryNotFoundError
            api = HfApi()
            try:
                logger.debug(f"Attempting to get model info for {model_load_path} via HfApi")
                model_info_obj = api.model_info(repo_id=model_load_path, token=auth_token)
                logger.debug(f"HfApi.model_info successful. SHA: {model_info_obj.sha}")
                
                # Try to simulate what from_pretrained does internally (simplified)
                from huggingface_hub import snapshot_download
                logger.debug(f"Attempting snapshot_download for {model_load_path}")
                downloaded_folder = snapshot_download(
                    repo_id=model_load_path,
                    token=auth_token,
                    cache_dir=os.getenv('HF_HOME') or (Path.home() / ".cache" / "huggingface" / "hub"),
                    allow_patterns=["*.json", "*.bin", "*.model", "*.txt", "*.md"],  # Expanded patterns
                    ignore_patterns=["*.h5", "*.ot", "*.msgpack"],  # Exclude non-PyTorch formats
                    local_files_only=False,
                    resume_download=True
                )
                logger.info(f"snapshot_download completed. Folder: {downloaded_folder}")
                
                # Detailed file listing with sizes
                file_list = []
                for f in Path(downloaded_folder).rglob('*'):
                    if f.is_file():
                        try:
                            file_list.append(f"{f.relative_to(downloaded_folder)} ({f.stat().st_size} bytes)")
                        except Exception as e:
                            file_list.append(f"{f.name} [size error]")
                logger.debug(f"Downloaded files:\n- " + "\n- ".join(file_list))
                
                # Verify critical files exist
                required_files = {
                    "config.json": False,
                    "pytorch_model.bin": False,
                    "tokenizer_config.json": False,
                    "vocab.txt": False,
                    "special_tokens_map.json": False
                }
                for fname in required_files:
                    if (Path(downloaded_folder) / fname).exists():
                        required_files[fname] = True
                logger.info(f"Required files check:\n{json.dumps(required_files, indent=2)}")
                logger.info(f"snapshot_download completed. Folder: {downloaded_folder}")
                
                # Detailed file listing with sizes
                file_list = []
                for f in Path(downloaded_folder).rglob('*'):
                    if f.is_file():
                        try:
                            file_list.append(f"{f.relative_to(downloaded_folder)} ({f.stat().st_size} bytes)")
                        except Exception as e:
                            file_list.append(f"{f.name} [size error]")
                logger.debug(f"Downloaded files:\n- " + "\n- ".join(file_list))
                
                # Verify critical files exist
                required_files = {
                    "config.json": False,
                    "pytorch_model.bin": False,
                    "tokenizer_config.json": False,
                    "vocab.txt": False,
                    "special_tokens_map.json": False
                }
                for fname in required_files:
                    if (Path(downloaded_folder) / fname).exists():
                        required_files[fname] = True
                logger.info(f"Required files check:\n{json.dumps(required_files, indent=2)}")

            except EntryNotFoundError:
                logger.error(f"HfApi confirms EntryNotFoundError for {model_load_path}.", exc_info=True)
            except Exception as e_snapshot:
                logger.error(f"Error during manual HfApi/snapshot_download attempt: {e_snapshot}", exc_info=True)
            # Re-raise original error
            raise e_hf_model_load # Re-raise to be caught by the main handler

        if self.model and self.tokenizer.pad_token_id is not None and self.tokenizer.pad_token_id >= self.model.config.vocab_size:
            logger.info(f"Resizing token embeddings for I/O Validator '{model_load_path}'. Old vocab: {self.model.config.vocab_size}, New (tokenizer): {len(self.tokenizer)}")
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.model:
            self.model.to(self.device)
            logger.debug(f"I/O Validator model '{self.model.config._name_or_path}' moved to {self.device}.")
        return self

    def _create_dataloader(self, texts: List[str], labels: Optional[List[int]], batch_size: int, shuffle: bool = False):
        if not self.tokenizer: raise RuntimeError("I/O Validator tokenizer not initialized.")
        logger.debug(f"Creating DataLoader for I/O Validator: {len(texts)} texts, batch_size={batch_size}, shuffle={shuffle}, max_length={self.DEFAULT_MAX_LENGTH}")
        encodings = self.tokenizer(texts, truncation=True, padding="max_length", max_length=self.DEFAULT_MAX_LENGTH, return_tensors="pt")
        ds_input_ids, ds_attn_mask = encodings.input_ids, encodings.attention_mask
        logger.debug(f"Encoded input_ids shape: {ds_input_ids.shape}, attention_mask shape: {ds_attn_mask.shape}")
        if labels is not None:
            logger.debug(f"Number of labels provided: {len(labels)}")
            if len(ds_input_ids) != len(labels): 
                min_len = min(len(ds_input_ids), len(labels))
                logger.warning(f"Mismatch between encoded inputs ({len(ds_input_ids)}) and labels ({len(labels)}). Truncating to {min_len}.")
                ds_input_ids, ds_attn_mask, labels = ds_input_ids[:min_len], ds_attn_mask[:min_len], labels[:min_len]
            dataset = TensorDataset(ds_input_ids, ds_attn_mask, torch.tensor(labels, dtype=torch.long))
        else: dataset = TensorDataset(ds_input_ids, ds_attn_mask)
        logger.debug(f"TensorDataset created for I/O Validator. Sample 0 input_ids: {dataset[0][0][:10]}...")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def train(self, train_texts: List[str], train_labels: List[int], eval_texts: Optional[List[str]] = None, eval_labels: Optional[List[int]] = None,
              batch_size: int = 8, learning_rate: float = 2e-5, epochs: int = 3, gradient_accumulation_steps: int = 1,
              early_stopping_patience: int = 0, warmup_ratio: float = 0.1, weight_decay: float = 0.01):
        if not self.model or not self.tokenizer: raise RuntimeError("I/O Validator model/tokenizer not setup.")
        logger.debug(f"I/O Validator training parameters: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}, grad_accum={gradient_accumulation_steps}, early_stop={early_stopping_patience}, warmup_ratio={warmup_ratio}, weight_decay={weight_decay}")
        train_loader = self._create_dataloader(train_texts, train_labels, batch_size, shuffle=True)
        eval_loader = self._create_dataloader(eval_texts, eval_labels, batch_size) if eval_texts and eval_labels else None
        if eval_loader: logger.debug(f"Evaluation DataLoader for I/O Validator created with {len(eval_loader.dataset)} samples.")

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        num_warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
        logger.debug(f"Optimizer: AdamW. Total train steps for I/O Validator: {total_steps}, Warmup steps: {num_warmup_steps}")
        best_eval_metric, epochs_no_improve = -float('inf'), 0

        logger.info("Starting I/O Validator training...")
        for epoch in range(epochs):
            if stop_signal_received: logger.info("I/O Validator training interrupted by signal."); break
            self.model.train(); total_loss, train_steps = 0, 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training (I/O Validator)", leave=False, disable=logger.level > logging.INFO)
            for step, batch_data in enumerate(progress_bar):
                b_input_ids, b_attn_mask, b_labels = [b.to(self.device) for b in batch_data]
                if step == 0 and epoch == 0: logger.debug(f"First batch - input_ids shape: {b_input_ids.shape}, attention_mask shape: {b_attn_mask.shape}, labels shape: {b_labels.shape}")
                outputs = self.model(b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
                loss = outputs.loss
                if gradient_accumulation_steps > 1: loss = loss / gradient_accumulation_steps
                loss.backward(); total_loss += loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1); train_steps +=1
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad()
                current_loss_display = loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)
                progress_bar.set_postfix({'loss': current_loss_display})
                if step % (len(train_loader)//5 +1) == 0 : logger.debug(f"Epoch {epoch+1}, Step {step+1}: Current batch loss {current_loss_display:.4f}")

            avg_train_loss = total_loss/train_steps if train_steps else 0
            logger.info(f"Epoch {epoch+1} (I/O Validator) avg train loss: {avg_train_loss:.4f}")
            if self.use_mlflow: self.mlflow_client.log_metric(f"io_validator_train_loss_epoch_{epoch+1}", avg_train_loss, step=epoch+1)

            if eval_loader:
                metrics = self._evaluate_from_loader(eval_loader)
                logger.info(f"Epoch {epoch+1} (I/O Validator) Eval: {metrics}")
                if self.use_mlflow:
                    for m_name, m_val in metrics.items(): self.mlflow_client.log_metric(f"io_validator_eval_{m_name}_epoch_{epoch+1}", m_val, step=epoch+1)

                current_metric = metrics.get('f1', metrics.get('accuracy', 0.0))
                if current_metric > best_eval_metric:
                    logger.debug(f"New best eval metric for I/O Validator: {current_metric:.4f} (was {best_eval_metric:.4f}). Saving 'best' model.")
                    best_eval_metric, epochs_no_improve = current_metric, 0; self._save_model("best")
                else: epochs_no_improve += 1
                logger.debug(f"Epochs without improvement for I/O Validator: {epochs_no_improve}")
                if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                    logger.info(f"I/O Validator early stopping triggered after {epochs_no_improve} epochs without improvement."); break

        logger.info("I/O Validator training finished."); self._save_model("latest")
        if early_stopping_patience > 0 and (self.model_dir / "best").exists() and best_eval_metric > -float('inf'):
            logger.info(f"Loading best I/O Validator model (metric: {best_eval_metric:.4f}).")
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir / "best").to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "best")
                logger.debug("Successfully loaded best I/O Validator model and tokenizer.")
            except Exception as e: logger.error(f"Failed to load best I/O Validator model: {e}", exc_info=True)
        elif self.use_mlflow: self.mlflow_client.log_params({"io_validator_final_model_type": "latest_after_full_epochs"})
        return self

    def _evaluate_from_loader(self, eval_loader: DataLoader) -> Dict[str, float]:
        if not self.model: raise RuntimeError("I/O Validator model not initialized for evaluation.")
        self.model.eval(); total_loss, eval_steps = 0,0; all_logits, all_labels = [], []
        logger.debug(f"Starting I/O Validator evaluation with {len(eval_loader.dataset)} samples.")
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(eval_loader, desc="Evaluating (I/O Validator)", leave=False, disable=logger.level > logging.INFO)):
                b_input_ids, b_attn_mask, b_labels = [b.to(self.device) for b in batch_data]
                if batch_idx == 0: logger.debug(f"Eval batch 0 - input_ids shape: {b_input_ids.shape}")
                outputs = self.model(b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
                total_loss += outputs.loss.item(); eval_steps += 1
                all_logits.append(outputs.logits.cpu().numpy()); all_labels.append(b_labels.cpu().numpy())
        preds = np.argmax(np.concatenate(all_logits), axis=1)
        true_labels = np.concatenate(all_labels)
        logger.debug(f"I/O Validator evaluation shapes: preds {preds.shape}, true_labels {true_labels.shape}")
        p, r, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
        acc = accuracy_score(true_labels, preds)
        avg_loss = total_loss/eval_steps if eval_steps else 0
        metrics = {'loss': avg_loss, 'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
        logger.debug(f"I/O Validator evaluation metrics computed: {metrics}")
        return metrics

    def predict(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        if not self.model or not self.tokenizer: raise RuntimeError("I/O Validator model/tokenizer not setup.")
        self.model.eval(); predictions_data = []
        logger.debug(f"I/O Validator predicting for {len(texts)} texts, batch_size={batch_size}")
        if texts: logger.debug(f"First text for I/O Validator prediction: '{texts[0][:100]}...'")
        predict_loader = self._create_dataloader(texts, None, batch_size)
        text_idx_offset = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(predict_loader, desc="Predicting (I/O Validator)", leave=False, disable=logger.level > logging.INFO)):
                b_input_ids, b_attn_mask = batch_data[0].to(self.device), batch_data[1].to(self.device)
                if batch_idx == 0: logger.debug(f"Prediction batch 0 - input_ids shape: {b_input_ids.shape}")
                logits = self.model(b_input_ids, attention_mask=b_attn_mask).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                logger.debug(f"Batch {batch_idx} - logits shape: {logits.shape}, probs shape: {probs.shape}, preds shape: {preds.shape}")
                for i in range(len(preds)):
                    current_text_idx = text_idx_offset + i
                    if current_text_idx >= len(texts): 
                        logger.error(f"Index out of bounds: current_text_idx={current_text_idx}, len(texts)={len(texts)}")
                        continue
                    current_text = texts[current_text_idx]
                    parts = current_text.split(self.separator, 1)
                    input_part = parts[0]
                    output_part = parts[1] if len(parts) > 1 else ""
                    pred_data_item = {'prediction': int(preds[i]), 'probability_positive': float(probs[i][1]),
                                             'input_text': input_part, 'output_text': output_part}
                    predictions_data.append(pred_data_item)
                    if i==0 and batch_idx==0: logger.debug(f"First I/O Validator prediction item: {pred_data_item}")
                text_idx_offset += len(preds)
        return predictions_data

    def classify_input_output_pair(self, input_text: str, output_text: str) -> Dict[str, Any]:
        safe_output_text = output_text if output_text is not None else ""
        full_text = f"{input_text}{self.separator}{safe_output_text}"
        logger.debug(f"I/O Validator classifying pair. Input: '{input_text[:50]}...', Output: '{safe_output_text[:50]}...'. Combined: '{full_text[:100]}...'")
        return self.predict([full_text])[0]

    def _save_model(self, suffix: str = ""):
        if not self.model or not self.tokenizer:
            logger.warning("Attempted to save I/O Validator model, but model or tokenizer is not available.")
            return
        save_path = self.model_dir / suffix if suffix else self.model_dir
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving I/O Validator model to {save_path}...")
        self.model.save_pretrained(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        else:
            logger.warning(f"Tokenizer not available for I/O Validator, cannot save to {save_path}")
        config_to_save = {"separator": self.separator, "model_base_id_on_train": self.model.config._name_or_path if self.model else self.model_id}
        with open(save_path / "model_config.json", 'w', encoding='utf-8') as f: json.dump(config_to_save, f, indent=2)
        logger.debug(f"Saved I/O Validator model config: {config_to_save} to {save_path / 'model_config.json'}")

    @classmethod
    def load(cls, model_dir: str, use_mlflow_during_load: bool = False):
        model_dir_path = Path(model_dir);
        logger.debug(f"Attempting to load I/O Validator model from directory: {model_dir_path}")
        if not model_dir_path.exists() or not model_dir_path.is_dir():
             raise FileNotFoundError(f"I/O Validator model directory not found: {model_dir_path}")

        instance = cls(model_dir=str(model_dir_path), use_mlflow=use_mlflow_during_load)
        instance.model_id = str(model_dir_path.resolve()) 
        logger.debug(f"I/O Validator load: instance.model_id set to local fine-tuned path '{instance.model_id}' for setup.")
        instance.setup() 

        cfg_path = model_dir_path / "model_config.json"
        if cfg_path.exists():
            logger.debug(f"Loading I/O Validator model_config.json from {cfg_path}")
            with open(cfg_path, 'r', encoding='utf-8') as f:
                loaded_cfg = json.load(f)
                instance.separator = loaded_cfg.get("separator", instance.separator)
                logger.debug(f"Loaded separator '{instance.separator}' for I/O Validator from config. Loaded config: {loaded_cfg}")
        else: logger.warning(f"I/O Validator model_config.json not found in {model_dir_path}")
        logger.info(f"I/O Validator model loaded from {model_dir_path}.")
        return instance

    def get_hardware_info(self) -> Dict[str, Any]:
        info = {"device": str(self.device), "cuda_available": torch.cuda.is_available(), "torch_version": torch.__version__, "transformers_version": transformers.__version__}
        if torch.cuda.is_available() and self.device and self.device.type == 'cuda':
             info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
        else: info["gpu_name"] = "N/A"
        try: import flash_attn; info["flash_attn_available"] = True
        except ImportError: info["flash_attn_available"] = False
        logger.debug(f"I/O Validator Hardware Info collected: {info}")
        return info

class ColBERTReranker:
    DEFAULT_MAX_LENGTH = 512; DEFAULT_PAD_TOKEN = "[PAD]"; DEFAULT_MODEL_ID = "sentence-transformers/msmarco-distilbert-base-tas-b"
    COLBERT_CONFIG_FILENAME = "colbert_reranker_config.json"; REFERENCE_TEXTS_SNAPSHOT_FILENAME = "reference_texts_snapshot.json"
    PRECOMPUTED_REF_EMBEDDINGS_FILENAME = "ref_embeddings.pt"
    BUILTIN_REFERENCE_TEXTS_BY_CLASS: Dict[str, List[str]] = {
        "Class 1: PII": ["SSN is 987-65-4321.", "User credit card ...1234, CVV 567."],
        "Class 2: Sensitive Personal Data": ["Employee review: Needs improvement.", "User search: 'flu symptoms'."],
        "Class 3: Confidential Personal Data": ["Customer email jane.doe@example.com.", "Shipping address: 123 Main St."],
        "Class 4: Internal Data": ["Q3 marketing strategy 'Project Phoenix'.", "Internal memo: team-building."],
        "Class 5: Public Data": ["Press release: new product.", "Company 'About Us' page."] }
    CLASS_DESCRIPTIONS = { "Class 1: PII": "Most sensitive...", "Class 2: Sensitive Personal Data": "Highly restricted...",
                           "Class 3: Confidential Personal Data": "Customer data...", "Class 4: Internal Data": "Company non-public...",
                           "Class 5: Public Data": "Public data" }

    def __init__(self, model_id_or_path: str = DEFAULT_MODEL_ID):
        self.model_id_or_path = model_id_or_path
        self.model: Optional[AutoModel] = None; self.tokenizer: Optional[AutoTokenizer] = None
        self.device: Optional[torch.device] = None
        self.reference_texts_for_model: Dict[str, List[str]] = {}
        self.reference_token_embeddings_by_class: Dict[str, List[torch.Tensor]] = {}
        self.is_fine_tuned: bool = False
        logger.debug(f"ColBERTReranker initialized with model_id_or_path: '{self.model_id_or_path}'")
        if not Path(self.model_id_or_path).is_dir():
            logger.debug(f"'{self.model_id_or_path}' is not a directory, using built-in ColBERT references by default.")
            self.reference_texts_for_model = copy.deepcopy(self.BUILTIN_REFERENCE_TEXTS_BY_CLASS)
        else:
            config_path = Path(self.model_id_or_path) / self.COLBERT_CONFIG_FILENAME
            if config_path.exists():
                logger.debug(f"Found ColBERT config at {config_path}, assuming fine-tuned model path.")
                self.is_fine_tuned = True 

    def _load_custom_references_from_jsonl(self, jsonl_path: Path):
        logger.info(f"Loading ColBERT custom references from {jsonl_path}...")
        custom_refs: Dict[str, List[str]] = {}
        valid_cls = set(self.CLASS_DESCRIPTIONS.keys()) if self.CLASS_DESCRIPTIONS else None
        logger.debug(f"Valid predefined class names for ColBERT: {valid_cls if valid_cls else 'Any (dynamic)'}")

        if not jsonl_path.exists(): logger.error(f"Custom ref JSONL not found: {jsonl_path}"); return
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        item = json.loads(line)
                        text, cls_name = item.get("text"), item.get("class_name")
                        if not text or not cls_name:
                            logger.debug(f"Skipping L{i+1} in {jsonl_path}: missing text or class_name. Item: {item}")
                            continue
                        if valid_cls and cls_name not in valid_cls:
                             logger.warning(f"L{i+1} in {jsonl_path}: Class name '{cls_name}' not in known ColBERT CLASS_DESCRIPTIONS. Adding it dynamically.")
                        custom_refs.setdefault(cls_name, []).append(text)
                        if i < 5 : logger.debug(f"Loaded custom ref L{i+1}: Class '{cls_name}', Text '{text[:50]}...'")
                    except json.JSONDecodeError: logger.warning(f"L{i+1} JSON error in {jsonl_path}")
            if not custom_refs: logger.warning(f"No valid custom refs in {jsonl_path}"); return
            self.reference_texts_for_model = custom_refs
            num_loaded_refs = sum(len(v) for v in custom_refs.values())
            num_loaded_classes = len(custom_refs)
            logger.info(f"Loaded {num_loaded_refs} custom ColBERT references for {num_loaded_classes} classes.")
            logger.debug(f"Final custom reference classes and counts: {{c: len(t) for c, t in self.reference_texts_for_model.items()}}")
        except Exception as e: logger.error(f"Error loading custom ref file {jsonl_path}: {e}", exc_info=True)

    def setup_model_and_references(self, cache_dir: Optional[Path] = None, force_recompute_embeddings: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ColBERT using device: {self.device} for model: {self.model_id_or_path}")
        logger.debug(f"ColBERT setup: cache_dir='{cache_dir}', force_recompute_embeddings={force_recompute_embeddings}")
        logger.debug(f"HF_TOKEN env var for ColBERT setup: {os.getenv('HF_TOKEN')}")
        logger.debug(f"HUGGING_FACE_HUB_TOKEN env var for ColBERT setup: {os.getenv('HUGGING_FACE_HUB_TOKEN')}")
        try:
            logger.info(f"Attempting to load tokenizer for ColBERT from: {self.model_id_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path)
            if self.tokenizer.pad_token is None:
                logger.debug("ColBERT tokenizer missing pad_token, adding default.")
                self.tokenizer.add_special_tokens({'pad_token': self.DEFAULT_PAD_TOKEN})
            logger.debug(f"ColBERT tokenizer.pad_token_id: {self.tokenizer.pad_token_id}, vocab_size: {self.tokenizer.vocab_size}")
        except Exception as e: logger.error(f"Fatal: ColBERT tokenizer load error: {e}", exc_info=True); raise

        has_flash_attn = False
        if self.device.type == 'cuda':
            try: import flash_attn; has_flash_attn = True; logger.info("Flash Attention 2 available for ColBERT.")
            except ImportError: logger.info("Flash Attention 2 not found for ColBERT."); logger.debug("flash_attn import failed for ColBERT", exc_info=logger.level==logging.DEBUG)
        model_kwargs = {}
        if has_flash_attn and version.parse(torch.__version__) >= version.parse("2.0"): model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.debug(f"ColBERT model_kwargs for from_pretrained: {model_kwargs}")

        try:
            logger.info(f"Attempting to load base model for ColBERT from: {self.model_id_or_path}")
            self.model = AutoModel.from_pretrained(self.model_id_or_path, **model_kwargs) 
        except Exception as e:
            logger.warning(f"Failed to load ColBERT with initial kwargs: {e}")
            if model_kwargs: logger.warning("Failed ColBERT load w/ Flash Attn. Retrying default."); model_kwargs={}
            try:
                logger.debug(f"Retrying ColBERT load with kwargs: {model_kwargs}")
                self.model = AutoModel.from_pretrained(self.model_id_or_path, **model_kwargs)
            except Exception as e2: logger.error(f"Fatal: ColBERT model load error: {e2}", exc_info=True); raise

        if self.model and self.tokenizer.pad_token_id is not None and self.tokenizer.pad_token_id >= self.model.config.vocab_size:
             logger.debug(f"Resizing ColBERT token embeddings. Old vocab size: {self.model.config.vocab_size}, new (tokenizer): {len(self.tokenizer)}")
             self.model.resize_token_embeddings(len(self.tokenizer))
        if self.model:
            self.model.to(self.device).eval()
            logger.info(f"ColBERT Model '{self.model.config._name_or_path}' loaded and set to eval mode on {self.device}.")
        else:
            logger.error(f"ColBERT Model object is None after attempting to load {self.model_id_or_path}. Cannot proceed with setup.")
            return

        if not self.reference_texts_for_model:
            logger.warning("No ColBERT references available (neither built-in nor custom loaded). Classification might not be meaningful.");
            self.reference_token_embeddings_by_class = {}
            return

        emb_path = cache_dir / self.PRECOMPUTED_REF_EMBEDDINGS_FILENAME if cache_dir else None
        ref_snapshot_path = cache_dir / self.REFERENCE_TEXTS_SNAPSHOT_FILENAME if cache_dir else None
        logger.debug(f"ColBERT ref embeddings path: {emb_path}, snapshot path: {ref_snapshot_path}")

        refs_match = False
        if ref_snapshot_path and ref_snapshot_path.exists() and emb_path and emb_path.exists():
            try:
                logger.debug(f"Checking existing reference snapshot at {ref_snapshot_path}")
                with open(ref_snapshot_path, 'r', encoding='utf-8') as f:
                    snapshot_refs = json.load(f)
                if snapshot_refs == self.reference_texts_for_model:
                    refs_match = True
                    logger.debug("Reference texts match snapshot.")
                else:
                    logger.info("Reference texts have changed since last embedding computation. Recomputing.")
                    logger.debug(f"Snapshot refs: {str(snapshot_refs)[:200]}..., Current refs: {str(self.reference_texts_for_model)[:200]}...")
            except Exception as e:
                logger.warning(f"Could not compare reference snapshot: {e}. Recomputing embeddings.")

        if not force_recompute_embeddings and emb_path and emb_path.exists() and refs_match:
            try:
                logger.debug(f"Attempting to load pre-computed ColBERT ref embeddings from {emb_path}")
                self.reference_token_embeddings_by_class = torch.load(emb_path, map_location='cpu')
                if set(self.reference_token_embeddings_by_class.keys()) == set(self.reference_texts_for_model.keys()):
                    num_loaded_cls = len(self.reference_token_embeddings_by_class)
                    num_loaded_embs = sum(len(v) for v in self.reference_token_embeddings_by_class.values())
                    logger.info(f"Loaded {num_loaded_embs} pre-computed ColBERT ref embeddings for {num_loaded_cls} classes from {emb_path}.")
                    return
                else:
                    logger.warning("Cached embeddings do not match current reference classes. Recomputing.")
                    logger.debug(f"Cached emb keys: {set(self.reference_token_embeddings_by_class.keys())}, Current ref keys: {set(self.reference_texts_for_model.keys())}")
            except Exception as e: logger.warning(f"Failed load pre-computed ColBERT embs: {e}. Recomputing.", exc_info=True)

        logger.info("Computing ColBERT reference embeddings...");
        self._compute_and_cache_reference_embeddings(embeddings_path=emb_path, ref_snapshot_path=ref_snapshot_path)

    def _compute_and_cache_reference_embeddings(self, embeddings_path: Optional[Path], ref_snapshot_path: Optional[Path]):
        if not self.model or not self.tokenizer: raise RuntimeError("ColBERT model/tokenizer not ready for embs.")
        self.reference_token_embeddings_by_class = {}
        all_texts_flat, class_text_counts = [], {}
        logger.debug("Starting computation of ColBERT reference embeddings.")

        for cls_name, texts in self.reference_texts_for_model.items():
            if not texts:
                logger.debug(f"Class '{cls_name}' has no reference texts, will have empty embeddings list.")
                self.reference_token_embeddings_by_class[cls_name] = []
                class_text_counts[cls_name] = 0
                continue
            all_texts_flat.extend(texts)
            class_text_counts[cls_name] = len(texts)
            logger.debug(f"For class '{cls_name}', added {len(texts)} texts to flat list for embedding.")

        if not all_texts_flat:
            logger.warning("No reference texts found across all classes to compute embeddings for ColBERT.")
            if embeddings_path:
                 try:
                    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.reference_token_embeddings_by_class, embeddings_path)
                    logger.info(f"Saved (empty) ColBERT ref embs to {embeddings_path}")
                    if ref_snapshot_path:
                        with open(ref_snapshot_path, 'w', encoding='utf-8') as f: json.dump(self.reference_texts_for_model, f, indent=2)
                        logger.debug(f"Saved empty reference snapshot to {ref_snapshot_path}")
                 except Exception as e: logger.error(f"Failed save empty ColBERT ref embs: {e}", exc_info=True)
            return

        logger.debug(f"Total of {len(all_texts_flat)} reference texts to embed in batches.")
        all_embs_gpu = self._get_token_embeddings_batched(all_texts_flat)
        current_idx = 0
        for cls_name, count in class_text_counts.items():
            if count > 0:
                cls_embs_gpu = all_embs_gpu[current_idx : current_idx + count]
                self.reference_token_embeddings_by_class[cls_name] = [emb.cpu() for emb in cls_embs_gpu]
                logger.debug(f"Assigned {len(cls_embs_gpu)} embeddings to class '{cls_name}'. First embedding shape (if any): {cls_embs_gpu[0].shape if cls_embs_gpu else 'N/A'}")
                current_idx += count

        if embeddings_path:
            try:
                embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.reference_token_embeddings_by_class, embeddings_path)
                logger.info(f"Saved ColBERT ref embs to {embeddings_path}")
                if ref_snapshot_path:
                    with open(ref_snapshot_path, 'w', encoding='utf-8') as f: json.dump(self.reference_texts_for_model, f, indent=2)
                    logger.info(f"Saved ColBERT reference snapshot to {ref_snapshot_path}")
            except Exception as e: logger.error(f"Failed save ColBERT ref embs/snapshot: {e}", exc_info=True)

    def _get_token_embeddings_batched(self, texts: List[str], batch_size: int = 32, enable_grad: bool = False) -> List[torch.Tensor]:
        if not self.tokenizer or not self.model: raise RuntimeError("ColBERT model/tokenizer not ready.")
        all_embs_gpu = []
        logger.debug(f"Getting token embeddings for {len(texts)} texts. Batch size: {batch_size}, enable_grad: {enable_grad}, max_length: {self.DEFAULT_MAX_LENGTH}")

        original_model_training_state = self.model.training
        if enable_grad: self.model.train(); logger.debug("Set ColBERT model to train mode for embeddings (grad enabled).")
        else: self.model.eval(); logger.debug("Set ColBERT model to eval mode for embeddings (no grad).")

        context_manager = torch.enable_grad() if enable_grad else torch.no_grad()

        with context_manager:
            for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing & Embedding ColBERT Batches", leave=False, disable=logger.level > logging.INFO):
                batch_texts = texts[i:i+batch_size]
                if i==0 and logger.level == logging.DEBUG : logger.debug(f"First batch texts (first text sample): '{batch_texts[0][:100]}...'")
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=self.DEFAULT_MAX_LENGTH).to(self.device)
                outputs = self.model(**inputs).last_hidden_state 
                if i==0 and logger.level == logging.DEBUG: logger.debug(f"Batch {i//batch_size} - input_ids shape: {inputs['input_ids'].shape}, last_hidden_state shape: {outputs.shape}")

                for j in range(outputs.size(0)): 
                    non_padded_mask = inputs['attention_mask'][j] == 1
                    token_embeddings = outputs[j][non_padded_mask] 
                    all_embs_gpu.append(token_embeddings)
                    if i==0 and j==0 and logger.level == logging.DEBUG: logger.debug(f"First item in first batch - token_embeddings shape: {token_embeddings.shape}")

        self.model.train(original_model_training_state) 
        logger.debug(f"Restored ColBERT model training state to: {self.model.training}. Produced {len(all_embs_gpu)} embedding tensors.")
        return all_embs_gpu

    def _colbert_maxsim(self, query_embs: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        if query_embs.ndim != 2 or doc_embs.ndim != 2 or query_embs.size(0) == 0 or doc_embs.size(0) == 0:
            logger.debug(f"ColBERT MaxSim: Empty or non-2D input. Query shape: {query_embs.shape}, Doc shape: {doc_embs.shape}. Returning 0.")
            dtype_to_use = query_embs.dtype if query_embs.numel() > 0 else (doc_embs.dtype if doc_embs.numel() > 0 else torch.float32)
            device_to_use = self.device if self.device else 'cpu'
            return torch.tensor(0.0, device=device_to_use, dtype=dtype_to_use)

        q_dev, d_dev = query_embs.to(self.device), doc_embs.to(self.device)
        q_norm = F.normalize(q_dev, p=2, dim=1) 
        d_norm = F.normalize(d_dev, p=2, dim=1) 

        sim_matrix = torch.matmul(q_norm, d_norm.T) 
        if sim_matrix.numel() == 0:
             logger.debug("ColBERT MaxSim: Similarity matrix is empty. Returning 0.")
             return torch.tensor(0.0, device=self.device, dtype=q_dev.dtype)
        
        max_sim_scores_per_query_token = torch.max(sim_matrix, dim=1)[0] 
        total_score = torch.sum(max_sim_scores_per_query_token)

        if logger.level == logging.DEBUG:
            logger.debug(f"ColBERT MaxSim: Query shape {query_embs.shape}, Doc shape {doc_embs.shape} on device {self.device}")
            logger.debug(f"  Sim matrix shape: {sim_matrix.shape}, Max scores per query token shape: {max_sim_scores_per_query_token.shape}, Final sum: {total_score.item()}")
        return total_score

    def classify_text(self, text: str) -> Dict[str, Any]:
        if not self.model: raise RuntimeError("ColBERT model not loaded.")
        logger.debug(f"ColBERT classifying text: '{text[:100]}...'")
        if not self.reference_token_embeddings_by_class and not self.reference_texts_for_model:
            logger.warning("ColBERT has no reference texts or embeddings loaded. Cannot classify.")
            return {"error": "ColBERT references not configured", "predicted_class": "N/A", "scores_by_class": {}}
        if not self.reference_token_embeddings_by_class and self.reference_texts_for_model:
            logger.error("ColBERT references exist but embeddings are missing. Please check setup.")
            return {"error": "ColBERT reference embeddings missing", "predicted_class": "N/A", "scores_by_class": {}}

        if not text.strip():
            logger.debug("ColBERT classify_text: Input text is empty or whitespace.")
            return {"input_text": text, "error": "Empty input text", "predicted_class": "N/A", "scores_by_class": {}}

        query_embs_list = self._get_token_embeddings_batched([text])
        if not query_embs_list:
            logger.error(f"Failed to generate embeddings for input text: '{text[:100]}...'")
            return {"input_text": text, "error": "Failed to generate embeddings for input text", "predicted_class": "N/A", "scores_by_class": {}}
        query_embs_gpu = query_embs_list[0]
        logger.debug(f"Query embedding shape for input text: {query_embs_gpu.shape}")

        scores = {}
        for cls_name, ref_embs_list_cpu in self.reference_token_embeddings_by_class.items():
            logger.debug(f"Calculating score for class: '{cls_name}' which has {len(ref_embs_list_cpu)} reference embeddings.")
            if not ref_embs_list_cpu:
                scores[cls_name] = 0.0
                logger.debug(f"  Class '{cls_name}' has no reference embeddings, score set to 0.0")
                continue

            class_total_score = 0.0
            num_valid_references_for_class = 0
            for idx, ref_cpu_embs in enumerate(ref_embs_list_cpu):
                if ref_cpu_embs.numel() == 0:
                    logger.debug(f"  Skipping empty reference embedding {idx} for class '{cls_name}'.")
                    continue
                if ref_cpu_embs.ndim == 1:
                    logger.warning(f"Skipping 1D reference embedding {idx} for class {cls_name}. Shape: {ref_cpu_embs.shape}")
                    continue

                maxsim_score = self._colbert_maxsim(query_embs_gpu, ref_cpu_embs.to(self.device)).item()
                logger.debug(f"  MaxSim score with ref {idx} ('{self.reference_texts_for_model.get(cls_name, ['N/A'])[idx][:30]}...') for class '{cls_name}': {maxsim_score:.4f}")
                class_total_score += maxsim_score
                num_valid_references_for_class +=1

            if num_valid_references_for_class > 0:
                scores[cls_name] = class_total_score / num_valid_references_for_class
                logger.debug(f"  Average MaxSim score for class '{cls_name}': {scores[cls_name]:.4f} (from {num_valid_references_for_class} refs)")
            else:
                scores[cls_name] = 0.0
                logger.debug(f"  No valid reference embeddings found for class '{cls_name}' after filtering, score set to 0.0")

        pred_cls = max(scores, key=scores.get) if scores else "N/A"
        class_desc = self.CLASS_DESCRIPTIONS.get(pred_cls, "Description not available") if pred_cls != "N/A" else "N/A"
        logger.debug(f"Predicted class: '{pred_cls}'. All scores: {scores}")

        return {"input_text": text, "predicted_class": pred_cls,
                "class_description": class_desc,
                "scores_by_class (avg_maxsim)": scores}

    def finetune(self, reference_jsonl_path: Path, output_model_dir: Path, base_model_id: str = DEFAULT_MODEL_ID,
                 epochs: int = 3, learning_rate: float = 1e-5, batch_size: int = 4, triplet_margin: float = 0.2):
        self.triplet_margin = triplet_margin 
        logger.info(f"Starting ColBERT fine-tuning. Base: {base_model_id}, Output: {output_model_dir}")
        logger.debug(f"Fine-tuning params: epochs={epochs}, lr={learning_rate}, batch_size={batch_size}, triplet_margin={triplet_margin}")
        output_model_dir.mkdir(parents=True, exist_ok=True)
        self._load_custom_references_from_jsonl(reference_jsonl_path)
        if not self.reference_texts_for_model: logger.error("No ColBERT refs for fine-tuning from JSONL. Aborting."); return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"ColBERT fine-tuning on device: {self.device}")
        logger.debug(f"HF_TOKEN env var for ColBERT finetune: {os.getenv('HF_TOKEN')}")
        logger.debug(f"HUGGING_FACE_HUB_TOKEN env var for ColBERT finetune: {os.getenv('HUGGING_FACE_HUB_TOKEN')}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None: self.tokenizer.add_special_tokens({'pad_token': self.DEFAULT_PAD_TOKEN})
        self.model = AutoModel.from_pretrained(base_model_id).to(self.device) 
        if self.model and self.tokenizer and self.tokenizer.pad_token_id is not None and self.model.config and self.tokenizer.pad_token_id >= self.model.config.vocab_size:
             self.model.resize_token_embeddings(len(self.tokenizer))
        if not self.model: logger.error("ColBERT model could not be loaded for fine-tuning. Aborting."); return
        logger.debug(f"Base model '{base_model_id}' and tokenizer loaded for fine-tuning.")

        triplets = self._prepare_triplets_for_finetuning()
        if not triplets: logger.error("No triplets generated for ColBERT fine-tuning. Aborting."); return
        logger.info(f"Prepared {len(triplets)} triplets for fine-tuning.")
        if triplets: logger.debug(f"Sample triplet 0: Anchor='{triplets[0][0][:50]}...', Pos='{triplets[0][1][:50]}...', Neg='{triplets[0][2][:50]}...'")

        class TripletDataset(Dataset):
            def __init__(self, d): self.d = d
            def __len__(self): return len(self.d)
            def __getitem__(self, i): return self.d[i]

        def collate_fn(b): return ([x[0] for x in b], [x[1] for x in b], [x[2] for x in b])
        dataloader = DataLoader(TripletDataset(triplets), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        logger.debug(f"Triplet DataLoader and AdamW optimizer (lr={learning_rate}) created.")

        self.model.train()
        epoch_acc = 0 # Initialize epoch_acc
        batches_processed_acc = 0 # Initialize batches_processed_acc for accuracy calculation

        for epoch in range(epochs):
            logger.info(f"ColBERT Fine-tune Epoch {epoch+1}/{epochs}"); total_loss_epoch = 0; batches_processed = 0
            epoch_acc = 0 # Reset for each epoch
            batches_processed_acc = 0 # Reset for each epoch

            prog_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Fine-tuning ColBERT", leave=False, disable=logger.level > logging.INFO)
            for batch_idx, (anchor_txts, pos_txts, neg_txts) in enumerate(prog_bar):
                if stop_signal_received: logger.info("ColBERT fine-tuning interrupted by signal."); break
                optimizer.zero_grad()

                all_triplet_texts = anchor_txts + pos_txts + neg_txts
                if batch_idx == 0: logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}: Processing {len(anchor_txts)} triplets. Total texts in batch: {len(all_triplet_texts)}")
                all_embs = self._get_token_embeddings_batched(all_triplet_texts, batch_size=len(all_triplet_texts), enable_grad=True)

                n_actual_triplets_in_batch = len(anchor_txts)
                anc_e = all_embs[:n_actual_triplets_in_batch]
                pos_e = all_embs[n_actual_triplets_in_batch : 2*n_actual_triplets_in_batch]
                neg_e = all_embs[2*n_actual_triplets_in_batch:]
                if batch_idx == 0: logger.debug(f"  Embeddings split: {len(anc_e)} anchors, {len(pos_e)} positives, {len(neg_e)} negatives.")

                accumulated_batch_loss = []
                current_batch_correct = 0
                current_batch_total_valid_triplets = 0

                for i in range(n_actual_triplets_in_batch):
                    current_anchor_emb = anc_e[i]
                    current_pos_emb = pos_e[i]
                    current_neg_emb = neg_e[i]

                    if current_anchor_emb.size(0) == 0 or current_pos_emb.size(0) == 0 or current_neg_emb.size(0) == 0:
                        logger.debug(f"Skipping triplet {i} in batch {batch_idx} due to empty embeddings (A:{current_anchor_emb.shape}, P:{current_pos_emb.shape}, N:{current_neg_emb.shape}). Anchor text: '{anchor_txts[i][:50]}...'")
                        continue
                    
                    current_batch_total_valid_triplets +=1
                    score_pos = self._colbert_maxsim(current_anchor_emb, current_pos_emb)
                    score_neg = self._colbert_maxsim(current_anchor_emb, current_neg_emb)
                    triplet_loss = F.relu(triplet_margin - score_pos + score_neg)
                    accumulated_batch_loss.append(triplet_loss)
                    if batch_idx == 0 and i < 2: logger.debug(f"  Triplet {i}: score_pos={score_pos.item():.4f}, score_neg={score_neg.item():.4f}, loss={triplet_loss.item():.4f}")
                    
                    with torch.no_grad():
                         if (score_pos > score_neg + self.triplet_margin): # Check accuracy condition
                            current_batch_correct +=1
                
                if current_batch_total_valid_triplets > 0:
                    epoch_acc += current_batch_correct / current_batch_total_valid_triplets # Add batch accuracy to epoch accuracy
                    batches_processed_acc +=1


                if accumulated_batch_loss:
                    mean_batch_loss = torch.stack(accumulated_batch_loss).mean()
                    mean_batch_loss.backward()
                    optimizer.step()

                    total_loss_epoch += mean_batch_loss.item()
                    batches_processed += 1
                    batch_accuracy_display = (current_batch_correct / current_batch_total_valid_triplets) * 100 if current_batch_total_valid_triplets > 0 else "N/A"
                    prog_bar.set_postfix({
                        'loss': mean_batch_loss.item(),
                        'acc': f"{batch_accuracy_display:.1f}%" if isinstance(batch_accuracy_display, float) else batch_accuracy_display
                    })
                    if batch_idx % (len(dataloader)//5 +1) == 0: logger.debug(f"  Batch {batch_idx} mean loss: {mean_batch_loss.item():.4f}")
                else:
                    prog_bar.set_postfix({'loss': 0.0, 'skipped_batch': True, 'acc': "N/A"})
                    logger.debug(f"  Batch {batch_idx} skipped as no valid triplets were processed.")

            if stop_signal_received: break
            avg_epoch_loss = total_loss_epoch / batches_processed if batches_processed else 0
            avg_acc_epoch = (epoch_acc / batches_processed_acc) * 100 if batches_processed_acc > 0 else 0
            logger.info(f"ColBERT Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_acc_epoch:.1f}%")

        self.model.eval()
        logger.info("ColBERT fine-tuning finished.");
        self._save_fine_tuned_model_assets(output_model_dir, base_model_id)
        self.model_id_or_path = str(output_model_dir.resolve())
        self.is_fine_tuned = True
        logger.info(f"Re-setting up ColBERT with fine-tuned model from {self.model_id_or_path} and recomputing reference embeddings.")
        self.setup_model_and_references(cache_dir=output_model_dir, force_recompute_embeddings=True)

    def _save_fine_tuned_model_assets(self, save_dir: Path, base_model_id_used: str):
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving fine-tuned ColBERT assets to {save_dir}. Base model used for tuning: {base_model_id_used}")
        if self.model: self.model.save_pretrained(save_dir)
        if self.tokenizer: self.tokenizer.save_pretrained(save_dir)

        snapshot_path = save_dir / self.REFERENCE_TEXTS_SNAPSHOT_FILENAME
        logger.debug(f"Saving reference texts snapshot (used for this fine-tuning) to {snapshot_path}")
        with open(snapshot_path, 'w', encoding='utf-8') as f: json.dump(self.reference_texts_for_model, f, indent=2)

        config_data = {
            "base_model_id_used_for_finetuning": base_model_id_used,
            "finetuned_from_hf_id_or_path_before_this_finetune": self.model.config._name_or_path if self.model else "N/A", 
            "timestamp": time.time(),
            "source_reference_texts_summary": {cls: len(texts) for cls, texts in self.reference_texts_for_model.items()}
        }
        config_path = save_dir / self.COLBERT_CONFIG_FILENAME
        logger.debug(f"Saving ColBERT fine-tuning config to {config_path}: {config_data}")
        with open(config_path, 'w', encoding='utf-8') as f: json.dump(config_data, f, indent=2)
        logger.info(f"Fine-tuned ColBERT assets (model, tokenizer, references, config) saved to {save_dir}")

    def _prepare_triplets_for_finetuning(self) -> List[Tuple[str, str, str]]:
        import random
        triplets = []; class_names = list(self.reference_texts_for_model.keys())
        logger.debug(f"Preparing triplets for ColBERT fine-tuning from {len(class_names)} classes.")
        if len(class_names) < 2:
            logger.warning("Need at least 2 classes with reference texts for ColBERT triplet generation. No triplets generated.")
            return []

        for cls_name, texts in self.reference_texts_for_model.items():
            if len(texts) < 2:
                logger.debug(f"Skipping class '{cls_name}' for triplet generation: needs at least 2 examples, found {len(texts)}.")
                continue

            other_classes_with_texts = {cn: txts_list for cn, txts_list in self.reference_texts_for_model.items() if cn != cls_name and txts_list}
            if not other_classes_with_texts:
                logger.debug(f"Skipping class '{cls_name}' for triplet generation: no other classes with texts available for negatives.")
                continue
            logger.debug(f"Generating triplets for class '{cls_name}' ({len(texts)} texts). Other classes for negatives: {list(other_classes_with_texts.keys())}")

            for i, anchor_text in enumerate(texts):
                positive_candidates = texts[:i] + texts[i+1:]
                if not positive_candidates: continue 
                positive_text = random.choice(positive_candidates)

                negative_class_name = random.choice(list(other_classes_with_texts.keys()))
                negative_text = random.choice(other_classes_with_texts[negative_class_name])

                triplets.append((anchor_text, positive_text, negative_text))
                if len(triplets) < 5 and logger.level == logging.DEBUG: 
                     logger.debug(f"  Created triplet: A='{anchor_text[:30]}...', P='{positive_text[:30]}...', N='{negative_text[:30]}...' (from class '{negative_class_name}')")

        random.shuffle(triplets)
        logger.debug(f"Total {len(triplets)} triplets generated and shuffled.")
        return triplets

    @classmethod
    def load_from_directory(cls, model_directory: Path):
        logger.info(f"Loading fine-tuned ColBERT from: {model_directory}")
        if not model_directory.is_dir(): raise FileNotFoundError(f"ColBERT model dir not found: {model_directory}")

        cfg_p = model_directory / cls.COLBERT_CONFIG_FILENAME
        ref_p = model_directory / cls.REFERENCE_TEXTS_SNAPSHOT_FILENAME
        logger.debug(f"ColBERT load paths: Config='{cfg_p}', References Snapshot='{ref_p}'")

        if not cfg_p.exists(): raise FileNotFoundError(f"Missing ColBERT config ({cls.COLBERT_CONFIG_FILENAME}) in {model_directory}")
        if not ref_p.exists(): raise FileNotFoundError(f"Missing ColBERT reference snapshot ({cls.REFERENCE_TEXTS_SNAPSHOT_FILENAME}) in {model_directory}")

        instance = cls(str(model_directory.resolve())) 
        instance.is_fine_tuned = True
        logger.debug(f"ColBERT instance created for loading, model_id_or_path set to '{instance.model_id_or_path}', is_fine_tuned=True.")

        try:
            with open(ref_p, 'r', encoding='utf-8') as f:
                instance.reference_texts_for_model = json.load(f)
            num_loaded_refs = sum(len(v) for v in instance.reference_texts_for_model.values())
            num_loaded_classes = len(instance.reference_texts_for_model)
            logger.info(f"Loaded {num_loaded_refs} reference texts for {num_loaded_classes} classes for fine-tuned ColBERT from {ref_p}")
            logger.debug(f"Loaded reference text details: {{c: len(t) for c,t in instance.reference_texts_for_model.items()}}")
        except Exception as e:
            logger.error(f"Failed to load ColBERT reference snapshot from {ref_p}: {e}", exc_info=True)
            raise
        
        with open(cfg_p, 'r', encoding='utf-8') as f_cfg: 
            loaded_colbert_config = json.load(f_cfg)
            logger.debug(f"Loaded ColBERT config from {cfg_p}: {loaded_colbert_config}")

        instance.setup_model_and_references(cache_dir=model_directory, force_recompute_embeddings=False)
        logger.info(f"Fine-tuned ColBERT model and its references loaded successfully from {model_directory}")
        return instance

    def get_hardware_info(self) -> Dict[str, Any]:
        info = {"device": str(self.device), "cuda_available": torch.cuda.is_available(), "torch_version": torch.__version__, "transformers_version": transformers.__version__}
        if torch.cuda.is_available() and self.device and self.device.type == 'cuda':
             info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
        else: info["gpu_name"] = "N/A"
        try: import flash_attn; info["flash_attn_available"] = True
        except ImportError: info["flash_attn_available"] = False
        logger.debug(f"ColBERT Hardware Info collected: {info}")
        return info

class ClassificationAPI:
    def __init__(self, modernbert_model_dir: Optional[str],
                 host: str, port: int,
                 policy_config_path: Optional[str] = "policy_config.json"):
        self.modernbert_model_dir = modernbert_model_dir # Directory for the I/O validation model
        self.host = host
        self.port = port
        self.policy_config_path = policy_config_path
        self.api_policy_config: Dict[str, Any] = {}
        self.modernbert_classifier: Optional[ModernBERTClassifier] = None # I/O validation model instance
        self.colbert_reranker: Optional[ColBERTReranker] = None
        self.app = Flask(__name__)
        CORS(self.app)
        self.request_count = 0
        logger.debug(f"ClassificationAPI initialized: I/O_Validator_dir='{modernbert_model_dir}', host='{host}', port={port}, policy_path='{policy_config_path}'")

    def setup(self, serve_modernbert: bool, serve_colbert: bool,
              colbert_model_id_or_dir: Optional[str],
              colbert_custom_ref_jsonl: Optional[str],
              colbert_cache_dir: Optional[str]):
        logger.info("Setting up ClassificationAPI...")
        logger.debug(f"API Setup Params: serve_io_validator={serve_modernbert}, serve_colbert={serve_colbert}, "
                     f"colbert_model_id_or_dir='{colbert_model_id_or_dir}', "
                     f"colbert_custom_ref_jsonl='{colbert_custom_ref_jsonl}', "
                     f"colbert_cache_dir='{colbert_cache_dir}'")

        if self.policy_config_path:
            policy_file = Path(self.policy_config_path)
            logger.debug(f"Attempting to load API policy from: {policy_file}")
            if policy_file.exists():
                try:
                    with open(policy_file, 'r', encoding='utf-8') as f:
                        self.api_policy_config = json.load(f)
                    logger.info(f"Loaded API policy configuration from {self.policy_config_path}")
                    logger.debug(f"Loaded policy config (first 200 chars): {str(self.api_policy_config)[:200]}")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in policy config file {self.policy_config_path}: {e}", exc_info=True)
                    logger.warning("Running without API policies due to load error.")
                except Exception as e:
                    logger.error(f"Failed to load policy config from {self.policy_config_path}: {e}", exc_info=True)
                    logger.warning("Running without API policies due to load error.")
            else:
                logger.warning(f"Policy config file not found: {self.policy_config_path}. No API policies loaded.")
        else:
            logger.warning("No policy config path provided. API policies will not be enforced by /service/validate.")

        if serve_modernbert and self.modernbert_model_dir: # serve_modernbert flag enables the I/O validator
            logger.info(f"I/O Validator serving enabled. Attempting to load from: {self.modernbert_model_dir}")
            try:
                self.modernbert_classifier = ModernBERTClassifier.load(self.modernbert_model_dir)
                logger.info("I/O Validator API model loaded successfully.")
            except FileNotFoundError:
                 logger.error(f"I/O Validator model directory not found: {self.modernbert_model_dir}. I/O Validator API will not be available.")
                 serve_modernbert=False
            except Exception as e:
                logger.error(f"Failed to load I/O Validator for API from {self.modernbert_model_dir}: {e}", exc_info=True)
                serve_modernbert=False
        elif serve_modernbert and not self.modernbert_model_dir:
            logger.warning("Serve I/O Validator requested but no model_dir provided. I/O Validator API will not be available.")
            serve_modernbert = False

        if serve_colbert:
            logger.info(f"ColBERT sensitivity serving enabled. Model ID/Dir: '{colbert_model_id_or_dir}'")
            try:
                colbert_path_obj = Path(colbert_model_id_or_dir) if colbert_model_id_or_dir else None

                if colbert_path_obj and colbert_path_obj.is_dir():
                    logger.info(f"Attempting to load fine-tuned ColBERT from directory: {colbert_path_obj}")
                    self.colbert_reranker = ColBERTReranker.load_from_directory(colbert_path_obj)
                else:
                    base_id = colbert_model_id_or_dir or ColBERTReranker.DEFAULT_MODEL_ID
                    logger.info(f"Initializing ColBERT with base model ID: {base_id}")
                    self.colbert_reranker = ColBERTReranker(base_id)

                    if colbert_custom_ref_jsonl:
                        custom_ref_path = Path(colbert_custom_ref_jsonl)
                        logger.debug(f"Custom ColBERT reference JSONL provided: {custom_ref_path}")
                        if custom_ref_path.exists():
                            self.colbert_reranker._load_custom_references_from_jsonl(custom_ref_path)
                        else:
                            logger.warning(f"ColBERT custom reference JSONL not found: {custom_ref_path}. Using defaults or built-in if available.")
                    else: logger.debug("No custom ColBERT reference JSONL provided for API setup.")

                    cache_p = Path(colbert_cache_dir) if colbert_cache_dir else Path.home()/".cache"/"classifier_tool"/"api_colbert_cache"
                    model_name_for_cache = "".join(c if c.isalnum() or c in ['-','_','.'] else '_' for c in Path(base_id).name)
                    final_cache_dir_for_model = cache_p / model_name_for_cache
                    final_cache_dir_for_model.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"ColBERT (non-fine-tuned) cache directory for API: {final_cache_dir_for_model}")
                    self.colbert_reranker.setup_model_and_references(cache_dir=final_cache_dir_for_model)

                logger.info(f"ColBERT API model setup complete. Fine-tuned: {self.colbert_reranker.is_fine_tuned if self.colbert_reranker else 'N/A'}.")
            except FileNotFoundError as e:
                 logger.error(f"ColBERT model file/directory not found: {e}. ColBERT API will not be available.", exc_info=True)
                 serve_colbert=False
            except Exception as e:
                logger.error(f"Failed to setup ColBERT for API: {e}", exc_info=True)
                serve_colbert=False
        else: logger.debug("ColBERT sensitivity serving not enabled for API.")

        @self.app.route('/health', methods=['GET'])
        def health():
            logger.debug("Health check endpoint hit.")
            io_validator_ok = self.modernbert_classifier is not None and self.modernbert_classifier.model is not None
            colbert_ok = self.colbert_reranker is not None and self.colbert_reranker.model is not None

            status_details = {
                "io_validator_loaded": io_validator_ok, # Changed key name
                "colbert_loaded": colbert_ok,
                "colbert_is_fine_tuned": self.colbert_reranker.is_fine_tuned if colbert_ok else None,
                "colbert_reference_classes": list(self.colbert_reranker.reference_texts_for_model.keys()) if colbert_ok and self.colbert_reranker.reference_texts_for_model else []
            }
            logger.debug(f"Health status details: {status_details}")

            overall_status = "ok"
            policy_readiness = {"status": "not_applicable_no_policies" if not self.api_policy_config else "ok", "issues": []}

            if self.api_policy_config:
                all_policies_runnable = True
                for policy_name, policy_rules in self.api_policy_config.items():
                    if policy_rules.get("modernbert_io_validation") and not io_validator_ok: # Check io_validator_ok
                        all_policies_runnable = False; msg = f"Policy '{policy_name}' requires I/O Validator, which is not loaded."
                        policy_readiness["issues"].append(msg); logger.debug(f"Health check policy issue: {msg}")
                    if policy_rules.get("colbert_input_sensitivity") or policy_rules.get("colbert_output_sensitivity"):
                        if not colbert_ok:
                            all_policies_runnable = False; msg = f"Policy '{policy_name}' requires ColBERT, which is not loaded."
                            policy_readiness["issues"].append(msg); logger.debug(f"Health check policy issue: {msg}")
                        elif policy_rules.get("require_colbert_fine_tuned") and (not self.colbert_reranker or not self.colbert_reranker.is_fine_tuned):
                            all_policies_runnable = False; msg = f"Policy '{policy_name}' requires fine-tuned ColBERT, but current ColBERT is not fine-tuned or not loaded."
                            policy_readiness["issues"].append(msg); logger.debug(f"Health check policy issue: {msg}")

                if not all_policies_runnable and (io_validator_ok or colbert_ok):
                    overall_status = "degraded"; policy_readiness["status"] = "degraded"
                elif not all_policies_runnable and not io_validator_ok and not colbert_ok:
                    overall_status = "error"; policy_readiness["status"] = "error_models_unavailable"
                elif not (io_validator_ok or colbert_ok) and self.api_policy_config :
                    overall_status = "error"; policy_readiness["status"] = "error_models_unavailable"
            logger.debug(f"Health check final: overall_status='{overall_status}', policy_readiness='{policy_readiness['status']}'")

            return jsonify({
                "status": overall_status,
                "model_availability": status_details,
                "policy_config_loaded": bool(self.api_policy_config),
                "policy_model_readiness": policy_readiness
            })

        if serve_modernbert and self.modernbert_classifier : # serve_modernbert enables I/O Validator endpoint
            logger.debug("I/O Validator /modernbert/classify endpoint will be available.") # Endpoint name kept for consistency
            @self.app.route('/modernbert/classify', methods=['POST'])
            def mb_classify(): # Function name kept
                logger.debug(f"I/O Validator /modernbert/classify endpoint hit. Request Remote Addr: {request.remote_addr}")
                if not self.modernbert_classifier or not self.modernbert_classifier.model:
                    logger.error("/modernbert/classify called but I/O Validator model not loaded.")
                    return jsonify({"error":"I/O Validator model not loaded or available"}),503

                data = request.get_json()
                if not data: logger.debug("Request to /modernbert/classify is not JSON."); return jsonify({"error": "Request body must be JSON"}), 400
                logger.debug(f"Received data for /modernbert/classify: {str(data)[:200]}")

                input_text = data.get('input_text')
                output_text = data.get('output_text')

                if input_text is None:
                    logger.debug("/modernbert/classify: 'input_text' is missing.")
                    return jsonify({"error":"'input_text' is required"}),400

                try:
                    result = self.modernbert_classifier.classify_input_output_pair(input_text, output_text if output_text is not None else "")
                    logger.debug(f"/modernbert/classify result for I/O Validator: {result}")
                    return jsonify(result)
                except Exception as e:
                    logger.error(f"Error in /modernbert/classify (I/O Validator): {e}", exc_info=True)
                    return jsonify({"error":str(e)}),500
        elif serve_modernbert: logger.warning("I/O Validator serving was requested but classifier is not available. Endpoint /modernbert/classify will not be functional.")

        if serve_colbert and self.colbert_reranker and self.colbert_reranker.model:
            logger.debug("ColBERT /colbert/classify_sensitivity endpoint will be available.")
            @self.app.route('/colbert/classify_sensitivity', methods=['POST'])
            def cb_classify():
                logger.debug(f"ColBERT /classify_sensitivity endpoint hit. Request Remote Addr: {request.remote_addr}")
                if not self.colbert_reranker or not self.colbert_reranker.model :
                    logger.error("/colbert/classify_sensitivity called but model not loaded.")
                    return jsonify({"error":"ColBERT model not loaded or available"}),503

                data=request.get_json()
                if not data: logger.debug("Request to /colbert/classify_sensitivity is not JSON."); return jsonify({"error": "Request body must be JSON"}), 400
                logger.debug(f"Received data for /colbert/classify_sensitivity: {str(data)[:200]}")

                txt=data.get('text')
                if txt is None: logger.debug("/colbert/classify_sensitivity: 'text' field missing."); return jsonify({"error":"'text' field required"}),400
                if not isinstance(txt, str): logger.debug("/colbert/classify_sensitivity: 'text' field not a string."); return jsonify({"error": "'text' field must be a string"}), 400

                try:
                    result = self.colbert_reranker.classify_text(txt)
                    logger.debug(f"/colbert/classify_sensitivity result: {result}")
                    return jsonify(result)
                except Exception as e:
                    logger.error(f"Error in /colbert/classify_sensitivity: {e}", exc_info=True)
                    return jsonify({"error":str(e)}),500
        elif serve_colbert: logger.warning("ColBERT serving was requested but reranker is not available. Endpoint /colbert/classify_sensitivity will not be functional.")

        logger.debug("/service/validate endpoint will be available (if policies are configured).")
        @self.app.route('/service/validate', methods=['POST'])
        def service_validate():
            self.request_count += 1
            logger.debug(f"/service/validate endpoint hit (request #{self.request_count}). Request Remote Addr: {request.remote_addr}")
            payload = request.get_json()
            if not payload:
                logger.debug("Request to /service/validate is not JSON.")
                return jsonify({"error": "Request body must be JSON"}), 400
            logger.debug(f"Received payload for /service/validate: {str(payload)[:300]}")

            api_class = payload.get("api_class")
            input_text = payload.get("input_text")
            output_text = payload.get("output_text")

            if not api_class or input_text is None:
                logger.debug(f"/service/validate: Missing 'api_class' ('{api_class}') or 'input_text' (present: {input_text is not None}).")
                return jsonify({"error": "'api_class' and 'input_text' are required fields"}), 400

            response_data = {"request_id": f"req_{time.time_ns()}", "request_summary": {"api_class": api_class, "input_text_len": len(input_text), "output_text_len": len(output_text) if output_text else 0}}
            current_overall_status = "PASS"
            violations = []
            error_message_detail = None

            policy = self.api_policy_config.get(api_class)
            if not policy:
                logger.warning(f"/service/validate: API class '{api_class}' not found in policy configuration.")
                response_data["overall_status"] = "REJECT_INVALID_POLICY"
                response_data["error_message"] = f"API class '{api_class}' not found in policy configuration."
                return jsonify(response_data), 400
            logger.debug(f"Applying policy for API class '{api_class}': {str(policy)[:200]}")
            response_data["policy_applied_summary"] = {k:v for k,v in policy.items() if "classes" not in k} 

            # --- I/O Validation (formerly ModernBERT) ---
            if policy.get("modernbert_io_validation"): # Key name kept for policy file consistency
                logger.debug("Performing I/O validation as per policy.")
                if output_text is None:
                    current_overall_status = "ERROR"; error_message_detail = "output_text is required for I/O validation."
                    logger.debug(f"Policy error: {error_message_detail}")
                    response_data["modernbert_io_validation"] = {"status": "error_missing_output_text", "message": error_message_detail}
                elif not self.modernbert_classifier or not self.modernbert_classifier.model:
                    current_overall_status = "ERROR"; error_message_detail = "I/O Validator model required by policy is not loaded."
                    logger.error(f"Policy error: {error_message_detail}")
                    response_data["modernbert_io_validation"] = {"status": "error_model_not_loaded", "message": error_message_detail}
                else:
                    try:
                        io_result = self.modernbert_classifier.classify_input_output_pair(input_text, output_text)
                        logger.debug(f"I/O validation result: {io_result}")
                        response_data["modernbert_io_validation"] = io_result
                        if io_result.get("prediction") == 0: 
                            violation_msg = "IO_Validation: Predicted as inappropriate pair."
                            violations.append(violation_msg); logger.debug(f"Policy violation: {violation_msg}")
                    except Exception as e:
                        logger.error(f"Error during I/O validation for policy '{api_class}': {e}", exc_info=True)
                        current_overall_status = "ERROR"; error_message_detail = f"I/O Validator exception: {str(e)}"
                        response_data["modernbert_io_validation"] = {"status": "error_exception_in_model", "message": error_message_detail}

            if current_overall_status == "ERROR": 
                response_data["overall_status"] = "ERROR"; response_data["error_message"] = error_message_detail
                logger.error(f"/service/validate returning ERROR due to: {error_message_detail}")
                return jsonify(response_data), 500

            # --- ColBERT Input Sensitivity ---
            if policy.get("colbert_input_sensitivity"):
                logger.debug("Performing ColBERT Input Sensitivity validation as per policy.")
                if not self.colbert_reranker or not self.colbert_reranker.model:
                    current_overall_status = "ERROR"; error_message_detail = "ColBERT model required by policy for input check is not loaded."
                    logger.error(f"Policy error: {error_message_detail}")
                    response_data["colbert_input_sensitivity"] = {"status": "error_model_not_loaded", "message": error_message_detail}
                elif policy.get("require_colbert_fine_tuned") and not self.colbert_reranker.is_fine_tuned:
                    current_overall_status = "ERROR"; error_message_detail = "Fine-tuned ColBERT model required by policy for input check, but loaded ColBERT is not fine-tuned."
                    logger.error(f"Policy error: {error_message_detail}")
                    response_data["colbert_input_sensitivity"] = {"status": "error_model_not_fine_tuned", "message": error_message_detail}
                else:
                    try:
                        cb_input_result = self.colbert_reranker.classify_text(input_text)
                        logger.debug(f"ColBERT input sensitivity result: {cb_input_result}")
                        response_data["colbert_input_sensitivity"] = cb_input_result
                        predicted_class = cb_input_result.get("predicted_class")
                        allowed_classes = policy.get("allowed_colbert_input_classes")
                        disallowed_classes = policy.get("disallowed_colbert_input_classes")

                        if allowed_classes and predicted_class not in allowed_classes:
                            violation_msg = f"ColBERT_Input_Sensitivity: Predicted class '{predicted_class}' not in allowed list: {allowed_classes}."
                            violations.append(violation_msg); logger.debug(f"Policy violation: {violation_msg}")
                        if disallowed_classes and predicted_class in disallowed_classes:
                             violation_msg = f"ColBERT_Input_Sensitivity: Predicted class '{predicted_class}' is in disallowed list: {disallowed_classes}."
                             violations.append(violation_msg); logger.debug(f"Policy violation: {violation_msg}")
                    except Exception as e:
                        err_str = str(e).lower()
                        if "401" in err_str or "unauthorized" in err_str:
                            logger.error(f"Authentication failed for Hugging Face resource. Please check your HF_TOKEN environment variable or login status (huggingface-cli login).")
                            logger.error(f"Original error: {str(e)}", exc_info=logger.level <= logging.DEBUG)
                        elif "404" in err_str or "not found" in err_str or "repositorynotfound" in err_str:
                            logger.error(f"Model or resource not found on Hugging Face Hub or locally. Check spelling and availability.")
                            logger.error(f"Original error: {str(e)}", exc_info=logger.level <= logging.DEBUG)
                        else:
                            logger.error(f"Error during ColBERT input sensitivity for policy '{api_class}': {e}", exc_info=True)
                        current_overall_status = "ERROR"; error_message_detail = f"ColBERT input sensitivity exception: {str(e)}"
                        response_data["colbert_input_sensitivity"] = {"status": "error_exception_in_model", "message": error_message_detail}

            if current_overall_status == "ERROR": 
                response_data["overall_status"] = "ERROR"; response_data["error_message"] = error_message_detail
                logger.error(f"/service/validate returning ERROR due to: {error_message_detail}")
                return jsonify(response_data), 500

            # --- ColBERT Output Sensitivity ---
            if policy.get("colbert_output_sensitivity"):
                logger.debug("Performing ColBERT Output Sensitivity validation as per policy.")
                if output_text is None or not output_text.strip() :
                    violation_msg = "ColBERT_Output_Sensitivity: output_text is missing or empty, but required for this policy check."
                    violations.append(violation_msg); logger.debug(f"Policy violation (or skip): {violation_msg}") # May not be a strict violation but a skip
                    response_data["colbert_output_sensitivity"] = {"status": "skipped_missing_output_text", "message": "Non-empty output_text is required for colbert_output_sensitivity check."}
                elif not self.colbert_reranker or not self.colbert_reranker.model:
                    current_overall_status = "ERROR"; error_message_detail = "ColBERT model required by policy for output check is not loaded."
                    logger.error(f"Policy error: {error_message_detail}")
                    response_data["colbert_output_sensitivity"] = {"status": "error_model_not_loaded", "message": error_message_detail}
                elif policy.get("require_colbert_fine_tuned") and not self.colbert_reranker.is_fine_tuned:
                    current_overall_status = "ERROR"; error_message_detail = "Fine-tuned ColBERT model required by policy for output check, but loaded ColBERT is not fine-tuned."
                    logger.error(f"Policy error: {error_message_detail}")
                    response_data["colbert_output_sensitivity"] = {"status": "error_model_not_fine_tuned", "message": error_message_detail}
                else:
                    try:
                        cb_output_result = self.colbert_reranker.classify_text(output_text)
                        logger.debug(f"ColBERT output sensitivity result: {cb_output_result}")
                        response_data["colbert_output_sensitivity"] = cb_output_result
                        predicted_class = cb_output_result.get("predicted_class")
                        allowed_classes = policy.get("allowed_colbert_output_classes")
                        disallowed_classes = policy.get("disallowed_colbert_output_classes")

                        if allowed_classes and predicted_class not in allowed_classes:
                            violation_msg = f"ColBERT_Output_Sensitivity: Predicted class '{predicted_class}' not in allowed list: {allowed_classes}."
                            violations.append(violation_msg); logger.debug(f"Policy violation: {violation_msg}")
                        if disallowed_classes and predicted_class in disallowed_classes:
                             violation_msg = f"ColBERT_Output_Sensitivity: Predicted class '{predicted_class}' is in disallowed list: {disallowed_classes}."
                             violations.append(violation_msg); logger.debug(f"Policy violation: {violation_msg}")
                    except Exception as e:
                        logger.error(f"Error during ColBERT output sensitivity for policy '{api_class}': {e}", exc_info=True)
                        current_overall_status = "ERROR"; error_message_detail = f"ColBERT output sensitivity exception: {str(e)}"
                        response_data["colbert_output_sensitivity"] = {"status": "error_exception_in_model", "message": error_message_detail}
            
            if current_overall_status == "ERROR":
                response_data["overall_status"] = "ERROR"; response_data["error_message"] = error_message_detail
                if violations: response_data["violations_detected_before_error"] = violations
                logger.error(f"/service/validate returning ERROR due to: {error_message_detail}. Violations before error: {violations}")
                return jsonify(response_data), 500
            elif violations:
                response_data["overall_status"] = "REJECT_POLICY_VIOLATION"; response_data["violation_reasons"] = violations
                logger.info(f"/service/validate REJECT_POLICY_VIOLATION for '{api_class}'. Reasons: {violations}")
                return jsonify(response_data), 200 # Still 200 OK, but with rejection status
            else:
                response_data["overall_status"] = "PASS"
                logger.info(f"/service/validate PASS for '{api_class}'.")
                return jsonify(response_data), 200

    def run(self, production: bool = True):
        can_serve_anything = False
        if self.modernbert_classifier and self.modernbert_classifier.model:
            can_serve_anything = True; logger.info("I/O Validator direct endpoint (/modernbert/classify) will be available.")
        if self.colbert_reranker and self.colbert_reranker.model:
            can_serve_anything = True; logger.info("ColBERT direct endpoint (/colbert/classify_sensitivity) will be available.")
        if self.api_policy_config:
            can_serve_anything = True; logger.info("/service/validate endpoint will be available.")

        if not can_serve_anything:
            logger.error("No models loaded and no policies configured for the API. Nothing to serve. Exiting.")
            sys.exit(1)

        logger.info(f"Starting API server on http://{self.host}:{self.port}")
        if production:
            logger.info("Running in production mode with Waitress (threads=8).")
            serve(self.app, host=self.host, port=self.port, threads=8)
        else:
            logger.info("Running in development mode with Flask's built-in server (use --dev-server for this).")
            self.app.run(host=self.host, port=self.port, debug=True, use_reloader=False)

# --- CLI Command Functions ---
def train_command(args: argparse.Namespace):
    logger.info("Running 'train' command (I/O Validator)...")
    logger.debug(f"Train command args: {args}")
    dp = DataProcessor(args.data_path)
    if not dp.load_and_validate(): logger.error("Data load failed for I/O Validator training."); sys.exit(1)
    texts, labels = dp.prepare_classification_data(separator=args.separator, balance_classes=args.balance_classes)
    if not texts: logger.error("No training samples after preparation for I/O Validator."); sys.exit(1)
    split = dp.perform_train_test_split(texts, labels, test_size=args.test_size, random_state=args.random_state)
    if not split['train_texts']: logger.error("No train data after split for I/O Validator."); sys.exit(1)

    classifier = ModernBERTClassifier(
        model_dir=args.model_dir, 
        use_mlflow=args.use_mlflow,
        base_model_id_for_training=args.base_model_for_training
    )
    classifier.separator = args.separator
    classifier.setup() 
    classifier.train(split['train_texts'], split['train_labels'], split['test_texts'], split['test_labels'],
                     args.batch_size, args.learning_rate, args.epochs, args.gradient_accumulation_steps,
                     args.early_stopping_patience, args.warmup_ratio, args.weight_decay)
    logger.info("I/O Validator 'train' command finished.")
    if args.run_server_after_train:
        logger.info("Starting server after training as per --run-server-after-train flag...")
        api = ClassificationAPI(args.model_dir, args.host, args.port, policy_config_path=None)
        api.setup(serve_modernbert=True, serve_colbert=False, colbert_model_id_or_dir=None, colbert_custom_ref_jsonl=None, colbert_cache_dir=None)
        api.run(production=not args.dev_server)

def serve_command(args: argparse.Namespace):
    logger.info("Running 'serve' command...")
    logger.debug(f"Serve command args: {args}")
    api = ClassificationAPI(
        modernbert_model_dir=args.modernbert_model_dir,
        host=args.host,
        port=args.port,
        policy_config_path=args.policy_config_path
    )
    api.setup(serve_modernbert=args.serve_modernbert,
              serve_colbert=args.serve_colbert_sensitivity,
              colbert_model_id_or_dir=args.colbert_model_id_or_dir,
              colbert_custom_ref_jsonl=args.colbert_custom_ref_jsonl,
              colbert_cache_dir=args.colbert_cache_dir)
    api.run(production=not args.dev_server)

def predict_modernbert_command_cli(args: argparse.Namespace): # Command name kept for consistency
    logger.info("Running 'predict-modernbert' (I/O Validator classification)...")
    logger.debug(f"Predict-modernbert command args: {args}")
    try:
        classifier = ModernBERTClassifier.load(args.model_dir)
    except FileNotFoundError:
        logger.error(f"I/O Validator model directory not found: {args.model_dir}. Cannot run prediction.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load I/O Validator model from {args.model_dir}: {e}", exc_info=True)
        sys.exit(1)

    if args.input_text is None or args.output_to_classify is None:
        logger.error("--input-text and --output-to-classify (can be empty string) are required for predict-modernbert.")
        sys.exit(1)
    result = classifier.classify_input_output_pair(args.input_text, args.output_to_classify)
    print(json.dumps(result, indent=2))

def rerank_data_classify_command_cli(args: argparse.Namespace):
    logger.info(f"Running 'rerank-data-classify' (ColBERT Sensitivity Classifier)...")
    logger.debug(f"Rerank-data-classify command args: {args}")
    if not args.text_to_classify: logger.error("--text-to-classify required."); sys.exit(1)
    try:
        reranker: ColBERTReranker
        cb_model_path = Path(args.colbert_model_dir) if args.colbert_model_dir else None

        cache_p = Path(args.cache_dir) if args.cache_dir else Path.home()/".cache"/"classifier_tool"/"colbert_cli_cache"
        cache_p.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ColBERT CLI cache base directory: {cache_p}")

        if cb_model_path and cb_model_path.is_dir():
            logger.info(f"Loading fine-tuned ColBERT from directory: {cb_model_path}")
            reranker = ColBERTReranker.load_from_directory(cb_model_path)
        else:
            base_id = args.colbert_model_id_or_dir or ColBERTReranker.DEFAULT_MODEL_ID
            logger.info(f"Initializing ColBERT with base model ID: {base_id}")
            reranker = ColBERTReranker(base_id)
            if args.custom_reference_jsonl:
                custom_ref_path = Path(args.custom_reference_jsonl)
                logger.debug(f"Custom reference JSONL provided for ColBERT CLI: {custom_ref_path}")
                if custom_ref_path.exists():
                    reranker._load_custom_references_from_jsonl(custom_ref_path)
                else:
                    logger.warning(f"Custom reference JSONL not found: {custom_ref_path}. Using defaults or built-ins.")

            model_name_for_cache = "".join(c if c.isalnum() or c in ['-','_','.'] else '_' for c in Path(base_id).name)
            final_cache_dir_for_model = cache_p / model_name_for_cache
            final_cache_dir_for_model.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Cache directory for this ColBERT model ('{base_id}'): {final_cache_dir_for_model}")
            reranker.setup_model_and_references(cache_dir=final_cache_dir_for_model)

        result = reranker.classify_text(args.text_to_classify)
        print(json.dumps(result, indent=2))
    except FileNotFoundError as e:
        logger.error(f"ColBERT model or reference file not found: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"ColBERT classification error: {e}", exc_info=True)
        sys.exit(1)

def finetune_colbert_command_cli(args: argparse.Namespace):
    logger.info("Running 'finetune-colbert'...")
    logger.debug(f"Finetune-colbert command args: {args}")
    ref_jsonl = Path(args.reference_jsonl)
    output_dir = Path(args.output_model_dir)
    if not ref_jsonl.exists(): logger.error(f"Reference JSONL not found: {ref_jsonl}"); sys.exit(1)

    reranker = ColBERTReranker(args.base_model_id)
    reranker.finetune(
        reference_jsonl_path=ref_jsonl,
        output_model_dir=output_dir,
        base_model_id=args.base_model_id,
        triplet_margin=args.triplet_margin,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )

def check_hardware_command(args: argparse.Namespace):
    logger.debug(f"Check-hardware command args: {args}")
    logger.info("--- I/O Validator Hardware Info ---")
    mb_checker = ModernBERTClassifier() # Creates instance with default (or no specific base for just hardware check)
    mb_checker.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device directly
    info_mb = mb_checker.get_hardware_info()
    for k,v in info_mb.items(): logger.info(f"I/O Validator - {k.replace('_',' ').capitalize()}: {v}")

    logger.info("--- ColBERT Sensitivity Classifier Hardware Info ---")
    cb_checker = ColBERTReranker()
    cb_checker.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device directly
    info_cb = cb_checker.get_hardware_info()
    for k,v in info_cb.items(): logger.info(f"ColBERT       - {k.replace('_',' ').capitalize()}: {v}")

def create_example_command(args: argparse.Namespace):
    logger.info(f"Creating example files in {args.output_dir}...")
    logger.debug(f"Create-example command args: {args}")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    mb_ex_path = out_dir / "sample_modernbert_training.jsonl" # Filename kept for consistency
    with open(mb_ex_path, 'w', encoding='utf-8') as f:
        json.dump({"input": "What is the capital of France?", "output_good_sample": "Paris is the capital of France.", "output_bad_sample": "Berlin is the capital of France."}, f); f.write("\n")
        json.dump({"input": "Is this product safe for children?", "output_good_sample": "This product is designed for ages 12 and up.", "output_bad_sample": "Yes, it's perfectly fine for toddlers."}, f); f.write("\n")
        json.dump({"input": "This movie was fantastic!", "label": 1}, f); f.write("\n")
        json.dump({"input": "I really disliked the service.", "label": 0}, f); f.write("\n")
    logger.info(f"Created I/O Validator (formerly ModernBERT) example training data: {mb_ex_path}")

    cb_ex_path = out_dir / "sample_colbert_references.jsonl"
    with open(cb_ex_path, 'w', encoding='utf-8') as f:
        for cn, exs in ColBERTReranker.BUILTIN_REFERENCE_TEXTS_BY_CLASS.items():
            if exs :
                for ex_text in exs[:2]: # Take first two examples from built-in
                    json.dump({"text": ex_text, "class_name": cn}, f); f.write("\n")
        # Add a couple of truly custom examples not in built-ins
        json.dump({"text": "Internal project codename 'Bluebird' launch plan details.", "class_name": "Custom_Project_Secrets"},f); f.write("\n")
        json.dump({"text": "Client support email: helpdesk@sensitivecorp.example", "class_name": "Custom_Client_Contact_Sensitive"},f); f.write("\n")
    logger.info(f"Created ColBERT example references: {cb_ex_path}")

    policy_ex_path = out_dir / "sample_policy_config.json"
    # Updated example policy configuration
    example_policy_config_for_create_example = {
      "DemoClass_IO_Validation_Only": {
        "description": "Demo: Only I/O validation (using the fine-tuned standard Transformer model like DeBERTa).",
        "modernbert_io_validation": True # This key corresponds to the I/O validation model
      },
       "DemoClass_InputSensitive_NoPII": {
        "description": "Demo: I/O Validation (e.g., DeBERTa) + ColBERT Input (base ColBERT model ok). Input must not be PII.",
        "modernbert_io_validation": True, # For I/O validation model
        "colbert_input_sensitivity": True,
        "require_colbert_fine_tuned": False, # For ColBERT model
        "disallowed_colbert_input_classes": ["Class 1: PII"]
      },
      "DemoClass_OutputPublic_FineTunedColBERT": {
        "description": "Demo: Output must be Public, requires fine-tuned ColBERT for sensitivity check. (I/O validation not explicitly enabled in this specific policy example).",
        "modernbert_io_validation": False, # I/O validation can be independently toggled
        "colbert_output_sensitivity": True,
        "require_colbert_fine_tuned": True, # For ColBERT model
        "allowed_colbert_output_classes": ["Class 5: Public Data"]
      },
      "DemoClass_NoChecks": {
        "description": "Demo: No specific validation checks enabled. All traffic passes through unless other mechanisms apply.",
        "modernbert_io_validation": False,
        "colbert_input_sensitivity": False,
        "colbert_output_sensitivity": False
      }
    }
    with open(policy_ex_path, 'w', encoding='utf-8') as f:
        json.dump(example_policy_config_for_create_example, f, indent=2)
    logger.info(f"Created example policy config for 'create-example': {policy_ex_path}")

    readme_p = out_dir / "README_examples.md"
    readme_content = f"""# Example Data for Classifier Service Tool

This directory contains example files to help you get started with the tool.

*   `{mb_ex_path.name}`: Sample data for training the I/O Validation model (e.g., a model like DeBERTa fine-tuned for sequence pair classification).
    The term "modernbert" in the filename is kept for consistency with older versions, but it now refers to a standard Transformer model.
*   `{cb_ex_path.name}`: Sample reference texts for the ColBERT-style sensitivity classifier.
*   `{policy_ex_path.name}`: An example policy configuration file. The descriptions within reflect the use of a standard Transformer for I/O validation and a ColBERT-style model for sensitivity.

## Using the Example Files

### 1. I/O Validation Model Training Data (`{mb_ex_path.name}`)
Use this file with the `train` command:
```bash
python <your_script_name.py> train \\
    --data-path ./{out_dir.name}/{mb_ex_path.name} \\
    --model-dir ./models/my_io_validator \\
    --base-model-for-training "{ModernBERTClassifier.DEFAULT_MODEL_A_ID}" \\
    --epochs 1 --batch-size 2 --log-level INFO 
    # Add other training parameters as needed
```

### 2. ColBERT Sensitivity Classifier References (`{cb_ex_path.name}`)
Use this file with the `finetune-colbert` command or with `rerank-data-classify` (if using a base ColBERT model):
```bash
# Fine-tuning ColBERT
python <your_script_name.py> finetune-colbert \\
    --reference-jsonl ./{out_dir.name}/{cb_ex_path.name} \\
    --output-model-dir ./models/my_colbert_sensitivity_finetuned \\
    --base-model-id "{ColBERTReranker.DEFAULT_MODEL_ID}" \\
    --epochs 2 --batch-size 2 --log-level INFO
    # Add other fine-tuning parameters

# Using with a base ColBERT model for direct classification
python <your_script_name.py> rerank-data-classify \\
    --text-to-classify "This is some sample text." \\
    --colbert-model-id-or-dir "{ColBERTReranker.DEFAULT_MODEL_ID}" \\
    --custom-reference-jsonl ./{out_dir.name}/{cb_ex_path.name} \\
    --log-level INFO
```

### 3. Policy Configuration (`{policy_ex_path.name}`)
Use this file with the `serve` command to start the API with these example policies:
```bash
python <your_script_name.py> serve \\
    --policy-config-path ./{out_dir.name}/{policy_ex_path.name} \\
    --serve-modernbert --modernbert-model-dir ./models/my_io_validator \\
    --serve-colbert-sensitivity --colbert-model-id-or-dir ./models/my_colbert_sensitivity_finetuned \\
    --log-level INFO
    # Ensure the model paths point to trained models
```
Replace `<your_script_name.py>` with the actual name of your script (e.g., `classify.py`).
Adjust model paths and other parameters as per your setup.
"""
    with open(readme_p, 'w', encoding='utf-8') as f: f.write(readme_content.replace("<your_script_name.py>", Path(__file__).name))
    logger.info(f"Created README for examples: {readme_p}")
    logger.info("Example files created successfully.")

def cli_main_logic():
    """Contains the main CLI argument parsing and command execution logic."""
    setup_signal_handling()
    
    # Updated main script description
    parser = argparse.ArgumentParser(
        description="Standard Transformer-Based Classification and Reranking Service with Policy Enforcement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Set the logging level for the script console output."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Train I/O Validator (formerly ModernBERT)
    train_p = subparsers.add_parser("train", help="Train I/O validation binary classifier (e.g., fine-tune DeBERTa).")
    train_p.add_argument("--base-model-for-training", default=ModernBERTClassifier.DEFAULT_MODEL_A_ID, help="Hugging Face model ID to use as the base for fine-tuning the I/O validation model.")
    train_p.add_argument("--data-path", nargs='+', required=True, help="Path(s) to JSONL training data for I/O validator.")
    train_p.add_argument("--model-dir", default="models/io_validator_custom", help="Directory to save/load fine-tuned I/O validation model.") # Updated help
    train_p.add_argument("--epochs", type=int, default=3); train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--learning-rate", type=float, default=2e-5); train_p.add_argument("--test-size", type=float, default=0.15)
    train_p.add_argument("--random-state", type=int, default=42); train_p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_p.add_argument("--early-stopping-patience", type=int, default=0, help="Set to >0 to enable early stopping based on eval F1/accuracy.")
    train_p.add_argument("--warmup-ratio", type=float, default=0.1); train_p.add_argument("--weight-decay", type=float, default=0.01)
    train_p.add_argument("--separator", default=" [SEP] ", help="Separator token for input/output pairs in I/O validation model.")
    train_p.add_argument("--balance-classes", action=argparse.BooleanOptionalAction, default=True, help="Balance classes during data preparation for I/O validation model training.")
    train_p.add_argument("--use-mlflow", action="store_true"); train_p.add_argument("--run-server-after-train", action="store_true")
    train_p.add_argument("--host", default="0.0.0.0"); train_p.add_argument("--port", type=int, default=5000)
    train_p.add_argument("--dev-server", action="store_true", help="Run Flask dev server instead of Waitress (for run-server-after-train or serve).")
    train_p.set_defaults(func=train_command)

    # Serve API
    serve_p = subparsers.add_parser("serve", help="Start API server with I/O Validator, ColBERT Sensitivity, and Policy Validation.")
    serve_p.add_argument("--serve-modernbert", action="store_true", help="Enable I/O Validator model and its direct /modernbert/classify endpoint.") # Flag name kept for consistency
    serve_p.add_argument("--modernbert-model-dir", help="Directory of fine-tuned I/O Validator model (required if --serve-modernbert or policies need it).") # Help text updated
    serve_p.add_argument("--serve-colbert-sensitivity", action="store_true", help="Enable ColBERT sensitivity model and its direct /colbert/classify_sensitivity endpoint.")
    serve_p.add_argument("--colbert-model-id-or-dir", help="Hugging Face ID or local directory of ColBERT model (base or fine-tuned).")
    serve_p.add_argument("--colbert-custom-ref-jsonl", help="Path to JSONL file with custom reference texts for base ColBERT model.")
    serve_p.add_argument("--colbert-cache-dir", help="Cache directory for ColBERT base model embeddings and downloaded files.")
    serve_p.add_argument("--policy-config-path", default="policy_config.json", help="Path to API policy configuration JSON file for /service/validate endpoint.")
    serve_p.add_argument("--host", default="0.0.0.0"); serve_p.add_argument("--port", type=int, default=5000)
    serve_p.add_argument("--dev-server", action="store_true", help="Run Flask dev server instead of Waitress.")
    serve_p.set_defaults(func=serve_command)

    # Predict with I/O Validator CLI
    pred_mb_p = subparsers.add_parser("predict-modernbert", help="CLI for I/O Validator classification of an input/output pair.") # Help updated
    pred_mb_p.add_argument("--model-dir", required=True, help="Directory of the fine-tuned I/O Validator model.") # Help updated
    pred_mb_p.add_argument("--input-text", required=True, help="Input text part of the pair.")
    pred_mb_p.add_argument("--output-to-classify", required=True, help="Output text part to classify against the input (can be empty string).")
    pred_mb_p.set_defaults(func=predict_modernbert_command_cli)

    # Classify Sensitivity with ColBERT CLI
    rerank_p = subparsers.add_parser("rerank-data-classify", help="Classify data sensitivity of a text using ColBERT.")
    rerank_p.add_argument("--text-to-classify", required=True, help="Text to classify for sensitivity.")
    rerank_p.add_argument("--colbert-model-dir", help="Path to a directory containing a fine-tuned ColBERT model (takes precedence over --colbert-model-id-or-dir).")
    rerank_p.add_argument("--colbert-model-id-or-dir", help="HF ID or path to a base ColBERT model (used if --colbert-model-dir is not provided).")
    rerank_p.add_argument("--custom-reference-jsonl", help="Path to JSONL with custom reference texts for a base ColBERT model.")
    rerank_p.add_argument("--cache-dir", help="Base cache directory for ColBERT non-fine-tuned setups.")
    rerank_p.set_defaults(func=rerank_data_classify_command_cli)

    # Finetune ColBERT
    finetune_cb_p = subparsers.add_parser("finetune-colbert", help="Fine-tune a ColBERT model for sensitivity classification.")
    finetune_cb_p.add_argument("--reference-jsonl", required=True, help="Path to JSONL file with reference texts for sensitivity classes.")
    finetune_cb_p.add_argument("--output-model-dir", required=True, help="Directory to save the fine-tuned ColBERT model and its assets.")
    finetune_cb_p.add_argument("--base-model-id", default=ColBERTReranker.DEFAULT_MODEL_ID, help="Base ColBERT model ID from Hugging Face to fine-tune from.")
    finetune_cb_p.add_argument("--epochs", type=int, default=3); finetune_cb_p.add_argument("--learning-rate", type=float, default=1e-5)
    finetune_cb_p.add_argument("--batch-size", type=int, default=4); finetune_cb_p.add_argument("--triplet-margin", type=float, default=0.2)
    finetune_cb_p.set_defaults(func=finetune_colbert_command_cli)

    # Utilities
    subparsers.add_parser("check-hardware", help="Check hardware and library versions.").set_defaults(func=check_hardware_command)
    ex_p = subparsers.add_parser("create-example", help="Create example data files (for training, references, policy).")
    ex_p.add_argument("--output-dir", default="classifier_tool_examples", help="Directory to create example files in.")
    ex_p.set_defaults(func=create_example_command)

    args = parser.parse_args()

    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        logger.error(f"Invalid log level: {args.log_level}. Defaulting to INFO.")
        numeric_log_level = logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_log_level)
    for handler in root_logger.handlers:
        handler.setLevel(numeric_log_level)
    logger.info(f"Logging level set to {args.log_level} ({numeric_log_level}) by CLI argument.")

    if hasattr(args, 'func'):
        try:
            logger.debug(f"Executing command: {args.command} with resolved arguments: {vars(args)}")
            args.func(args)
        except Exception as e:
            err_msg_lower = str(e).lower()
            if "401" in err_msg_lower or "unauthorized" in err_msg_lower :
                logger.error(f"Authentication failed for Hugging Face model download. Please check your HF_TOKEN environment variable.")
            elif "404" in err_msg_lower or "not found" in err_msg_lower or "repositorynotfound" in err_msg_lower:
                 # Check if base_model_id attribute exists, common in training/finetuning commands
                model_id_in_error = args.base_model_id if hasattr(args, 'base_model_id') else \
                                   (args.base_model_for_training if hasattr(args, 'base_model_for_training') else \
                                   (args.colbert_model_id_or_dir if hasattr(args, 'colbert_model_id_or_dir') else "the specified model"))
                logger.error(f"Model '{model_id_in_error}' not found on Hugging Face Hub or locally. Check model ID spelling and availability.")
            else:
                logger.error(f"An error occurred while executing command '{args.command}': {str(e)}", exc_info=True)
            sys.exit(1)
        except KeyboardInterrupt: 
            logger.info(f"Command '{args.command}' interrupted by user (KeyboardInterrupt). Exiting.")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    if ensure_venv(): # Only proceed if we are in the correct venv or after re-execution
        cli_main_logic()
