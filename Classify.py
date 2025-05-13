#!/usr/bin/env python
"""
ModernBERT Classification as a Service & ColBERT Data Reranking/Classification

A self-installing CLI/Python tool that:
1. Fine-tunes ModernBERT for binary classification of input-output text pairs.
2. Deploys a RESTful API server for the fine-tuned ModernBERT.
3. Provides data sensitivity classification using a ColBERT model (default or fine-tuned)
   by comparing input against reference examples via MaxSim.
4. Optionally allows fine-tuning the ColBERT model on custom reference examples for sensitivity classification.
5. Implements a policy-driven /service/validate endpoint.
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

# Attempt to import PyTorch and Transformers early
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    from torch.optim import AdamW
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
    from transformers.optimization import get_linear_schedule_with_warmup
    from packaging import version
except ImportError:
    pass # Handled by ensure_venv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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
    is_in_target_venv = sys.prefix == venv_path

    if is_in_target_venv:
        logger.info(f"Running inside the '{VENV_DIR}' virtual environment.")
        return True

    logger.info(f"Not running inside the target '{VENV_DIR}' virtual environment (current: {sys.prefix}).")
    venv_exists = os.path.isdir(venv_path)

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

    if not os.path.exists(python_executable):
        logger.error(f"Python executable not found at '{python_executable}'. Venv creation might have failed.")
        sys.exit(1)
    if not os.path.exists(pip_executable):
        logger.error(f"Pip executable not found at '{pip_executable}'. Venv creation might have failed.")
        sys.exit(1)

    try:
        logger.info("Attempting to upgrade pip in the virtual environment...")
        pip_upgrade_cmd = [pip_executable, "install", "--upgrade", "pip"]
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
            subprocess.run(torch_install_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')

            cuda_check_script = "import torch; print(torch.cuda.is_available())"
            cuda_check_cmd = [python_executable, "-c", cuda_check_script]
            result = subprocess.run(cuda_check_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
            cuda_available = result.stdout.strip().lower() == "true"

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
    try:
        os.execv(python_executable, [python_executable, script_path] + sys.argv[1:])
    except OSError as e:
        logger.error(f"os.execv failed: {e}. Executable='{python_executable}', Script='{script_path}'", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during script restart attempt: {e}", exc_info=True)
        sys.exit(1)
    return False

if __name__ == "__main__" and "pytest" not in sys.modules:
    if '--help' not in sys.argv and '-h' not in sys.argv :
        called_command = sys.argv[1] if len(sys.argv) > 1 else ""
        if called_command not in ["--help", "-h"]:
             ensure_venv()

import numpy as np
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve

class DataProcessor:
    """Handles JSONL data loading for ModernBERT fine-tuning."""
    def __init__(self, jsonl_paths: Union[str, List[str]]):
        if isinstance(jsonl_paths, str): self.jsonl_paths = [Path(jsonl_paths)]
        else: self.jsonl_paths = [Path(p) for p in jsonl_paths]
        self.data_entries: List[Dict[str, Any]] = []
        self.stats = { "total_files_processed": 0, "total_lines_read": 0, "valid_entries": 0,
                       "invalid_entries_missing_fields": 0, "invalid_entries_wrong_type": 0,
                       "invalid_entries_json_decode_error": 0, "files_with_errors": set() }

    def load_and_validate(self) -> bool:
        logger.info(f"Loading ModernBERT training data from {len(self.jsonl_paths)} path(s)...")
        for file_path_obj in self.jsonl_paths:
            file_path_str = str(file_path_obj)
            if not file_path_obj.exists() or not file_path_obj.is_file():
                logger.warning(f"File not found: {file_path_str}. Skipping."); self.stats["files_with_errors"].add(file_path_str); continue
            self.stats["total_files_processed"] += 1
            try:
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        self.stats["total_lines_read"] += 1; line = line.strip()
                        if not line: continue
                        try:
                            item = json.loads(line)
                            if 'input' in item and 'output_good_sample' in item and 'output_bad_sample' in item:
                                if not all(isinstance(item[k], str) for k in ['input', 'output_good_sample', 'output_bad_sample']):
                                    self.stats["invalid_entries_wrong_type"] += 1; continue
                                self.data_entries.append(item); self.stats["valid_entries"] += 1
                            elif 'input' in item and 'label' in item:
                                if not isinstance(item['input'], str) or not isinstance(item['label'], int) or item['label'] not in (0,1):
                                    self.stats["invalid_entries_wrong_type"] += 1; continue
                                self.data_entries.append({'input': item['input'], 'label': item['label']}); self.stats["valid_entries"] += 1
                            else: self.stats["invalid_entries_missing_fields"] += 1
                        except json.JSONDecodeError: self.stats["invalid_entries_json_decode_error"] += 1; self.stats["files_with_errors"].add(file_path_str)
                        except Exception: self.stats["invalid_entries_json_decode_error"] += 1; self.stats["files_with_errors"].add(file_path_str)
            except Exception as e: logger.error(f"Error processing file {file_path_str}: {e}", exc_info=True); self.stats["files_with_errors"].add(file_path_str)
        logger.info(f"Data loading: {self.stats['valid_entries']} valid entries from {self.stats['total_files_processed']} files.")
        return bool(self.data_entries)

    def prepare_classification_data(self, separator: str = " [SEP] ", balance_classes: bool = True) -> Tuple[List[str], List[int]]:
        texts, labels = [], []
        for item in self.data_entries:
            if 'output_good_sample' in item:
                texts.append(f"{item['input'].strip()}{separator}{item['output_good_sample'].strip()}"); labels.append(1)
                texts.append(f"{item['input'].strip()}{separator}{item['output_bad_sample'].strip()}"); labels.append(0)
            elif 'label' in item: texts.append(item['input'].strip()); labels.append(item['label'])

        if balance_classes and labels:
            pos_indices = [i for i, lbl in enumerate(labels) if lbl == 1]
            neg_indices = [i for i, lbl in enumerate(labels) if lbl == 0]
            if not pos_indices or not neg_indices: logger.warning("Cannot balance: one class has no samples.")
            elif len(pos_indices) != len(neg_indices):
                import random
                min_samples = min(len(pos_indices), len(neg_indices))
                chosen_pos = random.sample(pos_indices, min_samples)
                chosen_neg = random.sample(neg_indices, min_samples)
                balanced_texts, balanced_labels = [], []
                for i in chosen_pos + chosen_neg: balanced_texts.append(texts[i]); balanced_labels.append(labels[i])
                combined = list(zip(balanced_texts, balanced_labels)); random.shuffle(combined)
                texts, labels = [list(t) for t in zip(*combined)] if combined else ([], [])
                logger.info(f"Classes balanced. New count: {len(texts)}")
        logger.info(f"Prepared {len(texts)} samples for ModernBERT classification.")
        return texts, labels

    def perform_train_test_split(self, texts: List[str], labels: List[int], test_size: float = 0.2, random_state: int = 42) -> Dict[str, List[Any]]:
        if not texts or not labels: return {'train_texts': [], 'train_labels': [], 'test_texts': [], 'test_labels': []}
        stratify_param = labels if len(set(labels)) >= 2 else None
        if test_size <= 0 or int(len(texts) * test_size) < 1 :
            return {'train_texts': texts, 'train_labels': labels, 'test_texts': [], 'test_labels': []}
        X_train, X_test, y_train, y_test = sklearn_train_test_split(texts, labels, test_size=test_size, stratify=stratify_param, random_state=random_state)
        return {'train_texts': X_train, 'train_labels': y_train, 'test_texts': X_test, 'test_labels': y_test}

class ModernBERTClassifier:
    DEFAULT_MAX_LENGTH = 256; DEFAULT_PAD_TOKEN = "[PAD]"
    def __init__(self, model_dir: str = "model_files", use_mlflow: bool = False):
        self.model_dir = Path(model_dir).resolve(); self.model_id = "answerdotai/ModernBERT-base"
        self.model: Optional[AutoModelForSequenceClassification] = None; self.tokenizer: Optional[AutoTokenizer] = None
        self.device: Optional[torch.device] = None; self.separator: str = " [SEP] "
        self.use_mlflow = use_mlflow
        if self.use_mlflow:
            try: import mlflow; self.mlflow_client = mlflow
            except ImportError: logger.warning("MLflow not found, disabling."); self.use_mlflow = False

    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModernBERT using device: {self.device}")
        has_flash_attn = False
        if self.device.type == 'cuda':
            try: import flash_attn; has_flash_attn = True; logger.info("Flash Attention 2 available for ModernBERT.")
            except ImportError: logger.info("Flash Attention 2 not found for ModernBERT.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None: self.tokenizer.add_special_tokens({'pad_token': self.DEFAULT_PAD_TOKEN})

        model_kwargs = {"num_labels": 2}
        if has_flash_attn and version.parse(torch.__version__) >= version.parse("2.0"): model_kwargs["attn_implementation"] = "flash_attention_2"

        try: self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, **model_kwargs)
        except Exception as e:
            if model_kwargs.get("attn_implementation") == "flash_attention_2":
                logger.warning("Failed load ModernBERT w/ Flash Attn. Retrying default."); model_kwargs.pop("attn_implementation")
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, **model_kwargs)
            else: raise e
        if self.model and self.tokenizer.pad_token_id is not None and self.tokenizer.pad_token_id >= self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.model: self.model.to(self.device)
        return self

    def _create_dataloader(self, texts: List[str], labels: Optional[List[int]], batch_size: int, shuffle: bool = False):
        if not self.tokenizer: raise RuntimeError("ModernBERT tokenizer not initialized.")
        encodings = self.tokenizer(texts, truncation=True, padding="max_length", max_length=self.DEFAULT_MAX_LENGTH, return_tensors="pt")
        ds_input_ids, ds_attn_mask = encodings.input_ids, encodings.attention_mask
        if labels is not None:
            if len(ds_input_ids) != len(labels): # Ensure alignment
                min_len = min(len(ds_input_ids), len(labels))
                ds_input_ids, ds_attn_mask, labels = ds_input_ids[:min_len], ds_attn_mask[:min_len], labels[:min_len]
            dataset = TensorDataset(ds_input_ids, ds_attn_mask, torch.tensor(labels, dtype=torch.long))
        else: dataset = TensorDataset(ds_input_ids, ds_attn_mask)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def train(self, train_texts: List[str], train_labels: List[int], eval_texts: Optional[List[str]] = None, eval_labels: Optional[List[int]] = None,
              batch_size: int = 8, learning_rate: float = 2e-5, epochs: int = 3, gradient_accumulation_steps: int = 1,
              early_stopping_patience: int = 0, warmup_ratio: float = 0.1, weight_decay: float = 0.01):
        if not self.model or not self.tokenizer: raise RuntimeError("ModernBERT model/tokenizer not setup.")
        train_loader = self._create_dataloader(train_texts, train_labels, batch_size, shuffle=True)
        eval_loader = self._create_dataloader(eval_texts, eval_labels, batch_size) if eval_texts and eval_labels else None

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * warmup_ratio), num_training_steps=total_steps)
        best_eval_metric, epochs_no_improve = -float('inf'), 0

        logger.info("Starting ModernBERT training...")
        for epoch in range(epochs):
            if stop_signal_received: logger.info("ModernBERT training interrupted."); break
            self.model.train(); total_loss, train_steps = 0, 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training (ModernBERT)", leave=False)
            for step, batch_data in enumerate(progress_bar): # Renamed batch to batch_data
                b_input_ids, b_attn_mask, b_labels = [b.to(self.device) for b in batch_data]
                loss = self.model(b_input_ids, attention_mask=b_attn_mask, labels=b_labels).loss / gradient_accumulation_steps
                loss.backward(); total_loss += loss.item() * gradient_accumulation_steps; train_steps +=1
                if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad()
                progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
            logger.info(f"Epoch {epoch+1} (ModernBERT) avg train loss: {total_loss/train_steps if train_steps else 0:.4f}")

            if eval_loader:
                metrics = self._evaluate_from_loader(eval_loader)
                logger.info(f"Epoch {epoch+1} (ModernBERT) Eval: {metrics}")
                current_metric = metrics.get('f1', metrics.get('accuracy', 0.0))
                if current_metric > best_eval_metric: best_eval_metric, epochs_no_improve = current_metric, 0; self._save_model("best")
                else: epochs_no_improve += 1
                if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience: logger.info("ModernBERT early stopping."); break

        logger.info("ModernBERT training finished."); self._save_model("latest")
        if early_stopping_patience > 0 and (self.model_dir / "best").exists():
            logger.info(f"Loading best ModernBERT model (metric: {best_eval_metric:.4f}).")
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir / "best").to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "best") # Also load tokenizer from best
            except Exception as e: logger.error(f"Failed to load best ModernBERT: {e}", exc_info=True)
        return self

    def _evaluate_from_loader(self, eval_loader: DataLoader) -> Dict[str, float]:
        if not self.model: raise RuntimeError("ModernBERT model not initialized for evaluation.")
        self.model.eval(); total_loss, eval_steps = 0,0; all_logits, all_labels = [], []
        with torch.no_grad():
            for batch_data in tqdm(eval_loader, desc="Evaluating (ModernBERT)", leave=False): # Renamed batch to batch_data
                b_input_ids, b_attn_mask, b_labels = [b.to(self.device) for b in batch_data]
                outputs = self.model(b_input_ids, attention_mask=b_attn_mask, labels=b_labels)
                total_loss += outputs.loss.item(); eval_steps += 1
                all_logits.append(outputs.logits.cpu().numpy()); all_labels.append(b_labels.cpu().numpy())
        preds = np.argmax(np.concatenate(all_logits), axis=1)
        true_labels = np.concatenate(all_labels)
        p, r, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
        return {'loss': total_loss/eval_steps if eval_steps else 0, 'accuracy': accuracy_score(true_labels, preds), 'precision': p, 'recall': r, 'f1': f1}

    def predict(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        if not self.model or not self.tokenizer: raise RuntimeError("ModernBERT model/tokenizer not setup.")
        self.model.eval(); predictions_data = []
        predict_loader = self._create_dataloader(texts, None, batch_size)
        text_idx_offset = 0
        with torch.no_grad():
            for batch_data in tqdm(predict_loader, desc="Predicting (ModernBERT)", leave=False): # Renamed batch to batch_data
                b_input_ids, b_attn_mask = batch_data[0].to(self.device), batch_data[1].to(self.device)
                logits = self.model(b_input_ids, attention_mask=b_attn_mask).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                for i in range(len(preds)):
                    current_text = texts[text_idx_offset + i]
                    parts = current_text.split(self.separator, 1)
                    input_part = parts[0]
                    output_part = parts[1] if len(parts) > 1 else ""
                    predictions_data.append({'prediction': int(preds[i]), 'probability_positive': float(probs[i][1]),
                                             'input_text': input_part, 'output_text': output_part})
                text_idx_offset += len(preds)
        return predictions_data

    def classify_input_output_pair(self, input_text: str, output_text: str) -> Dict[str, Any]:
        # Ensure output_text is a string, even if empty, for concatenation
        safe_output_text = output_text if output_text is not None else ""
        return self.predict([f"{input_text}{self.separator}{safe_output_text}"])[0]


    def _save_model(self, suffix: str = ""):
        if not self.model or not self.tokenizer: return
        save_path = self.model_dir / suffix if suffix else self.model_dir
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving ModernBERT model to {save_path}...")
        self.model.save_pretrained(save_path); self.tokenizer.save_pretrained(save_path)
        with open(save_path / "model_config.json", 'w', encoding='utf-8') as f: json.dump({"separator": self.separator}, f)

    @classmethod
    def load(cls, model_dir: str, use_mlflow_during_load: bool = False):
        model_dir_path = Path(model_dir);
        if not model_dir_path.exists() or not model_dir_path.is_dir(): # Check if it's actually a directory
             raise FileNotFoundError(f"ModernBERT model directory not found: {model_dir_path}")

        # Ensure base model ID is set correctly before trying to load specifics from the directory
        instance = cls(model_dir=str(model_dir_path), use_mlflow=use_mlflow_during_load)
        instance.model_id = str(model_dir_path.resolve()) # Point to the local directory for loading
        instance.setup() # This will now try to load from instance.model_id (the directory)

        cfg_path = model_dir_path / "model_config.json"
        if cfg_path.exists():
            with open(cfg_path, 'r', encoding='utf-8') as f: instance.separator = json.load(f).get("separator", instance.separator)
        logger.info(f"ModernBERT model loaded from {model_dir_path}.")
        return instance

    def get_hardware_info(self) -> Dict[str, Any]:
        info = {"device": str(self.device), "cuda": torch.cuda.is_available(), "torch": torch.__version__, "transformers": transformers.__version__}
        if torch.cuda.is_available() and self.device and self.device.type == 'cuda':
             info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
        else: info["gpu_name"] = "N/A"
        try: import flash_attn; info["flash_attn"] = True
        except ImportError: info["flash_attn"] = False
        return info

class ColBERTReranker:
    DEFAULT_MAX_LENGTH = 512; DEFAULT_PAD_TOKEN = "[PAD]"; DEFAULT_MODEL_ID = "lightonai/GTE-ModernColBERT-v1"
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
        self.is_fine_tuned: bool = False # Attribute to track if the model is fine-tuned
        if not Path(self.model_id_or_path).is_dir(): # Default to built-in if not loading a fine-tuned dir
            self.reference_texts_for_model = copy.deepcopy(self.BUILTIN_REFERENCE_TEXTS_BY_CLASS)
        else: # If it's a directory, it might be a fine-tuned model path
             # is_fine_tuned will be set more definitively in load_from_directory
            config_path = Path(self.model_id_or_path) / self.COLBERT_CONFIG_FILENAME
            if config_path.exists():
                self.is_fine_tuned = True


    def _load_custom_references_from_jsonl(self, jsonl_path: Path):
        logger.info(f"Loading ColBERT custom references from {jsonl_path}...")
        custom_refs: Dict[str, List[str]] = {}
        # Use CLASS_DESCRIPTIONS keys as valid, or allow any if CLASS_DESCRIPTIONS is empty (e.g. fully custom setup)
        valid_cls = set(self.CLASS_DESCRIPTIONS.keys()) if self.CLASS_DESCRIPTIONS else None

        if not jsonl_path.exists(): logger.error(f"Custom ref JSONL not found: {jsonl_path}"); return
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        item = json.loads(line)
                        text, cls_name = item.get("text"), item.get("class_name")
                        if not text or not cls_name: continue
                        if valid_cls and cls_name not in valid_cls:
                             logger.warning(f"L{i+1} in {jsonl_path}: Class name '{cls_name}' not in known ColBERT CLASS_DESCRIPTIONS. Adding it dynamically.")
                             # Optionally add to CLASS_DESCRIPTIONS if you want to support truly dynamic classes
                             # self.CLASS_DESCRIPTIONS[cls_name] = "Custom class"
                        custom_refs.setdefault(cls_name, []).append(text)
                    except json.JSONDecodeError: logger.warning(f"L{i+1} JSON error in {jsonl_path}")
            if not custom_refs: logger.warning(f"No valid custom refs in {jsonl_path}"); return
            self.reference_texts_for_model = custom_refs # Override with custom
            logger.info(f"Loaded {sum(len(v) for v in custom_refs.values())} custom ColBERT references for {len(custom_refs)} classes.")
        except Exception as e: logger.error(f"Error loading custom ref file {jsonl_path}: {e}", exc_info=True)


    def setup_model_and_references(self, cache_dir: Optional[Path] = None, force_recompute_embeddings: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ColBERT using device: {self.device} for model: {self.model_id_or_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path)
            if self.tokenizer.pad_token is None: self.tokenizer.add_special_tokens({'pad_token': self.DEFAULT_PAD_TOKEN})
        except Exception as e: logger.error(f"Fatal: ColBERT tokenizer load error: {e}", exc_info=True); raise

        has_flash_attn = False
        if self.device.type == 'cuda':
            try: import flash_attn; has_flash_attn = True; logger.info("Flash Attention 2 available for ColBERT.")
            except ImportError: logger.info("Flash Attention 2 not found for ColBERT.")
        model_kwargs = {"attn_implementation": "flash_attention_2"} if has_flash_attn and version.parse(torch.__version__) >= version.parse("2.0") else {}

        try: self.model = AutoModel.from_pretrained(self.model_id_or_path, **model_kwargs)
        except Exception as e:
            if model_kwargs: logger.warning("Failed ColBERT load w/ Flash Attn. Retrying default."); model_kwargs={}
            try: self.model = AutoModel.from_pretrained(self.model_id_or_path, **model_kwargs)
            except Exception as e2: logger.error(f"Fatal: ColBERT model load error: {e2}", exc_info=True); raise

        if self.model and self.tokenizer.pad_token_id is not None and self.tokenizer.pad_token_id >= self.model.config.vocab_size:
             self.model.resize_token_embeddings(len(self.tokenizer))
        if self.model:
            self.model.to(self.device).eval()
            logger.info(f"ColBERT Model '{self.model.config._name_or_path}' loaded.")
        else: # Should not happen if previous try-except for model load failed, as it would raise
            logger.error(f"ColBERT Model object is None after attempting to load {self.model_id_or_path}. Cannot proceed with setup.")
            return


        if not self.reference_texts_for_model:
            logger.warning("No ColBERT references available (neither built-in nor custom loaded). Classification might not be meaningful.");
            self.reference_token_embeddings_by_class = {} # Ensure it's empty
            return

        emb_path = cache_dir / self.PRECOMPUTED_REF_EMBEDDINGS_FILENAME if cache_dir else None

        # Check if snapshot of references matches current references before loading embeddings
        ref_snapshot_path = cache_dir / self.REFERENCE_TEXTS_SNAPSHOT_FILENAME if cache_dir else None
        refs_match = False
        if ref_snapshot_path and ref_snapshot_path.exists() and emb_path and emb_path.exists():
            try:
                with open(ref_snapshot_path, 'r', encoding='utf-8') as f:
                    snapshot_refs = json.load(f)
                if snapshot_refs == self.reference_texts_for_model:
                    refs_match = True
                else:
                    logger.info("Reference texts have changed since last embedding computation. Recomputing.")
            except Exception as e:
                logger.warning(f"Could not compare reference snapshot: {e}. Recomputing embeddings.")

        if not force_recompute_embeddings and emb_path and emb_path.exists() and refs_match:
            try:
                self.reference_token_embeddings_by_class = torch.load(emb_path, map_location='cpu')
                # Verify that the loaded embeddings cover all current reference classes
                if set(self.reference_token_embeddings_by_class.keys()) == set(self.reference_texts_for_model.keys()):
                    logger.info(f"Loaded pre-computed ColBERT ref embeddings from {emb_path}.")
                    return # Successfully loaded
                else:
                    logger.warning("Cached embeddings do not match current reference classes. Recomputing.")
            except Exception as e: logger.warning(f"Failed load pre-computed ColBERT embs: {e}. Recomputing.", exc_info=True)

        logger.info("Computing ColBERT reference embeddings...");
        self._compute_and_cache_reference_embeddings(embeddings_path=emb_path, ref_snapshot_path=ref_snapshot_path)

    def _compute_and_cache_reference_embeddings(self, embeddings_path: Optional[Path], ref_snapshot_path: Optional[Path]):
        if not self.model or not self.tokenizer: raise RuntimeError("ColBERT model/tokenizer not ready for embs.")
        self.reference_token_embeddings_by_class = {}
        all_texts_flat, class_text_counts = [], {}

        # Ensure all classes in reference_texts_for_model are processed, even if empty
        for cls_name, texts in self.reference_texts_for_model.items():
            if not texts:
                self.reference_token_embeddings_by_class[cls_name] = []
                class_text_counts[cls_name] = 0 # Still record the class
                continue
            all_texts_flat.extend(texts)
            class_text_counts[cls_name] = len(texts)

        if not all_texts_flat: # No texts to embed across all classes
            if embeddings_path: # Save empty embeddings if path provided
                 try:
                    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.reference_token_embeddings_by_class, embeddings_path)
                    logger.info(f"Saved (empty) ColBERT ref embs to {embeddings_path}")
                    if ref_snapshot_path:
                        with open(ref_snapshot_path, 'w', encoding='utf-8') as f: json.dump(self.reference_texts_for_model, f, indent=2)
                 except Exception as e: logger.error(f"Failed save empty ColBERT ref embs: {e}", exc_info=True)
            return

        all_embs_gpu = self._get_token_embeddings_batched(all_texts_flat)
        current_idx = 0
        for cls_name, count in class_text_counts.items():
            if count > 0: # Only assign embeddings if there were texts for this class
                self.reference_token_embeddings_by_class[cls_name] = [emb.cpu() for emb in all_embs_gpu[current_idx : current_idx + count]]
                current_idx += count
            # else: it's already [] from initialization loop

        if embeddings_path:
            try:
                embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.reference_token_embeddings_by_class, embeddings_path)
                logger.info(f"Saved ColBERT ref embs to {embeddings_path}")
                if ref_snapshot_path: # Save current reference texts snapshot
                    with open(ref_snapshot_path, 'w', encoding='utf-8') as f: json.dump(self.reference_texts_for_model, f, indent=2)
                    logger.info(f"Saved ColBERT reference snapshot to {ref_snapshot_path}")
            except Exception as e: logger.error(f"Failed save ColBERT ref embs/snapshot: {e}", exc_info=True)


    def _get_token_embeddings_batched(self, texts: List[str], batch_size: int = 32, enable_grad: bool = False) -> List[torch.Tensor]:
        if not self.tokenizer or not self.model: raise RuntimeError("ColBERT model/tokenizer not ready.")
        all_embs_gpu = []

        original_model_training_state = self.model.training
        if enable_grad: self.model.train()
        else: self.model.eval()

        context_manager = torch.enable_grad() if enable_grad else torch.no_grad()

        with context_manager:
            for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing & Embedding ColBERT Batches", leave=False):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=self.DEFAULT_MAX_LENGTH).to(self.device)
                outputs = self.model(**inputs).last_hidden_state
                for j in range(outputs.size(0)): # Iterate through items in the batch
                    # Masking: Select embeddings only for non-padded tokens
                    non_padded_mask = inputs['attention_mask'][j] == 1
                    token_embeddings = outputs[j][non_padded_mask]
                    all_embs_gpu.append(token_embeddings) # Appends a [num_tokens, hidden_dim] tensor for each text

        self.model.train(original_model_training_state) # Restore original model state
        return all_embs_gpu

    def _colbert_maxsim(self, query_embs: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        if query_embs.ndim != 2 or doc_embs.ndim != 2 or query_embs.size(0) == 0 or doc_embs.size(0) == 0:
            dtype_to_use = query_embs.dtype if query_embs.numel() > 0 else (doc_embs.dtype if doc_embs.numel() > 0 else torch.float32)
            device_to_use = self.device if self.device else 'cpu'
            return torch.tensor(0.0, device=device_to_use, dtype=dtype_to_use)
        q_dev, d_dev = query_embs.to(self.device), doc_embs.to(self.device)
        sim_matrix = torch.matmul(F.normalize(q_dev, p=2, dim=1), F.normalize(d_dev, p=2, dim=1).T)
        if sim_matrix.numel() == 0: # Max over empty tensor error
             return torch.tensor(0.0, device=self.device, dtype=q_dev.dtype)
        return torch.sum(torch.max(sim_matrix, dim=1)[0])

    def classify_text(self, text: str) -> Dict[str, Any]:
        if not self.model: raise RuntimeError("ColBERT model not loaded.")
        if not self.reference_token_embeddings_by_class and not self.reference_texts_for_model:
            logger.warning("ColBERT has no reference texts or embeddings loaded. Cannot classify.")
            return {"error": "ColBERT references not configured", "predicted_class": "N/A", "scores_by_class": {}}
        if not self.reference_token_embeddings_by_class and self.reference_texts_for_model:
             # This case implies embeddings should have been computed but weren't (e.g. setup issue)
            logger.error("ColBERT references exist but embeddings are missing. Please check setup.")
            return {"error": "ColBERT reference embeddings missing", "predicted_class": "N/A", "scores_by_class": {}}


        if not text.strip(): return {"input_text": text, "error": "Empty input text", "predicted_class": "N/A", "scores_by_class": {}}

        query_embs_list = self._get_token_embeddings_batched([text])
        if not query_embs_list: # Should not happen if text is not empty
            return {"input_text": text, "error": "Failed to generate embeddings for input text", "predicted_class": "N/A", "scores_by_class": {}}
        query_embs_gpu = query_embs_list[0] # It's a list containing one tensor

        scores = {}
        # Iterate over classes that have actual embeddings
        for cls_name, ref_embs_list_cpu in self.reference_token_embeddings_by_class.items():
            if not ref_embs_list_cpu: # This class has no reference examples or embeddings
                scores[cls_name] = 0.0
                continue

            class_total_score = 0.0
            num_valid_references_for_class = 0
            for ref_cpu_embs in ref_embs_list_cpu:
                if ref_cpu_embs.numel() == 0: continue # Skip empty reference embedding

                # Ensure ref_cpu_embs is 2D [num_tokens, hidden_dim] before passing to _colbert_maxsim
                if ref_cpu_embs.ndim == 1: # Should ideally not happen if _get_token_embeddings_batched is correct
                    logger.warning(f"Skipping 1D reference embedding for class {cls_name}. Shape: {ref_cpu_embs.shape}")
                    continue

                class_total_score += self._colbert_maxsim(query_embs_gpu, ref_cpu_embs.to(self.device)).item()
                num_valid_references_for_class +=1

            if num_valid_references_for_class > 0:
                scores[cls_name] = class_total_score / num_valid_references_for_class
            else: # No valid reference embeddings for this class, even if texts existed
                scores[cls_name] = 0.0

        pred_cls = max(scores, key=scores.get) if scores else "N/A"
        # Ensure CLASS_DESCRIPTIONS is consulted safely
        class_desc = self.CLASS_DESCRIPTIONS.get(pred_cls, "Description not available") if pred_cls != "N/A" else "N/A"

        return {"input_text": text, "predicted_class": pred_cls,
                "class_description": class_desc,
                "scores_by_class (avg_maxsim)": scores}


    def finetune(self, reference_jsonl_path: Path, output_model_dir: Path, base_model_id: str = DEFAULT_MODEL_ID,
                 epochs: int = 3, learning_rate: float = 1e-5, batch_size: int = 4, triplet_margin: float = 0.2):
        logger.info(f"Starting ColBERT fine-tuning. Base: {base_model_id}, Output: {output_model_dir}")
        output_model_dir.mkdir(parents=True, exist_ok=True)
        self._load_custom_references_from_jsonl(reference_jsonl_path) # Loads into self.reference_texts_for_model
        if not self.reference_texts_for_model: logger.error("No ColBERT refs for fine-tuning from JSONL. Aborting."); return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None: self.tokenizer.add_special_tokens({'pad_token': self.DEFAULT_PAD_TOKEN})
        self.model = AutoModel.from_pretrained(base_model_id).to(self.device)
        if self.model and self.tokenizer.pad_token_id is not None and self.tokenizer.pad_token_id >= self.model.config.vocab_size:
             self.model.resize_token_embeddings(len(self.tokenizer))
        if not self.model: # Should be caught by AutoModel.from_pretrained raising an error.
            logger.error("ColBERT model could not be loaded for fine-tuning. Aborting."); return

        triplets = self._prepare_triplets_for_finetuning()
        if not triplets: logger.error("No triplets generated for ColBERT fine-tuning. Aborting."); return
        logger.info(f"Prepared {len(triplets)} triplets for fine-tuning.")

        class TripletDataset(Dataset):
            def __init__(self, d): self.d = d
            def __len__(self): return len(self.d)
            def __getitem__(self, i): return self.d[i]

        def collate_fn(b): return ([x[0] for x in b], [x[1] for x in b], [x[2] for x in b])
        dataloader = DataLoader(TripletDataset(triplets), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            logger.info(f"ColBERT Fine-tune Epoch {epoch+1}/{epochs}"); total_loss_epoch = 0; batches_processed = 0
            prog_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Fine-tuning ColBERT", leave=False)
            for anchor_txts, pos_txts, neg_txts in prog_bar:
                if stop_signal_received: logger.info("ColBERT fine-tuning interrupted."); break
                optimizer.zero_grad()

                # Concatenate all texts for batch embedding
                all_triplet_texts = anchor_txts + pos_txts + neg_txts
                # Get embeddings: list of tensors, one per text. Enable grad for these.
                all_embs = self._get_token_embeddings_batched(all_triplet_texts, batch_size=len(all_triplet_texts), enable_grad=True)

                n_actual_triplets_in_batch = len(anchor_txts) # This is how many triplets were intended for this batch
                anc_e = all_embs[:n_actual_triplets_in_batch]
                pos_e = all_embs[n_actual_triplets_in_batch : 2*n_actual_triplets_in_batch]
                neg_e = all_embs[2*n_actual_triplets_in_batch:]

                accumulated_batch_loss = [] # Store individual triplet losses for the batch

                for i in range(n_actual_triplets_in_batch):
                    current_anchor_emb = anc_e[i]
                    current_pos_emb = pos_e[i]
                    current_neg_emb = neg_e[i]

                    if current_anchor_emb.size(0) == 0 or current_pos_emb.size(0) == 0 or current_neg_emb.size(0) == 0:
                        logger.debug(f"Skipping triplet {i} in batch due to empty embeddings for anchor, positive, or negative text.")
                        continue

                    score_pos = self._colbert_maxsim(current_anchor_emb, current_pos_emb)
                    score_neg = self._colbert_maxsim(current_anchor_emb, current_neg_emb)

                    triplet_loss = F.relu(triplet_margin - score_pos + score_neg)
                    accumulated_batch_loss.append(triplet_loss)

                if accumulated_batch_loss: # If any valid triplets were processed
                    mean_batch_loss = torch.stack(accumulated_batch_loss).mean()
                    mean_batch_loss.backward()
                    optimizer.step()

                    total_loss_epoch += mean_batch_loss.item()
                    batches_processed += 1
                    prog_bar.set_postfix({'loss': mean_batch_loss.item()})
                else:
                    prog_bar.set_postfix({'loss': 0.0, 'skipped_batch': True})

            if stop_signal_received: break
            avg_epoch_loss = total_loss_epoch / batches_processed if batches_processed else 0
            logger.info(f"ColBERT Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")

        self.model.eval() # Set model to eval mode after training
        logger.info("ColBERT fine-tuning finished.");
        self._save_fine_tuned_model_assets(output_model_dir, base_model_id)
        self.model_id_or_path = str(output_model_dir.resolve())
        self.is_fine_tuned = True # Mark as fine-tuned
        # Re-setup with the new fine-tuned model and its references, recomputing embeddings for it.
        # The reference_texts_for_model is already set from _load_custom_references_from_jsonl
        logger.info(f"Re-setting up ColBERT with fine-tuned model from {self.model_id_or_path} and recomputing reference embeddings.")
        self.setup_model_and_references(cache_dir=output_model_dir, force_recompute_embeddings=True)


    def _save_fine_tuned_model_assets(self, save_dir: Path, base_model_id_used: str):
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.model: self.model.save_pretrained(save_dir)
        if self.tokenizer: self.tokenizer.save_pretrained(save_dir)
        # Save the reference_texts_for_model that this model was fine-tuned WITH
        with open(save_dir / self.REFERENCE_TEXTS_SNAPSHOT_FILENAME, 'w', encoding='utf-8') as f: json.dump(self.reference_texts_for_model, f, indent=2)
        cfg = {"base_model_id_used": base_model_id_used,
               "finetuned_from_hf_id_or_path": self.model_id_or_path, # Path it was originally loaded from before finetuning
               "timestamp": time.time()}
        with open(save_dir / self.COLBERT_CONFIG_FILENAME, 'w', encoding='utf-8') as f: json.dump(cfg, f, indent=2)
        logger.info(f"Fine-tuned ColBERT assets (model, tokenizer, references, config) saved to {save_dir}")

    def _prepare_triplets_for_finetuning(self) -> List[Tuple[str, str, str]]:
        import random
        triplets = []; class_names = list(self.reference_texts_for_model.keys())
        if len(class_names) < 2:
            logger.warning("Need at least 2 classes with reference texts for ColBERT triplet generation. No triplets generated.")
            return []

        for cls_name, texts in self.reference_texts_for_model.items():
            if len(texts) < 2: # Need at least two examples in the same class to form anchor/positive
                logger.debug(f"Skipping class '{cls_name}' for triplet generation: needs at least 2 examples, found {len(texts)}.")
                continue

            other_classes_with_texts = {cn: txts for cn, txts in self.reference_texts_for_model.items() if cn != cls_name and txts}
            if not other_classes_with_texts: # No other classes with texts to pick negatives from
                logger.debug(f"Skipping class '{cls_name}' for triplet generation: no other classes with texts available for negatives.")
                continue

            for i, anchor_text in enumerate(texts):
                # Choose positive from the same class, different text
                positive_candidates = texts[:i] + texts[i+1:]
                if not positive_candidates: continue # Should not happen if len(texts) >= 2
                positive_text = random.choice(positive_candidates)

                # Choose negative from a different class
                negative_class_name = random.choice(list(other_classes_with_texts.keys()))
                negative_text = random.choice(other_classes_with_texts[negative_class_name])

                triplets.append((anchor_text, positive_text, negative_text))

        random.shuffle(triplets)
        return triplets


    @classmethod
    def load_from_directory(cls, model_directory: Path):
        logger.info(f"Loading fine-tuned ColBERT from: {model_directory}")
        if not model_directory.is_dir(): raise FileNotFoundError(f"ColBERT model dir not found: {model_directory}")

        cfg_p = model_directory / cls.COLBERT_CONFIG_FILENAME
        ref_p = model_directory / cls.REFERENCE_TEXTS_SNAPSHOT_FILENAME

        if not cfg_p.exists(): raise FileNotFoundError(f"Missing ColBERT config ({cls.COLBERT_CONFIG_FILENAME}) in {model_directory}")
        if not ref_p.exists(): raise FileNotFoundError(f"Missing ColBERT reference snapshot ({cls.REFERENCE_TEXTS_SNAPSHOT_FILENAME}) in {model_directory}")

        instance = cls(str(model_directory.resolve())) # Initialize with path to fine-tuned model
        instance.is_fine_tuned = True # Explicitly mark as fine-tuned

        try:
            with open(ref_p, 'r', encoding='utf-8') as f:
                instance.reference_texts_for_model = json.load(f)
            logger.info(f"Loaded reference texts snapshot for fine-tuned ColBERT from {ref_p}")
        except Exception as e:
            logger.error(f"Failed to load ColBERT reference snapshot from {ref_p}: {e}", exc_info=True)
            raise

        # Setup model and its (now loaded) specific reference embeddings
        # The cache_dir for a loaded fine-tuned model should be within its own directory.
        # force_recompute_embeddings=False allows loading precomputed ones if they exist and match.
        instance.setup_model_and_references(cache_dir=model_directory, force_recompute_embeddings=False)

        logger.info(f"Fine-tuned ColBERT model and its references loaded successfully from {model_directory}")
        return instance

    def get_hardware_info(self) -> Dict[str, Any]:
        info = {"device": str(self.device), "cuda": torch.cuda.is_available(), "torch": torch.__version__, "transformers": transformers.__version__}
        if torch.cuda.is_available() and self.device and self.device.type == 'cuda': # check device exists
             info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
        else: info["gpu_name"] = "N/A"
        try: import flash_attn; info["flash_attn"] = True
        except ImportError: info["flash_attn"] = False
        return info

class ClassificationAPI:
    def __init__(self, modernbert_model_dir: Optional[str],
                 host: str, port: int,
                 policy_config_path: Optional[str] = "policy_config.json"): # Added policy_config_path
        self.modernbert_model_dir = modernbert_model_dir
        self.host = host
        self.port = port
        self.policy_config_path = policy_config_path
        self.api_policy_config: Dict[str, Any] = {}
        self.modernbert_classifier: Optional[ModernBERTClassifier] = None
        self.colbert_reranker: Optional[ColBERTReranker] = None
        self.app = Flask(__name__)
        CORS(self.app)
        self.request_count = 0

    def setup(self, serve_modernbert: bool, serve_colbert: bool,
              colbert_model_id_or_dir: Optional[str],
              colbert_custom_ref_jsonl: Optional[str],
              colbert_cache_dir: Optional[str]):

        # Load API policy configuration
        if self.policy_config_path:
            policy_file = Path(self.policy_config_path)
            if policy_file.exists():
                try:
                    with open(policy_file, 'r', encoding='utf-8') as f:
                        self.api_policy_config = json.load(f)
                    logger.info(f"Loaded API policy configuration from {self.policy_config_path}")
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

        if serve_modernbert and self.modernbert_model_dir:
            try:
                self.modernbert_classifier = ModernBERTClassifier.load(self.modernbert_model_dir)
                logger.info("ModernBERT API model loaded.")
            except FileNotFoundError:
                 logger.error(f"ModernBERT model directory not found: {self.modernbert_model_dir}. ModernBERT API will not be available.")
                 serve_modernbert=False # Cannot serve if not found
            except Exception as e:
                logger.error(f"Failed to load ModernBERT for API from {self.modernbert_model_dir}: {e}", exc_info=True)
                serve_modernbert=False

        if serve_colbert:
            try:
                colbert_path_obj = Path(colbert_model_id_or_dir) if colbert_model_id_or_dir else None

                if colbert_path_obj and colbert_path_obj.is_dir():
                    logger.info(f"Attempting to load fine-tuned ColBERT from directory: {colbert_path_obj}")
                    self.colbert_reranker = ColBERTReranker.load_from_directory(colbert_path_obj)
                else:
                    base_id = colbert_model_id_or_dir or ColBERTReranker.DEFAULT_MODEL_ID
                    logger.info(f"Initializing ColBERT with base model ID: {base_id}")
                    self.colbert_reranker = ColBERTReranker(base_id)

                    # Load custom references if provided, these will override built-in ones.
                    if colbert_custom_ref_jsonl:
                        custom_ref_path = Path(colbert_custom_ref_jsonl)
                        if custom_ref_path.exists():
                            self.colbert_reranker._load_custom_references_from_jsonl(custom_ref_path)
                        else:
                            logger.warning(f"ColBERT custom reference JSONL not found: {custom_ref_path}. Using defaults or built-in if available.")

                    # Determine cache directory for base model + its references
                    # Default cache for API server if not specified
                    cache_p = Path(colbert_cache_dir) if colbert_cache_dir else Path.home()/".cache"/"classifier_tool"/"api_colbert_cache"
                    # Create a sub-directory in cache based on model ID to avoid conflicts if base ID changes
                    model_name_for_cache = "".join(c if c.isalnum() or c in ['-','_','.'] else '_' for c in Path(base_id).name)
                    final_cache_dir_for_model = cache_p / model_name_for_cache
                    final_cache_dir_for_model.mkdir(parents=True, exist_ok=True)

                    self.colbert_reranker.setup_model_and_references(cache_dir=final_cache_dir_for_model)

                logger.info(f"ColBERT API model setup complete. Fine-tuned: {self.colbert_reranker.is_fine_tuned if self.colbert_reranker else 'N/A'}.")
            except FileNotFoundError as e:
                 logger.error(f"ColBERT model file/directory not found: {e}. ColBERT API will not be available.")
                 serve_colbert=False
            except Exception as e:
                logger.error(f"Failed to setup ColBERT for API: {e}", exc_info=True)
                serve_colbert=False

        @self.app.route('/health', methods=['GET'])
        def health():
            mb_ok = self.modernbert_classifier is not None and self.modernbert_classifier.model is not None
            cb_ok = self.colbert_reranker is not None and self.colbert_reranker.model is not None

            status_details = {
                "modernbert_loaded": mb_ok,
                "colbert_loaded": cb_ok,
                "colbert_is_fine_tuned": self.colbert_reranker.is_fine_tuned if cb_ok else None,
                "colbert_reference_classes": list(self.colbert_reranker.reference_texts_for_model.keys()) if cb_ok and self.colbert_reranker.reference_texts_for_model else []
            }

            overall_status = "ok"
            policy_readiness = {"status": "not_applicable_no_policies" if not self.api_policy_config else "ok", "issues": []}

            if self.api_policy_config:
                all_policies_runnable = True
                for policy_name, policy_rules in self.api_policy_config.items():
                    if policy_rules.get("modernbert_io_validation") and not mb_ok:
                        all_policies_runnable = False
                        policy_readiness["issues"].append(f"Policy '{policy_name}' requires ModernBERT, which is not loaded.")
                    if policy_rules.get("colbert_input_sensitivity") or policy_rules.get("colbert_output_sensitivity"):
                        if not cb_ok:
                            all_policies_runnable = False
                            policy_readiness["issues"].append(f"Policy '{policy_name}' requires ColBERT, which is not loaded.")
                        elif policy_rules.get("require_colbert_fine_tuned") and (not self.colbert_reranker or not self.colbert_reranker.is_fine_tuned):
                            all_policies_runnable = False
                            policy_readiness["issues"].append(f"Policy '{policy_name}' requires fine-tuned ColBERT, but current ColBERT is not fine-tuned or not loaded.")

                if not all_policies_runnable and (mb_ok or cb_ok): # Some models loaded, but not all policy reqs met
                    overall_status = "degraded"
                    policy_readiness["status"] = "degraded"
                elif not all_policies_runnable and not mb_ok and not cb_ok: # No models loaded, policies exist
                    overall_status = "error"
                    policy_readiness["status"] = "error_models_unavailable"
                elif not (mb_ok or cb_ok) and self.api_policy_config : # Policies exist, but no models loaded at all
                    overall_status = "error"
                    policy_readiness["status"] = "error_models_unavailable"


            return jsonify({
                "status": overall_status,
                "model_availability": status_details,
                "policy_config_loaded": bool(self.api_policy_config),
                "policy_model_readiness": policy_readiness
            })

        if serve_modernbert and self.modernbert_classifier : # Check classifier again after potential load failure
            @self.app.route('/modernbert/classify', methods=['POST'])
            def mb_classify():
                # This check is redundant due to the outer if, but good for safety
                if not self.modernbert_classifier or not self.modernbert_classifier.model:
                    return jsonify({"error":"ModernBERT model not loaded or available"}),503 # Service Unavailable

                data = request.get_json()
                if not data: return jsonify({"error": "Request body must be JSON"}), 400

                input_text = data.get('input_text')
                # Changed 'output_to_classify' to 'output_text' for consistency
                output_text = data.get('output_text')

                if input_text is None: # output_text can be None or empty string for single text classification cases if model handles it
                    return jsonify({"error":"'input_text' is required"}),400

                try:
                    # classify_input_output_pair expects output_text to be string or None
                    return jsonify(self.modernbert_classifier.classify_input_output_pair(input_text, output_text if output_text is not None else ""))
                except Exception as e:
                    logger.error(f"Error in /modernbert/classify: {e}", exc_info=True)
                    return jsonify({"error":str(e)}),500

        if serve_colbert and self.colbert_reranker and self.colbert_reranker.model: # Check again
            @self.app.route('/colbert/classify_sensitivity', methods=['POST'])
            def cb_classify():
                if not self.colbert_reranker or not self.colbert_reranker.model :
                    return jsonify({"error":"ColBERT model not loaded or available"}),503

                data=request.get_json()
                if not data: return jsonify({"error": "Request body must be JSON"}), 400
                txt=data.get('text')
                if txt is None: return jsonify({"error":"'text' field required"}),400
                if not isinstance(txt, str): return jsonify({"error": "'text' field must be a string"}), 400

                try: return jsonify(self.colbert_reranker.classify_text(txt))
                except Exception as e:
                    logger.error(f"Error in /colbert/classify_sensitivity: {e}", exc_info=True)
                    return jsonify({"error":str(e)}),500

        # New /service/validate endpoint
        @self.app.route('/service/validate', methods=['POST'])
        def service_validate():
            self.request_count += 1 # Example of simple request tracking
            payload = request.get_json()
            if not payload:
                return jsonify({"error": "Request body must be JSON"}), 400

            api_class = payload.get("api_class")
            input_text = payload.get("input_text")
            output_text = payload.get("output_text") # Can be None

            if not api_class or input_text is None: # input_text must be present
                return jsonify({"error": "'api_class' and 'input_text' are required fields"}), 400

            response_data = {"request": payload}
            current_overall_status = "PASS" # Optimistic start
            violations = [] # Store descriptions of policy violations
            error_message_detail = None # Store specific error message for ERROR status

            policy = self.api_policy_config.get(api_class)
            if not policy:
                response_data["overall_status"] = "REJECT_INVALID_POLICY"
                response_data["error_message"] = f"API class '{api_class}' not found in policy configuration."
                return jsonify(response_data), 400 # Bad Request as API class is invalid input

            response_data["policy_applied"] = policy

            # --- ModernBERT I/O Validation ---
            if policy.get("modernbert_io_validation"):
                if output_text is None: # Required for this check
                    current_overall_status = "ERROR"
                    error_message_detail = "output_text is required for modernbert_io_validation."
                    response_data["modernbert_io_validation"] = {"status": "error_missing_output_text", "message": error_message_detail}
                elif not self.modernbert_classifier or not self.modernbert_classifier.model:
                    current_overall_status = "ERROR"
                    error_message_detail = "ModernBERT model required by policy is not loaded."
                    response_data["modernbert_io_validation"] = {"status": "error_model_not_loaded", "message": error_message_detail}
                else:
                    try:
                        mb_result = self.modernbert_classifier.classify_input_output_pair(input_text, output_text)
                        response_data["modernbert_io_validation"] = mb_result
                        if mb_result.get("prediction") == 0: # Assuming 0 is REJECT
                            violations.append("ModernBERT_IO_Validation: Predicted as inappropriate pair.")
                    except Exception as e:
                        logger.error(f"Error during ModernBERT validation for policy '{api_class}': {e}", exc_info=True)
                        current_overall_status = "ERROR"
                        error_message_detail = str(e)
                        response_data["modernbert_io_validation"] = {"status": "error_exception_in_model", "message": error_message_detail}

            if current_overall_status == "ERROR":
                response_data["overall_status"] = "ERROR"
                response_data["error_message"] = error_message_detail
                return jsonify(response_data), 500

            # --- ColBERT Input Sensitivity ---
            if policy.get("colbert_input_sensitivity"):
                if not self.colbert_reranker or not self.colbert_reranker.model:
                    current_overall_status = "ERROR"
                    error_message_detail = "ColBERT model required by policy for input check is not loaded."
                    response_data["colbert_input_sensitivity"] = {"status": "error_model_not_loaded", "message": error_message_detail}
                elif policy.get("require_colbert_fine_tuned") and not self.colbert_reranker.is_fine_tuned:
                    current_overall_status = "ERROR"
                    error_message_detail = "Fine-tuned ColBERT model required by policy for input check, but loaded ColBERT is not fine-tuned."
                    response_data["colbert_input_sensitivity"] = {"status": "error_model_not_fine_tuned", "message": error_message_detail}
                else:
                    try:
                        cb_input_result = self.colbert_reranker.classify_text(input_text)
                        response_data["colbert_input_sensitivity"] = cb_input_result
                        predicted_class = cb_input_result.get("predicted_class")
                        allowed_classes = policy.get("allowed_colbert_input_classes")
                        disallowed_classes = policy.get("disallowed_colbert_input_classes")

                        if allowed_classes and predicted_class not in allowed_classes:
                            violations.append(f"ColBERT_Input_Sensitivity: Predicted class '{predicted_class}' not in allowed list: {allowed_classes}.")
                        if disallowed_classes and predicted_class in disallowed_classes:
                             violations.append(f"ColBERT_Input_Sensitivity: Predicted class '{predicted_class}' is in disallowed list: {disallowed_classes}.")
                    except Exception as e:
                        logger.error(f"Error during ColBERT input sensitivity for policy '{api_class}': {e}", exc_info=True)
                        current_overall_status = "ERROR"
                        error_message_detail = str(e)
                        response_data["colbert_input_sensitivity"] = {"status": "error_exception_in_model", "message": error_message_detail}

            if current_overall_status == "ERROR":
                response_data["overall_status"] = "ERROR"
                response_data["error_message"] = error_message_detail
                return jsonify(response_data), 500

            # --- ColBERT Output Sensitivity ---
            if policy.get("colbert_output_sensitivity"):
                if output_text is None or not output_text.strip() :
                    # This is a policy violation if the check is active and output is missing/empty
                    violations.append("ColBERT_Output_Sensitivity: output_text is missing or empty, but required for this policy check.")
                    response_data["colbert_output_sensitivity"] = {"status": "skipped_missing_output_text", "message": "Non-empty output_text is required for colbert_output_sensitivity check."}
                elif not self.colbert_reranker or not self.colbert_reranker.model:
                    current_overall_status = "ERROR"
                    error_message_detail = "ColBERT model required by policy for output check is not loaded."
                    response_data["colbert_output_sensitivity"] = {"status": "error_model_not_loaded", "message": error_message_detail}
                elif policy.get("require_colbert_fine_tuned") and not self.colbert_reranker.is_fine_tuned:
                    current_overall_status = "ERROR"
                    error_message_detail = "Fine-tuned ColBERT model required by policy for output check, but loaded ColBERT is not fine-tuned."
                    response_data["colbert_output_sensitivity"] = {"status": "error_model_not_fine_tuned", "message": error_message_detail}
                else:
                    try:
                        cb_output_result = self.colbert_reranker.classify_text(output_text)
                        response_data["colbert_output_sensitivity"] = cb_output_result
                        predicted_class = cb_output_result.get("predicted_class")
                        allowed_classes = policy.get("allowed_colbert_output_classes")
                        disallowed_classes = policy.get("disallowed_colbert_output_classes")

                        if allowed_classes and predicted_class not in allowed_classes:
                            violations.append(f"ColBERT_Output_Sensitivity: Predicted class '{predicted_class}' not in allowed list: {allowed_classes}.")
                        if disallowed_classes and predicted_class in disallowed_classes:
                             violations.append(f"ColBERT_Output_Sensitivity: Predicted class '{predicted_class}' is in disallowed list: {disallowed_classes}.")
                    except Exception as e:
                        logger.error(f"Error during ColBERT output sensitivity for policy '{api_class}': {e}", exc_info=True)
                        current_overall_status = "ERROR"
                        error_message_detail = str(e)
                        response_data["colbert_output_sensitivity"] = {"status": "error_exception_in_model", "message": error_message_detail}

            # Final overall_status determination
            if current_overall_status == "ERROR":
                response_data["overall_status"] = "ERROR"
                response_data["error_message"] = error_message_detail
                if violations: response_data["violations_detected_before_error"] = violations # Log violations even if an error occurred later
                return jsonify(response_data), 500
            elif violations:
                response_data["overall_status"] = "REJECT_POLICY_VIOLATION"
                response_data["violation_reasons"] = violations
                return jsonify(response_data), 200
            else:
                response_data["overall_status"] = "PASS"
                return jsonify(response_data), 200

    def run(self, production: bool = True):
        # Check if any model is loaded to provide any service
        can_serve_anything = False
        if self.modernbert_classifier and self.modernbert_classifier.model:
            can_serve_anything = True
            logger.info("ModernBERT direct endpoint will be available.")
        if self.colbert_reranker and self.colbert_reranker.model:
            can_serve_anything = True
            logger.info("ColBERT direct endpoint will be available.")

        if self.api_policy_config: # If policies are loaded, /service/validate is implicitly available
            can_serve_anything = True
            logger.info("/service/validate endpoint will be available (functionality depends on loaded models and policies).")

        if not can_serve_anything:
            logger.error("No models loaded and no policies configured for the API. Nothing to serve. Exiting.")
            sys.exit(1)

        logger.info(f"Starting API server on http://{self.host}:{self.port}")
        if production:
            logger.info("Running in production mode with Waitress.")
            serve(self.app, host=self.host, port=self.port, threads=8)
        else:
            logger.info("Running in development mode with Flask's built-in server (use --dev-server for this).")
            self.app.run(host=self.host, port=self.port, debug=True, use_reloader=False)


# --- CLI Command Functions ---
def train_command(args: argparse.Namespace):
    logger.info("Running 'train' command (ModernBERT)...")
    dp = DataProcessor(args.data_path)
    if not dp.load_and_validate(): logger.error("ModernBERT data load failed."); sys.exit(1)
    texts, labels = dp.prepare_classification_data(separator=args.separator)
    if not texts: logger.error("No ModernBERT training samples."); sys.exit(1)
    split = dp.perform_train_test_split(texts, labels, test_size=args.test_size, random_state=args.random_state)
    if not split['train_texts']: logger.error("No ModernBERT train data after split."); sys.exit(1)

    classifier = ModernBERTClassifier(model_dir=args.model_dir, use_mlflow=args.use_mlflow)
    classifier.separator = args.separator
    # For training, ModernBERT is initialized with its base model ID, not loaded from model_dir yet.
    # The .setup() method initializes the base model.
    classifier.setup()
    classifier.train(split['train_texts'], split['train_labels'], split['test_texts'], split['test_labels'],
                     args.batch_size, args.learning_rate, args.epochs, args.gradient_accumulation_steps,
                     args.early_stopping_patience, args.warmup_ratio, args.weight_decay)
    logger.info("ModernBERT 'train' command finished.")
    if args.run_server_after_train:
        # Policy config path not relevant here as it's just for serving what was trained
        api = ClassificationAPI(args.model_dir, args.host, args.port, policy_config_path=None)
        api.setup(serve_modernbert=True, serve_colbert=False, colbert_model_id_or_dir=None, colbert_custom_ref_jsonl=None, colbert_cache_dir=None)
        api.run(production=not args.dev_server)

def serve_command(args: argparse.Namespace):
    logger.info("Running 'serve' command...")
    api = ClassificationAPI(
        modernbert_model_dir=args.modernbert_model_dir,
        host=args.host,
        port=args.port,
        policy_config_path=args.policy_config_path # Pass policy config path
    )
    api.setup(serve_modernbert=args.serve_modernbert, # Explicit flag to serve modernbert
              serve_colbert=args.serve_colbert_sensitivity,
              colbert_model_id_or_dir=args.colbert_model_id_or_dir,
              colbert_custom_ref_jsonl=args.colbert_custom_ref_jsonl,
              colbert_cache_dir=args.colbert_cache_dir)
    api.run(production=not args.dev_server)


def predict_modernbert_command_cli(args: argparse.Namespace):
    logger.info("Running 'predict-modernbert'...")
    try:
        classifier = ModernBERTClassifier.load(args.model_dir)
    except FileNotFoundError:
        logger.error(f"ModernBERT model directory not found: {args.model_dir}. Cannot run prediction.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load ModernBERT model from {args.model_dir}: {e}", exc_info=True)
        sys.exit(1)

    if not args.input_text or args.output_to_classify is None: # output_to_classify can be an empty string
        logger.error("Both --input-text and --output-to-classify (can be empty string) are required for predict-modernbert.")
        sys.exit(1)
    result = classifier.classify_input_output_pair(args.input_text, args.output_to_classify)
    print(json.dumps(result, indent=2))


def rerank_data_classify_command_cli(args: argparse.Namespace):
    logger.info(f"Running 'rerank-data-classify' (ColBERT)...")
    if not args.text_to_classify: logger.error("--text-to-classify required."); sys.exit(1)
    try:
        reranker: ColBERTReranker
        cb_model_path = Path(args.colbert_model_dir) if args.colbert_model_dir else None

        cache_p = Path(args.cache_dir) if args.cache_dir else Path.home()/".cache"/"classifier_tool"/"colbert_cli_cache"
        cache_p.mkdir(parents=True, exist_ok=True)

        if cb_model_path and cb_model_path.is_dir():
            logger.info(f"Loading fine-tuned ColBERT from directory: {cb_model_path}")
            reranker = ColBERTReranker.load_from_directory(cb_model_path) # This handles its own cache/snapshot within model_dir
        else:
            base_id = args.colbert_model_id_or_dir or ColBERTReranker.DEFAULT_MODEL_ID
            logger.info(f"Initializing ColBERT with base model ID: {base_id}")
            reranker = ColBERTReranker(base_id)
            if args.custom_reference_jsonl:
                custom_ref_path = Path(args.custom_reference_jsonl)
                if custom_ref_path.exists():
                    reranker._load_custom_references_from_jsonl(custom_ref_path)
                else:
                    logger.warning(f"Custom reference JSONL not found: {custom_ref_path}. Using defaults or built-ins.")

            model_name_for_cache = "".join(c if c.isalnum() or c in ['-','_','.'] else '_' for c in Path(base_id).name)
            final_cache_dir_for_model = cache_p / model_name_for_cache
            final_cache_dir_for_model.mkdir(parents=True, exist_ok=True)
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
    ref_jsonl = Path(args.reference_jsonl)
    output_dir = Path(args.output_model_dir)
    if not ref_jsonl.exists(): logger.error(f"Reference JSONL not found: {ref_jsonl}"); sys.exit(1)
    if not output_dir: logger.error("--output-model-dir is required."); sys.exit(1) # Should be caught by argparse

    reranker = ColBERTReranker(args.base_model_id)
    reranker.finetune(ref_jsonl, output_dir, args.base_model_id,
                      args.epochs, args.learning_rate, args.batch_size, args.triplet_margin)

def check_hardware_command(args: argparse.Namespace):
    logger.info("--- ModernBERT Hardware Info ---")
    # Initialize a temporary instance to get info, doesn't need full setup
    mb_checker = ModernBERTClassifier()
    mb_checker.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Basic device set
    info_mb = mb_checker.get_hardware_info()
    for k,v in info_mb.items(): logger.info(f"ModernBERT - {k.replace('_',' ').capitalize()}: {v}")

    logger.info("--- ColBERT Hardware Info ---")
    cb_checker = ColBERTReranker()
    cb_checker.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Basic device set
    info_cb = cb_checker.get_hardware_info()
    for k,v in info_cb.items(): logger.info(f"ColBERT    - {k.replace('_',' ').capitalize()}: {v}")


def create_example_command(args: argparse.Namespace):
    logger.info(f"Creating example files in {args.output_dir}...")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ModernBERT example
    mb_ex_path = out_dir / "sample_modernbert_training.jsonl"
    with open(mb_ex_path, 'w', encoding='utf-8') as f:
        json.dump({"input": "What is the capital of France?", "output_good_sample": "Paris is the capital of France.", "output_bad_sample": "Berlin is the capital of France."}, f); f.write("\n")
        json.dump({"input": "Is this product safe for children?", "output_good_sample": "This product is designed for ages 12 and up.", "output_bad_sample": "Yes, it's perfectly fine for toddlers."}, f); f.write("\n")
        # Example for single-text classification training
        json.dump({"input": "This movie was fantastic!", "label": 1}, f); f.write("\n")
        json.dump({"input": "I really disliked the service.", "label": 0}, f); f.write("\n")
    logger.info(f"Created ModernBERT example: {mb_ex_path}")

    # ColBERT example
    cb_ex_path = out_dir / "sample_colbert_references.jsonl"
    with open(cb_ex_path, 'w', encoding='utf-8') as f:
        for cn, exs in ColBERTReranker.BUILTIN_REFERENCE_TEXTS_BY_CLASS.items():
            if exs :
                # Create a couple of examples per built-in class for better representation
                for ex_text in exs[:2]: # Take up to 2 examples
                    json.dump({"text": ex_text, "class_name": cn}, f); f.write("\n")
        # Add a custom class example
        json.dump({"text": "Internal project codename 'Bluebird' details.", "class_name": "Custom_Internal_Project"},f); f.write("\n")
        json.dump({"text": "Client contact: support@example.org", "class_name": "Custom_Client_Contact"},f); f.write("\n")
    logger.info(f"Created ColBERT example: {cb_ex_path}")

    # Policy config example (for create-example command)
    policy_ex_path = out_dir / "sample_policy_config.json" # Name matches what create-example says it creates
    example_policy_config_for_create_example = {
      "DemoClass_MB_Only": {
        "description": "Demo: Only ModernBERT IO validation.",
        "modernbert_io_validation": True
      },
       "DemoClass_InputSensitive_NoPII": {
        "description": "Demo: ModernBERT + ColBERT Input (base model ok). Input must not be PII.",
        "modernbert_io_validation": True,
        "colbert_input_sensitivity": True,
        "require_colbert_fine_tuned": False,
        "disallowed_colbert_input_classes": ["Class 1: PII"]
      },
      "DemoClass_OutputPublic_FineTuned": {
        "description": "Demo: Output must be Public, requires fine-tuned ColBERT.",
        "colbert_output_sensitivity": True,
        "require_colbert_fine_tuned": True,
        "allowed_colbert_output_classes": ["Class 5: Public Data"]
      }
    }
    with open(policy_ex_path, 'w', encoding='utf-8') as f:
        json.dump(example_policy_config_for_create_example, f, indent=2)
    logger.info(f"Created example policy config for 'create-example': {policy_ex_path}")


    readme_p = out_dir / "README_examples.md"
    readme_content = f"""# Example Data for Classifier Service Tool

This directory contains example files to help you get started with the tool.

*   `{mb_ex_path.name}`: Sample data for training a ModernBERT classifier using the `train` command. It includes examples for both input-output pair validation and single-text classification.
*   `{cb_ex_path.name}`: Sample reference texts for ColBERT. Use this with the `finetune-colbert` command or as `--custom-reference-jsonl` for base ColBERT models (via `rerank-data-classify` CLI or `serve` API). It includes examples for built-in classes and demonstrates adding custom classes.
*   `{policy_ex_path.name}`: An example policy configuration file generated by `create-example`. Use this with the `serve` command's `--policy-config-path` argument to enable the `/service/validate` endpoint with predefined policies. For a more comprehensive example policy file to study, refer to `example_policy_config.json` in the main documentation.

## How to Use

1.  **ModernBERT Training:**
    ```bash
    python <your_script_name.py> train --data-path {mb_ex_path} --model-dir ./models/my_example_modernbert --epochs 1
    ```

2.  **ColBERT Fine-tuning:**
    *(Ensure your `{cb_ex_path.name}` has enough diverse examples per class, and at least two classes for triplet generation).*
    ```bash
    python <your_script_name.py> finetune-colbert --reference-jsonl {cb_ex_path} --output-model-dir ./models/my_example_colbert --epochs 1
    ```

3.  **Serving the API with Policies:**
    First, ensure you have trained models (e.g., from steps above or your own).
    ```bash
    python <your_script_name.py> serve \\
        --serve-modernbert --modernbert-model-dir ./models/my_example_modernbert \\
        --serve-colbert-sensitivity --colbert-model-id-or-dir ./models/my_example_colbert \\
        --policy-config-path {policy_ex_path} \\
        --port 5000
    ```
    Then you can test the `/service/validate` endpoint:
    ```bash
    # Example using curl for "DemoClass_InputSensitive_NoPII" policy
    curl -X POST -H "Content-Type: application/json" \\
    -d '{{
      "api_class": "DemoClass_InputSensitive_NoPII",
      "input_text": "My SSN is 123-45-6789, can you help?",
      "output_text": "I cannot process SSNs directly. Please visit our secure portal."
    }}' http://localhost:5000/service/validate
    ```

See `python <your_script_name.py> --help` and `python <your_script_name.py> <command> --help` for all available commands and options.
"""
    with open(readme_p, 'w', encoding='utf-8') as f: f.write(readme_content.replace("<your_script_name.py>", Path(__file__).name))
    logger.info(f"Created README for examples: {readme_p}")
    logger.info("Example files created successfully.")


def main_cli_entry():
    setup_signal_handling()
    parser = argparse.ArgumentParser(description="Classifier Service Tool: ModernBERT & ColBERT with Policy Enforcement.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # ModernBERT Train
    train_p = subparsers.add_parser("train", help="Train ModernBERT binary classifier.")
    train_p.add_argument("--data-path", nargs='+', required=True, help="Path(s) to JSONL training data.")
    train_p.add_argument("--model-dir", default="models/modernbert_custom", help="Directory to save/load fine-tuned ModernBERT model.")
    train_p.add_argument("--epochs", type=int, default=3); train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--learning-rate", type=float, default=2e-5); train_p.add_argument("--test-size", type=float, default=0.15)
    train_p.add_argument("--random-state", type=int, default=42); train_p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_p.add_argument("--early-stopping-patience", type=int, default=0, help="Set to >0 to enable early stopping based on eval F1/accuracy.")
    train_p.add_argument("--warmup-ratio", type=float, default=0.1); train_p.add_argument("--weight-decay", type=float, default=0.01)
    train_p.add_argument("--separator", default=" [SEP] ", help="Separator token for input/output pairs in ModernBERT.")
    train_p.add_argument("--use-mlflow", action="store_true"); train_p.add_argument("--run-server-after-train", action="store_true")
    train_p.add_argument("--host", default="0.0.0.0"); train_p.add_argument("--port", type=int, default=5000)
    train_p.add_argument("--dev-server", action="store_true", help="Run Flask dev server instead of Waitress (for run-server-after-train or serve).")
    train_p.set_defaults(func=train_command)

    # Serve API
    serve_p = subparsers.add_parser("serve", help="Start API server with optional ModernBERT, ColBERT, and Policy Validation.")
    serve_p.add_argument("--serve-modernbert", action="store_true", help="Enable ModernBERT model and its direct /modernbert/classify endpoint.")
    serve_p.add_argument("--modernbert-model-dir", help="Directory of fine-tuned ModernBERT model (required if --serve-modernbert or policies need it).")
    serve_p.add_argument("--serve-colbert-sensitivity", action="store_true", help="Enable ColBERT model and its direct /colbert/classify_sensitivity endpoint.")
    serve_p.add_argument("--colbert-model-id-or-dir", help="Hugging Face ID or local directory of ColBERT model (base or fine-tuned).")
    serve_p.add_argument("--colbert-custom-ref-jsonl", help="Path to JSONL file with custom reference texts for base ColBERT model.")
    serve_p.add_argument("--colbert-cache-dir", help="Cache directory for ColBERT base model embeddings and downloaded files (defaults to ~/.cache/classifier_tool/api_colbert_cache).")
    serve_p.add_argument("--policy-config-path", default="policy_config.json", help="Path to API policy configuration JSON file for /service/validate endpoint.")
    serve_p.add_argument("--host", default="0.0.0.0"); serve_p.add_argument("--port", type=int, default=5000)
    serve_p.add_argument("--dev-server", action="store_true", help="Run Flask dev server instead of Waitress.")
    serve_p.set_defaults(func=serve_command)

    # Predict ModernBERT CLI
    pred_mb_p = subparsers.add_parser("predict-modernbert", help="CLI for ModernBERT classification of an input/output pair.")
    pred_mb_p.add_argument("--model-dir", required=True, help="Directory of the fine-tuned ModernBERT model.")
    pred_mb_p.add_argument("--input-text", required=True, help="Input text part of the pair.")
    pred_mb_p.add_argument("--output-to-classify", required=True, help="Output text part to classify against the input (can be empty string).")
    pred_mb_p.set_defaults(func=predict_modernbert_command_cli)

    # Rerank Data Classify (ColBERT CLI)
    rerank_p = subparsers.add_parser("rerank-data-classify", help="Classify data sensitivity of a text using ColBERT.")
    rerank_p.add_argument("--text-to-classify", required=True, help="Text to classify for sensitivity.")
    rerank_p.add_argument("--colbert-model-dir", help="Path to a directory containing a fine-tuned ColBERT model.")
    rerank_p.add_argument("--colbert-model-id-or-dir", help="HF ID or path to a base ColBERT model (used if --colbert-model-dir is not provided). Default: 'lightonai/GTE-ModernColBERT-v1'.")
    rerank_p.add_argument("--custom-reference-jsonl", help="Path to JSONL with custom reference texts for a base ColBERT model.")
    rerank_p.add_argument("--cache-dir", help="Base cache directory for ColBERT non-fine-tuned setups (defaults to ~/.cache/classifier_tool/colbert_cli_cache).")
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
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            logger.error(f"An error occurred while executing command '{args.command}': {e}", exc_info=True)
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli_entry()
