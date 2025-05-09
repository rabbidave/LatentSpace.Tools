#!/usr/bin/env python
"""
ModernBERT Classification as a Service & ColBERT Data Reranking/Classification

A self-installing CLI/Python tool that:
1. Fine-tunes ModernBERT for binary classification of input-output text pairs.
2. Deploys a RESTful API server for the fine-tuned ModernBERT.
3. Provides data sensitivity classification using a ColBERT model (default or fine-tuned)
   by comparing input against reference examples via MaxSim.
4. Optionally allows fine-tuning the ColBERT model on custom reference examples for sensitivity classification.
"""

import os
os.environ['HF_TOKEN'] = 'hf_insert_token' # Placeholder for Hugging Face Token
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
        if self.tokenizer.pad_token_id >= self.model.config.vocab_size: self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device); return self

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
        return self.predict([f"{input_text}{self.separator}{output_text}"])[0]

    def _save_model(self, suffix: str = ""):
        if not self.model or not self.tokenizer: return
        save_path = self.model_dir / suffix if suffix else self.model_dir
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving ModernBERT model to {save_path}...")
        self.model.save_pretrained(save_path); self.tokenizer.save_pretrained(save_path)
        with open(save_path / "model_config.json", 'w') as f: json.dump({"separator": self.separator}, f)

    @classmethod
    def load(cls, model_dir: str, use_mlflow_during_load: bool = False):
        model_dir_path = Path(model_dir);
        if not model_dir_path.exists(): raise FileNotFoundError(f"ModernBERT dir not found: {model_dir_path}")
        instance = cls(str(model_dir_path), use_mlflow_during_load); instance.setup() # setup reloads from path
        cfg_path = model_dir_path / "model_config.json"
        if cfg_path.exists():
            with open(cfg_path, 'r') as f: instance.separator = json.load(f).get("separator", instance.separator)
        logger.info(f"ModernBERT model loaded from {model_dir_path}.")
        return instance
    
    def get_hardware_info(self) -> Dict[str, Any]:
        info = {"device": str(self.device), "cuda": torch.cuda.is_available(), "torch": torch.__version__, "transformers": transformers.__version__}
        if torch.cuda.is_available(): info["gpu_name"] = torch.cuda.get_device_name(0)
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
        if not Path(self.model_id_or_path).is_dir(): # Default to built-in if not loading a fine-tuned dir
            self.reference_texts_for_model = copy.deepcopy(self.BUILTIN_REFERENCE_TEXTS_BY_CLASS)

    def _load_custom_references_from_jsonl(self, jsonl_path: Path):
        logger.info(f"Loading ColBERT custom references from {jsonl_path}...")
        custom_refs: Dict[str, List[str]] = {}
        valid_cls = set(self.CLASS_DESCRIPTIONS.keys())
        if not jsonl_path.exists(): logger.error(f"Custom ref JSONL not found: {jsonl_path}"); return
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        item = json.loads(line)
                        text, cls_name = item.get("text"), item.get("class_name")
                        if not text or not cls_name or cls_name not in valid_cls: continue
                        custom_refs.setdefault(cls_name, []).append(text)
                    except json.JSONDecodeError: logger.warning(f"L{i+1} JSON error in {jsonl_path}")
            if not custom_refs: logger.warning(f"No valid custom refs in {jsonl_path}"); return
            self.reference_texts_for_model = custom_refs
            logger.info(f"Loaded {sum(len(v) for v in custom_refs.values())} custom ColBERT references.")
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
        if self.tokenizer.pad_token_id >= self.model.config.vocab_size: self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device).eval()
        logger.info(f"ColBERT Model '{self.model.config._name_or_path}' loaded.")

        if not self.reference_texts_for_model: logger.warning("No ColBERT references available."); return
        emb_path = cache_dir / self.PRECOMPUTED_REF_EMBEDDINGS_FILENAME if cache_dir else None
        if not force_recompute_embeddings and emb_path and emb_path.exists():
            try:
                self.reference_token_embeddings_by_class = torch.load(emb_path, map_location='cpu')
                logger.info(f"Loaded pre-computed ColBERT ref embeddings from {emb_path}.")
                return # Successfully loaded
            except Exception as e: logger.warning(f"Failed load pre-computed ColBERT embs: {e}. Recomputing.", exc_info=True)
        
        logger.info("Computing ColBERT reference embeddings..."); self._compute_and_cache_reference_embeddings(emb_path)

    def _compute_and_cache_reference_embeddings(self, embeddings_path: Optional[Path]):
        if not self.model or not self.tokenizer: raise RuntimeError("ColBERT model/tokenizer not ready for embs.")
        self.reference_token_embeddings_by_class = {}
        all_texts_flat, class_text_counts = [], {}
        for cls_name, texts in self.reference_texts_for_model.items():
            if not texts: self.reference_token_embeddings_by_class[cls_name] = []; continue
            all_texts_flat.extend(texts); class_text_counts[cls_name] = len(texts)
        if not all_texts_flat: return

        all_embs_gpu = self._get_token_embeddings_batched(all_texts_flat)
        current_idx = 0
        for cls_name, count in class_text_counts.items():
            self.reference_token_embeddings_by_class[cls_name] = [emb.cpu() for emb in all_embs_gpu[current_idx : current_idx + count]]
            current_idx += count
        if embeddings_path:
            try:
                embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.reference_token_embeddings_by_class, embeddings_path); logger.info(f"Saved ColBERT ref embs to {embeddings_path}")
            except Exception as e: logger.error(f"Failed save ColBERT ref embs: {e}", exc_info=True)

    def _get_token_embeddings_batched(self, texts: List[str], batch_size: int = 32, enable_grad: bool = False) -> List[torch.Tensor]:
        if not self.tokenizer or not self.model: raise RuntimeError("ColBERT model/tokenizer not ready.")
        all_embs_gpu = []
        # Model train/eval state should be managed by the caller.

        context_manager = torch.enable_grad() if enable_grad else torch.no_grad()

        with context_manager:
            for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing & Embedding ColBERT Batches", leave=False):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=self.DEFAULT_MAX_LENGTH).to(self.device)
                outputs = self.model(**inputs).last_hidden_state
                for j in range(outputs.size(0)):
                    all_embs_gpu.append(outputs[j][inputs['attention_mask'][j] == 1])
        return all_embs_gpu

    def _colbert_maxsim(self, query_embs: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        if query_embs.ndim != 2 or doc_embs.ndim != 2 or query_embs.size(0) == 0 or doc_embs.size(0) == 0:
            # Ensure a tensor is returned, matching dtype and device if possible
            dtype_to_use = query_embs.dtype if query_embs.numel() > 0 else (doc_embs.dtype if doc_embs.numel() > 0 else torch.float32)
            device_to_use = self.device if self.device else 'cpu'
            return torch.tensor(0.0, device=device_to_use, dtype=dtype_to_use)
        q_dev, d_dev = query_embs.to(self.device), doc_embs.to(self.device) # Ensure on correct device
        sim_matrix = torch.matmul(F.normalize(q_dev, p=2, dim=1), F.normalize(d_dev, p=2, dim=1).T)
        return torch.sum(torch.max(sim_matrix, dim=1)[0])

    def classify_text(self, text: str) -> Dict[str, Any]:
        if not self.model or not self.reference_token_embeddings_by_class: raise RuntimeError("ColBERT not setup.")
        if not text.strip(): return {"error": "Empty input", "predicted_class": "N/A", "scores_by_class": {}}
        query_embs_gpu = self._get_token_embeddings_batched([text])[0]
        scores = {}
        for cls, ref_embs_list_cpu in self.reference_token_embeddings_by_class.items():
            if not ref_embs_list_cpu: scores[cls] = 0.0; continue
            scores[cls] = (sum(self._colbert_maxsim(query_embs_gpu, ref_cpu.to(self.device)) for ref_cpu in ref_embs_list_cpu) / len(ref_embs_list_cpu)).item()
        pred_cls = max(scores, key=scores.get) if scores else "N/A"
        return {"input_text": text, "predicted_class": pred_cls, 
                "class_description": self.CLASS_DESCRIPTIONS.get(pred_cls, "N/A"),
                "scores_by_class (avg_maxsim)": scores}

    def finetune(self, reference_jsonl_path: Path, output_model_dir: Path, base_model_id: str = DEFAULT_MODEL_ID,
                 epochs: int = 3, learning_rate: float = 1e-5, batch_size: int = 4, triplet_margin: float = 0.2):
        logger.info(f"Starting ColBERT fine-tuning. Base: {base_model_id}, Output: {output_model_dir}")
        output_model_dir.mkdir(parents=True, exist_ok=True)
        self._load_custom_references_from_jsonl(reference_jsonl_path)
        if not self.reference_texts_for_model: logger.error("No ColBERT refs for fine-tuning. Aborting."); return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None: self.tokenizer.add_special_tokens({'pad_token': self.DEFAULT_PAD_TOKEN})
        self.model = AutoModel.from_pretrained(base_model_id).to(self.device)
        if self.tokenizer.pad_token_id >= self.model.config.vocab_size: self.model.resize_token_embeddings(len(self.tokenizer))

        triplets = self._prepare_triplets_for_finetuning()
        if not triplets: logger.error("No triplets for ColBERT fine-tuning. Aborting."); return
        class TripletDataset(Dataset):
            def __init__(self, d): self.d = d
            def __len__(self): return len(self.d)
            def __getitem__(self, i): return self.d[i] # Return (anchor_text, positive_text, negative_text)
        
        def collate_fn(b): return ([x[0] for x in b], [x[1] for x in b], [x[2] for x in b])
        dataloader = DataLoader(TripletDataset(triplets), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            logger.info(f"ColBERT Fine-tune Epoch {epoch+1}/{epochs}"); total_loss = 0
            prog_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Fine-tuning ColBERT", leave=False)
            for anchor_txts, pos_txts, neg_txts in prog_bar:
                if stop_signal_received: logger.info("ColBERT fine-tuning interrupted."); break
                optimizer.zero_grad()
                all_embs = self._get_token_embeddings_batched(anchor_txts + pos_txts + neg_txts, batch_size=len(anchor_txts)*3, enable_grad=True)
                n = len(anchor_txts)
                anc_e, pos_e, neg_e = all_embs[:n], all_embs[n:2*n], all_embs[2*n:]
                
                # Initialize accumulated loss for the batch
                accumulated_batch_loss_tensor = torch.tensor(0.0, device=self.device, requires_grad=True)
                actual_triplets_in_batch = 0 # Count valid triplets processed in this batch
                current_batch_sum_loss_item = 0.0 # For logging

                for i in range(n): # n is len(anchor_txts)
                    current_anchor_emb = anc_e[i]
                    current_pos_emb = pos_e[i]
                    current_neg_emb = neg_e[i]

                    if current_anchor_emb.size(0) == 0 or \
                       current_pos_emb.size(0) == 0 or \
                       current_neg_emb.size(0) == 0:
                        logger.debug(f"Skipping triplet {i} in batch due to empty embeddings for anchor, positive, or negative text.")
                        continue

                    score_pos = self._colbert_maxsim(current_anchor_emb, current_pos_emb)
                    score_neg = self._colbert_maxsim(current_anchor_emb, current_neg_emb)
                    
                    loss_value_before_relu = triplet_margin - score_pos + score_neg
                    triplet_loss = F.relu(loss_value_before_relu)
                    
                    # Accumulate loss for the batch
                    if actual_triplets_in_batch == 0: # First valid triplet
                        accumulated_batch_loss_tensor = triplet_loss
                    else:
                        accumulated_batch_loss_tensor = accumulated_batch_loss_tensor + triplet_loss
                    
                    current_batch_sum_loss_item += triplet_loss.item()
                    actual_triplets_in_batch += 1
                
                if actual_triplets_in_batch > 0:
                    mean_batch_loss = accumulated_batch_loss_tensor / actual_triplets_in_batch
                    mean_batch_loss.backward() # Backward pass on the mean loss for the batch
                    optimizer.step() # Step optimizer after gradients are computed for the batch
                    
                    avg_loss_item_for_log = current_batch_sum_loss_item / actual_triplets_in_batch
                    total_loss += avg_loss_item_for_log # Accumulate epoch total loss (sum of batch averages)
                    prog_bar.set_postfix({'loss': avg_loss_item_for_log})
                else:
                    # If all triplets in the batch were skipped, no optimizer step
                    prog_bar.set_postfix({'loss': 0.0, 'skipped_batch': True})
            if stop_signal_received: break
            logger.info(f"ColBERT Epoch {epoch+1} avg loss: {total_loss/len(dataloader) if dataloader else 0:.4f}")
        
        logger.info("ColBERT fine-tuning finished."); self._save_fine_tuned_model_assets(output_model_dir, base_model_id)
        self.model_id_or_path = str(output_model_dir.resolve()) # Update instance to point to new model
        self.setup_model_and_references(cache_dir=output_model_dir, force_recompute_embeddings=True)


    def _save_fine_tuned_model_assets(self, save_dir: Path, base_model_id_used: str):
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.model: self.model.save_pretrained(save_dir)
        if self.tokenizer: self.tokenizer.save_pretrained(save_dir)
        with open(save_dir / self.REFERENCE_TEXTS_SNAPSHOT_FILENAME, 'w') as f: json.dump(self.reference_texts_for_model, f, indent=2)
        cfg = {"base_model_id_used": base_model_id_used, "finetuned_from": self.model_id_or_path, "timestamp": time.time()}
        with open(save_dir / self.COLBERT_CONFIG_FILENAME, 'w') as f: json.dump(cfg, f, indent=2)
        logger.info(f"Fine-tuned ColBERT assets saved to {save_dir}")

    def _prepare_triplets_for_finetuning(self) -> List[Tuple[str, str, str]]:
        import random
        triplets = []; class_names = list(self.reference_texts_for_model.keys())
        if len(class_names) < 2: logger.warning("Need >=2 classes for ColBERT triplets."); return []
        for cls_name, texts in self.reference_texts_for_model.items():
            if len(texts) < 2: continue
            other_cls = [cn for cn in class_names if cn != cls_name]
            if not other_cls: continue
            for i, anchor in enumerate(texts):
                pos = texts[random.choice([j for j in range(len(texts)) if j != i])]
                neg_cls_texts = self.reference_texts_for_model.get(random.choice(other_cls))
                if neg_cls_texts: triplets.append((anchor, pos, random.choice(neg_cls_texts)))
        random.shuffle(triplets); return triplets

    @classmethod
    def load_from_directory(cls, model_directory: Path):
        logger.info(f"Loading fine-tuned ColBERT from: {model_directory}")
        if not model_directory.is_dir(): raise FileNotFoundError(f"ColBERT model dir not found: {model_directory}")
        cfg_p = model_directory / cls.COLBERT_CONFIG_FILENAME; ref_p = model_directory / cls.REFERENCE_TEXTS_SNAPSHOT_FILENAME
        if not cfg_p.exists() or not ref_p.exists(): raise FileNotFoundError("Missing ColBERT config/ref snapshot.")
        
        instance = cls(str(model_directory.resolve())) # Initialize with path
        try:
            with open(ref_p, 'r') as f: instance.reference_texts_for_model = json.load(f)
        except Exception as e: logger.error(f"Failed load ColBERT ref snapshot: {e}"); raise
        instance.setup_model_and_references(cache_dir=model_directory, force_recompute_embeddings=False)
        logger.info(f"Fine-tuned ColBERT loaded from {model_directory}")
        return instance
        
    def get_hardware_info(self) -> Dict[str, Any]: # Same as ModernBERT's
        info = {"device": str(self.device), "cuda": torch.cuda.is_available(), "torch": torch.__version__, "transformers": transformers.__version__}
        if torch.cuda.is_available(): info["gpu_name"] = torch.cuda.get_device_name(0)
        try: import flash_attn; info["flash_attn"] = True
        except ImportError: info["flash_attn"] = False
        return info

class ClassificationAPI:
    def __init__(self, modernbert_model_dir: Optional[str], host: str, port: int):
        self.modernbert_model_dir = modernbert_model_dir; self.host=host; self.port=port
        self.modernbert_classifier: Optional[ModernBERTClassifier] = None
        self.colbert_reranker: Optional[ColBERTReranker] = None
        self.app = Flask(__name__); CORS(self.app); self.request_count = 0

    def setup(self, serve_modernbert: bool, serve_colbert: bool, colbert_model_id_or_dir: Optional[str], 
              colbert_custom_ref_jsonl: Optional[str], colbert_cache_dir: Optional[str]):
        if serve_modernbert and self.modernbert_model_dir:
            try: self.modernbert_classifier = ModernBERTClassifier.load(self.modernbert_model_dir); logger.info("ModernBERT API model loaded.")
            except Exception as e: logger.error(f"Failed load ModernBERT for API: {e}", exc_info=True); serve_modernbert=False # Don't serve if load fails
        
        if serve_colbert:
            try:
                colbert_path = Path(colbert_model_id_or_dir) if colbert_model_id_or_dir else None
                if colbert_path and colbert_path.is_dir(): # Load fine-tuned
                    self.colbert_reranker = ColBERTReranker.load_from_directory(colbert_path)
                else: # Base model + optional custom refs
                    base_id = colbert_model_id_or_dir or ColBERTReranker.DEFAULT_MODEL_ID
                    self.colbert_reranker = ColBERTReranker(base_id)
                    if colbert_custom_ref_jsonl: self.colbert_reranker._load_custom_references_from_jsonl(Path(colbert_custom_ref_jsonl))
                    cache = Path(colbert_cache_dir) if colbert_cache_dir else Path.home()/".cache"/"colbert_tool_api_cache"
                    self.colbert_reranker.setup_model_and_references(cache_dir=cache)
                logger.info("ColBERT API model setup.")
            except Exception as e: logger.error(f"Failed setup ColBERT for API: {e}", exc_info=True); serve_colbert=False

        @self.app.route('/health', methods=['GET'])
        def health(): return jsonify({"status":"ok", "modernbert":self.modernbert_classifier is not None, "colbert": self.colbert_reranker is not None and self.colbert_reranker.model is not None})

        if self.modernbert_classifier:
            @self.app.route('/modernbert/classify', methods=['POST'])
            @self.app.route('/service/validate', methods=['POST'])  # Add validation endpoint alias
            def mb_classify():
                if not self.modernbert_classifier: return jsonify({"error":"ModernBERT not loaded"}),500
                data=request.get_json(); inp,outp=data.get('input_text'),data.get('output_to_classify')
                if inp is None or outp is None: return jsonify({"error":"'input_text' and 'output_to_classify' required"}),400
                try: return jsonify(self.modernbert_classifier.classify_input_output_pair(inp,outp))
                except Exception as e: return jsonify({"error":str(e)}),500
        
        if self.colbert_reranker and self.colbert_reranker.model:
            @self.app.route('/colbert/classify_sensitivity', methods=['POST'])
            def cb_classify():
                if not self.colbert_reranker or not self.colbert_reranker.model : return jsonify({"error":"ColBERT not loaded"}),500
                data=request.get_json(); txt=data.get('text')
                if txt is None: return jsonify({"error":"'text' field required"}),400
                try: return jsonify(self.colbert_reranker.classify_text(txt))
                except Exception as e: return jsonify({"error":str(e)}),500
    
    def run(self, production: bool = True):
        if not self.modernbert_classifier and (not self.colbert_reranker or not self.colbert_reranker.model):
            logger.error("No models loaded for API. Exiting."); sys.exit(1)
        logger.info(f"Starting API server on http://{self.host}:{self.port}")
        if production: serve(self.app, host=self.host, port=self.port, threads=8)
        else: self.app.run(host=self.host, port=self.port, debug=True, use_reloader=False)

# --- CLI Command Functions ---
def train_command(args: argparse.Namespace):
    # (Assumed to be complete from previous version)
    logger.info("Running 'train' command (ModernBERT)...")
    dp = DataProcessor(args.data_path)
    if not dp.load_and_validate(): logger.error("ModernBERT data load failed."); sys.exit(1)
    texts, labels = dp.prepare_classification_data(separator=args.separator)
    if not texts: logger.error("No ModernBERT training samples."); sys.exit(1)
    split = dp.perform_train_test_split(texts, labels, test_size=args.test_size, random_state=args.random_state)
    if not split['train_texts']: logger.error("No ModernBERT train data after split."); sys.exit(1)
    
    classifier = ModernBERTClassifier(model_dir=args.model_dir, use_mlflow=args.use_mlflow)
    classifier.separator = args.separator
    classifier.setup()
    classifier.train(split['train_texts'], split['train_labels'], split['test_texts'], split['test_labels'],
                     args.batch_size, args.learning_rate, args.epochs, args.gradient_accumulation_steps,
                     args.early_stopping_patience, args.warmup_ratio, args.weight_decay)
    logger.info("ModernBERT 'train' command finished.")
    if args.run_server_after_train:
        api = ClassificationAPI(args.model_dir, args.host, args.port)
        # Only serve ModernBERT after its training here
        api.setup(serve_modernbert=True, serve_colbert=False, colbert_model_id_or_dir=None, colbert_custom_ref_jsonl=None, colbert_cache_dir=None)
        api.run(production=not args.dev_server)

def serve_command(args: argparse.Namespace):
    logger.info("Running 'serve' command...")
    api = ClassificationAPI(args.modernbert_model_dir, args.host, args.port)
    api.setup(serve_modernbert=True, # Default to attempting modernbert
              serve_colbert=args.serve_colbert_sensitivity, 
              colbert_model_id_or_dir=args.colbert_model_id_or_dir,
              colbert_custom_ref_jsonl=args.colbert_custom_ref_jsonl,
              colbert_cache_dir=args.colbert_cache_dir)
    api.run(production=not args.dev_server)

def predict_modernbert_command_cli(args: argparse.Namespace):
    # (Assumed to be complete)
    logger.info("Running 'predict-modernbert'...")
    classifier = ModernBERTClassifier.load(args.model_dir)
    if not args.input_text or args.output_to_classify is None: logger.error("Need --input-text and --output-to-classify."); sys.exit(1)
    result = classifier.classify_input_output_pair(args.input_text, args.output_to_classify)
    print(json.dumps(result, indent=2))


def rerank_data_classify_command_cli(args: argparse.Namespace):
    logger.info(f"Running 'rerank-data-classify' (ColBERT)...")
    if not args.text_to_classify: logger.error("--text-to-classify required."); sys.exit(1)
    try:
        reranker: ColBERTReranker
        cb_model_path = Path(args.colbert_model_dir) if args.colbert_model_dir else None
        if cb_model_path and cb_model_path.is_dir():
            reranker = ColBERTReranker.load_from_directory(cb_model_path)
        else:
            base_id = args.colbert_model_id_or_dir or ColBERTReranker.DEFAULT_MODEL_ID
            reranker = ColBERTReranker(base_id)
            if args.custom_reference_jsonl: reranker._load_custom_references_from_jsonl(Path(args.custom_reference_jsonl))
            cache_p = Path(args.cache_dir) if args.cache_dir else Path.home()/".cache"/"colbert_cli_cache"
            model_name_for_cache = "".join(c if c.isalnum() or c in ['-','_','.'] else '_' for c in Path(base_id).name)
            reranker.setup_model_and_references(cache_dir=cache_p/model_name_for_cache)
        
        result = reranker.classify_text(args.text_to_classify)
        print(json.dumps(result, indent=2))
    except Exception as e: logger.error(f"ColBERT classify error: {e}", exc_info=True); sys.exit(1)

def finetune_colbert_command_cli(args: argparse.Namespace):
    logger.info("Running 'finetune-colbert'...")
    if not args.reference_jsonl or not args.output_model_dir: logger.error("Need --reference-jsonl and --output-model-dir."); sys.exit(1)
    reranker = ColBERTReranker(args.base_model_id) # Initialized with base
    reranker.finetune(Path(args.reference_jsonl), Path(args.output_model_dir), args.base_model_id,
                      args.epochs, args.learning_rate, args.batch_size, args.triplet_margin)

def check_hardware_command(args: argparse.Namespace): 
    # (Assumed to be complete)
    logger.info("--- Hardware Check ---")
    info = ModernBERTClassifier().get_hardware_info() # Can use either class's method
    for k,v in info.items(): logger.info(f"{k.replace('_',' ').capitalize()}: {v}")

def create_example_command(args: argparse.Namespace):
    # (Updated to include ColBERT example JSONL)
    logger.info(f"Creating example files in {args.output_dir}...")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mb_ex_path = out_dir / "sample_modernbert_training.jsonl"
    with open(mb_ex_path, 'w') as f:
        json.dump({"input": "Q?", "output_good_sample": "Good A.", "output_bad_sample": "Bad A."}, f); f.write("\n")
    cb_ex_path = out_dir / "sample_colbert_references.jsonl"
    with open(cb_ex_path, 'w') as f:
        for cn, exs in ColBERTReranker.BUILTIN_REFERENCE_TEXTS_BY_CLASS.items():
            if exs : json.dump({"text": exs[0], "class_name": cn}, f); f.write("\n")
    readme_p = out_dir / "README_examples.md"
    # (README content from previous response, updated paths if necessary)
    readme_content = f"""# Example Data
    - `{mb_ex_path.name}`: For ModernBERT `train`.
    - `{cb_ex_path.name}`: For ColBERT `finetune-colbert` or `--custom-reference-jsonl`.
    See CLI --help for commands.
    """
    with open(readme_p, 'w') as f: f.write(readme_content)
    logger.info("Example files created.")


def main_cli_entry():
    setup_signal_handling()
    parser = argparse.ArgumentParser(description="Classifier Service Tool: ModernBERT & ColBERT.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # ModernBERT Train
    train_p = subparsers.add_parser("train", help="Train ModernBERT binary classifier.")
    train_p.add_argument("--data-path", nargs='+', required=True); train_p.add_argument("--model-dir", default="models/modernbert_custom")
    train_p.add_argument("--epochs", type=int, default=3); train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--learning-rate", type=float, default=2e-5); train_p.add_argument("--test-size", type=float, default=0.15)
    train_p.add_argument("--random-state", type=int, default=42); train_p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_p.add_argument("--early-stopping-patience", type=int, default=0); train_p.add_argument("--warmup-ratio", type=float, default=0.1)
    train_p.add_argument("--weight-decay", type=float, default=0.01); train_p.add_argument("--separator", default=" [SEP] ")
    train_p.add_argument("--use-mlflow", action="store_true"); train_p.add_argument("--run-server-after-train", action="store_true")
    train_p.add_argument("--host", default="0.0.0.0"); train_p.add_argument("--port", type=int, default=5000)
    train_p.add_argument("--dev-server", action="store_true"); train_p.set_defaults(func=train_command)

    # Serve API
    serve_p = subparsers.add_parser("serve", help="Start API server.")
    serve_p.add_argument("--modernbert-model-dir", help="Dir of ModernBERT model (if serving).")
    serve_p.add_argument("--serve-colbert-sensitivity", action="store_true", help="Enable ColBERT API.")
    serve_p.add_argument("--colbert-model-id-or-dir", help="HF ID or dir of ColBERT model.")
    serve_p.add_argument("--colbert-custom-ref-jsonl", help="JSONL for custom ColBERT refs.")
    serve_p.add_argument("--colbert-cache-dir", help="Cache for ColBERT base model+custom refs.")
    serve_p.add_argument("--host", default="0.0.0.0"); serve_p.add_argument("--port", type=int, default=5000)
    serve_p.add_argument("--dev-server", action="store_true"); serve_p.set_defaults(func=serve_command)

    # Predict ModernBERT CLI
    pred_mb_p = subparsers.add_parser("predict-modernbert", help="CLI for ModernBERT classification.")
    pred_mb_p.add_argument("--model-dir", default="models/modernbert_custom", required=True)
    pred_mb_p.add_argument("--input-text", required=True); pred_mb_p.add_argument("--output-to-classify", required=True)
    pred_mb_p.set_defaults(func=predict_modernbert_command_cli)

    # Rerank Data Classify (ColBERT CLI)
    rerank_p = subparsers.add_parser("rerank-data-classify", help="Classify data sensitivity with ColBERT.")
    rerank_p.add_argument("--text-to-classify", required=True)
    rerank_p.add_argument("--colbert-model-dir", help="Path to fine-tuned ColBERT dir.")
    rerank_p.add_argument("--colbert-model-id-or-dir", help="HF ID or path to base ColBERT model (if not using --colbert-model-dir).")
    rerank_p.add_argument("--custom-reference-jsonl", help="Custom refs for base ColBERT model.")
    rerank_p.add_argument("--cache-dir", help="Base cache for ColBERT non-fine-tuned setups.")
    rerank_p.set_defaults(func=rerank_data_classify_command_cli)

    # Finetune ColBERT
    finetune_cb_p = subparsers.add_parser("finetune-colbert", help="Fine-tune ColBERT model.")
    finetune_cb_p.add_argument("--reference-jsonl", required=True); finetune_cb_p.add_argument("--output-model-dir", required=True)
    finetune_cb_p.add_argument("--base-model-id", default=ColBERTReranker.DEFAULT_MODEL_ID)
    finetune_cb_p.add_argument("--epochs", type=int, default=3); finetune_cb_p.add_argument("--learning-rate", type=float, default=1e-5)
    finetune_cb_p.add_argument("--batch-size", type=int, default=4); finetune_cb_p.add_argument("--triplet-margin", type=float, default=0.2)
    finetune_cb_p.set_defaults(func=finetune_colbert_command_cli)
    
    # Utilities
    subparsers.add_parser("check-hardware", help="Check hardware.").set_defaults(func=check_hardware_command)
    ex_p = subparsers.add_parser("create-example", help="Create example data."); ex_p.add_argument("--output-dir", default="classifier_tool_examples")
    ex_p.set_defaults(func=create_example_command)

    args = parser.parse_args()
    if hasattr(args, 'func'): args.func(args)
    else: parser.print_help()

if __name__ == "__main__":
    main_cli_entry()
