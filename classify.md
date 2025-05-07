
# Policy-Driven ModernBERT & ColBERT Classification Service

This script provides a command-line interface and a RESTful API service implementing **graduated controls** for data validation and classification based on predefined API sensitivity levels. It leverages two types of models:

1.  **ModernBERT Binary Classification:** Fine-tuned models (`answerdotai/ModernBERT-base` base) validate if an `output_text` is appropriate for a given `input_text`. Useful for ensuring API responses match requests or adhere to specific formats/content rules.
2.  **ColBERT Data Sensitivity Classification:** Models (`lightonai/GTE-ModernColBERT-v1` base, optionally fine-tuned) classify the sensitivity of a given text (e.g., PII, Sensitive, Confidential) by comparing its token embeddings against reference examples using the MaxSim technique.

The core feature is the **policy enforcement layer** that applies different combinations of these checks based on an API's designated classification level (Class1 to Class5).

The tool is self-installing, creating a Python virtual environment (`.venv_policy_classifier_service`) with all necessary dependencies.

## Features

*   **Graduated Controls:** Enforces validation policies based on API classification (`API_CLASSIFICATION_REQUIREMENTS`).
*   **ModernBERT:**
    *   Fine-tune for specific input-output pair validation tasks.
    *   Save/load fine-tuned models.
    *   CLI for direct pair prediction.
*   **ColBERT Data Sensitivity:**
    *   Classify text sensitivity using a base model + built-in/custom references.
    *   Fine-tune the ColBERT model on custom references for improved accuracy.
    *   Save/load fine-tuned ColBERT models + associated data.
    *   CLI for direct sensitivity classification.
    *   Caching of reference embeddings.
*   **Policy-Driven API Server:**
    *   Primary endpoint (`/service/validate`) applies controls based on `api_class`.
    *   Loads required ModernBERT, base ColBERT, and fine-tuned ColBERT models.
    *   Returns detailed validation results and overall pass/reject status.
*   **Utilities:**
    *   Hardware check.
    *   Example data generation.

## Setup

The script manages its own virtual environment. Run any command to trigger setup if needed.

```bash
# Example: Show help (will trigger venv setup)
python Classify.py --help

```

## Core Concepts

### API Classification Requirements

The heart of the graduated controls is the `API_CLASSIFICATION_REQUIREMENTS` dictionary defined within the script. It maps API classes (e.g., "Class1", "Class2") to specific validation requirements:

* `modernbert_io_validation`: (Boolean) Whether to validate the (input, output) pair using a fine-tuned ModernBERT model.
* `colbert_input_sensitivity`: (Boolean) Whether to classify the sensitivity of the input text using ColBERT.
* `colbert_output_sensitivity`: (Boolean) Whether to classify the sensitivity of the output text using ColBERT.
* `require_colbert_fine_tuned`: (Boolean) If ColBERT checks are enabled, does the policy _require_ a fine-tuned ColBERT model (True), or is the base ColBERT model acceptable (False)?

### Model Roles

* __ModernBERT:__ Used when `modernbert_io_validation` is `True`. Requires a model fine-tuned on relevant good/bad examples for the API context.
* __ColBERT (Base or Fine-tuned):__ Used when `colbert_input_sensitivity` or `colbert_output_sensitivity` is `True`.
   * If `require_colbert_fine_tuned` is `False`, the service will use the configured base ColBERT model (default `lightonai/GTE-ModernColBERT-v1`) with either built-in references or custom ones provided via JSONL during setup.
   * If `require_colbert_fine_tuned` is `True`, the service _must_ be configured with the path to a directory containing a ColBERT model fine-tuned using the `finetune-colbert` command.

## CLI Usage

Use `python Classify.py <command> --help` for details.

### 1. ModernBERT Fine-tuning and Prediction

**a. Train a ModernBERT Model:**
(Command: `train`)

```bash
python Classify.py train \
    --data-path path/to/your/training_data.jsonl \
    --model-dir models/my_modernbert_validator \
    # ... other training args ...

```

**b. Predict with ModernBERT Directly (CLI):**
(Command: `predict-modernbert`)

```bash
python Classify.py predict-modernbert \
    --model-dir models/my_modernbert_validator \
    --input-text "Input Query" \
    --output-to-classify "Generated Response"

```

### 2. ColBERT Sensitivity Classification & Fine-tuning

**a. Classify Sensitivity using Base ColBERT (CLI):**
(Command: `classify-colbert`)

```bash
# Using built-in references
python Classify.py classify-colbert \
    --text-to-classify "My SSN is 123-45-6789." \
    --cache-dir ./colbert_cache

# Using custom references
python Classify.py classify-colbert \
    --text-to-classify "help@example.com" \
    --custom-reference-jsonl path/to/custom_refs.jsonl \
    --colbert-base-model-id-or-dir lightonai/GTE-ModernColBERT-v1 \
    --cache-dir ./colbert_cache

```

**b. Fine-tune a ColBERT Model:**
(Command: `finetune-colbert`)

```bash
python Classify.py finetune-colbert \
    --reference-jsonl path/to/custom_refs.jsonl \
    --output-model-dir models/my_finetuned_colbert \
    # ... other fine-tuning args ...

```

**c. Classify Sensitivity using Fine-tuned ColBERT (CLI):**
(Command: `classify-colbert`)

```bash
python Classify.py classify-colbert \
    --text-to-classify "Internal project code: Phoenix" \
    --colbert-model-dir models/my_finetuned_colbert
    # Fine-tuned model dir implies using its specific references

```

### Example Classification Output

Here is an example of the output from the `classify-colbert` command:

```json
{
  "input_text": "This is a test sentence.",
  "predicted_class": "Class 4: Internal Data",
  "class_description": "Company non-public...",
  "scores_by_class (avg_maxsim)": {
    "Class 1: PII": 5.612870454788208,
    "Class 2: Sensitive Personal Data": 5.708996772766113,
    "Class 3: Confidential Personal Data": 5.620632171630859,
    "Class 4: Internal Data": 5.821521043777466,
    "Class 5: Public Data": 5.6775946617126465
  }
}
```
```json
{
  "input_text": "My SSN is 123-45-6789.",
  "predicted_class": "Class 1: PII",
  "class_description": "Most sensitive...",
  "scores_by_class (avg_maxsim)": {
    "Class 1: PII": 9.662119388580322,
    "Class 2: Sensitive Personal Data": 8.076715469360352,
    "Class 3: Confidential Personal Data": 8.788877487182617,
    "Class 4: Internal Data": 8.027799129486084,
    "Class 5: Public Data": 8.07073450088501
  }
}
```

### 4. API Server

(Command: `serve-policy-api`)
Starts the API server, loading the specified models to enforce policies.

```bash
python Classify.py serve-policy-api \
    --modernbert-model-dir models/my_modernbert_validator \
    --colbert-base-model-id-or-dir lightonai/GTE-ModernColBERT-v1 \
    --colbert-finetuned-model-dir models/my_finetuned_colbert \
    --port 5000 \
    # --colbert-custom-ref-jsonl-for-base path/to/custom_refs.jsonl # If base needs custom refs
    # --colbert-cache-dir ./api_cache # For base model + custom refs cache
    # --dev-server # For development

```

### 5. Utilities

**a. Create Example Files:**
(Command: `create-example`)

```bash
python Classify.py create-example --output-dir ./my_classifier_examples

```

**b. Check Hardware:**
(Command: `check-hardware`)

```bash
python Classify.py check-hardware

```

## Policy-Driven API Endpoint

The primary way to interact with the service for graduated controls is via the `/service/validate` endpoint.

* **Endpoint:** `POST /service/validate`

* **Request Body (JSON):**

```json
{
  "api_class": "Class1", // Or "Class2", "Class3", etc. Required.
  "input_text": "The user's input to the original API.", // Required.
  "output_text": "The API's generated response." // Optional, but needed if policy requires output checks.
}

```

* **Response Body (JSON):**
   * `request`: Echos the input request fields.
   * `policy_applied`: The dictionary from `API_CLASSIFICATION_REQUIREMENTS` for the requested `api_class`.
   * `modernbert_io_validation`: (Object | Null) Result from `ModernBERTClassifier.classify_input_output_pair` if the check was run. Contains `prediction`, `probability_positive`, etc., or `status`/`reason` if skipped/error.
   * `colbert_input_sensitivity`: (Object | Null) Result from `ColBERTReranker.classify_text` on the input if the check was run. Contains `predicted_class`, `scores_by_class`, etc., or `status`/`reason`.
   * `colbert_output_sensitivity`: (Object | Null) Result from `ColBERTReranker.classify_text` on the output if the check was run.
   * `overall_status`: (String)
      * `"PASS"`: All required checks were performed and passed according to basic policy rules (e.g., ModernBERT prediction != 0, critical sensitivity classes not detected where forbidden).
      * `"REJECT_POLICY_VIOLATION"`: A required check failed (e.g., ModernBERT predicted 0, sensitive data detected).
      * `"REJECT_INVALID_POLICY"`: The provided `api_class` was not found.
      * `"ERROR"`: An internal error occurred during processing (e.g., required model not loaded).

* **Example Response Snippet:**

```json
{
  "request": {
    "api_class": "Class2",
    "input_text": "What is my address?",
    "output_text": "123 Main St"
  },
  "policy_applied": {
    "description": "Highly restricted APIs...",
    "modernbert_io_validation": true,
    "colbert_input_sensitivity": true,
    "colbert_output_sensitivity": true,
    "require_colbert_fine_tuned": false
  },
  "modernbert_io_validation": {
    "prediction": 1,
    "probability_positive": 0.95, ...
  },
  "colbert_input_sensitivity": {
    "predicted_class": "Class5_Public", ...
  },
  "colbert_output_sensitivity": {
    "predicted_class": "Class2_SensitivePersonal", ...
  },
  "overall_status": "REJECT_POLICY_VIOLATION" // Rejected due to sensitive output
}

```

## Directory Structure & Caching

* __ModernBERT Models:__ Saved via `train --model-dir`. Contains standard Hugging Face model/tokenizer files + `model_config.json`.
* __Fine-tuned ColBERT Models:__ Saved via `finetune-colbert --output-model-dir`. Contains model/tokenizer files + `colbert_reranker_config.json`, `reference_texts_snapshot.json`, `ref_embeddings.pt`.
* __ColBERT Cache:__ Specified via `--cache-dir` or `--colbert-cache-dir`. Used to store pre-computed embeddings for _base_ ColBERT models when used with specific reference sets (built-in or custom JSONL). This avoids recomputing embeddings on each startup or CLI run. The structure might be like `./cache_dir/base_model_cache/<sanitized_model_name>/ref_embeddings.pt`.

## Considerations

* **Model Loading:** The `serve-policy-api` command needs paths to potentially three different models (ModernBERT, base ColBERT, fine-tuned ColBERT). Ensure the correct paths are provided based on the policies you intend to support.
* __Policy Logic:__ The current `validate_interaction` implements basic checks (ModernBERT prediction=0 -> reject, sensitive class detection -> reject). More complex rejection logic based on combinations of results might be needed for specific use cases.
* **Resource Usage:** Loading multiple large language models requires significant RAM/VRAM.

```text

```