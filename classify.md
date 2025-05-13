
# Policy-Driven Classification Service with ModernBERT & ColBERT

This document describes `classify.py`, a command-line interface (CLI) and RESTful API service designed to enforce data handling policies through **graduated controls**. It uses two specialized transformer models to validate and classify text data based on predefined API sensitivity levels defined in an external policy configuration file.

## Core Components

### ModernBERT: Input-Output Validation & Single-Text Classification

- **Purpose**:
   - Validate if `output_text` is an appropriate response for a given `input_text`.
   - Perform binary classification on a single `input_text` (e.g., sentiment analysis).
- **Implementation**:
   - Uses fine-tuned `answerdotai/ModernBERT-base` models.
   - Trained on pairs of (input, good_output) and (input, bad_output) for I/O validation, or (text, label 0/1) for single-text classification.

### ColBERT: Data Sensitivity Classification

- **Purpose**: Classify text into predefined sensitivity categories (e.g., PII, Confidential, Public, or custom classes).
- **Implementation**:
   - Uses `lightonai/GTE-ModernColBERT-v1` (as a base or fine-tuned).
   - Employs the **MaxSim** technique: Compares token embeddings of input text against reference examples for each sensitivity class, summing maximum similarity scores per token to determine the best matching class.
- **Example Output**: See `example_colbert_output.json` (provided at the end of this document) for an example of ColBERT's classification output structure.

## Policy Enforcement Layer

The service's core strength lies in its **policy enforcement layer**, primarily accessed via the `/service/validate` endpoint. This layer intelligently combines checks based on:

- An API's designated "API Class" (a string identifier).
- Corresponding rules defined in an external `policy_config.json` file.

Key features of this policy-driven approach:

- **Graduated Controls**: Implement precise validation policies tailored to the sensitivity and requirements of different APIs.
- **Input-Output Coherence**: Ensures API responses are relevant and appropriate using ModernBERT validation.
- **Data Sensitivity Detection**: Identifies potentially sensitive data in inputs or outputs using ColBERT classification.
- **Externalized Policy Configuration**: Manage validation rules via the `policy_config.json` file, allowing updates without code changes.
- **Extensible Model Framework**: Supports fine-tuning of both ModernBERT and ColBERT models for domain-specific accuracy.
- **Self-Contained Setup**: Auto-creates a Python virtual environment (`.venv_classifier_service_tool`) with all necessary dependencies on its first run.

## Setup & Configuration

The script manages its own Python virtual environment. On its first run (or if not in the target venv), `classify.py` will:
1.  Create a virtual environment in `./.venv_classifier_service_tool`.
2.  Install all required Python packages into this venv.
3.  Automatically re-execute itself within the newly prepared environment.

```bash
# First-run setup (auto-creates virtual environment)
python classify.py --help
```

**Environment Requirements**:
- **Hugging Face Token**: For downloading models from the Hugging Face Hub, ensure the `HUGGING_FACE_HUB_TOKEN` (or `HF_TOKEN`) environment variable is set with your valid token. The script **does not** store or hardcode this token.
- Python 3.8+ is recommended. Sufficient RAM (e.g., 8GB+) is advised, especially when loading multiple or large models.

## Policy Configuration (`policy_config.json`)

The behavior of the `/service/validate` endpoint is governed by an external JSON configuration file (e.g., `policy_config.json`). You must specify the path to this file using the `--policy-config-path` argument when starting the server with the `serve` command.

*   **Refer to `example_policy_config.json` (provided at the end of this document) for a comprehensive example of its structure and available rule keys.**
*   **Key Fields in a Policy Definition (within `policy_config.json`):**
    The table below outlines the main configurable rules for an API class. An API class is identified by a unique string key in the `policy_config.json` file.

    | Field                           | Type          | Description                                                                                                |
    |---------------------------------|---------------|------------------------------------------------------------------------------------------------------------|
    | `description`                   | string        | Human-readable description of the policy.                                                                  |
    | `modernbert_io_validation`      | boolean       | If `true`, validate the (input\_text, output\_text) pair using the loaded ModernBERT model.                  |
    | `colbert_input_sensitivity`     | boolean       | If `true`, classify the `input_text` for sensitivity using the loaded ColBERT model.                       |
    | `colbert_output_sensitivity`    | boolean       | If `true`, classify the `output_text` for sensitivity using the loaded ColBERT model.                      |
    | `require_colbert_fine_tuned`  | boolean       | If `true` (and ColBERT checks are enabled), the policy demands a fine-tuned ColBERT model.                   |
    | `allowed_colbert_input_classes` | list[string]  | Optional. If provided, the predicted ColBERT class for `input_text` *must* be in this list.                 |
    | `disallowed_colbert_input_classes`| list[string]  | Optional. If provided, the predicted ColBERT class for `input_text` *must not* be in this list.             |
    | `allowed_colbert_output_classes`| list[string]  | Optional. If provided, the predicted ColBERT class for `output_text` *must* be in this list.                |
    | `disallowed_colbert_output_classes`| list[string] | Optional. If provided, the predicted ColBERT class for `output_text` *must not* be in this list.            |

## API Endpoints

### Primary Validation Endpoint: `/service/validate`

`POST /service/validate` - This is the main interaction point for policy-based validation.

**Request Body (JSON):**
```json
{
  "api_class": "YourAPINameOrClassID_v1", // Required. Identifier for the policy to apply.
  "input_text": "The user's input to the original API.", // Required.
  "output_text": "The API's generated response." // Optional, but required if policy involves output checks or ModernBERT I/O validation.
}
```

**Response Body (JSON):**
The response details the original request, the specific policy rules applied, the results of individual checks, and an overall status. The `overall_status` can be:

*   `"PASS"`: All required checks defined in the policy were performed and passed.
*   `"REJECT_POLICY_VIOLATION"`: A required check failed (e.g., ModernBERT predicted inappropriate, ColBERT detected a disallowed class). `violation_reasons` list will contain details.
*   `"REJECT_INVALID_POLICY"`: The provided `api_class` was not found in `policy_config.json`. `error_message` will contain details.
*   `"ERROR"`: An internal processing error occurred (e.g., a required model was not loaded but policy mandated its use, model inference error). `error_message` will contain details.

*   **Refer to `example_service_validate_pass_response.json` and `example_service_validate_reject_response.json` (provided at the end of this document) for detailed example response structures.**

**Testing the `/service/validate` Endpoint:**
Ensure the server is running (see "Starting the API Server" section). Use `curl` or a similar tool:
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{
  "api_class": "YourAPINameOrClassID_v1",
  "input_text": "User question?",
  "output_text": "Service response."
}' http://localhost:5000/service/validate
```

### Direct Model Endpoints

These endpoints provide direct access to the underlying models, useful for testing or simpler use cases not requiring the full policy layer. They are available if the respective models are loaded via the `serve` command.

*   **ModernBERT Classification:** `POST /modernbert/classify`
    *   Request Body (JSON): `{"input_text": "...", "output_text": "..."}` (output\_text can be empty string for single-text models)
*   **ColBERT Sensitivity Check:** `POST /colbert/classify_sensitivity`
    *   Request Body (JSON): `{"text": "..."}`

### Health Check Endpoint

`GET /health` - Provides the operational status of the service.

**Response (JSON Example - actual structure from script):**
```json
{
  "status": "ok", // "ok", "degraded", or "error"
  "model_availability": {
    "modernbert_loaded": true,
    "colbert_loaded": true,
    "colbert_is_fine_tuned": true,
    "colbert_reference_classes": ["Class 1: PII", "Class 5: Public Data"]
  },
  "policy_config_loaded": true,
  "policy_model_readiness": {
    "status": "ok", // "ok", "degraded", "error_models_unavailable", or "not_applicable_no_policies"
    "issues": []   // List of strings describing policy readiness issues, if any
  }
}
```

## CLI Commands for Model Operations & Fine-Tuning

Use `python classify.py <command> --help` for detailed options for each command.

### 1. ModernBERT Operations

*   **Train a ModernBERT Model (`train`)**
    Data format in JSONL:
    `{"input": "q", "output_good_sample": "good_a", "output_bad_sample": "bad_a"}` (for I/O validation)
    OR `{"input": "text", "label": 0/1}` (for single-text classification)
    ```bash
    python classify.py train --data-path ./data/modernbert_train.jsonl --model-dir ./models/modernbert --epochs 3 --learning-rate 2e-5 --batch-size 16
    ```
*   **Predict with ModernBERT via CLI (`predict-modernbert`)**
    ```bash
    python classify.py predict-modernbert --model-dir ./models/modernbert --input-text "How do I reset my password?" --output-to-classify "Please contact support@example.com"
    ```

### 2. ColBERT Operations

*   **Classify Sensitivity with ColBERT via CLI (`rerank-data-classify`)**
    ```bash
    # Using a fine-tuned model (references are part of the model package)
    python classify.py rerank-data-classify --text-to-classify "Patient ID: 12345, Diagnosis: XYZ" --colbert-model-dir ./models/colbert-medical

    # Using a base model with custom references
    python classify.py rerank-data-classify --text-to-classify "Order confirmation #98765" \
        --colbert-model-id-or-dir lightonai/GTE-ModernColBERT-v1 \
        --custom-reference-jsonl ./config/sensitivity_refs.jsonl \
        --cache-dir ./colbert_cache
    ```
*   **Fine-tune a ColBERT Model (`finetune-colbert`)**
    Reference JSONL format: `{"text": "example of ClassA text", "class_name": "ClassA"}` (one per line)
    ```bash
    python classify.py finetune-colbert --reference-jsonl ./data/colbert_train.jsonl --output-model-dir ./models/colbert-custom --base-model-id "lightonai/GTE-ModernColBERT-v1" --epochs 5 --batch-size 4
    ```
    *(Note: The script uses an internal default for `max_seq_length` for ColBERT, which is not a direct CLI argument for `finetune-colbert`.)*

## Utility Commands

*   **Create Example Files (`create-example`)**
    Generates sample data files for ModernBERT training, ColBERT custom references, and a basic policy configuration.
    ```bash
    python classify.py create-example --output-dir ./classifier_tool_examples
    ```
    This creates:
    *   `./classifier_tool_examples/sample_modernbert_training.jsonl`
    *   `./classifier_tool_examples/sample_colbert_references.jsonl`
    *   `./classifier_tool_examples/sample_policy_config.json` (a basic policy set for quickstart)
    *   `./classifier_tool_examples/README_examples.md` (with usage instructions for these files)
*   **Check Hardware (`check-hardware`)**
    Displays information about detected hardware (CPU/GPU) and relevant library versions (PyTorch, Transformers, Flash Attention).
    ```bash
    python classify.py check-hardware
    ```

## Starting the API Server (`serve` command)

The `serve` command starts the Flask API server. The availability of `/service/validate` and direct model endpoints depends on the arguments provided and the loaded policy configuration.

```bash
python classify.py serve \
    --serve-modernbert --modernbert-model-dir /path/to/your/modernbert_model \
    --serve-colbert-sensitivity --colbert-model-id-or-dir /path/to/your/colbert_model_or_id \
    --policy-config-path /path/to/your/policy_config.json \
    --colbert-custom-ref-jsonl /path/to/colbert_base_model_custom_refs.jsonl \
    --colbert-cache-dir /path/to/colbert_model_cache \
    --host 0.0.0.0 --port 5000 \
    # --dev-server  # Uncomment for development mode (uses Flask's built-in server with reloader)
```
*   Adjust model paths and flags based on which services and policies you need to enable.
*   If `--dev-server` is not used, the server runs in production mode using Waitress.
*   **Note on `--colbert-model-id-or-dir` for the `serve` command:** This argument specifies the ColBERT model. If the path provided is a directory structured as a fine-tuned ColBERT model (containing `colbert_reranker_config.json`, model files, and `reference_texts_snapshot.json`), that fine-tuned model will be loaded. Otherwise, the value is treated as a Hugging Face model ID (e.g., `lightonai/GTE-ModernColBERT-v1`) or a path to a base model directory. If it's a base model, you might need to also provide `--colbert-custom-ref-jsonl` if you are not using its default built-in references (or if it has none).

## Development & Maintenance

**Dependency Management**:
To refresh or update the virtual environment if dependencies change or issues arise:
```bash
# Remove the old virtual environment
rm -rf .venv_classifier_service_tool
# Re-run any command to trigger a fresh install of dependencies into a new venv
python classify.py --help
```

---
**Example JSON Files (Referenced Above)**

**1. `example_policy_config.json`**
*(This file defines different API classes and their validation rules)*
```json
{
  "DefaultPII_Check_Input": {
    "description": "Checks input for PII using base ColBERT, and validates I/O with ModernBERT. PII in input is rejected.",
    "modernbert_io_validation": true,
    "colbert_input_sensitivity": true,
    "colbert_output_sensitivity": false,
    "require_colbert_fine_tuned": false,
    "disallowed_colbert_input_classes": ["Class 1: PII"]
  },
  "StrictPublicOutput_FineTuned": {
    "description": "Ensures output is strictly public data using a fine-tuned ColBERT. I/O also validated.",
    "modernbert_io_validation": true,
    "colbert_input_sensitivity": false,
    "colbert_output_sensitivity": true,
    "require_colbert_fine_tuned": true,
    "allowed_colbert_output_classes": ["Class 5: Public Data"]
  },
  "ModernBERT_Only_Validation": {
    "description": "Only performs ModernBERT input-output validation. No sensitivity checks.",
    "modernbert_io_validation": true,
    "colbert_input_sensitivity": false,
    "colbert_output_sensitivity": false
  },
  "FullChecks_AllowInternalData_FineTuned": {
    "description": "All checks enabled with fine-tuned ColBERT. Allows Internal and Public data for input/output.",
    "modernbert_io_validation": true,
    "colbert_input_sensitivity": true,
    "colbert_output_sensitivity": true,
    "require_colbert_fine_tuned": true,
    "allowed_colbert_input_classes": ["Class 4: Internal Data", "Class 5: Public Data"],
    "allowed_colbert_output_classes": ["Class 4: Internal Data", "Class 5: Public Data"]
  },
  "ErrorDemo_MissingFineTunedColBERT": {
    "description": "This policy requires fine-tuned ColBERT. If only base ColBERT is loaded, it will result in an ERROR status during /service/validate.",
    "modernbert_io_validation": false,
    "colbert_input_sensitivity": true,
    "require_colbert_fine_tuned": true
  }
}
```

**2. `example_colbert_output.json`**
*(Example of what the ColBERT `/colbert/classify_sensitivity` endpoint or internal classification might return)*
```json
{
  "input_text": "My SSN is 123-45-6789 and I live at 1600 Pennsylvania Ave.",
  "predicted_class": "Class 1: PII",
  "class_description": "Most sensitive...",
  "scores_by_class (avg_maxsim)": {
    "Class 1: PII": 11.753210067749023,
    "Class 2: Sensitive Personal Data": 9.012345671234567,
    "Class 3: Confidential Personal Data": 9.876543210987654,
    "Class 4: Internal Data": 8.543210987654321,
    "Class 5: Public Data": 8.123456789012345
  }
}
```

**3. `example_service_validate_pass_response.json`**
*(Example of a successful `/service/validate` response where all checks passed)*
```json
{
  "request": {
    "api_class": "FullChecks_AllowInternalData_FineTuned",
    "input_text": "Regarding Q4 financial projections for Project Phoenix.",
    "output_text": "The Q4 projections for Project Phoenix show a 10% growth. This is internal company information."
  },
  "policy_applied": {
    "description": "All checks enabled with fine-tuned ColBERT. Allows Internal and Public data for input/output.",
    "modernbert_io_validation": true,
    "colbert_input_sensitivity": true,
    "colbert_output_sensitivity": true,
    "require_colbert_fine_tuned": true,
    "allowed_colbert_input_classes": ["Class 4: Internal Data", "Class 5: Public Data"],
    "allowed_colbert_output_classes": ["Class 4: Internal Data", "Class 5: Public Data"]
  },
  "modernbert_io_validation": {
    "prediction": 1,
    "probability_positive": 0.9921875,
    "input_text": "Regarding Q4 financial projections for Project Phoenix.",
    "output_text": "The Q4 projections for Project Phoenix show a 10% growth. This is internal company information."
  },
  "colbert_input_sensitivity": {
    "input_text": "Regarding Q4 financial projections for Project Phoenix.",
    "predicted_class": "Class 4: Internal Data",
    "class_description": "Company non-public...",
    "scores_by_class (avg_maxsim)": {
      "Class 1: PII": 7.5,
      "Class 2: Sensitive Personal Data": 8.1,
      "Class 3: Confidential Personal Data": 8.8,
      "Class 4: Internal Data": 10.5,
      "Class 5: Public Data": 9.2
    }
  },
  "colbert_output_sensitivity": {
    "input_text": "The Q4 projections for Project Phoenix show a 10% growth. This is internal company information.",
    "predicted_class": "Class 4: Internal Data",
    "class_description": "Company non-public...",
    "scores_by_class (avg_maxsim)": {
      "Class 1: PII": 7.2,
      "Class 2: Sensitive Personal Data": 7.9,
      "Class 3: Confidential Personal Data": 8.5,
      "Class 4: Internal Data": 11.1,
      "Class 5: Public Data": 9.0
    }
  },
  "overall_status": "PASS"
}
```

**4. `example_service_validate_reject_response.json`**
*(Example of a `/service/validate` response where a policy check failed, leading to rejection)*
```json
{
  "request": {
    "api_class": "DefaultPII_Check_Input",
    "input_text": "My name is John Doe and my SSN is 987-65-4321.",
    "output_text": "Hello John, I have noted your SSN."
  },
  "policy_applied": {
    "description": "Checks input for PII using base ColBERT, and validates I/O with ModernBERT. PII in input is rejected.",
    "modernbert_io_validation": true,
    "colbert_input_sensitivity": true,
    "colbert_output_sensitivity": false,
    "require_colbert_fine_tuned": false,
    "disallowed_colbert_input_classes": ["Class 1: PII"]
  },
  "modernbert_io_validation": {
    "prediction": 0,
    "probability_positive": 0.10546875,
    "input_text": "My name is John Doe and my SSN is 987-65-4321.",
    "output_text": "Hello John, I have noted your SSN."
  },
  "colbert_input_sensitivity": {
    "input_text": "My name is John Doe and my SSN is 987-65-4321.",
    "predicted_class": "Class 1: PII",
    "class_description": "Most sensitive...",
    "scores_by_class (avg_maxsim)": {
      "Class 1: PII": 12.1,
      "Class 2: Sensitive Personal Data": 9.3,
      "Class 3: Confidential Personal Data": 9.5,
      "Class 4: Internal Data": 8.2,
      "Class 5: Public Data": 8.0
    }
  },
  "overall_status": "REJECT_POLICY_VIOLATION",
  "violation_reasons": [
    "ModernBERT_IO_Validation: Predicted as inappropriate pair.",
    "ColBERT_Input_Sensitivity: Predicted class 'Class 1: PII' is in disallowed list: ['Class 1: PII']."
  ]
}
