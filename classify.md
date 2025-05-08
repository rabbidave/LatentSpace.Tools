# Policy-Driven Classification Service with ModernBERT & ColBERT

This document describes a command-line interface (CLI) and RESTful API service designed to enforce data handling policies through **graduated controls**. It uses two specialized transformer models to validate and classify text data based on predefined API sensitivity levels:

1. __ModernBERT for Input-Output Validation:__ Leverages fine-tuned `answerdotai/ModernBERT-base` models to determine if an API's `output_text` is an appropriate response to a given `input_text`. This is crucial for ensuring response relevance, adherence to formats, or content guidelines.
2. **ColBERT for Data Sensitivity Classification:** Employs `lightonai/GTE-ModernColBERT-v1` (base or fine-tuned) to classify the sensitivity of text (e.g., PII, Sensitive, Confidential). It achieves this by comparing text token embeddings against reference examples using the MaxSim (maximum similarity) technique.

The service's core strength lies in its **policy enforcement layer**. This layer intelligently combines these checks based on an API's designated "API Class" (e.g., Class1 to Class5), allowing for flexible and robust data governance.

The tool is self-installing, creating a Python venv (`.venv_policy_classifier_service`) with all necessary dependencies on its first run.

## Why This Tool?

* **Graduated Controls:** Implement precise validation policies tailored to the sensitivity of different APIs.
* **Input-Output Coherence:** Ensure API responses are relevant and appropriate for the requests they serve using ModernBERT.
* **Data Detection:** Identify potentially sensitive data (PII, confidential info) in inputs or outputs using ColBERT before it's mishandled.
* __Centralized Policy Management:__ Define and manage data validation rules in one place (`API_CLASSIFICATION_REQUIREMENTS`).
* **Extensible Model Framework:** Fine-tune both ModernBERT and ColBERT models for domain-specific accuracy.
* **Self-Contained & Easy Setup:** Automatic virtual environment creation simplifies deployment and dependency management.

## Core Components & Concepts

### 1. ModernBERT: Input-Output Validation

* __Purpose:__ To validate if an `output_text` is a "good" or "appropriate" response given an `input_text`.
* **How it Works:** A binary classifier fine-tuned on examples of good and bad input-output pairs.
* **Use Case:** Ensuring chatbot responses are relevant, API outputs match expected formats, or preventing undesirable content generation.

It's important to note that ModernBERT can also be trained as a **single-text binary classifier**. By using the `train` command with data in the `{"input": "...", "label": ...}` format (where `label` is 0 or 1), the model can be trained to predict a binary outcome based solely on the `input` text, without requiring a separate `output_text` for training. This is useful for tasks like sentiment analysis or simple text categorization.

### 2. ColBERT: Data Sensitivity Classification

* **Purpose:** To classify a piece of text into predefined sensitivity categories (e.g., PII, Confidential, Public).
* **How it Works:**
   * Generates token-level embeddings for the input text and for a set of reference texts representing each sensitivity class.
   * Uses the **MaxSim** technique: For each token in the input text, it finds the maximum similarity score against all tokens in a reference document's embeddings. These per-token maximums are summed to get a query-document score.
   * The input text is assigned the class of the reference set it scores highest against.

* **Use Case:** Detecting PII in user queries, classifying the sensitivity of generated API responses, ensuring data is handled according to its classification.
* **Models:**
   * **Base ColBERT:** Uses a pre-trained model (e.g., `lightonai/GTE-ModernColBERT-v1`) with built-in or custom-provided reference examples.
   * **Fine-tuned ColBERT:** The base ColBERT model can be further fine-tuned on your specific reference examples for improved domain-specific accuracy.

### 3. The Policy Enforcement Layer & API Classification

The service's intelligence resides in the `API_CLASSIFICATION_REQUIREMENTS` dictionary (defined within `Classify.py`). This configuration maps an `api_class` (a string like "Class1", "Class2", etc., that you assign to your APIs) to a specific set of validation rules:

* `modernbert_io_validation` (Boolean): If `True`, the (input_text, output_text) pair will be validated by a fine-tuned ModernBERT model.
* `colbert_input_sensitivity` (Boolean): If `True`, the `input_text` will be classified for sensitivity using ColBERT.
* `colbert_output_sensitivity` (Boolean): If `True`, the `output_text` will be classified for sensitivity using ColBERT.
* `require_colbert_fine_tuned` (Boolean):
   * If `True` (and ColBERT checks are enabled): The policy *demands* a fine-tuned ColBERT model.
   * If `False` (and ColBERT checks are enabled): The base ColBERT model (with appropriate references) is acceptable.

**This policy-driven approach is the primary way to use the service for comprehensive validation.**

## Setup

The script manages its own Python virtual environment. Simply run any command, and the script will automatically create (`.venv_policy_classifier_service`), activate, and install dependencies if it's the first time or if it's not running in the venv. It may restart itself to ensure it's operating within the correct environment.

```bash
# Example: Show help (this will trigger the venv setup if needed)
python Classify.py --help

```

## Using the Policy-Driven Service (Primary API)

The main interaction point for policy-based validation is the `/service/validate` API endpoint.

* **Endpoint:** `POST /service/validate`

* __Purpose:__ To apply graduated controls based on the provided `api_class`.

* **Request Body (JSON):**

```json
{
  "api_class": "Class1", // Required. Your API's classification level.
  "input_text": "The user's input to the original API.", // Required.
  "output_text": "The API's generated response." // Optional, but required if policy involves output checks.
}

```

* **Server Configuration:** When starting the policy service (using the `serve-policy-api` command, detailed later), you'll need to provide paths to:

   * A fine-tuned ModernBERT model (if any policy uses `modernbert_io_validation`).
   * A base ColBERT model ID (can be default) or path.
   * Optionally, a fine-tuned ColBERT model directory (if any policy uses `require_colbert_fine_tuned: true`).
   * Optionally, custom reference examples for the base ColBERT model.

* **Response Body (JSON):** The response details the checks performed and the overall outcome.

   * `request`: An echo of the input request.
   * `policy_applied`: The specific policy rules fetched from `API_CLASSIFICATION_REQUIREMENTS` for the given `api_class`.
   * `modernbert_io_validation`: Result from ModernBERT if run (prediction, probability, etc.), or a status if skipped/error.
   * `colbert_input_sensitivity`: Result from ColBERT on input text if run (predicted class, scores, etc.), or a status.
   * `colbert_output_sensitivity`: Result from ColBERT on output text if run.
   * `overall_status`:
      * `"PASS"`: All required checks performed and passed.
      * `"REJECT_POLICY_VIOLATION"`: A required check failed (e.g., ModernBERT inappropriate, or sensitive data detected where forbidden).
      * `"REJECT_INVALID_POLICY"`: The `api_class` was not found.
      * `"ERROR"`: Internal processing error (e.g., a required model was not loaded).

* **Example Response Snippet:**

```json
{
  "request": {
    "api_class": "Class2",
    "input_text": "What is my address?",
    "output_text": "123 Main St"
  },
  "policy_applied": {
    "description": "Highly restricted APIs...", // Example field in policy
    "modernbert_io_validation": true,
    "colbert_input_sensitivity": true,
    "colbert_output_sensitivity": true,
    "require_colbert_fine_tuned": false
  },
  "modernbert_io_validation": {
    "prediction": 1, // Assuming 1 is 'appropriate'
    "probability_positive": 0.95
    // ... other ModernBERT details
  },
  "colbert_input_sensitivity": {
    "predicted_class": "Class5_Public", // Input was non-sensitive
    // ... other ColBERT details
  },
  "colbert_output_sensitivity": {
    "predicted_class": "Class2_SensitivePersonal", // Output was sensitive
    // ... other ColBERT details
  },
  "overall_status": "REJECT_POLICY_VIOLATION" // Rejected because sensitive output violated policy for Class2
}

```

* **Testing the `/service/validate` Endpoint:**

Use tools like `curl` or PowerShell's `Invoke-RestMethod`. (Ensure the server is running via the `serve-policy-api` command).

**Using curl (Linux/macOS/WSL):**

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{
  "api_class": "Class2",
  "input_text": "What is my name?",
  "output_text": "Your name is John Doe."
}' http://localhost:5000/service/validate

```

**Using PowerShell:**

```powershell
$body = @{
    api_class = "Class2"
    input_text = "What is my name?"
    output_text = "Your name is John Doe."
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:5000/service/validate -Method Post -ContentType "application/json" -Body $body

```

## Model-Specific Operations & Fine-Tuning (CLI)

Beyond the policy service, you can directly train, fine-tune, and test the ModernBERT and ColBERT models using the CLI. Use `python Classify.py <command> --help` for detailed options.

### 1. ModernBERT: Input-Output Validation

__a. Train a ModernBERT Model:__
(Command: `train`)
You need a `.jsonl` file where each line is a JSON object, typically with `{"input": "...", "output_good_sample": "...", "output_bad_sample": "..."}` to teach the model what's appropriate. As mentioned above, for single-text binary classification, the data should be in the `{"input": "...", "label": ...}` format.

```bash
python Classify.py train \
    --data-path path/to/your/training_data.jsonl \
    --model-dir models/my_modernbert_validator \
    # ... other training arguments like --epochs, --learning-rate ...

```

**b. Predict with a Trained ModernBERT (CLI):**
(Command: `predict-modernbert`)
Test your fine-tuned ModernBERT model directly.

```bash
python Classify.py predict-modernbert \
    --model-dir models/my_modernbert_validator \
    --input-text "User Query: Tell me a joke." \
    --output-to-classify "Response: I am a large language model."
```

### 2. ColBERT: Data Sensitivity Classification

**a. Classify Sensitivity with Base ColBERT (CLI):**
**Windows Note:** When using text with spaces or special characters, use PowerShell escaping:
```powershell
python Classify.py classify-colbert --text-to-classify '\"Text with spaces & special chars\"' ...
```
Or pass text via file:
```powershell
python Classify.py classify-colbert --text-to-classify (Get-Content input.txt -Raw) ...
```
(Command: `classify-colbert`)
Use a pre-trained ColBERT model with either built-in sensitivity reference examples or your own.

```bash
# Using built-in references (PII, Sensitive, Confidential, etc.)
python Classify.py classify-colbert \
    --text-to-classify "My social security number is 123-45-6789." \
    --cache-dir ./colbert_cache # Caches embeddings for faster subsequent runs

# Using custom references from a JSONL file
# custom_refs.jsonl: {"text": "example of TypeA data", "class_name": "MyCustomClassA"} (one per line)
python Classify.py classify-colbert \
    --text-to-classify "user@company.com is an internal email." \
    --custom-reference-jsonl path/to/custom_refs.jsonl \
    --colbert-base-model-id-or-dir lightonai/GTE-ModernColBERT-v1 \
    --cache-dir ./colbert_cache

```

**b. Fine-tune a ColBERT Model:**
(Command: `finetune-colbert`)
Improve sensitivity classification for your specific data by fine-tuning ColBERT on your reference examples. The reference JSONL should contain texts mapped to your defined sensitivity classes.

```bash
python Classify.py finetune-colbert \
    --reference-jsonl path/to/your_sensitivity_references.jsonl \
    --output-model-dir models/my_finetuned_colbert \
    # ... other fine-tuning arguments like --epochs, --learning-rate ...

```

**c. Classify Sensitivity with Fine-tuned ColBERT (CLI):**
(Command: `classify-colbert`)
When you provide a directory containing a fine-tuned ColBERT model, it automatically uses the references it was fine-tuned on.

```bash
python Classify.py classify-colbert \
    --text-to-classify "Project Dragonfire is our secret internal project." \
    --colbert-model-dir models/my_finetuned_colbert
    # No need for --custom-reference-jsonl; it's part of the fine-tuned model package.

```

## Starting the API Server(s)

There are two ways to serve models:

1. **Policy-Driven Service (Recommended for main use):** Command `serve-policy-api`.
2. **Direct Model Service (For testing individual models):** Command `serve`.

### `serve-policy-api` (Primary Service Endpoint)

This command starts the API server exposing the `/service/validate` endpoint. You must configure it with paths to the models required by your policies.

```bash
python Classify.py serve-policy-api \
    --modernbert-model-dir path/to/your/fine-tuned_modernbert \
    --colbert-base-model-id-or-dir lightonai/GTE-ModernColBERT-v1 \ # Or path to a local base model
    # Optional: if your policies require a fine-tuned ColBERT
    --colbert-fine-tuned-model-dir path/to/your/fine-tuned_colbert \
    # Optional: if using base ColBERT with custom references not bundled with a fine-tuned model
    --colbert-custom-ref-jsonl path/to/base_colbert_custom_refs.jsonl \
    --colbert-cache-dir ./api_colbert_cache \
    --port 5000

```

### `serve` (Direct Model Endpoints)

This command can start a server exposing individual model endpoints like `/modernbert/classify` and `/colbert/classify_sensitivity`. Useful for direct testing or simpler use cases not requiring the full policy layer.

The `/modernbert/classify` endpoint is primarily designed for input-output validation and expects both `input_text` and `output_to_classify` in the request body. However, it can still be used with a ModernBERT model trained for single-text binary classification (using `{"input": "...", "label": ...}` training data). In this scenario, the `output_to_classify` parameter still needs to be provided in the request body, even if it's an empty string or a placeholder. The model will primarily use the `input_text` for prediction based on its single-text training.

```bash
# Example: Serve only a ModernBERT model
python Classify.py serve --modernbert-model-dir models/my_modernbert_validator

# Example: Serve only a ColBERT model (fine-tuned)
python Classify.py serve --serve-colbert-sensitivity --colbert-model-id-or-dir models/my_finetuned_colbert

# Example: Serve both, with ColBERT using base model + custom refs
python Classify.py serve \
    --modernbert-model-dir models/my_modernbert_validator \
    --serve-colbert-sensitivity \
    --colbert-model-id-or-dir lightonai/GTE-ModernColBERT-v1 \
    --colbert-custom-ref-jsonl path/to/custom_refs.jsonl \
    --colbert-cache-dir ./colbert_cache

```

## Utilities

**a. Create Example Data Files:**
(Command: `create-example`)
Generates sample JSONL files for ModernBERT training and ColBERT custom references.

```bash
python Classify.py create-example --output-dir ./my_classifier_examples
# This will create:
# ./my_classifier_examples/sample_modernbert_training.jsonl
# ./my_classifier_examples/sample_colbert_references.jsonl

```

**b. Check Hardware:**
(Command: `check-hardware`)
Displays information about available hardware (CPU/GPU, PyTorch, Transformers version).

```bash
python Classify.py check-hardware

```

## Directory Structure & Caching

* **Fine-tuned ModernBERT Models (`--model-dir` for `train`):**
   * Contains standard Hugging Face model files (`pytorch_model.bin`, `config.json`, etc.) and tokenizer files.
   * Includes a `model_config.json` with tool-specific settings (e.g., separator token).

* **Fine-tuned ColBERT Models (`--output-model-dir` for `finetune-colbert`):**
   * Contains Hugging Face model and tokenizer files.
   * `colbert_reranker_config.json`: Metadata about the fine-tuning.
   * `reference_texts_snapshot.json`: A copy of the reference texts used for fine-tuning.
   * `ref_embeddings.pt`: Pre-computed embeddings for these reference texts.

* **ColBERT Cache (`--cache-dir` or `--colbert-cache-dir`):**
   * Used by **base** ColBERT models (when not using a fine-tuned ColBERT directory) to store pre-computed embeddings of reference texts (either built-in or from a `--custom-reference-jsonl`).
   * This avoids re-calculating embeddings on every run or server start.
   * Structure might be: `./your_cache_dir/sanitized_base_model_name/ref_embeddings.pt`.

## Important Considerations

* __Model Loading for Policy Service:__ The `serve-policy-api` command is flexible. You only need to provide paths for models that your defined `API_CLASSIFICATION_REQUIREMENTS` will actually use. For example, if no policy requires `modernbert_io_validation`, you don't need to supply `--modernbert-model-dir`.
* __Policy Logic Customization:__ The `overall_status` determination in the `/service/validate` endpoint is based on straightforward rules (e.g., ModernBERT predicting "inappropriate" means reject, or detection of a highly sensitive class means reject). For more nuanced decision-making (e.g., "reject if ModernBERT fails AND ColBERT detects PII"), you would need to modify the `validate_interaction` logic within `Classify.py`.
* **Resource Consumption:** Language models are resource-intensive. Loading multiple models (ModernBERT, base ColBERT, fine-tuned ColBERT) simultaneously for the policy service will require substantial RAM and VRAM (if using GPUs). Plan your hardware accordingly.
* **Security:** Ensure that any custom reference data or training data used does not inadvertently contain sensitive information that could be exposed or learned by the models in undesirable ways.
