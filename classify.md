
# Data Classification & Monitoring Service
###### (Policy-Driven, Local-First, etc)

This document describes `classify.py`, a **Data Classification and Monitoring Service** that delivers plug-and-play policy-based validation for any application (read: JSON object). It orchestrates fine-tuned transformers, GGUF-based vision-language models (VLMs), and smart retrieval pipelines to provide a governance layer for multimodal content including text, image, and video.

## Overview

**Classification as a Service**: Integrate seamlessly using a single `/service/validate` endpoint that accepts content and policy specs. The service handles classification, policy enforcement, and contextual assistance.

**Highlights:**

* **Drop-in Integration**: One endpoint for all types of validation
* **Multi-Modal Ready**: Supports text, image, video, and (soon) audio
* **Policy-Driven**: Configurable via external JSON
* **Context-Aware Help**: Built-in RAG-driven documentation assistant
* **Enterprise-Focused**: Governance, observability, and compliance ready
* **VLM-Aware Docs**: Enhances markdown for indexing and retrieval

## Core Components

### 1. I/O Validator

* **Use**: Ensures `output_text` aligns with `input_text`; binary classification of `input_text`.
* **Model**: Transformer-based (e.g., DeBERTa v3); fine-tuned for response appropriateness.

### 2. Sensitivity Classifier

* **Use**: Detect PII/confidentiality sensitivity.
* **Model**: MaxSim-enabled ColBERT; compares against example classes.

### 3. Vision-Language Processor

* **Use**: Image and video content analysis.
* **Model**: LLaVA, PaliGemma, etc., via `llama-cpp-python`.
* **Features**: Frame sampling, custom prompts, policy check on derived text.

### 4. VLM Markdown Processor

* **Use**: Chunk markdown intelligently for RAG.
* **Model**: GGUF models (e.g., Pleias-RAG-1B).
* **Fallback**: Python parser for environments without VLM support.

### 5. Policy-Driven RAG Framework

* **RAGRetriever**: Manages dense vector search using SentenceTransformers.
    * Stores document texts, metadata, and their corresponding N-dimensional embedding vectors.
    * Ensures index integrity by matching document counts with embedding array dimensions.
* **Features**: Embedding cache for models, metadata filtering during retrieval.

### 6. Self-Documenting RAG System

* **Function**: Builds searchable documentation index.
* **Source**: Local or remote markdown docs.
* **Purpose**: Contextual suggestions when validation fails.

## Policy Enforcement

Driven via `/service/validate` and an external JSON file (`policy_config.json`).

**Supports:**

* Input-output coherence
* Sensitivity detection
* VLM media checks
* Required fields & regex validation
* Custom validation rules
* Documentation hints
* Self-contained env setup

## Integration Examples

### API Gateway

```js
// Pre-check user request
const resp = await fetch('/service/validate', {...});
```

### ETL Pipeline

```python
for record in data:
  result = requests.post('/service/validate', json={...})
```

### Content Moderation

```bash
curl -X POST /service/validate \
  -F 'json_payload=...' -F 'uploaded_image=@img.jpg'
```

## Setup

On first run:

* Creates venv
* Installs dependencies (Transformers, Flask, etc.)
* Re-launches in venv

```bash
python classify.py --help
```

## Policy File Example

```json
{
  "StrictPolicy": {
    "modernbert_io_validation": true,
    "colbert_input_sensitivity": true,
    "item_processing_rules": [...],
    "custom_validation_rules": [...]
  }
}
```

## Endpoints

* `POST /service/validate` — main validation
* `POST /modernbert/classify` — I/O check
* `POST /colbert/classify_sensitivity` — text classification
* `POST /rag/query` — retrieve help docs

## CLI Examples

### Start Server

```bash
python classify.py serve --policy-config-path ./policy.json
```

### Create RAG Index

```bash
python classify.py rag index --corpus-path docs.jsonl
```

## Data Formats

* I/O Validation:

```json
{"input": "Q?", "output_good_sample": "A"}
```

* ColBERT Sensitivity:

```json
{"text": "SSN: 123-45-6789", "class_name": "PII"}
```


## System Requirements

* Python 3.8+
* RAM: 16–32GB+ for VLMs
* GPU: Optional, improves performance

## Testing

```bash
python classify.py test --test-type all
```

---

