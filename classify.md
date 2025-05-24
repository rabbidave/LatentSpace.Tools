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

<summary>

##### Key Concepts & Components:

</summary>



<details>

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

* **Function**: Builds searchable documentation index (from this document or others).
* **Source**: Local or remote markdown docs.
* **Purpose**: Contextual suggestions when validation fails.

</details>

## Policy Enforcement

Driven via `/service/validate` and an external JSON file (`policy_config.json`).

**Supports:**

* Input-output coherence
* Sensitivity detection
* VLM media checks
* Required fields & regex validation
* Custom validation rules
* Documentation hints (via RAG if policy violations occur)
* Self-contained env setup

## Integration Examples

### API Gateway

```javascript
// Pre-check user request and AI-generated response
async function validateAIResponse(userQuery, aiResponse) {
    const validationPayload = {
        api_class: 'UserInputHandlingPolicy', // Defined in your policy_config.json
        input_text: userQuery,
        output_text: aiResponse,
        request_id: `gateway-req-${Date.now()}`
    };

    try {
        const validationResponse = await fetch('http://classify-service:8080/service/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(validationPayload)
        });

        if (!validationResponse.ok) {
            throw new Error(`Validation service error: ${validationResponse.status}`);
        }

        const result = await validationResponse.json();

        if (result.overall_status === 'REJECT_POLICY_VIOLATION') {
            console.error('Policy Violation:', result.violation_reasons);
            // Use documentation suggestions to provide guidance
            const suggestions = result.documentation_suggestions?.suggestions || [];
            
            return { 
                success: false,
                error: 'Content policy violation', 
                violations: result.violation_reasons,
                helpfulSuggestions: suggestions.map(s => ({
                    title: s.title,
                    preview: s.content_preview
                }))
            };
        }

        // Validation passed
        return {
            success: true,
            response: aiResponse
        };

    } catch (error) {
        console.error('Validation service communication error:', error);
        // Decide whether to fail open or closed
        return {
            success: false,
            error: 'Validation service unavailable',
            details: error.message
        };
    }
}

// Example usage
const userQuery = "Tell me about project X";
const aiResponse = "Project X is a super secret initiative, details are classified.";

const validationResult = await validateAIResponse(userQuery, aiResponse);

if (!validationResult.success) {
    // Handle policy violation or service error
    console.log('Cannot return response:', validationResult.error);
    if (validationResult.helpfulSuggestions) {
        console.log('Helpful information:', validationResult.helpfulSuggestions);
    }
} else {
    // Safe to return the AI response
    console.log('Approved response:', validationResult.response);
}
```


**Policy (`UserInputHandlingPolicy` in `policy_config.json`):**
```json
{
  "UserInputHandlingPolicy": {
    "description": "Validates AI responses against user inputs for sensitivity and appropriateness.",
    "modernbert_io_validation": true,
    "colbert_input_sensitivity": true,
    "allowed_colbert_input_classes": ["Class 3: Internal", "Class 4: Public"],
    "colbert_output_sensitivity": true,
    "disallowed_colbert_output_classes": ["Class 1: PII", "Class 2: Confidential"],
    "documentation_assistance": {
      "enabled": true,
      "index_path": "./tool_documentation",
      "max_total_suggestions": 3
    }
  }
}
```

**How it works:**
1. The API Gateway intercepts a user query and the AI's intended response.
2. It sends both to `/service/validate` with the appropriate `api_class`.
3. The service might use `ModernBERTClassifier` to check if the `output_text` is coherent with `input_text`.
4. `ColBERTReranker` checks `input_text` to ensure it's not overly sensitive itself, and more importantly, checks `output_text` to prevent leaking PII or Confidential information.
5. If `output_text` (e.g., "Project X is super secret") is classified as "Confidential" and this class is disallowed by the policy, a `REJECT_POLICY_VIOLATION` is returned.
6. The `documentation_suggestions` in the response would then point to sections in this document (if indexed by RAG) about handling confidential data or specific project mentions.

</details>

### ETL Pipeline

```python
# ETL pipeline for processing documents, with classification monitoring
import requests
import uuid
import logging

logger = logging.getLogger(__name__)

class DocumentValidator:
    def __init__(self, service_url='http://classify-service:8080'):
        self.service_url = service_url
        self.validate_endpoint = f"{service_url}/service/validate"
    
    def validate_document(self, doc_content, doc_id, pipeline_stage, metadata=None):
        """
        Validate document content against policy for a specific pipeline stage.
        
        Args:
            doc_content: The document text to validate
            doc_id: Unique identifier for the document
            pipeline_stage: Current stage in the pipeline (e.g., 'Ingestion', 'Transformation')
            metadata: Optional additional context
            
        Returns:
            tuple: (is_valid: bool, validation_result: dict)
        """
        validation_payload = {
            'api_class': f'Pipeline_{pipeline_stage}_IntegrityPolicy',
            'input_text': doc_content,
            'request_id': f"etl-{pipeline_stage}-{doc_id}",
        }
        
        # Add optional metadata
        if metadata:
            validation_payload['metadata_fields'] = metadata
        
        try:
            response = requests.post(
                self.validate_endpoint, 
                json=validation_payload,
                timeout=30  # Add timeout for reliability
            )
            response.raise_for_status()
            
            result = response.json()
            is_valid = result.get('overall_status') == 'PASS'
            
            if not is_valid:
                self._log_compliance_violation(doc_id, pipeline_stage, result)
                
            return is_valid, result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Validation service call failed for doc {doc_id} at stage {pipeline_stage}: {e}")
            return False, {'error': str(e)}
    
    def _log_compliance_violation(self, doc_id, stage, validation_result):
        """Log compliance violations with helpful context."""
        violations = validation_result.get('violation_reasons', [])
        logger.error(f"Compliance VIOLATION for doc {doc_id} at stage {stage}: {violations}")
        
        # Log any helpful suggestions
        suggestions = validation_result.get('documentation_suggestions', {}).get('suggestions', [])
        if suggestions:
            logger.info(f"Remediation suggestions for {doc_id}: {[s['title'] for s in suggestions]}")

# Example usage in an ETL pipeline
def process_financial_documents(documents):
    validator = DocumentValidator()
    processed_docs = []
    quarantined_docs = []
    
    for doc in documents:
        doc_id = str(uuid.uuid4())
        
        # Stage 1: Ingestion validation
        is_valid, result = validator.validate_document(
            doc_content=doc['content'],
            doc_id=doc_id,
            pipeline_stage='Ingestion',
            metadata={
                'document_source': 'financial_reports_q1_2024',
                'processing_step': 'pii_scan'
            }
        )
        
        if not is_valid:
            quarantined_docs.append({
                'doc_id': doc_id,
                'stage_failed': 'Ingestion',
                'violations': result.get('violation_reasons', [])
            })
            continue
            
        # Stage 2: Additional processing...
        processed_doc = transform_document(doc)
        
        # Stage 3: Pre-storage validation
        is_valid, result = validator.validate_document(
            doc_content=processed_doc['content'],
            doc_id=doc_id,
            pipeline_stage='PreStorage'
        )
        
        if is_valid:
            processed_docs.append(processed_doc)
        else:
            quarantined_docs.append({
                'doc_id': doc_id,
                'stage_failed': 'PreStorage',
                'violations': result.get('violation_reasons', [])
            })
    
    return processed_docs, quarantined_docs

def transform_document(doc):
    """Placeholder for document transformation logic."""
    return doc

# Example test
if __name__ == "__main__": # This __main__ block is specific to the ETL example
    # Configure basic logging for the ETL example if run directly
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    test_docs = [
        {"content": "Q1 revenue was $1.2M with growth in key segments."},
        {"content": "Employee SSN 123-45-6789 and salary $85,000."},
    ]
    
    # valid_docs, quarantined = process_financial_documents(test_docs)
    # print(f"Processed: {len(valid_docs)}, Quarantined: {len(quarantined)}")
    pass # Comment out actual execution for documentation purposes
```



**Policy (`Pipeline_Ingestion_IntegrityPolicy` in `policy_config.json`):**
```json
{
  "Pipeline_Ingestion_IntegrityPolicy": {
    "description": "Ensures data ingested into the pipeline meets PII and content standards.",
    "colbert_input_sensitivity": true,
    "disallowed_colbert_input_classes": ["Class 1: PII"],
    "custom_validation_rules": [
      {
        "type": "text_length_limit", 
        "max_length": 100000,
        "text_fields": ["input_text"]
      }
    ],
    "documentation_assistance": {
      "enabled": true,
      "index_path": "./tool_documentation_etl",
      "max_total_suggestions": 2
    }
  }
}
```

**How it works:**
1. As documents flow through an ETL pipeline, specific stages can call `/service/validate`.
2. The `api_class` can be dynamic based on the `pipeline_stage`.
3. The `input_text` would be the content of the document/record being processed.
4. Policies check for PII, formatting, size limits, etc.
5. Violations result in logging, quarantine, or alerts.
6. `documentation_suggestions` guide data stewards on remediation.

</details>


**Policy (`UserGeneratedContentPolicy` in `policy_config.json`):**
```json
{
  "UserGeneratedContentPolicy": {
    "description": "Moderates user-generated text and images for inappropriate content and PII.",
    "colbert_input_sensitivity": true,
    "disallowed_colbert_input_classes": ["Class 1: PII", "HateSpeechClass", "ViolentContentClass"],
    "item_processing_rules": [
      {
        "item_type": "image",
        "vlm_processing": {
          "required": true,
          "prompt": "Analyze this image for nudity, violence, hate symbols, or other policy-violating content. Describe any text present."
        },
        "derived_text_checks": {
          "colbert_sensitivity": true,
          "disallowed_classes": ["ExplicitContentClass", "ViolentImageryClass"],
          "blocked_keywords": ["gore", "graphic_violence_depiction"]
        }
      }
    ],
    "custom_validation_rules": [
      {
        "type": "text_length_limit",
        "max_length": 5000,
        "text_fields": ["input_text"]
      }
    ],
    "documentation_assistance": {
      "enabled": true,
      "index_path": "./community_guidelines_rag",
      "max_total_suggestions": 3
    }
  }
}
```

**How it works:**
1. User submits post with text and image.
2. Backend sends multipart request to `/service/validate`.
3. Text is checked for PII/profanity.
4. Image is analyzed by VLM.
5. VLM description is checked for policy violations.
6. Violations trigger rejection with helpful suggestions.

</details>

## Setup

On first run:

* Creates venv
* Installs dependencies (Transformers, Flask, etc.)
* Re-launches in venv

```bash
python classify.py --help
```

## Policy File Example

(Refer to detailed examples within each Integration Example section above or a separate `policy_config.md` for full schema and options.)
```json
{
  "StrictPIIPolicy": {
    "description": "Focuses on identifying and rejecting PII in input text.",
    "modernbert_io_validation": false,
    "colbert_input_sensitivity": true,
    "disallowed_colbert_input_classes": ["Class 1: PII"],
    "documentation_assistance": {
      "enabled": true,
      "index_path": "./internal_data_handling_docs_rag",
      "max_total_suggestions": 2
    }
  }
}
```

## Endpoints

* `POST /service/validate` — main policy-driven validation endpoint
* `POST /modernbert/classify` — direct access to I/O Validator model
* `POST /colbert/classify_sensitivity` — direct access to Sensitivity Classifier model
* `POST /rag/query` — query a loaded RAG index (typically the global documentation index)
* `GET /status` — check service and component health

## CLI Examples

(For full CLI options, run `python classify.py [command] --help`)

### Start Server

```bash
# Ensure HF_TOKEN is set in your environment if models need downloading
# PowerShell: $env:HF_TOKEN="your_hf_token_here"
# Bash: export HF_TOKEN="your_hf_token_here"

python classify.py serve \
    --policy-config-path ./path/to/your/policy_config.json \
    --port 8080
```

### Create Example Files & Auto-Build Docs RAG

```bash
# This will create sample files and build a RAG index from the specified docs
# Ensure HF_TOKEN is set if the embedding model isn't cached

python classify.py create-example \
    --output-dir ./my_service_examples \
    --auto-build-docs-rag \
    --docs-url "https://raw.githubusercontent.com/your-org/your-repo/main/your-docs.md" \
    --docs-vlm-model-path /path/to/local/model.gguf \
    --processing-strategy vlm 
```

### Create RAG Index from Custom Corpus

```bash
python classify.py rag index \
  --corpus-path ./path/to/my_corpus.jsonl \
  --index-path ./my_custom_rag_index \
  --embedding-model-id all-mpnet-base-v2
```

## Data Formats

### I/O Validation Training (Illustrative)

```json
{
  "input": "Question: Capital of France?",
  "output_good_sample": "Paris",
  "output_bad_sample": "London"
}
```

### ColBERT Sensitivity Reference Examples (Illustrative)

```json
{"text": "SSN: 123-45-6789", "class_name": "Class 1: PII"}
{"text": "Public company earnings report", "class_name": "Class 4: Public"}
```

## Markdown Intelligence

The service leverages Vision-Language Models (VLMs) for advanced processing of markdown documentation when building RAG indexes. This offers significant advantages over traditional regex or basic text-splitting methods.

**Why VLM Helps for RAG Indexing:**

* **Semantic Chunking**: VLMs understand content and structure to create semantically coherent chunks. Related ideas stay grouped, improving retrieval context.

* **Structure Preservation**: VLMs identify and preserve document structure (headers, lists). This metadata enables targeted search/filtering.

* **Code Block Handling**: VLMs treat code blocks as complete units, preventing awkward splits. Metadata flags code-heavy chunks.

* **Contextual Metadata Extraction**: VLMs extract topics, keywords, audience, and difficulty level, enriching metadata for powerful queries.

* **Enhanced Fallback**: Python-based fallback processor respects headers and code blocks when VLM unavailable.

* **Secondary Chunking**: Large chunks get split with overlap to respect embedding model limits.

**Benefits Over Traditional Processing:**

* **Better Search Relevance**: Semantic chunks create better embeddings and more relevant retrieval.

* **Context-Aware Documentation Assistance**: More precise and relevant help for policy violations.

* **Improved Technical Content Handling**: Proper code block and technical jargon handling.

* **Automatic Content Categorization**: Extracted metadata enables auto-tagging and categorization.

## System Requirements

* Python 3.8+
* RAM: 16–32GB+ for VLMs (especially VLM Markdown Processor). Placeholder models and CPU-only operation require less.
* GPU: Optional but recommended for performance
  * CUDA-enabled GPU for `llama-cpp-python` with `n_gpu_layers > 0`

## Testing

```bash
# Run all test categories
python classify.py test --test-type all --verbose

# Run specific test category
python classify.py test --test-type rag
```

---
