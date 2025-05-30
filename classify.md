
# Data Classification & Monitoring Service
###### (Policy-Driven, VLM-Enhanced, Local-First)

This document describes `classify.py`, a **Data Classification and Monitoring Service** that delivers plug-and-play policy-based validation for any application (read: JSON object).

It orchestrates fine-tuned transformers, GGUF-based vision-language models (VLMs), and smart retrieval pipelines to provide a governance layer for multimodal content including text, image, and video.

## **'N-JSON(s)' @ N-Granularity**

Integrate seamlessly using a single `/service/validate` endpoint that accepts content and policy specs. 

The service handles classification, policy enforcement, and contextual assistance.

**Highlights:**

* **Drop-in Integration**: Single-Endpoint Validation
* **Natively Multi-Modal**: via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for GGUF models
* **Policy-Driven**: Configurable via external JSON
* **VLM-Enhanced Policy Generation**: Bootstrap policies from markdown requirements
* **Context-Aware Help**: Self-Documenting RAG for policy violations

---


### 🧠 1. VLM-Powered Policy Generation

The service can utilize a Vision-Language Model (VLM) to interpret a markdown document describing service requirements, data handling obligations, or API guidelines, and then **generate a draft `ClassificationAPI` policy JSON object**. This kickstarts your policy configuration.

```bash
# Generate a policy JSON from your requirements markdown
python classify.py generate-policy-from-markdown \
    --markdown-file ./path/to/your/requirements.md \
    --vlm-model-path "your-gguf-model-repo/model.gguf" \
    --output-policy-file ./generated_policy.json \
    --policy-name MyNewGeneratedPolicy
```
*(This generated policy can then be added to your main `policy_config.json`)*

### 📚 2. VLM-Driven RAG Indexing for Documentation

VLMs enable advanced processing of markdown documentation when building RAG indexes, enhancing the "Documentation Assistance" feature.

*   **🎯 Semantic Chunking**: VLMs understand content; related ideas stay grouped, improving retrieval context.
*   **🏗️ Structure Preservation**: VLMs identify and preserve document metadata enabling targeted search/filtering.
*   **💻 Code Block Handling**: VLMs treat code blocks as complete units, preventing awkward splits. Metadata flags code-heavy chunks.
*   **🏷️ Contextual Metadata**: VLMs can extract topics, keywords, and more, enriching metadata for powerful queries (future enhancements).
*   **✂️ Secondary Chunking**: Large chunks (from VLM or fallback) get split with overlap to respect embedding model limits.

**Benefits Over Traditional RAG Processing:**

*   **🔍 Better Search Relevance**: Semantic chunks create better embeddings and more relevant retrieval for documentation assistance.
*   **💡 Context-Aware Documentation Assistance**: More precise and relevant help for policy violations.

---

## 🛡️ PII Policy Example

> See `enhanced_policy_config.json` (created by `create-example`) for more schema details and options.

The following demonstrates a **strict PII detection policy** that focuses on identifying and rejecting personally identifiable information in input text:

```json
{
  "StrictPIIPolicy": {
    "description": "Focuses on identifying and rejecting PII in input text.",
    "modernbert_io_validation": false,
    "colbert_input_sensitivity": true,
    "disallowed_colbert_input_classes": ["Class 1: PII"],
    "documentation_assistance": {
      "enabled": true,
      "index_path": "./tool_examples/internal_data_handling_docs_rag", // Path to a RAG index
      "max_total_suggestions": 2
    }
  }
}
```

### Key Policy Components:

- **`colbert_input_sensitivity`**: Enables sensitivity analysis.
- **`disallowed_colbert_input_classes`**: Specifically blocks PII content.
- **`documentation_assistance`**: Provides contextual help when violations occur, using the RAG index specified in `index_path`.

---

## 🔌 API Endpoints

* `POST /service/validate` — **Main policy-driven validation endpoint.**
* `POST /modernbert/classify` — Direct access to I/O Validator model.
* `POST /colbert/classify_sensitivity` — Direct access to Sensitivity Classifier model.
* `POST /rag/query` — Query a loaded RAG index (typically the global documentation index if configured).
* `GET /status` — Check service and component health.

---

## 🔗 Integration Examples

### 1. API Gateway Integration

**Scenario**: Pre-validate AI responses before returning them to users to prevent data leaks.

<details>
<summary><strong>📋 Click to expand API Gateway example</strong></summary>

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

        // ✅ Validation passed
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

// 🎯 Example usage
const userQuery = "Tell me about project X";
const aiResponse = "Project X is a super secret initiative, details are classified.";

const validationResult = await validateAIResponse(userQuery, aiResponse);

if (!validationResult.success) {
    // ❌ Handle policy violation or service error
    console.log('Cannot return response:', validationResult.error);
    if (validationResult.helpfulSuggestions) {
        console.log('💡 Helpful information:', validationResult.helpfulSuggestions);
    }
} else {
    // ✅ Safe to return the AI response
    console.log('Approved response:', validationResult.response);
}
```

**🔧 Required Policy Configuration** (`UserInputHandlingPolicy` in `policy_config.json`):

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
      "index_path": "./tool_examples/tool_documentation", // Ensure this RAG index exists
      "max_total_suggestions": 3
    }
  }
}
```

**🔄 How it works:**
1. **Intercept**: API Gateway captures user query and AI's intended response.
2. **Validate**: Send both to `/service/validate` with appropriate `api_class`.
3. **I/O Check**: `ModernBERTClassifier` verifies output coherence with input.
4. **Sensitivity Scan**: `ColBERTReranker` checks for PII/confidential data leakage.
5. **Policy Enforcement**: If output contains "Confidential" class content → `REJECT_POLICY_VIOLATION`.
6. **Guidance**: `documentation_suggestions` provide remediation steps from the specified RAG index.

</details>

---

### 2. Data Pipeline Integration

**Scenario**: Validate documents during processing to ensure compliance throughout the data pipeline.

<details>
<summary><strong>📋 Click to expand ETL Pipeline example</strong></summary>

```python
# ETL pipeline for processing documents, with classification monitoring
import requests
import uuid
import logging
import json # Added for json.dumps in example
import time # Added for request_id in example

logger = logging.getLogger(__name__) # Ensure logger is defined for example

class DocumentValidator:
    def __init__(self, service_url='http://classify-service:8080'):
        self.service_url = service_url
        self.validate_endpoint = f"{service_url}/service/validate"
    
    def validate_document(self, doc_content, doc_id, pipeline_stage, metadata=None):
        """
        Validate document content against policy for a specific pipeline stage.
        """
        validation_payload = {
            'api_class': f'Pipeline_{pipeline_stage}_IntegrityPolicy',
            'input_text': doc_content,
            'request_id': f"etl-{pipeline_stage}-{doc_id}",
        }
        
        if metadata:
            validation_payload['metadata_fields'] = metadata # Note: API expects 'metadata' not 'metadata_fields' at top level
        
        try:
            response = requests.post(
                self.validate_endpoint, 
                json=validation_payload,
                timeout=30
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
        """📝 Log compliance violations with helpful context."""
        violations = validation_result.get('violation_reasons', [])
        logger.error(f"🚫 Compliance VIOLATION for doc {doc_id} at stage {stage}: {violations}")
        
        suggestions = validation_result.get('documentation_suggestions', {}).get('suggestions', [])
        if suggestions:
            logger.info(f"💡 Remediation suggestions for {doc_id}: {[s['title'] for s in suggestions]}")

# 🏭 Example usage in an ETL pipeline
def process_financial_documents(documents):
    validator = DocumentValidator()
    processed_docs = []
    quarantined_docs = []
    
    for doc in documents:
        doc_id = str(uuid.uuid4())
        
        is_valid, result = validator.validate_document(
            doc_content=doc['content'],
            doc_id=doc_id,
            pipeline_stage='Ingestion',
            metadata={'document_source': 'financial_reports_q1_2024', 'processing_step': 'pii_scan'}
        )
        
        if not is_valid:
            quarantined_docs.append({'doc_id': doc_id, 'stage_failed': 'Ingestion', 'violations': result.get('violation_reasons', [])})
            continue
            
        processed_doc = doc # Placeholder for transform_document
        
        is_valid, result = validator.validate_document(doc_content=processed_doc['content'], doc_id=doc_id, pipeline_stage='PreStorage')
        
        if is_valid:
            processed_docs.append(processed_doc)
        else:
            quarantined_docs.append({'doc_id': doc_id, 'stage_failed': 'PreStorage', 'violations': result.get('violation_reasons', [])})
    
    return processed_docs, quarantined_docs

# 🧪 Example test
if __name__ == "__main__":
    # Configure logger for example run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    test_docs = [
        {"content": "Q1 revenue was $1.2M with growth in key segments."},
        {"content": "Employee SSN 123-45-6789 and salary $85,000."},
    ]
    
    # valid_docs, quarantined = process_financial_documents(test_docs)
    # print(f"✅ Processed: {len(valid_docs)}, 🚫 Quarantined: {len(quarantined)}")
    # print(f"Quarantined details: {quarantined}") 
```

**🔧 Required Policy Configuration** (`Pipeline_Ingestion_IntegrityPolicy` in `policy_config.json`):

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
      "index_path": "./tool_examples/tool_documentation_etl",
      "max_total_suggestions": 2
    }
  }
}
```

**🔄 How it works:**
1. **Stage-Based Validation**: Different pipeline stages use different `api_class` policies.
2. **Content Analysis**: `input_text` contains the document/record being processed.
3. **Multi-Layer Checks**: Policies verify PII, formatting, size limits, etc.
4. **Compliance Tracking**: Violations trigger logging, quarantine, or alerts.
5. **Guided Remediation**: `documentation_suggestions` help data stewards fix issues.

</details>

---

### 3. Multimodal Moderation 

**Scenario**: Moderate user-generated content including text and images for policy compliance.

<details>
<summary><strong>📋 Click to expand Content Moderation example</strong></summary>

```bash
# 🖼️ Example: Multipart form data with image upload
curl -X POST http://classify-service:8080/service/validate \
  -F 'json_payload={"api_class":"UserGeneratedContentPolicy","input_text":"Check out this cool image I found!","input_items":[{"id":"user_image_1","type":"image","filename_in_form":"uploaded_image"}]}' \
  -F 'uploaded_image=@./path/to/your/test_image.jpg'
```

```python
# 🐍 Python example for content moderation
import requests
import json # Ensure json is imported for payload
import time # Ensure time is imported for request_id

def moderate_user_content(text_content, image_file_path=None):
    """
    Moderate user-generated content for policy compliance.
    """
    payload = {
        "api_class": "UserGeneratedContentPolicy",
        "input_text": text_content,
        "request_id": f"moderation-{int(time.time())}"
    }
    
    if image_file_path:
        payload["input_items"] = [{
            "id": "user_uploaded_image", "type": "image", "filename_in_form": "user_image"
        }]
    
    try:
        if image_file_path:
            with open(image_file_path, 'rb') as f_image:
                files = {'user_image': f_image}
                data = {'json_payload': json.dumps(payload)}
                response = requests.post(
                    'http://classify-service:8080/service/validate',
                    data=data, files=files, timeout=45
                )
        else:
            response = requests.post(
                'http://classify-service:8080/service/validate', json=payload, timeout=30
            )
        
        response.raise_for_status()
        result = response.json()
        
        return {
            'approved': result.get('overall_status') == 'PASS',
            'violations': result.get('violation_reasons', []),
            'suggestions': result.get('documentation_suggestions', {}),
            'processing_details': result.get('processing_details', {})
        }
        
    except requests.exceptions.RequestException as e:
        return {
            'approved': False, 'error': f'Moderation service error: {str(e)}',
            'violations': ['Service unavailable']
        }

# 🎯 Example usage
# text = "Hey everyone, here's my personal info: email john@example.com"
# image_path = "./user_uploads/suspicious_image.jpg" # Make sure this path is valid for testing
# moderation_result = moderate_user_content(text, image_path)
# if moderation_result['approved']:
#     print("✅ Content approved for publication")
# else:
#     print(f"❌ Content blocked: {moderation_result['violations']}")
#     if 'suggestions' in moderation_result and moderation_result['suggestions'].get('suggestions'):
#         print(f"💡 Guidelines: {moderation_result['suggestions']['suggestions']}")
```

**🔧 Required Policy Configuration** (`UserGeneratedContentPolicy` in `policy_config.json`):

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
      "index_path": "./tool_examples/community_guidelines_rag",
      "max_total_suggestions": 3
    }
  }
}
```

**🔄 How it works:**
1. **Text Analysis**: User's text checked for PII, hate speech, violent content.
2. **Image Processing**: VLM analyzes uploaded images for policy violations.
3. **Multi-Modal Validation**: VLM description text also checked for violations.
4. **Keyword Filtering**: Blocked keywords provide additional safeguards.
5. **Contextual Help**: Violations trigger community guideline suggestions from the RAG index.

</details>

---

## 🚀 Setup & Tooling Commands

On first run, the script will:
* Create a virtual environment (`.venv_classifier_service_tool`).
* Install all required dependencies (Transformers, Flask, llama-cpp-python, etc.).
* Re-launch itself using the Python interpreter from the virtual environment.

```bash
python classify.py --help
```

### 1. Start API Server

```bash
# 🔑 Ensure HF_TOKEN is set in your environment if models need downloading
# PowerShell: $env:HF_TOKEN="your_hf_token_here"
# Bash: export HF_TOKEN="your_hf_token_here"

python classify.py serve \
    --policy-config-path ./path/to/your/policy_config.json \
    --port 8080 \
    --global-rag-retriever-index-path ./tool_examples/tool_documentation # Optional: for /rag/query
```

### 2. Create Example Files & Documentation RAG Index

This command generates sample documentation, a policy configuration (`enhanced_policy_config.json`), and can automatically build a RAG index from specified documentation sources. This RAG index can then be referenced in your policies for `documentation_assistance`.

```bash
# 📚 This will create sample files and build a RAG index from the specified docs
# Ensure HF_TOKEN is set if the embedding model isn't cached

python classify.py create-example \
    --output-dir ./my_service_examples \
    --auto-build-docs-rag \
    --docs-url "https://raw.githubusercontent.com/your-org/your-repo/main/your-docs.md" \
    --docs-rag-index-name "my_docs_rag" \
    # Optional: For VLM-based chunking of documentation
    # --docs-vlm-model-path "/path/to/local/doc_processing_model.gguf" \
    # --processing-strategy vlm 
```
*The `index_path` in your policy's `documentation_assistance` section should point to the RAG index created here (e.g., `./my_service_examples/my_docs_rag`).*

### 3. Generate API Policy from Markdown (VLM-driven)

Use a VLM to draft a new policy definition by analyzing a markdown document that outlines requirements or guidelines.

```bash
python classify.py generate-policy-from-markdown \
    --markdown-file ./path/to/your/service_requirements.md \
    --vlm-model-path "your-gguf-model-repo/policy_generation_model.gguf" \
    # Optional: --gguf-filename "specific_file.gguf" (if vlm-model-path is repo_id)
    --output-policy-file ./new_generated_policy.json \
    --policy-name MyServicePolicyFromMarkdown
```
*The output JSON will contain `{"MyServicePolicyFromMarkdown": { ...generated policy... }}`. You can then integrate this into your main `policy_config.json`.*

### 4. Create Custom RAG Index

Build a RAG index from any JSONL corpus for use in `documentation_assistance` or general querying.

```bash
python classify.py rag index \
  --corpus-path ./path/to/my_corpus.jsonl \
  --index-path ./my_custom_rag_index \
  --embedding-model-id all-mpnet-base-v2 \
  --metadata-fields category topic_id # Optional: list of metadata fields in your JSONL
```

### 5. Index Python Codebase for RAG

Create a RAG index directly from a Python script to enable querying its structure (functions, classes).

```bash
python classify.py index-codebase \
    --code-file-path ./classify.py \
    --index-path ./codebase_index_for_classify_py \
    --code-chunk-strategy functions
```

---

## 📊 Data Formats

### I/O Validation Format (for training ModernBERT - conceptual)

```json
{
  "input": "Question: Capital of France?",
  "output_good_sample": "Paris",
  "output_bad_sample": "London"
}
```

### Sensitivity Classification Format (for training ColBERT - conceptual)

```json
{"text": "SSN: 123-45-6789", "class_name": "Class 1: PII"}
{"text": "Public company earnings report", "class_name": "Class 4: Public"}
```

---

## 💻 System Requirements

* **Python 3.8+**
* **RAM**: 16–32GB+ recommended for optimal performance with VLMs (especially for VLM Markdown Processor and Policy Generation). Placeholder models and CPU-only operation require less.
* **GPU**: CUDA-enabled GPU (optional but recommended) for hardware acceleration with `llama-cpp-python` (set `n_gpu_layers > 0`).

---

## 🧪 Testing

```bash
# Run all test categories with verbose output
python classify.py test --test-type all --verbose

# Run a specific test category (e.g., RAG functionality)
python classify.py test --test-type rag --verbose
```
---
## 🏙️ Architecture
```

 classify.py - High-Level Architecture

    +---------------------------+      +------------------------------+      +-----------------------------+
    |   Developer / Operator    |----->|   Raw Markdown (for Policy)  |<-----| VLM Model (GGUF for Policy) |
    |      (using CLI)          |      +------------------------------+      +-----------------------------+
    +---------------------------+                 |
                 |                                |
                 | (1a. `generate-policy-...`)   v
                 |                      +------------------------------+
                 |                      | VLM Policy Generation Tool   |
                 |                      | (MarkdownReformatterVLM)     |
                 |                      +------------------------------+
                 |                                |
                 |                                v
                 |                      +------------------------------+
                 +--------------------->| Generated Policy Config File |
                                        |      (api_policy.json)       |
                                        +--------------+---------------+
                                                       | (Loaded at API start)
                                                       |
    +---------------------------+      +------------------------------+      +-----------------------------+
    |   Developer / Operator    |----->| Raw Content (Docs Markdown,  |<-----| VLM/Embedding Models (GGUF, |
    |      (using CLI)          |      |       Python Code Files)     |      |  SentenceTransformers HF)   |
    +---------------------------+      +------------------------------+      +-----------------------------+
                 |                                |
                 | (1b. `index-docs`,             |
                 |      `index-codebase`,         v
                 |      `rag index`)   +------------------------------+
                 |                      | RAG Index Build Process      |
                 |                      | (VLM/Fallback/AST Chunking,  |
                 |                      |      Embedding)              |
                 |                      +------------------------------+
                 |                                |
                 |                                v
                 |                      +------------------------------+      +-----------------------------+
                 +--------------------->| RAG Index Artifacts Store    |<-----| Embedding Models (HF)       |
                                        | (docs.jsonl, embeddings.npy, |      | (for RAG query encoding)    |
                                        |  config.json)                |      +-----------------------------+
                                        +--------------+---------------+
                                                       | (Loaded at API start & on-demand by policy)
                                                       |
    ---------------------------------- API SERVER RUNTIME -------------------------------------------
                                                       |
                                                       v
    +---------------------------+      +------------------------------------------------------------------+
    |      API Client           |<---->|                      Classification API Server                   |
    | (e.g., Web App, Script)   |      |                  (Flask / Waitress + Custom Logic)               |
    +---------------------------+      |                                                                  |
                                       |  +-------------------------+  +--------------------------------+ |
                                       |  | Policy Loader & Cache   |  | RAG Retriever Loader & Cache   | |
                                       |  +-------------------------+  +--------------------------------+ |
                                       |              |                               |                   |
                                       |              v                               v                   |
                                       |  +------------------------------------------------------------+  |
                                       |  |                       Validation Engine                     | |
                                       |  |    (Applies rules from loaded Policy for current API Class) | |
                                       |  +-----------------------+------------------+-----------------+  |
                                       |              |           |                  |                    |
                                       |  (Policy     |           | (Data for VLM)   | (Data for RAG)     |
                                       |   Checks)    v           v                  v                    |
                                       |  +----------------+  +----------------+  +----------------+      |
                                       |  | BERT/ColBERT   |  | VLM for Items  |  | RAG Querier    |      |
                                       |  | Model Services |  | Model Service  |  | (Doc Assist)   |      |
                                       |  +-------+--------+  +-------+--------+  +-------+--------+      |
                                       |          | (HF Models)         | (GGUF Model)         | (Uses RAG Index)
                                       |          v                     v                      v          |
                                       |  +----------------+  +----------------+  +----------------+      |
                                       |  | HuggingFace    |  | GGUF (llama.cpp|  | RAG Index      |      |
                                       |  | Model Artifacts|  | based) Models  |  | Artifacts      |      |
                                       |  +----------------+  +----------------+  +----------------+      |
                                       +------------------------------------------------------------------+

Key Data/Artifact Flows:

(1a) VLM Policy Generation:
     Raw Markdown (Policy Desc.) + VLM Model --> [VLM Policy Gen Tool] --> Generated Policy Config File

(1b) RAG Index Generation:
     Raw Content (Docs/Code) + VLM/Embedding Models --> [RAG Index Build] --> RAG Index Artifacts

(2) API Initialization:
     Generated Policy Config File --> [API Server/Policy Loader]
     RAG Index Artifacts --> [API Server/RAG Retriever Loader]

(3) API Request Validation:
     API Client Request --> [API Server/Validation Engine]
        --> Applies rules from loaded Policy
        --> Invokes BERT/ColBERT Model Services (using HF Models)
        --> Invokes VLM for Items Service (using GGUF Model)
        --> (If violations & configured) Invokes RAG Querier (using RAG Index) for Documentation Assistance
        --> API Client Response
