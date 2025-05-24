# Data Classification & Monitoring Service
###### (Policy-Driven, Local-First, etc)

This document describes `classify.py`, a **Data Classification and Monitoring Service** that delivers plug-and-play policy-based validation for any application (read: JSON object).

It orchestrates fine-tuned transformers, GGUF-based vision-language models (VLMs), and smart retrieval pipelines to provide a governance layer for multimodal content including text, image, and video.

## **'Validation Endpoint' @ N-Granularity**

Integrate seamlessly using a single `/service/validate` endpoint that accepts content and policy specs. The service handles classification, policy enforcement, and contextual assistance.

**Highlights:**

* **Drop-in Integration**: One endpoint for all types of validation
* **Multi-Modal Ready**: Supports text, image, video, and (soon) audio
* **Policy-Driven**: Configurable via external JSON
* **Context-Aware Help**: Built-in RAG-driven documentation assistant
* **Enterprise-Focused**: Governance, observability, and compliance ready
* **VLM-Aware Docs**: Enhances markdown for indexing and retrieval

---

## 🛡️ PII Policy Example

> `policy_config.md` for schema and options.

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
      "index_path": "./internal_data_handling_docs_rag",
      "max_total_suggestions": 2
    }
  }
}
```

### Key Policy Components:

- **`colbert_input_sensitivity`**: Enables sensitivity analysis using ColBERT
- **`disallowed_colbert_input_classes`**: Specifically blocks PII content
- **`documentation_assistance`**: Provides contextual help when violations occur

---

## 🔌 API Endpoints

* `POST /service/validate` — **main policy-driven validation endpoint**
* `POST /modernbert/classify` — direct access to I/O Validator model
* `POST /colbert/classify_sensitivity` — direct access to Sensitivity Classifier model
* `POST /rag/query` — query a loaded RAG index (typically the global documentation index)
* `GET /status` — check service and component health

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
      "index_path": "./tool_documentation",
      "max_total_suggestions": 3
    }
  }
}
```

**🔄 How it works:**
1. **Intercept**: API Gateway captures user query and AI's intended response
2. **Validate**: Send both to `/service/validate` with appropriate `api_class`
3. **I/O Check**: `ModernBERTClassifier` verifies output coherence with input
4. **Sensitivity Scan**: `ColBERTReranker` checks for PII/confidential data leakage
5. **Policy Enforcement**: If output contains "Confidential" class content → `REJECT_POLICY_VIOLATION`
6. **Guidance**: `documentation_suggestions` provide remediation steps

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
        """📝 Log compliance violations with helpful context."""
        violations = validation_result.get('violation_reasons', [])
        logger.error(f"🚫 Compliance VIOLATION for doc {doc_id} at stage {stage}: {violations}")
        
        # Log any helpful suggestions
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
        
        # 📥 Stage 1: Ingestion validation
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
            
        # ⚙️ Stage 2: Document transformation
        processed_doc = transform_document(doc)
        
        # 💾 Stage 3: Pre-storage validation
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
    """🔄 Placeholder for document transformation logic."""
    return doc

# 🧪 Example test
if __name__ == "__main__":
    test_docs = [
        {"content": "Q1 revenue was $1.2M with growth in key segments."},
        {"content": "Employee SSN 123-45-6789 and salary $85,000."},  # ⚠️ Contains PII
    ]
    
    # valid_docs, quarantined = process_financial_documents(test_docs)
    # print(f"✅ Processed: {len(valid_docs)}, 🚫 Quarantined: {len(quarantined)}")
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
      "index_path": "./tool_documentation_etl",
      "max_total_suggestions": 2
    }
  }
}
```

**🔄 How it works:**
1. **Stage-Based Validation**: Different pipeline stages use different `api_class` policies
2. **Content Analysis**: `input_text` contains the document/record being processed
3. **Multi-Layer Checks**: Policies verify PII, formatting, size limits, etc.
4. **Compliance Tracking**: Violations trigger logging, quarantine, or alerts
5. **Guided Remediation**: `documentation_suggestions` help data stewards fix issues

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
  -F 'uploaded_image=@./test_image.jpg'
```

```python
# 🐍 Python example for content moderation
import requests

def moderate_user_content(text_content, image_file_path=None):
    """
    Moderate user-generated content for policy compliance.
    
    Args:
        text_content: User's text input
        image_file_path: Optional path to uploaded image
        
    Returns:
        dict: Moderation result with approval status
    """
    
    # 📝 Prepare the JSON payload
    payload = {
        "api_class": "UserGeneratedContentPolicy",
        "input_text": text_content,
        "request_id": f"moderation-{int(time.time())}"
    }
    
    # 🖼️ Add image item if provided
    if image_file_path:
        payload["input_items"] = [{
            "id": "user_uploaded_image",
            "type": "image", 
            "filename_in_form": "user_image"
        }]
    
    try:
        if image_file_path:
            # 📤 Multipart request with image
            files = {'user_image': open(image_file_path, 'rb')}
            data = {'json_payload': json.dumps(payload)}
            
            response = requests.post(
                'http://classify-service:8080/service/validate',
                data=data,
                files=files,
                timeout=45  # Longer timeout for image processing
            )
            files['user_image'].close()
        else:
            # 📤 JSON-only request
            response = requests.post(
                'http://classify-service:8080/service/validate',
                json=payload,
                timeout=30
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
            'approved': False,
            'error': f'Moderation service error: {str(e)}',
            'violations': ['Service unavailable']
        }

# 🎯 Example usage
import json
import time

text = "Hey everyone, here's my personal info: email john@example.com"
image_path = "./user_uploads/suspicious_image.jpg"

moderation_result = moderate_user_content(text, image_path)

if moderation_result['approved']:
    print("✅ Content approved for publication")
else:
    print(f"❌ Content blocked: {moderation_result['violations']}")
    if 'suggestions' in moderation_result:
        print(f"💡 Guidelines: {moderation_result['suggestions']}")
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
      "index_path": "./community_guidelines_rag",
      "max_total_suggestions": 3
    }
  }
}
```

**🔄 How it works:**
1. **Text Analysis**: User's text checked for PII, hate speech, violent content
2. **Image Processing**: VLM analyzes uploaded images for policy violations
3. **Multi-Modal Validation**: VLM description text also checked for violations
4. **Keyword Filtering**: Blocked keywords provide additional safeguards
5. **Contextual Help**: Violations trigger community guideline suggestions

</details>

---

## 🚀 Setup

On first run:

* Creates venv
* Installs dependencies (Transformers, Flask, etc.)
* Re-launches in venv

```bash
python classify.py --help
```

### Start Server

```bash
# 🔑 Ensure HF_TOKEN is set in your environment if models need downloading
# PowerShell: $env:HF_TOKEN="your_hf_token_here"
# Bash: export HF_TOKEN="your_hf_token_here"

python classify.py serve \
    --policy-config-path ./path/to/your/policy_config.json \
    --port 8080
```

### Create Example Files & Auto-Build Docs RAG

```bash
# 📚 This will create sample files and build a RAG index from the specified docs
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

---

## 📊 Data Formats

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

---

## 🧠 Markdown Intelligence

The service leverages Vision-Language Models (VLMs) for advanced processing of markdown documentation when building RAG indexes. This offers significant advantages over traditional regex or basic text-splitting methods.

**Why VLM Helps for RAG Indexing:**

* **🎯 Semantic Chunking**: VLMs understand content and structure to create semantically coherent chunks. Related ideas stay grouped, improving retrieval context.

* **🏗️ Structure Preservation**: VLMs identify and preserve document structure (headers, lists). This metadata enables targeted search/filtering.

* **💻 Code Block Handling**: VLMs treat code blocks as complete units, preventing awkward splits. Metadata flags code-heavy chunks.

* **🏷️ Contextual Metadata Extraction**: VLMs extract topics, keywords, audience, and difficulty level, enriching metadata for powerful queries.

* **🔄 Enhanced Fallback**: Python-based fallback processor respects headers and code blocks when VLM unavailable.

* **✂️ Secondary Chunking**: Large chunks get split with overlap to respect embedding model limits.

**Benefits Over Traditional Processing:**

* **🔍 Better Search Relevance**: Semantic chunks create better embeddings and more relevant retrieval.

* **💡 Context-Aware Documentation Assistance**: More precise and relevant help for policy violations.

* **⚙️ Improved Technical Content Handling**: Proper code block and technical jargon handling.

* **🏷️ Automatic Content Categorization**: Extracted metadata enables auto-tagging and categorization.

---

## 💻 System Requirements

* **Python 3.8+**
* **RAM**: 16–32GB+ for VLMs (especially VLM Markdown Processor). Placeholder models and CPU-only operation require less.
* **GPU**: Optional but recommended for performance
  * CUDA-enabled GPU for `llama-cpp-python` with `n_gpu_layers > 0`

---

## 🧪 Testing

```bash
# Run all test categories
python classify.py test --test-type all --verbose

# Run specific test category
python classify.py test --test-type rag
```

---
