# AI Security Architecture

## via AI Passports & GFCI-plugs 🛂

AI security is fundamentally about observability with context. We track inputs, outputs, and model behavior patterns, then compare against established baselines to detect anomalies — logged to a per-integration API with per-deploy specifics and end-to-end configuration details.

<details>
<summary>🖼️ End-to-End Architecture Diagram</summary>

![End-to-End Architecture](end-to-end.jpg)
</details>

---

## Core Components

### 1. Input/Output Logging
- Track prompts, responses, and processing patterns
- Establish baselines of normal vs. abnormal behavior
- Apply [data classification](https://github.com/rabbidave/LatentSpace.Tools/blob/main/classify.md) (PICR,1/2/3) for [all system actions](https://github.com/rabbidave/LatentSpace.Tools/blob/main/MCPHandshake.md)

### 2. Real-Time Monitoring
- Compare current behavior against established patterns
- Alert on deviations that exceed thresholds
- Use Lightweight Heuristics w/ Stepwise Validation

### 3. Verification & Testing
- Validate models against known attack patterns
- Implement continuous security testing
- Maintain audit trails for compliance

### 4. System Breadcrumb

The **System Breadcrumb** is a SHA-256 (or contemporaneous equivalent) hash of the initial `use_case_registry.json`, used as a shared secret for deployment identity verification and auditability. It provides immutable context anchoring for logs and requests.

**Artifacts:**
- `use_case_registry.json` – Source of truth for the AI Passport
- `system_breadcrumb.txt` – Hash artifact derived from the above
- `tool_invocation.log` – Operational log entries referencing the breadcrumb

<details>
<summary><strong>🔐 Python: Generate System Breadcrumb</strong></summary>

```python
# generate_breadcrumb.py
import json, hashlib

def generate_system_breadcrumb(json_path: str) -> str:
    with open(json_path, 'r') as f:
        data = json.load(f)
    encoded = json.dumps(data, sort_keys=True).encode('utf-8')
    breadcrumb = hashlib.sha256(encoded).hexdigest()

    with open('system_breadcrumb.txt', 'w') as out:
        out.write(breadcrumb)

    return breadcrumb
```

</details>

---

## Role-Based Implementation

Each team has clear responsibilities:

- **Enterprise**: Set security standards, define classifications
- **IT/Ops**: Configure runtime environments, validation parameters
- **Application Teams**: Implement controls, monitor business metrics

![Personas and Roles](personas.jpg)

---

<details>
<summary>🧭 C4 Architecture Views</summary>

| View       | Description              | Link                                       |
|------------|--------------------------|--------------------------------------------|
| Context    | High-level system context | ![Context](C4%20-%20Context.jpg)           |
| Container  | Deployment components     | ![Container](C4%20-%20Container.png)       |
| Component  | Key functional elements   | ![Component](C4%20-%20Component.png)       |
| Code       | Implementation details    | ![Code](C4%20-%20Code.png)                 |
| Personas   | User/stakeholder roles    | ![Personas](C4%20-%20Personas.png)         |
</details>

---

## Implementation Resources

- [API Schema Definition](schema.json)
- [Logging Implementation](LoggingAPI.py)
- [Tool Classification](MCPHandshake.md)
- `use_case_registry.json` – Defines the deployment’s purpose and scope
- `system_breadcrumb.txt` – Canonical SHA-256 fingerprint
- `tool_invocation.log` – Signed and referenceable activity logs

---

<details>
<summary>⚙️ AI Control Plane API Documentation</summary>

### Overview

This API provides a system of record for configuration changes across AI deployments. It supports:
- Schema evolution
- Role-based access control
- Audit trail of all changes
- Backwards compatibility

### Authentication

All requests require a bearer token specific to your role:
- Enterprise Infrastructure: `enterprise-token`
- LOB IT: `lob-token`
- Application Teams: `app-token`

Refer to the `get_current_role` function in [`LoggingAPI.py`](LoggingAPI.py) for token-to-role mapping.

### 1. Enterprise Infrastructure Path (`POST /api/v1/config/schema`)

Enterprise teams manage core infrastructure configurations. This path allows updating the enterprise schema, defining standards like data classification and alerting.

**Adding GPU Configuration Example:**
This `curl` command demonstrates adding a GPU configuration to the enterprise schema.
```bash
curl -X POST http://localhost:8000/api/v1/config/schema \
  -H "Authorization: Bearer enterprise-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-inference",
    "targetMetric": "gpu_utilization",
    "dataClassification": "Restricted",
    "quickAlertHeuristic": {
      "threshold": 0.85,
      "window": "5m"
    },
    "gpu_config": {
      "type": "A100",
      "memory": "80GB"
    },
    "reason": "Production scale-up for high-throughput inference"
  }'
```

<details>
<summary><strong>📋 Example Response</strong></summary>

```json
{
  "status": "success",
  "message": "Enterprise schema updated",
  "updated_schema": {
    "type": "object",
    "required": ["name", "targetMetric", "dataClassification", "quickAlertHeuristic"],
    "properties": {
      "name": {"type": "string"},
      "targetMetric": {"type": "string"},
      "dataClassification": {
        "type": "string",
        "enum": ["Public", "Internal", "Confidential", "Restricted"]
      },
      "quickAlertHeuristic": {
        "type": "object",
        "required": ["threshold", "window"],
        "properties": {
          "threshold": {"type": "number"},
          "window": {"type": "string"}
        }
      },
      "author": {"type": "string"},
      "reason": {"type": "string"},
      "timestamp": {"type": "string", "format": "date-time"},
      "gpu_config": {                  # New field added
        "type": "object",
        "properties": {
          "type": {"type": "string"},
          "memory": {"type": "string"}
        }
      }
    },
    "additionalProperties": true
  }
}
```
</details>
*Note: The `author` and `timestamp` fields are automatically managed by the API. The response shows the updated schema structure, including the newly added `gpu_config`.*

### 2. LOB IT Path (`POST /api/v1/runtime/schema`)

LOB IT teams manage runtime configurations and model validation parameters.

**Adding Operational Metric Example:**
```bash
curl -X POST http://localhost:8000/api/v1/runtime/schema \
  -H "Authorization: Bearer lob-token" \
  -H "Content-Type: application/json" \
  -d '{
    "modelVersion": "v2.1.0",
    "validationParameters": {
      "minBatchSize": 32,
      "maxLatencyMs": 100
    },
    "operational_metric": "inference_throughput",
    "reason": "Adding throughput monitoring for batch processing"
  }'
```
*(Successful responses will be similar to the enterprise update, showing "Runtime schema updated" and the `updated_schema` for "runtime" including the new `operational_metric` field.)*

### 3. Application Team Path (`POST /api/v1/integrations/schema`)

Application teams manage custom integrations and business metrics.

**Adding Custom Metrics Example:**
```bash
curl -X POST http://localhost:8000/api/v1/integrations/schema \
  -H "Authorization: Bearer app-token" \
  -H "Content-Type: application/json" \
  -d '{
    "customThresholds": {
      "accuracy": 0.95,
      "latency_p99": 250
    },
    "businessMetric": "revenue",
    "callbackUrl": "https://app-endpoint/callback",
    "reason": "Adding latency monitoring for SLA compliance"
  }'
```
*(Successful responses will show "Integration schema updated" and the `updated_schema` for "integration" including the new `callbackUrl` field.)*

### 4. Audit Trail (`GET /api/v1/audit/{schema_type}`)

View the history of changes for any schema type (`enterprise`, `runtime`, or `integration`). This is vital for compliance and tracking configuration evolution.

**Example Request:**
```bash
curl -X GET http://localhost:8000/api/v1/audit/enterprise \
  -H "Authorization: Bearer enterprise-token"```

<details>
<summary><strong>📋 Example Response</strong></summary>

```json
{
  "changes": [
    {
      "timestamp": "2024-02-13T14:30:00Z",
      "schema_type": "enterprise",
      "author": "enterprise-enterprise",
      "change": {
        "name": "gpu-inference",
        "targetMetric": "gpu_utilization",
        "dataClassification": "Restricted",
        "quickAlertHeuristic": {
          "threshold": 0.85,
          "window": "5m"
        },
        "gpu_config": {
          "type": "A100",
          "memory": "80GB"
        },
        "reason": "Production scale-up for high-throughput inference",
        "author": "enterprise-enterprise",
        "timestamp": "2024-02-13T14:30:00Z"
      }
    }
    // ... other historical changes for 'enterprise' schema
  ]
}
```
</details>

<details>
<summary>📜 Schema Evolution Rules</summary>

1.  **Required Fields** (must be present in the request body for the respective schema type, unless already defined and not being changed):
    *   Enterprise: `name`, `targetMetric`, `dataClassification`, `quickAlertHeuristic`
    *   LOB IT: `modelVersion`, `validationParameters`
    *   App Teams: `customThresholds`, `businessMetric`
2.  **All Changes Require**:
    *   `reason`: A string explaining the purpose of the change (optional for GET).
    *   Appropriate role authorization (validated via token).
    *   The request body must validate against the current base schema structure for required fields and their types.
3.  **New Fields**:
    *   If `additionalProperties` is true in the base schema (which it is for all defined schemas), new fields can be added.
    *   The API infers the type of new fields (e.g., string, number, object) based on the provided value.
    *   These new fields are then incorporated into the schema definition for future validations.
    *   New fields cannot override the type or structure of existing, defined fields in `base_schemas`.
</details>

### Error Handling

Common error responses include:
<details>
<summary><strong>📋 Access Denied (403 Forbidden)</strong></summary>

```json
{
  "detail": "Enterprise access required"
}
```
</details>
<details>
<summary><strong>📋 Invalid Token (401 Unauthorized)</strong></summary>

```json
{
  "detail": "Invalid token"
}```
</details>
<details>
<summary><strong>📋 Validation Error (400 Bad Request)</strong></summary>

```json
{
  "detail": "[ErrorDetail(message=\"'reason' is a required property\", ...)]"
}
```
*(Actual jsonschema error message might be more verbose)*
</details>

## Business Benefits

- **Compliance**: Meet regulatory requirements with audit trails
- **Security**: Detect and mitigate novel AI-specific threats
- **Operational**: Faster incident response with clear accountability

---

## Getting Started

1. Define your Tools & Sensitivity (`tool_invocation.log`)
2. Create a use case registry entry (`use_case_registry.json`)
3. Generate your system breadcrumb **Artifacts:** (`system_breadcrumb.txt`)
4. Attach breadcrumb ID to all Sensitive Tool Invocation logs; Log to the API

---

## Additional Resources

- [Original Architecture Overview](a16zSummary.png)
- [Detailed Architecture](a16zDetail.png)
- [Annotated Architecture](a16zDetailAnnotated.png)
- [LLMs and Observability (Video)](LLMs%20x%20Observability.mp4)

---
```
