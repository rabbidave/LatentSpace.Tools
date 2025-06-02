# AI Security Architecture

## via AI Passports

AI security is fundamentally about observability with context. We track inputs, outputs, and model behavior patterns, then compare against established baselines to detect anomalies — logged to a per-integration API with per-deploy specifics and end-to-end configuration details (Yay! SR11-7v2!! Go Governance!!).

![End-to-End Architecture](end-to-end.jpg)

---

## Core Components

### 1. Input/Output Logging
- Track prompts, responses, and processing patterns
- Establish baselines of normal vs. abnormal behavior
- Apply data classification (PICR) for appropriate handling

### 2. Real-Time Monitoring
- Compare current behavior against established patterns
- Alert on deviations that exceed thresholds
- Apply lightweight checks broadly, heavier verification selectively

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

# Example usage:
# python generate_breadcrumb.py
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

## 🧭 C4 Architecture Views

| View       | Description              | Link                                       |
|------------|--------------------------|--------------------------------------------|
| Context    | High-level system context | ![Context](C4%20-%20Context.jpg)           |
| Container  | Deployment components     | ![Container](C4%20-%20Container.png)       |
| Component  | Key functional elements   | ![Component](C4%20-%20Component.png)       |
| Code       | Implementation details    | ![Code](C4%20-%20Code.png)                 |
| Personas   | User/stakeholder roles    | ![Personas](C4%20-%20Personas.png)         |

---

## Implementation Resources

- [API Schema Definition](schema.json)  
- [Logging Implementation](LoggingAPI.py)  
- [Validation Documentation](validation-docs.md)  
- `use_case_registry.json` – Defines the deployment’s purpose and scope  
- `system_breadcrumb.txt` – Canonical SHA-256 fingerprint  
- `tool_invocation.log` – Signed and referenceable activity logs  

---

## Business Benefits

- **Compliance**: Meet regulatory requirements with audit trails  
- **Security**: Detect and mitigate novel AI-specific threats  
- **Operational**: Faster incident response with clear accountability  

---

## Getting Started

1. Define your data classification scheme using PICR  
2. Create a use case registry entry (`use_case_registry.json`)  
3. Generate your system breadcrumb  
4. Attach breadcrumb ID to all tool invocation logs  
5. Deploy monitoring, alerting, and validation checks  

---

## Additional Resources

- [Original Architecture Overview](a16zSummary.png)  
- [Detailed Architecture](a16zDetail.png)  
- [Annotated Architecture](a16zDetailAnnotated.png)  
- [LLMs and Observability (Video)](LLMs%20x%20Observability.mp4)  
