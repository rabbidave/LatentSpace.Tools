# Pebble: Async AI Collaboration

This document outlines the architecture and protocol for a framework designed to enable robust, auditable, and stateful collaboration between AI agents, automated systems, and human operators.

The core principle of this system is that:

> **Agents pick up a pebble, verify its crytographic `proof`, then act on and update its `state`.**

This approach, inspired by computational theory on catalytic computing, solves the critical problem of maintaining stateful coherence in distributed, asynchronous workflows. The pebble itself becomes the universal, self-contained state artifact that travels between runtimes.

### Index

<details>
<summary>Expand to see the document structure...</summary>

1. [The Pebble: The Universal State Artifact](#1-the-pebble-the-universal-state-artifact)
2. [The Asynchronous DAG: The Architectural Vision](#2-the-asynchronous-dag-the-architectural-vision)
3. [Canonical Reference: The Self-Executing Instruction Template](#3-canonical-reference-the-self-executing-instruction-template)

</details>

---

### 1. The Pebble: The Universal State Artifact

The fundamental unit of this framework is the **`<pebble>`**. It is not merely a message, but a self-contained, verifiable artifact representing a single node in a computational graph.

A pebble consists of three core components:

* **`<action>`**: A description of the work that was just completed to produce this pebble.
* **`<state>`**: The complete, resulting state of the world after the action was taken.
* **`<proof>`**: A cryptographic hash (e.g., [AI Passport](https://github.com/rabbidave/LatentSpace.Tools/blob/main/AI_Passports.md) breadcrumb) that guarantees the authenticity and integrity of the `state`.

By packaging state and proof together, the pebble becomes a **digital catalyst**, allowing any participant to trust and build upon the work of others without requiring a centralized, monolithic state machine.

### 2. The Asynchronous DAG: The Architectural Vision

The pebbles are not sent from point to point; they are placed on a conceptual **Shared Computational DAG**. This acts as a persistent, auditable log of the entire process. Different actors, each with their own runtime and capabilities, interact with this shared graph as their single source of truth.

```ascii
+-----------------------------------------------------------------------------+
|                      Shared Computational DAG / Pebble Log                  |
| (A logical, persistent, and auditable record of all state transitions)      |
|                                                                             |
| +------------------+     +------------------+     +------------------+      |
| | Pebble_N         | --> | Pebble_N+1       | --> | Pebble_N+2       | -->..|
| | (From Agent)     |     | (From System)    |     | (From Human)     |      |
| +------------------+     +------------------+     +------------------+      |
|                                                                             |
+----------------------------------^-----------------------------+------------+
                                   | (2. Place Pebble_N+1)         | (1. Retrieve Pebble_N)
                                   |                               |
+----------------------------------+-------------------------------v----------+
|        Any Authorized Participant (Agent, System, or Human)                 |
|                                                                             |
|            +------------------------------------------------------+         |
|            |      Internal Workflow: Verify, Act, Formulate       |         |
|            |                                                      |         |
|            |  1. Verify pebble.proof                              |         |
|            |  2. Act on pebble.state (using own tools/logic)      |         |
|            |  3. Formulate new action, state, and proof           |         |
|            |                                                      |         |
|            +------------------------------------------------------+         |
|                                                                             |
+-----------------------------------------------------------------------------+


```

### 3. Canonical Reference: The Self-Executing Instruction Template

This section contains the final, atomic instruction template. It is a self-sufficient artifact designed to bootstrap any agent on how to participate in the Pebble Protocol. It uses a `curl` command as the axiomatic example of an agent placing its own verifiable pebble onto the shared graph, reflecting the MCP Handshake v1.1 schema in a logically coherent use case.

<details>
<summary><strong>Click to Expand:</strong> The Complete Agent Instruction Template</summary>

```markdown
# INSTRUCTION TEMPLATE: The Asynchronous Pebble Protocol

This document defines the complete protocol for an AI agent operating in a distributed, asynchronous environment. This is a self-contained artifact.

## 1. The Core Philosophy: The Pebble as a Universal Artifact

The foundation of your operation is inspired by the work of Ryan Williams, which proved a small, reusable memory space (a **catalyst**) can be as effective as a lot of time. In this protocol, the **pebble is that catalyst.**

It is a self-contained, stateful artifact designed for a world of asynchronous, distributed work. It is not a message for a single system; it is a node to be placed on a shared computational Directed Acyclic Graph (DAG). This means **any agent, system, or human operator**—provided they understand this schema—can pick up a pebble, verify its `proof`, act on its `state`, and then place a new, evolved pebble back onto the graph.

The axiomatic `curl` command below is the physical manifestation of an agent performing this fundamental action: securely submitting its completed work as a verifiable, stateful node for the next participant in the asynchronous chain.

## 2. The Canonical Action: Placing a Pebble on the DAG (MCP v1.1 Schema)

The following example demonstrates a successful, multi-agent workflow where a task is verified and completed without requiring security escalation, according to a simple policy.

**Scenario:** An `IntakeAgent-001` receives a request for an **Automated Expense Categorization** service. It passes the work to a specialized classification service (`classify.py`). This scanner (`ocr-categorizer/v2.1`) processes receipt images, **finds no PII**, and classifies the action as `Class3_Confidential`. A final agent, `AIGovernanceAgent-007`, reviews the clean scan, confirms it complies with a "StrictPIIPolicy" (which only adds friction for `Class1_PII`), and places this final, "approved" pebble onto the DAG.

```sh
# This is the canonical example of the AIGovernanceAgent-007 submitting its final
# work after a successful, non-escalated verification by another service.

# The -d payload is a JSON file representing the final "approved" pebble.
curl -X POST https://dag.internal/place_pebble \
-H "Content-Type: application/json" \
-d '{
  "action": {
    "step": "ApproveAndCompleteIntake",
    "source": "AIGovernanceAgent-007",
    "completed_task": "AIM_CASE0012345_FinalReview"
  },
  "state": {
    "use_case_id": "EXP-CAT-001",
    "status": "ApprovedForStandardDeployment",
    "use_case_registry": {
      "overall_mcp_data_classification": {
        "value": "Class3_Confidential",
        "attesting_agent_id": "AIGovernanceAgent-007",
        "reason": "All constituent actions verified as low-risk (no PII)."
      },
      "verified_tool_actions": [
        {
          "action_name": "categorizeExpenseFromImage",
          "data_classification": {
            "value": "Class3_Confidential",
            "attesting_agent_id": "https://internal.api/ocr-categorizer/v2.1",
            "reason": "Scan complete. No PII markers found in image or extracted text."
          },
          "mcp_extensions_required": []
        }
      ]
    }
  },
  "proof": "e6f1a2b3c4d5e6f1a2b3c4d5e6f1a2b3c4d5e6f1a2b3c4d5e6f1a2b3c4d5e6f1"
}'

```

## 3. Protocol Explanation

* **The `action` block:** A record of the work this final agent (`AIGovernanceAgent-007`) just completed—in this case, approving the task based on the provided evidence.
* **The `state` block:** The agent's final understanding of the world, which now represents a completed task. The logical coherence is key:
   * The `use_case_id` is `EXP-CAT-001`, which directly relates to the `action_name` `categorizeExpenseFromImage`.
   * The `attesting_agent_id` is now a __versioned API endpoint__: `"https://internal.api/ocr-categorizer/v2.1"`. This provides a machine-readable, non-repudiable link to the exact service and version that performed the classification, enhancing auditability.
   * The `mcp_extensions_required` list is now an __empty array `[]`__. This is a deliberate and meaningful signal. Based on a policy (e.g., "Only add friction for Class 1 PII"), the `AIGovernanceAgent-007` has determined that since the data was classified as `Class3_Confidential` (and not PII), no additional security controls are needed. The next system to process this pebble knows it can proceed with a standard, low-friction workflow.

* __The `proof` field:__ A cryptographic hash that guarantees the integrity and authenticity of this final, approved `state`. It is generated as `SHA256(canonical_json(state) + secret_ai_passport_breadcrumb)`. Any participant can verify this proof to trust that the state has not been tampered with since its approval.

```sh

</details>
```
