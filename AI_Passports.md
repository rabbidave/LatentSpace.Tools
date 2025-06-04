## Auditable AI Intake w/ AI Passports 🛂

This document outlines the process for onboarding AI use cases, leveraging ServiceNow as a key interface within an enterprise **AI Intake Platform**.

*   Every AI initiative, from initial proposal to deployment, is meticulously documented and reviewed for security and compliance.
*   Aligns with relevant frameworks like [**MCP Handshake**](https://github.com/rabbidave/LatentSpace.Tools/blob/main/MCPHandshake.md), [**Data Classification**](https://github.com/rabbidave/LatentSpace.Tools/blob/main/classify.md), as well as information handling standards
*   Identifies specific tool actions and their classifications upfront, resulting in auditable artifacts.
*   The Auditable AI Intake approach allows for cohesive data gathering, integrating information from external systems

---

### Secure Front Door: AI Use Case Submission via ServiceNow Portal

Any employee (e.g., "Steve from Accounting") can propose an AI use case through a dedicated catalog item—for instance, via a ServiceNow portal form—which acts as a key data capture point within the enterprise's AI Intake Platform. While the portal itself employs standard secure submission protocols, the robust Man-in-the-Middle (MITM) protection for the AI use case's auditable lifecycle is anchored by the **`system_breadcrumb.txt`**. This breadcrumb is a cryptographic hash of the `use_case_registry.json`, which is meticulously compiled from the details submitted through this initial 'Front Door'. Thus, the integrity of the data captured here is paramount, as it directly underpins the non-repudiable audit trail secured by the breadcrumb. The form emphasizes detailing the specific functions/actions within each tool.

<details>
<summary>📋 ServiceNow Portal Form: Initial Request Fields </summary>

*   **Requested For:** [Auto-filled or User Input, potentially cross-referenced via the intake platform]
*   **Department:** [User Input]
*   **Short Description:** (e.g., "AI for automated expense report categorization and payout")
*   **Business Problem/Opportunity:** "Describe the problem this AI will solve or the benefit it will provide."
*   **Overall Use Case Data Classification (Your Best Effort - MCP Handshake 1-5):** [Dropdown: Class 1, Class 2, Class 3, Class 4, Class 5]
    *   *(Consider the most sensitive data the *entire use case* will handle. Link to internal MCP Handshake Data Classification guide.)*
*   **Overall Use Case Relevant Jurisdiction(s) (e.g., GDPR, CCPA, HIPAA):** [Text field or multi-select list]
*   **Proposed AI Tools & Specific Actions/Functions to be Invoked:**
    *   *(This section is repeatable for each distinct external tool or internal system/API the AI will interact with. Information might be pre-filled or validated against a central tool registry managed by the single intake platform.)*
    *   **Tool/System Name & Vendor/Owner:** (e.g., "VendorPay Secure Processor by FinSecure Inc.", "Internal HRIS API by HR Tech Team")
        *   *(Repeatable sub-section for each specific Action/Function within this Tool/System)*
        *   **Specific Action/Function Name:** (e.g., `initiateSinglePayment`, `getEmployeeBankAccountDetails`, `categorizeExpenseText`, `list_transactions`)
            *   *(This maps to a known MCP-proxied function or a function that *will need* to be proxied via MCP).*
        *   **Purpose of this Action:** (e.g., "To send payment instructions for approved expenses," "To retrieve bank details for payout," "To classify receipt text into expense categories")
        *   **Data Types Created/Accessed/Modified by this Action:** (e.g., "Payment amount, payee ID, currency", "Employee bank account number, routing number", "Text strings, category codes")
        *   **Your Best-Effort MCP Data Classification for THIS SPECIFIC ACTION (Class 1-5):** [Dropdown: Class 1, Class 2, Class 3, Class 4, Class 5]
            *   *(Link to [MCP Handshake](https://github.com/rabbidave/LatentSpace.Tools/blob/main/MCPHandshake.md) & [Data Classification](https://github.com/rabbidave/LatentSpace.Tools/blob/main/classify.md) guides. Consider the data handled *by this specific action*).*
*   **Attestation:** "I attest that the information provided regarding overall use case classification, jurisdictions, and the specific AI tool actions and their individual data classifications is accurate to the best of my current understanding. I have consulted the linked MCP Handshake Data Classification guide." [Checkbox - Mandatory]
*   **Primary Business Contact:** [Defaults to Requested For, or can be specified]

</details>

---

### Step 1: Parent Ticket Creation & Initial Triage (ServiceNow Workflow)

1.  **Automated Ticket Creation:** An "AI Use Case Onboarding" ticket (e.g., `AIM_CASE0012345`) is created in ServiceNow, acting as the record within the single intake platform.
2.  **Assignment:** Routed to "AI Governance Team" or "AI Intake Coordinator."
3.  **Initial Review:** The team reviews submitted information, *paying close attention to the declared tool actions and their proposed classifications*. This initial check is critical for flagging high-risk actions early.

---

### Step 2: Structured Information Gathering & Verification (ServiceNow Tasks)

The sub-tasks provide a concrete starting point for verification, potentially drawing on or pushing data to other systems connected to the single intake platform (e.g., a master data management system for data classifications or a GRC tool for compliance checks).

<details>
<summary>⚙️ Example ServiceNow Sub-Tasks </summary>

*   **Task 1: Use Case Scope & Objective Validation**
    *   **Assigned to:** Business Contact, AI Intake Coordinator.
    *   **Objective:** Confirm the overall business goals and how the proposed AI (and its specific tool actions) will achieve them.
*   **Task 2: Verification of Tool Actions, Data Classifications & MCP Requirements**
    *   **Assigned to:** Data Steward, IT Security (including MCP Handshake SMEs), AI Intake Coordinator, Business Contact.
    *   **Objective:**
        *   For **each declared Specific Action/Function**:
            *   Verify its existence and intended behavior within the specified Tool/System.
            *   Rigorously confirm the **MCP Handshake data classification (Class 1-5)** based on the actual data it processes.
            *   Determine the **mandatory MCP Handshake security extensions** required for this specific action (e.g., standard MCP 2.1, transaction-bound tokens, atomic consumption, dual-agent validation).
        *   Confirm the **overall use case data classification** based on the highest sensitivity of its constituent verified actions.
*   **Task 3: Jurisdictional & Regulatory Impact Assessment**
    *   **Assigned to:** Legal/Compliance Team, Data Privacy Officer.
    *   **Objective:** Assess regulatory impact based on the overall use case jurisdiction and the specific data handled by each verified tool action.
*   **Task 4: Technical Design for MCP Integration**
    *   **Assigned to:** IT/Application Teams, Security Architects.
    *   **Objective:** Design the technical integration, detailing how *each tool action* will be securely invoked via the MCP framework (Local MCP Client, Remote MCP Service, state store interactions) according to its verified classification and required extensions.

</details>

---

### Step 3: `use_case_registry.json` Creation & Approval

The `use_case_registry.json` accurately reflects this granular detail. This registry file might be versioned and stored in a central artifact repository, linked from the ServiceNow ticket within the single intake platform.

<details>
<summary><strong>📄 Artifact Example: <code>use_case_registry.json</code> (w /Tool Actions)</strong></summary>

```json
{
  "passport_schema_version": "1.2.0",
  "use_case_id": "ACC-EXP-001",
  "use_case_name": "Automated Expense Report Categorization & Payout",
  // ... other use case level fields like description, owner, status, service_now_ticket_ref (linking back to the single intake platform record) ...
  "overall_mcp_data_classification": "Class1_PII", // Final overall classification
  "jurisdictions": ["US-Federal", "California-CCPA", "EU-GDPR"],
  "data_inputs_summary": [ /* High-level summary of data inputs if needed */ ],
  "data_outputs_summary": [ /* High-level summary of data outputs if needed */ ],
  "tools_and_actions": [
    {
      "tool_system_id": "VendorPay_API-v1.5",
      "tool_system_name": "VendorPay Secure Payment Processor",
      "vendor_owner": "FinSecure Inc.",
      "actions": [
        {
          "action_name": "initiateSinglePayment",
          "description": "Processes payment instructions to employee bank accounts.",
          "data_elements_processed": ["payee_tokenized_id", "payment_amount", "currency_code", "expense_report_id"],
          "mcp_classification_attested_by_user": "Class1_PII",
          "mcp_classification_verified": "Class1_PII",
          "mcp_handshake_extensions_required": ["TransactionBoundTokens", "AtomicTokenConsumption", "DualAgentValidation"],
          "target_api_endpoint_details": "POST /v1/payments"
        },
        {
          "action_name": "getPaymentStatus",
          "description": "Retrieves the status of a previously initiated payment.",
          "data_elements_processed": ["payment_transaction_id", "status_code", "status_message"],
          "mcp_classification_attested_by_user": "Class2_SensitivePersonalData",
          "mcp_classification_verified": "Class2_SensitivePersonalData",
          "mcp_handshake_extensions_required": ["TransactionBoundTokens"],
          "target_api_endpoint_details": "GET /v1/payments/{transaction_id}/status"
        }
      ]
    },
    {
      "tool_system_id": "Internal-OCR-Categorizer-v2.1",
      "tool_system_name": "Internal OCR & Expense Categorization Engine",
      "vendor_owner": "In-House Development AI Team",
      "actions": [
        {
          "action_name": "categorizeExpenseFromImage",
          "description": "Extracts text from receipt images, classifies expense types.",
          "data_elements_processed": ["receipt_image_binary", "extracted_text", "expense_category_code", "confidence_score"],
          "mcp_classification_attested_by_user": "Class3_ConfidentialPersonalData",
          "mcp_classification_verified": "Class3_ConfidentialPersonalData",
          "mcp_handshake_extensions_required": ["TransactionBoundTokens", "EnhancedValidation (Schema)"],
          "target_api_endpoint_details": "POST /v2/categorize/image"
        }
      ]
    }
    // ... other tools and their specific actions ...
  ],
  // ... stakeholders, attestations_log_link etc. ...
}
```

</details>

1.  **Attachment & Approval Workflow:** The detailed `use_case_registry.json` is attached to the ServiceNow ticket and routed for approval. Approvers have a clear view of the specific operations and their individual risk profiles.
2.  **Version Control Integration:** Recommended for the `use_case_registry.json`.

---

### Step 4: `system_breadcrumb.txt` Generation & Logging Control

The `use_case_registry.json` is hashed with all the detailed and accurate information regarding specific tool actions. The consistent and accurate inclusion of this `system_breadcrumb_id` in all relevant operational logs (e.g., `tool_invocation.log`) is a critical control. Model Risk Management (MRM) / 2nd Line of Defense teams will have oversight responsibilities to periodically validate the presence and integrity of these breadcrumbs in log streams as part of their monitoring duties, ensuring that the audit trail remains anchored and trustworthy.

<details>
<summary><strong>🍞 Artifact Example: <code>system_breadcrumb.txt</code></strong></summary>

```text
a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2
```
*(Example hash based on `use_case_registry.json`)*.
</details>

---

### Step 5: Operationalizing, Logging, & MRM/2nd Line Monitoring and Audit Oversight

Operational logs are generated by the system during tool invocations, providing detailed records of AI actions. While these logs are essential for operational troubleshooting and security incident response by operational teams, the Model Risk Management (MRM) / 2nd Line of Defense assumes specific ongoing monitoring and audit oversight responsibilities. These responsibilities are often managed as distinct sub-tasks within the AI Governance framework after deployment, focusing on the integrity of the control framework and model performance rather than individual transaction reviews.

<details>
<summary><strong>📜 Artifact Example: <code>tool_invocation.log</code> ( i.e. System Actions)</strong></summary>

The `tool.name` field in the MCP Handshake log clearly identifies the *specific action* that was invoked. The `system_breadcrumb_id` links this specific invocation back to its approved configuration.

```json
{
  "log_entry_id": "log-uuid-...",
  "system_breadcrumb_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
  "mcp_handshake_schema": "MCP.Handshake.v1",
  "transaction": { /* ... */ },
  "tool": {
    "name": "VendorPay_API-v1.5/initiateSinglePayment", // Tool_System_ID + Action_Name from registry
    "version": "1.5.2", // Actual version of the API endpoint/function called
    "sensitivity": "Class1_PII", // Verified classification of THIS SPECIFIC ACTION
    "parameters_hash": "sha256-of-actual-parameters-for-initiateSinglePayment...",
    "target_api": {
      "name": "VendorPay Payment Initiation API", // Corresponds to tool_system_name
      "operation": "POST /v1/payments" // Corresponds to target_api_endpoint_details
    }
  },
  "authentication": { /* ... Ephemeral token bound to this specific action and its params ... */ },
  "validation": {
    "status": "APPROVED",
    "checks_performed": [ /* MCP Handshake checks relevant to Class1_PII actions */
      "dual_agent_confirmation_received"
    ],
    "tier_level_applied": "Class1_PII" // Validation tier applied for this action
  },
  "audit": {
    "integration_id": "ACC-EXP-001_VendorPay_initiateSinglePayment", // More specific integration point
    /* ... */
  },
  /* ... execution_result, error_handling ... */
}
```

</details>

<details>
<summary>🔍 MRM/2nd Line Ongoing Monitoring & Audit Sub-Tasks</summary>

*   **System Breadcrumb Validation:**
    *   Periodically audit operational logs (e.g., samples from `tool_invocation.log` streams or aggregated logging platforms) to confirm the consistent presence and correctness of the `system_breadcrumb_id`.
    *   Verify that the `system_breadcrumb_id` in logs matches the officially registered breadcrumb for the deployed AI use case.
*   **Model Performance Monitoring:**
    *   Monitor key performance indicators (KPIs) and metrics of the AI model against the baselines and thresholds defined in the `use_case_registry.json`.
    *   Review model drift, accuracy, fairness, and other relevant performance characteristics over time.
    *   Validate that model behavior remains within acceptable parameters and aligns with its intended purpose and risk profile.
*   **Compliance & Control Adherence:**
    *   Assess adherence to the data classifications and MCP Handshake security extensions specified for each tool action in the `use_case_registry.json` through sampled log reviews or automated checks.
    *   Review access patterns and aggregated tool invocation frequencies for anomalies or deviations from expected behavior defined in the AI Passport.
    *   Ensure that logging mechanisms themselves are functioning correctly and haven't been tampered with.
*   **Issue Escalation & Reporting:**
    *   Report findings, deviations, and risks to relevant stakeholders, including AI Governance, business owners, and IT/Operations.
    *   Track remediation of any identified issues related to model performance, control failures, or audit discrepancies.

</details>

---
