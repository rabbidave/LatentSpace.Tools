### Target-State Architecture for Zero-Standing Privileges

A secure integration pattern enabling AI assistants to interact with sensitive business systems through coordinated, transaction-specific authentication protocols with built-in defense-in-depth.

#### Overview

The MCP Handshake Architecture provides an enterprise-grade security framework for AI integrations, implementing a defense-in-depth strategy with clear separation of concerns. It uses a two-phase handshake mechanism ensuring transaction-specific authorization with zero standing privileges, aligning with modern zero trust principles and data classification requirements.

#### Key Components and Terminology

- **AI Assistant** implements the **Local MCP Client** - initiates requests but cannot directly access sensitive APIs
- **Confirmation Agent** implements the **Remote MCP Service** - acts as a secure gateway validating all operations
- **State Store** - provides atomic token management (typically Redis, DynamoDB, or similar with TTL support)
- **User Identity Provider** - external system for user authentication and session token issuance
- **Target Enterprise APIs** - back-end systems containing sensitive data or operations

#### Core Architecture Principles

##### 1. Dual-Agent Authority with Coordinated Components

The architecture implements separation of powers through a dual-validation pattern:

- **Local MCP Client (implemented by AI Assistant)**: Initiates transaction requests and manages client-side workflow, but cannot directly access sensitive systems.
- **Remote MCP Service (implemented by Confirmation Agent)**: Acts as a secure gateway that independently validates operations, manages token lifecycle, and is the only component with access to sensitive API credentials.
- **Secure State Store**: Tracks ephemeral token states and ensures atomic consumption.
   Each component maintains isolated security contexts connected through cryptographically verified handshakes.

##### 2. Ephemeral Action Authorization with Replay Protection

Every sensitive operation requires explicit, time-bound authorization:

- **Phase 1: Request Authorization**: Authenticated user requests an operation.
- **Phase 2: Nonce Generation & Parameter Binding**: A unique nonce (ephemeral token) is generated and cryptographically bound to the parameter hash.
- **Phase 3: Atomic Execution & Token Consumption**: Operation proceeds after validation; token is atomically consumed.
   This provides two-factor replay protection (ephemeral token + parameter hash binding).

##### 3. Tiered Access Control

Access is tiered based on data classification:

1. **Public (Tier 1)**: Basic validation, minimal auth (e.g., public reference data).
2. **Internal (Tier 2)**: PKI verification, parameter sanitization (e.g., internal reports).
3. **Confidential (Tier 3)**: Comprehensive validation (Regex, Schema, AST), parameter transformation (e.g., financial operations, PII access).
4. **Restricted (Tier 4)**: All lower-tier validations + independent secondary validation, highest sensitivity (e.g., admin actions, critical changes).

#### Implementation Reference Architecture

```ini
┌─────────────────┐                   ┌─────────────────────────┐
│                 │                   │                         │
│   AI Assistant  │                   │ User Identity Provider  │
│  (Primary Agent)│                   │      (Session Auth)     │
│                 │                   │                         │
└───────┬─────────┘                   └───────────┬─────────────┘
        │                                         │ Session Token
        │                                         │ (e.g., JWT)
        │ 1. Auth Req (Tool + Params + Metadata)  ▼
        ├─────────────────────────────────>┌─────────────────┐
        │         (Session Token)          │                 │
        │                                  │ Confirmation    │
        │ 2. Ephemeral Tx Token <----------│ Agent + State   │
        │                                  │ Store           │
        │ 3. Execute Tool (Tool + Params)  │                 │
        ├─────────────────────────────────>│                 │
        │   (Session Token +               │                 │
        │    Ephemeral Tx Token)           │                 │
        │                                  │                 │
        │ 4. Result + Proof <--------------│                 │
        │                                  └───────┬─────────┘
        │                                          │
        │                                          │ Validated Call
        │                                          ▼
        │                            ┌─────────────────────────┐
        │                            │                         │
        │                            │    Secure VPC/Cloud     │
        │                            │    Environment          │
        │                            │  ┌───────────────────┐  │
        │                            │  │                   │  │
        │                            │  │ Enterprise APIs   │  │
        │                            │  │ & Services        │  │
        │                            │  │                   │  │
        │                            │  └───────────────────┘  │
        │                            │                         │
        │                            └─────────────────────────┘

```

---

<!-- Page Break -->

### Reference Implementation Schema (MCP.Handshake.v1)

```json
{
  "schema": "MCP.Handshake.v1",
  "transaction": {
    "id": "uuid-for-this-specific-request",
    "timestamp": "ISO-8601-timestamp",
    "user": { "id": "authenticated-user-id", "roles": ["role1", "role2"] }
  },
  "tool": {
    "name": "target-operation-name", "version": "1.0.0",
    "sensitivity": "CONFIDENTIAL", "parameters_hash": "sha256-of-parameters-object",
    "target_api": { "name": "enterprise-api-identifier", "operation": "specific-api-operation" }
  },
  "authentication": {
    "session_token": "jwt-or-other-identity-token", "ephemeral_token": "single-use-transaction-bound-token",
    "expiry": "ISO-8601-timestamp-short-lifespan",
    "token_state": { "consumed": false, "consumption_timestamp": null }
  },
  "validation": {
    "status": "APPROVED", "timestamp": "ISO-8601-timestamp",
    "checks_performed": ["parameter_validation", "pattern_validation", "code_analysis"],
    "tier_level": "CONFIDENTIAL", "reason": "Optional explanation if DENIED"
  },
  "audit": {
    "request_ip": "client-ip-address", "client_id": "application-identifier",
    "integration_id": "specific-integration-identifier",
    "receipt": { "transaction_proof": "cryptographic-signature-of-transaction-details", "timestamp": "ISO-8601-timestamp" }
  },
  "error_handling": {
    "status_code": null, "error_type": null, "message": null, "retry_allowed": true
  }
}

```

### Operational Lifecycle

**Integration Setup Phase:** Enterprise, IT/Ops, and Application teams collaborate to define classifications, configure environments, and implement integration logic.
**Transaction Execution Flow:**

1. User authenticates; Local MCP Client collects request details.
2. **Handshake Phase 1 (Request Authorization)**: Local Client sends request; Remote Service validates session, hashes parameters, generates ephemeral token bound to hash.
3. **Handshake Phase 2 (Execute Operation)**: Local Client sends parameters and tokens; Remote Service re-verifies hash, atomically consumes token, performs tiered validation.
4. Operation executes if all checks pass; results and proof returned.

---

#### Data Classification Mapping

| Data Class | Description                     | Examples                                | Security Extensions Required      |
|------------|---------------------------------|-----------------------------------------|-----------------------------------|
| **Class 1: PII** | Most sensitive personal data      | SSN, payment methods, credentials       | per-integration specifics         |
| **Class 2: Sensitive Personal Data** | Financial txns, personal details | Txn history, refunds, balance         | Transaction-bound tokens + add'l  |
| **Class 3: Confidential Personal Data** | Business-sensitive operations   | Customer profiles, invoices, processing | Transaction-bound tokens + enhanced validation |
| **Class 4: Internal Data** | Standard business operations    | Exchange rates, general account info  | Standard MCP 2.1 authorization    |
| **Class 5: Public Data** | Non-sensitive operations        | Public API endpoints, documentation   | No additional authorization       |

#### Required Custom Extensions

1. **Transaction-Bound Ephemeral Tokens (Class 1-3)**: Cryptographically bind tokens to operation parameters (toolName, paramsHash, userId, dataClass, short expiry).
2. **Atomic Token Consumption (Class 1-3)**: Prevent replay via one-time use (e.g., Redis `EVAL` for GET & DEL).

**Class 4-5 Operations (Internal/Public Data)**: Standard single-phase MCP 2.1 (bearer token).

```ini
┌─────────────────┐    Standard MCP 2.1    ┌──────────────────┐
│   AI Assistant  │◄──── Single Phase ─────┤ Standard MCP 2.1 │
│ (Class 4-5 ops) │      Bearer Token      │   Authorization  │
└─────────────────┘                        └──────────────────┘

```

**Class 1-3 Operations (PII/Sensitive/Confidential)**: Two-phase zero-trust.

```ini
┌─────────────────┐                        ┌──────────────────┐
│   AI Assistant  │                        │ Standard MCP 2.1 │
│ (Class 1-3 ops) │◄─── Session Token ─────┤   Authorization  │
└─────────┬───────┘                        └──────────────────┘
          │ Sensitive Operations (send_money, refund, etc.)
          ▼
┌─────────────────┐    2-Phase Flow        ┌──────────────────┐
│ Enhanced Local  │◄─── Phase 1: Auth  ────┤ Zero-Trust MCP   │
│ MCP Client      │◄─── Phase 2: Execute ──┤ Extension Service│
└─────────────────┘                        └────────┬─────────┘
                                                    │ Class 1-2 Only
                                                    ▼
                                         ┌──────────────────┐
                                         │ Confirmation     │
                                         │ Agent Validator  │
                                         └──────────────────┘

```

#### Financial API Tool Classification Examples

```typescript
const TOOL_CLASSIFICATIONS = {
  "create_payment_method": 1, "update_customer_payment": 1, // Class 1
  "send_money": 2, "refund_transaction": 2,                 // Class 2
  "create_invoice": 3, "process_payment": 3,                // Class 3
  "list_transactions": 4, "get_account_balance": 4,         // Class 4
  "get_exchange_rate": 5                                    // Class 5
};

```

Class 4-5 operations use standard MCP 2.1. Class 1-3 layer zero-trust extensions, determined by `TOOL_CLASSIFICATIONS`.

#### Implementation Priority

1. Phase 1: Class 4-5 ops with standard MCP 2.1.
2. Phase 2: Add transaction-bound tokens for Class 3 ops.
3. Phase 3: Integrate dual-agent validation for Class 1-2 ops.
4. Phase 4: Full zero-trust pipeline with comprehensive audit.

---
