## Target-State Architecture for Zero-Standing Privileges

A secure integration pattern enabling AI assistants to interact with sensitive business systems through coordinated, transaction-specific authentication protocols with built-in defense-in-depth.

## Overview

The MCP Handshake Architecture provides an enterprise-grade security framework for AI integrations, implementing a defense-in-depth strategy with clear separation of concerns. It uses a two-phase handshake mechanism ensuring transaction-specific authorization with zero standing privileges, aligning with modern zero trust principles and data classification requirements.

### Key Components and Terminology

- **AI Assistant** implements the **Local MCP Client** - initiates requests but cannot directly access sensitive APIs
- **Confirmation Agent** implements the **Remote MCP Service** - acts as a secure gateway validating all operations
- **State Store** - provides atomic token management (typically Redis, DynamoDB, or similar with TTL support)
- **User Identity Provider** - external system for user authentication and session token issuance
- **Target Enterprise APIs** - back-end systems containing sensitive data or operations

## Core Architecture Principles

### 1. Dual-Agent Authority with Coordinated Components

The architecture implements separation of powers through a dual-validation pattern:

- **Local MCP Client (implemented by AI Assistant)**: Initiates transaction requests and manages client-side workflow, but cannot directly access sensitive systems.

- **Remote MCP Service (implemented by Confirmation Agent)**: Acts as a secure gateway that independently validates operations, manages token lifecycle, and is the only component with access to sensitive API credentials. This separation ensures that even if the AI Assistant is compromised, it cannot directly access target systems.

- **Secure State Store**: Tracks ephemeral token states and ensures atomic consumption, typically implemented using Redis, DynamoDB, or a similar system with TTL support and atomic operations to prevent race conditions during token validation.

Each component maintains isolated security contexts connected through cryptographically verified handshakes, with initial trust bootstrapped through TLS, certificate validation, and secure secret management.

### 2. Ephemeral Action Authorization with Replay Protection

Every sensitive operation requires explicit, time-bound authorization with built-in replay protection:
- **Phase 1: Request Authorization**: Authenticated user requests a specific operation with parameters
- **Phase 2: Nonce Generation & Parameter Binding**: A unique nonce (ephemeral token) is generated and cryptographically bound to the parameter hash
- **Phase 3: Atomic Execution & Token Consumption**: Operation proceeds only after validation succeeds, and the token is atomically consumed

This approach provides two-factor replay protection:
1. The ephemeral token acts as a nonce (number used once) that is invalidated after use
2. The parameter hash binds this nonce to the exact operation parameters

For example, if calling a function `transferFunds({fromAccount: "12345", toAccount: "67890", amount: 500})`, the `parameter_hash` would be `SHA256(JSON.stringify({fromAccount: "12345", toAccount: "67890", amount: 500}))`. This ensures that the exact same parameters must be provided in Phase 2, preventing an attacker from changing parameters (e.g., changing the amount or destination account) between authorization and execution.

### 3. Tiered Access Control

The architecture implements a tiered approach to API security based on data classification:

1. **Public (Tier 1)**: 
   - Basic input validation and sanitization
   - Available with minimal authentication
   - Lowest sensitivity level
   - *Examples*: Public reference data, open documentation, non-personalized information

2. **Internal (Tier 2)**:
   - Public key verification for request authenticity
   - Parameter sanitization and schema validation
   - Medium-low sensitivity
   - *Examples*: Internal reports, departmental dashboards, non-sensitive operations

3. **Confidential (Tier 3)**:
   - Comprehensive validation using multiple techniques based on context:
     - Regex pattern validation for structured inputs
     - Schema validation for complex objects
     - Code analysis (AST or alternatives) for execution requests
   - Parameter transformation or sanitization required
   - Medium-high sensitivity
   - *Examples*: Financial operations, PII access, business transactions

4. **Restricted (Tier 4)**:
   - Includes all lower-tier validations
   - Secondary validation by independent system
   - Highest sensitivity level
   - May require time-delayed execution or human approval workflows
   - *Examples*: Administrative actions, critical infrastructure changes, high-value transactions

## Implementation Reference Architecture

```
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

## Reference Implementation Schema

```json
{
  "schema": "MCP.Handshake.v1",
  "transaction": {
    "id": "uuid-for-this-specific-request",
    "timestamp": "ISO-8601-timestamp",
    "user": {
      "id": "authenticated-user-id",
      "roles": ["role1", "role2"]
    }
  },
  "tool": {
    "name": "target-operation-name",
    "version": "1.0.0",
    "sensitivity": "CONFIDENTIAL", // PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED
    "parameters_hash": "sha256-of-parameters-object",
    "target_api": {
      "name": "enterprise-api-identifier",
      "operation": "specific-api-operation"
    }
  },
  "authentication": {
    "session_token": "jwt-or-other-identity-token",
    "ephemeral_token": "single-use-transaction-bound-token",
    "expiry": "ISO-8601-timestamp-short-lifespan",
    "token_state": {
      "consumed": false,
      "consumption_timestamp": null
    }
  },
  "validation": {
    "status": "APPROVED", // APPROVED, DENIED, PENDING
    "timestamp": "ISO-8601-timestamp",
    "checks_performed": ["parameter_validation", "pattern_validation", "code_analysis"],
    "tier_level": "CONFIDENTIAL",
    "reason": "Optional explanation if DENIED"
  },
  "audit": {
    "request_ip": "client-ip-address",
    "client_id": "application-identifier",
    "integration_id": "specific-integration-identifier",
    "receipt": {
      "transaction_proof": "cryptographic-signature-of-transaction-details", // Typically HMAC or digital signature of transaction data
      "timestamp": "ISO-8601-timestamp"
    }
  },
  "error_handling": {
    "status_code": null, // HTTP status code if error occurred
    "error_type": null,  // AUTH_ERROR, VALIDATION_ERROR, EXECUTION_ERROR, etc.
    "message": null,     // Human-readable error message
    "retry_allowed": true // Whether retry is permitted for this error
  }
}
```

## Operational Lifecycle

### Integration Setup Phase
1. **Enterprise Team** (Provides infra, standards, and requirements):
   - Defines data classification scheme and sensitivity tiers
   - Establishes validation requirements per tier
   - Publishes standards and requirements

2. **IT/Ops Team**:
   - Configures runtime environment
   - Sets up monitoring and logging
   - Deploys Remote MCP Service and State Store infrastructure

3. **Application Team**:
   - Implements Local MCP Client integration
   - Configures field mappings and classifications
   - Develops business-specific integration logic

### Transaction Execution Flow

1. **Authentication & Authorization**:
   - User authenticates using standard OAuth/JWT flows
   - Local MCP Client collects operation request details
   - Client determines sensitivity tier for the requested operation

2. **Handshake Phase 1: Request Authorization**:
   - Local MCP Client sends request with tool name, parameters, tier metadata
   - Remote MCP Service validates session token and user permissions
   - Parameters are hashed to create a unique operation fingerprint
   - Server generates transaction ID and ephemeral token with short expiry
   - Token is cryptographically bound to the parameter hash

3. **Handshake Phase 2: Execute Operation**:
   - Local MCP Client immediately sends execution request with:
     - Original parameters (will be re-hashed server-side for verification)
     - Session token
     - Ephemeral transaction token
   - Remote MCP Service:
     - Re-hashes parameters and verifies match with original hash
     - Atomically consumes the token to prevent replay attacks
     - Performs validation checks based on sensitivity tier:
       - Tier 1-2: Basic input validation
       - Tier 3: Pattern validation, AST validation
       - Tier 4: Secondary confirmation agent validation

4. **Operation Execution & Response**:
   - Target API operation proceeds only after all validations succeed
   - Execution is logged with complete audit trail
   - Results and cryptographic proof receipts returned to client
   - State Store maintains record of consumed token
