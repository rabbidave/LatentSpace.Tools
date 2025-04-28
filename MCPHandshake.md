# Handshake MCP Architecture

A secure integration pattern for connecting AI assistants with sensitive operations through coordinated, transaction-specific authentication protocols.

## Overview

The Handshake MCP Architecture provides a robust security framework for AI assistants to interact with sensitive systems while maintaining strong security boundaries. It implements a two-phase handshake mechanism that ensures transaction-specific authorization without standing privileges, aligned with modern zero trust principles.

## Core Concepts

### 1. Coordinated Components

This architecture consists of two primary components that work through a secure handshake mechanism:

- **Local MCP Server**: Deployed on the user's device or local environment; responsible for initiating transaction requests.
- **Remote MCP Service**: Serverless function or hibernated service that awaits handshake signals to execute sensitive operations.

These components establish trust through a cryptographically verified handshake protocol while maintaining independent security contexts. User authentication (establishing user identity to the Remote Service) may precede the transaction-specific handshake.

### 2. Just-in-Time Resource Allocation

Resources are allocated only when needed for a specific transaction:

- Remote services remain dormant or scale to zero when inactive.
- New, isolated instances can be spawned for each operation or handshake phase.
- Compute resources are consumed ephemerally, aligning cost with activity.

### 3. Transaction-Bound Ephemeral Auth

Each sensitive operation requires a distinct, two-phase handshake explicitly bound to that specific transaction instance:

- **Phase 1: Authorization Request**: The authenticated user requests authorization for a specific target operation with defined parameters.
- **Phase 2: Ephemeral Token Grant & Consumption**: Upon successful authorization, a unique, single-use Ephemeral Transaction Token is generated and returned. This token is:
  - Tied cryptographically to the specific transaction UUID, authorized user, target operation, and parameters.
  - Limited in scope strictly to the single authorized business operation instance.
  - Given a very short expiration period (e.g., seconds), sufficient only to be immediately used.
  - Managed via a secure state store where its validity is tracked.

**Token Consumption**: The Ephemeral Transaction Token is presented alongside the operation request and is atomically consumed upon validation before the sensitive operation executes, preventing replay attacks.

## Architecture Diagram

```
┌─────────────────┐                   ┌─────────────────────────┐
│                 │                   │                         │
│   AI Assistant  │                   │ User Identity Provider  │
│  (MCP Client /  │                   │      (Session Auth)     │
│ Local MCP Srvr) │                   │                         │
│                 │                   └───────────┬─────────────┘
└───────┬─────────┘                               │ Session Token
        │                                         │ (e.g., JWT)
        │                                         │
        │ 1. Auth Req (Tool + Params + Metadata)  ▼
        ├─────────────────────────────────>┌─────────────────┐
        │         (Session Token)          │                 │
        │                                  │   Remote MCP    │
        │ 2. Ephemeral Tx Token <----------│     Service     │
        │                                  │                 │
        │ 3. Execute Tool (Tool + Params)  │       +         │
        ├─────────────────────────────────>│                 │
        │   (Session Token +               │   State Store   │
        │    Ephemeral Tx Token Header)    │                 │
        │                                  │                 │
        │ 4. Result + Proof <--------------│                 │
        │                                  └───────┬─────────┘
        │                                          │
        │                                          │ Consumes Tx Token,
        │                                          │ Calls Service API
        │                                          ▼
        │                            ┌─────────────────────────┐
        │                            │                         │
        │                            │    Secure VPC/Cloud     │
        │                            │    Environment          │
        │                            │  ┌───────────────────┐  │
        │                            │  │                   │  │
        │                            │  │ Sensitive Service │  │
        │                            │  │ APIs with keys    │  │
        │                            │  │                   │  │
        │                            │  └───────────────────┘  │
        │                            │                         │
        │                            └─────────────────────────┘
```

## Operational Flow

### Initialization:
1. User activates the local MCP client/server.
2. User authenticates to the Remote MCP Service via standard OAuth/JWT flows, obtaining a primary Session Token.
3. Local client prepares for transaction-specific handshakes.

### Transaction Binding & Authorization Request:
1. AI assistant initiates a transaction (e.g., "Create invoice for X").
2. Local client identifies the target tool and parameters, determining the required authentication flow based on the tool's sensitivity classification (public, internal, confidential, or restricted).
3. Local client sends an Authorization Request call to the Remote MCP Service, passing the target tool, parameters, tool sensitivity metadata, and the user's Session Token.
4. Remote service validates the Session Token, generates a unique transactionId, hashes parameters, and checks user permissions according to the tool's sensitivity tier.

### Ephemeral Token Grant (Handshake Phase 1 Complete):
1. Remote service, only upon explicit request from the MCP client, generates a unique, single-use Ephemeral Transaction Token (nonce).
2. Remote service stores the transaction state (transactionId, userId, targetTool, paramsHash, status: pending, short expiry) in a secure state store.
3. Remote service returns the transactionId and the Ephemeral Transaction Token to the Local client.

### Operation Execution (Handshake Phase 2):
1. Local client immediately constructs the operation request.
2. Local client sends the request to the Remote MCP Service, passing:
   - The user's Session Token
   - The Ephemeral Transaction Token
   - The required operation parameters
   - The sensitivity classification metadata
3. Remote service validates the Session Token again.
4. Remote service retrieves the transaction state from the store using the Ephemeral Transaction Token.
5. The service atomically consumes the token and validates its expiry, user, tool, and parameter hash against the current request.
6. For restricted sensitivity tier operations, a confirmation agent validates the request before execution (similar to the dual-agent system in agent(s).py).

### Sensitive Operation & Proof Generation:
1. If ephemeral token validation succeeds and sensitivity tier validation requirements are met, the Remote MCP Service executes the sensitive operation.
   - For public tier: Minimal validation
   - For internal tier: Public key verification
   - For confidential tier: Regex and AST validation
   - For restricted tier: Secondary agent confirmation
2. Cryptographic proofs (receipts) are generated, bound to the transactionId.
3. Results and the complete proof chain are returned securely to the Local client.
4. Local client validates the cryptographic receipts.
5. Validation level and sensitivity tier are included in the audit trail.

### Handshake Termination:
1. The Ephemeral Transaction Token is already invalidated/consumed.
2. Operation receipt and audit logs are persisted.
3. Remote service instance may return to hibernation.
4. Secure connection related to the specific operation call is closed.

## Security Benefits

### Zero Standing Privileges
- No long-lived tokens grant direct access to sensitive operations.
- The user's Session Token only grants the privilege to request a transaction authorization, not execute it.
- The Ephemeral Transaction Token grants privilege only for the specifically authorized operation instance.
- Principle of least privilege is enforced dynamically at the transaction level.

### Isolated Contexts
- Each handshake establishes an isolated security context for the specific transaction.
- Compromise of one session token does not automatically grant execution rights.
- Clear security boundaries between handshake participants and phases.

### Reduced Attack Surface
- Sensitive components only execute logic after successful validation of both session and ephemeral tokens.
- Minimal network exposure of critical services; ephemeral tokens reduce the window for credential misuse.
- Defense-in-depth through layered authentication (session + transaction).

### Enhanced Audit Trail
- Distinct transactionId for each authorized operation attempt.
- Clear tracking of authorization requests, ephemeral token issuance, consumption attempts, and operation execution.
- Detailed logs associate user identity, client, requested operation, parameters, and timing for each step.

## Implementation Considerations

### Tiered Access Control

The architecture implements a tiered approach to API security based on metadata classification in the JSON header:

1. **Public Tools**: 
   - No authentication required
   - Available to any client
   - Lowest sensitivity level

2. **Internal APIs**:
   - Requires public key verification
   - Basic security checks
   - Medium-low sensitivity

3. **Confidential APIs**:
   - Requires Regex pattern validation
   - Requires AST (Abstract Syntax Tree) validation for code execution
   - Medium-high sensitivity
   - May implement parameter sanitization

4. **Restricted APIs**:
   - Requires all previous validations
   - Requires separate confirmation agent validation
   - Employs a dual-agent approval system similar to the one described in agent(s).py
   - Highest sensitivity level
   - May implement additional waiting periods or human approval

### Transaction-Bound Token Lifecycle
- A two-phase protocol (authorize->execute) manages the lifecycle when required by policy.
- Ephemeral Transaction Tokens are typically nonces or opaque strings whose state is managed server-side.
- State must include bindings to transactionId, userId, targetTool, paramsHash, sensitivity tier, and a short expiry.
- Atomic Consumption upon validation is the primary revocation mechanism, preventing replays.
- Parameter hashing ensures the token cannot be reused for the same tool with different inputs.
