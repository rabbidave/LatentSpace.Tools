# How-To Guide: Zero-Trust Extensions for Agent Operations

This guide details how to enhance a PayPal agent integration, built upon MCP 2.1 (bearer token), with the zero-trust extensions specified in the "MCP Handshake" These extensions are crucial for securing sensitive operations (Class 1-3) by layering transaction-specific authorizations and graduated controls onto the standard MCP 2.1 framework.

## Prerequisites

-   An existing PayPal agent integration using MCP 2.1 (referred to as `standardMCPClient`).
-   Understanding of the "MCP 2.1 Addendum: Zero-Trust Extensions" (Data Classifications, Ephemeral Token Binding, Atomic Consumption, Dual Authorization Flows).
-   Ability to deploy a "Zero-Trust MCP Extension Service" (e.g., serverless functions) to manage the two-phase handshake for sensitive operations.
-   A secure state store (e.g., Redis) for managing ephemeral authorization states.

## Architecture: Layering Zero-Trust onto MCP 2.1

As outlined in the addendum, this approach uses dual flows:

1.  **Class 4-5 Operations (Internal/Public Data)**: Continue to use the standard single-phase MCP 2.1 authorization (e.g., bearer token via `standardMCPClient`).
2.  **Class 1-3 Operations (PII/Sensitive/Confidential)**: Employ a two-phase zero-trust handshake managed by the new "Zero-Trust MCP Extension Service." This involves:
    *   Initial session validation (can be part of standard MCP 2.1).
    *   **Phase 1 (Authorization Request)**: The client requests authorization from the Extension Service, which issues a transaction-bound ephemeral token if the initial identity/intent is valid.
    *   **Phase 2 (Execution Request)**: The client uses this ephemeral token to request action execution from the Extension Service. The service validates the token, re-verifies parameters, applies graduated controls (including a Confirmation Agent for Class 1-2), and, if all checks pass, performs the PayPal operation.

## Implementation Steps

### 1. Define Tool Classifications (from Addendum)

Adopt the precise tool classifications as specified in the addendum. This mapping is critical for routing and applying correct security controls.
<details>
<summary><strong>📋 TypeScript: Tool Classifications and Token Interface</strong></summary>

```typescript
// tool-classifications.ts (shared or accessible by client and extension service)
const TOOL_CLASSIFICATIONS: Record<string, number> = {
  // Class 1: PII Operations
  "create_payment_method": 1,
  "update_customer_payment": 1,
  "get_payment_credentials": 1,

  // Class 2: Sensitive Financial
  "send_money": 2,
  "refund_transaction": 2,
  "process_large_payment": 2,

  // Class 3: Confidential Business
  "create_invoice": 3,
  "process_payment": 3,
  "get_customer_profile": 3, // Note: Moved to Class 3 as per addendum example

  // Class 4: Internal Operations
  "list_transactions": 4,
  "get_account_balance": 4,

  // Class 5: Public Data
  "get_exchange_rate": 5,
  "get_supported_currencies": 5
};

interface EphemeralTokenPayload { // Matches EphemeralTokenBinding from addendum
  transactionId: string;
  toolName: string;
  paramsHash: string;
  userId: string;      // Identity from session
  dataClass: 1 | 2 | 3;
  expiryTimestamp: number; // Unix timestamp for TTL
}
```
</details>

### 2. Create the Zero-Trust MCP Extension Service

This service handles the two-phase handshake for Class 1-3 operations.
<details>
<summary><strong>📋 TypeScript: Zero-Trust MCP Extension Service (Serverless Example)</strong></summary>

```typescript
// zero-trust-mcp-extension-service.ts
import { APIGatewayProxyHandler } from 'aws-lambda';
import { v4 as uuidv4 } from 'uuid';
import *_crypto from 'crypto';
import Redis from 'ioredis';
// Assuming tool-classifications.ts is accessible
// import { TOOL_CLASSIFICATIONS, EphemeralTokenPayload } from './tool-classifications';

const redis = new Redis(process.env.REDIS_URL!);
const EPHEMERAL_TOKEN_TTL_SECONDS = 30; // As per addendum

// --- Helper Functions (Stubs for core logic, focus on handshake) ---
async function verifyIdentityAndSession(sessionToken: string): Promise<string | null> {
  console.log(`Verifying identity via session: ${sessionToken ? 'present' : 'absent'}`);
  // TODO: Implement robust session validation, return unique userId or null
  return sessionToken ? `user_${sessionToken.substring(0, 8)}` : null;
}

async function performPayPalOperationInternal(toolName: string, parameters: any, userId: string): Promise<{ success: boolean; data?: any; error?: string }> {
  console.log(`Performing PayPal op: ${toolName} for user ${userId}`, parameters);
  // TODO: Securely call PayPal API. Credentials must ONLY be accessible here.
  return { success: true, data: { message: `${toolName} executed by extension service.` } };
}

async function callConfirmationAgent(toolName: string, parameters: any, userId: string, dataClass: number): Promise<{ approved: boolean; reason?: string }> {
  console.log(`CONFIRMATION AGENT: Validating Class ${dataClass} tool: ${toolName} for user ${userId}`);
  // TODO: Implement call to Confirmation Agent Validator for Class 1-2 operations.
  // This agent performs independent secondary validation.
  if (dataClass <= 2) {
    return { approved: true }; // Placeholder for successful confirmation
  }
  return { approved: true }; // Not applicable for Class 3+ within this specific check
}

async function recordSecurityTransaction(eventDetails: any): Promise<void> {
  console.log("SECURITY TRANSACTION LOG:", eventDetails);
  // TODO: Log to a secure, immutable audit system.
}

// --- Redis Atomic Consumption (from Addendum) ---
const validateAndConsumeEphemeralToken = async (tokenKey: string): Promise<string | null> => {
  const script = `
    local tokenData = redis.call('GET', KEYS)
    if tokenData then
      redis.call('DEL', KEYS)
      return tokenData
    else
      return nil
    end
  `;
  // Ensure correct usage of redis.eval: KEYS refers to the first element in the keys array.
  // The '1' indicates the number of keys being passed.
  return await redis.eval(script, 1, tokenKey); 
};


// --- API Handlers ---
// Phase 1: Request Authorization for a Sensitive Operation
export const requestSensitiveOperationAuth: APIGatewayProxyHandler = async (event) => {
  try {
    const { sessionToken, toolName, parameters } = JSON.parse(event.body || '{}');
    if (!sessionToken || !toolName || !parameters) {
      return { statusCode: 400, body: JSON.stringify({ error: 'sessionToken, toolName, and parameters are required' }) };
    }

    const userId = await verifyIdentityAndSession(sessionToken);
    if (!userId) {
      return { statusCode: 401, body: JSON.stringify({ error: 'Invalid session or identity verification failed' }) };
    }

    const dataClass = (TOOL_CLASSIFICATIONS[toolName] || 0) as (1 | 2 | 3); // Assert to 1,2,3 or handle error
    if (dataClass > 3 || dataClass < 1) {
        await recordSecurityTransaction({ phase: 'AUTH_REQUEST', toolName, status: 'REJECTED', reason: 'Operation not eligible for zero-trust flow or invalid class' });
        return { statusCode: 400, body: JSON.stringify({ error: `Tool ${toolName} is not a Class 1-3 operation.` })};
    }

    const transactionId = uuidv4();
    const ephemeralToken = _crypto.randomBytes(32).toString('hex');
    const paramsHash = _crypto.createHash('sha256').update(JSON.stringify(parameters)).digest('hex');
    const expiryTimestamp = Math.floor(Date.now() / 1000) + EPHEMERAL_TOKEN_TTL_SECONDS;

    const tokenPayload: EphemeralTokenPayload = { transactionId, toolName, paramsHash, userId, dataClass, expiryTimestamp };
    
    await redis.setex(`tx:${ephemeralToken}`, EPHEMERAL_TOKEN_TTL_SECONDS, JSON.stringify(tokenPayload));

    await recordSecurityTransaction({ phase: 'AUTH_REQUEST', transactionId, userId, toolName, dataClass, paramsHash, status: 'EPHEMERAL_TOKEN_ISSUED', expiry: expiryTimestamp });
    return { statusCode: 200, body: JSON.stringify({ transactionId, ephemeralToken }) };

  } catch (error: any) {
    console.error("Auth Request Phase Error:", error);
    await recordSecurityTransaction({ phase: 'AUTH_REQUEST', event: 'ERROR', error: error.message });
    return { statusCode: 500, body: JSON.stringify({ error: 'Authorization request processing failed' }) };
  }
};

// Phase 2: Execute a Sensitive Operation with an Ephemeral Token
export const executeSensitiveOperation: APIGatewayProxyHandler = async (event) => {
  let tokenPayload: EphemeralTokenPayload | null = null;
  try {
    const { sessionToken, ephemeralToken, toolName, parameters } = JSON.parse(event.body || '{}');
    if (!sessionToken || !ephemeralToken || !toolName || !parameters) {
        return { statusCode: 400, body: JSON.stringify({ error: 'sessionToken, ephemeralToken, toolName, and parameters are required' }) };
    }

    const currentUserId = await verifyIdentityAndSession(sessionToken);
    if (!currentUserId) {
      return { statusCode: 401, body: JSON.stringify({ error: 'Invalid session or identity re-verification failed for execution' }) };
    }

    const tokenDataString = await validateAndConsumeEphemeralToken(`tx:${ephemeralToken}`); // Atomic consumption
    if (!tokenDataString) {
      await recordSecurityTransaction({ phase: 'EXECUTION', ephemeralToken, status: 'REJECTED', reason: 'Ephemeral token invalid, expired, or already consumed' });
      return { statusCode: 403, body: JSON.stringify({ error: 'Invalid, expired, or consumed ephemeral token' }) };
    }
    tokenPayload = JSON.parse(tokenDataString) as EphemeralTokenPayload;

    if (tokenPayload.expiryTimestamp < Math.floor(Date.now() / 1000)) {
        await recordSecurityTransaction({ phase: 'EXECUTION', transactionId: tokenPayload.transactionId, status: 'REJECTED', reason: 'Ephemeral token TTL expired (post-consumption check)' });
        return { statusCode: 403, body: JSON.stringify({ error: 'Ephemeral token TTL expired' }) };
    }

    const currentParamsHash = _crypto.createHash('sha256').update(JSON.stringify(parameters)).digest('hex');
    if (tokenPayload.userId !== currentUserId || tokenPayload.toolName !== toolName || tokenPayload.paramsHash !== currentParamsHash) {
      await recordSecurityTransaction({ phase: 'EXECUTION', transactionId: tokenPayload.transactionId, status: 'REJECTED', reason: 'Execution request mismatch with token binding (userId, toolName, or paramsHash)' });
      return { statusCode: 403, body: JSON.stringify({ error: 'Mismatch between execution request and token binding' }) };
    }

    // Graduated Controls: Parameter Validation & Confirmation Agent for Class 1-2
    // Enhanced validation for Class 3
    let validationSuccess = true;
    let validationError = "";

    if (tokenPayload.dataClass <= 3) { // Enhanced validation for Class 3
        console.log(`Performing enhanced validation for Class ${tokenPayload.dataClass} tool: ${toolName}`);
        // TODO: Implement detailed schema/regex validation for parameters
        // Example:
        // if (!parameters.amount || typeof parameters.amount !== 'number' || parameters.amount <= 0) {
        //    validationSuccess = false; validationError = "Invalid amount for payment.";
        // }
    }

    if (validationSuccess && tokenPayload.dataClass <= 2) { // Confirmation Agent for Class 1-2
      const confirmation = await callConfirmationAgent(toolName, parameters, currentUserId, tokenPayload.dataClass);
      if (!confirmation.approved) {
        validationSuccess = false;
        validationError = confirmation.reason || "Denied by Confirmation Agent";
      }
    }

    if (!validationSuccess) {
        await recordSecurityTransaction({ phase: 'EXECUTION', transactionId: tokenPayload.transactionId, status: 'REJECTED', reason: `Graduated control validation failed: ${validationError}` });
        return { statusCode: 403, body: JSON.stringify({ error: `Validation failed: ${validationError}` }) };
    }

    const result = await performPayPalOperationInternal(toolName, parameters, currentUserId);
    await recordSecurityTransaction({ phase: 'EXECUTION', transactionId: tokenPayload.transactionId, userId: currentUserId, toolName, dataClass: tokenPayload.dataClass, status: result.success ? 'SUCCESS' : 'FAILURE', result: result.data || result.error });
    return { statusCode: result.success ? 200 : 400, body: JSON.stringify(result) };

  } catch (error: any) {
    console.error("Sensitive Operation Execution Error:", error);
    const txId = tokenPayload ? tokenPayload.transactionId : 'N/A_TOKEN_ERROR';
    await recordSecurityTransaction({ phase: 'EXECUTION', transactionId: txId, event: 'ERROR', error: error.message });
    return { statusCode: 500, body: JSON.stringify({ error: 'Sensitive operation execution failed' }) };
  }
};
```
</details>

### 3. Update Client-Side Logic (Orchestrator Function)

The client (AI Assistant or its backend) uses an orchestrator function like `executeSecureTool` from the addendum to decide which flow to use.
<details>
<summary><strong>📋 TypeScript: Client-Side Orchestrator</strong></summary>

```typescript
// client-side-orchestrator.ts
// Assuming standardMCPClient is an existing client for Class 4-5 operations
// And TOOL_CLASSIFICATIONS is available (imported from tool-classifications.ts)

class PayPalZeroTrustClient {
    // This new client specifically handles the two-phase handshake for Class 1-3
    constructor(private extensionServiceUrl: string, private sessionTokenProvider: () => Promise<string>) {}

    async requestAuth(toolName: string, parameters: any): Promise<{ transactionId: string, ephemeralToken: string }> {
        const sessionToken = await this.sessionTokenProvider();
        const response = await fetch(`${this.extensionServiceUrl}/requestSensitiveOperationAuth`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sessionToken, toolName, parameters }),
        });
        if (!response.ok) throw new Error(`Auth request failed: ${await response.text()}`);
        return response.json();
    }

    async executeWithAuth(authDetails: { ephemeralToken: string }, toolName: string, parameters: any): Promise<any> {
        const sessionToken = await this.sessionTokenProvider();
        const response = await fetch(`${this.extensionServiceUrl}/executeSensitiveOperation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sessionToken, ephemeralToken: authDetails.ephemeralToken, toolName, parameters }),
        });
        if (!response.ok) throw new Error(`Execution with auth failed: ${await response.text()}`);
        return response.json();
    }
}

// Orchestrator function (as per Addendum)
async function executeSecureTool(
    toolName: string,
    parameters: any,
    standardMCPClient: any, // Existing client for Class 4-5
    zeroTrustClient: PayPalZeroTrustClient // New client for Class 1-3
): Promise<any> {
  const dataClass = TOOL_CLASSIFICATIONS[toolName];

  if (!dataClass) {
    throw new Error(`Unknown tool or classification for: ${toolName}`);
  }

  if (dataClass <= 3) { // Class 1-3: Use zero-trust extensions
    console.log(`[Client] Using Zero-Trust flow for ${toolName} (Class ${dataClass})`);
    const authDetails = await zeroTrustClient.requestAuth(toolName, parameters);
    return await zeroTrustClient.executeWithAuth(authDetails, toolName, parameters);
  } else { // Class 4-5: Use standard MCP 2.1
    console.log(`[Client] Using Standard MCP 2.1 flow for ${toolName} (Class ${dataClass})`);
    return await standardMCPClient.executeTool(toolName, parameters); // Assuming this method exists
  }
}
```
</details>

### 4. AI Assistant Integration

The AI assistant or its backend for frontend (BFF) would use `executeSecureTool`.
<details>
<summary><strong>📋 TypeScript: AI Assistant Integration Example</strong></summary>

```typescript
// ai-assistant-integration.ts
// --- Setup ---
const extensionServiceUrl = process.env.ZERO_TRUST_MCP_EXTENSION_URL!; // Ensure this is set
const sessionTokenProvider = async () => "user-session-jwt-token"; // Replace with actual token logic

// Mock for standard MCP client for Class 4-5 tools
const mockStandardMCPClient = {
  executeTool: async (toolName: string, parameters: any) => {
    console.log(`STANDARD MCP: Executing ${toolName}`, parameters);
    if (toolName === "get_exchange_rate") return { rate: 1.1, source: "Standard MCP" };
    if (toolName === "get_supported_currencies") return { currencies: ["USD", "EUR", "GBP"], source: "Standard MCP" };
    return { success: true, message: `${toolName} handled by standard MCP`, source: "Standard MCP" };
  }
};
const zeroTrustClient = new PayPalZeroTrustClient(extensionServiceUrl, sessionTokenProvider);

// --- Example Usage ---
async function processUserRequest(userIntent: string, userParams: any) {
  console.log(`Processing user request: ${userIntent} with params:`, userParams);
  try {
    const result = await executeSecureTool(userIntent, userParams, mockStandardMCPClient, zeroTrustClient);
    console.log("Operation Result:", result);
    // Return result to user or use in AI response
  } catch (error) {
    console.error("Failed to process user request for intent:", userIntent, error);
    // Handle error appropriately
  }
}

// Example calls:
// processUserRequest("send_money", { amount: 50, recipient: "friend@example.com", currency: "USD" });
// processUserRequest("get_exchange_rate", { from: "USD", to: "EUR" });
// processUserRequest("create_payment_method", { type: "CREDIT_CARD", details: "...", customerId: "cust_123" });
```
</details>

### 5. Deploy and Secure Services

-   Deploy the `zero-trust-mcp-extension-service` (handlers: `requestSensitiveOperationAuth`, `executeSensitiveOperation`).
-   Securely manage all credentials (PayPal API keys, Redis connection strings) using a secrets manager.
-   Enforce HTTPS, network segmentation (VPCs), and robust IAM policies.

### 6. Implement Comprehensive Audit Logging

Ensure `recordSecurityTransaction` in the Extension Service logs all relevant details for each phase of the sensitive operation lifecycle, including transaction IDs, user IDs, tool names, data classifications, parameter hashes, and outcomes.

## Testing Strategy

-   **Class 4-5 Tools**: Verify they continue to work through the `standardMCPClient` via `executeSecureTool`.
-   **Class 1-3 Tools**:
    -   Test successful two-phase handshake.
    -   Test token expiration (wait > TTL between auth and execute).
    -   Test parameter tampering (modify params between auth and execute).
    -   Test token replay (use same ephemeral token twice).
    -   Test failures in `callConfirmationAgent` for Class 1-2.
    -   Test failures in enhanced parameter validation for Class 3.
-   **Security Logs**: Verify audit logs capture all necessary details for each scenario.

```
