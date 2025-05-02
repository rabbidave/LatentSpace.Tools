# How-To Guide: Adding MCP Handshake Architecture to PayPal Agent Toolkit

This guide explains how to upgrade the existing PayPal MCP server from the [agent-toolkit](https://github.com/paypal/agent-toolkit) to incorporate the MCP Handshake Architecture for enhanced security through transaction-specific authorization.

## Prerequisites

- Working installation of the PayPal agent-toolkit
- Understanding of the MCP Handshake Architecture (as defined in MCPHandshake.md)
- Access to deploy a Remote MCP Service
- A secure state store (Redis or DynamoDB)

## Architecture Overview

The MCP Handshake Architecture adds a security layer between your AI assistant and the PayPal API, implementing:
- Session token authentication
- Ephemeral transaction-specific tokens
- Tiered sensitivity validation
- Comprehensive audit logging

## Implementation Steps

### 1. Define Tool Sensitivity Classifications

First, classify PayPal API operations by sensitivity level:

```typescript
// Define tool sensitivity mapping
const TOOL_SENSITIVITY: Record<string, string> = {
  // Public Tools (no authentication required)
  "get_exchange_rate": "PUBLIC",
  
  // Internal Tools (require public key verification)
  "get_customer_profile": "INTERNAL",
  "list_transactions": "INTERNAL",
  
  // Confidential Tools (require additional validation)
  "create_invoice": "CONFIDENTIAL",
  "process_payment": "CONFIDENTIAL",
  
  // Restricted Tools (require confirmation agent)
  "refund_transaction": "RESTRICTED",
  "create_payment_method": "RESTRICTED",
  "send_money": "RESTRICTED"
};
```

### 2. Create the Remote MCP Service

Build a serverless function or API that acts as the mediator:

```typescript
// remote-mcp-service.ts
import { APIGatewayProxyHandler } from 'aws-lambda';
import { v4 as uuidv4 } from 'uuid';
import * as crypto from 'crypto';
import Redis from 'ioredis';

const redis = new Redis(process.env.REDIS_URL);
const EPHEMERAL_TOKEN_TTL = 30; // seconds

export const authorize: APIGatewayProxyHandler = async (event) => {
  const { sessionToken, tool, parameters, metadata } = JSON.parse(event.body || '{}');
  
  // Validate session token
  const userId = await validateSessionToken(sessionToken);
  if (!userId) {
    return { statusCode: 401, body: JSON.stringify({ error: 'Invalid session' }) };
  }
  
  // Generate transaction ID and ephemeral token
  const transactionId = uuidv4();
  const ephemeralToken = crypto.randomBytes(32).toString('hex');
  const paramsHash = crypto.createHash('sha256').update(JSON.stringify(parameters)).digest('hex');
  
  // Store transaction state
  await redis.setex(
    `tx:${ephemeralToken}`,
    EPHEMERAL_TOKEN_TTL,
    JSON.stringify({
      transactionId,
      userId,
      tool,
      paramsHash,
      status: 'pending',
      sensitivity: TOOL_SENSITIVITY[tool] || 'INTERNAL',
      createdAt: Date.now()
    })
  );
  
  return {
    statusCode: 200,
    body: JSON.stringify({ transactionId, ephemeralToken })
  };
};

export const execute: APIGatewayProxyHandler = async (event) => {
  const { sessionToken, ephemeralToken, tool, parameters } = JSON.parse(event.body || '{}');
  
  // Validate session token again
  const userId = await validateSessionToken(sessionToken);
  if (!userId) {
    return { statusCode: 401, body: JSON.stringify({ error: 'Invalid session' }) };
  }
  
  // Retrieve and validate ephemeral token
  const txData = await redis.get(`tx:${ephemeralToken}`);
  if (!txData) {
    return { statusCode: 403, body: JSON.stringify({ error: 'Invalid or expired token' }) };
  }
  
  const tx = JSON.parse(txData);
  
  // Validate transaction matches
  const currentParamsHash = crypto.createHash('sha256').update(JSON.stringify(parameters)).digest('hex');
  if (tx.userId !== userId || tx.tool !== tool || tx.paramsHash !== currentParamsHash) {
    return { statusCode: 403, body: JSON.stringify({ error: 'Transaction mismatch' }) };
  }
  
  // Delete ephemeral token (atomic consumption)
  await redis.del(`tx:${ephemeralToken}`);
  
  // Apply sensitivity-based validation
  const validationResult = await validateBySensitivity(tx.sensitivity, tool, parameters);
  if (!validationResult.success) {
    return { statusCode: 403, body: JSON.stringify({ error: validationResult.error }) };
  }
  
  // Execute PayPal API call
  const result = await executePayPalOperation(tool, parameters);
  
  // Log audit trail
  await logAuditTrail({
    transactionId: tx.transactionId,
    userId,
    tool,
    sensitivity: tx.sensitivity,
    status: result.success ? 'completed' : 'failed',
    result: result.success ? 'success' : result.error
  });
  
  return {
    statusCode: result.success ? 200 : 400,
    body: JSON.stringify(result)
  };
};
```

### 3. Implement Sensitivity-Based Validation

```typescript
async function validateBySensitivity(sensitivity: string, tool: string, parameters: any): Promise<{success: boolean, error?: string}> {
  switch (sensitivity) {
    case 'PUBLIC':
      return { success: true };
      
    case 'INTERNAL':
      // Verify public key signature
      if (!verifyPublicKeySignature(parameters)) {
        return { success: false, error: 'Invalid signature' };
      }
      return { success: true };
      
    case 'CONFIDENTIAL':
      // Regex and AST validation
      if (!validateRegexPatterns(tool, parameters)) {
        return { success: false, error: 'Parameters failed regex validation' };
      }
      if (!validateASTStructure(tool, parameters)) {
        return { success: false, error: 'Invalid parameter structure' };
      }
      return { success: true };
      
    case 'RESTRICTED':
      // Use confirmation agent (similar to agent(s).py dual-agent system)
      const confirmation = await getConfirmationAgentApproval(tool, parameters);
      if (!confirmation.approved) {
        return { success: false, error: `Denied by confirmation agent: ${confirmation.reason}` };
      }
      return { success: true };
      
    default:
      return { success: false, error: 'Unknown sensitivity level' };
  }
}
```

### 4. Update the Local MCP Client

Modify the PayPal agent-toolkit to use the handshake protocol:

```typescript
// paypal-mcp-client.ts
import { Tool } from '@modelcontextprotocol/sdk/types.js';

class PayPalHandshakeMCPClient {
  private remoteServiceUrl: string;
  private sessionToken: string;
  
  constructor(remoteServiceUrl: string, sessionToken: string) {
    this.remoteServiceUrl = remoteServiceUrl;
    this.sessionToken = sessionToken;
  }
  
  async executeWithHandshake(toolName: string, parameters: any): Promise<any> {
    // Phase 1: Authorization
    const authResponse = await fetch(`${this.remoteServiceUrl}/authorize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionToken: this.sessionToken,
        tool: toolName,
        parameters,
        metadata: { sensitivity: TOOL_SENSITIVITY[toolName] }
      })
    });
    
    if (!authResponse.ok) {
      throw new Error(`Authorization failed: ${await authResponse.text()}`);
    }
    
    const { transactionId, ephemeralToken } = await authResponse.json();
    
    // Phase 2: Execution
    const executeResponse = await fetch(`${this.remoteServiceUrl}/execute`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'X-Ephemeral-Token': ephemeralToken
      },
      body: JSON.stringify({
        sessionToken: this.sessionToken,
        ephemeralToken,
        tool: toolName,
        parameters
      })
    });
    
    if (!executeResponse.ok) {
      throw new Error(`Execution failed: ${await executeResponse.text()}`);
    }
    
    return await executeResponse.json();
  }
}

// Update existing PayPal tools to use handshake
function createSecurePayPalTools(client: PayPalHandshakeMCPClient): Tool[] {
  const originalTools = createPayPalTools();
  
  return originalTools.map(tool => ({
    ...tool,
    inputSchema: {
      ...tool.inputSchema,
      required: [...(tool.inputSchema.required || []), 'sensitivity']
    },
    handler: async (params: any) => {
      return await client.executeWithHandshake(tool.name, params);
    }
  }));
}
```

### 5. Deploy the Remote MCP Service

Deploy the Remote MCP Service as a serverless function:

```yaml
# serverless.yml
service: paypal-mcp-handshake

provider:
  name: aws
  runtime: nodejs18.x
  environment:
    REDIS_URL: ${env:REDIS_URL}
    PAYPAL_CLIENT_ID: ${env:PAYPAL_CLIENT_ID}
    PAYPAL_CLIENT_SECRET: ${env:PAYPAL_CLIENT_SECRET}

functions:
  authorize:
    handler: handler.authorize
    events:
      - http:
          path: /authorize
          method: post
  
  execute:
    handler: handler.execute
    events:
      - http:
          path: /execute
          method: post
```

### 6. Update AI Assistant Integration

Update your AI assistant to use the new handshake client:

```typescript
// app.ts
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { PayPalHandshakeMCPClient } from './paypal-mcp-client.js';

const server = new Server({
  name: 'secure-paypal-server',
  version: '2.0.0',
});

// Initialize with user session
const sessionToken = await getUserSessionToken(); // From your auth system
const handshakeClient = new PayPalHandshakeMCPClient(
  process.env.REMOTE_MCP_SERVICE_URL,
  sessionToken
);

// Register secure tools
const secureTools = createSecurePayPalTools(handshakeClient);
secureTools.forEach(tool => server.addTool(tool));
```

### 7. Implement Audit Logging

```typescript
async function logAuditTrail(data: {
  transactionId: string;
  userId: string;
  tool: string;
  sensitivity: string;
  status: string;
  result: string;
}) {
  // Log to your audit system
  await axios.post(process.env.AUDIT_LOG_ENDPOINT, {
    ...data,
    timestamp: new Date().toISOString(),
    serviceVersion: '2.0.0'
  });
}
```

## Testing the Implementation

1. **Test Authorization Flow**:
```typescript
// test-handshake.ts
const response = await handshakeClient.executeWithHandshake('get_customer_profile', {
  customerId: '12345'
});
console.log('Customer profile:', response);
```

2. **Test Restricted Operations**:
```typescript
// This should trigger confirmation agent
const refundResponse = await handshakeClient.executeWithHandshake('refund_transaction', {
  transactionId: 'TX123',
  amount: 100.00
});
```

3. **Test Token Expiration**:
```typescript
// Wait > 30 seconds between auth and execute to test expiration
const { ephemeralToken } = await authorize(...);
await new Promise(resolve => setTimeout(resolve, 31000));
const result = await execute(...); // Should fail
```

## Security Considerations

1. **Secure the Remote MCP Service**:
   - Deploy in a secure VPC
   - Use HTTPS only
   - Implement rate limiting
   - Monitor for anomalies

2. **Protect Sensitive Credentials**:
   - Store PayPal API credentials in a secrets manager
   - Rotate credentials regularly
   - Use IAM roles for service access

3. **Audit Everything**:
   - Log all authorization requests
   - Log all execution attempts
   - Monitor for unusual patterns

## Migration Checklist

- [ ] Define sensitivity tiers for all PayPal tools
- [ ] Implement Remote MCP Service
- [ ] Set up secure state store (Redis/DynamoDB)
- [ ] Update Local MCP Client library
- [ ] Deploy Remote MCP Service
- [ ] Update AI assistant integration
- [ ] Implement comprehensive logging
- [ ] Test all tools with new handshake flow
- [ ] Update documentation
- [ ] Train team on new security model

## Troubleshooting

### Common Issues

1. **Token Expiration**: If operations fail with "Invalid or expired token", increase `EPHEMERAL_TOKEN_TTL`
2. **Transaction Mismatch**: Ensure parameters are serialized consistently
3. **Validation Failures**: Check sensitivity tier configuration and validation logic

### Monitoring

Set up alerts for:
- Failed authorization attempts
- Expired token usage
- Abnormal request patterns
- High-volume operations

## Complete Example: Refund Transaction Flow

Here's a complete example showing how a sensitive operation flows through the handshake architecture:

```typescript
// 1. Client initiates refund
const refundParams = {
  transactionId: 'TX123456',
  amount: 50.00,
  reason: 'Customer request'
};

// 2. Handshake client requests authorization
const authRequest = await fetch('https://api.example.com/authorize', {
  method: 'POST',
  body: JSON.stringify({
    sessionToken: userSessionToken,
    tool: 'refund_transaction',
    parameters: refundParams,
    metadata: { sensitivity: 'RESTRICTED' }
  })
});

const { transactionId, ephemeralToken } = await authRequest.json();

// 3. Execute with ephemeral token
const executeRequest = await fetch('https://api.example.com/execute', {
  method: 'POST',
  body: JSON.stringify({
    sessionToken: userSessionToken,
    ephemeralToken,
    tool: 'refund_transaction',
    parameters: refundParams
  })
});

// 4. Remote service:
//    - Validates tokens
//    - Runs confirmation agent check
//    - Executes PayPal refund
//    - Logs audit trail

const result = await executeRequest.json();
console.log('Refund result:', result);
```
