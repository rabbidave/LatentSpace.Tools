# MCP 2.1 Addendum: Zero-Trust Extensions for Sensitive Operations

**Version:** 1.0  
**Date:** 2025-05-22  
**Status:** Implementation Guidance

## Overview

While MCP 2.1 provides native authorization capabilities, sensitive financial operations require additional zero-trust security measures beyond standard bearer token patterns. This addendum defines custom extensions needed for Class 1-3 data operations.

## Data Classification Mapping

| Data Class | Description | Examples | Security Extensions Required |
|------------|-------------|----------|------------------------------|
| **Class 1: PII** | Most sensitive personal data | SSN, payment methods, account credentials | per-integration specifics |
| **Class 2: Sensitive Personal Data** | Financial transactions, personal details | Transaction history, refunds, account balance | Transaction-bound tokens + add'l |
| **Class 3: Confidential Personal Data** | Business-sensitive operations | Customer profiles, invoices, payment processing | Transaction-bound tokens + enhanced validation |
| **Class 4: Internal Data** | Standard business operations | Exchange rates, general account info | Standard MCP 2.1 authorization |
| **Class 5: Public Data** | Non-sensitive operations | Public API endpoints, documentation | No additional authorization |

## Required Custom Extensions

### 1. Transaction-Bound Ephemeral Tokens
**For:** Class 1-3 operations  
**Purpose:** Cryptographically bind tokens to specific operation parameters

```typescript
interface EphemeralTokenBinding {
  transactionId: string;
  toolName: string;
  paramsHash: string;  // SHA-256 of JSON.stringify(parameters)
  userId: string;
  dataClass: 1 | 2 | 3;
  expiry: number;      // 30-second TTL
}
```

### 2. Atomic Token Consumption
**For:** Class 1-3 operations  
**Purpose:** Prevent replay attacks through one-time-use validation

```typescript
// Redis-based atomic consumption
const validateAndConsume = async (ephemeralToken: string) => {
  return await redis.eval(`
    local token = redis.call('GET', KEYS[1])
    if token then
      redis.call('DEL', KEYS[1])
      return token
    else
      return nil
    end
  `, 1, `tx:${ephemeralToken}`);
};
```



## Implementation Architecture

### Dual Authorization Flows

**Class 4-5 Operations** (Internal/Public Data) use standard **single-phase MCP 2.1**:
```
┌─────────────────┐    Standard MCP 2.1    ┌──────────────────┐
│   AI Assistant  │◄──── Single Phase ─────┤ Standard MCP 2.1 │
│                 │      Bearer Token       │   Authorization  │
│ (Class 4-5 ops) │◄───────────────────────┤   (get_balance,  │
└─────────────────┘                        │   exchange_rate) │
                                           └──────────────────┘
```

**Class 1-3 Operations** (PII/Sensitive/Confidential) require **two-phase zero-trust**:
```
┌─────────────────┐                        ┌──────────────────┐
│   AI Assistant  │                        │ Standard MCP 2.1 │
│                 │◄─── Session Token ─────┤   Authorization  │
│ (Class 1-3 ops) │                        │                  │
└─────────┬───────┘                        └──────────────────┘
          │
          │ Sensitive Operations
          │ (send_money, refund, etc.)
          ▼
┌─────────────────┐    2-Phase Flow        ┌──────────────────┐
│ Enhanced Local  │◄─── Phase 1: Auth ────┤ Zero-Trust MCP   │
│ MCP Client      │◄─── Phase 2: Execute ──┤ Extension Service│
└─────────────────┘                        └────────┬─────────┘
                                                    │
                                          Class 1-2 Only
                                                    ▼
                                         ┌──────────────────┐
                                         │ Confirmation     │
                                         │ Agent Validator  │
                                         └──────────────────┘
```

## Financial API Tool Classification Examples

```typescript
const TOOL_CLASSIFICATIONS = {
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
  "get_customer_profile": 3,
  
  // Class 4: Internal Operations  
  "list_transactions": 4,
  "get_account_balance": 4,
  
  // Class 5: Public Data
  "get_exchange_rate": 5,
  "get_supported_currencies": 5
};
```

## Integration with MCP 2.1

Class 4-5 operations use standard MCP 2.1 authorization. Class 1-3 operations layer the zero-trust extensions on top:

```typescript
async function executeSecureTool(toolName: string, parameters: any) {
  const dataClass = TOOL_CLASSIFICATIONS[toolName];
  
  if (dataClass <= 3) {
    // Use zero-trust extensions
    return await executeWithZeroTrustHandshake(toolName, parameters, dataClass);
  } else {
    // Use standard MCP 2.1
    return await standardMCPClient.executeTool(toolName, parameters);
  }
}
```

## Security Benefits

- **Zero Standing Privileges**: Class 1-3 tokens are single-use and parameter-bound
- **Defense in Depth**: Multiple validation layers scale with data sensitivity  
- **Comprehensive Audit**: Every sensitive operation has unique transaction tracking
- **Standards Compliance**: Built on MCP 2.1 foundation with targeted security enhancements

## Implementation Priority

1. **Phase 1**: Implement Class 4-5 operations with standard MCP 2.1
2. **Phase 2**: Add transaction-bound tokens for Class 3 operations  
3. **Phase 3**: Integrate dual-agent validation for Class 1-2 operations
4. **Phase 4**: Full zero-trust pipeline with comprehensive audit logging

This approach provides maximum security for sensitive operations while maintaining developer productivity for routine tasks.
