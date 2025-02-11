# Per-Integration AI Control Plane API - Validation Details

## 1. Role-Based Field Ownership 

### 1.1 Enterprise Infrastructure (Required Fields)
Enterprise Infrastructure owns the foundational validation requirements - analogous to building code inspectors ensuring basic safety standards are met.

**Health Check Definition Example:**
```json
{
  "name": "model-inference-health",  // Enterprise-defined name schema
  "targetMetric": "inference_latency_ms",  // Standard enterprise metric
  "dataClassification": "Restricted",  // PICR classification
  "quickAlertHeuristic": {
    "threshold": 0.85,  // Dot product similarity threshold
    "window": "5m"      // Rolling window for evaluation
  }
}
```

**Key Points:**
- Enterprise Infrastructure validates the presence and format of required fields
- They don't prescribe specific values (except for classification levels)
- Think of it as "the outlet must be GFCI" not "what you plug into it"

### 1.2 LOB IT/Cluster Operators (Runtime Fields)
LOB IT manages model deployment and runtime configuration, interfacing between Enterprise requirements and business needs.

**Model Version Control Example:**
```json
{
  // Enterprise-required fields above
  "modelVersion": "v2.1.0",        // LOB IT controlled
  "validationParameters": {
    "minBatchSize": 32,            // LOB IT configured
    "maxLatencyMs": 100,           // LOB IT configured
    "alertContacts": ["team-ml-ops@company.com"]
  }
}
```

**Key Points:**
- LOB IT can add fields but cannot remove Enterprise requirements
- They handle promotion through environments
- Model version changes trigger Enterprise notifications but don't require approval

### 1.3 Application Teams (Integration Fields)
Application teams implement specific integrations and custom metrics for their use cases.

**Custom Integration Example:**
```json
{
  // Enterprise & LOB fields above
  "processingParameters": {
    "customThresholds": {
      "accuracy": 0.95,            // App team defined
      "businessMetric": "revenue"  // App team defined
    },
    "callbackUrl": "https://app-endpoint/callback"
  }
}
```

## 2. Quick Alert Heuristic Configuration

### 2.1 Enterprise Requirements
Enterprise Infrastructure defines the structure and minimum requirements for alerts:

```json
{
  "quickAlertHeuristic": {
    "metric": "vector_similarity",
    "algorithm": "dot_product",     // Enterprise standardized on dot product
    "threshold": 0.85,
    "window": "5m",
    "minimumSamples": 100
  }
}
```

### 2.2 LOB IT Implementation
LOB IT configures specific thresholds and windows based on their model characteristics:

```json
{
  "quickAlertHeuristic": {
    "metric": "vector_similarity",
    "algorithm": "dot_product",
    "threshold": 0.92,              // Stricter for this model
    "window": "15m",                // Longer window for batch processing
    "minimumSamples": 250,          // Higher sample requirement
    "alertPolicy": {
      "cooldown": "1h",
      "escalation": "tier2"
    }
  }
}
```

### 2.3 Application Team Usage
Application teams consume alerts and can add custom handlers:

```json
{
  "quickAlertHeuristic": {
    // Enterprise & LOB fields above
    "customHandlers": {
      "onAlert": "https://app-team-endpoint/alert",
      "metricStore": "datadog",
      "dashboardId": "ml-monitoring-01"
    }
  }
}
```

## 3. Validation Chain Summary

The validation chain enforces separation of duties:

1. Enterprise Infrastructure validates:
   - Presence of required fields
   - PICR classification enforcement
   - Base metric collection
   - Alert heuristic structure

2. LOB IT validates:
   - Model version control
   - Runtime configurations
   - Environment-specific parameters

3. Application Teams validate:
   - Custom metrics
   - Business-specific thresholds
   - Integration endpoints

Think of it as a building inspection:
- Enterprise Infrastructure ensures the building meets code
- LOB IT manages tenant improvements
- Application Teams handle furniture and fixtures

## 4. Security Model

The separation of duties is enforced through role-based access control:

```json
{
  "roles": {
    "enterprise": {
      "paths": ["/api/v1/config/*"],
      "operations": ["validate", "read"]
    },
    "lob-it": {
      "paths": ["/api/v1/runtime/*"],
      "operations": ["deploy", "configure"]
    },
    "app-team": {
      "paths": ["/api/v1/integrations/*"],
      "operations": ["integrate", "monitor"]
    }
  }
}
```

Each role's scope is limited to their respective areas of responsibility, preventing cross-boundary modifications while enabling collaboration.