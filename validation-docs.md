# AI Control Plane API Documentation

## Overview

This API provides a system of record for configuration changes across AI deployments. It supports:
- Schema evolution
- Role-based access control
- Audit trail of all changes
- Backwards compatibility

## Authentication

All requests require a bearer token specific to your role:
- Enterprise Infrastructure: `enterprise-token`
- LOB IT: `lob-token`
- Application Teams: `app-token`

## 1. Enterprise Infrastructure Path

Enterprise teams manage core infrastructure configurations.

### Adding GPU Configuration

```bash
curl -X POST http://localhost:8000/api/v1/config/schema \
  -H "Authorization: Bearer enterprise-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-inference",
    "targetMetric": "gpu_utilization",
    "dataClassification": "Restricted",
    "quickAlertHeuristic": {
      "threshold": 0.85,
      "window": "5m"
    },
    "gpu_config": {
      "type": "A100",
      "memory": "80GB"
    },
    "reason": "Production scale-up for high-throughput inference"
  }'
```

Response:
```json
{
  "status": "success",
  "message": "Enterprise schema updated",
  "updated_schema": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "targetMetric": {"type": "string"},
      "dataClassification": {
        "type": "string",
        "enum": ["Public", "Internal", "Confidential", "Restricted"]
      },
      "quickAlertHeuristic": {
        "type": "object",
        "properties": {
          "threshold": {"type": "number"},
          "window": {"type": "string"}
        }
      },
      "gpu_config": {
        "type": "object",
        "properties": {
          "type": {"type": "string"},
          "memory": {"type": "string"}
        }
      }
    }
  }
}
```

## 2. LOB IT Path

LOB IT teams manage runtime configurations and model validation.

### Adding Operational Metric

```bash
curl -X POST http://localhost:8000/api/v1/runtime/schema \
  -H "Authorization: Bearer lob-token" \
  -H "Content-Type: application/json" \
  -d '{
    "modelVersion": "v2.1.0",
    "validationParameters": {
      "minBatchSize": 32,
      "maxLatencyMs": 100
    },
    "operational_metric": "inference_throughput",
    "reason": "Adding throughput monitoring for batch processing"
  }'
```

## 3. Application Team Path

Application teams manage custom integrations and business metrics.

### Adding Custom Metrics

```bash
curl -X POST http://localhost:8000/api/v1/integrations/schema \
  -H "Authorization: Bearer app-token" \
  -H "Content-Type: application/json" \
  -d '{
    "customThresholds": {
      "accuracy": 0.95,
      "latency_p99": 250
    },
    "businessMetric": "revenue",
    "callbackUrl": "https://app-endpoint/callback",
    "reason": "Adding latency monitoring for SLA compliance"
  }'
```

## 4. Audit Trail

View the history of changes for any schema type:

```bash
curl -X GET http://localhost:8000/api/v1/audit/enterprise \
  -H "Authorization: Bearer enterprise-token"
```

Response:
```json
{
  "changes": [
    {
      "timestamp": "2024-02-13T14:30:00Z",
      "schema_type": "enterprise",
      "author": "enterprise-infrastructure",
      "change": {
        "name": "gpu-inference",
        "gpu_config": {
          "type": "A100",
          "memory": "80GB"
        }
      }
    }
  ]
}
```

## Schema Evolution Rules

1. Required Fields:
   - Enterprise: name, targetMetric, dataClassification, quickAlertHeuristic
   - LOB IT: modelVersion, validationParameters
   - App Teams: customThresholds, businessMetric

2. All changes require:
   - Reason for change
   - Appropriate role authorization
   - Valid schema according to role

3. New fields:
   - Must match their declared type
   - Cannot override existing fields
   - Are automatically added to schema

## Error Handling

Common error responses:

```json
{
  "detail": "Enterprise access required"
}
```

```json
{
  "detail": "Invalid token"
}
```

```json
{
  "detail": "Required field 'reason' missing"
}
```

## Best Practices

1. Always include a clear reason for changes
2. Use descriptive names for new fields
3. Document changes in your change management system
4. Verify changes in audit log
5. Test backwards compatibility

## Understanding Configuration vs. State

These configurations are a system of record and NOT:
- Real-time system state
- Performance metrics
- Resource allocation
- Runtime parameters

Example of proper usage:

```bash
# Record approved GPU configuration
curl -X POST http://localhost:8000/api/v1/config/schema \
  -H "Authorization: Bearer enterprise-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-inference",
    "targetMetric": "gpu_utilization",
    "dataClassification": "Restricted",
    "quickAlertHeuristic": {
      "threshold": 0.85,
      "window": "5m"
    },
    "gpu_config": {
      "type": "A100",
      "memory": "80GB"
    },
    "reason": "Establishing baseline for production GPU inference"
  }'
```

This records that:
- A100 GPUs are approved for use
- 80GB memory configuration is standard
- Utilization should be monitored
- Changes were properly authorized

The actual runtime allocation and usage are managed separately by your orchestration system.