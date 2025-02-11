from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
import json
from pydantic import BaseModel, validator
from typing import Dict, Any

app = FastAPI()

# Load the OpenAPI specification from the JSON file
with open("K8sforAIAPI.json", "r") as f:
    openapi_spec = json.load(f)

@app.get("/openapi.json")
async def get_openapi_json():
    return JSONResponse(openapi_spec)

# Data Classification Enum
class DataClassification(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, *args, **kwargs):
        classifications = ["Public", "Internal", "Confidential", "Restricted"]
        if v not in classifications:
            raise ValueError(f"Invalid data classification. Must be one of: {', '.join(classifications)}")
        return v

# Model Validation Request Model
class ModelValidationRequest(BaseModel):
    modelHash: str
    dataClassification: DataClassification
    validationParameters: Dict[str, Any] | None = None

# Health Check Config Model (using Dict for now, can create a Pydantic model later if needed)
class HealthCheckConfig(BaseModel):
    name: str
    targetMetric: str
    dataClassification: DataClassification
    quickAlertHeuristic: Dict[str, Any]
    description: str | None = None

# Batch Processing Config Model (using Dict for now, can create a Pydantic model later if needed)
class BatchProcessingConfig(BaseModel):
    modelId: str
    dataSource: str
    dataClassification: DataClassification
    processingParameters: Dict[str, Any] | None = None

# Mock endpoints based on the OpenAPI spec. These would need to be implemented with actual logic.
@app.post("/api/v1/config/healthchecks")
async def create_health_check_config(health_check_config: HealthCheckConfig):
    # Mock implementation: return the received config
    return health_check_config

@app.post("/api/v1/runtime/models/{modelId}/validate")
async def validate_model(modelId: str, request: ModelValidationRequest):
    # Mock implementation: return a fixed validation result
    return {
        "valid": True,
        "timestamp": "2024-01-01T00:00:00Z",
        "details": request.dict()
    }

@app.post("/api/v1/integrations/batch")
async def create_batch_processing_job(batch_processing_config: BatchProcessingConfig):
    # Mock implementation: return a pending batch job
    return {"jobId": "mock-job-123", "status": "Pending", "created": "2024-01-01T00:00:00Z"}

# Override the default OpenAPI schema to use the one from the file
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=openapi_spec["info"]["title"],
        version=openapi_spec["info"]["version"],
        description=openapi_spec["info"]["description"],
        routes=app.routes,
    )
    #  Remove the automatically generated schema, and use the one from the file.
    openapi_schema["components"]["schemas"] = openapi_spec["components"]["schemas"]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi