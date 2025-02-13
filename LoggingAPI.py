from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import jsonschema
from jsonschema import validate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Control Plane API")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# File paths for schemas and audit logs
SCHEMA_DIR = "schemas"
AUDIT_DIR = "audit"
os.makedirs(SCHEMA_DIR, exist_ok=True)
os.makedirs(AUDIT_DIR, exist_ok=True)

# Base schemas with evolution support
base_schemas = {
    "enterprise": {
        "type": "object",
        "required": ["name", "targetMetric", "dataClassification", "quickAlertHeuristic"],
        "properties": {
            "name": {"type": "string"},
            "targetMetric": {"type": "string"},
            "dataClassification": {
                "type": "string",
                "enum": ["Public", "Internal", "Confidential", "Restricted"]
            },
            "quickAlertHeuristic": {
                "type": "object",
                "required": ["threshold", "window"],
                "properties": {
                    "threshold": {"type": "number"},
                    "window": {"type": "string"}
                }
            },
            "author": {"type": "string"},
            "reason": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"}
        },
        "additionalProperties": True
    },
    "runtime": {
        "type": "object",
        "required": ["modelVersion", "validationParameters"],
        "properties": {
            "modelVersion": {"type": "string"},
            "validationParameters": {
                "type": "object",
                "additionalProperties": True
            },
            "author": {"type": "string"},
            "reason": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"}
        },
        "additionalProperties": True
    },
    "integration": {
        "type": "object",
        "required": ["customThresholds", "businessMetric"],
        "properties": {
            "customThresholds": {
                "type": "object",
                "additionalProperties": {"type": "number"}
            },
            "businessMetric": {"type": "string"},
            "author": {"type": "string"},
            "reason": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"}
        },
        "additionalProperties": True
    }
}

def log_audit_event(schema_type: str, change: Dict[str, Any], author: str):
    """Log schema changes to audit file"""
    timestamp = datetime.utcnow().isoformat()
    audit_entry = {
        "timestamp": timestamp,
        "schema_type": schema_type,
        "author": author,
        "change": change
    }
    
    audit_file = os.path.join(AUDIT_DIR, f"{schema_type}_audit.jsonl")
    with open(audit_file, "a") as f:
        f.write(json.dumps(audit_entry) + "\n")

def get_current_role(token: str = Depends(oauth2_scheme)) -> str:
    role_tokens = {
        "enterprise-token": "enterprise",
        "lob-token": "lob-it",
        "app-token": "app-team"
    }
    if token not in role_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    return role_tokens[token]

def validate_and_update_schema(schema_type: str, new_fields: Dict[str, Any], author: str) -> dict:
    """Validate new fields and update schema with audit trail"""
    current_schema = load_schema(schema_type)
    
    # Add audit fields
    new_fields["timestamp"] = datetime.utcnow().isoformat()
    new_fields["author"] = author
    
    try:
        validate(instance=new_fields, schema=current_schema)
    except jsonschema.exceptions.ValidationError as e:
        # Handle schema evolution
        if "additionalProperties" in current_schema and current_schema["additionalProperties"]:
            properties = current_schema["properties"]
            for field, value in new_fields.items():
                if field not in properties:
                    properties[field] = {"type": type(value).__name__}
            
            try:
                validate(instance=new_fields, schema=current_schema)
            except jsonschema.exceptions.ValidationError as e:
                raise HTTPException(status_code=400, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    
    log_audit_event(schema_type, new_fields, author)
    return current_schema

@app.post("/api/v1/config/schema")
async def update_enterprise_schema(
    new_fields: Dict[str, Any],
    role: str = Depends(get_current_role)
):
    """Update enterprise infrastructure schema"""
    if role != "enterprise":
        raise HTTPException(status_code=403, detail="Enterprise access required")
    
    updated_schema = validate_and_update_schema("enterprise", new_fields, f"enterprise-{role}")
    save_schema("enterprise", updated_schema)
    
    return {
        "status": "success",
        "message": "Enterprise schema updated",
        "updated_schema": updated_schema
    }

@app.post("/api/v1/runtime/schema")
async def update_runtime_schema(
    new_fields: Dict[str, Any],
    role: str = Depends(get_current_role)
):
    """Update LOB IT runtime schema"""
    if role != "lob-it":
        raise HTTPException(status_code=403, detail="LOB IT access required")
    
    updated_schema = validate_and_update_schema("runtime", new_fields, f"lob-{role}")
    save_schema("runtime", updated_schema)
    
    return {
        "status": "success",
        "message": "Runtime schema updated",
        "updated_schema": updated_schema
    }

@app.post("/api/v1/integrations/schema")
async def update_integration_schema(
    new_fields: Dict[str, Any],
    role: str = Depends(get_current_role)
):
    """Update application team integration schema"""
    if role != "app-team":
        raise HTTPException(status_code=403, detail="App team access required")
    
    updated_schema = validate_and_update_schema("integration", new_fields, f"app-{role}")
    save_schema("integration", updated_schema)
    
    return {
        "status": "success",
        "message": "Integration schema updated",
        "updated_schema": updated_schema
    }

@app.get("/api/v1/audit/{schema_type}")
async def get_audit_log(
    schema_type: str,
    role: str = Depends(get_current_role)
):
    """Retrieve audit log for schema changes"""
    audit_file = os.path.join(AUDIT_DIR, f"{schema_type}_audit.jsonl")
    if not os.path.exists(audit_file):
        return {"changes": []}
    
    with open(audit_file, "r") as f:
        changes = [json.loads(line) for line in f]
    
    return {"changes": changes}

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_schemas()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)