from typing import Dict, Optional
from pydantic import BaseModel, Field

class RootResponse(BaseModel):
    service: str = Field(..., example="WoundInsight Clinical AI API")
    status: str = Field(..., example="running")
    version: str = Field(default="1.0.0", example="1.0.0")
    docs_url: str = Field(default="/docs", example="/docs")

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    device: str = Field(..., example="cuda:0")
    gpu_available: bool = Field(..., example=True)
    gpu_name: Optional[str] = Field(None, example="NVIDIA GeForce RTX 4060 Laptop GPU")
    models_loaded: bool = Field(..., example=True)
    checkpoints: Dict[str, str] = Field(..., description="Active checkpoint basenames")
    database_connected: bool = Field(..., description="Persistence database connectivity status")
