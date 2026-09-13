from pydantic import BaseModel, Field

class ErrorResponse(BaseModel):
    status: str = Field(default="error", example="error")
    error: str = Field(..., description="Human-readable error description")
    error_code: str = Field(..., description="Machine-readable error classification code")
