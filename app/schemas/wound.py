from typing import Optional, List
from pydantic import BaseModel, Field

class WoundCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Wound name or label (e.g., 'Left Heel Ulcer')")
    location: Optional[str] = Field(None, max_length=100, description="Anatomical location (e.g., 'Left Lower Extremity - Plantar')")

class WoundResponse(BaseModel):
    id: str = Field(..., description="Unique wound identifier (UUID)")
    user_id: str = Field(..., description="Owner user ID")
    name: str = Field(..., description="Wound name or label")
    location: Optional[str] = Field(None, description="Anatomical location")
    created_at: str = Field(..., description="Creation ISO timestamp")

class WoundListResponse(BaseModel):
    wounds: List[WoundResponse] = Field(..., description="List of user's tracked wounds")
    total: int = Field(..., description="Total count of wounds")
