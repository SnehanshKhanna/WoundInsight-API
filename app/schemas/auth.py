from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class UserRegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password (min 6 characters)")
    name: Optional[str] = Field(None, description="User's full or preferred name")

class UserLoginRequest(BaseModel):
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")

class UserResponse(BaseModel):
    id: str = Field(..., description="Unique user identifier (UUID)")
    email: str = Field(..., description="User email address")
    name: Optional[str] = Field(None, description="User's full or preferred name")
    created_at: str = Field(..., description="Account creation ISO timestamp")

class AuthTokenResponse(BaseModel):
    access_token: str = Field(..., description="Stateless JWT bearer token")
    token_type: str = Field(default="bearer", description="Token type")
    user: UserResponse = Field(..., description="Current authenticated user details")
