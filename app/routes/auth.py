import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status, Depends

from app.schemas.auth import UserRegisterRequest, UserLoginRequest, AuthTokenResponse, UserResponse
from app.schemas.error import ErrorResponse
from app.models.database_models import UserModel
from app.db.repositories import UserRepository
from app.utils.auth import hash_password, verify_password, create_access_token, get_current_user

logger = logging.getLogger("wound_api.routes.auth")
router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

@router.post(
    "/register",
    response_model=AuthTokenResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Email already registered or validation error"}
    },
    summary="Register a New User Account"
)
async def register(req: UserRegisterRequest):
    """Registers a new user, hashes password securely via bcrypt, and issues a JWT bearer token."""
    existing_user = UserRepository.get_by_email(req.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists."
        )

    user_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()
    hashed_pw = hash_password(req.password)

    user_model = UserModel(
        id=user_id,
        email=req.email.lower().strip(),
        password_hash=hashed_pw,
        name=req.name.strip() if req.name else None,
        created_at=now_iso
    )
    UserRepository.create_user(user_model)

    # Generate JWT bearer token
    token = create_access_token({"sub": user_id, "email": user_model.email})

    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse(
            id=user_model.id,
            email=user_model.email,
            name=user_model.name,
            created_at=user_model.created_at
        )
    )

@router.post(
    "/login",
    response_model=AuthTokenResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid email or password"}
    },
    summary="Authenticate User and Obtain JWT Token"
)
async def login(req: UserLoginRequest):
    """Authenticates credentials against bcrypt hash and returns a signed JWT bearer token."""
    user = UserRepository.get_by_email(req.email)
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password. Please verify your credentials.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = create_access_token({"sub": user.id, "email": user.email})

    return AuthTokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            created_at=user.created_at
        )
    )

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get Authenticated User Profile"
)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """Returns profile information for the bearer token's authenticated user."""
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        name=current_user.get("name"),
        created_at=current_user["created_at"]
    )
