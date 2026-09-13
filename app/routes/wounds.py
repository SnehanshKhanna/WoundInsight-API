import uuid
import logging
from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, HTTPException, status, Depends

from app.schemas.wound import WoundCreateRequest, WoundResponse, WoundListResponse
from app.schemas.error import ErrorResponse
from app.models.database_models import WoundModel
from app.db.repositories import WoundRepository
from app.utils.auth import get_current_user

logger = logging.getLogger("wound_api.routes.wounds")
router = APIRouter(prefix="/api/v1/wounds", tags=["Wounds"])

@router.post(
    "",
    response_model=WoundResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a Tracked Wound Profile"
)
async def create_wound(
    req: WoundCreateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Creates a new wound tracking profile owned by the authenticated user."""
    wound_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()

    wound = WoundModel(
        id=wound_id,
        user_id=current_user["id"],
        name=req.name.strip(),
        location=req.location.strip() if req.location else None,
        created_at=now_iso
    )
    WoundRepository.create_wound(wound)

    return WoundResponse(
        id=wound.id,
        user_id=wound.user_id,
        name=wound.name,
        location=wound.location,
        created_at=wound.created_at
    )

@router.get(
    "",
    response_model=WoundListResponse,
    summary="List Wounds Owned by Authenticated User"
)
async def list_wounds(current_user: dict = Depends(get_current_user)):
    """Retrieves all wound profiles created by the current authenticated user."""
    wounds = WoundRepository.list_by_user(current_user["id"])
    return WoundListResponse(
        wounds=[
            WoundResponse(
                id=w.id,
                user_id=w.user_id,
                name=w.name,
                location=w.location,
                created_at=w.created_at
            )
            for w in wounds
        ],
        total=len(wounds)
    )

@router.get(
    "/{wound_id}",
    response_model=WoundResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Wound profile not found or not owned by user"}
    },
    summary="Get Wound Profile by ID"
)
async def get_wound_by_id(
    wound_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Retrieves a single wound profile, strictly verifying ownership by the authenticated user."""
    wound = WoundRepository.get_by_id(wound_id)
    if not wound or wound.user_id != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Wound profile with ID '{wound_id}' was not found."
        )

    return WoundResponse(
        id=wound.id,
        user_id=wound.user_id,
        name=wound.name,
        location=wound.location,
        created_at=wound.created_at
    )
