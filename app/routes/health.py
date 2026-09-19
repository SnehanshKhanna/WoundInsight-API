from fastapi import APIRouter, Depends
from app.schemas.health import HealthResponse
from app.services.inference_service import inference_service
from app.db.database import check_db_health

router = APIRouter(tags=["Health & Status"])

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & Model Readiness Status"
)
async def get_health(db_status: bool = Depends(check_db_health)):
    """Returns runtime health status, compute device info, loaded checkpoint names, and database status."""
    return inference_service.get_status(db_connected=db_status)
