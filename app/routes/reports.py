import logging
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import FileResponse
from app.services.report_service import report_service
from app.db.repositories import AnalysisRepository
from app.utils.auth import get_current_user

logger = logging.getLogger("wound_api.routes.reports")
router = APIRouter(tags=["Reports"])

@router.get(
    "/api/v1/analyses/{analysis_id}/report",
    summary="Retrieve Diagnostic Report Image",
    responses={
        200: {"content": {"image/png": {}}, "description": "Diagnostic Report Figure PNG"},
        404: {"description": "Report image not found or unauthorized"}
    }
)
async def get_analysis_report(
    analysis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Retrieves the generated 6-panel diagnostic visual report PNG, verifying ownership."""
    record = AnalysisRepository.get_by_id(analysis_id)
    if not record or record["user_id"] != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Diagnostic report for analysis '{analysis_id}' was not found."
        )

    report_file = report_service.get_report_file(analysis_id)
    if not report_file or not report_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Diagnostic report for analysis '{analysis_id}' was not found."
        )
    return FileResponse(path=str(report_file), media_type="image/png", filename=f"report_{analysis_id}.png")
