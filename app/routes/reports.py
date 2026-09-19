import logging
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import RedirectResponse
from app.services.storage_service import storage_service
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

    report_path = f"users/{current_user['id']}/wounds/{record['wound_id']}/analyses/{analysis_id}/report.png"
    
    try:
        signed_url = storage_service.get_signed_url(report_path)
        return RedirectResponse(url=signed_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    except Exception as e:
        logger.error(f"Failed to generate signed URL for report {analysis_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Diagnostic report for analysis '{analysis_id}' was not found."
        )
