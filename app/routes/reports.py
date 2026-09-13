import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from app.services.report_service import report_service

logger = logging.getLogger("wound_api.routes.reports")
router = APIRouter(tags=["Reports"])

@router.get(
    "/api/v1/analyses/{analysis_id}/report",
    summary="Retrieve Diagnostic Report Image",
    responses={
        200: {"content": {"image/png": {}}, "description": "Diagnostic Report Figure PNG"},
        404: {"description": "Report image not found"}
    }
)
async def get_analysis_report(analysis_id: str):
    """Retrieves the generated 6-panel diagnostic visual report PNG for the specified analysis_id."""
    report_file = report_service.get_report_file(analysis_id)
    if not report_file or not report_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Diagnostic report for analysis '{analysis_id}' was not found."
        )
    return FileResponse(path=str(report_file), media_type="image/png", filename=f"report_{analysis_id}.png")
