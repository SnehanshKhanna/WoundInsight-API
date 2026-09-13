import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, status, Depends
from fastapi.responses import FileResponse, JSONResponse

from app.schemas.analysis import AnalysisResponse, AnalysisListResponse
from app.schemas.error import ErrorResponse
from app.utils.image_validation import validate_image_upload
from app.utils.auth import get_current_user
from app.services.inference_service import inference_service
from app.services.storage_service import storage_service
from app.db.repositories import AnalysisRepository, WoundRepository

logger = logging.getLogger("wound_api.routes.analysis")
router = APIRouter(tags=["Analysis"])

@router.post(
    "/api/v1/analyses",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image format, size, wound_id, or corrupted payload"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        500: {"model": ErrorResponse, "description": "Inference pipeline failure"}
    },
    summary="Submit Wound Photo for Comprehensive Multi-Task Analysis"
)
async def analyze_wound_image(
    image: UploadFile = File(..., description="Single photographic image of a wound (JPEG, PNG, WEBP, BMP)"),
    wound_id: str = Form(..., description="Associated user-owned wound profile identifier"),
    current_user: dict = Depends(get_current_user)
):
    """
    Accepts one wound image upload (`multipart/form-data`) associated with a user wound profile
    and executes the complete multi-task clinical AI pipeline:
    - Binary Wound Bed Segmentation (DeepLabV3+)
    - Multi-Class Tissue Segmentation (Fibrin/Slough, Granulation, Callus)
    - Dual-Branch Etiology Classification (DFU, Pressure, Surgical, Venous)
    - Physical & Pixel Morphometrics
    - Clinical Triage Severity Scoring
    - Epistemic Uncertainty Estimation
    - Grad-CAM Attribution Heatmaps
    - Visual Diagnostic Report Generation
    - Persistent Record Storage with strict user & wound ownership
    """
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded. Please provide an image under the 'image' field."
        )

    # Verify that the targeted wound belongs to the authenticated user
    wound = WoundRepository.get_by_id(wound_id)
    if not wound or wound.user_id != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Wound profile '{wound_id}' is invalid or does not belong to the authenticated user."
        )

    # Read uploaded file stream
    try:
        contents = await image.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file stream: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to read the uploaded file stream."
        )

    # Validate image payload
    pil_image = validate_image_upload(
        file_bytes=contents,
        filename=image.filename,
        content_type=image.content_type
    )

    # Run inference and persist
    try:
        result = inference_service.process_and_persist(
            pil_image=pil_image,
            file_bytes=contents,
            original_filename=image.filename,
            user_id=current_user["id"],
            wound_id=wound_id
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference pipeline execution error on {image.filename}: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error": "An internal error occurred while processing the wound image.",
                "error_code": "INFERENCE_PIPELINE_ERROR"
            }
        )

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    include_in_schema=False,
    summary="Legacy Compatibility Endpoint for Wound Analysis"
)
async def legacy_analyze_wound_image(
    image: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    wound_id: Optional[str] = Form(None)
):
    """Legacy unauthenticated fallback route for backwards compatibility."""
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded. Please provide an image under the 'image' field."
        )
    contents = await image.read()
    pil_image = validate_image_upload(file_bytes=contents, filename=image.filename, content_type=image.content_type)
    return inference_service.process_and_persist(
        pil_image=pil_image,
        file_bytes=contents,
        original_filename=image.filename,
        user_id=user_id,
        wound_id=wound_id
    )

@router.get(
    "/api/v1/analyses/{analysis_id}",
    response_model=AnalysisResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Analysis record not found or not owned by user"}
    },
    summary="Get Analysis Record by ID"
)
async def get_analysis_by_id(
    analysis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Retrieves a saved wound analysis by its unique analysis_id, enforcing ownership."""
    record = AnalysisRepository.get_by_id(analysis_id)
    if not record or record["user_id"] != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID '{analysis_id}' was not found."
        )
    return record

@router.get(
    "/api/v1/analyses",
    response_model=AnalysisListResponse,
    summary="List Historical Wound Analyses (Paginated & Filtered to Current User)"
)
async def list_analyses(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Offset index for pagination"),
    wound_id: Optional[str] = Query(None, description="Optional filter by wound_id"),
    current_user: dict = Depends(get_current_user)
):
    """Lists past wound analyses strictly scoped to the authenticated user."""
    items, total = AnalysisRepository.list_analyses(
        limit=limit,
        offset=offset,
        user_id=current_user["id"],
        wound_id=wound_id
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "analyses": items
    }

@router.get(
    "/api/v1/analyses/{analysis_id}/image",
    summary="Retrieve Stored Original Wound Photo",
    responses={
        200: {"description": "Original image binary file"},
        404: {"description": "Original image not found or unauthorized"}
    }
)
async def get_analysis_original_image(
    analysis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Retrieves the original uploaded wound photo for an analysis, verifying ownership."""
    record = AnalysisRepository.get_by_id(analysis_id)
    if not record or record["user_id"] != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis record '{analysis_id}' was not found."
        )

    img_path = storage_service.get_uploaded_image_path(analysis_id)
    if not img_path or not img_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original image for analysis '{analysis_id}' was not found."
        )

    ext = img_path.suffix.lower()
    media_type = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
    return FileResponse(path=str(img_path), media_type=media_type, filename=img_path.name)

@router.get(
    "/api/v1/analyses/{analysis_id}/gradcam",
    summary="Retrieve Standalone Grad-CAM Attribution Overlay Image",
    responses={
        200: {"content": {"image/png": {}}, "description": "Standalone Grad-CAM Attribution Overlay PNG"},
        404: {"description": "Grad-CAM image not found or unauthorized"}
    }
)
async def get_analysis_gradcam_image(
    analysis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Retrieves the standalone Grad-CAM attribution overlay image, verifying ownership."""
    record = AnalysisRepository.get_by_id(analysis_id)
    if not record or record["user_id"] != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis record '{analysis_id}' was not found."
        )

    gradcam_path = storage_service.get_gradcam_path(analysis_id)
    if not gradcam_path or not gradcam_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Grad-CAM visualization for analysis '{analysis_id}' was not found."
        )

    return FileResponse(path=str(gradcam_path), media_type="image/png", filename=f"gradcam_{analysis_id}.png")
