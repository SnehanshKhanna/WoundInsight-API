import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse

from app.schemas.analysis import AnalysisResponse, AnalysisListResponse
from app.schemas.error import ErrorResponse
from app.utils.image_validation import validate_image_upload
from app.services.inference_service import inference_service
from app.services.storage_service import storage_service
from app.db.repositories import AnalysisRepository

logger = logging.getLogger("wound_api.routes.analysis")
router = APIRouter(tags=["Analysis"])

@router.post(
    "/api/v1/analyses",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image format, size, or corrupted payload"},
        500: {"model": ErrorResponse, "description": "Inference pipeline failure"}
    },
    summary="Submit Wound Photo for Comprehensive Multi-Task Analysis"
)
@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    include_in_schema=False,
    summary="Legacy Compatibility Endpoint for Wound Analysis"
)
async def analyze_wound_image(
    image: UploadFile = File(..., description="Single photographic image of a wound (JPEG, PNG, WEBP, BMP)"),
    user_id: Optional[str] = Form(None, description="Optional user or clinician identifier")
):
    """
    Accepts one wound image upload (`multipart/form-data`) and runs the complete multi-task clinical AI pipeline:
    - **Binary Wound Bed Segmentation** (DeepLabV3+ ResNet34)
    - **Multi-Class Tissue Segmentation** (0: Background, 1: Fibrin/Slough, 2: Granulation, 3: Callus)
    - **Dual-Branch Etiology Classification** (DFU, Pressure, Surgical, Venous Ulcers)
    - **Physical & Pixel Morphometrics** (Area, perimeter, circularity, contour regularity)
    - **Clinical Triage Severity** (0-100 composite risk score, triage grade, recommended clinical protocol)
    - **Epistemic Uncertainty Estimation** (MC-Dropout + TTA confidence score & clinician review flag)
    - **Grad-CAM Attribution** (Global context, central ROI stream, and fused attributions)
    - **Visual Diagnostic Report Generation**
    - **Persistent Record Storage** (Database & filesystem storage)
    """
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded. Please provide an image under the 'image' field."
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
            user_id=user_id
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

@router.get(
    "/api/v1/analyses/{analysis_id}",
    response_model=AnalysisResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Analysis record not found"}
    },
    summary="Get Analysis Record by ID"
)
async def get_analysis_by_id(analysis_id: str):
    """Retrieves a previously saved wound analysis by its unique analysis_id."""
    record = AnalysisRepository.get_by_id(analysis_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID '{analysis_id}' not found."
        )
    return record

@router.get(
    "/api/v1/analyses",
    response_model=AnalysisListResponse,
    summary="List Historical Wound Analyses (Paginated)"
)
async def list_analyses(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Offset index for pagination"),
    user_id: Optional[str] = Query(None, description="Filter analyses by user_id")
):
    """Lists past wound analyses ordered from newest to oldest."""
    items, total = AnalysisRepository.list_analyses(limit=limit, offset=offset, user_id=user_id)
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
        404: {"description": "Original image not found"}
    }
)
async def get_analysis_original_image(analysis_id: str):
    """Retrieves the original uploaded wound photo for an analysis."""
    img_path = storage_service.get_uploaded_image_path(analysis_id)
    if not img_path or not img_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original image for analysis '{analysis_id}' was not found."
        )
    
    # Infer mime type from extension
    ext = img_path.suffix.lower()
    media_type = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
    return FileResponse(path=str(img_path), media_type=media_type, filename=img_path.name)
