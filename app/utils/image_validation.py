import io
from pathlib import Path
from typing import Optional
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException, status
from app.config import settings

def validate_image_upload(file_bytes: bytes, filename: str, content_type: Optional[str] = None) -> Image.Image:
    """
    Validates uploaded file payload:
      1. Size check (<= MAX_FILE_SIZE_BYTES).
      2. File extension check.
      3. MIME type check if present.
      4. Pillow decode integrity check.
      5. Minimum / maximum resolution sanity check.
    Returns RGB PIL Image on success or raises HTTPException(400).
    """
    # 1. Size verification
    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    if len(file_bytes) > settings.MAX_FILE_SIZE_BYTES:
        max_mb = settings.MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size of {max_mb:.1f} MB."
        )

    # 2. Extension check
    ext = Path(filename).suffix.lower() if filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
        )

    # 3. MIME type check
    if content_type and content_type.lower() not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type '{content_type}'. Must be one of: {', '.join(sorted(settings.ALLOWED_MIME_TYPES))}"
        )

    # 4. Pillow decode verification
    try:
        image_stream = io.BytesIO(file_bytes)
        img = Image.open(image_stream)
        img.verify()  # Verify image header / integrity

        # Re-open stream because verify() closes the handle in PIL
        image_stream.seek(0)
        img = Image.open(image_stream)
        img.load()    # Force load image data
        rgb_img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid or decipherable image."
        )

    # 5. Dimension sanity check
    w, h = rgb_img.size
    if w < 32 or h < 32:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image resolution ({w}x{h}) is too small. Minimum supported resolution is 32x32 pixels."
        )
    if w > 8192 or h > 8192:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image resolution ({w}x{h}) exceeds maximum allowed dimension of 8192x8192 pixels."
        )

    return rgb_img
