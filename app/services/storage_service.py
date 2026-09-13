import os
import re
from pathlib import Path
from typing import Optional
from app.config import settings

def sanitize_filename(filename: str) -> str:
    """Removes potentially dangerous path traversal characters."""
    clean = re.sub(r"[^\w\.-]", "_", Path(filename).name)
    return clean or "upload.jpg"

class StorageService:
    """Service managing persistent disk storage of uploaded wound photos and generated diagnostic reports."""

    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        self.report_dir = settings.REPORT_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save_uploaded_image(self, file_bytes: bytes, filename: str, analysis_id: str) -> Path:
        """Saves original uploaded image payload to storage/uploads/{analysis_id}_{filename}."""
        safe_name = sanitize_filename(filename)
        stored_name = f"{analysis_id}_{safe_name}"
        dest_path = self.upload_dir / stored_name
        with open(dest_path, "wb") as f:
            f.write(file_bytes)
        return dest_path

    def get_uploaded_image_path(self, analysis_id: str) -> Optional[Path]:
        """Finds stored uploaded image by analysis_id prefix."""
        for item in self.upload_dir.iterdir():
            if item.is_file() and item.name.startswith(f"{analysis_id}_"):
                return item
        return None

    def get_report_path(self, analysis_id: str) -> Optional[Path]:
        """Returns path to the generated diagnostic report PNG."""
        report_path = self.report_dir / f"report_{analysis_id}.png"
        if report_path.exists() and report_path.is_file():
            return report_path
        return None

    def get_report_destination(self, analysis_id: str) -> Path:
        """Returns targeted output path for a new diagnostic report figure."""
        return self.report_dir / f"report_{analysis_id}.png"

    def save_gradcam_image(self, overlay_rgb, analysis_id: str) -> Path:
        """Saves Grad-CAM attribution overlay figure to storage/reports/gradcam_{analysis_id}.png."""
        from PIL import Image
        dest_path = self.report_dir / f"gradcam_{analysis_id}.png"
        img = Image.fromarray(overlay_rgb)
        img.save(dest_path, format="PNG")
        return dest_path

    def get_gradcam_path(self, analysis_id: str) -> Optional[Path]:
        """Returns path to the standalone Grad-CAM image PNG if it exists."""
        gradcam_path = self.report_dir / f"gradcam_{analysis_id}.png"
        if gradcam_path.exists() and gradcam_path.is_file():
            return gradcam_path
        return None

storage_service = StorageService()
