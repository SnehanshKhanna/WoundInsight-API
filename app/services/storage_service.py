import re
import logging
from pathlib import Path
from typing import Optional
from supabase import create_client, Client
from app.config import settings

logger = logging.getLogger("wound_api.storage")

def sanitize_filename(filename: str) -> str:
    """Removes potentially dangerous path traversal characters."""
    clean = re.sub(r"[^\w\.-]", "_", Path(filename).name)
    return clean or "upload.jpg"

class StorageService:
    """Service managing persistent cloud storage in Supabase."""
    
    def __init__(self):
        self.bucket_name = "clinical-artifacts"
        self.supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
        
    def upload_original_image(self, user_id: str, wound_id: str, analysis_id: str, filename: str, file_bytes: bytes) -> str:
        safe_name = sanitize_filename(filename)
        path = f"users/{user_id}/wounds/{wound_id}/analyses/{analysis_id}/original_{safe_name}"
        self.supabase.storage.from_(self.bucket_name).upload(path, file_bytes)
        logger.info(f"Uploaded original image to {path}")
        return path

    def upload_gradcam_image(self, user_id: str, wound_id: str, analysis_id: str, image_bytes: bytes) -> str:
        path = f"users/{user_id}/wounds/{wound_id}/analyses/{analysis_id}/gradcam.png"
        self.supabase.storage.from_(self.bucket_name).upload(
            path, 
            image_bytes,
            file_options={"content-type": "image/png"}
        )
        logger.info(f"Uploaded Grad-CAM image to {path}")
        return path

    def upload_report_image(self, user_id: str, wound_id: str, analysis_id: str, image_bytes: bytes) -> str:
        path = f"users/{user_id}/wounds/{wound_id}/analyses/{analysis_id}/report.png"
        self.supabase.storage.from_(self.bucket_name).upload(
            path, 
            image_bytes,
            file_options={"content-type": "image/png"}
        )
        logger.info(f"Uploaded report image to {path}")
        return path

    def get_signed_url(self, storage_path: str, expires_in: int = 60) -> str:
        res = self.supabase.storage.from_(self.bucket_name).create_signed_url(storage_path, expires_in)
        return res["signedURL"]

    def delete_analysis_artifacts(self, user_id: str, wound_id: str, analysis_id: str) -> None:
        """Best-effort cleanup of any files uploaded for this analysis."""
        prefix = f"users/{user_id}/wounds/{wound_id}/analyses/{analysis_id}/"
        try:
            # list takes the folder path
            files = self.supabase.storage.from_(self.bucket_name).list(prefix)
            if not files:
                return
            paths_to_delete = [f"{prefix}{f['name']}" for f in files if f.get('name')]
            if paths_to_delete:
                self.supabase.storage.from_(self.bucket_name).remove(paths_to_delete)
                logger.info(f"Cleaned up {len(paths_to_delete)} artifacts for analysis {analysis_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup artifacts for analysis {analysis_id}: {e}")

storage_service = StorageService()
