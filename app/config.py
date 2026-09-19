import os
import sys
from pathlib import Path
import torch

from typing import Optional
from dotenv import load_dotenv

# Base Directory of WoundInsight-API project
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Ensure project root is on sys.path so that 'src.pipeline' imports succeed cleanly
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

class Settings:
    # Service Metadata
    SERVICE_NAME: str = "WoundInsight Clinical AI API"
    VERSION: str = "1.0.0"
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"

    # Hugging Face Model Hub Configuration
    HF_MODEL_REPO_ID: str = os.getenv("HF_MODEL_REPO_ID", "SnehanshKhanna/WoundInsight-models")
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN", None))
    HF_REVISION: Optional[str] = os.getenv("HF_REVISION", None)
    HF_FORCE_DOWNLOAD: bool = os.getenv("HF_FORCE_DOWNLOAD", "false").lower() in ("true", "1", "yes")

    # Local Checkpoint Directory
    CHECKPOINTS_DIR: Path = BASE_DIR / "checkpoints"

    # Relative Remote File Paths on Hugging Face Repository
    WOUND_MODEL_REMOTE_PATH: str = "segmentation/retrained_best_wound_model.pth"
    TISSUE_MODEL_REMOTE_PATH: str = "tissue_segmentation/retrained_best_tissue_model.pth"
    CLASSIFIER_MODEL_REMOTE_PATH: str = "classification/retrained_best_dual_branch_classifier.pth"

    # Production Model Checkpoint Paths
    WOUND_MODEL_PATH: str = os.getenv(
        "WOUND_MODEL_PATH",
        str(BASE_DIR / "checkpoints" / "segmentation" / "retrained_best_wound_model.pth")
    )
    TISSUE_MODEL_PATH: str = os.getenv(
        "TISSUE_MODEL_PATH",
        str(BASE_DIR / "checkpoints" / "tissue_segmentation" / "retrained_best_tissue_model.pth")
    )
    CLASSIFIER_MODEL_PATH: str = os.getenv(
        "CLASSIFIER_MODEL_PATH",
        str(BASE_DIR / "checkpoints" / "classification" / "retrained_best_dual_branch_classifier.pth")
    )

    # Compute Device
    DEVICE: str = os.getenv(
        "API_DEVICE",
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # Fallback Spatial Scale (assumed 0.15 mm/pixel for clinical photography heuristic)
    FALLBACK_SCALE: float = float(os.getenv("FALLBACK_SCALE", "0.15"))

    # Database Configuration
    DATABASE_URL: str = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable must be set for PostgreSQL connection.")

    # Supabase Configuration
    SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
    if not SUPABASE_URL:
        raise ValueError("SUPABASE_URL environment variable must be set.")

    SUPABASE_SERVICE_ROLE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError("SUPABASE_SERVICE_ROLE_KEY environment variable must be set.")

    # Allowed Upload Extensions & MIME Types
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ALLOWED_MIME_TYPES: set = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp"
    }

    # Upload Size Limit (15 MB)
    MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024

    # CORS Configuration
    CORS_ORIGINS: list[str] = ["*"]

    # JWT Authentication Configuration
    JWT_SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY")
    if not JWT_SECRET_KEY:
        raise ValueError("JWT_SECRET_KEY environment variable must be set for production security.")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))  # 7 days default

    def ensure_storage_dirs(self) -> None:
        """Ensures checkpoint directories exist on disk."""
        self.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        (self.CHECKPOINTS_DIR / "classification").mkdir(parents=True, exist_ok=True)
        (self.CHECKPOINTS_DIR / "segmentation").mkdir(parents=True, exist_ok=True)
        (self.CHECKPOINTS_DIR / "tissue_segmentation").mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_storage_dirs()
