import os
import sys
from pathlib import Path
import torch

# Base Directory of WoundInsight-API project
BASE_DIR = Path(__file__).resolve().parent.parent

# Ensure project root is on sys.path so that 'src.pipeline' imports succeed cleanly
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

class Settings:
    # Service Metadata
    SERVICE_NAME: str = "WoundInsight Clinical AI API"
    VERSION: str = "1.0.0"
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"

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

    # Storage Directories
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    REPORT_DIR: Path = STORAGE_DIR / "reports"
    DATABASE_DIR: Path = STORAGE_DIR / "database"

    # Database Configuration (SQLite default with support for external PostgreSQL/MySQL via env)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{DATABASE_DIR / 'woundinsight.db'}"
    )

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

    def ensure_storage_dirs(self) -> None:
        """Ensures all persistence directories exist on disk."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.DATABASE_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_storage_dirs()
