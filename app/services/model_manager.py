import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import HfHubHTTPError

from app.config import settings

logger = logging.getLogger("wound_api.model_manager")

class ModelManager:
    """
    Manages resolution, downloading, and local caching of WoundInsight deep learning models
    from Hugging Face Hub (default: SnehanshKhanna/WoundInsight-models).
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.repo_id = settings.HF_MODEL_REPO_ID
        self.token = settings.HF_TOKEN
        self.revision = settings.HF_REVISION
        self.base_checkpoints_dir = settings.CHECKPOINTS_DIR

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def get_model_specs(self) -> Dict[str, Dict[str, Any]]:
        """Returns metadata for all three production deep learning models."""
        return {
            "wound_segmentation": {
                "name": "Binary Wound Bed Segmenter (DeepLabV3+ ResNeSt50)",
                "remote_filename": settings.WOUND_MODEL_REMOTE_PATH,
                "local_path": Path(settings.WOUND_MODEL_PATH),
                "expected_min_bytes": 100_000_000,
            },
            "tissue_segmentation": {
                "name": "Supervised 4-Class Tissue Segmenter (DeepLabV3+ ResNeSt50)",
                "remote_filename": settings.TISSUE_MODEL_REMOTE_PATH,
                "local_path": Path(settings.TISSUE_MODEL_PATH),
                "expected_min_bytes": 100_000_000,
            },
            "classification": {
                "name": "Dual-Branch Etiology Classifier (ResNet34 Dual-Stream)",
                "remote_filename": settings.CLASSIFIER_MODEL_REMOTE_PATH,
                "local_path": Path(settings.CLASSIFIER_MODEL_PATH),
                "expected_min_bytes": 100_000_000,
            }
        }

    def is_cached(self, local_path: str | Path, min_bytes: int = 1) -> bool:
        """Checks if a checkpoint file is cached locally and is non-empty."""
        path = Path(local_path)
        return path.is_file() and path.stat().st_size >= min_bytes

    def download_model(
        self,
        model_key: str,
        force: bool = False,
        target_local_path: Optional[Path] = None,
        custom_repo_id: Optional[str] = None
    ) -> Path:
        """
        Ensures a specific model checkpoint is available locally.
        If already cached locally and not force, reuses local cache without network call.
        If missing or force is True, downloads the file from Hugging Face Hub.
        """
        specs = self.get_model_specs()
        if model_key not in specs:
            raise KeyError(f"Unknown model key '{model_key}'. Allowed: {list(specs.keys())}")

        spec = specs[model_key]
        dest_path = Path(target_local_path) if target_local_path else spec["local_path"]
        remote_filename = spec["remote_filename"]
        repo_id = custom_repo_id or self.repo_id
        force_dl = force or settings.HF_FORCE_DOWNLOAD

        # 1. Check local cache
        if not force_dl and self.is_cached(dest_path, min_bytes=spec["expected_min_bytes"]):
            size_mb = dest_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"[Model Cache Hit] {spec['name']} found locally at: {dest_path} ({size_mb:.2f} MB)"
            )
            return dest_path

        # 2. Download from Hugging Face Hub
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[Model Download] Fetching '{remote_filename}' from HF Hub ({repo_id}) to {dest_path}..."
        )

        try:
            # If target is within a base directory mirroring repo structure
            # hf_hub_download writes directly to local_dir / filename
            base_dir = dest_path.parent
            # Check if dest_path ends with remote_filename
            try:
                rel = dest_path.relative_to(dest_path.parents[len(Path(remote_filename).parts) - 1])
                match_structure = str(rel).replace("\\", "/") == remote_filename
            except Exception:
                match_structure = False

            if match_structure:
                local_dir_root = dest_path.parents[len(Path(remote_filename).parts) - 1]
                downloaded_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_filename,
                    local_dir=str(local_dir_root),
                    token=self.token,
                    revision=self.revision,
                    force_download=force_dl
                )
            else:
                # Download into parent and move/ensure dest_path
                temp_dir = dest_path.parent / ".hf_download_temp"
                temp_dir.mkdir(parents=True, exist_ok=True)
                downloaded_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_filename,
                    local_dir=str(temp_dir),
                    token=self.token,
                    revision=self.revision,
                    force_download=force_dl
                )
                if Path(downloaded_file).resolve() != dest_path.resolve():
                    shutil.move(downloaded_file, dest_path)
                shutil.rmtree(temp_dir, ignore_errors=True)

            final_path = Path(dest_path)
            if not final_path.is_file() or final_path.stat().st_size == 0:
                raise RuntimeError(f"Download completed but file is missing or empty at: {final_path}")

            size_mb = final_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"[Model Download Complete] {spec['name']} saved to: {final_path} ({size_mb:.2f} MB)"
            )
            return final_path

        except HfHubHTTPError as e:
            logger.error(
                f"Failed to download {spec['name']} from HF repo '{repo_id}' ({remote_filename}): {e}"
            )
            raise RuntimeError(
                f"Hugging Face Hub HTTP Error downloading {remote_filename} from {repo_id}: {e}. "
                f"Check repo ID, network connectivity, or supply HF_TOKEN if repository is private."
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error downloading {spec['name']}: {e}", exc_info=True)
            raise

    def ensure_models_available(self, force: bool = False) -> Dict[str, Path]:
        """
        Ensures all three production models exist locally.
        Missing models are downloaded automatically from Hugging Face Hub.
        Cached models are reused immediately without network requests.
        """
        resolved_paths = {}
        for key in ["wound_segmentation", "tissue_segmentation", "classification"]:
            resolved_paths[key] = self.download_model(model_key=key, force=force)
        return resolved_paths

    def get_status(self) -> Dict[str, Any]:
        """Returns the cache status of all model checkpoints."""
        specs = self.get_model_specs()
        status = {
            "hf_repo_id": self.repo_id,
            "has_token_configured": bool(self.token),
            "models": {}
        }
        for key, spec in specs.items():
            lp = spec["local_path"]
            is_present = self.is_cached(lp, min_bytes=spec["expected_min_bytes"])
            status["models"][key] = {
                "name": spec["name"],
                "cached": is_present,
                "local_path": str(lp),
                "remote_filename": spec["remote_filename"],
                "file_size_bytes": lp.stat().st_size if is_present else None
            }
        return status

model_manager = ModelManager.get_instance()
