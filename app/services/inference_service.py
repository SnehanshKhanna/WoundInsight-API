import os
import time
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import torch

from src.pipeline import MasterWoundSystem
from app.config import settings
from app.services.storage_service import storage_service
from app.services.report_service import report_service
from app.models.database_models import AnalysisRecordModel
from app.db.repositories import AnalysisRepository
from app.db.database import check_db_health

logger = logging.getLogger("wound_api.inference_service")

class InferenceService:
    _instance: Optional["InferenceService"] = None

    def __init__(self):
        self.system: Optional[MasterWoundSystem] = None
        self.loaded_at: Optional[str] = None
        self.device = settings.DEVICE

    @classmethod
    def get_instance(cls) -> "InferenceService":
        if cls._instance is None:
            cls._instance = InferenceService()
        return cls._instance

    def initialize_models(self) -> None:
        """Loads all three production models onto designated device."""
        if self.system is not None:
            return

        logger.info(f"Loading production models on device: {self.device}")
        for path_name, path_val in [
            ("Wound Segmenter", settings.WOUND_MODEL_PATH),
            ("Tissue Segmenter", settings.TISSUE_MODEL_PATH),
            ("Classifier", settings.CLASSIFIER_MODEL_PATH),
        ]:
            if not os.path.exists(path_val):
                raise FileNotFoundError(f"Missing required model checkpoint for {path_name}: {path_val}")

        self.system = MasterWoundSystem(
            wound_seg_path=settings.WOUND_MODEL_PATH,
            tissue_seg_path=settings.TISSUE_MODEL_PATH,
            classifier_path=settings.CLASSIFIER_MODEL_PATH,
            fallback_scale=settings.FALLBACK_SCALE,
            device=self.device
        )
        self.loaded_at = datetime.now(timezone.utc).isoformat()
        logger.info("All 3 AI models successfully loaded and ready.")

    def is_ready(self) -> bool:
        return self.system is not None

    def get_status(self) -> Dict[str, Any]:
        gpu_name = None
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "CUDA Device"

        return {
            "status": "healthy" if self.is_ready() else "loading",
            "device": self.device,
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "models_loaded": self.is_ready(),
            "checkpoints": {
                "wound_segmentation": os.path.basename(settings.WOUND_MODEL_PATH),
                "tissue_segmentation": os.path.basename(settings.TISSUE_MODEL_PATH),
                "etiology_classifier": os.path.basename(settings.CLASSIFIER_MODEL_PATH),
            },
            "database_connected": check_db_health()
        }

    def process_and_persist(
        self,
        pil_image: Image.Image,
        file_bytes: bytes,
        original_filename: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("Inference engine is not initialized.")

        analysis_id = str(uuid.uuid4())
        created_at_iso = datetime.now(timezone.utc).isoformat()

        # 1. Persist original uploaded image file
        img_dest = storage_service.save_uploaded_image(
            file_bytes=file_bytes,
            filename=original_filename,
            analysis_id=analysis_id
        )

        # 2. Run finalized multi-task AI inference pipeline
        t0 = time.perf_counter()
        raw_results = self.system.process_image(pil_image, run_explainability=True)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        # 3. Generate & persist clinical diagnostic visual report
        report_dest = report_service.generate_and_save_report(
            system=self.system,
            raw_results=raw_results,
            analysis_id=analysis_id
        )

        # 4. Format structured probability map
        diag_probs = raw_results["diagnostics"]["class_probabilities"]
        formatted_probs = {
            "DFU": float(diag_probs.get("Diabetic Foot Ulcer", 0.0)),
            "Pressure": float(diag_probs.get("Pressure Ulcer", 0.0)),
            "Surgical": float(diag_probs.get("Surgical Wound", 0.0)),
            "Venous": float(diag_probs.get("Venous Ulcer", 0.0)),
        }

        morph = raw_results["morphometrics"]
        tissue = raw_results["tissue_breakdown_supervised"]
        sev = raw_results["severity_assessment"]
        qa = raw_results["safety_qa"]
        expl = raw_results.get("explainability")

        # 5. Persist record to SQLite database
        db_record = AnalysisRecordModel(
            analysis_id=analysis_id,
            user_id=user_id,
            original_filename=original_filename,
            image_storage_path=str(img_dest),
            report_storage_path=str(report_dest),
            created_at=created_at_iso,
            predicted_etiology=raw_results["diagnostics"]["predicted_type"],
            etiology_confidence=float(raw_results["diagnostics"]["confidence"]),
            etiology_probabilities=formatted_probs,
            wound_detected=bool(morph["area_pixels"] > 0),
            wound_area_pixels=int(morph["area_pixels"]),
            wound_area_cm2=float(morph["area_cm2"]),
            perimeter_mm=float(morph["perimeter_mm"]),
            circularity=float(morph["circularity"]),
            is_irregular=bool(morph["is_irregular"]),
            fibrin_slough_percent=float(tissue["fibrin_slough_percent"]),
            granulation_percent=float(tissue["granulation_percent"]),
            callus_percent=float(tissue["callus_percent"]),
            severity_score=float(sev["severity_score"]),
            severity_grade=sev["severity_grade"],
            recommended_action=sev["recommended_action"],
            ai_confidence_score=float(qa["ai_confidence_score"]),
            clinician_review_flag=bool(qa["requires_clinician_review"]),
            explainability_metadata=expl,
            inference_time_ms=latency_ms
        )
        AnalysisRepository.save_analysis(db_record)

        # 6. Construct standardized response dictionary
        report_filename = f"report_{analysis_id}.png"
        return {
            "analysis_id": analysis_id,
            "status": "success",
            "timestamp": created_at_iso,
            "original_filename": original_filename,
            "user_id": user_id,
            "wound": {
                "detected": bool(morph["area_pixels"] > 0),
                "area_pixels": int(morph["area_pixels"]),
                "area_cm2": float(morph["area_cm2"]),
                "perimeter_mm": float(morph["perimeter_mm"]),
                "circularity": float(morph["circularity"]),
                "is_irregular": bool(morph["is_irregular"]),
            },
            "tissue": {
                "fibrin_slough_percent": float(tissue["fibrin_slough_percent"]),
                "granulation_percent": float(tissue["granulation_percent"]),
                "callus_percent": float(tissue["callus_percent"]),
            },
            "etiology": {
                "predicted_type": raw_results["diagnostics"]["predicted_type"],
                "confidence": float(raw_results["diagnostics"]["confidence"]),
                "probabilities": formatted_probs,
            },
            "severity": {
                "severity_score": float(sev["severity_score"]),
                "severity_grade": sev["severity_grade"],
                "recommended_action": sev["recommended_action"],
            },
            "uncertainty": {
                "ai_confidence_score": float(qa["ai_confidence_score"]),
                "requires_clinician_review": bool(qa["requires_clinician_review"]),
            },
            "visualizations": {
                "report_image_url": f"/api/v1/analyses/{analysis_id}/report",
                "report_filename": report_filename,
                "original_image_url": f"/api/v1/analyses/{analysis_id}/image",
            },
            "explainability": expl,
            "academic_notice": "Academic prototype. Not certified for standalone clinical diagnostic decisions.",
            "_meta": {
                "inference_time_ms": latency_ms
            }
        }

inference_service = InferenceService.get_instance()
