import os
import io
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from PIL import Image
import torch
from fastapi import HTTPException, status

from src.pipeline import MasterWoundSystem
from app.config import settings
from app.services.storage_service import storage_service
from app.services.report_service import report_service
from app.services.model_manager import model_manager
from app.models.database_models import AnalysisRecordModel
from app.db.repositories import AnalysisRepository

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

        model_manager.ensure_models_available()

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

    def get_status(self, db_connected: bool = True) -> Dict[str, Any]:
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
            "database_connected": db_connected
        }

    def process_and_persist(
        self,
        pil_image: Image.Image,
        file_bytes: bytes,
        original_filename: str,
        user_id: str,
        wound_id: str
    ) -> Dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("Inference engine is not initialized.")

        # 1. Generate analysis_id
        analysis_id = str(uuid.uuid4())
        created_at_iso = datetime.now(timezone.utc).isoformat()

        try:
            # 2. Perform inference
            t0 = time.perf_counter()
            raw_results = self.system.process_image(pil_image, run_explainability=True)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

            # 3. Generate Grad-CAM bytes
            gradcam_bytes = None
            expl_raw = raw_results["_raw"].get("explainability_data")
            if expl_raw and "fused_overlay" in expl_raw:
                buffer = io.BytesIO()
                img = Image.fromarray(expl_raw["fused_overlay"])
                img.save(buffer, format="PNG")
                gradcam_bytes = buffer.getvalue()

            # 4. Generate report bytes
            report_bytes = report_service.generate_report_bytes(
                system=self.system,
                raw_results=raw_results
            )

            # 5. Upload original image
            img_dest = storage_service.upload_original_image(
                user_id=user_id,
                wound_id=wound_id,
                analysis_id=analysis_id,
                filename=original_filename,
                file_bytes=file_bytes
            )

            # 6. Upload Grad-CAM
            gradcam_dest = None
            if gradcam_bytes:
                gradcam_dest = storage_service.upload_gradcam_image(
                    user_id=user_id,
                    wound_id=wound_id,
                    analysis_id=analysis_id,
                    image_bytes=gradcam_bytes
                )

            # 7. Upload report
            report_dest = storage_service.upload_report_image(
                user_id=user_id,
                wound_id=wound_id,
                analysis_id=analysis_id,
                image_bytes=report_bytes
            )

            # 8. Insert database record
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

            db_record = AnalysisRecordModel(
                analysis_id=analysis_id,
                user_id=user_id,
                wound_id=wound_id,
                original_filename=original_filename,
                image_storage_path=img_dest,
                report_storage_path=report_dest,
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

        except Exception as e:
            logger.error(f"Inference pipeline failed for {analysis_id}. Rolling back storage artifacts: {e}", exc_info=True)
            storage_service.delete_analysis_artifacts(user_id, wound_id, analysis_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred during analysis processing and persistence."
            )

        return {
            "analysis_id": analysis_id,
            "status": "success",
            "timestamp": created_at_iso,
            "original_filename": original_filename,
            "user_id": user_id,
            "wound_id": wound_id,
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
                "report_filename": f"report_{analysis_id}.png",
                "original_image_url": f"/api/v1/analyses/{analysis_id}/image",
                "gradcam_image_url": f"/api/v1/analyses/{analysis_id}/gradcam"
            },
            "explainability": expl,
            "academic_notice": "Academic prototype. Not certified for standalone clinical diagnostic decisions.",
            "_meta": {
                "inference_time_ms": latency_ms
            }
        }

inference_service = InferenceService.get_instance()
