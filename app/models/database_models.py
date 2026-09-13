from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class AnalysisRecordModel:
    analysis_id: str
    original_filename: str
    image_storage_path: str
    report_storage_path: str
    created_at: str
    predicted_etiology: str
    etiology_confidence: float
    etiology_probabilities: Dict[str, float]
    wound_detected: bool
    wound_area_pixels: int
    wound_area_cm2: float
    perimeter_mm: float
    circularity: float
    is_irregular: bool
    fibrin_slough_percent: float
    granulation_percent: float
    callus_percent: float
    severity_score: float
    severity_grade: str
    recommended_action: str
    ai_confidence_score: float
    clinician_review_flag: bool
    inference_time_ms: float
    user_id: Optional[str] = None
    explainability_metadata: Optional[Dict[str, Any]] = None
