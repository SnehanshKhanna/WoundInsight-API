from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class WoundMorphometrics(BaseModel):
    detected: bool = Field(..., description="Whether an active wound bed was delineated")
    area_pixels: int = Field(..., description="Raw image-derived pixel area")
    area_cm2: float = Field(..., description="Estimated physical area in cm² (based on 0.15 mm/px heuristic)")
    perimeter_mm: float = Field(..., description="Estimated physical perimeter in mm (based on 0.15 mm/px heuristic)")
    circularity: float = Field(..., description="Isoperimetric quotient bounded [0, 1]")
    is_irregular: bool = Field(..., description="True if circularity < 0.55 indicating irregular lesion contour")

class TissueBreakdown(BaseModel):
    fibrin_slough_percent: float = Field(..., description="Percentage of segmented wound bed comprising Fibrin / Slough tissue")
    granulation_percent: float = Field(..., description="Percentage of segmented wound bed comprising Granulation tissue")
    callus_percent: float = Field(..., description="Percentage of segmented wound bed comprising Callus tissue")

class EtiologyProbabilities(BaseModel):
    DFU: float = Field(..., description="Diabetic Foot Ulcer probability percentage")
    Pressure: float = Field(..., description="Pressure Ulcer probability percentage")
    Surgical: float = Field(..., description="Surgical Wound probability percentage")
    Venous: float = Field(..., description="Venous Ulcer probability percentage")

class EtiologyDiagnostics(BaseModel):
    predicted_type: str = Field(..., description="Predicted wound classification category")
    confidence: float = Field(..., description="Predicted class confidence percentage")
    probabilities: EtiologyProbabilities = Field(..., description="All four class probability distributions")

class SeverityAssessment(BaseModel):
    severity_score: float = Field(..., description="Composite clinical triage risk score (0 - 100)")
    severity_grade: str = Field(..., description="Triage grade (Low, Moderate, High, Critical)")
    recommended_action: str = Field(..., description="Clinical protocol guidance based on composite risk")

class SafetyQA(BaseModel):
    ai_confidence_score: float = Field(..., description="Epistemic uncertainty confidence estimate (0 - 100%)")
    requires_clinician_review: bool = Field(..., description="Flag indicating if low AI confidence or high ambiguity requires manual review")

class VisualOutputs(BaseModel):
    report_image_url: str = Field(..., description="URL endpoint to access the diagnostic visual report")
    report_filename: str = Field(..., description="Saved report image filename")
    original_image_url: Optional[str] = Field(None, description="URL endpoint to access the original stored image")
    gradcam_image_url: Optional[str] = Field(None, description="URL endpoint to access the standalone fused Grad-CAM attribution image")

class ExplainabilityAttribution(BaseModel):
    available: bool = Field(..., description="Whether Grad-CAM attribution is generated")
    method: str = Field(default="Grad-CAM (Dual-Branch ResNet34)", description="Attribution algorithm")
    target_class: Optional[str] = Field(None, description="Attributed etiology category")
    global_target_layer: Optional[str] = Field(None, description="Global branch target convolutional layer")
    roi_target_layer: Optional[str] = Field(None, description="ROI branch target convolutional layer")
    fusion_method: Optional[str] = Field(None, description="Coordinate mapping and fusion formula")
    academic_notice: str = Field(
        default="Attribution visualization indicates model feature activation patterns, not clinical causality.",
        description="Standard academic interpretability notice"
    )

class AnalysisResponse(BaseModel):
    analysis_id: str = Field(..., description="Unique analysis identifier (UUID)")
    status: str = Field(default="success", example="success")
    timestamp: str = Field(..., description="ISO 8601 analysis timestamp")
    original_filename: str = Field(..., description="Original uploaded filename")
    user_id: Optional[str] = Field(None, description="User or session identifier (if provided)")
    wound_id: Optional[str] = Field(None, description="Associated wound identifier")
    wound: WoundMorphometrics
    tissue: TissueBreakdown
    etiology: EtiologyDiagnostics
    severity: SeverityAssessment
    uncertainty: SafetyQA
    visualizations: VisualOutputs
    explainability: Optional[ExplainabilityAttribution] = None
    academic_notice: str = Field(
        default="Academic prototype. Not certified for standalone clinical diagnostic decisions.",
        description="Standard academic research and decision-support disclaimer."
    )
    _meta: Optional[Dict[str, Any]] = None

class AnalysisSummaryItem(BaseModel):
    analysis_id: str
    created_at: str
    original_filename: str
    user_id: Optional[str] = None
    wound_id: Optional[str] = None
    predicted_etiology: str
    etiology_confidence: float
    wound_area_cm2: float
    severity_grade: str
    severity_score: float
    ai_confidence_score: float
    clinician_review_flag: bool
    report_image_url: str

class AnalysisListResponse(BaseModel):
    total: int = Field(..., description="Total count of recorded analyses")
    limit: int = Field(..., description="Limit applied")
    offset: int = Field(..., description="Offset applied")
    analyses: List[AnalysisSummaryItem] = Field(..., description="List of recorded analysis summaries")
