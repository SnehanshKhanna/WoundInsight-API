import os
import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4

# Force dummy environment variables for static tests
os.environ["DATABASE_URL"] = "postgres://dummy:dummy@localhost:5432/dummy"
os.environ["SUPABASE_URL"] = "https://dummy.supabase.co"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "dummy"
os.environ["JWT_SECRET_KEY"] = "dummy_secret"

from app.models.database_models import UserModel, WoundModel, AnalysisRecordModel
from app.db.repositories import UserRepository, WoundRepository, AnalysisRepository
from app.services.storage_service import StorageService

# In-memory storage for mocks
mock_db = {
    "users": {},
    "wounds": {},
    "analyses": {}
}

def mock_create_user(user: UserModel):
    mock_db["users"][user.id] = user
    return user

def mock_get_user_by_id(user_id: str):
    return mock_db["users"].get(user_id)
    
def mock_get_user_by_email(email: str):
    for u in mock_db["users"].values():
        if u.email.lower() == email.lower():
            return u
    return None

def mock_create_wound(wound: WoundModel):
    mock_db["wounds"][wound.id] = wound
    return wound
    
def mock_get_wound_by_id(wound_id: str):
    return mock_db["wounds"].get(wound_id)
    
def mock_list_wounds_by_user(user_id: str):
    return [w for w in mock_db["wounds"].values() if w.user_id == user_id]

def mock_save_analysis(record: AnalysisRecordModel):
    mock_db["analyses"][record.analysis_id] = record
    return record.analysis_id
    
def mock_get_analysis_by_id(analysis_id: str):
    record = mock_db["analyses"].get(analysis_id)
    if not record: return None
    probs = record.etiology_probabilities
    expl = record.explainability_metadata
    
    return {
        "analysis_id": str(record.analysis_id),
        "status": "success",
        "timestamp": str(record.created_at),
        "original_filename": record.original_filename,
        "image_storage_path": record.image_storage_path,
        "report_storage_path": record.report_storage_path,
        "user_id": str(record.user_id),
        "wound_id": str(record.wound_id),
        "wound": {
            "detected": bool(record.wound_detected),
            "area_pixels": record.wound_area_pixels,
            "area_cm2": record.wound_area_cm2,
            "perimeter_mm": record.perimeter_mm,
            "circularity": record.circularity,
            "is_irregular": bool(record.is_irregular),
        },
        "tissue": {
            "fibrin_slough_percent": record.fibrin_slough_percent,
            "granulation_percent": record.granulation_percent,
            "callus_percent": record.callus_percent,
        },
        "etiology": {
            "predicted_type": record.predicted_etiology,
            "confidence": record.etiology_confidence,
            "probabilities": probs,
        },
        "severity": {
            "severity_score": record.severity_score,
            "severity_grade": record.severity_grade,
            "recommended_action": record.recommended_action,
        },
        "uncertainty": {
            "ai_confidence_score": record.ai_confidence_score,
            "requires_clinician_review": bool(record.clinician_review_flag),
        },
        "visualizations": {
            "report_image_url": f"/api/v1/analyses/{record.analysis_id}/report",
            "report_filename": f"report_{record.analysis_id}.png",
            "original_image_url": f"/api/v1/analyses/{record.analysis_id}/image",
            "gradcam_image_url": f"/api/v1/analyses/{record.analysis_id}/gradcam"
        },
        "explainability": expl,
        "academic_notice": "Academic prototype. Not certified for standalone clinical diagnostic decisions.",
        "_meta": {
            "inference_time_ms": record.inference_time_ms
        }
    }

def mock_list_analyses(limit=50, offset=0, user_id=None, wound_id=None):
    results = []
    for rec in mock_db["analyses"].values():
        if user_id and rec.user_id != user_id:
            continue
        if wound_id and rec.wound_id != wound_id:
            continue
            
        results.append({
            "analysis_id": str(rec.analysis_id),
            "created_at": str(rec.created_at),
            "original_filename": rec.original_filename,
            "user_id": str(rec.user_id),
            "wound_id": str(rec.wound_id),
            "predicted_etiology": rec.predicted_etiology,
            "etiology_confidence": rec.etiology_confidence,
            "wound_area_cm2": rec.wound_area_cm2,
            "severity_grade": rec.severity_grade,
            "severity_score": rec.severity_score,
            "ai_confidence_score": rec.ai_confidence_score,
            "clinician_review_flag": bool(rec.clinician_review_flag),
            "report_image_url": f"/api/v1/analyses/{rec.analysis_id}/report"
        })
    # sort by created_at desc
    results.sort(key=lambda x: x["created_at"], reverse=True)
    return results[offset:offset+limit], len(results)

from app.main import app
from app.db.database import check_db_health

@pytest.fixture(autouse=True, scope="session")
def session_mock_supabase_and_db():
    if os.environ.get("SUPABASE_INTEGRATION_TESTS") == "1":
        yield
        return

    # Override FastAPI dependency for health route
    app.dependency_overrides[check_db_health] = lambda: True

    # The actual database initialization logic should be skipped
    patch_init = patch('app.main.init_db')
    patch_health = patch('app.db.database.check_db_health', return_value=True)

    # Repository logic replacements
    patch_u_create = patch.object(UserRepository, 'create_user', side_effect=mock_create_user)
    patch_u_get = patch.object(UserRepository, 'get_by_id', side_effect=mock_get_user_by_id)
    patch_u_email = patch.object(UserRepository, 'get_by_email', side_effect=mock_get_user_by_email)
    
    patch_w_create = patch.object(WoundRepository, 'create_wound', side_effect=mock_create_wound)
    patch_w_get = patch.object(WoundRepository, 'get_by_id', side_effect=mock_get_wound_by_id)
    patch_w_list = patch.object(WoundRepository, 'list_by_user', side_effect=mock_list_wounds_by_user)
    
    patch_a_save = patch.object(AnalysisRepository, 'save_analysis', side_effect=mock_save_analysis)
    patch_a_get = patch.object(AnalysisRepository, 'get_by_id', side_effect=mock_get_analysis_by_id)
    patch_a_list = patch.object(AnalysisRepository, 'list_analyses', side_effect=mock_list_analyses)
    
    # Storage logic replacements
    patch_s_orig = patch.object(StorageService, 'upload_original_image', return_value="mock/path.jpg")
    patch_s_grad = patch.object(StorageService, 'upload_gradcam_image', return_value="mock/gradcam.png")
    patch_s_rept = patch.object(StorageService, 'upload_report_image', return_value="mock/report.png")
    patch_s_del = patch.object(StorageService, 'delete_analysis_artifacts', return_value=None)
    patch_s_url = patch.object(StorageService, 'get_signed_url', return_value="https://mock.supabase.co/storage/v1/object/sign/clinical-artifacts/mock.png")
    
    # Supabase Client Initialization patch (prevents actual network call during StorageService init)
    patch_create_client = patch('app.services.storage_service.create_client', return_value=MagicMock())

    with patch_init, patch_health, patch_u_create, patch_u_get, patch_u_email, \
         patch_w_create, patch_w_get, patch_w_list, patch_a_save, patch_a_get, patch_a_list, \
         patch_create_client, patch_s_orig, patch_s_grad, patch_s_rept, patch_s_del, patch_s_url:
        yield

@pytest.fixture(autouse=True)
def clear_mock_db():
    # Clear mock DB for each test function
    mock_db["users"].clear()
    mock_db["wounds"].clear()
    mock_db["analyses"].clear()


