import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from app.db.database import get_db_connection
from app.models.database_models import AnalysisRecordModel, UserModel, WoundModel

logger = logging.getLogger("wound_api.repo")

class UserRepository:
    """Repository handling persistence operations for user accounts."""

    @staticmethod
    def create_user(user: UserModel) -> UserModel:
        sql = """
        INSERT INTO users (id, email, password_hash, name, created_at)
        VALUES (?, ?, ?, ?, ?);
        """
        with get_db_connection() as conn:
            conn.execute(sql, (user.id, user.email, user.password_hash, user.name, user.created_at))
        logger.info(f"Registered new user account: {user.email} (id: {user.id})")
        return user

    @staticmethod
    def get_by_id(user_id: str) -> Optional[UserModel]:
        sql = "SELECT id, email, password_hash, name, created_at FROM users WHERE id = ?;"
        with get_db_connection() as conn:
            cur = conn.execute(sql, (user_id,))
            row = cur.fetchone()
            if not row:
                return None
            return UserModel(
                id=row["id"],
                email=row["email"],
                password_hash=row["password_hash"],
                name=row["name"],
                created_at=row["created_at"]
            )

    @staticmethod
    def get_by_email(email: str) -> Optional[UserModel]:
        sql = "SELECT id, email, password_hash, name, created_at FROM users WHERE LOWER(email) = LOWER(?);"
        with get_db_connection() as conn:
            cur = conn.execute(sql, (email,))
            row = cur.fetchone()
            if not row:
                return None
            return UserModel(
                id=row["id"],
                email=row["email"],
                password_hash=row["password_hash"],
                name=row["name"],
                created_at=row["created_at"]
            )

class WoundRepository:
    """Repository handling persistence operations for patient wound profiles."""

    @staticmethod
    def create_wound(wound: WoundModel) -> WoundModel:
        sql = """
        INSERT INTO wounds (id, user_id, name, location, created_at)
        VALUES (?, ?, ?, ?, ?);
        """
        with get_db_connection() as conn:
            conn.execute(sql, (wound.id, wound.user_id, wound.name, wound.location, wound.created_at))
        logger.info(f"Created new wound profile '{wound.name}' (id: {wound.id}) for user {wound.user_id}")
        return wound

    @staticmethod
    def get_by_id(wound_id: str) -> Optional[WoundModel]:
        sql = "SELECT id, user_id, name, location, created_at FROM wounds WHERE id = ?;"
        with get_db_connection() as conn:
            cur = conn.execute(sql, (wound_id,))
            row = cur.fetchone()
            if not row:
                return None
            return WoundModel(
                id=row["id"],
                user_id=row["user_id"],
                name=row["name"],
                location=row["location"],
                created_at=row["created_at"]
            )

    @staticmethod
    def list_by_user(user_id: str) -> List[WoundModel]:
        sql = "SELECT id, user_id, name, location, created_at FROM wounds WHERE user_id = ? ORDER BY created_at DESC;"
        with get_db_connection() as conn:
            cur = conn.execute(sql, (user_id,))
            rows = cur.fetchall()
            return [
                WoundModel(
                    id=row["id"],
                    user_id=row["user_id"],
                    name=row["name"],
                    location=row["location"],
                    created_at=row["created_at"]
                )
                for row in rows
            ]

class AnalysisRepository:
    """Repository handling persistence operations for wound analysis records."""

    @staticmethod
    def save_analysis(record: AnalysisRecordModel) -> str:
        sql = """
        INSERT INTO analyses (
            analysis_id, user_id, wound_id, original_filename, image_storage_path, report_storage_path,
            created_at, predicted_etiology, etiology_confidence, etiology_probabilities,
            wound_detected, wound_area_pixels, wound_area_cm2, perimeter_mm, circularity, is_irregular,
            fibrin_slough_percent, granulation_percent, callus_percent,
            severity_score, severity_grade, recommended_action,
            ai_confidence_score, clinician_review_flag, explainability_metadata, inference_time_ms
        ) VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?
        );
        """
        params = (
            record.analysis_id,
            record.user_id,
            record.wound_id,
            record.original_filename,
            record.image_storage_path,
            record.report_storage_path,
            record.created_at,
            record.predicted_etiology,
            record.etiology_confidence,
            json.dumps(record.etiology_probabilities),
            1 if record.wound_detected else 0,
            record.wound_area_pixels,
            record.wound_area_cm2,
            record.perimeter_mm,
            record.circularity,
            1 if record.is_irregular else 0,
            record.fibrin_slough_percent,
            record.granulation_percent,
            record.callus_percent,
            record.severity_score,
            record.severity_grade,
            record.recommended_action,
            record.ai_confidence_score,
            1 if record.clinician_review_flag else 0,
            json.dumps(record.explainability_metadata) if record.explainability_metadata else None,
            record.inference_time_ms
        )
        with get_db_connection() as conn:
            conn.execute(sql, params)
        logger.info(f"Saved analysis record {record.analysis_id} into database.")
        return record.analysis_id

    @staticmethod
    def get_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM analyses WHERE analysis_id = ?;"
        with get_db_connection() as conn:
            cur = conn.execute(sql, (analysis_id,))
            row = cur.fetchone()
            if not row:
                return None
            return AnalysisRepository._row_to_dict(row)

    @staticmethod
    def list_analyses(
        limit: int = 50,
        offset: int = 0,
        user_id: Optional[str] = None,
        wound_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        where_clauses = []
        params: list = []

        if user_id:
            where_clauses.append("user_id = ?")
            params.append(user_id)
        if wound_id:
            where_clauses.append("wound_id = ?")
            params.append(wound_id)

        where_str = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        count_sql = f"SELECT COUNT(*) FROM analyses{where_str}"
        list_sql = f"SELECT * FROM analyses{where_str} ORDER BY created_at DESC LIMIT ? OFFSET ?;"

        with get_db_connection() as conn:
            total_cur = conn.execute(count_sql, params)
            total_count = total_cur.fetchone()[0]

            query_params = list(params) + [limit, offset]
            list_cur = conn.execute(list_sql, query_params)
            rows = list_cur.fetchall()
            items = [AnalysisRepository._row_to_summary(row) for row in rows]

        return items, total_count

    @staticmethod
    def _row_to_summary(row) -> Dict[str, Any]:
        return {
            "analysis_id": row["analysis_id"],
            "created_at": row["created_at"],
            "original_filename": row["original_filename"],
            "user_id": row["user_id"],
            "wound_id": row["wound_id"] if "wound_id" in row.keys() else None,
            "predicted_etiology": row["predicted_etiology"],
            "etiology_confidence": row["etiology_confidence"],
            "wound_area_cm2": row["wound_area_cm2"],
            "severity_grade": row["severity_grade"],
            "severity_score": row["severity_score"],
            "ai_confidence_score": row["ai_confidence_score"],
            "clinician_review_flag": bool(row["clinician_review_flag"]),
            "report_image_url": f"/api/v1/analyses/{row['analysis_id']}/report"
        }

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        probs = json.loads(row["etiology_probabilities"]) if row["etiology_probabilities"] else {}
        expl = json.loads(row["explainability_metadata"]) if row["explainability_metadata"] else None

        return {
            "analysis_id": row["analysis_id"],
            "status": "success",
            "timestamp": row["created_at"],
            "original_filename": row["original_filename"],
            "user_id": row["user_id"],
            "wound_id": row["wound_id"] if "wound_id" in row.keys() else None,
            "wound": {
                "detected": bool(row["wound_detected"]),
                "area_pixels": row["wound_area_pixels"],
                "area_cm2": row["wound_area_cm2"],
                "perimeter_mm": row["perimeter_mm"],
                "circularity": row["circularity"],
                "is_irregular": bool(row["is_irregular"]),
            },
            "tissue": {
                "fibrin_slough_percent": row["fibrin_slough_percent"],
                "granulation_percent": row["granulation_percent"],
                "callus_percent": row["callus_percent"],
            },
            "etiology": {
                "predicted_type": row["predicted_etiology"],
                "confidence": row["etiology_confidence"],
                "probabilities": probs,
            },
            "severity": {
                "severity_score": row["severity_score"],
                "severity_grade": row["severity_grade"],
                "recommended_action": row["recommended_action"],
            },
            "uncertainty": {
                "ai_confidence_score": row["ai_confidence_score"],
                "requires_clinician_review": bool(row["clinician_review_flag"]),
            },
            "visualizations": {
                "report_image_url": f"/api/v1/analyses/{row['analysis_id']}/report",
                "report_filename": f"report_{row['analysis_id']}.png",
                "original_image_url": f"/api/v1/analyses/{row['analysis_id']}/image",
                "gradcam_image_url": f"/api/v1/analyses/{row['analysis_id']}/gradcam"
            },
            "explainability": expl,
            "academic_notice": "Academic prototype. Not certified for standalone clinical diagnostic decisions.",
            "_meta": {
                "inference_time_ms": row["inference_time_ms"]
            }
        }
