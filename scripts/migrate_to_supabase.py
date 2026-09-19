import os
import sys
import sqlite3
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("migrate_to_supabase")

def run_migration(dry_run: bool = False):
    target = os.environ.get("MIGRATION_TARGET")
    if not target or target.lower() not in ["development", "testing"]:
        logger.error("Safety Check Failed: MIGRATION_TARGET must be 'development' or 'testing'.")
        return

    logger.info(f"--- Starting Migration Preflight (Target: {target}, Dry Run: {dry_run}) ---")
    
    sqlite_db_path = BASE_DIR / "storage" / "database" / "woundinsight.db"
    report_dir = BASE_DIR / "storage" / "reports"

    if not sqlite_db_path.exists():
        logger.error(f"Local SQLite database not found at {sqlite_db_path}. Aborting.")
        return

    sl_conn = sqlite3.connect(str(sqlite_db_path))
    sl_conn.row_factory = sqlite3.Row

    # 1. Source-Side Validation
    users = sl_conn.execute("SELECT * FROM users").fetchall()
    wounds = sl_conn.execute("SELECT * FROM wounds").fetchall()
    analyses = sl_conn.execute("SELECT * FROM analyses").fetchall()

    user_ids = {u["id"] for u in users}
    wound_ids = {w["id"]: w["user_id"] for w in wounds}
    
    stats = {
        "users": {"total": len(users), "valid": 0, "invalid": 0},
        "wounds": {"total": len(wounds), "valid": 0, "invalid": 0},
        "analyses": {"total": len(analyses), "valid": 0, "invalid": 0},
        "artifacts": {"originals_missing": 0, "reports_missing": 0, "gradcams_missing": 0, "storage_conflicts": 0}
    }

    # Validate Users
    for u in users:
        if u["id"] and u["email"]: stats["users"]["valid"] += 1
        else: stats["users"]["invalid"] += 1

    # Validate Wounds
    for w in wounds:
        if w["id"] and w["user_id"] in user_ids: stats["wounds"]["valid"] += 1
        else: stats["wounds"]["invalid"] += 1

    # Validate Analyses & Files
    valid_analyses = []
    for a in analyses:
        is_valid = True
        a_id = a["analysis_id"]
        w_id = a["wound_id"]
        
        if not a_id or w_id not in wound_ids:
            is_valid = False
        else:
            old_img_path = Path(a["image_storage_path"])
            if not old_img_path.exists():
                stats["artifacts"]["originals_missing"] += 1
                is_valid = False
            
            old_report_path = Path(a["report_storage_path"])
            if not old_report_path.exists():
                stats["artifacts"]["reports_missing"] += 1

            old_gradcam_path = report_dir / f"gradcam_{a_id}.png"
            if not old_gradcam_path.exists():
                stats["artifacts"]["gradcams_missing"] += 1

        if is_valid:
            stats["analyses"]["valid"] += 1
            valid_analyses.append(a)
        else:
            stats["analyses"]["invalid"] += 1

    # Output Preflight Report
    logger.info("Source Validation Report:")
    logger.info(f"  Users: Total {stats['users']['total']} | Valid {stats['users']['valid']} | Invalid {stats['users']['invalid']}")
    logger.info(f"  Wounds: Total {stats['wounds']['total']} | Valid {stats['wounds']['valid']} | Invalid {stats['wounds']['invalid']}")
    logger.info(f"  Analyses: Total {stats['analyses']['total']} | Valid {stats['analyses']['valid']} | Invalid {stats['analyses']['invalid']}")
    logger.info(f"  Artifacts Missing: Originals {stats['artifacts']['originals_missing']} | Reports {stats['artifacts']['reports_missing']} | GradCAMs {stats['artifacts']['gradcams_missing']}")

    if dry_run and (not os.environ.get("DATABASE_URL") or "dummy" in os.environ.get("DATABASE_URL", "")):
        logger.info("Dry run requested without real credentials. Source validation complete. Skipping target validation.")
        return

    # 2. Target Connectivity & Target Validation
    import psycopg2
    from psycopg2.extras import Json
    from supabase import create_client, Client
    from app.config import settings

    logger.info("Connecting to target infrastructure...")
    try:
        supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
        pg_conn = psycopg2.connect(settings.DATABASE_URL)
        pg_conn.autocommit = False
    except Exception as e:
        logger.error(f"Failed to connect to target infrastructure: {e}")
        return

    bucket_name = "clinical-artifacts"

    def artifact_exists(key: str) -> bool:
        # Check storage target existence using prefix list
        folder = key.rsplit("/", 1)[0]
        filename = key.rsplit("/", 1)[1]
        try:
            res = supabase.storage.from_(bucket_name).list(folder)
            if res:
                for item in res:
                    if item["name"] == filename: return True
        except Exception:
            pass
        return False

    try:
        if not dry_run:
            with pg_conn.cursor() as cur:
                # Migrate Users
                for u in users:
                    cur.execute("INSERT INTO users (id, email, password_hash, name, created_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                                (u["id"], u["email"], u["password_hash"], u["name"], u["created_at"]))
                # Migrate Wounds
                for w in wounds:
                    cur.execute("INSERT INTO wounds (id, user_id, name, location, created_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                                (w["id"], w["user_id"], w["name"], w["location"], w["created_at"]))
            pg_conn.commit()

        # Migrate Analyses & Artifacts safely
        with pg_conn.cursor() as cur:
            for a in valid_analyses:
                a_id = a["analysis_id"]
                u_id = a["user_id"]
                w_id = a["wound_id"]
                
                # Setup Paths
                old_img_path = Path(a["image_storage_path"])
                new_img_key = f"users/{u_id}/wounds/{w_id}/analyses/{a_id}/original_{old_img_path.name}"
                
                old_report_path = Path(a["report_storage_path"])
                new_report_key = f"users/{u_id}/wounds/{w_id}/analyses/{a_id}/report.png"
                
                old_gradcam_path = report_dir / f"gradcam_{a_id}.png"
                new_gradcam_key = f"users/{u_id}/wounds/{w_id}/analyses/{a_id}/gradcam.png"
                
                # Check DB Idempotency
                cur.execute("SELECT 1 FROM analyses WHERE analysis_id = %s", (a_id,))
                db_exists = bool(cur.fetchone())

                # Check Storage Idempotency
                img_conflict = artifact_exists(new_img_key)
                report_conflict = artifact_exists(new_report_key) if old_report_path.exists() else False
                gradcam_conflict = artifact_exists(new_gradcam_key) if old_gradcam_path.exists() else False

                if img_conflict or report_conflict or gradcam_conflict:
                    stats["artifacts"]["storage_conflicts"] += 1
                    logger.warning(f"Storage conflict for analysis {a_id}. Artifacts already exist. Skipping upload.")
                else:
                    if not dry_run:
                        with open(old_img_path, "rb") as f:
                            supabase.storage.from_(bucket_name).upload(new_img_key, f.read())
                        if old_report_path.exists():
                            with open(old_report_path, "rb") as f:
                                supabase.storage.from_(bucket_name).upload(new_report_key, f.read(), file_options={"content-type": "image/png"})
                        if old_gradcam_path.exists():
                            with open(old_gradcam_path, "rb") as f:
                                supabase.storage.from_(bucket_name).upload(new_gradcam_key, f.read(), file_options={"content-type": "image/png"})
                
                # Insert if DB doesn't exist
                if not db_exists and not dry_run:
                    probs_json = json.loads(a["etiology_probabilities"]) if a["etiology_probabilities"] else {}
                    expl_meta = json.loads(a["explainability_metadata"]) if a["explainability_metadata"] else None

                    cur.execute(
                        """
                        INSERT INTO analyses (
                            analysis_id, user_id, wound_id, original_filename, image_storage_path, report_storage_path,
                            created_at, predicted_etiology, etiology_confidence, etiology_probabilities,
                            wound_detected, wound_area_pixels, wound_area_cm2, perimeter_mm, circularity, is_irregular,
                            fibrin_slough_percent, granulation_percent, callus_percent,
                            severity_score, severity_grade, recommended_action,
                            ai_confidence_score, clinician_review_flag, explainability_metadata, inference_time_ms
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                        (
                            a_id, u_id, w_id, a["original_filename"], new_img_key, new_report_key,
                            a["created_at"], a["predicted_etiology"], a["etiology_confidence"], Json(probs_json),
                            bool(a["wound_detected"]), a["wound_area_pixels"], a["wound_area_cm2"], 
                            a["perimeter_mm"], a["circularity"], bool(a["is_irregular"]),
                            a["fibrin_slough_percent"], a["granulation_percent"], a["callus_percent"],
                            a["severity_score"], a["severity_grade"], a["recommended_action"],
                            a["ai_confidence_score"], bool(a["clinician_review_flag"]), 
                            Json(expl_meta) if expl_meta else None, a["inference_time_ms"]
                        )
                    )
                    pg_conn.commit()

        if stats["artifacts"]["storage_conflicts"] > 0:
            logger.warning(f"Target Validation: Found {stats['artifacts']['storage_conflicts']} analyses with pre-existing storage conflicts.")
        
        if dry_run:
            logger.info("Dry run complete. No modifications were made to target infrastructure.")
        else:
            logger.info("Migration strictly completed.")

    except Exception as e:
        if not dry_run: pg_conn.rollback()
        logger.error(f"Migration failed: {e}", exc_info=True)
    finally:
        sl_conn.close()
        pg_conn.close()

if __name__ == "__main__":
    is_dry_run = "--dry-run" in sys.argv
    run_migration(dry_run=is_dry_run)
