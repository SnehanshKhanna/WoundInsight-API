import os
import json
import sqlite3
from pathlib import Path
import uuid

BASE_DIR = Path(r"c:\Users\Lenovo\Desktop\Final Year Project\WoundInsight-API")
sqlite_db_path = BASE_DIR / "storage" / "database" / "woundinsight.db"
report_dir = BASE_DIR / "storage" / "reports"

def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except:
        return False

def audit():
    if not sqlite_db_path.exists():
        print(f"ERROR: SQLite DB not found at {sqlite_db_path}")
        return
        
    conn = sqlite3.connect(sqlite_db_path)
    conn.row_factory = sqlite3.Row
    
    users = conn.execute("SELECT * FROM users").fetchall()
    wounds = conn.execute("SELECT * FROM wounds").fetchall()
    analyses = conn.execute("SELECT * FROM analyses").fetchall()
    
    user_ids = [u["id"] for u in users]
    wound_ids = [w["id"] for w in wounds]
    analysis_ids = [a["analysis_id"] for a in analyses]
    
    print("--- TASK 2: Source dataset audit ---")
    print(f"A. users: {len(users)}")
    print(f"B. wounds: {len(wounds)}")
    print(f"C. analyses: {len(analyses)}")
    
    # D. Relationship Integrity
    analyses_bad_user = [a for a in analyses if a["user_id"] not in user_ids]
    analyses_bad_wound = [a for a in analyses if a["wound_id"] not in wound_ids]
    wounds_bad_user = [w for w in wounds if w["user_id"] not in user_ids]
    
    print("\nD. relationship integrity:")
    print(f"   analyses with no matching user: {len(analyses_bad_user)}")
    print(f"   analyses with no matching wound: {len(analyses_bad_wound)}")
    print(f"   wounds with no matching user: {len(wounds_bad_user)}")
    
    # E. Artifact Availability
    missing_originals = 0
    missing_reports = 0
    missing_gradcams = 0
    total_originals = 0
    total_reports = 0
    total_gradcams = 0
    
    for a in analyses:
        a_id = a["analysis_id"]
        # Original
        orig = Path(a["image_storage_path"])
        if orig.exists():
            total_originals += 1
        else:
            missing_originals += 1
            
        # Report
        rep = Path(a["report_storage_path"])
        if rep.exists():
            total_reports += 1
        else:
            missing_reports += 1
            
        # GradCAM
        gc = report_dir / f"gradcam_{a_id}.png"
        if gc.exists():
            total_gradcams += 1
        else:
            missing_gradcams += 1
            
    print("\nE. artifact availability:")
    print(f"   analyses with missing original image: {missing_originals}")
    print(f"   analyses with missing report: {missing_reports}")
    print(f"   analyses with missing Grad-CAM: {missing_gradcams}")
    print(f"   total existing original images: {total_originals}")
    print(f"   total existing reports: {total_reports}")
    print(f"   total existing Grad-CAM files: {total_gradcams}")
    
    # F. Duplicates
    dup_users = len(user_ids) - len(set(user_ids))
    dup_wounds = len(wound_ids) - len(set(wound_ids))
    dup_analyses = len(analysis_ids) - len(set(analysis_ids))
    
    target_paths = set()
    dup_targets = 0
    for a in analyses:
        p = f"users/{a['user_id']}/wounds/{a['wound_id']}/analyses/{a['analysis_id']}/original_{Path(a['image_storage_path']).name}"
        if p in target_paths: dup_targets += 1
        target_paths.add(p)
        
    print("\nF. duplicate/conflicting IDs:")
    print(f"   duplicate user IDs: {dup_users}")
    print(f"   duplicate wound IDs: {dup_wounds}")
    print(f"   duplicate analysis IDs: {dup_analyses}")
    print(f"   duplicate target Storage paths: {dup_targets}")
    
    # G. Malformed data
    invalid_uuids = 0
    for lst in (user_ids, wound_ids, analysis_ids):
        for id_val in lst:
            if not is_valid_uuid(id_val): invalid_uuids += 1
            
    malformed_json_probs = 0
    malformed_json_expl = 0
    invalid_nulls = 0
    
    for a in analyses:
        if a["etiology_probabilities"]:
            try: json.loads(a["etiology_probabilities"])
            except: malformed_json_probs += 1
        if a["explainability_metadata"]:
            try: json.loads(a["explainability_metadata"])
            except: malformed_json_expl += 1
            
        if a["created_at"] is None or a["image_storage_path"] is None:
            invalid_nulls += 1
            
    for u in users:
        if u["email"] is None or u["password_hash"] is None:
            invalid_nulls += 1
            
    print("\nG. malformed data:")
    print(f"   invalid UUIDs: {invalid_uuids}")
    print(f"   malformed JSON in etiology_probabilities: {malformed_json_probs}")
    print(f"   malformed JSON in explainability_metadata: {malformed_json_expl}")
    print(f"   invalid/null values in required fields: {invalid_nulls}")

if __name__ == "__main__":
    audit()
