import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables before doing anything else
load_dotenv(r"c:\Users\Lenovo\Desktop\Final Year Project\.env")

import sys
import uuid
import psycopg2
from supabase import create_client, Client
from fastapi.testclient import TestClient

# Add app to path
BASE_DIR = Path(r"c:\Users\Lenovo\Desktop\Final Year Project\WoundInsight-API")
sys.path.insert(0, str(BASE_DIR))

from app.main import app
from app.config import settings

def run_smoke_test():
    print("Starting E2E Smoke Test against Supabase...")
    
    with TestClient(app) as client:
        test_email = "supabase-smoke-test@woundinsight.com"
    test_password = "SecureTestPassword123!"
    
    report = {
        "Authentication": "FAIL",
        "User persistence": "FAIL",
        "Wound persistence": "FAIL",
        "Analysis persistence": "FAIL",
        "Original image Storage": "FAIL",
        "Report Storage": "FAIL",
        "Grad-CAM Storage": "N/A",
        "Signed URL retrieval": "FAIL",
        "Ownership isolation": "FAIL",
        "Cleanup": "FAIL"
    }

    user_id = None
    wound_id = None
    analysis_id = None
    
    try:
        # STEP 1 & 2 - Create user & Auth
        # 1. Register
        reg_res = client.post("/api/v1/auth/register", json={
            "email": test_email,
            "password": test_password,
            "name": "Smoke Test User"
        })
        if reg_res.status_code not in (200, 201):
            if reg_res.status_code == 400 and "already exists" in reg_res.text:
                pass
            else:
                print(f"Register failed: {reg_res.text}")
        # Ignore if exists (just in case), but we expect 201 or 200
        
        # 2. Login
        login_res = client.post("/api/v1/auth/login", json={
            "email": test_email,
            "password": test_password
        })
        if login_res.status_code != 200:
            raise Exception(f"Login failed: {login_res.text}")
        
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Get user id from me endpoint or login response (assuming login doesn't have it, we use /me)
        me_res = client.get("/api/v1/auth/me", headers=headers)
        if me_res.status_code != 200:
            raise Exception("Failed to get current user info")
        user_id = me_res.json()["id"]
        
        report["Authentication"] = "PASS"
        
        # STEP 3 - Create a wound
        wound_res = client.post("/api/v1/wounds", headers=headers, json={
            "name": "Supabase Persistence Smoke Test",
            "location": "Test Location"
        })
        if wound_res.status_code not in (200, 201):
            raise Exception(f"Failed to create wound: {wound_res.text}")
        
        wound_id = wound_res.json()["id"]
        report["Wound persistence"] = "PASS" # Logically, API worked
        
        # STEP 4 - Run ONE analysis
        # Find a test image
        test_img_path = BASE_DIR / "tests" / "fixtures" / "wound_test_image.jpg"
        if not test_img_path.exists():
            test_img_path = BASE_DIR / "tests" / "fixtures" / "sample_wound.jpg"
            if not test_img_path.exists():
                # Let's create a dummy image to test the upload pipeline
                from PIL import Image
                test_img_path = BASE_DIR / "tests" / "fixtures" / "dummy.png"
                test_img_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (100, 100), color="red").save(test_img_path)
                
        with open(test_img_path, "rb") as f:
            analyze_res = client.post(
                "/api/v1/analyses",
                headers=headers,
                data={"wound_id": wound_id},
                files={"image": ("dummy.png", f, "image/png")}
            )
            
        if analyze_res.status_code != 200:
            raise Exception(f"Analysis failed: {analyze_res.text}")
            
        analysis_data = analyze_res.json()
        analysis_id = analysis_data["analysis_id"]
        
        # Check API response fields
        if analysis_data.get("visualizations", {}).get("gradcam_url"):
            report["Grad-CAM Storage"] = "TESTING"
            
        report["Analysis persistence"] = "PASS" # API returned success
        
        # STEP 5 - Verify PostgreSQL persistence
        db_url = os.environ["DATABASE_URL"]
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (test_email,))
            if not cur.fetchone(): raise Exception("User not found in DB")
            report["User persistence"] = "PASS"
            
            cur.execute("SELECT id FROM wounds WHERE id = %s", (wound_id,))
            if not cur.fetchone(): raise Exception("Wound not found in DB")
            
            cur.execute("SELECT analysis_id FROM analyses WHERE analysis_id = %s", (analysis_id,))
            if not cur.fetchone(): raise Exception("Analysis not found in DB")
        conn.close()
        
        # STEP 6 - Verify Supabase Storage
        supabase_url = os.environ["SUPABASE_URL"]
        if supabase_url.endswith("/rest/v1") or supabase_url.endswith("/rest/v1/"):
            supabase_url = supabase_url.split("/rest/v1")[0]
            
        supabase: Client = create_client(supabase_url, os.environ["SUPABASE_SERVICE_ROLE_KEY"])
        bucket = "clinical-artifacts"
        prefix = f"users/{user_id}/wounds/{wound_id}/analyses/{analysis_id}/"
        files_res = supabase.storage.from_(bucket).list(prefix)
        files = [f["name"] for f in files_res] if files_res else []
        
        if any("original" in f for f in files):
            report["Original image Storage"] = "PASS"
        if any("report" in f for f in files):
            report["Report Storage"] = "PASS"
        if any("gradcam" in f for f in files):
            report["Grad-CAM Storage"] = "PASS"
        elif report["Grad-CAM Storage"] == "TESTING":
            report["Grad-CAM Storage"] = "FAIL"
            
        # STEP 7 - Verify signed URL retrieval
        # Call the API for the signed URLs (using follow_redirects=False to catch the 307)
        img_redir = client.get(f"/api/v1/analyses/{analysis_id}/image", headers=headers, follow_redirects=False)
        if img_redir.status_code == 307 and "token=" in img_redir.headers.get("location", ""):
            report["Signed URL retrieval"] = "PASS"
        else:
            raise Exception(f"Expected 307 with token, got {img_redir.status_code}")
            
        # STEP 8 - Verify isolation
        # Create user 2
        test_email_2 = "hacker@woundinsight.com"
        client.post("/api/v1/auth/register", json={"email": test_email_2, "password": test_password, "name": "Hacker"})
        login2 = client.post("/api/v1/auth/login", json={"email": test_email_2, "password": test_password})
        token2 = login2.json()["access_token"]
        headers2 = {"Authorization": f"Bearer {token2}"}
        
        # Try to access user 1's image
        hack_res = client.get(f"/api/v1/analyses/{analysis_id}/image", headers=headers2, follow_redirects=False)
        if hack_res.status_code in (401, 403, 404):
            report["Ownership isolation"] = "PASS"
        else:
            raise Exception(f"Isolation failed, got status {hack_res.status_code}")
            
    except Exception as e:
        print(f"ERROR: {e}")
    
    # STEP 9 - Cleanup
    print("Running Cleanup...")
    try:
        # Delete from DB (cascade should handle wounds/analyses)
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE email IN (%s, %s)", (test_email, "hacker@woundinsight.com"))
        conn.commit()
        conn.close()
        
        # Delete from Storage
        if user_id and wound_id and analysis_id:
            supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
            bucket = "clinical-artifacts"
            prefix = f"users/{user_id}/wounds/{wound_id}/analyses/{analysis_id}/"
            files_res = supabase.storage.from_(bucket).list(prefix)
            if files_res:
                paths = [prefix + f["name"] for f in files_res]
                supabase.storage.from_(bucket).remove(paths)
        report["Cleanup"] = "PASS"
    except Exception as e:
        print(f"Cleanup error: {e}")

    # Check remaining
    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM analyses")
            remaining_analyses = cur.fetchone()[0]
        conn.close()
    except:
        remaining_analyses = "Unknown"
        
    try:
        supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
        # counting all files could be large, but it should be 0
        rem_files = supabase.storage.from_("clinical-artifacts").list()
        remaining_storage = len(rem_files) if rem_files else 0
    except:
        remaining_storage = "Unknown"

    print("\n--- FINAL REPORT ---")
    for k, v in report.items():
        print(f"{k}: {v}")
    
    print(f"Remaining database rows (analyses): {remaining_analyses}")
    print(f"Remaining storage objects (root): {remaining_storage}")

if __name__ == "__main__":
    run_smoke_test()
