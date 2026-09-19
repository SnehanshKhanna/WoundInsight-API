import sys
from pathlib import Path
from fastapi.testclient import TestClient

API_ROOT = Path(__file__).resolve().parent.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.main import app

def test_persistence_and_retrieval():
    sample_path = API_ROOT / "tests" / "sample_images" / "sample_dfu.png"
    assert sample_path.exists()

    with TestClient(app) as client:
        # Authenticate user and create wound profile
        reg = client.post("/api/v1/auth/register", json={"email": "persist@test.org", "password": "password123", "name": "Persist User"})
        token = reg.json().get("access_token") or client.post("/api/v1/auth/login", json={"email": "persist@test.org", "password": "password123"}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        w_res = client.post("/api/v1/wounds", json={"name": "Persistence Wound"}, headers=headers)
        wound_id = w_res.json()["id"]

        # 1. Post new analysis
        with open(sample_path, "rb") as f:
            img_bytes = f.read()

        r_post = client.post(
            "/api/v1/analyses",
            files={"image": ("sample_dfu.png", img_bytes, "image/png")},
            data={"wound_id": wound_id},
            headers=headers
        )
        assert r_post.status_code == 200
        post_data = r_post.json()
        analysis_id = post_data["analysis_id"]

        # 2. Query record by analysis_id (conforms to AnalysisResponse schema)
        r_get = client.get(f"/api/v1/analyses/{analysis_id}", headers=headers)
        assert r_get.status_code == 200
        get_data = r_get.json()
        assert get_data["analysis_id"] == analysis_id
        assert get_data["user_id"] == post_data["user_id"]
        assert get_data["etiology"]["predicted_type"] == post_data["etiology"]["predicted_type"]
        assert get_data["wound"]["area_pixels"] == post_data["wound"]["area_pixels"]
        assert get_data["severity"]["severity_score"] == post_data["severity"]["severity_score"]

        # 3. Query list of analyses (conforms to AnalysisListResponse schema)
        r_list = client.get("/api/v1/analyses?limit=10&offset=0", headers=headers)
        assert r_list.status_code == 200
        list_data = r_list.json()
        assert list_data["total"] >= 1
        found = any(a["analysis_id"] == analysis_id for a in list_data["analyses"])
        assert found, f"Analysis {analysis_id} not found in listing"

        # 4. Filter by wound_id
        r_wound_list = client.get(f"/api/v1/analyses?wound_id={wound_id}", headers=headers)
        assert r_wound_list.status_code == 200
        wound_list_data = r_wound_list.json()
        assert wound_list_data["total"] >= 1
        assert all(a["wound_id"] == wound_id for a in wound_list_data["analyses"])

        # 5. Retrieve diagnostic report image
        r_report = client.get(f"/api/v1/analyses/{analysis_id}/report", headers=headers, follow_redirects=False)
        assert r_report.status_code == 307
        assert "location" in r_report.headers

        # 6. Retrieve stored original image
        r_orig = client.get(f"/api/v1/analyses/{analysis_id}/image", headers=headers, follow_redirects=False)
        assert r_orig.status_code == 307
        assert "location" in r_orig.headers

        # 7. Non-existent record returns 404
        r_404 = client.get("/api/v1/analyses/nonexistent-uuid-12345", headers=headers)
        assert r_404.status_code == 404

        r_report_404 = client.get("/api/v1/analyses/nonexistent-uuid-12345/report", headers=headers)
        assert r_report_404.status_code == 404

        print("\n[Persistence & Retrieval] ALL TESTS PASSED.")

if __name__ == "__main__":
    test_persistence_and_retrieval()
