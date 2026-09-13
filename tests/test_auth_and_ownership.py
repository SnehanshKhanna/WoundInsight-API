import sys
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import init_db

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

import uuid

def test_auth_and_ownership_flow(client):
    uid_a = uuid.uuid4().hex[:8]
    email_a = f"alice_{uid_a}@example.com"
    uid_b = uuid.uuid4().hex[:8]
    email_b = f"bob_{uid_b}@example.com"

    # 1. Register User A
    reg_a = client.post(
        "/api/v1/auth/register",
        json={"email": email_a, "password": "password123", "name": "Alice Smith"}
    )
    assert reg_a.status_code == 201, reg_a.text
    token_a = reg_a.json()["access_token"]
    user_a_id = reg_a.json()["user"]["id"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # Duplicate registration should fail
    dup = client.post(
        "/api/v1/auth/register",
        json={"email": email_a, "password": "password123"}
    )
    assert dup.status_code == 400

    # 2. Login User A
    login_fail = client.post(
        "/api/v1/auth/login",
        json={"email": email_a, "password": "wrongpassword"}
    )
    assert login_fail.status_code == 401

    login_ok = client.post(
        "/api/v1/auth/login",
        json={"email": email_a, "password": "password123"}
    )
    assert login_ok.status_code == 200
    assert "access_token" in login_ok.json()

    # 3. GET /me
    me_resp = client.get("/api/v1/auth/me", headers=headers_a)
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == email_a
    assert me_resp.json()["id"] == user_a_id

    # 4. Register User B
    reg_b = client.post(
        "/api/v1/auth/register",
        json={"email": email_b, "password": "password456", "name": "Bob Jones"}
    )
    assert reg_b.status_code == 201
    token_b = reg_b.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # 5. User A creates Wound 1
    w_create = client.post(
        "/api/v1/wounds",
        headers=headers_a,
        json={"name": "Left Heel Ulcer", "location": "Left Heel"}
    )
    assert w_create.status_code == 201, w_create.text
    wound_a_id = w_create.json()["id"]
    assert w_create.json()["user_id"] == user_a_id

    # 6. User A lists wounds
    w_list_a = client.get("/api/v1/wounds", headers=headers_a)
    assert w_list_a.status_code == 200
    assert len(w_list_a.json()["wounds"]) >= 1

    # User B lists wounds (should be empty for Bob)
    w_list_b = client.get("/api/v1/wounds", headers=headers_b)
    assert w_list_b.status_code == 200
    assert len(w_list_b.json()["wounds"]) == 0

    # User B cannot access User A's wound
    w_get_unauth = client.get(f"/api/v1/wounds/{wound_a_id}", headers=headers_b)
    assert w_get_unauth.status_code == 404

    # 7. User B tries to analyze with User A's wound_id -> should fail
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), color=(200, 100, 100)).save(buf, format="PNG")
    img_bytes = buf.getvalue()

    spoof_analysis = client.post(
        "/api/v1/analyses",
        headers=headers_b,
        data={"wound_id": wound_a_id},
        files={"image": ("test.png", img_bytes, "image/png")}
    )
    assert spoof_analysis.status_code == 400
    assert "not belong" in spoof_analysis.text

    # 8. User A analyzes with User A's wound_id -> should succeed
    real_analysis = client.post(
        "/api/v1/analyses",
        headers=headers_a,
        data={"wound_id": wound_a_id},
        files={"image": ("test.png", img_bytes, "image/png")}
    )
    assert real_analysis.status_code == 200, real_analysis.text
    analysis_data = real_analysis.json()
    analysis_id = analysis_data["analysis_id"]
    assert analysis_data["user_id"] == user_a_id
    assert analysis_data["wound_id"] == wound_a_id
    assert "gradcam_image_url" in analysis_data["visualizations"]

    # 9. Isolation on GET analysis
    get_ok = client.get(f"/api/v1/analyses/{analysis_id}", headers=headers_a)
    assert get_ok.status_code == 200

    get_denied = client.get(f"/api/v1/analyses/{analysis_id}", headers=headers_b)
    assert get_denied.status_code == 404

    # 10. Isolation on report and image
    img_ok = client.get(f"/api/v1/analyses/{analysis_id}/image", headers=headers_a)
    assert img_ok.status_code == 200

    img_denied = client.get(f"/api/v1/analyses/{analysis_id}/image", headers=headers_b)
    assert img_denied.status_code == 404

    rep_ok = client.get(f"/api/v1/analyses/{analysis_id}/report", headers=headers_a)
    assert rep_ok.status_code == 200

    rep_denied = client.get(f"/api/v1/analyses/{analysis_id}/report", headers=headers_b)
    assert rep_denied.status_code == 404

    # 11. Standalone Grad-CAM endpoint
    cam_ok = client.get(f"/api/v1/analyses/{analysis_id}/gradcam", headers=headers_a)
    assert cam_ok.status_code == 200
    assert cam_ok.headers["content-type"] == "image/png"
