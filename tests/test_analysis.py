import sys
import time
from pathlib import Path
from fastapi.testclient import TestClient

API_ROOT = Path(__file__).resolve().parent.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.main import app

def test_inference_pipeline_on_real_image():
    sample_path = API_ROOT / "tests" / "sample_images" / "sample_dfu.png"
    assert sample_path.exists(), f"Sample test image missing at {sample_path}"

    with TestClient(app) as client:
        with open(sample_path, "rb") as f:
            img_bytes = f.read()

        t0 = time.perf_counter()
        response = client.post(
            "/api/v1/analyses",
            files={"image": ("sample_dfu.png", img_bytes, "image/png")},
            data={"user_id": "clinician_test_01"}
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        # Invariants
        assert data["status"] == "success"
        assert len(data["analysis_id"]) > 10
        assert data["user_id"] == "clinician_test_01"
        assert data["original_filename"] == "sample_dfu.png"

        # 1. Wound Morphometrics
        w = data["wound"]
        assert isinstance(w["detected"], bool)
        assert isinstance(w["area_pixels"], int)
        assert isinstance(w["area_cm2"], (int, float))
        assert isinstance(w["perimeter_mm"], (int, float))
        assert 0.0 <= w["circularity"] <= 1.0
        assert isinstance(w["is_irregular"], bool)

        # 2. Tissue Breakdown
        t = data["tissue"]
        assert 0.0 <= t["fibrin_slough_percent"] <= 100.0
        assert 0.0 <= t["granulation_percent"] <= 100.0
        assert 0.0 <= t["callus_percent"] <= 100.0
        total_tissue = t["fibrin_slough_percent"] + t["granulation_percent"] + t["callus_percent"]
        assert total_tissue <= 100.1, f"Tissue sum exceeds 100%: {total_tissue}"

        # 3. Etiology Classification
        e = data["etiology"]
        assert e["predicted_type"] in ["Diabetic Foot Ulcer", "Pressure Ulcer", "Surgical Wound", "Venous Ulcer"]
        assert 0.0 <= e["confidence"] <= 100.0
        probs = e["probabilities"]
        for k in ["DFU", "Pressure", "Surgical", "Venous"]:
            assert k in probs
        prob_sum = sum(probs.values())
        assert 99.0 <= prob_sum <= 101.0

        # 4. Severity Assessment
        s = data["severity"]
        assert 0.0 <= s["severity_score"] <= 100.0
        assert s["severity_grade"] in ["Low (Mild)", "Moderate (Chronic Risk)", "High (Severe / Stagnant)", "Critical (Urgent)"]
        assert len(s["recommended_action"]) > 5

        # 5. Safety & Epistemic Uncertainty
        u = data["uncertainty"]
        assert 0.0 <= u["ai_confidence_score"] <= 100.0
        assert isinstance(u["requires_clinician_review"], bool)

        # 6. Visualizations
        vis = data["visualizations"]
        assert "report_image_url" in vis
        assert "report_filename" in vis
        assert "original_image_url" in vis

        # 7. Explainability
        expl = data.get("explainability")
        assert expl is not None
        assert expl["available"] is True
        assert "Dual-Branch" in expl["method"]
        assert "layer4" in expl["global_target_layer"]
        assert "layer4" in expl["roi_target_layer"]

        print(f"\n[Real Image Inference] SUCCESS ({elapsed_ms:.1f} ms)")
        print(f"  Diagnosis: {e['predicted_type']} ({e['confidence']}%)")
        print(f"  Wound Area: {w['area_pixels']} px ({w['area_cm2']} cm2)")
        print(f"  Tissue: Gran: {t['granulation_percent']}% | Fibrin: {t['fibrin_slough_percent']}% | Callus: {t['callus_percent']}%")
        print(f"  Severity: {s['severity_grade']} ({s['severity_score']}/100)")
        print(f"  AI Confidence: {u['ai_confidence_score']}% (Review: {u['requires_clinician_review']})")

def test_legacy_analyze_endpoint():
    sample_path = API_ROOT / "tests" / "sample_images" / "sample_wound.jpg"
    assert sample_path.exists()

    with TestClient(app) as client:
        with open(sample_path, "rb") as f:
            img_bytes = f.read()

        response = client.post(
            "/analyze",
            files={"image": ("sample_wound.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"

def test_error_handling_invalid_inputs():
    with TestClient(app) as client:
        # 1. Non-image text file
        r_txt = client.post(
            "/api/v1/analyses",
            files={"image": ("note.txt", b"This is a text note, not an image.", "text/plain")}
        )
        assert r_txt.status_code == 400
        assert "Unsupported" in r_txt.json()["detail"]

        # 2. Corrupted file
        r_corrupt = client.post(
            "/api/v1/analyses",
            files={"image": ("corrupted.png", b"CORRUPTED_NOT_A_PNG_HEADER_DATA", "image/png")}
        )
        assert r_corrupt.status_code == 400
        assert "not a valid" in r_corrupt.json()["detail"]

        # 3. Empty 0-byte file
        r_empty = client.post(
            "/api/v1/analyses",
            files={"image": ("empty.jpg", b"", "image/jpeg")}
        )
        assert r_empty.status_code == 400
        assert "empty" in r_empty.json()["detail"]

if __name__ == "__main__":
    test_inference_pipeline_on_real_image()
    test_legacy_analyze_endpoint()
    test_error_handling_invalid_inputs()
    print("Analysis tests PASSED.")
