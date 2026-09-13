import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure WoundInsight-API is first on sys.path
API_ROOT = Path(__file__).resolve().parent.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.main import app

def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "WoundInsight" in data["service"]

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True
        assert data["database_connected"] is True
        assert "wound_segmentation" in data["checkpoints"]
        assert "tissue_segmentation" in data["checkpoints"]
        assert "etiology_classifier" in data["checkpoints"]
        assert data["checkpoints"]["wound_segmentation"] == "retrained_best_wound_model.pth"
        assert data["checkpoints"]["tissue_segmentation"] == "retrained_best_tissue_model.pth"
        assert data["checkpoints"]["etiology_classifier"] == "retrained_best_dual_branch_classifier.pth"

if __name__ == "__main__":
    test_root_endpoint()
    test_health_endpoint()
    print("Health tests PASSED.")
