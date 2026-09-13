import os
import sys
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import patch
from PIL import Image

API_ROOT = Path(__file__).resolve().parent.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from huggingface_hub import HfApi
from app.config import settings
from app.services.model_manager import model_manager
from app.services.inference_service import inference_service
from app.db.database import init_db

# Expected cryptographic SHA256 hashes of production checkpoints
EXPECTED_SHA256 = {
    "classification/retrained_best_dual_branch_classifier.pth": "3bcaba36ea28eb9b5a72cbfcf800bf75d61569f7d25fba611b3d79b438980ad5",
    "segmentation/retrained_best_wound_model.pth": "ef161f1d0c450d20c58ecdd82619b504e4e33107773bcf3b9a00fd06008ea3d0",
    "tissue_segmentation/retrained_best_tissue_model.pth": "487356f5ebb7aa39c42957f0bdf9fb2ea5a7c92ccad309b8b95e42dbe46521b5"
}

def test_hf_configuration():
    """Verifies that the Hugging Face Hub configuration is properly set."""
    assert settings.HF_MODEL_REPO_ID == "SnehanshKhanna/WoundInsight-models", (
        f"Unexpected HF_MODEL_REPO_ID: {settings.HF_MODEL_REPO_ID}"
    )
    # Token should not be required for public repo
    assert settings.HF_TOKEN is None or isinstance(settings.HF_TOKEN, str)

    # Check that remote paths are correctly configured
    assert settings.WOUND_MODEL_REMOTE_PATH == "segmentation/retrained_best_wound_model.pth"
    assert settings.TISSUE_MODEL_REMOTE_PATH == "tissue_segmentation/retrained_best_tissue_model.pth"
    assert settings.CLASSIFIER_MODEL_REMOTE_PATH == "classification/retrained_best_dual_branch_classifier.pth"
    print("  [PASS] Hugging Face Hub configuration parameters verified.")

def test_hf_remote_resolution():
    """Verifies that each of the three expected remote files can be resolved on Hugging Face Hub."""
    api = HfApi()
    info = api.model_info(repo_id=settings.HF_MODEL_REPO_ID, files_metadata=True)
    assert info is not None

    remote_files = {s.rfilename: s for s in info.siblings}
    for remote_path, exp_hash in EXPECTED_SHA256.items():
        assert remote_path in remote_files, f"Remote file {remote_path} missing in repo {settings.HF_MODEL_REPO_ID}"
        file_meta = remote_files[remote_path]
        assert file_meta.size > 100_000_000, f"Remote file {remote_path} size is unexpectedly small: {file_meta.size}"
        if file_meta.lfs:
            assert file_meta.lfs.sha256 == exp_hash, (
                f"Remote SHA256 mismatch for {remote_path}: {file_meta.lfs.sha256} vs {exp_hash}"
            )

    print("  [PASS] Remote HF files resolution and cryptographic SHA256 hashes verified.")

def test_hf_caching_behavior():
    """Verifies that existing local checkpoints are recognized as cached and NO download occurs."""
    specs = model_manager.get_model_specs()
    for key, spec in specs.items():
        local_path = spec["local_path"]
        assert local_path.exists(), f"Local checkpoint does not exist: {local_path}"
        assert model_manager.is_cached(local_path, min_bytes=spec["expected_min_bytes"]), (
            f"Checkpoint not recognized as cached: {local_path}"
        )

    # Mock hf_hub_download to ensure it is NEVER called when models are already cached
    with patch("app.services.model_manager.hf_hub_download") as mock_download:
        for key in ["wound_segmentation", "tissue_segmentation", "classification"]:
            resolved = model_manager.download_model(key, force=False)
            assert resolved.exists()

        mock_download.assert_not_called()
    print("  [PASS] Local caching verified: zero network calls when cached checkpoints exist.")

def test_hf_download_isolated_target():
    """Verifies downloading mechanism into an isolated temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=settings.HF_MODEL_REPO_ID,
            filename="classification/.gitkeep",
            local_dir=temp_dir
        )
        assert Path(downloaded).exists(), f"Downloaded file not found at: {downloaded}"
    print("  [PASS] Hugging Face Hub download mechanism into isolated directory verified.")

def test_numerical_equivalence_on_1014():
    """
    Performs full inference on 1014.png and verifies complete equivalence with baseline values.
    """
    import torch
    torch.manual_seed(42)

    init_db()
    inference_service.initialize_models()

    sample_path = API_ROOT / "tests" / "sample_images" / "1014.png"
    assert sample_path.exists(), f"Test image 1014.png not found at {sample_path}"

    img = Image.open(sample_path).convert("RGB")
    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    torch.manual_seed(42)
    res = inference_service.process_and_persist(
        pil_image=img,
        file_bytes=img_bytes,
        original_filename="1014.png",
        user_id="test_hf_verification"
    )

    # 1. Structure Invariants
    for required_key in [
        "analysis_id", "status", "timestamp", "original_filename",
        "wound", "tissue", "etiology", "severity", "uncertainty",
        "visualizations", "explainability", "academic_notice"
    ]:
        assert required_key in res, f"Missing key '{required_key}' in inference output"

    assert res["status"] == "success"
    assert res["original_filename"] == "1014.png"

    # 2. Wound Segmentation & Morphometrics
    w = res["wound"]
    assert w["detected"] is True
    assert 5900 <= w["area_pixels"] <= 6100, f"Area pixels out of range: {w['area_pixels']}"
    assert round(w["area_cm2"], 2) == 1.35
    assert 40.0 <= w["perimeter_mm"] <= 43.0
    assert 0.95 <= w["circularity"] <= 1.0
    assert w["is_irregular"] is False

    # 3. Tissue Breakdown
    t = res["tissue"]
    assert 73.0 <= t["fibrin_slough_percent"] <= 76.0
    assert 3.5 <= t["granulation_percent"] <= 5.5
    assert 20.0 <= t["callus_percent"] <= 23.0

    # 4. Etiology Classification
    e = res["etiology"]
    assert e["predicted_type"] == "Diabetic Foot Ulcer"
    assert round(e["confidence"], 1) == 98.9
    probs = e["probabilities"]
    assert round(probs["DFU"], 1) == 98.9
    assert round(probs["Pressure"], 1) == 0.4
    assert round(probs["Surgical"], 1) == 0.5
    assert round(probs["Venous"], 1) == 0.2

    # 5. Severity Assessment
    s = res["severity"]
    assert 48.0 <= s["severity_score"] <= 49.5
    assert s["severity_grade"] == "Moderate (Chronic Risk)"
    assert "pressure offloading" in s["recommended_action"]

    # 6. Epistemic Uncertainty & Clinician Review Flag
    u = res["uncertainty"]
    assert 85.0 <= u["ai_confidence_score"] <= 90.0
    assert u["requires_clinician_review"] is False

    # 7. Explainability
    expl = res["explainability"]
    assert expl["available"] is True
    assert expl["target_class"] == "Diabetic Foot Ulcer"
    assert "Dual-Branch" in expl["method"]

    print("  [PASS] Numerical and behavioral equivalence verified on 1014.png.")

def test_cpu_inference_compatibility():
    """
    Verifies that the entire multi-task AI pipeline runs reliably on CPU.
    Validates binary wound bed segmentation, tissue composition, etiology classification,
    and Grad-CAM explainability without CUDA dependencies.
    """
    import torch
    from src.pipeline import MasterWoundSystem

    sample_path = API_ROOT / "tests" / "sample_images" / "1014.png"
    assert sample_path.exists(), f"Test image 1014.png not found at {sample_path}"
    img = Image.open(sample_path).convert("RGB")

    torch.manual_seed(42)
    cpu_system = MasterWoundSystem(
        wound_seg_path=settings.WOUND_MODEL_PATH,
        tissue_seg_path=settings.TISSUE_MODEL_PATH,
        classifier_path=settings.CLASSIFIER_MODEL_PATH,
        fallback_scale=settings.FALLBACK_SCALE,
        device="cpu"
    )
    assert cpu_system.device == "cpu"
    assert cpu_system.device_type == "cpu"

    torch.manual_seed(42)
    res = cpu_system.process_image(img, run_explainability=True)

    assert res["diagnostics"]["predicted_type"] == "Diabetic Foot Ulcer"
    assert res["morphometrics"]["area_pixels"] > 0
    assert res["morphometrics"]["area_cm2"] > 0
    assert "tissue_breakdown_supervised" in res
    assert "severity_assessment" in res
    assert "safety_qa" in res
    assert res["explainability"] is not None
    assert res["explainability"]["available"] is True
    print("  [PASS] CPU inference compatibility verified (segmentation, classification, explainability).")

if __name__ == "__main__":
    print("\n--- [SUITE: HUGGING FACE MODEL MANAGEMENT & VERIFICATION] ---")
    test_hf_configuration()
    test_hf_remote_resolution()
    test_hf_caching_behavior()
    test_hf_download_isolated_target()
    test_numerical_equivalence_on_1014()
    test_cpu_inference_compatibility()
    print("\nALL HUGGING FACE & CPU TESTS PASSED SUCCESSFULLY!")

