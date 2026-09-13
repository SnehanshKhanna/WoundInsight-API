import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parent.parent
WOUNDINSIGHT_ORIG = API_ROOT.parent / "WoundInsight"

def is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

def test_no_woundinsight_imports():
    """Verifies that no loaded application or ML module is imported from the original WoundInsight directory."""
    if str(API_ROOT) not in sys.path:
        sys.path.insert(0, str(API_ROOT))

    import app.main
    import src.pipeline
    import src.explainability

    for mod_name, mod in list(sys.modules.items()):
        if mod and hasattr(mod, "__file__") and mod.__file__:
            mod_file = Path(mod.__file__).resolve()
            if mod_name.startswith("app.") or mod_name == "app" or mod_name.startswith("src.") or mod_name == "src":
                # Ensure it is inside WoundInsight-API
                assert is_subpath(mod_file, API_ROOT), f"Module {mod_name} ({mod_file}) not inside {API_ROOT}"
                # Ensure it is NOT inside original WoundInsight source directories
                assert not is_subpath(mod_file, WOUNDINSIGHT_ORIG / "src"), f"Module {mod_name} loaded from original WoundInsight/src: {mod_file}"
                assert not is_subpath(mod_file, WOUNDINSIGHT_ORIG / "backend"), f"Module {mod_name} loaded from original WoundInsight/backend: {mod_file}"

    print("[PASS] Architectural isolation verified: 0 runtime code imports from original WoundInsight.")

def test_checkpoint_isolation():
    from app.config import settings
    for ckpt_attr in ["WOUND_MODEL_PATH", "TISSUE_MODEL_PATH", "CLASSIFIER_MODEL_PATH"]:
        ckpt_path = Path(getattr(settings, ckpt_attr)).resolve()
        assert ckpt_path.exists(), f"Checkpoint {ckpt_attr} does not exist at {ckpt_path}"
        assert is_subpath(ckpt_path, API_ROOT), f"Checkpoint {ckpt_attr} path points outside API_ROOT: {ckpt_path}"
        assert not is_subpath(ckpt_path, WOUNDINSIGHT_ORIG), f"Checkpoint {ckpt_attr} points to original WoundInsight: {ckpt_path}"

    print("[PASS] Checkpoint isolation verified: all checkpoints located strictly within WoundInsight-API.")

if __name__ == "__main__":
    test_no_woundinsight_imports()
    test_checkpoint_isolation()
    print("Isolation tests completed successfully.")
