import sys
import os
import time
from pathlib import Path

# Set UTF-8 encoding for Windows standard output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_health import test_root_endpoint, test_health_endpoint
from tests.test_analysis import (
    test_inference_pipeline_on_real_image,
    test_legacy_analyze_endpoint,
    test_error_handling_invalid_inputs
)
from tests.test_persistence import test_persistence_and_retrieval
from tests.test_isolation import test_no_woundinsight_imports, test_checkpoint_isolation

def run_suite():
    print("=" * 80)
    print("WOUNDINSIGHT-API: INDEPENDENT COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 80)
    start_total = time.perf_counter()

    print("\n--- [SUITE 1: HEALTH & READINESS ENDPOINTS] ---")
    test_root_endpoint()
    print("  [PASS] GET / (Root endpoint) verified")
    test_health_endpoint()
    print("  [PASS] GET /health (Health & models status) verified")

    print("\n--- [SUITE 2: INFERENCE PIPELINE & ERROR HANDLING] ---")
    test_inference_pipeline_on_real_image()
    print("  [PASS] POST /api/v1/analyses (Full multi-task inference) verified")
    test_legacy_analyze_endpoint()
    print("  [PASS] POST /analyze (Backward compatibility) verified")
    test_error_handling_invalid_inputs()
    print("  [PASS] Input validation & error responses (400) verified")

    print("\n--- [SUITE 3: PERSISTENCE & ASSET RETRIEVAL] ---")
    test_persistence_and_retrieval()
    print("  [PASS] Database persistence & retrieval verified")
    print("  [PASS] Report & original image streaming verified")

    print("\n--- [SUITE 4: ARCHITECTURAL ISOLATION] ---")
    test_no_woundinsight_imports()
    test_checkpoint_isolation()
    print("  [PASS] Complete repository isolation verified (0 external runtime dependencies)")

    total_time = time.perf_counter() - start_total
    print("\n" + "=" * 80)
    print(f"ALL TESTS COMPLETED SUCCESSFULLY IN {total_time:.2f}s!")
    print("=" * 80)

if __name__ == "__main__":
    run_suite()
