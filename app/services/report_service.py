import io
import logging
from typing import Dict, Any

logger = logging.getLogger("wound_api.report_service")

class ReportService:
    """Coordinates generation and resolution of diagnostic visual reports in-memory."""

    def generate_report_bytes(
        self,
        system,
        raw_results: Dict[str, Any]
    ) -> bytes:
        """Generates the multi-panel clinical diagnostic report figure in-memory and returns the PNG bytes."""
        buffer = io.BytesIO()

        # system.generate_report_figure uses matplotlib.pyplot.savefig which natively accepts a file-like object
        system.generate_report_figure(
            orig_rgb=raw_results["_raw"]["orig_rgb"],
            binary_mask=raw_results["_raw"]["binary_mask"],
            unc_map=raw_results["_raw"]["uncertainty_map"],
            tissue_data=raw_results["_raw"]["tissue_data"],
            diag_data=raw_results["diagnostics"],
            morph_data=raw_results["morphometrics"],
            sev_data=raw_results["severity_assessment"],
            save_path=buffer,
            explainability_data=raw_results["_raw"].get("explainability_data")
        )
        
        return buffer.getvalue()

report_service = ReportService()
