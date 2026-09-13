import logging
from pathlib import Path
from typing import Dict, Any, Optional
from app.services.storage_service import storage_service

logger = logging.getLogger("wound_api.report_service")

class ReportService:
    """Coordinates generation and resolution of diagnostic visual reports."""

    def generate_and_save_report(
        self,
        system,
        raw_results: Dict[str, Any],
        analysis_id: str
    ) -> Path:
        """Generates the multi-panel clinical diagnostic report figure and saves it to storage."""
        dest_path = storage_service.get_report_destination(analysis_id)

        system.generate_report_figure(
            orig_rgb=raw_results["_raw"]["orig_rgb"],
            binary_mask=raw_results["_raw"]["binary_mask"],
            unc_map=raw_results["_raw"]["uncertainty_map"],
            tissue_data=raw_results["_raw"]["tissue_data"],
            diag_data=raw_results["diagnostics"],
            morph_data=raw_results["morphometrics"],
            sev_data=raw_results["severity_assessment"],
            save_path=str(dest_path),
            explainability_data=raw_results["_raw"].get("explainability_data")
        )
        logger.info(f"Generated visual diagnostic report at: {dest_path}")
        return dest_path

    def get_report_file(self, analysis_id: str) -> Optional[Path]:
        return storage_service.get_report_path(analysis_id)

report_service = ReportService()
