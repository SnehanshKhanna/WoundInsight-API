from app.routes.health import router as health_router
from app.routes.analysis import router as analysis_router
from app.routes.reports import router as reports_router

__all__ = ["health_router", "analysis_router", "reports_router"]
