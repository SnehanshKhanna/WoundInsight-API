import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.database import init_db
from app.services.inference_service import inference_service
from app.services.model_manager import model_manager
from app.routes.health import router as health_router
from app.routes.analysis import router as analysis_router
from app.routes.reports import router as reports_router
from app.schemas.health import RootResponse

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("wound_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: sets up SQLite database and loads deep learning models once."""
    logger.info("Initializing WoundInsight-API persistence layer (SQLite)...")
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise e

    logger.info(f"Ensuring AI model checkpoints are available locally (HF Repo: {settings.HF_MODEL_REPO_ID})...")
    try:
        model_manager.ensure_models_available()
    except Exception as e:
        logger.error(f"FATAL: Failed to resolve/download AI models: {e}", exc_info=True)
        raise e

    logger.info("Initializing AI Inference Engine (3 production deep learning models)...")
    try:
        inference_service.initialize_models()
        logger.info("Master Clinical Pipeline ready to process requests.")
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize AI models: {e}", exc_info=True)
        raise e

    yield

    logger.info("Shutting down WoundInsight-API service.")

# Create FastAPI application instance
app = FastAPI(
    title=settings.SERVICE_NAME,
    description=(
        "Production REST API for Automated Multi-Task Wound Diagnostic AI System. "
        "Performs Binary Bed Segmentation (DeepLabV3+), 4-Class Tissue Composition (Granulation/Fibrin/Callus), "
        "Dual-Branch Etiology Classification (DFU/Pressure/Surgical/Venous), Morphometrics, "
        "Severity Triage Scoring, Epistemic Uncertainty Estimation, and Grad-CAM Explainability."
    ),
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static report and upload directories for direct asset streaming if needed
app.mount("/static/reports", StaticFiles(directory=str(settings.REPORT_DIR)), name="reports_static")

# Register API Routers
app.include_router(health_router)
app.include_router(analysis_router)
app.include_router(reports_router)

@app.get(
    "/",
    response_model=RootResponse,
    summary="Root Service Information",
    tags=["General"]
)
async def root():
    """Returns basic service status and documentation links."""
    return {
        "service": settings.SERVICE_NAME,
        "status": "running",
        "version": settings.VERSION,
        "docs_url": settings.DOCS_URL
    }
