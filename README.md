# WoundInsight-API

A self-contained, persistence-ready FastAPI REST service providing automated clinical wound analysis powered by the finalized WoundInsight deep learning pipeline.

---

## Capabilities

1. **Binary Wound Bed Segmentation**: DeepLabV3+ (ResNet34) delineates the active wound margin with component-filtering heuristic.
2. **Multi-Class Tissue Segmentation**: DeepLabV3+ (ResNet34) quantifies percentage composition of Granulation (healing), Fibrin/Slough (devitalized), and Callus tissue.
3. **Dual-Branch Etiology Classification**: DualBranchResNet34 analyzes both global contextual framing and central 65% lesion morphology to classify:
   - Diabetic Foot Ulcer (DFU)
   - Pressure Ulcer
   - Surgical Wound
   - Venous Ulcer
4. **Physical & Pixel Morphometrics**: Computes wound area (pixels and cm²), perimeter (mm), circularity index, and boundary irregularity.
5. **Clinical Triage Severity**: Computes 0–100 composite risk score, triage grade (Low, Moderate, High, Critical), and protocol guidance.
6. **Epistemic Uncertainty Estimation**: Evaluates boundary certainty via Monte Carlo Dropout (8 forward passes) + Horizontal-Flip Test-Time Augmentation (TTA).
7. **Grad-CAM Visual Explainability**: Computes feature attribution maps across `global_backbone.layer4` and `roi_backbone.layer4`.
8. **Diagnostic Visual Report Generation**: Generates 6-panel clinical diagnostic figures.
9. **Persistence & History**: Persistent SQLite database and filesystem storage for uploaded images and analysis records.

---

## Directory Structure

```
WoundInsight-API/
├── app/
│   ├── main.py                   # FastAPI entrypoint and lifespan management
│   ├── config.py                 # Central configuration
│   ├── routes/
│   │   ├── health.py             # GET /health
│   │   ├── analysis.py           # POST /api/v1/analyses, GET /api/v1/analyses/{id}, GET /api/v1/analyses
│   │   └── reports.py            # GET /api/v1/analyses/{id}/report
│   ├── services/
│   │   ├── inference_service.py  # Coordinates ML inference and persistence
│   │   ├── storage_service.py    # Manages disk storage of uploads and reports
│   │   └── report_service.py     # Generates and resolves report figures
│   ├── schemas/
│   │   ├── analysis.py           # Analysis Pydantic models
│   │   ├── health.py             # Health check models
│   │   └── error.py              # Error response model
│   ├── models/
│   │   └── database_models.py    # Data representations
│   ├── db/
│   │   ├── database.py           # SQLite connection & schema initialization
│   │   └── repositories.py       # Data access repository
│   └── utils/
│       └── image_validation.py   # Upload validation & Pillow integrity checks
├── src/
│   ├── pipeline.py               # Finalized MasterWoundSystem
│   └── explainability.py         # Finalized DualBranchGradCAM
├── checkpoints/
│   ├── segmentation/retrained_best_wound_model.pth
│   ├── tissue_segmentation/retrained_best_tissue_model.pth
│   └── classification/retrained_best_dual_branch_classifier.pth
├── storage/
│   ├── database/                 # SQLite database file (woundinsight.db)
│   ├── uploads/                  # Uploaded original images
│   └── reports/                  # Generated diagnostic visual reports
├── tests/
│   ├── sample_images/            # Self-contained test images
│   ├── test_health.py            # Health verification
│   ├── test_analysis.py          # Inference & schema verification
│   └── test_persistence.py       # Database & retrieval verification
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service information and status |
| `GET` | `/health` | Compute device, model readiness, and database health |
| `POST` | `/api/v1/analyses` | Submit wound photo (`multipart/form-data`) for full analysis |
| `GET` | `/api/v1/analyses/{analysis_id}` | Retrieve stored analysis record by UUID |
| `GET` | `/api/v1/analyses` | List past analyses (with `limit` and `offset` pagination) |
| `GET` | `/api/v1/analyses/{analysis_id}/report` | Retrieve 6-panel diagnostic PNG report |
| `GET` | `/api/v1/analyses/{analysis_id}/image` | Retrieve original stored wound photo |

---

## Model Checkpoints & Hugging Face Hub

The API requires three production deep learning model weights totaling approximately 678 MB:
- `checkpoints/segmentation/retrained_best_wound_model.pth` (~257.08 MB)
- `checkpoints/tissue_segmentation/retrained_best_tissue_model.pth` (~257.09 MB)
- `checkpoints/classification/retrained_best_dual_branch_classifier.pth` (~163.69 MB)

Production models are hosted publicly on Hugging Face:
👉 **[SnehanshKhanna/WoundInsight-models](https://huggingface.co/SnehanshKhanna/WoundInsight-models)**

The built-in `ModelManager` (`app/services/model_manager.py`) automatically ensures all checkpoints are present at startup using the official `huggingface_hub` Python client:
- **Local Cache**: Existing models in `checkpoints/` are reused instantly with zero network delay.
- **Cold Start / Container**: In clean environments or Docker containers, missing models are downloaded on application startup and cached locally for the lifetime of the container instance.
- **Zero Per-Request Overhead**: Once loaded in memory, inference requests execute with zero download overhead.

---

## Google Cloud Run Deployment

WoundInsight-API is packaged and configured for deployment as a containerized microservice on **Google Cloud Run** with automated CPU fallback.

### 1. Architecture Highlights
- **Container Base**: `python:3.11-slim` with CPU-optimized PyTorch wheels (`--index-url https://download.pytorch.org/whl/cpu`), minimizing the container image footprint to ~1.2 GiB (compared to ~5 GiB for CUDA images).
- **Dynamic Port Binding**: Automatically respects Cloud Run's `$PORT` environment variable (defaults to `8080`).
- **Dynamic Model Fetching**: Model checkpoints are excluded from the Docker build context via `.dockerignore` and automatically fetched at instance startup from Hugging Face Hub.
- **Compute Fallback**: Operates on CPU in Cloud Run while automatically supporting local GPU (`cuda:0`) development.
- **Ephemeral Storage Notice**: Cloud Run container filesystems are ephemeral. The built-in SQLite database and generated visual diagnostic reports are persisted across requests *within the lifetime of the specific container instance*, but are discarded on instance recycling. External managed storage (e.g., Cloud SQL, Cloud Storage) can be integrated for long-term multi-instance persistence.

### 2. Sizing & Resource Requirements
- **Memory**: Provision **at least 2 GiB to 4 GiB** (recommended: `4Gi`) to comfortably accommodate the three deep learning models in memory (~1.5 GiB working set) alongside concurrent image processing.
- **vCPU**: 1 to 2 vCPUs (recommended: `2`).
- **Timeout**: Set to at least `120s` (to allow model downloading on cold starts).

### 3. Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PORT` | `8080` | Port listened to by Uvicorn (assigned by Cloud Run) |
| `API_DEVICE` | `cpu` | Force CPU execution on Cloud Run (`cuda:0` auto-detected locally) |
| `HF_MODEL_REPO_ID` | `SnehanshKhanna/WoundInsight-models` | Hugging Face Hub repository containing model checkpoints |
| `HF_TOKEN` | *(Optional)* | Hugging Face access token (only needed if repository is private) |
| `HF_REVISION` | *(Optional)* | Git revision or branch name on Hugging Face |
| `HF_FORCE_DOWNLOAD` | `false` | Set to `true` to force re-download of checkpoints on startup |

### 4. Build and Deploy Commands

#### Build Container Image (using Google Cloud Build or Artifact Registry):
```bash
# Set Google Cloud project and image tag
export PROJECT_ID="your-gcp-project-id"
export IMAGE_TAG="gcr.io/${PROJECT_ID}/woundinsight-api:latest"

# Submit build to Google Cloud Build
gcloud builds submit --tag ${IMAGE_TAG} .
```

#### Deploy to Google Cloud Run:
```bash
gcloud run deploy woundinsight-api \
  --image ${IMAGE_TAG} \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 120s \
  --port 8080 \
  --set-env-vars API_DEVICE=cpu,HF_MODEL_REPO_ID=SnehanshKhanna/WoundInsight-models
```

### 5. Verification on Cloud Run

Once deployed, query the service endpoints:
- **Service Root**: `GET https://<cloud-run-url>/`
- **Health Check**: `GET https://<cloud-run-url>/health`
  ```json
  {
    "status": "healthy",
    "device": "cpu",
    "gpu_available": false,
    "gpu_name": null,
    "models_loaded": true,
    "checkpoints": {
      "wound_segmentation": "retrained_best_wound_model.pth",
      "tissue_segmentation": "retrained_best_tissue_model.pth",
      "etiology_classifier": "retrained_best_dual_branch_classifier.pth"
    },
    "database_connected": true
  }
  ```
- **Interactive Documentation**: `GET https://<cloud-run-url>/docs`
- **Submit Analysis**: `POST https://<cloud-run-url>/api/v1/analyses` (`multipart/form-data`, field `image`)

---

## Academic Notice & Clinical Disclaimer

> **ACADEMIC PROTOTYPE NOTICE**:  
> WoundInsight-API is an academic research prototype developed as part of an engineering final-year capstone project. It is not certified, cleared, or approved by the FDA, CE, or any healthcare regulatory body for standalone clinical diagnostic decisions. Output figures, tissue breakdown percentages, and triage severity classifications are intended solely for educational, research, and technical evaluation purposes. Clinician review is strictly required.

---

## Quickstart (Local Development)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive documentation is available at `http://localhost:8000/docs`.

### 3. Run Automated Tests
```bash
python tests/run_all_tests.py
```

