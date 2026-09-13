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

## Model Checkpoints

The API requires three production deep learning model weights totaling approximately 678 MB:
- `checkpoints/segmentation/retrained_best_wound_model.pth` (~257.08 MB)
- `checkpoints/tissue_segmentation/retrained_best_tissue_model.pth` (~257.09 MB)
- `checkpoints/classification/retrained_best_dual_branch_classifier.pth` (~163.69 MB)

> **Important (Git & Model Storage)**:
> Each of these checkpoint files exceeds GitHub's 100 MB individual file limit. Consequently, the binary checkpoint files (`*.pth`) are excluded from this Git repository via `.gitignore` to maintain a clean and standard repository structure.
> 
> - **Local Development**: Ensure the three `.pth` files are present in their designated directories under `checkpoints/` before running the API or automated tests.
> - **Production Deployment**: For production containerization or cloud deployments, models will be fetched or mounted via an external model-storage solution (e.g., AWS S3, Google Cloud Storage, Hugging Face Hub, or automated asset download script).

---

## Quickstart

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
pytest tests/ -v
```
