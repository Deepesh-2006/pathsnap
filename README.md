# PathSnap: Map-Matching & Route Reconstruction

End-to-end GPS map-matching pipeline built for UE24CS352A (Machine Learning). Given noisy GPS trajectories and a local road network (`.nodes`, `.arcs`), the system scores top-10 candidate road segments per GPS point with a tuned XGBoost classifier and decodes a connected route using an HMM + Viterbi pass.

## Repository Layout

```
pathsnap/
├── config.py                # paths, candidate radius/top-K settings, seed
├── requirements.txt
├── src/
│   ├── data_utils.py        # raw file loaders, local equirectangular projection, segment math
│   ├── eda.py               # dataset inventory, gap/connectivity checks, EDA plots
│   ├── features.py          # KDTree candidate generation and 12 geometric/kinematic features
│   ├── train_models.py      # 6-model comparison (5-fold GroupKFold + student_test)
│   ├── tune_model.py        # Optuna Bayesian tuning for XGBoost + HMM/Viterbi evaluation
│   ├── hmm_viterbi.py       # Viterbi decoder with road-graph transitions and gap handling
│   └── make_maps.py         # Folium HTML map generator
├── api/
│   └── main.py              # FastAPI backend (Laptop 1)
├── ui/
│   └── index.html           # Evaluator UI with Leaflet map (Laptop 2)
├── artifacts/               # best_model.joblib, model_meta.json
└── outputs/                 # CSV/JSON summaries, matplotlib plots, Folium HTML maps
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note on macOS: `lightgbm` requires OpenMP (`brew install libomp`).

Check `config.py` and update `DATA_DIR` if your `train/` and `student_test/` folders live somewhere else.

## Running the Pipeline

```bash
# 1. Exploratory data analysis
python src/eda.py

# 2. Candidate generation & feature extraction
python src/features.py

# 3. Train and compare 6 models across 5-fold GroupKFold + student_test
python src/train_models.py

# 4. Bayesian hyperparameter tuning (Optuna) + HMM/Viterbi comparison
python src/tune_model.py

# 5. Generate interactive Folium maps
python src/make_maps.py
```

## Two-Laptop Deployment

### Laptop 1: Model & FastAPI Server

Point `PATHSNAP_EVAL_DIR` to the folder containing the 10 evaluation trips (defaults to `student_test` if unset):

```bash
export PATHSNAP_EVAL_DIR="/path/to/evaluator_test"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` — server and model status
- `GET /trips` — lists available trip IDs
- `POST /predict/{trip_id}` — runs candidate extraction, XGBoost scoring, and Viterbi decoding; computes evaluation metrics server-side when `.route` is present

### Laptop 2: Evaluator UI

Serve `ui/index.html` (or open it in a browser):

```bash
python -m http.server 8080 --directory ui
```

Enter `http://<laptop-1-ip>:8000` in the API Base URL field, click **Connect & Load Trips**, pick a trip, and click **Reconstruct Route**.

## Team Work Split

- **Member 1 — Data Pipeline, EDA & Feature Engineering** (`config.py`, `src/data_utils.py`, `src/eda.py`, `src/features.py`): Raw file loading, local equirectangular projection, dataset inventory/gap checks, `cKDTree` top-10 candidate generation, and 12 geometric/kinematic features (Report Section 2.2).
- **Member 2 — Multi-Model Experimentation & Evaluation** (`src/train_models.py`, `outputs/model_comparison.*`): 5-fold `GroupKFold` CV and `student_test` evaluation across 6 models (Logistic Regression, Random Forest, HistGradientBoosting, LightGBM, XGBoost, MLPClassifier), comparison plots, and loss curves (Report Section 3).
- **Member 3 — Bayesian Tuning, HMM/Viterbi & Folium Maps** (`src/tune_model.py`, `src/hmm_viterbi.py`, `src/make_maps.py`): Optuna TPE hyperparameter search on XGBoost, Viterbi sequence decoding with road-graph transitions and GPS gap handling, classifier-only vs. HMM comparison, and Folium HTML maps (Report Sections 4, 5, and 6).
- **Member 4 — FastAPI Backend, Evaluator UI & Deployment** (`api/main.py`, `ui/index.html`, `README.md`, `requirements.txt`): Server-side inference and evaluation endpoints (`/health`, `/trips`, `/predict/{trip_id}`), Leaflet evaluator UI, and two-laptop LAN deployment (Report Section 7).

