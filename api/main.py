import os
import time
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import list_trips, load_trip
from src.features import FEATURE_COLS, extract_trip_candidates
from src.hmm_viterbi import viterbi_decode_trip, compute_route_continuity

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", str(config.TEST_DIR)))
MODEL_PATH = config.ARTIFACTS_DIR / "best_model.joblib"

app = FastAPI(title="PathSnap Map-Matching API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_artifacts():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "eval_dir": str(EVAL_DIR), "model_loaded": model is not None}


@app.get("/trips")
def get_trips():
    if not EVAL_DIR.exists():
        raise HTTPException(status_code=500, detail="Evaluation directory not found")
    return {"trips": list_trips(EVAL_DIR)}


@app.post("/predict/{trip_id}")
def predict_trip(trip_id: str):
    trip_dir = EVAL_DIR / trip_id
    if not trip_dir.exists():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")

    t0 = time.perf_counter()
    trip = load_trip(EVAL_DIR, trip_id)
    has_ref = trip["route"] is not None

    df_trip, meta = extract_trip_candidates(trip, with_labels=has_ref)
    X = df_trip[FEATURE_COLS].to_numpy(dtype=np.float32)
    probs = model.predict_proba(X)[:, 1]

    hmm_obs_arcs, hmm_route = viterbi_decode_trip(df_trip, probs, meta["geom"])
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    nodes = trip["nodes"]
    arcs = trip["arcs"]
    track = trip["track"]

    # Build polyline segments for the UI map
    pred_segments = [
        [
            [float(nodes[arcs[a, 0], 1]), float(nodes[arcs[a, 0], 0])],
            [float(nodes[arcs[a, 1], 1]), float(nodes[arcs[a, 1], 0])],
        ]
        for a in hmm_route
    ]
    gps_points = [[float(lat), float(lon)] for lon, lat in track[:, :2]]

    metrics = {
        "n_gps_obs": int(len(track)),
        "n_predicted_arcs": int(len(hmm_route)),
        "route_continuity": round(compute_route_continuity(hmm_obs_arcs, arcs), 4),
        "inference_ms": elapsed_ms,
    }

    # Server-side accuracy calculation using internal reference data (never exposing raw .route)
    if has_ref and meta["ref_arcs"] is not None:
        ref_obs = meta["ref_arcs"]
        metrics["obs_top1_accuracy"] = round(float(np.mean(hmm_obs_arcs == ref_obs)), 4)

        gt_set = set(trip["route"].tolist())
        pred_set = set(hmm_route)
        inter = len(gt_set & pred_set)
        p = inter / max(len(pred_set), 1)
        r = inter / max(len(gt_set), 1)
        metrics["route_set_f1"] = round(2 * p * r / max(p + r, 1e-9), 4)

    return {
        "trip_id": trip_id,
        "metrics": metrics,
        "gps_points": gps_points,
        "predicted_segments": pred_segments,
        "predicted_arc_ids": hmm_route,
    }
