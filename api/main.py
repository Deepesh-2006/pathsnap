import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import list_trips, load_trip
from src.features import FEATURE_COLS, extract_trip_candidates
from src.hmm_viterbi import viterbi_decode_trip, compute_route_continuity

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", str(config.TEST_DIR)))
MODEL_PATH = config.ARTIFACTS_DIR / "best_model.joblib"

app = FastAPI(title="PathSnap Map-Matching API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
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
    if not (EVAL_DIR / trip_id).exists():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")

    t0 = time.perf_counter()
    trip = load_trip(EVAL_DIR, trip_id)
    has_ref = trip["route"] is not None
    df_t, meta = extract_trip_candidates(trip, with_labels=has_ref)
    probs = model.predict_proba(df_t[FEATURE_COLS].to_numpy(dtype=np.float32))[:, 1]
    hmm_obs, hmm_rt = viterbi_decode_trip(df_t, probs, meta["geom"])
    ms = round((time.perf_counter() - t0) * 1000.0, 1)

    nds, arcs, trk = trip["nodes"], trip["arcs"], trip["track"]
    segs = [[[float(nds[arcs[a, 0], 1]), float(nds[arcs[a, 0], 0])], [float(nds[arcs[a, 1], 1]), float(nds[arcs[a, 1], 0])]] for a in hmm_rt]
    pts = [[float(lat), float(lon)] for lon, lat in trk[:, :2]]

    metrics = {
        "n_gps_obs": int(len(trk)),
        "n_predicted_arcs": int(len(hmm_rt)),
        "route_continuity": round(compute_route_continuity(hmm_obs, arcs), 4),
        "inference_ms": ms,
    }

    if has_ref and meta["ref_arcs"] is not None:
        metrics["obs_top1_accuracy"] = round(float(np.mean(hmm_obs == meta["ref_arcs"])), 4)
        gt_s, pred_s = set(trip["route"].tolist()), set(hmm_rt)
        inter = len(gt_s & pred_s)
        p, r = inter / max(len(pred_s), 1), inter / max(len(gt_s), 1)
        metrics["route_set_f1"] = round(2 * p * r / max(p + r, 1e-9), 4)

    return {"trip_id": trip_id, "metrics": metrics, "gps_points": pts, "predicted_segments": segs, "predicted_arc_ids": hmm_rt}
