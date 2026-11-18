import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import list_trips, load_trip
from src.features import FEATURE_COLS, extract_trip_candidates
from src.hmm_viterbi import viterbi_decode_trip, compute_route_continuity
from src.train_models import eval_observation_top1_and_continuity

optuna.logging.set_verbosity(optuna.logging.WARNING)


def eval_hmm(clf) -> dict:
    t1_c, t1_h, c_c, c_h, f1_c, f1_h = [], [], [], [], [], []
    for tid in list_trips(config.TEST_DIR):
        trip = load_trip(config.TEST_DIR, tid)
        df_t, meta = extract_trip_candidates(trip, with_labels=True)
        probs = clf.predict_proba(df_t[FEATURE_COLS].to_numpy(dtype=np.float32))[:, 1]

        w = df_t[["obs_idx", "arc_id", "label"]].copy()
        w["prob"] = probs
        clf_obs = w.loc[w.groupby("obs_idx", sort=False)["prob"].idxmax(), "arc_id"].to_numpy(dtype=np.int64)
        hmm_obs, hmm_rt = viterbi_decode_trip(df_t, probs, meta["geom"])
        ref = meta["ref_arcs"]

        t1_c.append(float(np.mean(clf_obs == ref)))
        t1_h.append(float(np.mean(hmm_obs == ref)))
        c_c.append(compute_route_continuity(clf_obs, trip["arcs"]))
        c_h.append(compute_route_continuity(hmm_obs, trip["arcs"]))

        gt_s = set(trip["route"].tolist())
        for ps, bkt in [(set(clf_obs.tolist()), f1_c), (set(hmm_rt), f1_h)]:
            inter = len(gt_s & ps)
            p, r = inter / max(len(ps), 1), inter / max(len(gt_s), 1)
            bkt.append(2 * p * r / max(p + r, 1e-9))

    return {
        "classifier_only": {"obs_top1_acc": round(float(np.mean(t1_c)), 4), "route_continuity": round(float(np.mean(c_c)), 4), "route_set_f1": round(float(np.mean(f1_c)), 4)},
        "classifier_plus_hmm": {"obs_top1_acc": round(float(np.mean(t1_h)), 4), "route_continuity": round(float(np.mean(c_h)), 4), "route_set_f1": round(float(np.mean(f1_h)), 4)},
    }


def main():
    tr_df = pd.read_pickle(config.OUTPUTS_DIR / "train_features.pkl")
    te_df = pd.read_pickle(config.OUTPUTS_DIR / "test_features.pkl")

    X_tr, y_tr, grps = tr_df[FEATURE_COLS].to_numpy(dtype=np.float32), tr_df["label"].to_numpy(dtype=np.int8), tr_df["trip_id"].to_numpy()
    X_te, y_te = te_df[FEATURE_COLS].to_numpy(dtype=np.float32), te_df["label"].to_numpy(dtype=np.int8)
    arcs_map = {tid: load_trip(config.TEST_DIR, tid)["arcs"] for tid in list_trips(config.TEST_DIR)}
    splits = list(GroupKFold(n_splits=5).split(X_tr, y_tr, groups=grps))

    def obj(trial: optuna.Trial) -> float:
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 350),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }
        f1s = []
        for tr_i, val_i in splits:
            m = XGBClassifier(**p, tree_method="hist", n_jobs=-1, random_state=config.SEED, eval_metric="logloss")
            m.fit(X_tr[tr_i], y_tr[tr_i])
            f1s.append(f1_score(y_tr[val_i], m.predict(X_tr[val_i]), average="macro"))
        return float(np.mean(f1s))

    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=config.SEED))
    study.optimize(obj, n_trials=15)
    opt_t = time.perf_counter() - t0
    study.trials_dataframe().to_csv(config.OUTPUTS_DIR / "optuna_trials.csv", index=False)

    bp = study.best_params
    untuned = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, tree_method="hist", n_jobs=-1, random_state=config.SEED, eval_metric="logloss").fit(X_tr, y_tr)
    tuned = XGBClassifier(**bp, tree_method="hist", n_jobs=-1, random_state=config.SEED, eval_metric="logloss").fit(X_tr, y_tr)

    rows = []
    for lbl, mdl, cv_f1 in [("Untuned XGBoost", untuned, 0.8865), ("Tuned XGBoost", tuned, study.best_value)]:
        ti = time.perf_counter()
        probs = mdl.predict_proba(X_te)[:, 1]
        inf_t = time.perf_counter() - ti
        preds = (probs >= 0.5).astype(np.int8)
        top1, cont = eval_observation_top1_and_continuity(te_df, probs, arcs_map)
        rows.append({
            "Model": lbl,
            "CV Macro F1": round(float(cv_f1), 4),
            "Top-1 Accuracy": round(top1, 4),
            "Student-Test Macro F1": round(float(f1_score(y_te, preds, average="macro")), 4),
            "Log Loss": round(float(log_loss(y_te, np.clip(probs, 1e-6, 1 - 1e-6))), 4),
            "Route Continuity": round(cont, 4),
            "Inference Time (s)": round(inf_t, 4),
        })
    pd.DataFrame(rows).to_csv(config.OUTPUTS_DIR / "tuned_vs_untuned.csv", index=False)

    hmm_res = eval_hmm(tuned)
    with open(config.OUTPUTS_DIR / "hmm_comparison.json", "w") as f:
        json.dump(hmm_res, f, indent=2)

    joblib.dump(tuned, config.ARTIFACTS_DIR / "best_model.joblib")
    with open(config.ARTIFACTS_DIR / "model_meta.json", "w") as f:
        json.dump({
            "model_type": "XGBClassifier",
            "features": FEATURE_COLS,
            "top_k_candidates": config.TOP_K,
            "best_params": bp,
            "optuna_trials": 15,
            "optuna_best_cv_macro_f1": round(float(study.best_value), 4),
            "optuna_time_s": round(opt_t, 2),
        }, f, indent=2)


if __name__ == "__main__":
    main()
