import json
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

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import list_trips, load_trip
from src.features import FEATURE_COLS, extract_trip_candidates
from src.hmm_viterbi import viterbi_decode_trip, compute_route_continuity
from src.train_models import eval_observation_top1_and_continuity

optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluate_hmm_on_student_test(clf) -> dict:
    obs_top1_clf, obs_top1_hmm = [], []
    cont_clf, cont_hmm = [], []
    route_f1_clf, route_f1_hmm = [], []

    for trip_id in list_trips(config.TEST_DIR):
        trip = load_trip(config.TEST_DIR, trip_id)
        df_trip, meta = extract_trip_candidates(trip, with_labels=True)
        X = df_trip[FEATURE_COLS].to_numpy(dtype=np.float32)
        probs = clf.predict_proba(X)[:, 1]

        work = df_trip[["obs_idx", "arc_id", "label"]].copy()
        work["prob"] = probs
        best_rows = work.loc[work.groupby("obs_idx", sort=False)["prob"].idxmax()]
        clf_obs_arcs = best_rows["arc_id"].to_numpy(dtype=np.int64)
        ref_arcs = meta["ref_arcs"]

        hmm_obs_arcs, hmm_route = viterbi_decode_trip(df_trip, probs, meta["geom"])

        obs_top1_clf.append(float(np.mean(clf_obs_arcs == ref_arcs)))
        obs_top1_hmm.append(float(np.mean(hmm_obs_arcs == ref_arcs)))

        arcs = trip["arcs"]
        cont_clf.append(compute_route_continuity(clf_obs_arcs, arcs))
        cont_hmm.append(compute_route_continuity(hmm_obs_arcs, arcs))

        # Sequence-level arc set F1 against .route
        gt_set = set(trip["route"].tolist())
        clf_set = set(clf_obs_arcs.tolist())
        hmm_set = set(hmm_route)

        for pred_set, bucket in [(clf_set, route_f1_clf), (hmm_set, route_f1_hmm)]:
            inter = len(gt_set & pred_set)
            p = inter / max(len(pred_set), 1)
            r = inter / max(len(gt_set), 1)
            bucket.append(2 * p * r / max(p + r, 1e-9))

    return {
        "classifier_only": {
            "obs_top1_acc": round(float(np.mean(obs_top1_clf)), 4),
            "route_continuity": round(float(np.mean(cont_clf)), 4),
            "route_set_f1": round(float(np.mean(route_f1_clf)), 4),
        },
        "classifier_plus_hmm": {
            "obs_top1_acc": round(float(np.mean(obs_top1_hmm)), 4),
            "route_continuity": round(float(np.mean(cont_hmm)), 4),
            "route_set_f1": round(float(np.mean(route_f1_hmm)), 4),
        },
    }


def main():
    train_df = pd.read_pickle(config.OUTPUTS_DIR / "train_features.pkl")
    test_df = pd.read_pickle(config.OUTPUTS_DIR / "test_features.pkl")

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int8)
    groups = train_df["trip_id"].to_numpy()

    X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int8)

    test_arcs_map = {tid: load_trip(config.TEST_DIR, tid)["arcs"] for tid in list_trips(config.TEST_DIR)}
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_train, y_train, groups=groups))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 350),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }
        fold_f1s = []
        for tr_idx, val_idx in splits:
            clf = XGBClassifier(
                **params,
                tree_method="hist",
                n_jobs=-1,
                random_state=config.RANDOM_SEED,
                eval_metric="logloss",
            )
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            preds = clf.predict(X_train[val_idx])
            fold_f1s.append(f1_score(y_train[val_idx], preds, average="macro"))
        return float(np.mean(fold_f1s))

    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=config.RANDOM_SEED))
    study.optimize(objective, n_trials=15)
    opt_time = time.perf_counter() - t0

    trials_df = study.trials_dataframe()
    trials_df.to_csv(config.OUTPUTS_DIR / "optuna_trials.csv", index=False)

    best_params = study.best_params
    print(f"Optuna finished 15 trials in {opt_time:.1f}s. Best CV Macro F1: {study.best_value:.4f}")
    print("Best params:", best_params)

    # Compare untuned vs tuned
    untuned = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.08,
        tree_method="hist", n_jobs=-1, random_state=config.RANDOM_SEED, eval_metric="logloss"
    )
    untuned.fit(X_train, y_train)

    tuned = XGBClassifier(
        **best_params, tree_method="hist",
        n_jobs=-1, random_state=config.RANDOM_SEED, eval_metric="logloss"
    )
    tuned.fit(X_train, y_train)

    comp_rows = []
    for label, mdl, cv_f1 in [("Untuned XGBoost", untuned, 0.8865), ("Tuned XGBoost", tuned, study.best_value)]:
        t_inf = time.perf_counter()
        probs = mdl.predict_proba(X_test)[:, 1]
        inf_time = time.perf_counter() - t_inf
        preds = (probs >= 0.5).astype(np.int8)
        top1, cont = eval_observation_top1_and_continuity(test_df, probs, test_arcs_map)
        comp_rows.append({
            "Model": label,
            "CV Macro F1": round(float(cv_f1), 4) if cv_f1 is not None else "see model_comparison.csv",
            "Top-1 Accuracy": round(top1, 4),
            "Student-Test Macro F1": round(float(f1_score(y_test, preds, average="macro")), 4),
            "Log Loss": round(float(log_loss(y_test, np.clip(probs, 1e-6, 1 - 1e-6))), 4),
            "Route Continuity": round(cont, 4),
            "Inference Time (s)": round(inf_time, 4),
        })

    pd.DataFrame(comp_rows).to_csv(config.OUTPUTS_DIR / "tuned_vs_untuned.csv", index=False)
    print("Tuned vs Untuned:", comp_rows)

    # Evaluate HMM + Viterbi on student_test
    hmm_res = evaluate_hmm_on_student_test(tuned)
    with open(config.OUTPUTS_DIR / "hmm_comparison.json", "w") as f:
        json.dump(hmm_res, f, indent=2)
    print("HMM comparison:", json.dumps(hmm_res, indent=2))

    # Save final artifacts
    joblib.dump(tuned, config.ARTIFACTS_DIR / "best_model.joblib")
    with open(config.ARTIFACTS_DIR / "model_meta.json", "w") as f:
        json.dump({
            "model_type": "XGBClassifier",
            "features": FEATURE_COLS,
            "top_k_candidates": config.TOP_K_CANDIDATES,
            "best_params": best_params,
            "optuna_trials": 15,
            "optuna_best_cv_macro_f1": round(float(study.best_value), 4),
            "optuna_time_s": round(opt_time, 2),
        }, f, indent=2)


if __name__ == "__main__":
    main()
