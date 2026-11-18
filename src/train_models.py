import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, log_loss, confusion_matrix
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import list_trips, load_trip
from src.features import FEATURE_COLS
from src.hmm_viterbi import compute_route_continuity


def get_models():
    return {
        "LogisticRegression": (
            "Linear / Probabilistic",
            "StandardScaler + C=1.0, max_iter=300",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=300, random_state=config.RANDOM_SEED),
            ),
        ),
        "RandomForest": (
            "Ensemble (Bagging)",
            "n_estimators=80, max_depth=12, min_samples_leaf=15",
            lambda: RandomForestClassifier(
                n_estimators=80,
                max_depth=12,
                min_samples_leaf=15,
                n_jobs=-1,
                random_state=config.RANDOM_SEED,
            ),
        ),
        "HistGradientBoosting": (
            "Boosting (Histogram)",
            "max_iter=150, max_depth=8, lr=0.08",
            lambda: HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=8,
                learning_rate=0.08,
                random_state=config.RANDOM_SEED,
            ),
        ),
        "LightGBM": (
            "Boosting (GBDT)",
            "n_estimators=200, num_leaves=31, lr=0.08",
            lambda: LGBMClassifier(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.08,
                n_jobs=-1,
                random_state=config.RANDOM_SEED,
                verbose=-1,
            ),
        ),
        "XGBoost": (
            "Boosting (GBDT)",
            "n_estimators=200, max_depth=6, lr=0.08, hist",
            lambda: XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.08,
                tree_method="hist",
                n_jobs=-1,
                random_state=config.RANDOM_SEED,
                eval_metric="logloss",
            ),
        ),
        "MLPClassifier": (
            "Neural Network",
            "StandardScaler + (64, 32), early_stopping=True",
            lambda: make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=25,
                    early_stopping=True,
                    n_iter_no_change=5,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ),
    }


def eval_observation_top1_and_continuity(test_df: pd.DataFrame, probs: np.ndarray, test_arcs_map: dict) -> tuple[float, float]:
    work = test_df[["trip_id", "obs_idx", "arc_id", "label"]].copy()
    work["prob"] = probs

    # Pick candidate with max predicted probability per (trip_id, obs_idx)
    idx_best = work.groupby(["trip_id", "obs_idx"], sort=False)["prob"].idxmax()
    best_rows = work.loc[idx_best]
    top1_acc = float(best_rows["label"].mean())

    conts = []
    for trip_id, grp in best_rows.groupby("trip_id", sort=False):
        pred_arcs = grp["arc_id"].to_numpy(dtype=np.int64)
        conts.append(compute_route_continuity(pred_arcs, test_arcs_map[trip_id]))
    return top1_acc, float(np.mean(conts))


def main():
    train_df = pd.read_pickle(config.OUTPUTS_DIR / "train_features.pkl")
    test_df = pd.read_pickle(config.OUTPUTS_DIR / "test_features.pkl")

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int8)
    groups = train_df["trip_id"].to_numpy()

    X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int8)

    test_arcs_map = {}
    for tid in list_trips(config.TEST_DIR):
        test_arcs_map[tid] = load_trip(config.TEST_DIR, tid)["arcs"]

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_train, y_train, groups=groups))

    results = []
    conf_matrices = {}
    mlp_loss_curve = None

    for name, (family, cfg_str, builder) in get_models().items():
        print(f"=== {name} ===")
        cv_f1s = []
        for fold, (tr_idx, val_idx) in enumerate(splits):
            clf = builder()
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            val_pred = clf.predict(X_train[val_idx])
            cv_f1s.append(f1_score(y_train[val_idx], val_pred, average="macro"))

        cv_macro_f1 = float(np.mean(cv_f1s))

        # Fit on full 80 training trips
        final_clf = builder()
        t0 = time.perf_counter()
        final_clf.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        test_probs = final_clf.predict_proba(X_test)[:, 1]
        test_pred = (test_probs >= 0.5).astype(np.int8)
        infer_time = time.perf_counter() - t1

        test_macro_f1 = float(f1_score(y_test, test_pred, average="macro"))
        test_logloss = float(log_loss(y_test, np.clip(test_probs, 1e-6, 1 - 1e-6)))
        top1_acc, route_cont = eval_observation_top1_and_continuity(test_df, test_probs, test_arcs_map)
        conf_matrices[name] = confusion_matrix(y_test, test_pred).tolist()

        if name == "MLPClassifier":
            mlp_step = final_clf.named_steps["mlpclassifier"]
            mlp_loss_curve = mlp_step.loss_curve_

        row = {
            "Model": name,
            "Algorithm Family": family,
            "Key Configuration": cfg_str,
            "CV Macro F1": round(cv_macro_f1, 4),
            "Top-1 Accuracy": round(top1_acc, 4),
            "Route Continuity": round(route_cont, 4),
            "Student-Test Macro F1": round(test_macro_f1, 4),
            "Log Loss": round(test_logloss, 4),
            "Train Time (s)": round(train_time, 2),
            "Inference Time (s)": round(infer_time, 4),
        }
        results.append(row)
        print(row)

    res_df = pd.DataFrame(results)
    res_df.to_csv(config.OUTPUTS_DIR / "model_comparison.csv", index=False)
    with open(config.OUTPUTS_DIR / "model_comparison.json", "w") as f:
        json.dump({"results": results, "confusion_matrices": conf_matrices}, f, indent=2)

    # Comparison bar plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    models = res_df["Model"]
    x = np.arange(len(models))

    axes[0].bar(x, res_df["CV Macro F1"], color="#3b6998")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=30, ha="right")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("5-Fold GroupCV Macro F1")

    w = 0.35
    axes[1].bar(x - w / 2, res_df["Top-1 Accuracy"], width=w, label="Top-1 Acc", color="#2ca02c")
    axes[1].bar(x + w / 2, res_df["Student-Test Macro F1"], width=w, label="Macro F1", color="#ff7f0e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=30, ha="right")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title("Student-Test Top-1 & Macro F1")
    axes[1].legend()

    axes[2].bar(x, res_df["Log Loss"], color="#d62728")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=30, ha="right")
    axes[2].set_title("Student-Test Log Loss (lower is better)")

    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "model_comparison.png", dpi=150)
    plt.close()

    if mlp_loss_curve is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(mlp_loss_curve) + 1), mlp_loss_curve, marker="o", color="#1f77b4")
        plt.xlabel("Epoch")
        plt.ylabel("Training Log Loss")
        plt.title("MLPClassifier Training Loss Curve")
        plt.tight_layout()
        plt.savefig(config.PLOTS_DIR / "mlp_loss_curve.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
