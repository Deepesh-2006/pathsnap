import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import list_trips, load_trip, make_local_projector


def inspect_partition(partition_name: str, partition_dir: Path) -> list[dict]:
    rows = []
    for trip_id in list_trips(partition_dir):
        data = load_trip(partition_dir, trip_id)
        track = data["track"]
        nodes = data["nodes"]
        arcs = data["arcs"]
        route = data["route"]

        n_obs = len(track)
        n_nodes = len(nodes)
        n_arcs = len(arcs)
        n_route = len(route) if route is not None else 0

        nans_track = int(np.isnan(track).sum())
        nans_nodes = int(np.isnan(nodes).sum())

        ts = track[:, 2]
        dt = np.diff(ts) if n_obs > 1 else np.array([])
        non_1s = int(np.sum(np.abs(dt - 1.0) > 1e-3)) if len(dt) > 0 else 0
        max_dt = float(np.max(dt)) if len(dt) > 0 else 0.0
        min_dt = float(np.min(dt)) if len(dt) > 0 else 0.0

        # Project track to meters around its mean to measure spatial extent
        proj = make_local_projector(float(track[:, 0].mean()), float(track[:, 1].mean()))
        tx, ty = proj(track[:, 0], track[:, 1])
        step_dist = np.sqrt(np.diff(tx) ** 2 + np.diff(ty) ** 2) if n_obs > 1 else np.array([0.0])

        # Road network connectivity (undirected check for components)
        u = arcs[:, 0]
        v = arcs[:, 1]
        adj = csr_matrix(
            (np.ones(len(arcs) * 2, dtype=np.int8), (np.concatenate([u, v]), np.concatenate([v, u]))),
            shape=(n_nodes, n_nodes),
        )
        n_comp, _ = connected_components(adj, directed=False)
        deg = np.bincount(np.concatenate([u, v]), minlength=n_nodes)
        isolated = int(np.sum(deg == 0))

        # Check how often consecutive arcs in .route share a node (to_node == next from_node)
        route_cont = np.nan
        if route is not None and len(route) > 1:
            r_arcs = arcs[route]
            connected = np.sum(r_arcs[:-1, 1] == r_arcs[1:, 0])
            route_cont = float(connected / (len(route) - 1))

        rows.append({
            "partition": partition_name,
            "trip_id": trip_id,
            "n_obs": n_obs,
            "n_nodes": n_nodes,
            "n_arcs": n_arcs,
            "n_route": n_route,
            "nans_track": nans_track,
            "nans_nodes": nans_nodes,
            "non_1s_gaps": non_1s,
            "min_dt_s": min_dt,
            "max_dt_s": max_dt,
            "total_track_km": float(np.sum(step_dist) / 1000.0),
            "max_gps_jump_m": float(np.max(step_dist)),
            "n_components": int(n_comp),
            "isolated_nodes": isolated,
            "route_continuity": route_cont,
            "lon_min": float(track[:, 0].min()),
            "lon_max": float(track[:, 0].max()),
            "lat_min": float(track[:, 1].min()),
            "lat_max": float(track[:, 1].max()),
        })
    return rows


def main():
    train_rows = inspect_partition("train", config.TRAIN_DIR)
    test_rows = inspect_partition("student_test", config.TEST_DIR)
    df = pd.DataFrame(train_rows + test_rows)
    df.to_csv(config.OUTPUTS_DIR / "eda_trips.csv", index=False)

    summary = {}
    for part in ("train", "student_test"):
        sub = df[df["partition"] == part]
        summary[part] = {
            "num_trips": int(len(sub)),
            "total_gps_obs": int(sub["n_obs"].sum()),
            "obs_per_trip_min_med_max": [int(sub["n_obs"].min()), float(sub["n_obs"].median()), int(sub["n_obs"].max())],
            "nans_total": int(sub["nans_track"].sum() + sub["nans_nodes"].sum()),
            "non_1s_fraction": float(sub["non_1s_gaps"].sum() / max(sub["n_obs"].sum() - len(sub), 1)),
            "max_dt_s": float(sub["max_dt_s"].max()),
            "min_dt_s": float(sub["min_dt_s"].min()),
            "median_nodes": float(sub["n_nodes"].median()),
            "median_arcs": float(sub["n_arcs"].median()),
            "median_route_len": float(sub["n_route"].median()),
            "mean_route_continuity": float(sub["route_continuity"].mean()),
        }

    with open(config.OUTPUTS_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    # Plots
    plt.figure(figsize=(8, 4))
    for part, color in [("train", "#2b5c8f"), ("student_test", "#d95f02")]:
        sub = df[df["partition"] == part]
        plt.hist(sub["n_obs"], bins=15, alpha=0.65, label=part, color=color, edgecolor="white")
    plt.xlabel("GPS observations per trip")
    plt.ylabel("Trip count")
    plt.title("Distribution of Trip Lengths (GPS Observations)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "eda_trip_lengths.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(df["n_nodes"], df["n_arcs"], c=df["partition"].map({"train": "#2b5c8f", "student_test": "#d95f02"}), alpha=0.8)
    axes[0].set_xlabel("Road network nodes")
    axes[0].set_ylabel("Road network arcs")
    axes[0].set_title("Road Network Size per Trip")

    axes[1].hist(df["max_dt_s"], bins=20, color="#4c72b0", edgecolor="white")
    axes[1].set_xlabel("Max timestamp gap per trip (s)")
    axes[1].set_ylabel("Trip count")
    axes[1].set_title("Maximum GPS Time Gap per Trip")
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / "eda_network_and_gaps.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
