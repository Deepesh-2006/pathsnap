from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import (
    list_trips,
    load_trip,
    make_local_projector,
    point_to_segments_dist_and_proj,
    segment_bearings,
    angle_diff,
)


FEATURE_COLS = [
    "dist_m",
    "dist_rank",
    "dist_diff_best",
    "abs_heading_diff",
    "cos_heading_diff",
    "arc_len_m",
    "proj_t",
    "gps_speed_mps",
    "dt_prev_s",
    "prev_dist_m",
    "next_dist_m",
    "step_along_arc",
]


def prepare_trip_geometry(trip: dict) -> dict:
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    lon0 = float(track[:, 0].mean())
    lat0 = float(track[:, 1].mean())
    proj = make_local_projector(lon0, lat0)

    tx, ty = proj(track[:, 0], track[:, 1])
    nx, ny = proj(nodes[:, 0], nodes[:, 1])

    u = arcs[:, 0]
    v = arcs[:, 1]
    ax, ay = nx[u], ny[u]
    bx, by = nx[v], ny[v]

    mid_x = 0.5 * (ax + bx)
    mid_y = 0.5 * (ay + by)
    arc_len = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
    arc_bearing = segment_bearings(ax, ay, bx, by)

    tree = cKDTree(np.column_stack([mid_x, mid_y]))

    # Compute GPS headings and speeds in local XY
    n_obs = len(track)
    ts = track[:, 2]
    dt_prev = np.ones(n_obs, dtype=np.float64)
    if n_obs > 1:
        dt_prev[1:] = np.maximum(np.diff(ts), 1e-3)
        dt_prev[0] = dt_prev[1]

    gps_heading = np.zeros(n_obs, dtype=np.float64)
    gps_speed = np.zeros(n_obs, dtype=np.float64)
    has_heading = np.zeros(n_obs, dtype=bool)

    for i in range(n_obs):
        i0 = max(0, i - 1)
        i1 = min(n_obs - 1, i + 1)
        dx = tx[i1] - tx[i0]
        dy = ty[i1] - ty[i0]
        step = np.hypot(dx, dy)
        dt = max(ts[i1] - ts[i0], 1e-3)
        gps_speed[i] = step / dt
        if step >= 1.0:
            gps_heading[i] = np.arctan2(dy, dx)
            has_heading[i] = True

    return {
        "tx": tx,
        "ty": ty,
        "ts": ts,
        "dt_prev": dt_prev,
        "gps_heading": gps_heading,
        "gps_speed": gps_speed,
        "has_heading": has_heading,
        "ax": ax,
        "ay": ay,
        "bx": bx,
        "by": by,
        "arc_len": arc_len,
        "arc_bearing": arc_bearing,
        "tree": tree,
        "nx": nx,
        "ny": ny,
        "arcs": arcs,
        "proj": proj,
    }


def assign_reference_arcs(trip: dict, geom: dict) -> np.ndarray | None:
    route = trip.get("route")
    if route is None or len(route) == 0:
        return None

    tx, ty = geom["tx"], geom["ty"]
    ax, ay, bx, by = geom["ax"][route], geom["ay"][route], geom["bx"][route], geom["by"][route]

    n_obs = len(tx)
    ref_arcs = np.empty(n_obs, dtype=np.int64)
    for i in range(n_obs):
        d, _, _, _ = point_to_segments_dist_and_proj(tx[i], ty[i], ax, ay, bx, by)
        ref_arcs[i] = route[int(np.argmin(d))]
    return ref_arcs


def extract_trip_candidates(
    trip: dict,
    top_k: int = config.TOP_K_CANDIDATES,
    search_k: int = 60,
    with_labels: bool = True,
) -> tuple[pd.DataFrame, dict]:
    geom = prepare_trip_geometry(trip)
    tx, ty = geom["tx"], geom["ty"]
    ax, ay, bx, by = geom["ax"], geom["ay"], geom["bx"], geom["by"]
    n_obs = len(tx)
    n_arcs = len(ax)
    k_query = min(search_k, n_arcs)
    k_keep = min(top_k, n_arcs)

    ref_arcs = assign_reference_arcs(trip, geom) if with_labels else None

    # Query KDTree in batch for all GPS points
    _, idx_matrix = geom["tree"].query(np.column_stack([tx, ty]), k=k_query)
    if k_query == 1:
        idx_matrix = idx_matrix.reshape(-1, 1)

    obs_ids = []
    cand_arcs = []
    feats = []
    labels = []
    missed = 0

    for i in range(n_obs):
        pool = idx_matrix[i]
        dists, ts_proj, _, _ = point_to_segments_dist_and_proj(
            tx[i], ty[i], ax[pool], ay[pool], bx[pool], by[pool]
        )
        order = np.argsort(dists)[:k_keep]
        chosen = pool[order]
        chosen_d = dists[order]
        chosen_t = ts_proj[order]

        min_d = chosen_d[0]

        i_prev = max(0, i - 1)
        i_next = min(n_obs - 1, i + 1)
        prev_d, prev_t, _, _ = point_to_segments_dist_and_proj(
            tx[i_prev], ty[i_prev], ax[chosen], ay[chosen], bx[chosen], by[chosen]
        )
        next_d, next_t, _, _ = point_to_segments_dist_and_proj(
            tx[i_next], ty[i_next], ax[chosen], ay[chosen], bx[chosen], by[chosen]
        )

        if geom["has_heading"][i]:
            hdiff = angle_diff(geom["gps_heading"][i], geom["arc_bearing"][chosen])
            abs_hdiff = np.abs(hdiff)
            cos_hdiff = np.cos(hdiff)
        else:
            abs_hdiff = np.full(k_keep, np.pi / 2.0)
            cos_hdiff = np.zeros(k_keep)

        arc_l = geom["arc_len"][chosen]
        step_along = (next_t - prev_t) * arc_l

        block = np.column_stack([
            chosen_d,
            np.arange(k_keep, dtype=np.float32),
            chosen_d - min_d,
            abs_hdiff,
            cos_hdiff,
            arc_l,
            chosen_t,
            np.full(k_keep, geom["gps_speed"][i]),
            np.full(k_keep, geom["dt_prev"][i]),
            prev_d,
            next_d,
            step_along,
        ]).astype(np.float32)

        obs_ids.append(np.full(k_keep, i, dtype=np.int32))
        cand_arcs.append(chosen.astype(np.int64))
        feats.append(block)

        if ref_arcs is not None:
            y_i = ref_arcs[i]
            match = (chosen == y_i).astype(np.int8)
            if match.sum() == 0:
                missed += 1
            labels.append(match)

    df = pd.DataFrame(np.vstack(feats), columns=FEATURE_COLS)
    df["trip_id"] = trip["trip_id"]
    df["obs_idx"] = np.concatenate(obs_ids)
    df["arc_id"] = np.concatenate(cand_arcs)
    if ref_arcs is not None:
        df["label"] = np.concatenate(labels)

    meta = {
        "n_obs": n_obs,
        "missed_candidates": missed,
        "miss_rate": float(missed / max(n_obs, 1)),
        "ref_arcs": ref_arcs,
        "geom": geom,
    }
    return df, meta


def build_partition_dataset(partition_dir: Path, cache_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    dfs = []
    trip_stats = {}
    for trip_id in list_trips(partition_dir):
        trip = load_trip(partition_dir, trip_id)
        df_trip, meta = extract_trip_candidates(trip, with_labels=True)
        dfs.append(df_trip)
        trip_stats[trip_id] = {
            "n_obs": meta["n_obs"],
            "missed_candidates": meta["missed_candidates"],
            "miss_rate": meta["miss_rate"],
        }
    full_df = pd.concat(dfs, ignore_index=True)
    if cache_path is not None:
        full_df.to_pickle(cache_path)
    return full_df, trip_stats


if __name__ == "__main__":
    print("Building student_test features...")
    test_df, test_stats = build_partition_dataset(config.TEST_DIR, config.OUTPUTS_DIR / "test_features.pkl")
    miss_test = sum(v["missed_candidates"] for v in test_stats.values()) / sum(v["n_obs"] for v in test_stats.values())
    print(f"student_test rows={len(test_df)}, candidate miss rate={miss_test:.4f}")

    print("Building train features...")
    train_df, train_stats = build_partition_dataset(config.TRAIN_DIR, config.OUTPUTS_DIR / "train_features.pkl")
    miss_train = sum(v["missed_candidates"] for v in train_stats.values()) / sum(v["n_obs"] for v in train_stats.values())
    print(f"train rows={len(train_df)}, candidate miss rate={miss_train:.4f}")
