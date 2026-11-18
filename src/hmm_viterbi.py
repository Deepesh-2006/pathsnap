from collections import defaultdict
from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))


def build_adjacency(arcs: np.ndarray) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Directed adjacency: out_arcs[node] and in_arcs[node]."""
    out_nodes: dict[int, set[int]] = defaultdict(set)
    for u, v in arcs:
        out_nodes[int(u)].add(int(v))
    return out_nodes, {}


def viterbi_decode_trip(
    df_trip,
    probs: np.ndarray,
    geom: dict,
    emit_weight: float = 1.0,
    stay_bonus: float = 0.0,
    direct_pen: float = -0.3,
    two_hop_pen: float = -1.2,
    shared_node_pen: float = -3.0,
    disc_pen: float = -7.5,
) -> tuple[np.ndarray, list[int]]:
    """
    Decode the most likely sequence of arc_ids for a single trip using Viterbi.
    Returns (per_obs_arc_ids, collapsed_route_arc_ids).
    """
    arcs = geom["arcs"]
    nx, ny = geom["nx"], geom["ny"]
    ts = geom["ts"]

    # Build lightweight directed adjacency only for nodes touched by candidates + 1-hop neighbors
    out_nodes: dict[int, set[int]] = defaultdict(set)
    cand_arcs_all = np.unique(df_trip["arc_id"].to_numpy(dtype=np.int64))
    touched_nodes = set(arcs[cand_arcs_all, 1].tolist())
    for u, v in arcs:
        if int(u) in touched_nodes:
            out_nodes[int(u)].add(int(v))

    obs_indices = df_trip["obs_idx"].to_numpy(dtype=np.int32)
    arc_ids = df_trip["arc_id"].to_numpy(dtype=np.int64)
    dists = df_trip["dist_m"].to_numpy(dtype=np.float64)

    n_obs = int(obs_indices.max()) + 1
    # Group candidate slices by obs_idx (df_trip is already ordered by obs_idx)
    splits = np.flatnonzero(np.diff(obs_indices)) + 1
    groups_arc = np.split(arc_ids, splits)
    groups_prob = np.split(probs, splits)
    groups_dist = np.split(dists, splits)

    dp_prev = None
    backptrs = []

    for t in range(n_obs):
        c_arcs = groups_arc[t]
        c_probs = np.clip(groups_prob[t], 1e-6, 1.0)
        c_dists = groups_dist[t]
        emit = emit_weight * np.log(c_probs) - 0.5 * (c_dists / 40.0) ** 2

        if t == 0:
            dp_prev = emit
            continue

        prev_arcs = groups_arc[t - 1]
        dt = float(ts[t] - ts[t - 1])
        gap_relax = 0.4 if dt > 3.0 else 1.0

        k_prev = len(prev_arcs)
        k_curr = len(c_arcs)
        trans = np.full((k_prev, k_curr), disc_pen * gap_relax, dtype=np.float64)

        u1 = arcs[prev_arcs, 0]
        v1 = arcs[prev_arcs, 1]
        u2 = arcs[c_arcs, 0]
        v2 = arcs[c_arcs, 1]

        for i in range(k_prev):
            a1 = prev_arcs[i]
            vi = int(v1[i])
            ui = int(u1[i])
            nbrs_1 = out_nodes.get(vi, ())
            for j in range(k_curr):
                a2 = c_arcs[j]
                uj = int(u2[j])
                vj = int(v2[j])
                if a1 == a2:
                    trans[i, j] = stay_bonus
                elif vi == uj:
                    trans[i, j] = direct_pen
                elif uj in nbrs_1:
                    trans[i, j] = two_hop_pen * gap_relax
                elif ui == uj or vi == vj or ui == vj:
                    trans[i, j] = shared_node_pen * gap_relax
                elif dt > 3.0:
                    # Euclidean gap penalty when GPS dropped out
                    gap_m = np.hypot(nx[uj] - nx[vi], ny[uj] - ny[vi])
                    trans[i, j] = -1.5 - 0.015 * min(gap_m, 400.0)

        scores = dp_prev[:, None] + trans
        best_prev = np.argmax(scores, axis=0)
        dp_prev = scores[best_prev, np.arange(k_curr)] + emit
        backptrs.append(best_prev)

    # Backtrack
    seq_idx = np.empty(n_obs, dtype=np.int32)
    seq_idx[-1] = int(np.argmax(dp_prev))
    for t in range(n_obs - 2, -1, -1):
        seq_idx[t] = backptrs[t][seq_idx[t + 1]]

    pred_obs_arcs = np.array([groups_arc[t][seq_idx[t]] for t in range(n_obs)], dtype=np.int64)

    collapsed = []
    for a in pred_obs_arcs.tolist():
        if not collapsed or collapsed[-1] != a:
            collapsed.append(a)

    return pred_obs_arcs, collapsed


def compute_route_continuity(pred_obs_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Fraction of arc transitions that are connected (same arc or to_node == next from_node)."""
    if len(pred_obs_arcs) <= 1:
        return 1.0
    changes = pred_obs_arcs[:-1] != pred_obs_arcs[1:]
    if not np.any(changes):
        return 1.0
    a_prev = pred_obs_arcs[:-1][changes]
    a_next = pred_obs_arcs[1:][changes]
    connected = arcs[a_prev, 1] == arcs[a_next, 0]
    return float(np.mean(connected))
