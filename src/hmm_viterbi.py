from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))


def viterbi_decode_trip(
    df_trip,
    probs: np.ndarray,
    geom: dict,
    emit_w: float = 1.0,
    stay_b: float = 0.0,
    dir_p: float = -0.3,
    hop2_p: float = -1.2,
    node_p: float = -3.0,
    disc_p: float = -7.5,
) -> tuple[np.ndarray, list[int]]:
    arcs, nx, ny, ts = geom["arcs"], geom["nx"], geom["ny"], geom["ts"]

    out_nds: dict[int, set[int]] = defaultdict(set)
    c_all = np.unique(df_trip["arc_id"].to_numpy(dtype=np.int64))
    touched = set(arcs[c_all, 1].tolist())
    for u, v in arcs:
        if int(u) in touched:
            out_nds[int(u)].add(int(v))

    obs_idx = df_trip["obs_idx"].to_numpy(dtype=np.int32)
    aids = df_trip["arc_id"].to_numpy(dtype=np.int64)
    dists = df_trip["dist_m"].to_numpy(dtype=np.float64)

    n_obs = int(obs_idx.max()) + 1
    splits = np.flatnonzero(np.diff(obs_idx)) + 1
    g_arc = np.split(aids, splits)
    g_prob = np.split(probs, splits)
    g_dist = np.split(dists, splits)

    dp_prev = None
    bptrs = []

    for t in range(n_obs):
        ca = g_arc[t]
        cp = np.clip(g_prob[t], 1e-6, 1.0)
        cd = g_dist[t]
        emit = emit_w * np.log(cp) - 0.5 * (cd / 40.0) ** 2

        if t == 0:
            dp_prev = emit
            continue

        pa = g_arc[t - 1]
        dt = float(ts[t] - ts[t - 1])
        g_rel = 0.4 if dt > 3.0 else 1.0

        kp, kc = len(pa), len(ca)
        trans = np.full((kp, kc), disc_p * g_rel, dtype=np.float64)
        u1, v1 = arcs[pa, 0], arcs[pa, 1]
        u2, v2 = arcs[ca, 0], arcs[ca, 1]

        for i in range(kp):
            a1, ui, vi = pa[i], int(u1[i]), int(v1[i])
            nbrs = out_nds.get(vi, ())
            for j in range(kc):
                a2, uj, vj = ca[j], int(u2[j]), int(v2[j])
                if a1 == a2:
                    trans[i, j] = stay_b
                elif vi == uj:
                    trans[i, j] = dir_p
                elif uj in nbrs:
                    trans[i, j] = hop2_p * g_rel
                elif ui == uj or vi == vj or ui == vj:
                    trans[i, j] = node_p * g_rel
                elif dt > 3.0:
                    gm = np.hypot(nx[uj] - nx[vi], ny[uj] - ny[vi])
                    trans[i, j] = -1.5 - 0.015 * min(gm, 400.0)

        sc = dp_prev[:, None] + trans
        bp = np.argmax(sc, axis=0)
        dp_prev = sc[bp, np.arange(kc)] + emit
        bptrs.append(bp)

    seq = np.empty(n_obs, dtype=np.int32)
    seq[-1] = int(np.argmax(dp_prev))
    for t in range(n_obs - 2, -1, -1):
        seq[t] = bptrs[t][seq[t + 1]]

    pred_obs = np.array([g_arc[t][seq[t]] for t in range(n_obs)], dtype=np.int64)
    collapsed = []
    for a in pred_obs.tolist():
        if not collapsed or collapsed[-1] != a:
            collapsed.append(a)
    return pred_obs, collapsed


def compute_route_continuity(pred_obs: np.ndarray, arcs: np.ndarray) -> float:
    if len(pred_obs) <= 1:
        return 1.0
    chg = pred_obs[:-1] != pred_obs[1:]
    if not np.any(chg):
        return 1.0
    ap, an = pred_obs[:-1][chg], pred_obs[1:][chg]
    return float(np.mean(arcs[ap, 1] == arcs[an, 0]))
