from pathlib import Path
import sys
import joblib
import numpy as np
import folium

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import list_trips, load_trip
from src.features import FEATURE_COLS, extract_trip_candidates
from src.hmm_viterbi import viterbi_decode_trip


def build_trip_map(trip: dict, pred_arcs: list[int], out_html: Path) -> None:
    trk, nds, arcs, ref = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    m = folium.Map(location=[float(trk[:, 1].mean()), float(trk[:, 0].mean())], zoom_start=15, tiles="OpenStreetMap")

    pad = 0.003
    uc, vc = nds[arcs[:, 0]], nds[arcs[:, 1]]
    in_box = (
        (uc[:, 0] >= trk[:, 0].min() - pad) & (uc[:, 0] <= trk[:, 0].max() + pad) &
        (uc[:, 1] >= trk[:, 1].min() - pad) & (uc[:, 1] <= trk[:, 1].max() + pad)
    )
    fg_net = folium.FeatureGroup(name="Road Network (local)", show=True)
    n_lines = [[[float(uc[i, 1]), float(uc[i, 0])], [float(vc[i, 1]), float(vc[i, 0])]] for i in np.flatnonzero(in_box)[:3000]]
    if n_lines:
        folium.PolyLine(n_lines, color="#b0b8c1", weight=2, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)

    if ref is not None and len(ref) > 0:
        fg_ref = folium.FeatureGroup(name="Reference Route", show=True)
        r_lines = [[[float(nds[arcs[a, 0], 1]), float(nds[arcs[a, 0], 0])], [float(nds[arcs[a, 1], 1]), float(nds[arcs[a, 1], 0])]] for a in ref]
        folium.PolyLine(r_lines, color="#2ca02c", weight=6, opacity=0.75).add_to(fg_ref)
        fg_ref.add_to(m)

    if pred_arcs:
        fg_pred = folium.FeatureGroup(name="Predicted Route (Model + HMM)", show=True)
        p_lines = [[[float(nds[arcs[a, 0], 1]), float(nds[arcs[a, 0], 0])], [float(nds[arcs[a, 1], 1]), float(nds[arcs[a, 1], 0])]] for a in pred_arcs]
        folium.PolyLine(p_lines, color="#d62728", weight=3.5, opacity=0.9, dash_array="6, 4").add_to(fg_pred)
        fg_pred.add_to(m)

    fg_gps = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    pts = [[float(lat), float(lon)] for lon, lat in trk[:, :2]]
    for pt in pts[::max(1, len(pts) // 250)]:
        folium.CircleMarker(pt, radius=3, color="#1f77b4", fill=True, fill_opacity=0.85).add_to(fg_gps)
    fg_gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    legend = """
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white;
                padding: 10px 14px; border: 1px solid #ccc; border-radius: 6px; font-size: 13px;">
      <b>PathSnap Map Legend</b><br>
      <span style="color:#1f77b4;">&#9679;</span> Raw GPS Points<br>
      <span style="color:#b0b8c1;">&#9473;</span> Road Network<br>
      <span style="color:#2ca02c;">&#9473;</span> Reference Route<br>
      <span style="color:#d62728;">&#9476;</span> Predicted Route (HMM)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    m.save(str(out_html))


def main():
    clf = joblib.load(config.ARTIFACTS_DIR / "best_model.joblib")
    for tid in list_trips(config.TEST_DIR)[:3]:
        trip = load_trip(config.TEST_DIR, tid)
        df_t, meta = extract_trip_candidates(trip, with_labels=True)
        probs = clf.predict_proba(df_t[FEATURE_COLS].to_numpy(dtype=np.float32))[:, 1]
        _, rt = viterbi_decode_trip(df_t, probs, meta["geom"])
        out_p = config.MAPS_DIR / f"trip_{tid}_map.html"
        build_trip_map(trip, rt, out_p)
        print(f"Saved {out_p}")


if __name__ == "__main__":
    main()
