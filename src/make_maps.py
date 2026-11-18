from pathlib import Path
import joblib
import numpy as np
import folium

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_utils import list_trips, load_trip
from src.features import FEATURE_COLS, extract_trip_candidates
from src.hmm_viterbi import viterbi_decode_trip


def build_trip_map(trip: dict, pred_route_arcs: list[int], out_html: Path) -> None:
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    ref_route = trip["route"]

    center_lat = float(track[:, 1].mean())
    center_lon = float(track[:, 0].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

    # 1. Local road network arcs within padded trip bbox
    pad = 0.003
    lon_min, lon_max = track[:, 0].min() - pad, track[:, 0].max() + pad
    lat_min, lat_max = track[:, 1].min() - pad, track[:, 1].max() + pad

    u_coords = nodes[arcs[:, 0]]
    v_coords = nodes[arcs[:, 1]]
    in_box = (
        (u_coords[:, 0] >= lon_min) & (u_coords[:, 0] <= lon_max) &
        (u_coords[:, 1] >= lat_min) & (u_coords[:, 1] <= lat_max)
    )
    fg_net = folium.FeatureGroup(name="Road Network (local)", show=True)
    net_lines = [
        [[float(u_coords[i, 1]), float(u_coords[i, 0])], [float(v_coords[i, 1]), float(v_coords[i, 0])]]
        for i in np.flatnonzero(in_box)[:3000]
    ]
    if net_lines:
        folium.PolyLine(net_lines, color="#b0b8c1", weight=2, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)

    # 2. Reference route (if available)
    if ref_route is not None and len(ref_route) > 0:
        fg_ref = folium.FeatureGroup(name="Reference Route", show=True)
        ref_lines = [
            [[float(nodes[arcs[a, 0], 1]), float(nodes[arcs[a, 0], 0])],
             [float(nodes[arcs[a, 1], 1]), float(nodes[arcs[a, 1], 0])]]
            for a in ref_route
        ]
        folium.PolyLine(ref_lines, color="#2ca02c", weight=6, opacity=0.75).add_to(fg_ref)
        fg_ref.add_to(m)

    # 3. Predicted HMM route
    if pred_route_arcs:
        fg_pred = folium.FeatureGroup(name="Predicted Route (Model + HMM)", show=True)
        pred_lines = [
            [[float(nodes[arcs[a, 0], 1]), float(nodes[arcs[a, 0], 0])],
             [float(nodes[arcs[a, 1], 1]), float(nodes[arcs[a, 1], 0])]]
            for a in pred_route_arcs
        ]
        folium.PolyLine(pred_lines, color="#d62728", weight=3.5, opacity=0.9, dash_array="6, 4").add_to(fg_pred)
        fg_pred.add_to(m)

    # 4. Raw GPS observations (subsample markers if very long, keep full polyline)
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    gps_pts = [[float(lat), float(lon)] for lon, lat in track[:, :2]]
    step = max(1, len(gps_pts) // 250)
    for pt in gps_pts[::step]:
        folium.CircleMarker(pt, radius=3, color="#1f77b4", fill=True, fill_opacity=0.85).add_to(fg_gps)
    fg_gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white;
                padding: 10px 14px; border: 1px solid #ccc; border-radius: 6px; font-size: 13px;">
      <b>PathSnap Map Legend</b><br>
      <span style="color:#1f77b4;">&#9679;</span> Raw GPS Points<br>
      <span style="color:#b0b8c1;">&#9473;</span> Road Network<br>
      <span style="color:#2ca02c;">&#9473;</span> Reference Route<br>
      <span style="color:#d62728;">&#9476;</span> Predicted Route (HMM)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(str(out_html))


def main():
    clf = joblib.load(config.ARTIFACTS_DIR / "best_model.joblib")
    trips = list_trips(config.TEST_DIR)
    for tid in trips[:3]:
        trip = load_trip(config.TEST_DIR, tid)
        df_trip, meta = extract_trip_candidates(trip, with_labels=True)
        probs = clf.predict_proba(df_trip[FEATURE_COLS].to_numpy(dtype=np.float32))[:, 1]
        _, hmm_route = viterbi_decode_trip(df_trip, probs, meta["geom"])
        out_path = config.MAPS_DIR / f"trip_{tid}_map.html"
        build_trip_map(trip, hmm_route, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
