from pathlib import Path
import numpy as np


EARTH_RADIUS_M = 6371000.0


def list_trips(partition_dir: Path) -> list[str]:
    return sorted(p.name for p in partition_dir.iterdir() if p.is_dir())


def load_trip(partition_dir: Path, trip_id: str) -> dict:
    trip_dir = partition_dir / trip_id
    track = np.loadtxt(trip_dir / f"{trip_id}.track", dtype=np.float64)
    nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", dtype=np.float64)
    arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64)

    if track.ndim == 1:
        track = track.reshape(1, -1)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)

    route_path = trip_dir / f"{trip_id}.route"
    route = None
    if route_path.exists():
        route = np.loadtxt(route_path, dtype=np.int64)
        if route.ndim == 0:
            route = route.reshape(1)

    return {
        "trip_id": trip_id,
        "track": track,  # (T, 3): lon, lat, timestamp
        "nodes": nodes,  # (N, 2): lon, lat
        "arcs": arcs,    # (M, 2): from_node, to_node
        "route": route,  # (R,) or None
    }


def make_local_projector(lon0: float, lat0: float):
    """Equirectangular projection centered at (lon0, lat0) in meters."""
    cos_lat = np.cos(np.deg2rad(lat0))
    deg_to_m_lat = np.deg2rad(1.0) * EARTH_RADIUS_M
    deg_to_m_lon = deg_to_m_lat * cos_lat

    def project(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = (lon - lon0) * deg_to_m_lon
        y = (lat - lat0) * deg_to_m_lat
        return x, y

    return project


def point_to_segments_dist_and_proj(
    px: float,
    py: float,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized distance from point (px, py) to segments A->B.
    Returns (dist, t_clamped, proj_x, proj_y).
    """
    vx = bx - ax
    vy = by - ay
    seg_len_sq = vx * vx + vy * vy

    wx = px - ax
    wy = py - ay

    # Guard against zero-length degenerate segments
    t = np.where(seg_len_sq > 1e-12, (wx * vx + wy * vy) / np.maximum(seg_len_sq, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)

    proj_x = ax + t * vx
    proj_y = ay + t * vy
    dx = px - proj_x
    dy = py - proj_y
    dist = np.sqrt(dx * dx + dy * dy)
    return dist, t, proj_x, proj_y


def segment_bearings(ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    """Angle in radians in [-pi, pi] from A to B in local XY plane."""
    return np.arctan2(by - ay, bx - ax)


def angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Smallest signed angular difference in [-pi, pi]."""
    d = (a - b + np.pi) % (2.0 * np.pi) - np.pi
    return d
