"""
Simplified reverse Lagrangian backtracking for the Aapda Setu demo.

This is NOT OpenDrift / a baroclinic ocean model. It advects an ensemble of
particles backward with a single, spatially uniform current + windage vector
plus a growing random walk. Good enough to show the investigation loop on
stage; not a substitute for a real oil-weathering model.

Direction convention (must match data/environment.json):
    Both current and wind are stored as oceanographic "toward" directions.
    0° = toward true north, 90° = toward true east, clockwise.
    Example: current direction 210° means water is flowing toward the southwest.

    Meteorological winds are usually reported as the direction they blow FROM.
    We do NOT use that convention here — wind 240° means the wind stress pushes
    oil toward 240° (WSW). Call this out in Q&A if a judge asks.

Forward oil velocity used in reverse:
    v_oil = current + windage * wind
    reverse step uses -v_oil so particles walk toward the probable release zone.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

KM_PER_DEG_LAT = 111.32
# 15-minute integration step — coarse but stable for a 3–24 h demo slider.
DT_HOURS = 0.25


def _toward_uv(speed_ms: float, direction_deg_toward: float) -> tuple[float, float]:
    """Convert (speed, toward-direction) into eastward (u) and northward (v) m/s."""
    rad = math.radians(direction_deg_toward)
    u_east = speed_ms * math.sin(rad)
    v_north = speed_ms * math.cos(rad)
    return u_east, v_north


def _meters_to_latlon_delta(d_east_m: float, d_north_m: float, lat: float) -> tuple[float, float]:
    km_per_deg_lon = KM_PER_DEG_LAT * math.cos(math.radians(lat))
    dlat = (d_north_m / 1000.0) / KM_PER_DEG_LAT
    dlon = (d_east_m / 1000.0) / max(km_per_deg_lon, 1e-6)
    return dlat, dlon


def _point_in_ring(lat: float, lon: float, ring: Sequence[tuple[float, float]]) -> bool:
    """Even-odd ray cast. `ring` is [(lat, lon), ...] and may be closed."""
    pts = list(ring)
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    inside = False
    n = len(pts)
    for i in range(n):
        lat_i, lon_i = pts[i]
        lat_j, lon_j = pts[(i + 1) % n]
        intersect = ((lat_i > lat) != (lat_j > lat)) and (
            lon < (lon_j - lon_i) * (lat - lat_i) / (lat_j - lat_i + 1e-15) + lon_i
        )
        if intersect:
            inside = not inside
    return inside


def _seed_particles(
    polygon: Sequence[tuple[float, float]],
    n_particles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform-ish seeds inside the polygon bbox, rejection-sampled into the ring."""
    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    # Pad the bbox slightly so a few particles start on the slick margin.
    pad_lat = 0.15 * (lat_max - lat_min + 1e-6)
    pad_lon = 0.15 * (lon_max - lon_min + 1e-6)

    pts = []
    attempts = 0
    while len(pts) < n_particles and attempts < n_particles * 40:
        attempts += 1
        lat = float(rng.uniform(lat_min - pad_lat, lat_max + pad_lat))
        lon = float(rng.uniform(lon_min - pad_lon, lon_max + pad_lon))
        if _point_in_ring(lat, lon, polygon) or rng.random() < 0.12:
            pts.append((lat, lon))
    # Fallback if the polygon is degenerate.
    while len(pts) < n_particles:
        pts.append(
            (
                float(np.mean(lats) + rng.normal(0, 0.01)),
                float(np.mean(lons) + rng.normal(0, 0.01)),
            )
        )
    return np.array(pts, dtype=float)


def backtrack(
    spill_polygon_latlon: Sequence[tuple[float, float]],
    current_vector: dict,
    wind_vector: dict,
    hours_back: float,
    n_particles: int = 150,
    windage: float = 0.03,
    spread_deg: float = 0.05,
    seed: int = 26143,
) -> dict:
    """
    Integrate `n_particles` backward for `hours_back` hours.

    Parameters
    ----------
    spill_polygon_latlon : sequence of (lat, lon)
    current_vector : {"speed_ms", "direction_deg_toward"}
    wind_vector : {"speed_ms", "direction_deg_toward"}
    hours_back : duration of the reverse window
    n_particles : ensemble size
    windage : fraction of wind speed imparted to the slick (typical 2–4%)
    spread_deg : base angular/spatial diffusion scale (degrees of lat/lon
                 equivalent); perturbation grows with elapsed reverse time
    seed : RNG seed so a given slider value is reproducible during a demo

    Returns
    -------
    dict with keys:
        positions : (n_particles, 2) array of final (lat, lon)
        centroid : (lat, lon)
        radius_km : RMS radius around the centroid (approx source-zone size)
        hours_back, n_particles
    """
    rng = np.random.default_rng(seed)
    hours_back = float(max(hours_back, DT_HOURS))
    n_steps = int(round(hours_back / DT_HOURS))
    dt_s = DT_HOURS * 3600.0

    u_c, v_c = _toward_uv(float(current_vector["speed_ms"]), float(current_vector["direction_deg_toward"]))
    u_w, v_w = _toward_uv(float(wind_vector["speed_ms"]), float(wind_vector["direction_deg_toward"]))
    # Forward oil drift; reverse uses the negative.
    u_fwd = u_c + windage * u_w
    v_fwd = v_c + windage * v_w

    pos = _seed_particles(spill_polygon_latlon, n_particles, rng)

    for step in range(n_steps):
        # Uncertainty grows as we walk further back in time (random-walk envelope).
        grow = 1.0 + 0.08 * (step + 1)
        # Spatial kick in metres, converted per-particle at its latitude.
        # spread_deg ~ 0.05° ≈ 5.5 km; we use a much smaller per-step kick so
        # the cloud still tightens/loosens visibly with the hour slider.
        kick_m = 260.0 * grow * spread_deg / 0.05
        d_east = -u_fwd * dt_s + rng.normal(0.0, kick_m, size=n_particles)
        d_north = -v_fwd * dt_s + rng.normal(0.0, kick_m, size=n_particles)
        for i in range(n_particles):
            dlat, dlon = _meters_to_latlon_delta(float(d_east[i]), float(d_north[i]), float(pos[i, 0]))
            pos[i, 0] += dlat
            pos[i, 1] += dlon

    centroid = pos.mean(axis=0)
    # Equirectangular distance to centroid -> RMS radius in km.
    mean_lat = float(centroid[0])
    dlat_km = (pos[:, 0] - centroid[0]) * KM_PER_DEG_LAT
    dlon_km = (pos[:, 1] - centroid[1]) * KM_PER_DEG_LAT * math.cos(math.radians(mean_lat))
    dist_km = np.sqrt(dlat_km**2 + dlon_km**2)
    radius_km = float(np.sqrt(np.mean(dist_km**2)))
    # Floor + a modest buffer so AIS points near the cloud still count.
    # Grows with hours_back because the random walk inflates RMS radius.
    radius_km = max(radius_km * 1.15, 10.0)

    return {
        "positions": pos,
        "centroid": (float(centroid[0]), float(centroid[1])),
        "radius_km": radius_km,
        "hours_back": hours_back,
        "n_particles": n_particles,
        "forward_uv_ms": (u_fwd, v_fwd),
    }
