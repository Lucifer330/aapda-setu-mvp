"""
Evidence scoring: correlate AIS tracks with the reverse-drift source zone.

This is the single source of truth for candidate scores.

Overall (equal weights, illustrative):
    overall = (spatial + temporal + course) / 3
    High > 0.66, Medium > 0.33, otherwise Low.

Diagnostic/behavioural score is computed for explainability and is NOT
a fourth overall weight. Ranking supports investigation; it does not
assign legal responsibility.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

KM_PER_DEG_LAT = 111.32
HIGH_THRESHOLD = 0.66
MEDIUM_THRESHOLD = 0.33


def _to_utc(ts) -> datetime:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.to_pydatetime()


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _overlap_hours(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> float:
    start = max(a0, b0)
    end = min(a1, b1)
    return max(0.0, (end - start).total_seconds() / 3600.0)


def evidence_level(score: float) -> str:
    """Single banding used for overall evidence and per-component labels."""
    if score > HIGH_THRESHOLD:
        return "High"
    if score > MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def review_action(level: str) -> str:
    return "Required" if level in ("High", "Medium") else "Optional"


def score_tracks(
    ais_df: pd.DataFrame,
    source_centroid: tuple[float, float],
    source_radius_km: float,
    spill_lat: float,
    spill_lon: float,
    detection_time: datetime,
    hours_back: float,
    forward_uv_ms: tuple[float, float] = (0.0, 0.0),
) -> pd.DataFrame:
    if detection_time.tzinfo is None:
        detection_time = detection_time.replace(tzinfo=timezone.utc)

    release_start = detection_time - pd.Timedelta(hours=float(hours_back))
    release_end = detection_time
    window_h = max(float(hours_back), 1e-6)
    c_lat, c_lon = source_centroid
    u_fwd, v_fwd = forward_uv_ms
    oil_toward = (math.degrees(math.atan2(u_fwd, v_fwd)) + 360.0) % 360.0

    rows = []
    for vessel_id, g in ais_df.groupby("vessel_id"):
        g = g.sort_values("timestamp")
        times = [_to_utc(t) for t in g["timestamp"].tolist()]
        lats = g["lat"].to_numpy(dtype=float)
        lons = g["lon"].to_numpy(dtype=float)
        courses = g["course_deg"].to_numpy(dtype=float)

        dists = np.array([_haversine_km(la, lo, c_lat, c_lon) for la, lo in zip(lats, lons)])
        in_window = np.array([(release_start <= t <= release_end) for t in times])

        if in_window.any():
            d_win = dists[in_window]
            fade = 1.6 * max(source_radius_km, 1e-6)
            spatial = float(np.max(np.clip(1.0 - d_win / fade, 0.0, 1.0)))
        else:
            spatial = 0.0

        dist_spill = np.array(
            [_haversine_km(la, lo, spill_lat, spill_lon) for la, lo in zip(lats, lons)]
        )
        in_theatre = dist_spill <= 120.0
        if in_theatre.any():
            theatre_times = [t for t, ok in zip(times, in_theatre) if ok]
            t0, t1 = min(theatre_times), max(theatre_times)
            temporal = min(1.0, _overlap_hours(t0, t1, release_start, release_end) / window_h)
        else:
            temporal = 0.0

        near = dists <= (3.0 * source_radius_km)
        if near.any():
            errs = []
            for la, lo, crs, is_near, t in zip(lats, lons, courses, near, times):
                if not is_near or t < release_start or t > release_end:
                    continue
                brng = _bearing_deg(la, lo, spill_lat, spill_lon)
                errs.append(_ang_diff(float(crs), brng))
            mean_err = float(np.mean(errs)) if errs else 180.0
            course_score = max(0.0, 1.0 - mean_err / 180.0)
        else:
            course_score = 0.0

        diag_errs = [
            _ang_diff(float(crs), oil_toward)
            for crs, t in zip(courses, times)
            if release_start <= t <= release_end
        ]
        diagnostic = max(0.0, 1.0 - float(np.mean(diag_errs)) / 180.0) if diag_errs else 0.0

        overall = (spatial + temporal + course_score) / 3.0
        level = evidence_level(overall)
        mid = release_start + (release_end - release_start) / 2
        idx = int(np.argmin([abs((t - mid).total_seconds()) for t in times]))

        rows.append(
            {
                "vessel_id": vessel_id,
                "spatial_score": round(spatial, 3),
                "temporal_score": round(temporal, 3),
                "course_score": round(course_score, 3),
                "diagnostic_score": round(diagnostic, 3),
                "overall_score": round(overall, 3),
                "evidence_level": level,
                "investigation_review": review_action(level),
                "in_time_window": bool(temporal > 0),
                "spatially_compatible": bool(spatial > 0),
                "marker_lat": float(lats[idx]),
                "marker_lon": float(lons[idx]),
            }
        )

    return pd.DataFrame(rows).sort_values("overall_score", ascending=False).reset_index(drop=True)


def funnel_counts(scores: pd.DataFrame) -> dict:
    n = int(len(scores))
    return {
        "tracks_in_region": n,
        "in_time_window": int(scores["in_time_window"].sum()) if n else 0,
        "spatially_compatible": int(scores["spatially_compatible"].sum()) if n else 0,
        "candidate_vessels": int(scores["evidence_level"].isin(["High", "Medium"]).sum()) if n else 0,
    }


def explain_candidate(row: pd.Series) -> list[str]:
    reasons = []
    if row["spatial_score"] > HIGH_THRESHOLD:
        reasons.append("Entered the probable source zone during the estimated source time window.")
    elif row["spatial_score"] > MEDIUM_THRESHOLD:
        reasons.append("Passed near the probable source zone (partial spatial compatibility).")
    else:
        reasons.append("Did not enter the probable source zone in this window.")

    if row["temporal_score"] > HIGH_THRESHOLD:
        reasons.append("AIS coverage overlaps most of the estimated source time window.")
    elif row["temporal_score"] > MEDIUM_THRESHOLD:
        reasons.append("Partial overlap with the estimated source time window.")
    else:
        reasons.append("Little or no presence in the estimated source time window.")

    if row["course_score"] > HIGH_THRESHOLD:
        reasons.append("Recorded heading is consistent with a path toward the observed slick.")
    elif row["course_score"] > MEDIUM_THRESHOLD:
        reasons.append("Heading is only partly aligned with a path toward the slick.")
    else:
        reasons.append("Heading is not aligned with a path toward the slick.")

    reasons.append(
        "Behavioural/diagnostic score compares heading with the scenario oil-drift "
        "direction; it is not included in the overall score."
    )
    return reasons


def closest_point_in_window(
    track: pd.DataFrame,
    detection_time: datetime,
    hours_back: float,
) -> Optional[pd.Series]:
    if track.empty:
        return None
    if detection_time.tzinfo is None:
        detection_time = detection_time.replace(tzinfo=timezone.utc)
    release_start = detection_time - pd.Timedelta(hours=float(hours_back))
    mid = release_start + (detection_time - release_start) / 2
    times = track["timestamp"].map(_to_utc)
    idx = int(np.argmin([abs((t - mid).total_seconds()) for t in times]))
    return track.iloc[idx]
