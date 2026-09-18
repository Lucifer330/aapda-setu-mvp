"""
Evidence scoring: correlate AIS tracks with the reverse-drift source zone.

Weights are equal (1/3 each) and ILLUSTRATIVE. A production system would
calibrate them on labelled incidents (and add identity, AIS-gap, and
oil-fingerprinting features). Scores support investigation ranking; they
do not assign legal responsibility.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

KM_PER_DEG_LAT = 111.32


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
    """Initial bearing from point 1 toward point 2, 0–360 clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def _ang_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two headings in degrees."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _overlap_hours(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> float:
    start = max(a0, b0)
    end = min(a1, b1)
    return max(0.0, (end - start).total_seconds() / 3600.0)


def evidence_level(score: float) -> str:
    # High > 0.66, Medium in (0.33, 0.66], Low <= 0.33.
    # Equality at 0.33 is treated as Low so a vessel that only scores on
    # one of the three equal-weight components does not look like a candidate.
    if score > 0.66:
        return "High"
    if score > 0.33:
        return "Medium"
    return "Low"


def score_tracks(
    ais_df: pd.DataFrame,
    source_centroid: tuple[float, float],
    source_radius_km: float,
    spill_lat: float,
    spill_lon: float,
    detection_time: datetime,
    hours_back: float,
) -> pd.DataFrame:
    """
    Compute spatial / temporal / course scores for each vessel.

    Spatial: peak soft-membership of in-window points vs the source-zone
             circle (1 inside the RMS radius, fading to 0 at 1.6× radius).
    Temporal: overlap of in-theatre track times with
              [detection - hours_back, detection] / hours_back.
    Course: 1 - mean_angular_error/180 for in-window points near the zone,
            where angular error is |vessel course − bearing to spill|.
    Overall: mean of the three (equal weights — calibrate with real data).
    """
    if detection_time.tzinfo is None:
        detection_time = detection_time.replace(tzinfo=timezone.utc)

    release_start = detection_time - pd.Timedelta(hours=float(hours_back))
    release_end = detection_time
    window_h = max(float(hours_back), 1e-6)
    c_lat, c_lon = source_centroid

    rows = []
    for vessel_id, g in ais_df.groupby("vessel_id"):
        g = g.sort_values("timestamp")
        times = [_to_utc(t) for t in g["timestamp"].tolist()]
        lats = g["lat"].to_numpy(dtype=float)
        lons = g["lon"].to_numpy(dtype=float)
        courses = g["course_deg"].to_numpy(dtype=float)

        dists = np.array([_haversine_km(la, lo, c_lat, c_lon) for la, lo in zip(lats, lons)])
        in_window = np.array([(release_start <= t <= release_end) for t in times])

        # Spatial: peak membership of in-window points (did the vessel actually
        # enter the estimated source zone, not an average diluted by the rest
        # of a long transit). Soft kernel fades to 0 at 1.6× RMS radius.
        if in_window.any():
            d_win = dists[in_window]
            fade = 1.6 * max(source_radius_km, 1e-6)
            spatial = float(np.max(np.clip(1.0 - d_win / fade, 0.0, 1.0)))
        else:
            spatial = 0.0

        # Temporal: overlap of the track with the release window, but only
        # using positions that are in the same theatre (~120 km of the slick).
        # A vessel transiting Gujarat while the window is open should not get
        # a full temporal score.
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
                if not is_near:
                    continue
                if t < release_start or t > release_end:
                    continue
                brng = _bearing_deg(la, lo, spill_lat, spill_lon)
                errs.append(_ang_diff(float(crs), brng))
            mean_err = float(np.mean(errs)) if errs else 180.0
            course_score = max(0.0, 1.0 - mean_err / 180.0)
        else:
            course_score = 0.0

        # Equal weights — illustrative only; would be calibrated on real cases.
        overall = (spatial + temporal + course_score) / 3.0

        # Marker: AIS position closest in time to the midpoint of the release window.
        mid = release_start + (release_end - release_start) / 2
        idx = int(np.argmin([abs((t - mid).total_seconds()) for t in times]))

        rows.append(
            {
                "vessel_id": vessel_id,
                "spatial_score": round(spatial, 3),
                "temporal_score": round(temporal, 3),
                "course_score": round(course_score, 3),
                "overall_score": round(overall, 3),
                "evidence_level": evidence_level(overall),
                "marker_lat": float(lats[idx]),
                "marker_lon": float(lons[idx]),
            }
        )

    out = pd.DataFrame(rows).sort_values("overall_score", ascending=False).reset_index(drop=True)
    return out


def closest_point_in_window(
    track: pd.DataFrame,
    detection_time: datetime,
    hours_back: float,
) -> Optional[pd.Series]:
    """Return the track row whose timestamp is nearest the release-window midpoint."""
    if track.empty:
        return None
    if detection_time.tzinfo is None:
        detection_time = detection_time.replace(tzinfo=timezone.utc)
    release_start = detection_time - pd.Timedelta(hours=float(hours_back))
    mid = release_start + (detection_time - release_start) / 2
    times = track["timestamp"].map(_to_utc)
    idx = int(np.argmin([abs((t - mid).total_seconds()) for t in times]))
    return track.iloc[idx]
