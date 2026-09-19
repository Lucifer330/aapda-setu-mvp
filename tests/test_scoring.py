"""Central scoring is the single source of truth for all pages."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_correlation import (  # noqa: E402
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    evidence_level,
    funnel_counts,
    score_tracks,
)
from drift_model import backtrack  # noqa: E402


def _load_bundle(hours_back: int = 8):
    data = ROOT / "data"
    with (data / "sample_mask_polygon.json").open(encoding="utf-8") as f:
        poly_meta = json.load(f)
    with (data / "environment.json").open(encoding="utf-8") as f:
        env = json.load(f)
    ais = pd.read_csv(data / "synthetic_ais.csv")
    ais["timestamp"] = pd.to_datetime(ais["timestamp"], utc=True)
    poly_ll = [(p["lat"], p["lon"]) for p in poly_meta["polygon_latlon"]]
    det = datetime.strptime(
        env["detection_time_utc"].replace(" UTC", "").strip(), "%Y-%m-%d %H:%M"
    ).replace(tzinfo=timezone.utc)
    drift = backtrack(
        spill_polygon_latlon=poly_ll,
        current_vector=env["current"],
        wind_vector=env["wind"],
        hours_back=hours_back,
        n_particles=150,
        windage=0.03,
        spread_deg=0.05,
    )
    scores = score_tracks(
        ais_df=ais,
        source_centroid=drift["centroid"],
        source_radius_km=drift["radius_km"],
        spill_lat=float(env["spill_lat"]),
        spill_lon=float(env["spill_lon"]),
        detection_time=det,
        hours_back=hours_back,
        forward_uv_ms=drift["forward_uv_ms"],
    )
    return scores


def test_overall_is_equal_weight_mean():
    scores = _load_bundle(8)
    for _, r in scores.iterrows():
        expected = round(
            (r["spatial_score"] + r["temporal_score"] + r["course_score"]) / 3.0, 3
        )
        assert r["overall_score"] == expected


def test_evidence_thresholds_centralized():
    assert evidence_level(0.919) == "High"
    assert evidence_level(0.573) == "Medium"
    assert evidence_level(0.32) == "Low"
    assert evidence_level(0.331) == "Medium"
    assert HIGH_THRESHOLD == 0.66
    assert MEDIUM_THRESHOLD == 0.33


def test_eight_hour_window_is_stable():
    a = _load_bundle(8)
    b = _load_bundle(8)
    pd.testing.assert_frame_equal(a, b)
    by_id = a.set_index("vessel_id")
    assert by_id.loc["V2", "overall_score"] == 0.919
    assert by_id.loc["V2", "evidence_level"] == "High"
    assert by_id.loc["V1", "overall_score"] == 0.830
    assert by_id.loc["V1", "evidence_level"] == "High"
    assert by_id.loc["V4", "overall_score"] == 0.573
    assert by_id.loc["V4", "evidence_level"] == "Medium"


def test_funnel_uses_scored_table():
    scores = _load_bundle(8)
    funnel = funnel_counts(scores)
    assert funnel["tracks_in_region"] == 8
    assert funnel["candidate_vessels"] == int(
        scores["evidence_level"].isin(["High", "Medium"]).sum()
    )
    assert "diagnostic_score" in scores.columns
    assert "investigation_review" in scores.columns
