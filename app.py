"""
AAPDA SETU — oil spill source investigation prototype (SIH26143).

Run from the project root:
    python src/data_gen.py
    streamlit run app.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

import folium
from folium.plugins import HeatMap
from branca.element import MacroElement
from jinja2 import Template

ROOT = Path(__file__).resolve().parent


def _load_src_module(mod_name: str) -> ModuleType:
    """Load src/<mod>.py by absolute path.

    Streamlit Community Cloud clones into /mount/src/<repo>. A project folder
    named src/ plus a top-level `from ais_correlation import ...` is then
    resolved against /mount/src, not this repository. File-based import avoids
    that collision.
    """
    path = ROOT / "src" / f"{mod_name}.py"
    if not path.is_file():
        raise ImportError(f"Missing {path}")
    spec = importlib.util.spec_from_file_location(f"aapda_{mod_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ais_correlation = _load_src_module("ais_correlation")
evidence_level = _ais_correlation.evidence_level
explain_candidate = _ais_correlation.explain_candidate
funnel_counts = _ais_correlation.funnel_counts
score_tracks = _ais_correlation.score_tracks
backtrack = _load_src_module("drift_model").backtrack
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"
KM_PER_DEG_LAT = 111.32
PRESET_HOURS = [3, 6, 8, 12, 24]
NAV_ITEMS = ["Dashboard", "Incidents", "Vessels", "Analytics", "Evidence", "About"]
INCIDENT_ID = "AS-DEMO-2026-0114"

LEVEL_COLORS = {
    "High": "#c2410c",
    "Medium": "#b45309",
    "Low": "#64748b",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: "Source Sans 3", "Segoe UI", sans-serif; }
header[data-testid="stHeader"] { display: none; }
.stApp { background: #F7F8FA; }
.block-container { padding-top: 1.4rem; padding-bottom: 2.4rem; max-width: 1320px; }

.as-nav {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 10px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.as-brand { display: flex; align-items: center; gap: 10px; }
.as-mark {
    width: 28px; height: 28px; border-radius: 4px;
    background: #1e3a5f; color: #fff; font-size: 11px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.as-name { font-weight: 700; color: #111827; font-size: 1.02rem; line-height: 1.1; }
.as-name span { display: block; font-weight: 400; color: #6b7280; font-size: 0.72rem; }
.as-nav-right { display: flex; align-items: center; gap: 16px; font-size: 0.82rem; color: #4b5563; }
.as-dot { width: 8px; height: 8px; border-radius: 50%; background: #16a34a; display: inline-block; margin-right: 6px; }
.as-demo-chip {
    display: inline-block; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em;
    color: #9a3412; background: #fff7ed; border: 1px solid #fdba74;
    padding: 2px 8px; border-radius: 4px; margin-left: 8px;
}

.as-h1 { font-size: 1.45rem; font-weight: 700; color: #111827; margin: 4px 0 0 0; }
.as-sub { color: #4b5563; font-size: 0.95rem; margin: 4px 0 14px 0; max-width: 720px; }
.as-h2 { font-size: 0.92rem; font-weight: 650; color: #111827; margin: 0 0 8px 0; }
.as-meta { color: #6b7280; font-size: 0.8rem; }

.as-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 8px 0 16px 0; }
.as-metric {
    background: #fff; border: 1px solid #e5e7eb; border-radius: 6px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04); padding: 12px 14px;
}
.as-metric .k { font-size: 0.72rem; color: #6b7280; font-weight: 600; }
.as-metric .v { font-size: 1.15rem; color: #111827; font-weight: 650; margin-top: 4px; }
.as-metric .h { font-size: 0.75rem; color: #6b7280; margin-top: 3px; }

.as-panel {
    background: #fff; border: 1px solid #e5e7eb; border-radius: 6px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04); padding: 14px 16px;
}
.as-kv { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
.as-kv td { padding: 7px 0; border-top: 1px solid #f3f4f6; vertical-align: top; }
.as-kv td.k { color: #6b7280; width: 42%; }
.as-kv td.v { color: #111827; }

.as-flow { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 8px 0 16px 0; }
.as-flow span {
    background: #fff; border: 1px solid #e5e7eb; color: #374151;
    font-size: 0.75rem; padding: 4px 8px; border-radius: 4px;
}
.as-flow .arr { background: none; border: none; color: #9ca3af; padding: 0 2px; }

.as-banner {
    background: #fffbeb; border: 1px solid #fde68a; color: #78350f;
    font-size: 0.82rem; padding: 8px 12px; border-radius: 6px; margin-bottom: 14px;
}
.as-foot { color: #6b7280; font-size: 0.78rem; margin-top: 22px; }
.as-badge-high { color: #9a3412; background: #fff7ed; border: 1px solid #fdba74; padding: 1px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 650; }
.as-badge-med { color: #92400e; background: #fffbeb; border: 1px solid #fcd34d; padding: 1px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 650; }
.as-badge-low { color: #4b5563; background: #f3f4f6; border: 1px solid #e5e7eb; padding: 1px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 650; }

.as-funnel { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 8px 0 14px 0; }
.as-funnel-step {
    background: #fff; border: 1px solid #e5e7eb; border-radius: 6px;
    padding: 10px 12px; box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
.as-funnel-step .n { font-size: 1.2rem; font-weight: 650; color: #111827; }
.as-funnel-step .l { font-size: 0.72rem; color: #6b7280; font-weight: 600; margin-top: 2px; }
.as-why li { margin: 0 0 6px 0; color: #374151; font-size: 0.86rem; }

div[data-testid="stHorizontalBlock"] button { border-radius: 4px; }
@media (max-width: 1100px) {
    .as-metrics, .as-funnel { grid-template-columns: repeat(2, 1fr); }
}
</style>
"""


def _ensure_sample_data() -> None:
    needed = [
        DATA / "sample_spill.png",
        DATA / "sample_mask.png",
        DATA / "sample_mask_polygon.json",
        DATA / "synthetic_ais.csv",
        DATA / "environment.json",
    ]
    if all(p.exists() for p in needed):
        return
    gen_main = _load_src_module("data_gen").main
    gen_main()


@st.cache_data(show_spinner=False)
def load_scenario() -> dict:
    _ensure_sample_data()
    with (DATA / "sample_mask_polygon.json").open(encoding="utf-8") as f:
        poly_meta = json.load(f)
    with (DATA / "environment.json").open(encoding="utf-8") as f:
        env = json.load(f)
    ais = pd.read_csv(DATA / "synthetic_ais.csv")
    ais["timestamp"] = pd.to_datetime(ais["timestamp"], utc=True)
    return {"poly_meta": poly_meta, "env": env, "ais": ais}


def load_images() -> tuple[Image.Image, Image.Image]:
    spill = Image.open(DATA / "sample_spill.png").convert("L")
    mask = Image.open(DATA / "sample_mask.png").convert("L")
    return spill, mask


def polygon_area_km2(ring: list[tuple[float, float]]) -> float:
    pts = list(ring)
    if pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return 0.0
    lat0 = sum(p[0] for p in pts) / len(pts)
    lon0 = sum(p[1] for p in pts) / len(pts)
    km_lon = KM_PER_DEG_LAT * math.cos(math.radians(lat0))
    xy = [((lon - lon0) * km_lon, (lat - lat0) * KM_PER_DEG_LAT) for lat, lon in pts]
    acc = 0.0
    n = len(xy)
    for i in range(n):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % n]
        acc += x1 * y2 - x2 * y1
    return abs(acc) / 2.0


def overlay_slick(spill: Image.Image, polygon_pixels: list) -> Image.Image:
    """Draw the demo spill outline with Pillow."""
    img = spill.convert("RGB")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    pts = [(float(p[0]), float(p[1])) for p in polygon_pixels]
    if not pts:
        return img
    if pts[0] != pts[-1]:
        pts = pts + [pts[0]]
    draw.line(pts, fill=(194, 65, 12), width=3)
    return img


def detection_dt(env: dict) -> datetime:
    raw = env["detection_time_utc"].replace(" UTC", "").strip()
    return datetime.strptime(raw, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


def fmt_ll(lat: float, lon: float) -> str:
    return f"{lat:.3f}°N, {lon:.3f}°E"


def source_time_window(det: datetime, hours_back: int) -> str:
    start = det - timedelta(hours=int(hours_back))
    return (
        f"{start.strftime('%Y-%m-%d %H:%M')} UTC  to  "
        f"{det.strftime('%Y-%m-%d %H:%M')} UTC"
    )


@st.cache_data(show_spinner=False)
def compute_products(hours_back: int) -> dict:
    """Single cached reverse-drift + scoring pass for a backtracking window."""
    scenario = load_scenario()
    poly_ll = [(p["lat"], p["lon"]) for p in scenario["poly_meta"]["polygon_latlon"]]
    env = scenario["env"]
    det = detection_dt(env)
    drift = backtrack(
        spill_polygon_latlon=poly_ll,
        current_vector=env["current"],
        wind_vector=env["wind"],
        hours_back=int(hours_back),
        n_particles=150,
        windage=0.03,
        spread_deg=0.05,
    )
    scores = score_tracks(
        ais_df=scenario["ais"],
        source_centroid=drift["centroid"],
        source_radius_km=drift["radius_km"],
        spill_lat=float(env["spill_lat"]),
        spill_lon=float(env["spill_lon"]),
        detection_time=det,
        hours_back=int(hours_back),
        forward_uv_ms=drift["forward_uv_ms"],
    )
    return {"drift": drift, "scores": scores}


def candidate_table(scores: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Candidate Vessel": scores["vessel_id"],
            "Spatial Compatibility": scores["spatial_score"],
            "Temporal Compatibility": scores["temporal_score"],
            "Trajectory Consistency": scores["course_score"],
            "Behavioural / Diagnostic Evidence": scores["diagnostic_score"],
            "Overall Score": scores["overall_score"],
            "Evidence Level": scores["evidence_level"],
            "Investigation Review": scores["investigation_review"],
        }
    )


def funnel_html(funnel: dict) -> str:
    return f"""
        <div class="as-funnel">
          <div class="as-funnel-step"><div class="l">AIS tracks in region</div>
            <div class="n">{funnel['tracks_in_region']}</div></div>
          <div class="as-funnel-step"><div class="l">In source time window</div>
            <div class="n">{funnel['in_time_window']}</div></div>
          <div class="as-funnel-step"><div class="l">Spatially compatible</div>
            <div class="n">{funnel['spatially_compatible']}</div></div>
          <div class="as-funnel-step"><div class="l">Candidate vessels</div>
            <div class="n">{funnel['candidate_vessels']}</div></div>
        </div>
        """


class MapLegend(MacroElement):
    def __init__(self):
        super().__init__()
        self._name = "MapLegend"
        self._template = Template(
            """
            {% macro html(this, kwargs) %}
            <div style="
                position: fixed; bottom: 28px; left: 28px; z-index: 9999;
                background: rgba(255,255,255,0.94); color: #111827;
                border: 1px solid #e5e7eb; padding: 10px 12px; border-radius: 6px;
                font-size: 12px; font-family: 'Source Sans 3', sans-serif;
                line-height: 1.7; box-shadow: 0 1px 2px rgba(16,24,40,0.08);">
                <div style="font-weight:650;color:#6b7280;font-size:10px;letter-spacing:0.08em;">MAP LAYERS</div>
                <div><span style="color:#c2410c;">&#9632;</span> Observed slick</div>
                <div><span style="color:#0f766e;">&#9679;</span> Reverse drift</div>
                <div><span style="color:#1d4ed8;">&#9711;</span> Probable source zone</div>
                <div><span style="color:#64748b;">&#8212;</span> AIS track</div>
                <div><span style="color:#1e3a5f;">&#9670;</span> Candidate vessel</div>
            </div>
            {% endmacro %}
            """
        )


def build_map(
    poly_ll: list[tuple[float, float]],
    particles: np.ndarray,
    centroid: tuple[float, float],
    radius_km: float,
    ais: pd.DataFrame,
    scores: pd.DataFrame,
    show_heat: bool,
    spill_lat: float,
    spill_lon: float,
    selected_vessel: str,
) -> folium.Map:
    m = folium.Map(
        location=[(spill_lat + centroid[0]) / 2.0, (spill_lon + centroid[1]) / 2.0],
        zoom_start=8,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    folium.Polygon(
        locations=[[lat, lon] for lat, lon in poly_ll],
        color="#c2410c",
        weight=3,
        fill=True,
        fill_color="#ea580c",
        fill_opacity=0.30,
        tooltip="Observed oil slick",
    ).add_to(m)
    folium.CircleMarker(
        location=[spill_lat, spill_lon],
        radius=6,
        color="#ffffff",
        fill=True,
        fill_color="#c2410c",
        fill_opacity=1.0,
        tooltip="Spill location",
    ).add_to(m)

    if show_heat and len(particles):
        heat = [[float(lat), float(lon), 0.65] for lat, lon in particles]
        HeatMap(
            heat,
            radius=18,
            blur=15,
            min_opacity=0.28,
            gradient={0.2: "#99f6e4", 0.55: "#14b8a6", 0.85: "#0f766e"},
            name="Reverse drift",
        ).add_to(m)

    folium.Circle(
        location=[centroid[0], centroid[1]],
        radius=radius_km * 1000.0,
        color="#1d4ed8",
        weight=2,
        fill=True,
        fill_color="#1d4ed8",
        fill_opacity=0.05,
        dash_array="6,6",
        tooltip=f"Probable source zone · ~{radius_km:.1f} km RMS radius",
    ).add_to(m)
    folium.CircleMarker(
        location=[centroid[0], centroid[1]],
        radius=5,
        color="#1d4ed8",
        fill=True,
        fill_opacity=1.0,
        tooltip="Source-zone centroid",
    ).add_to(m)

    level_by_id = dict(zip(scores["vessel_id"], scores["evidence_level"]))
    marker_by_id = {
        r["vessel_id"]: (r["marker_lat"], r["marker_lon"], r["evidence_level"], r["overall_score"])
        for _, r in scores.iterrows()
    }

    for vid, g in ais.groupby("vessel_id"):
        g = g.sort_values("timestamp")
        level = level_by_id.get(vid, "Low")
        is_sel = vid == selected_vessel
        color = "#1e3a5f" if is_sel else LEVEL_COLORS[level]
        folium.PolyLine(
            locations=list(zip(g["lat"].tolist(), g["lon"].tolist())),
            color=color,
            weight=5 if is_sel else (3 if level == "High" else 2),
            opacity=0.95 if is_sel else 0.82,
            tooltip=f"{vid} · {level}",
        ).add_to(m)
        ml, mo, ev, sc = marker_by_id[vid]
        folium.CircleMarker(
            location=[ml, mo],
            radius=9 if is_sel else 6,
            color="#ffffff" if is_sel else color,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            popup=folium.Popup(
                f"<b>{vid}</b><br>Evidence level: {ev}<br>Overall: {sc:.3f}<br>For investigation review",
                max_width=240,
            ),
        ).add_to(m)
        folium.Marker(
            location=[ml, mo],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:11px;color:#111827;font-weight:650;'
                    f"background:#fff;border:1px solid #e5e7eb;padding:1px 4px;"
                    f'border-radius:3px;white-space:nowrap;">{vid}</div>'
                )
            ),
        ).add_to(m)

    MapLegend().add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def evidence_badge(level: str) -> str:
    cls = {"High": "as-badge-high", "Medium": "as-badge-med", "Low": "as-badge-low"}[level]
    return f"<span class='{cls}'>{level}</span>"


def render_nav(page: str) -> str:
    brand, status = st.columns([1.4, 1.1])
    with brand:
        st.markdown(
            "<div style='font-weight:700;font-size:1.05rem;color:#111827;'>AAPDA SETU</div>"
            "<div style='font-size:0.8rem;color:#6b7280;'>Marine Oil Spill Investigation</div>",
            unsafe_allow_html=True,
        )
    with status:
        st.markdown(
            "<div style='text-align:right;font-size:0.82rem;color:#4b5563;padding-top:4px;'>"
            "<span style='display:inline-block;width:8px;height:8px;background:#16a34a;"
            "border-radius:50%;margin-right:6px;vertical-align:middle;'></span>"
            "System status · operational (demo)<br/>Analyst · demo session</div>",
            unsafe_allow_html=True,
        )
    cols = st.columns(len(NAV_ITEMS))
    chosen = page
    for col, name in zip(cols, NAV_ITEMS):
        if col.button(
            name,
            key=f"nav_{name}",
            width="stretch",
            type="primary" if name == page else "secondary",
        ):
            chosen = name
    if chosen != page:
        st.session_state.page = chosen
        st.rerun()
    return page


def render_report(ctx: dict) -> None:
    scores = ctx["scores"]
    ids = ctx["ids"]
    high_ids = ", ".join(scores.loc[scores["evidence_level"] == "High", "vessel_id"].tolist()) or "None"
    med_ids = ", ".join(scores.loc[scores["evidence_level"] == "Medium", "vessel_id"].tolist()) or "None"
    summary_bits = []
    for _, r in scores.head(3).iterrows():
        summary_bits.append(
            f"{r['vessel_id']} · {r['evidence_level']} "
            f"(spatial {r['spatial_score']:.3f}, temporal {r['temporal_score']:.3f}, "
            f"course {r['course_score']:.3f}, overall {r['overall_score']:.3f})"
        )
    st.markdown(
        f"""
        <div class="as-panel">
          <div class="as-h2">Investigation report</div>
          <table class="as-kv">
            <tr><td class="k">Incident</td><td class="v">{INCIDENT_ID} · Arabian Sea demonstration case (SIH26143)</td></tr>
            <tr><td class="k">Detection time</td><td class="v">{ctx['env']['detection_time_utc']}</td></tr>
            <tr><td class="k">Spill location</td><td class="v">{fmt_ll(ctx['spill_lat'], ctx['spill_lon'])}</td></tr>
            <tr><td class="k">Spill confidence</td><td class="v">N/A (static sample mask — not model-inferred)</td></tr>
            <tr><td class="k">Backtracking window</td><td class="v">{ctx['hours_back']} hours</td></tr>
            <tr><td class="k">Probable source zone</td><td class="v">{ctx['zone_txt']} (ensemble centroid and RMS radius)</td></tr>
            <tr><td class="k">Estimated source time window</td><td class="v">{ctx['window_txt']}</td></tr>
            <tr><td class="k">Candidate vessels</td>
                <td class="v">High: {high_ids}<br/>Medium: {med_ids}<br/>Tracks ranked: {', '.join(ids)}</td></tr>
            <tr><td class="k">Evidence summary</td><td class="v">{"<br/>".join(summary_bits)}</td></tr>
            <tr><td class="k">Investigation status</td>
                <td class="v">Open — candidate generation complete; analyst review required</td></tr>
          </table>
          <p class="as-meta" style="margin:12px 0 0 0;">
            Candidate generation supports investigation; final determination remains with the human analyst.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_backtrack_controls() -> bool:
    st.markdown('<div class="as-h2">Backtracking window</div>', unsafe_allow_html=True)
    preset = st.columns(len(PRESET_HOURS) + 2)
    for h, col in zip(PRESET_HOURS, preset):
        if col.button(
            f"{h}h",
            key=f"preset_{h}",
            width="stretch",
            type="primary" if st.session_state.hours_back == h else "secondary",
        ):
            st.session_state.hours_back = h
            st.session_state.hours_slider_widget = h
            st.rerun()
    with preset[-2]:
        st.slider(
            "Hours",
            min_value=3,
            max_value=24,
            step=1,
            value=int(st.session_state.hours_back),
            key="hours_slider_widget",
            label_visibility="collapsed",
        )
    with preset[-1]:
        show_heat = st.checkbox("Particle heatmap", value=bool(st.session_state.show_heat))
        st.session_state.show_heat = bool(show_heat)
        return bool(show_heat)


def page_dashboard(ctx: dict) -> None:
    st.markdown('<div class="as-h1">Marine Oil Spill Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="as-sub">Detect spills, reconstruct probable origin, and correlate vessel movements for investigation.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="as-banner"><span class="as-demo-chip">DEMO / SYNTHETIC</span> '
        "This workspace uses a synthetic SAR scene and illustrative AIS tracks. "
        "It is not a live operational feed.</div>",
        unsafe_allow_html=True,
    )
    n_cand = ctx["funnel"]["candidate_vessels"]
    st.markdown(
        f"""
        <div class="as-metrics">
          <div class="as-metric"><div class="k">Active incidents</div>
            <div class="v">1</div><div class="h">Case {INCIDENT_ID}</div></div>
          <div class="as-metric"><div class="k">Under investigation</div>
            <div class="v">1</div><div class="h">Open — investigation review</div></div>
          <div class="as-metric"><div class="k">Candidate vessels</div>
            <div class="v">{n_cand}</div><div class="h">{ctx['n_high']} High · {ctx['n_med']} Medium</div></div>
          <div class="as-metric"><div class="k">Latest processed scene</div>
            <div class="v">{ctx['env']['detection_time_utc']}</div><div class="h">Synthetic scene timestamp</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="as-flow">
            <span>DETECT</span><span class="arr">→</span>
            <span>TRACE</span><span class="arr">→</span>
            <span>CORRELATE</span><span class="arr">→</span>
            <span>RANK</span><span class="arr">→</span>
            <span>REVIEW</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="as-h2">Operational map</div>', unsafe_allow_html=True)
    fmap = build_map(
        poly_ll=ctx["poly_ll"],
        particles=ctx["drift"]["positions"],
        centroid=ctx["drift"]["centroid"],
        radius_km=ctx["drift"]["radius_km"],
        ais=ctx["ais"],
        scores=ctx["scores"],
        show_heat=st.session_state.show_heat,
        spill_lat=ctx["spill_lat"],
        spill_lon=ctx["spill_lon"],
        selected_vessel=st.session_state.selected_vessel,
    )
    st_folium(fmap, width=None, height=560, returned_objects=[], key="dashboard_map")
    st.caption(
        "Observed slick, reverse-drift particles, probable source zone, AIS tracks, and candidate markers. "
        "DEMO / SYNTHETIC layers."
    )
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown('<div class="as-h2">Investigation summary</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="as-panel">
            <table class="as-kv">
              <tr><td class="k">Incident ID</td><td class="v">{INCIDENT_ID} <span class="as-demo-chip">DEMO</span></td></tr>
              <tr><td class="k">Detection time</td><td class="v">{ctx['env']['detection_time_utc']}</td></tr>
              <tr><td class="k">Location</td><td class="v">{fmt_ll(ctx['spill_lat'], ctx['spill_lon'])}</td></tr>
              <tr><td class="k">Estimated spill area</td><td class="v">{ctx['area_km2']:.2f} km²</td></tr>
              <tr><td class="k">Detection confidence</td><td class="v">N/A (demo spill segmentation / static mask)</td></tr>
              <tr><td class="k">Source time window</td><td class="v">{ctx['window_txt']}</td></tr>
              <tr><td class="k">Probable source zone</td><td class="v">{ctx['zone_txt']}</td></tr>
              <tr><td class="k">Investigation status</td><td class="v">Open — Investigation Review</td></tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open incident workspace", type="primary"):
            st.session_state.page = "Incidents"
            st.rerun()
    with right:
        st.markdown('<div class="as-h2">AIS filtering funnel</div>', unsafe_allow_html=True)
        st.markdown(funnel_html(ctx["funnel"]), unsafe_allow_html=True)
        st.caption("Counts are from the scored demo tracks for the current backtracking window.")
        st.markdown('<div class="as-h2">Top-ranked candidates</div>', unsafe_allow_html=True)
        st.caption("Same scores as Vessels, Analytics, and Evidence.")
        for _, r in ctx["scores"].head(4).iterrows():
            st.markdown(
                f"**{r['vessel_id']}** · overall {r['overall_score']:.3f} · {evidence_badge(r['evidence_level'])}",
                unsafe_allow_html=True,
            )


def page_incidents(ctx: dict) -> None:
    st.markdown('<div class="as-h1">Incident workspace</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="as-sub">CASE {INCIDENT_ID} · Open — Investigation Review · Arabian Sea demonstration</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="as-banner"><span class="as-demo-chip">DEMO / SYNTHETIC</span> '
        "Observed spill → probable source zone → relevant AIS window → candidate vessels → evidence review.</div>",
        unsafe_allow_html=True,
    )
    show_heat = render_backtrack_controls()
    left, right = st.columns([0.68, 0.32], gap="medium")
    with left:
        st.markdown('<div class="as-h2">Operational map</div>', unsafe_allow_html=True)
        fmap = build_map(
            poly_ll=ctx["poly_ll"],
            particles=ctx["drift"]["positions"],
            centroid=ctx["drift"]["centroid"],
            radius_km=ctx["drift"]["radius_km"],
            ais=ctx["ais"],
            scores=ctx["scores"],
            show_heat=show_heat,
            spill_lat=ctx["spill_lat"],
            spill_lon=ctx["spill_lon"],
            selected_vessel=st.session_state.selected_vessel,
        )
        st_folium(fmap, width=None, height=620, returned_objects=[], key="incident_map")
        st.caption(
            f"Probable source zone: {ctx['zone_txt']}. Estimated source time window: {ctx['window_txt']} "
            f"(detection minus backtracking window; not a precise release clock)."
        )
    with right:
        st.markdown('<div class="as-h2">Incident information</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="as-panel">
            <table class="as-kv">
              <tr><td class="k">Incident ID</td><td class="v">{INCIDENT_ID}</td></tr>
              <tr><td class="k">Detection time</td><td class="v">{ctx['env']['detection_time_utc']}</td></tr>
              <tr><td class="k">Location</td><td class="v">{fmt_ll(ctx['spill_lat'], ctx['spill_lon'])}</td></tr>
              <tr><td class="k">Estimated spill area</td><td class="v">{ctx['area_km2']:.2f} km²</td></tr>
              <tr><td class="k">Detection confidence</td><td class="v">N/A (demo spill segmentation)</td></tr>
              <tr><td class="k">Source time window</td><td class="v">{ctx['window_txt']}</td></tr>
              <tr><td class="k">Source zone</td><td class="v">{ctx['zone_txt']}</td></tr>
              <tr><td class="k">Investigation status</td><td class="v">Open — Investigation Review</td></tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Candidate ranking does not assign legal responsibility.")
        st.markdown('<div class="as-h2" style="margin-top:12px;">AIS funnel</div>', unsafe_allow_html=True)
        st.markdown(funnel_html(ctx["funnel"]), unsafe_allow_html=True)
        st.markdown('<div class="as-h2">Candidate vessels</div>', unsafe_allow_html=True)
        for _, r in ctx["scores"].iterrows():
            mark = " · selected" if r["vessel_id"] == st.session_state.selected_vessel else ""
            st.markdown(
                f"`{r['vessel_id']}` {evidence_badge(r['evidence_level'])} · {r['overall_score']:.3f}{mark}",
                unsafe_allow_html=True,
            )
        st.selectbox("Focus candidate vessel", options=ctx["ids"], key="selected_vessel")


def page_vessels(ctx: dict) -> None:
    st.markdown('<div class="as-h1">Candidate vessels</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="as-sub">Evidence levels support investigation review. They do not identify a legally responsible vessel.</p>',
        unsafe_allow_html=True,
    )
    st.caption("Illustrative AIS tracks · DEMO / SYNTHETIC · scores from the centralized scoring module")
    st.selectbox("Focus candidate vessel", options=ctx["ids"], key="selected_vessel")
    sel = ctx["scores"][ctx["scores"]["vessel_id"] == st.session_state.selected_vessel].iloc[0]
    reasons = explain_candidate(sel)

    st.markdown(
        f"""
        <div class="as-panel">
          <div class="as-h2">Selected candidate</div>
          <table class="as-kv">
            <tr><td class="k">Candidate vessel</td><td class="v"><b>{sel['vessel_id']}</b></td></tr>
            <tr><td class="k">Spatial compatibility</td>
                <td class="v">{sel['spatial_score']:.3f} · {evidence_level(sel['spatial_score'])}</td></tr>
            <tr><td class="k">Temporal compatibility</td>
                <td class="v">{sel['temporal_score']:.3f} · {evidence_level(sel['temporal_score'])}</td></tr>
            <tr><td class="k">Trajectory consistency</td>
                <td class="v">{sel['course_score']:.3f} · {evidence_level(sel['course_score'])}</td></tr>
            <tr><td class="k">Behavioural / diagnostic</td>
                <td class="v">{sel['diagnostic_score']:.3f} · {evidence_level(sel['diagnostic_score'])} (not in overall)</td></tr>
            <tr><td class="k">Overall evidence</td><td class="v">{sel['overall_score']:.3f}</td></tr>
            <tr><td class="k">Evidence level</td><td class="v">{evidence_badge(sel['evidence_level'])}</td></tr>
            <tr><td class="k">Investigation review</td><td class="v">{sel['investigation_review']}</td></tr>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View evidence", type="primary"):
        st.session_state.page = "Evidence"
        st.rerun()

    st.markdown('<div class="as-h2" style="margin-top:14px;">Why this candidate?</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='as-panel as-why'><ul>"
        + "".join(f"<li>{r}</li>" for r in reasons)
        + "</ul>"
        "<p class='as-meta'>Source corridor + historical vessel trajectory + relevant time window "
        "are shown on the Incident and Evidence maps.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="as-h2">Ranked tracks</div>', unsafe_allow_html=True)
    st.dataframe(candidate_table(ctx["scores"]), hide_index=True, width="stretch")


def page_analytics(ctx: dict) -> None:
    st.markdown('<div class="as-h1">Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="as-sub">Component scores for the selected candidate. Overall = (spatial + temporal + trajectory) / 3. '
        "Behavioural/diagnostic is shown separately.</p>",
        unsafe_allow_html=True,
    )
    st.selectbox("Candidate vessel", options=ctx["ids"], key="selected_vessel")
    sel = ctx["scores"][ctx["scores"]["vessel_id"] == st.session_state.selected_vessel].iloc[0]
    diag = float(sel["diagnostic_score"])

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure(
            go.Bar(
                x=[sel["spatial_score"], sel["temporal_score"], sel["course_score"], diag],
                y=["Spatial", "Temporal", "Trajectory", "Behavioural / Diagnostic"],
                orientation="h",
                marker_color=["#1d4ed8", "#0f766e", "#b45309", "#64748b"],
                text=[
                    f"{sel['spatial_score']:.3f}",
                    f"{sel['temporal_score']:.3f}",
                    f"{sel['course_score']:.3f}",
                    f"{diag:.3f}",
                ],
                textposition="outside",
            )
        )
        fig.update_layout(
            template="plotly_white",
            height=280,
            margin=dict(l=10, r=48, t=8, b=8),
            xaxis=dict(range=[0, 1.15], title="Score (0–1)", gridcolor="#f3f4f6"),
            yaxis=dict(autorange="reversed"),
            font=dict(color="#1f2937", size=12),
            showlegend=False,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Overall evidence is the mean of spatial, temporal, and course. "
            "Behavioural/diagnostic compares heading with the scenario oil-drift direction; it is not a fourth overall weight."
        )
    with c2:
        fig2 = go.Figure(
            go.Bar(
                x=ctx["scores"]["vessel_id"],
                y=ctx["scores"]["overall_score"],
                marker_color=[
                    "#c2410c" if lv == "High" else "#b45309" if lv == "Medium" else "#94a3b8"
                    for lv in ctx["scores"]["evidence_level"]
                ],
            )
        )
        fig2.update_layout(
            template="plotly_white",
            height=280,
            margin=dict(l=10, r=10, t=8, b=8),
            yaxis=dict(range=[0, 1], title="Overall score", gridcolor="#f3f4f6"),
            xaxis_title="Candidate vessel",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#1f2937", size=12),
        )
        st.plotly_chart(fig2, width="stretch")
        st.caption("Overall scores across all illustrative AIS tracks — identical to the Candidate Vessels table.")
    st.markdown(
        f"**{sel['vessel_id']}** · overall {sel['overall_score']:.3f} · {evidence_badge(sel['evidence_level'])} · "
        f"investigation review: {sel['investigation_review']}",
        unsafe_allow_html=True,
    )


def page_evidence(ctx: dict) -> None:
    st.markdown('<div class="as-h1">Evidence</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="as-sub">Structured view of the same computed products: satellite sample, reverse drift, AIS correlation, and timeline.</p>',
        unsafe_allow_html=True,
    )
    sat, drift_t, ais_t, time_t = st.tabs(["Satellite", "Drift analysis", "AIS correlation", "Timeline"])
    with sat:
        st.caption("Synthetic SAR-like scene · mask is a static sample, not model inference")
        c1, c2 = st.columns(2)
        overlay = overlay_slick(ctx["spill_img"], ctx["poly_meta"]["polygon_pixels"])
        with c1:
            st.image(ctx["spill_img"], caption="Original scene (synthetic)", width="stretch")
        with c2:
            st.image(overlay, caption="Demo spill segmentation outline (static/pre-drawn mask)", width="stretch")
        st.markdown(
            f"Estimated area **{ctx['area_km2']:.2f} km²** · detection confidence **N/A (demo spill segmentation)**"
        )
    with drift_t:
        st.caption("Reverse Lagrangian ensemble using the scenario current and wind vectors")
        fmap = build_map(
            poly_ll=ctx["poly_ll"],
            particles=ctx["drift"]["positions"],
            centroid=ctx["drift"]["centroid"],
            radius_km=ctx["drift"]["radius_km"],
            ais=ctx["ais"],
            scores=ctx["scores"],
            show_heat=True,
            spill_lat=ctx["spill_lat"],
            spill_lon=ctx["spill_lon"],
            selected_vessel=st.session_state.selected_vessel,
        )
        st_folium(fmap, width=None, height=520, returned_objects=[], key="evidence_map")
        st.caption(
            f"Source zone {ctx['zone_txt']} · window {ctx['window_txt']} · "
            f"current {ctx['env']['current']['speed_ms']} m/s toward {ctx['env']['current']['direction_deg_toward']}° · "
            f"wind {ctx['env']['wind']['speed_ms']} m/s toward {ctx['env']['wind']['direction_deg_toward']}°"
        )
    with ais_t:
        st.caption("Vessel trajectories relative to the inferred source area and time window")
        st.dataframe(candidate_table(ctx["scores"]), hide_index=True, width="stretch")
        st.selectbox("Focus candidate vessel", options=ctx["ids"], key="selected_vessel")
        sel = ctx["scores"][ctx["scores"]["vessel_id"] == st.session_state.selected_vessel].iloc[0]
        st.markdown("**Why this candidate?**")
        for line in explain_candidate(sel):
            st.markdown(f"- {line}")
    with time_t:
        st.markdown(
            f"""
            1. **Detection** — {ctx['env']['detection_time_utc']} at {fmt_ll(ctx['spill_lat'], ctx['spill_lon'])} (synthetic scene; demo spill segmentation).
            2. **Backtracking** — reverse drift over **{ctx['hours_back']} h** → zone {ctx['zone_txt']}.
            3. **Source window** — {ctx['window_txt']}.
            4. **AIS correlation** — {ctx['funnel']['tracks_in_region']} tracks in region; {ctx['funnel']['in_time_window']} in the time window; {ctx['funnel']['spatially_compatible']} spatially compatible.
            5. **Candidate ranking** — overall = (spatial + temporal + trajectory) / 3.
            6. **Analyst review** — {ctx['n_high']} High and {ctx['n_med']} Medium candidates queued; ranking is not legal proof.
            """
        )

    st.markdown('<div class="as-h2">Evidence summary</div>', unsafe_allow_html=True)
    top = ctx["scores"].iloc[0]
    st.markdown(
        f"Lead candidate for review: **{top['vessel_id']}** · {evidence_badge(top['evidence_level'])} · "
        f"overall {top['overall_score']:.3f}. Ranking supports investigation; it does not determine legal responsibility.",
        unsafe_allow_html=True,
    )
    if st.button("Export investigation report", type="primary"):
        st.session_state.show_report = True
    if st.session_state.show_report:
        render_report(ctx)


def page_about() -> None:
    st.markdown('<div class="as-h1">About Aapda Setu</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="as-sub">Smart India Hackathon problem statement SIH26143 — oil spill source investigation prototype.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        ### Implemented workflow

        DETECT (scene → demo spill segmentation) → TRACE (reverse-drift ensemble) → CORRELATE (historical AIS) → RANK (evidence scores) → REVIEW (human analyst).

        ### Real in this prototype

        - Reverse-drift mathematics and the spatial / temporal / trajectory scoring formulas
        - Map layers, AIS funnel counts, and the investigation report assembled from those computed values

        ### Simulated for demonstration

        - SAR image (not a Sentinel-1 product)
        - AIS tracks (illustrative, not a live feed)
        - Environment vectors (single current/wind case, not an ocean-model hindcast)
        - Spill mask (pre-drawn; confidence is N/A — demo spill segmentation)

        Candidate generation supports investigation; final determination remains with the human analyst.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Aapda Setu — Marine Oil Spill Intelligence",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "hours_back" not in st.session_state:
        st.session_state.hours_back = 8
    if "show_heat" not in st.session_state:
        st.session_state.show_heat = True
    if "show_report" not in st.session_state:
        st.session_state.show_report = False
    if "hours_slider_widget" in st.session_state:
        st.session_state.hours_back = int(st.session_state.hours_slider_widget)

    render_nav(st.session_state.page)

    scenario = load_scenario()
    spill_img, _mask_img = load_images()
    poly_meta = scenario["poly_meta"]
    env = scenario["env"]
    ais = scenario["ais"]
    poly_ll = [(p["lat"], p["lon"]) for p in poly_meta["polygon_latlon"]]
    det = detection_dt(env)
    spill_lat = float(env["spill_lat"])
    spill_lon = float(env["spill_lon"])
    area_km2 = polygon_area_km2(poly_ll)
    hours_back = int(st.session_state.hours_back)

    products = compute_products(hours_back)
    drift = products["drift"]
    scores = products["scores"]
    ids = scores["vessel_id"].tolist()
    if "selected_vessel" not in st.session_state or st.session_state.selected_vessel not in ids:
        st.session_state.selected_vessel = ids[0]

    clat, clon = drift["centroid"]
    ctx = {
        "poly_meta": poly_meta,
        "env": env,
        "ais": ais,
        "poly_ll": poly_ll,
        "det": det,
        "spill_lat": spill_lat,
        "spill_lon": spill_lon,
        "area_km2": area_km2,
        "hours_back": hours_back,
        "drift": drift,
        "scores": scores,
        "ids": ids,
        "funnel": funnel_counts(scores),
        "n_high": int((scores["evidence_level"] == "High").sum()),
        "n_med": int((scores["evidence_level"] == "Medium").sum()),
        "zone_txt": f"{fmt_ll(clat, clon)} · ~{drift['radius_km']:.1f} km",
        "window_txt": source_time_window(det, hours_back),
        "spill_img": spill_img,
    }

    page = st.session_state.page
    if page == "Dashboard":
        page_dashboard(ctx)
    elif page == "Incidents":
        page_incidents(ctx)
    elif page == "Vessels":
        page_vessels(ctx)
    elif page == "Analytics":
        page_analytics(ctx)
    elif page == "Evidence":
        page_evidence(ctx)
    else:
        page_about()

    st.markdown(
        '<p class="as-foot">'
        "DEMO / SYNTHETIC — synthetic SAR and illustrative AIS. Not a trained production model. "
        "Candidate ranking supports investigation; it does not determine legal responsibility."
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
