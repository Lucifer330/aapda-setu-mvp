"""
AAPDA SETU — oil spill source investigation prototype (SIH26143).

Run from the project root:
    python src/data_gen.py
    streamlit run app.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
sys.path.insert(0, str(ROOT / "src"))

from ais_correlation import _ang_diff, _to_utc, score_tracks  # noqa: E402
from drift_model import backtrack  # noqa: E402

DATA = ROOT / "data"
KM_PER_DEG_LAT = 111.32
PRESET_HOURS = [3, 6, 8, 12, 24]

LEVEL_COLORS = {
    "High": "#d4764e",
    "Medium": "#c9a227",
    "Low": "#7d8b99",
}
LEVEL_BG = {
    "High": "#3a241c",
    "Medium": "#2f2a16",
    "Low": "#1b2834",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: "IBM Plex Sans", sans-serif;
}
.stApp { background: #0b1c2c; }
header[data-testid="stHeader"] { background: #0b1c2c; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1480px; }

.as-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    border-bottom: 1px solid #1e3a52;
    padding-bottom: 12px;
    margin-bottom: 14px;
}
.as-brand { letter-spacing: 0.18em; font-size: 0.72rem; color: #7ad4cc; font-weight: 600; }
.as-title { font-size: 1.55rem; font-weight: 700; color: #f4f8fb; margin: 2px 0 0 0; line-height: 1.2; }
.as-sub { font-size: 0.92rem; color: #9bb0c3; margin-top: 4px; }
.as-demo {
    background: #12283c;
    border: 1px solid #2aa9a1;
    color: #d7f3f0;
    padding: 8px 12px;
    font-size: 0.78rem;
    line-height: 1.35;
    max-width: 340px;
}
.as-demo b { color: #7ad4cc; letter-spacing: 0.12em; font-size: 0.7rem; }

.as-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 12px 0 8px 0; }
.as-metric {
    background: #12283c;
    border: 1px solid #1e3a52;
    padding: 10px 12px;
}
.as-metric .k { font-size: 0.68rem; letter-spacing: 0.12em; color: #7ad4cc; font-weight: 600; }
.as-metric .v { font-size: 1.05rem; color: #f4f8fb; font-weight: 600; margin-top: 4px; }
.as-metric .h { font-size: 0.75rem; color: #9bb0c3; margin-top: 2px; }

.as-flow {
    display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
    font-size: 0.72rem; letter-spacing: 0.08em; color: #9bb0c3;
    margin: 6px 0 14px 0;
}
.as-flow span {
    background: #12283c; border: 1px solid #1e3a52; padding: 4px 8px; color: #d5e4ef;
}
.as-flow .arr { background: none; border: none; color: #2aa9a1; padding: 0 2px; }

.as-panel-label {
    font-size: 0.72rem; letter-spacing: 0.14em; color: #7ad4cc; font-weight: 600;
    margin-bottom: 6px;
}
.as-note { font-size: 0.78rem; color: #9bb0c3; }
.as-report {
    background: #12283c; border: 1px solid #1e3a52; padding: 16px 18px; margin-top: 8px;
}
.as-report h3 { margin: 0 0 10px 0; color: #f4f8fb; font-size: 1.05rem; }
.as-report table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.as-report td { padding: 6px 8px; border-top: 1px solid #1e3a52; vertical-align: top; }
.as-report td.k { color: #9bb0c3; width: 240px; }
.as-report td.v { color: #e8eef4; }
.as-foot { color: #8aa0b4; font-size: 0.8rem; margin-top: 18px; }

[data-testid="stSlider"] label { color: #9bb0c3 !important; }
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
    from data_gen import main as gen_main

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
    gray = np.asarray(spill.convert("L"), dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    ax.imshow(gray, cmap="gray", origin="upper")
    xs = [p[0] for p in polygon_pixels] + [polygon_pixels[0][0]]
    ys = [p[1] for p in polygon_pixels] + [polygon_pixels[0][1]]
    ax.plot(xs, ys, color="#d4764e", linewidth=2.2)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    bio = BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    bio.seek(0)
    return Image.open(bio).convert("RGB")


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


def environmental_consistency(
    track: pd.DataFrame,
    drift: dict,
    detection_time: datetime,
    hours_back: float,
) -> float:
    """
    Alignment of the vessel's in-window heading with the oil-drift direction
    implied by the scenario current + windage. Not a fourth overall-score
    weight — a diagnostic that uses the same vectors as reverse backtracking.
    """
    u, v = drift["forward_uv_ms"]
    oil_toward = (math.degrees(math.atan2(u, v)) + 360.0) % 360.0
    release_start = detection_time - timedelta(hours=float(hours_back))
    errs = []
    for _, row in track.iterrows():
        t = _to_utc(row["timestamp"])
        if t < release_start or t > detection_time:
            continue
        errs.append(_ang_diff(float(row["course_deg"]), oil_toward))
    if not errs:
        return 0.0
    return round(max(0.0, 1.0 - float(np.mean(errs)) / 180.0), 3)


class MapLegend(MacroElement):
    def __init__(self):
        super().__init__()
        self._name = "MapLegend"
        self._template = Template(
            """
            {% macro html(this, kwargs) %}
            <div style="
                position: fixed; bottom: 28px; left: 28px; z-index: 9999;
                background: rgba(11,28,44,0.92); color: #e8eef4;
                border: 1px solid #1e3a52; padding: 10px 12px;
                font-size: 12px; font-family: sans-serif;
                line-height: 1.7;">
                <div style="letter-spacing:0.12em;color:#7ad4cc;font-size:10px;">LEGEND</div>
                <div><span style="color:#d4764e;">&#9632;</span> Detected Slick</div>
                <div><span style="color:#5ec8c5;">&#9679;</span> Reverse Drift</div>
                <div><span style="color:#7ad4cc;">&#9711;</span> Probable Source Zone</div>
                <div><span style="color:#8aa0b4;">&#8212;</span> AIS Track</div>
                <div><span style="color:#e8eef4;">&#9670;</span> Candidate Vessel</div>
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
        color="#d4764e",
        weight=3,
        fill=True,
        fill_color="#d4764e",
        fill_opacity=0.28,
        tooltip="Detected slick",
    ).add_to(m)
    folium.CircleMarker(
        location=[spill_lat, spill_lon],
        radius=6,
        color="#f4f8fb",
        fill=True,
        fill_color="#d4764e",
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
            gradient={0.2: "#7ad4cc", 0.55: "#2aa9a1", 0.85: "#0e6b66"},
            name="Reverse drift",
        ).add_to(m)

    folium.Circle(
        location=[centroid[0], centroid[1]],
        radius=radius_km * 1000.0,
        color="#7ad4cc",
        weight=2,
        fill=True,
        fill_color="#2aa9a1",
        fill_opacity=0.06,
        dash_array="6,6",
        tooltip=f"Probable source zone · ~{radius_km:.1f} km RMS radius",
    ).add_to(m)
    folium.CircleMarker(
        location=[centroid[0], centroid[1]],
        radius=5,
        color="#7ad4cc",
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
        color = "#f4f8fb" if is_sel else LEVEL_COLORS[level]
        folium.PolyLine(
            locations=list(zip(g["lat"].tolist(), g["lon"].tolist())),
            color=color,
            weight=5 if is_sel else (3 if level == "High" else 2),
            opacity=0.95 if is_sel else 0.85,
            tooltip=f"{vid} · {level}",
        ).add_to(m)
        ml, mo, ev, sc = marker_by_id[vid]
        folium.CircleMarker(
            location=[ml, mo],
            radius=9 if is_sel else 6,
            color="#f4f8fb" if is_sel else color,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            popup=folium.Popup(
                f"<b>{vid}</b><br>Evidence: {ev}<br>Overall: {sc:.3f}",
                max_width=220,
            ),
        ).add_to(m)
        folium.Marker(
            location=[ml, mo],
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:11px;color:#f4f8fb;font-weight:600;'
                    f'text-shadow:0 0 4px #0b1c2c;white-space:nowrap;">{vid}</div>'
                )
            ),
        ).add_to(m)

    MapLegend().add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def evidence_badge(level: str) -> str:
    color = {"High": "#d4764e", "Medium": "#c9a227", "Low": "#7d8b99"}[level]
    return (
        f"<span style='display:inline-block;min-width:64px;text-align:center;"
        f"border:1px solid {color};color:{color};padding:1px 8px;font-size:0.78rem;"
        f"letter-spacing:0.08em;'>{level.upper()}</span>"
    )


def main() -> None:
    st.set_page_config(
        page_title="AAPDA SETU — Oil Spill Source Investigation",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

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

    if "hours_back" not in st.session_state:
        st.session_state.hours_back = 8
    if "show_report" not in st.session_state:
        st.session_state.show_report = False

    st.markdown(
        f"""
        <div class="as-top">
          <div>
            <div class="as-brand">AAPDA SETU</div>
            <div class="as-title">Oil Spill Source Investigation</div>
            <div class="as-sub">Satellite–AIS Intelligence for Oil Spill Source Investigation</div>
            <div class="as-sub" style="margin-top:2px;">SIH26143 · Arabian Sea demonstration case</div>
          </div>
          <div class="as-demo">
            <b>DEMO DATA</b><br/>
            This prototype uses synthetic SAR and illustrative AIS data for demonstration.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="as-flow">
            <span>SATELLITE</span><span class="arr">→</span>
            <span>SPILL DETECTION</span><span class="arr">→</span>
            <span>REVERSE DRIFT</span><span class="arr">→</span>
            <span>AIS CORRELATION</span><span class="arr">→</span>
            <span>EVIDENCE</span><span class="arr">→</span>
            <span>HUMAN REVIEW</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctrl_a, ctrl_b, ctrl_c = st.columns([1.6, 2.2, 1.0])
    with ctrl_a:
        st.markdown('<div class="as-panel-label">BACKTRACKING WINDOW</div>', unsafe_allow_html=True)
        preset_cols = st.columns(len(PRESET_HOURS))
        for h, col in zip(PRESET_HOURS, preset_cols):
            if col.button(
                f"{h}h",
                key=f"preset_{h}",
                use_container_width=True,
                type="primary" if st.session_state.hours_back == h else "secondary",
            ):
                st.session_state.hours_back = h
                st.rerun()
    with ctrl_b:
        st.slider(
            "Fine adjust (hours)",
            min_value=3,
            max_value=24,
            step=1,
            key="hours_back",
        )
    with ctrl_c:
        st.write("")
        show_heat = st.checkbox("Particle heatmap", value=True)

    hours_back = int(st.session_state.hours_back)

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
        spill_lat=spill_lat,
        spill_lon=spill_lon,
        detection_time=det,
        hours_back=hours_back,
    )

    n_high = int((scores["evidence_level"] == "High").sum())
    n_med = int((scores["evidence_level"] == "Medium").sum())
    clat, clon = drift["centroid"]
    zone_txt = f"{fmt_ll(clat, clon)} · ~{drift['radius_km']:.1f} km"
    window_txt = source_time_window(det, hours_back)

    st.markdown(
        f"""
        <div class="as-metrics">
          <div class="as-metric">
            <div class="k">DETECTED SLICK</div>
            <div class="v">{area_km2:.2f} km²</div>
            <div class="h">{fmt_ll(spill_lat, spill_lon)}</div>
          </div>
          <div class="as-metric">
            <div class="k">DETECTION TIME</div>
            <div class="v">{env['detection_time_utc']}</div>
            <div class="h">Mask confidence: N/A (static sample)</div>
          </div>
          <div class="as-metric">
            <div class="k">PROBABLE SOURCE ZONE</div>
            <div class="v">~{drift['radius_km']:.1f} km RMS</div>
            <div class="h">{fmt_ll(clat, clon)}</div>
          </div>
          <div class="as-metric">
            <div class="k">CANDIDATE VESSELS</div>
            <div class="v">{n_high} High · {n_med} Medium</div>
            <div class="h">{len(scores)} AIS tracks ranked</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ids = scores["vessel_id"].tolist()
    if "selected_vessel" not in st.session_state or st.session_state.selected_vessel not in ids:
        st.session_state.selected_vessel = ids[0]

    left, right = st.columns([0.68, 0.32], gap="medium")
    with left:
        st.markdown('<div class="as-panel-label">INCIDENT MAP</div>', unsafe_allow_html=True)
        fmap = build_map(
            poly_ll=poly_ll,
            particles=drift["positions"],
            centroid=drift["centroid"],
            radius_km=drift["radius_km"],
            ais=ais,
            scores=scores,
            show_heat=show_heat,
            spill_lat=spill_lat,
            spill_lon=spill_lon,
            selected_vessel=st.session_state.selected_vessel,
        )
        st_folium(fmap, width=None, height=680, returned_objects=[], key="incident_map")
        st.markdown(
            f'<div class="as-note">Probable source zone: {zone_txt}<br/>'
            f"Estimated source time window: {window_txt} "
            f"(detection time minus backtracking window; not a precise release clock)</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="as-panel-label">AIS CORRELATION</div>', unsafe_allow_html=True)
        st.caption("Illustrative AIS tracks")

        table_html = [
            "<table style='width:100%;border-collapse:collapse;font-size:0.82rem;'>",
            "<tr style='color:#7ad4cc;letter-spacing:0.06em;font-size:0.7rem;'>",
            "<th align='left'>Candidate Vessel</th><th>Spatial</th><th>Temporal</th>"
            "<th>Course</th><th align='left'>Evidence</th></tr>",
        ]
        for _, r in scores.iterrows():
            hl = "background:#163044;" if r["vessel_id"] == st.session_state.selected_vessel else ""
            table_html.append(
                f"<tr style='{hl}border-top:1px solid #1e3a52;'>"
                f"<td style='padding:6px 4px;font-weight:600;'>{r['vessel_id']}</td>"
                f"<td align='center'>{r['spatial_score']:.3f}</td>"
                f"<td align='center'>{r['temporal_score']:.3f}</td>"
                f"<td align='center'>{r['course_score']:.3f}</td>"
                f"<td style='padding:6px 4px;'>{evidence_badge(r['evidence_level'])}</td>"
                f"</tr>"
            )
        table_html.append("</table>")
        st.markdown("".join(table_html), unsafe_allow_html=True)

        st.selectbox(
            "Selected vessel",
            options=ids,
            key="selected_vessel",
        )

        sel = scores[scores["vessel_id"] == st.session_state.selected_vessel].iloc[0]
        track = ais[ais["vessel_id"] == sel["vessel_id"]]
        env_score = environmental_consistency(track, drift, det, hours_back)

        st.markdown('<div class="as-panel-label" style="margin-top:12px;">EVIDENCE BREAKDOWN</div>', unsafe_allow_html=True)
        st.markdown(
            f"**{sel['vessel_id']}** · overall {sel['overall_score']:.3f} · {evidence_badge(sel['evidence_level'])}",
            unsafe_allow_html=True,
        )
        st.caption(
            "Overall evidence is the equal-weight mean of spatial, temporal, and course. "
            "Environmental consistency is a diagnostic from the same current/wind vectors; "
            "it is not a fourth weight in the overall score."
        )

        breakdown = pd.DataFrame(
            {
                "component": [
                    "Spatial match",
                    "Temporal match",
                    "Course / trajectory match",
                    "Environmental consistency",
                ],
                "score": [
                    sel["spatial_score"],
                    sel["temporal_score"],
                    sel["course_score"],
                    env_score,
                ],
            }
        )
        fig = go.Figure(
            go.Bar(
                x=breakdown["score"],
                y=breakdown["component"],
                orientation="h",
                marker_color=["#2aa9a1", "#5b8fb8", "#c9a227", "#7ad4cc"],
                text=[f"{v:.3f}" for v in breakdown["score"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#12283c",
            plot_bgcolor="#12283c",
            height=230,
            margin=dict(l=10, r=40, t=8, b=8),
            xaxis=dict(range=[0, 1.15], title="Score (0–1)", gridcolor="#1e3a52"),
            yaxis=dict(autorange="reversed"),
            font=dict(color="#e8eef4", size=12),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"**Overall evidence level:** {evidence_badge(sel['evidence_level'])} "
            f"(computed {sel['overall_score']:.3f})",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Scenario current {env['current']['speed_ms']} m/s toward "
            f"{env['current']['direction_deg_toward']}° · wind "
            f"{env['wind']['speed_ms']} m/s toward {env['wind']['direction_deg_toward']}° · windage 3%."
        )

        if st.button("GENERATE INVESTIGATION REPORT", use_container_width=True):
            st.session_state.show_report = True

    with st.expander("Synthetic SAR scene (not a live Sentinel-1 product)", expanded=False):
        c1, c2 = st.columns(2)
        overlay = overlay_slick(spill_img, poly_meta["polygon_pixels"])
        with c1:
            st.image(spill_img, caption="Synthetic SAR-like scene", use_container_width=True)
        with c2:
            st.image(overlay, caption="Mask outline (static sample)", use_container_width=True)

    if st.session_state.show_report:
        high_ids = ", ".join(scores.loc[scores["evidence_level"] == "High", "vessel_id"].tolist()) or "None"
        med_ids = ", ".join(scores.loc[scores["evidence_level"] == "Medium", "vessel_id"].tolist()) or "None"
        summary_bits = []
        for _, r in scores.head(3).iterrows():
            summary_bits.append(
                f"{r['vessel_id']} {r['evidence_level']} "
                f"(spatial {r['spatial_score']:.3f}, temporal {r['temporal_score']:.3f}, "
                f"course {r['course_score']:.3f}, overall {r['overall_score']:.3f})"
            )
        st.markdown(
            f"""
            <div class="as-report">
              <h3>Investigation report</h3>
              <table>
                <tr><td class="k">Incident</td>
                    <td class="v">Arabian Sea slick · AAPDA SETU demonstration case (SIH26143)</td></tr>
                <tr><td class="k">Detection Time</td>
                    <td class="v">{env['detection_time_utc']}</td></tr>
                <tr><td class="k">Spill Location</td>
                    <td class="v">{fmt_ll(spill_lat, spill_lon)}</td></tr>
                <tr><td class="k">Spill Confidence</td>
                    <td class="v">N/A (static sample mask — not model-inferred)</td></tr>
                <tr><td class="k">Backtracking Window</td>
                    <td class="v">{hours_back} hours</td></tr>
                <tr><td class="k">Probable Source Zone</td>
                    <td class="v">{zone_txt} (ensemble centroid and RMS radius)</td></tr>
                <tr><td class="k">Estimated Source Time Window</td>
                    <td class="v">{window_txt}</td></tr>
                <tr><td class="k">Candidate Vessels</td>
                    <td class="v">High: {high_ids}<br/>Medium: {med_ids}<br/>Tracks ranked: {', '.join(ids)}</td></tr>
                <tr><td class="k">Evidence Summary</td>
                    <td class="v">{"<br/>".join(summary_bits)}</td></tr>
                <tr><td class="k">Investigation Status</td>
                    <td class="v">Open — candidate generation complete; analyst review required</td></tr>
              </table>
              <p style="color:#9bb0c3;font-size:0.85rem;margin:12px 0 0 0;">
                Candidate generation supports investigation; final determination remains with the human analyst.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p class="as-foot">'
        "This prototype uses synthetic SAR and illustrative AIS data for demonstration. "
        "Not a trained production model. Candidate ranking supports investigation; "
        "it does not determine legal responsibility."
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
