"""
Generate synthetic demo data for Aapda Setu.

This script is intentionally self-contained so a teammate or SIH judge can
regenerate the scenario and see exactly how the SAR-like image, spill mask,
environment vectors, and AIS tracks were constructed.

Usage (from the project root):
    python src/data_gen.py
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Scenario constants (documented for Q&A)
# ---------------------------------------------------------------------------
# Spill is placed in the Arabian Sea west of Mumbai / Gujarat shipping lanes.
SPILL_LAT = 19.50
SPILL_LON = 71.00
DETECTION_TIME = datetime(2026, 1, 14, 6, 32, tzinfo=timezone.utc)

# Image georeferencing: 512x512 pixels covering a ~22 km square patch.
IMAGE_SIZE = 512
PATCH_KM = 22.0
KM_PER_DEG_LAT = 111.32

# Environment used by the reverse-drift model.
# Direction convention (oceanographic "toward"):
#   0° = toward north, 90° = toward east, clockwise from true north.
# Current 210° = flowing toward southwest.
# Wind 240° = blowing toward west-southwest (NOT meteorological "from").
ENVIRONMENT = {
    "current": {"speed_ms": 0.4, "direction_deg_toward": 210.0},
    "wind": {"speed_ms": 6.0, "direction_deg_toward": 240.0},
    "notes": (
        "Both current and wind use 'toward' direction (oceanographic). "
        "This is documented again in src/drift_model.py."
    ),
}

RNG = np.random.default_rng(26143)  # SIH problem-statement seed


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    d = project_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def km_per_deg_lon(lat: float) -> float:
    return KM_PER_DEG_LAT * math.cos(math.radians(lat))


class PixelGeo:
    def __init__(self, cx: float, cy: float, lat0: float, lon0: float):
        self.cx = cx
        self.cy = cy
        self.lat0 = lat0
        self.lon0 = lon0
        self.km_per_px = PATCH_KM / IMAGE_SIZE

    def pixel_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        dx_km = (x - self.cx) * self.km_per_px
        dy_km = (self.cy - y) * self.km_per_px  # image y down -> north up
        lat = self.lat0 + dy_km / KM_PER_DEG_LAT
        lon = self.lon0 + dx_km / km_per_deg_lon(self.lat0)
        return lat, lon


def value_noise(shape: int, grid: int, rng: np.random.Generator) -> np.ndarray:
    """Coarse value noise bilinearly upsampled — a cheap Perlin-ish texture."""
    coarse = rng.random((grid, grid))
    img = Image.fromarray((coarse * 255).astype(np.uint8), mode="L")
    img = img.resize((shape, shape), resample=Image.Resampling.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def make_slick_polygon(cx: float, cy: float, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Irregular elongated polygon (not an ellipse) for the oil slick."""
    n = 14
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        # Elongated along ~30° with noisy radius so it looks like a slick filament.
        rx = 78.0 + float(rng.normal(0, 10))
        ry = 28.0 + float(rng.normal(0, 6))
        x = rx * math.cos(t)
        y = ry * math.sin(t)
        ang = math.radians(28.0)
        xr = x * math.cos(ang) - y * math.sin(ang)
        yr = x * math.sin(ang) + y * math.cos(ang)
        # Occasional indent to avoid a smooth blob.
        if i in (3, 9):
            xr *= 0.62
            yr *= 0.70
        pts.append((int(round(cx + xr)), int(round(cy + yr))))
    return pts


def rasterize_polygon(size: int, polygon: list[tuple[int, int]]) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(polygon, fill=255)
    return np.asarray(img)


def generate_sar_and_mask() -> dict:
    """Build a speckled SAR-like ocean scene plus a binary slick mask."""
    n = IMAGE_SIZE
    # Multi-scale noise + multiplicative speckle (typical of SAR intensity).
    tex = (
        0.45 * value_noise(n, 8, RNG)
        + 0.30 * value_noise(n, 18, RNG)
        + 0.25 * value_noise(n, 40, RNG)
    )
    tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-8)
    speckle = RNG.gamma(shape=2.2, scale=0.45, size=(n, n))
    ocean = 0.42 + 0.22 * tex
    ocean = np.clip(ocean * speckle / speckle.mean(), 0.12, 0.95)

    # Slick off-center (northwest of image center).
    cx, cy = 210.0, 195.0
    polygon = make_slick_polygon(cx, cy, RNG)
    mask = rasterize_polygon(n, polygon)

    # Oil damps Bragg scatter -> darker returns inside the slick.
    darken = 0.38 + 0.08 * RNG.random((n, n))
    ocean = np.where(mask > 0, ocean * darken, ocean)
    # Soften the slick edge slightly so it is not a hard cartoon cutout.
    mask_img = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=1.6))
    mask_soft = np.asarray(mask_img).astype(np.float32) / 255.0
    ocean = ocean * (1.0 - 0.15 * mask_soft) + 0.08 * mask_soft * ocean

    gray = (np.clip(ocean, 0, 1) * 255).astype(np.uint8)
    spill_img = Image.fromarray(gray, mode="L")
    mask_img_bin = Image.fromarray(mask, mode="L")

    geo = PixelGeo(cx=cx, cy=cy, lat0=SPILL_LAT, lon0=SPILL_LON)
    poly_ll = [geo.pixel_to_latlon(x, y) for x, y in polygon]
    # Close the ring for GeoJSON-style consumers.
    if poly_ll[0] != poly_ll[-1]:
        poly_ll.append(poly_ll[0])

    out = data_dir()
    spill_path = out / "sample_spill.png"
    mask_path = out / "sample_mask.png"
    spill_img.save(spill_path)
    mask_img_bin.save(mask_path)

    meta = {
        "detection_time_utc": DETECTION_TIME.strftime("%Y-%m-%d %H:%M UTC"),
        "spill_lat": SPILL_LAT,
        "spill_lon": SPILL_LON,
        "image_size": IMAGE_SIZE,
        "patch_km": PATCH_KM,
        "polygon_pixels": polygon,
        "polygon_latlon": [{"lat": lat, "lon": lon} for lat, lon in poly_ll],
        "notes": "Synthetic SAR-like scene; mask is hand-constructed, not model output.",
    }
    with (out / "sample_mask_polygon.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with (out / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **ENVIRONMENT,
                "spill_lat": SPILL_LAT,
                "spill_lon": SPILL_LON,
                "detection_time_utc": meta["detection_time_utc"],
            },
            f,
            indent=2,
        )

    print(f"Wrote {spill_path}")
    print(f"Wrote {mask_path}")
    print(f"Wrote {out / 'sample_mask_polygon.json'}")
    print(f"Wrote {out / 'environment.json'}")
    return meta


def destination_point(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    """Move `dist_km` along `bearing_deg` (toward, clockwise from north)."""
    dlat = (dist_km * math.cos(math.radians(bearing_deg))) / KM_PER_DEG_LAT
    dlon = (dist_km * math.sin(math.radians(bearing_deg))) / km_per_deg_lon(lat)
    return lat + dlat, lon + dlon


def sample_times(
    start: datetime,
    end: datetime,
    step_min: int,
    dense_start: datetime | None = None,
    dense_end: datetime | None = None,
    dense_step_min: int = 15,
) -> list[datetime]:
    times = []
    t = start
    while t <= end:
        times.append(t)
        if dense_start and dense_end and dense_start <= t < dense_end:
            t += timedelta(minutes=dense_step_min)
        else:
            t += timedelta(minutes=step_min)
    if times[-1] != end:
        times.append(end)
    return times


def interpolate_at(
    waypoints: list[tuple[datetime, float, float, float, float]],
    times: list[datetime],
    rng: np.random.Generator,
) -> list[dict]:
    rows = []
    for t in times:
        if t <= waypoints[0][0]:
            a = b = waypoints[0]
            u = 0.0
        elif t >= waypoints[-1][0]:
            a = b = waypoints[-1]
            u = 0.0
        else:
            a = b = waypoints[0]
            for k in range(len(waypoints) - 1):
                if waypoints[k][0] <= t <= waypoints[k + 1][0]:
                    a, b = waypoints[k], waypoints[k + 1]
                    break
            span = (b[0] - a[0]).total_seconds() or 1.0
            u = (t - a[0]).total_seconds() / span
        lat = a[1] + u * (b[1] - a[1]) + float(rng.normal(0, 0.0035))
        lon = a[2] + u * (b[2] - a[2]) + float(rng.normal(0, 0.0035))
        spd = a[3] + u * (b[3] - a[3]) + float(rng.normal(0, 0.25))
        crs = (a[4] + u * (b[4] - a[4]) + float(rng.normal(0, 2.0))) % 360.0
        rows.append(
            {
                "timestamp": t,
                "lat": lat,
                "lon": lon,
                "speed_knots": max(0.5, spd),
                "course_deg": crs,
            }
        )
    return rows


def generate_ais(meta: dict) -> None:
    """
    Eight vessel tracks around the spill detection time.

    Combined drift (current + 3% windage) is roughly 0.56 m/s toward ~230°.
    An 8-hour reverse origin therefore sits ~16 km toward ~50° (northeast)
    of the slick. Candidates steam southwest along that axis:

    - ~9 h before detection they are 25–40 km NE of the slick
    - ~8 h before they pass the reverse-origin neighbourhood
    - then they continue SW (plausible release + drift onto the observed slick)

    V3–V8 are decoys: wrong time, opposite heading, or a different area.
    """
    det = DETECTION_TIME
    spill_lat, spill_lon = SPILL_LAT, SPILL_LON
    axis = 48.0  # reverse-drift bearing (toward NE)

    tracks: dict[str, list[dict]] = {}

    # V1 — candidate tanker, 12 kn SW along the reverse axis.
    t_pass = det - timedelta(hours=8)
    v1_wps = [
        (t_pass - timedelta(hours=2), *destination_point(spill_lat, spill_lon, axis, 38.0), 12.4, 228.0),
        (t_pass, *destination_point(spill_lat, spill_lon, axis, 16.0), 12.6, 226.0),
        (t_pass + timedelta(hours=9), *destination_point(spill_lat, spill_lon, 226.0, 100.0), 12.9, 224.0),
    ]
    tracks["V1"] = interpolate_at(
        v1_wps,
        sample_times(v1_wps[0][0], v1_wps[-1][0], 50, t_pass - timedelta(hours=2), t_pass + timedelta(hours=2), 15),
        RNG,
    )

    # V2 — candidate, slightly west of V1, pass ~7 h before detection.
    t_pass2 = det - timedelta(hours=7)
    axis2 = 42.0
    v2_wps = [
        (t_pass2 - timedelta(hours=1.5), *destination_point(spill_lat, spill_lon, axis2, 34.0), 11.3, 222.0),
        (t_pass2, *destination_point(spill_lat, spill_lon, axis2, 17.0), 11.5, 220.0),
        (t_pass2 + timedelta(hours=8), *destination_point(spill_lat, spill_lon, 220.0, 82.0), 11.4, 218.0),
    ]
    tracks["V2"] = interpolate_at(
        v2_wps,
        sample_times(v2_wps[0][0], v2_wps[-1][0], 50, t_pass2 - timedelta(hours=1.5), t_pass2 + timedelta(hours=1.5), 15),
        RNG,
    )

    # V3 — same neighbourhood but AFTER detection (wrong temporal window).
    t_late = det + timedelta(hours=4)
    v3_wps = [
        (t_late, *destination_point(spill_lat, spill_lon, 40.0, 20.0), 10.0, 225.0),
        (t_late + timedelta(hours=12), *destination_point(spill_lat, spill_lon, 225.0, 80.0), 10.4, 228.0),
    ]
    tracks["V3"] = interpolate_at(v3_wps, sample_times(v3_wps[0][0], v3_wps[-1][0], 45), RNG)

    # V4 — in the theatre but steaming AWAY (northeast) during the release window.
    t_away = det - timedelta(hours=9)
    v4_wps = [
        (t_away, *destination_point(spill_lat, spill_lon, 200.0, 18.0), 14.0, 48.0),
        (t_away + timedelta(hours=14), *destination_point(spill_lat, spill_lon, 48.0, 95.0), 14.2, 50.0),
    ]
    tracks["V4"] = interpolate_at(v4_wps, sample_times(v4_wps[0][0], v4_wps[-1][0], 50), RNG)

    # V5 — different area: north toward Gujarat / Kandla approaches.
    v5_wps = [
        (det - timedelta(hours=10), 21.6, 69.4, 9.5, 330.0),
        (det + timedelta(hours=6), 22.4, 69.1, 9.1, 335.0),
    ]
    tracks["V5"] = interpolate_at(v5_wps, sample_times(v5_wps[0][0], v5_wps[-1][0], 55), RNG)

    # V6 — Mumbai harbour approaches, well east of the slick.
    v6_wps = [
        (det - timedelta(hours=8), 18.85, 72.55, 8.0, 10.0),
        (det + timedelta(hours=8), 19.15, 72.72, 7.4, 25.0),
    ]
    tracks["V6"] = interpolate_at(v6_wps, sample_times(v6_wps[0][0], v6_wps[-1][0], 50), RNG)

    # V7 — southbound far west (open Arabian Sea).
    v7_wps = [
        (det - timedelta(hours=11), 20.8, 68.4, 13.5, 175.0),
        (det + timedelta(hours=5), 18.9, 68.55, 13.8, 178.0),
    ]
    tracks["V7"] = interpolate_at(v7_wps, sample_times(v7_wps[0][0], v7_wps[-1][0], 50), RNG)

    # V8 — westbound, leaving the area; only a short overlap with the window.
    v8_wps = [
        (det - timedelta(hours=2), 19.65, 70.55, 15.0, 265.0),
        (det + timedelta(hours=10), 19.50, 68.85, 15.4, 268.0),
    ]
    tracks["V8"] = interpolate_at(v8_wps, sample_times(v8_wps[0][0], v8_wps[-1][0], 55), RNG)

    out_csv = data_dir() / "synthetic_ais.csv"
    lines = ["vessel_id,timestamp,lat,lon,speed_knots,course_deg"]
    for vid, rows in tracks.items():
        for r in rows:
            ts = r["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ")
            lines.append(
                f"{vid},{ts},{r['lat']:.6f},{r['lon']:.6f},"
                f"{r['speed_knots']:.2f},{r['course_deg']:.1f}"
            )
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv} ({sum(len(v) for v in tracks.values())} AIS points, {len(tracks)} vessels)")


def main() -> None:
    meta = generate_sar_and_mask()
    generate_ais(meta)
    print("Sample data generation complete.")


if __name__ == "__main__":
    main()
