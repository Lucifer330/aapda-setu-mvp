# AAPDA SETU — Oil Spill Source Investigation (MVP)

Maritime incident-investigation dashboard for **Smart India Hackathon** problem statement **SIH26143**.

**AAPDA SETU**  
Satellite–AIS Intelligence for Oil Spill Source Investigation

This is a **demonstration prototype**, not an operational or legally conclusive system.

## Setup

Python **3.12** is the intended runtime (matches Streamlit Community Cloud).

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/data_gen.py
streamlit run app.py
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/data_gen.py
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501). Default backtracking is **8 hours**. Use the **3h / 6h / 8h / 12h / 24h** controls or the slider: the particle cloud, probable source zone, and AIS ranking all update together.

## Deploy on Streamlit Community Cloud

Community Cloud runs the app from a **public GitHub repository** (always free for public apps).

1. Push this project to GitHub (`app.py` at the repo root, plus `requirements.txt` and `data/`).
2. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **Create app** → select the repository, branch `main`, and main file `app.py`.
4. Open **Advanced settings** and set Python to **3.12** (required). Community Cloud does not reliably read `.python-version`. If the app was already created on 3.14, delete it and deploy again — Python cannot be changed in place.
5. Main file: `app.py`. No secrets are required.
6. Deploy. The app URL will be `https://<subdomain>.streamlit.app`.

The demo uses synthetic SAR and illustrative AIS data; keep the on-screen DEMO DATA notice when sharing the link.

## REAL vs SIMULATED

### REAL

- Architecture: satellite scene → slick polygon → reverse-drift ensemble → AIS correlation → ranked evidence → human review
- Processing pipeline: reverse Lagrangian backtracking (`src/drift_model.py`) and spatial / temporal / course scoring (`src/ais_correlation.py`)
- Proposed analytical workflow: an analyst inspects the map, varies the backtracking window, reads component scores, and generates an investigation report. Ranking supports investigation; it does not determine legal responsibility

### SIMULATED

- SAR scene: generated speckle texture, **not** a real Sentinel-1 product
- AIS tracks: eight illustrative vessels in `data/synthetic_ais.csv`, **not** a live AIS feed
- Demonstration environment data: a single current/wind vector in `data/environment.json`, **not** an ocean-model hindcast
- Spill mask: a pre-drawn polygon; mask confidence is **N/A (static sample)**

The on-screen **DEMO DATA** banner states this explicitly.

### Next-phase data replacement

| Demo input | Production replacement |
| --- | --- |
| `data/sample_spill.png` | Sentinel-1 GRD/SLC from the Copernicus / CDSE API |
| `data/sample_mask.png` | Trained slick segmenter (e.g. U-Net) with a real confidence score |
| `data/synthetic_ais.csv` | Terrestrial + satellite AIS (MMSI identity, live or replay) |
| `data/environment.json` | Current and wind hindcast fields forcing a Lagrangian model (e.g. OpenDrift) |

## Scenario (fixed demo case)

| Field | Value |
| --- | --- |
| Location | ~19.50°N, 71.00°E (Arabian Sea, west of Mumbai / Gujarat lanes) |
| Detection time | 2026-01-14 06:32 UTC |
| Current | 0.4 m/s toward 210° (southwest) |
| Wind | 6 m/s toward 240° (west-southwest) |

Direction convention: both current and wind use oceanographic **toward** directions (0° = north, clockwise). This is **not** meteorological “wind from”. See comments in `src/drift_model.py`.

Vessels **V1** and **V2** pass near the reverse-drift origin 6–10 hours before detection; **V4** is in theatre on an opposing course; **V3, V5–V8** are decoys (wrong time or different area).

## Project layout

```
app.py                 Streamlit dashboard
src/data_gen.py        Regenerates synthetic sample data
src/drift_model.py     Reverse Lagrangian backtracking
src/ais_correlation.py Evidence scoring
data/                  Generated demo artifacts
```
