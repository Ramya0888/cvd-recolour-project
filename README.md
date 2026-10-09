# Adaptive Colour Remapping for Colour-Vision Deficiency

**Pair-based image recolouring with memory-colour protection** — a full-stack web app that makes confusable colours distinguishable for colour-blind (CVD) viewers, while keeping the image natural for everyone else.

> Final-year Image Processing project — Department of CSE, College of Engineering Guindy, Anna University.

---

## Overview

About 1 in 12 men have colour-vision deficiency (most commonly red–green). Reds and greens collapse to near-identical tones, so information in charts, maps and photos is lost. Standard *Daltonization* tools shift colours **globally**, which often makes the whole image look unnatural.

This project instead recolours **only the colours that are actually confusable**, and **protects natural tones** (skin, sky, foliage) so the result still looks real.

### Two novelties
1. **Selective, pair-based remapping** — detect the specific colour pairs that become confusable under CVD and shift only those (not the whole image).
2. **Memory-colour protection** — skin / sky / foliage are detected and shielded from aggressive recolouring.

---

## Features

- Upload any image (JPEG / PNG, transparency handled).
- Simulate **deuteranopia, protanopia, tritanopia** with adjustable severity.
- K-means clustering in Lab space + **CIEDE2000 (ΔE)** confusable-pair detection.
- Adaptive lightness remapping with a tunable **target ΔE**.
- Toggle **memory-colour protection** on/off and see the effect live.
- Baseline **global Daltonization** for side-by-side comparison.
- Live metrics: **fix-rate, ΔE before→after, SSIM, % pixels changed, protected clusters**.
- Four-panel view: *Original · CVD view · Corrected · Corrected (CVD view)*.

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| Backend | Python 3, FastAPI, Uvicorn |
| Image processing | OpenCV, scikit-image, scikit-learn, NumPy, colorspacious |
| Frontend | React, Axios, Recharts |
| Core techniques | RGB / Lab / LMS colour spaces, CIEDE2000, K-means, SSIM |

No GPU and no model training required — runs on a laptop.

---

## System Architecture

```
User ──> React Frontend ──(HTTP/JSON)──> FastAPI (/process)
                                              │
                                              ▼
        ┌──────────── Image-Processing Pipeline (Python/OpenCV) ────────────┐
        │ 1 Preprocess → 2 Simulate CVD → 3 Cluster → 4 Detect pairs →      │
        │ 5 Protect memory → 6 Remap → 7 Baseline → 8 Metrics               │
        └───────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                          Results (images + metrics JSON)
                                  └── rendered in the four-panel view
```

---

## Project Structure

```
cvd-recolour-project/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app + endpoints
│   │   └── modules/
│   │       ├── preprocessing.py    # Module 1
│   │       ├── cvd_simulation.py   # Module 2
│   │       ├── clustering.py       # Module 3
│   │       ├── pair_detection.py   # Module 4
│   │       ├── memory_colour.py    # Module 5 (Novelty 2)
│   │       ├── remapping.py        # Module 6 (Novelty 1)
│   │       ├── baseline.py         # Module 7
│   │       └── metrics.py          # Module 8
│   ├── requirements.txt
│   └── generate_test_charts.py     # makes confusable red/green test charts
├── frontend/                       # React app (src/App.js, src/App.css)
├── data/sample_images/             # test images (Kodak, Ishihara, charts)
└── results/                        # generated outputs (gitignored)
```

---

## Setup & Installation

### 1. Backend

```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: <http://localhost:8000/docs>

### 2. Frontend

```bash
cd frontend
npm install
npm start
```

App: <http://localhost:3000> (expects the backend running on port 8000).

> Run the backend and frontend in two separate terminals.

### 3. (Optional) Generate test charts

```bash
cd backend
python generate_test_charts.py
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`  | `/health` | Health check |
| `POST` | `/upload` | Upload & preprocess an image → returns `session_id` |
| `POST` | `/process/{session_id}` | Run the full pipeline → image URLs + metrics |
| `POST` | `/simulate/{session_id}` | CVD simulation only |
| `POST` | `/cluster/{session_id}` | K-means clustering |
| `POST` | `/detect-pairs/{session_id}` | Confusable-pair detection |
| `POST` | `/classify-memory-colours/{session_id}` | Memory-colour classification |
| `POST` | `/remap/{session_id}` | Adaptive remapping |
| `POST` | `/baseline/{session_id}` | Global Daltonization baseline |
| `POST` | `/metrics/{session_id}` | Evaluation metrics |

`/process` key parameters: `cvd_type`, `severity` (0–100), `k` (clusters), `target_delta_e`, `protect_memory` (true/false).

---

## How It Works (Pipeline)

1. **Preprocessing** — validate, flatten transparency, resize, convert to RGB.
2. **CVD Simulation** — RGB → LMS, collapse the faulty cone, back to RGB (Viénot/Machado).
3. **Clustering** — K-means in Lab space → K representative colours + pixel labels.
4. **Pair Detection** — CIEDE2000 ΔE high in normal view but low in CVD view → confusable.
5. **Memory-Colour Protection** — classify skin/sky/foliage via HSV; cap their shift.
6. **Adaptive Remapping** — push only confusable pairs apart along lightness to the target ΔE.
7. **Baseline** — standard global Daltonization for comparison.
8. **Metrics** — fix-rate, ΔE before→after, SSIM, % pixels changed.

---

## Results

| Image type | Fix rate | SSIM | Notes |
|------------|---------|------|-------|
| Red-green chart | 100% | 0.98 | Dramatic, complete correction |
| Townhouse photo | 67% | 0.94 | Gentle, adaptive on muted tones |
| Portrait (face) | 50% | 0.97 | Protection guards skin tones |
| Parrots (tritanopia) | limited | high | Known limitation (blue-yellow) |

**Target-ΔE trade-off:** raising the target strengthens correction (ΔE after 5.4 → 13.2) but lowers naturalness (SSIM 0.989 → 0.882). Default is **ΔE = 12**.

---

## Limitations

- The HSV skin rule can mis-flag warm tones (e.g. red clothing).
- Strongest for red-green CVD; tritanopia (blue-yellow) correction is limited because the remap shifts lightness.
- Evaluated with validated CVD simulators, not colour-blind participants.
- Fixed cluster count; no face-aware protection yet.

## Future Work

- Face-aware protection (gate the skin label behind face detection).
- Per-type correction (shift along the blue-yellow axis for tritanopia).
- Human evaluation with real CVD participants.
- Video / real-time support and a browser-extension deployment.

---

## References

1. G. R. Kuhn, M. M. Oliveira, L. A. F. Fernandes, "An Efficient Naturalness-Preserving Image-Recoloring Method for Dichromats," *IEEE TVCG*, 14(6), 2008. **(base paper)**
2. G. M. Machado, M. M. Oliveira, L. A. F. Fernandes, "A Physiologically-Based Model for Simulation of Color Vision Deficiency," *IEEE TVCG*, 15(6), 2009.
3. H. Brettel, F. Viénot, J. D. Mollon, "Computerized simulation of color appearance for dichromats," *JOSA A*, 14(10), 1997.
4. M. R. Luo, G. Cui, B. Rigg, "The development of the CIE 2000 colour-difference formula: CIEDE2000," *Color Research & Application*, 26(5), 2001.
5. Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity (SSIM)," *IEEE TIP*, 13(4), 2004.

---

## Team

| Name | Roll Number |
|------|-------------|
| Saranya E | 2023103580 |
| Nithyasri K | 2023103586 |
| Bharathi N | 2023103589 |
| Ramya S | 2023103601 |

Department of Computer Science and Engineering, College of Engineering Guindy, Anna University, Chennai – 25.
