# OrbitGuard AI — Setup Guide

**Version:** 1.0
**Date:** 2026-10-11

---

## 1. Prerequisites

- **Python 3.10+** (tested on 3.12)
- **pip** or **venv** for package management
- Internet access for TLE downloads and NOAA space weather feeds

---

## 2. Installation

```bash
# Clone the repository
git clone https://github.com/randoduck/Astraeus.git
cd Astraeus

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install flask numpy scipy pandas scikit-learn \
            sgp4 xgboost lightgbm catboost torch \
            pytest matplotlib
```

---

## 3. Directory Structure

```
Astraeus/
├── app.py                      # Flask web server & dashboard
├── config.py                   # Centralised configuration
├── data_collection.py          # TLE download & caching
├── feature_extraction.py       # TLE → orbital feature vectors
├── conjunction_detector.py     # Pairwise conjunction screening
├── avoidance_maneuver.py       # CW-based collision avoidance
├── space_weather.py            # NOAA SWPC integration
├── predict_collision_risk.py   # ML ensemble + conjunction pipeline
├── model_training.py           # ML model training scripts
├── tests/                      # Automated test suite (pytest)
│   ├── conftest.py
│   ├── test_orbital_propagation.py
│   ├── test_conjunction_detector.py
│   ├── test_avoidance_maneuver.py
│   └── test_space_weather.py
├── docs/                       # Documentation
│   ├── SCIENTIFIC_VALIDATION_REPORT.md
│   ├── VALIDATION_METHODOLOGY.md
│   ├── LIMITATIONS.md
│   └── SETUP.md (this file)
├── templates/                  # HTML templates
├── static/                     # CSS, JS, images
├── tle_cache/                  # Cached TLE data (auto-created)
├── saved_models/               # Trained ML models
└── fuel_opt/                   # Fuel optimisation module
```

---

## 4. Running the Server

```bash
# Start the Flask development server on port 5001
PORT=5001 python app.py
```

Open http://127.0.0.1:5001 in your browser.

### Default Credentials

| Field | Value |
|---|---|
| Username | `admin` |
| Password | `astraeus2025` |

> ⚠️ **Change these credentials before deploying to any network.**

---

## 5. Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run a specific test module
python -m pytest tests/test_orbital_propagation.py -v

# Run with coverage (requires pytest-cov)
pip install pytest-cov
python -m pytest tests/ --cov=. --cov-report=term-missing
```

Expected output: **81 tests, 100% passing.**

---

## 6. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5001` | Flask server port |
| `HOST` | `0.0.0.0` | Flask bind address |
| `ORBITGUARD_SECRET` | `orbitguard_ai_2026...` | Flask secret key |
| `ORBITGUARD_TLE_DIR` | `tle_data` | TLE file directory |
| `ORBITGUARD_CACHE_DIR` | `tle_cache` | TLE cache directory |
| `ORBITGUARD_CONJUNCTION_DIST_KM` | `25.0` | Conjunction threshold (km) |
| `ORBITGUARD_SCREENING_KM` | `100.0` | Screening volume radius (km) |
| `ORBITGUARD_PROP_HOURS` | `72.0` | Propagation horizon (hours) |
| `ORBITGUARD_PROP_STEP_SEC` | `60.0` | Propagation step (seconds) |
| `ORBITGUARD_HBR_M` | `20.0` | Combined hard-body radius (m) |
| `ORBITGUARD_POS_SIGMA_KM` | `1.0` | Default position sigma (km) |
| `ORBITGUARD_UPDATE_SEC` | `300` | Background update interval (s) |
| `ORBITGUARD_ALWAYS_FRESH` | `1` | Bypass download time windows |

---

## 7. API Endpoints

### Core Data
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | System status |
| GET | `/api/satellites` | All tracked objects |
| GET | `/api/debris` | Detected debris objects |

### Conjunction Detection
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/conjunctions` | Cached conjunction events |
| POST | `/api/conjunctions/screen` | On-demand screening |

### Space Weather
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/space-weather` | NOAA SWPC aggregated data |

### Collision Avoidance
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/avoidance/recommendations` | Current recommendations |
| POST | `/api/avoidance/plan` | Plan manoeuvre for event |

---

## 8. Training ML Models

If you need to retrain the debris detection and trajectory models:

```bash
# Download latest TLE data
python data_collection.py

# Extract features
python feature_extraction.py

# Train models
python model_training.py
```

Trained models are saved to `saved_models/`. The server loads them
automatically on startup.

---

## 9. Troubleshooting

| Issue | Solution |
|---|---|
| "No TLE data found" | Run `python data_collection.py` first |
| Port 5001 in use | Set `PORT=5002 python app.py` |
| ML models not loading | Run `python model_training.py` to train |
| NOAA feed timeouts | Check internet; cached data will be used |
| Import errors | Ensure venv is activated and deps installed |
