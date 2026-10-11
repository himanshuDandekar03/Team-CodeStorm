# OrbitGuard AI — Team CodeStorm

> **Advanced Satellite Tracking, Conjunction Prediction, Space Weather Analysis, and Fuel-Efficient Collision Avoidance System**

An end-to-end aerospace and machine learning platform built to protect orbital assets from debris collisions in Low Earth Orbit (LEO) and beyond.

---

## 🚀 Key Capabilities

- **Real-Time Orbital Tracking & SGP4 Propagation**: High-precision ephemeris computation using SGP4/SDP4 models with sub-second accuracy across multi-day screening windows.
- **Conjunction Screening & Collision Probability ($P_c$)**: Multi-satellite all-pairs close approach analysis using 3D Cartesian Euclidean filters and Foster-1992 2D encounter-plane probability integration.
- **Space Weather & Atmospheric Drag**: Live NOAA SWPC integration (Planetary K-index, Radio Flux F10.7) and Jacchia-Roberts dynamic atmospheric scale-height density corrections.
- **Multi-Burn Collision Avoidance**: Hohmann, Bi-elliptic, and cross-track maneuver planning with Tsiolkovsky delta-v and fuel mass budget optimization.
- **Interactive 3D Visualizer & Operations Dashboard**: WebGL Cesium/Three.js-powered real-time orbital visualization, alert panels, conjunction breakdown tables, and REST APIs.
- **Automated Test Suite**: 28 comprehensive unit & integration tests validating orbital mechanics, physics bounds, and API contracts.

---

## 🛠️ Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/himanshuDandekar03/Team-CodeStorm.git
cd Team-CodeStorm
python3 -m venv .venv
source .venv/bin/activate
cd Astraeus
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Launch the OrbitGuard AI Dashboard
```bash
python app.py
```
Open [http://localhost:5001](http://localhost:5001) in your browser to view the real-time operational dashboard.

---

## 📁 Repository Structure

```
Team-CodeStorm/
├── Astraeus/
│   ├── app.py                      # Flask application and REST API endpoints
│   ├── avoidance_maneuver.py       # Collision avoidance maneuver optimizer
│   ├── config.py                   # Centralized runtime configuration & thresholds
│   ├── conjunction_detector.py     # SGP4 propagation & 2D Foster Pc engine
│   ├── data_collection.py          # CelesTrak TLE ingestion with cached fallbacks
│   ├── space_weather.py            # NOAA SWPC live telemetry & drag modeling
│   ├── fuel_opt/                   # Fuel budgeting & orbital mechanics utilities
│   ├── static/ & templates/        # 3D orbital viewer & mission control UI
│   ├── docs/                       # Scientific validation, setup & architecture docs
│   └── tests/                      # Automated test suite (28 tests)
└── README.md                       # Project overview
```

---

## 🛡️ Scientific Validation & Standards

- **Orbital Mechanics**: Compliant with Vallado & SGP4 standards.
- **Collision Risk**: Formulated using 2D encounter-plane coordinate transformation and Foster (1992) probability integration.
- **Space Weather Telemetry**: Validated against NOAA Space Weather Prediction Center (SWPC) specifications.
