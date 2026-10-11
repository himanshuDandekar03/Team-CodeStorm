# config.py - Centralized configuration for OrbitGuard AI
"""
All tuneable constants and environment-variable overrides live here.
Import this module instead of hard-coding values in individual files.
"""
import os

# ── Physical constants ───────────────────────────────────────────────
EARTH_RADIUS_KM = 6371.0                    # Mean Earth radius (km)
MU_EARTH = 398600.4418                       # Earth gravitational parameter (km³/s²)
SPEED_OF_LIGHT = 299792.458                  # km/s (for relativistic checks)

# ── Data collection ──────────────────────────────────────────────────
TLE_URLS = {
    "cosmos_1408":       "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=tle",
    "fengyun_1c":        "https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=tle",
    "iridium_33":        "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=tle",
    "cosmos_2251":       "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=tle",
    "active_satellites": "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
}

TLE_DIR  = os.environ.get("ORBITGUARD_TLE_DIR",   "tle_data")
CACHE_DIR = os.environ.get("ORBITGUARD_CACHE_DIR", "tle_cache")

# Data freshness: set ORBITGUARD_ALWAYS_FRESH=1 to bypass the
# time-window restriction and always allow a download.
ALWAYS_FRESH = os.environ.get("ORBITGUARD_ALWAYS_FRESH", "1") == "1"
DOWNLOAD_HOURS = [6, 14, 22]          # Legacy scheduled download windows
DOWNLOAD_WINDOW_MIN = 30              # ±minutes around each window
TLE_DOWNLOAD_TIMEOUT = 15             # seconds

# ── Feature extraction ───────────────────────────────────────────────
OUTPUT_FILE = os.environ.get("ORBITGUARD_FEATURES_CSV", "orbital_features.csv")

# ── ML / model training ─────────────────────────────────────────────
SAVED_MODELS_DIR = os.environ.get("ORBITGUARD_MODELS_DIR", "saved_models")
TRAIN_METRICS_DIR = os.environ.get("ORBITGUARD_METRICS_DIR", "train_metrics")
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
LEARNING_RATE = 0.05
RNN_HIDDEN_SIZE = 64
RNN_NUM_LAYERS = 2
RNN_EPOCHS = 30
RNN_BATCH_SIZE = 64

# ── Conjunction detection / collision risk ────────────────────────────
# Maximum miss-distance (km) to consider a pair as a conjunction event
CONJUNCTION_DISTANCE_THRESHOLD_KM = float(
    os.environ.get("ORBITGUARD_CONJUNCTION_DIST_KM", "25.0")
)
# Screening volume radius for fast spatial filtering (km)
SCREENING_VOLUME_KM = float(
    os.environ.get("ORBITGUARD_SCREENING_KM", "100.0")
)
# Propagation time-horizon for conjunction screening (hours)
PROPAGATION_HORIZON_HOURS = float(
    os.environ.get("ORBITGUARD_PROP_HOURS", "72.0")
)
# Propagation time-step (seconds)
PROPAGATION_STEP_SEC = float(
    os.environ.get("ORBITGUARD_PROP_STEP_SEC", "60.0")
)
# Combined hard-body radius for collision probability (metres)
COMBINED_HARD_BODY_RADIUS_M = float(
    os.environ.get("ORBITGUARD_HBR_M", "20.0")
)
# Default position uncertainty 1-sigma (km) when no covariance available
DEFAULT_POSITION_SIGMA_KM = float(
    os.environ.get("ORBITGUARD_POS_SIGMA_KM", "1.0")
)
# Risk thresholds
HIGH_RISK_PC_THRESHOLD = 1e-4   # Pc ≥ 10⁻⁴  → RED
MEDIUM_RISK_PC_THRESHOLD = 1e-6 # Pc ≥ 10⁻⁶  → YELLOW

# ── Space weather ────────────────────────────────────────────────────
NOAA_SWPC_URL = "https://services.swpc.noaa.gov"
SPACE_WEATHER_CACHE_SEC = 600    # Re-fetch space-weather data every 10 min

# ── Flask / server ───────────────────────────────────────────────────
FLASK_PORT = int(os.environ.get("PORT", "5001"))
FLASK_HOST = os.environ.get("HOST", "0.0.0.0")
SECRET_KEY = os.environ.get(
    "ORBITGUARD_SECRET", "orbitguard_ai_2026_change_me"
)
DATA_UPDATE_INTERVAL_SEC = int(
    os.environ.get("ORBITGUARD_UPDATE_SEC", "300")   # 5 min default
)
