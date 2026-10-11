# space_weather.py — NOAA SWPC Space Weather Integration for OrbitGuard AI
"""
Fetches real-time space weather data from the NOAA Space Weather Prediction
Center (SWPC) public JSON feeds:

  • Solar flux (F10.7)
  • Geomagnetic indices (Kp, Ap)
  • Solar wind parameters (speed, density, Bz)
  • GOES X-ray flux (flare classification)
  • Atmospheric drag multiplier based on F10.7
"""
import json
import time
import threading
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError

try:
    from config import NOAA_SWPC_URL, SPACE_WEATHER_CACHE_SEC
except ImportError:
    NOAA_SWPC_URL = "https://services.swpc.noaa.gov"
    SPACE_WEATHER_CACHE_SEC = 600

# ── Verified Live Endpoints ───────────────────────────────────────────
_ENDPOINTS = {
    "kp_index":        "/products/noaa-planetary-k-index.json",
    "f107_summary":    "/products/summary/10cm-flux.json",
    "f107_history":    "/products/10cm-flux-30-day.json",
    "solar_wind_speed": "/products/summary/solar-wind-speed.json",
    "solar_wind_mag":  "/products/summary/solar-wind-mag-field.json",
    "xray_flux":       "/json/goes/primary/xrays-6-hour.json",
    "alerts":          "/products/alerts.json",
}

# ── Thread-safe cache ─────────────────────────────────────────────────
_cache = {}
_cache_lock = threading.Lock()


def _fetch_json(endpoint_key):
    """Fetch JSON from NOAA with caching and robust error handling."""
    with _cache_lock:
        cached = _cache.get(endpoint_key)
        if cached and (time.time() - cached["ts"]) < SPACE_WEATHER_CACHE_SEC:
            return cached["data"]

    url = NOAA_SWPC_URL + _ENDPOINTS[endpoint_key]
    try:
        req = Request(url, headers={"User-Agent": "OrbitGuardAI/1.0"})
        with urlopen(req, timeout=8) as resp:
            raw_text = resp.read().decode('utf-8', errors='ignore').strip()
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as jde:
                if "Extra data" in str(jde) and jde.pos > 0:
                    data = json.loads(raw_text[:jde.pos])
                else:
                    raise
        with _cache_lock:
            _cache[endpoint_key] = {"ts": time.time(), "data": data}
        return data
    except Exception as exc:
        print(f"⚠️ Space weather fetch ({endpoint_key}) note: {exc}")
        with _cache_lock:
            if cached:
                return cached["data"]
        return None


# ── Public API ────────────────────────────────────────────────────────
def get_kp_index():
    """
    Return the latest Kp index and history from NOAA.
    """
    data = _fetch_json("kp_index")
    history = []
    latest_kp = 3.0
    latest_ts = ""

    if isinstance(data, list) and data:
        for item in data:
            if isinstance(item, dict):
                kp_val = item.get("Kp") or item.get("kp")
                ts_val = item.get("time_tag", "")
                if kp_val is not None:
                    try:
                        history.append((ts_val, float(kp_val)))
                    except (ValueError, TypeError):
                        pass
        if history:
            latest_ts, latest_kp = history[-1]

    return {
        "kp": round(latest_kp, 2),
        "timestamp": latest_ts,
        "kp_history": history[-24:] if history else [],
    }


def get_solar_flux():
    """
    Return the latest F10.7 solar flux (SFU) from NOAA.
    """
    data = _fetch_json("f107_summary")
    if isinstance(data, dict) and "flux" in data:
        try:
            return {
                "f107": float(data["flux"]),
                "timestamp": str(data.get("time_tag", ""))
            }
        except (ValueError, TypeError):
            pass

    # Fallback to history
    hist = _fetch_json("f107_history")
    if isinstance(hist, list) and hist:
        last = hist[-1]
        if isinstance(last, dict) and "flux" in last:
            return {
                "f107": float(last["flux"]),
                "timestamp": str(last.get("time_tag", ""))
            }

    return {"f107": 120.0, "timestamp": ""}


def get_solar_wind():
    """
    Return latest solar wind parameters.
    """
    speed_data = _fetch_json("solar_wind_speed")
    mag_data = _fetch_json("solar_wind_mag")

    speed = 450.0
    bz = 0.0
    bt = 5.0
    ts = ""

    if isinstance(speed_data, dict):
        speed = float(speed_data.get("proton_speed") or 450.0)
        ts = speed_data.get("time_tag", "")

    if isinstance(mag_data, dict):
        bz = float(mag_data.get("bz_gsm") or 0.0)
        bt = float(mag_data.get("bt") or 5.0)

    return {
        "speed_km_s": round(speed, 1),
        "bz_nT": round(bz, 2),
        "bt_nT": round(bt, 2),
        "timestamp": ts
    }


def classify_xray(flux_w_m2):
    """Classify X-ray flux into standard flare class (A, B, C, M, X)."""
    if flux_w_m2 <= 0:
        return "A0.0"
    import math
    try:
        exp = math.log10(flux_w_m2)
        if exp >= -4:
            return f"X{flux_w_m2 / 1e-4:.1f}"
        elif exp >= -5:
            return f"M{flux_w_m2 / 1e-5:.1f}"
        elif exp >= -6:
            return f"C{flux_w_m2 / 1e-6:.1f}"
        elif exp >= -7:
            return f"B{flux_w_m2 / 1e-7:.1f}"
        else:
            return f"A{flux_w_m2 / 1e-8:.1f}"
    except Exception:
        return "C1.0"


def get_xray_flux():
    """
    Return latest GOES X-ray flux and flare classification.
    """
    data = _fetch_json("xray_flux")
    flux = 1e-7
    ts = ""
    flare_class = "B1.0"

    if isinstance(data, list) and data:
        for entry in reversed(data):
            if isinstance(entry, dict) and entry.get("energy") == "0.1-0.8nm":
                flux_val = entry.get("flux")
                if flux_val is not None:
                    try:
                        flux = float(flux_val)
                        ts = entry.get("time_tag", "")
                        flare_class = classify_xray(flux)
                        break
                    except (ValueError, TypeError):
                        pass

    return {
        "flux_w_m2": flux,
        "flare_class": flare_class,
        "timestamp": ts
    }


def atmospheric_drag_multiplier(f107=None, altitude_km=400):
    """
    Thermospheric density scaling factor based on F10.7 solar flux.
    Returns relative multiplier over quiet-Sun baseline (F10.7 = 70).
    """
    if f107 is None:
        sw = get_solar_flux()
        f107 = sw.get("f107", 70.0)

    import math
    scale_factor = 1.0 + 0.015 * (f107 - 70.0) * math.exp(-altitude_km / 500.0)
    return max(0.2, scale_factor)


def get_geomagnetic_storm_level(kp=None):
    """
    Classify NOAA Space Weather Geomagnetic Storm Scale (G0–G5).
    """
    if kp is None:
        kp_data = get_kp_index()
        kp = kp_data.get("kp", 0.0)

    if kp >= 9:
        return "G5", "Extreme Geomagnetic Storm"
    elif kp >= 8:
        return "G4", "Severe Geomagnetic Storm"
    elif kp >= 7:
        return "G3", "Strong Geomagnetic Storm"
    elif kp >= 6:
        return "G2", "Moderate Geomagnetic Storm"
    elif kp >= 5:
        return "G1", "Minor Geomagnetic Storm"
    elif kp >= 4:
        return "G0", "Active Unsettled Conditions"
    else:
        return "Quiet", "Quiet Geomagnetic Conditions"


def get_space_weather_summary():
    """
    Single aggregated dashboard payload for all space weather indicators.
    """
    kp_data = get_kp_index()
    f107_data = get_solar_flux()
    wind_data = get_solar_wind()
    xray_data = get_xray_flux()

    storm_level, storm_desc = get_geomagnetic_storm_level(kp_data["kp"])
    drag_mult = atmospheric_drag_multiplier(f107_data["f107"])

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kp_index": kp_data["kp"],
        "kp_history": kp_data["kp_history"],
        "f107_flux": f107_data["f107"],
        "solar_wind": wind_data,
        "xray": xray_data,
        "storm_level": storm_level,
        "storm_description": storm_desc,
        "drag_multiplier": round(drag_mult, 3),
        "status": "ok",
    }
